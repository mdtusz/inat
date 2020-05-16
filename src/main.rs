use std::collections::HashMap;
use std::io;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{OutputBuffer, StreamData};
use log::info;
use portaudio as pa;
use sample::conv::ToFrameSliceMut;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use tui::backend::TermionBackend;
use tui::buffer::Buffer;
use tui::layout::{Constraint, Direction, Layout, Rect};
use tui::style::{Color, Style};
use tui::symbols::{bar, block, line};
use tui::widgets::{Block, Borders, Paragraph, Sparkline, Text, Widget};
use tui::Terminal;

mod engine;
mod ui;

use crate::engine::{ConnectionKind, DspNode, Frame as F, Graph, NodeId, Sampler};
use crate::ui::{Mode, UiState};

#[derive(Clone)]
struct Transport {
    pub frame: usize,
    pub next_step_frame: usize,
    pub playing: bool,
    pub step: usize,
    pub sequence_length: usize,
    pub tempo: f64,
}

impl Transport {
    fn new() -> Self {
        Self {
            frame: 0,
            next_step_frame: 0,
            playing: true,
            step: 0,
            sequence_length: 64,
            tempo: 110.0,
        }
    }

    fn play_pause(&mut self) {
        self.playing = !self.playing;

        // Reset frames if restarting.
        if self.playing {
            self.frame = 0;
            self.step = 0;
        }
    }

    fn tick(&mut self) {
        self.frame += 1;
    }

    fn step(&mut self) {
        self.step_by(1);
    }

    fn step_by(&mut self, step: i16) {
        let mut offset = step;

        while offset < 0 {
            offset += self.sequence_length as i16;
        }

        self.step = (self.step + offset as usize) % self.sequence_length;
    }
}

struct App {
    graph: Graph<DspNode>,
    transport: Transport,
    samples: HashMap<String, NodeId>,
    ui: UiState,
    steps: Vec<bool>,
}

impl App {
    fn new() -> Self {
        let mut graph = Graph::new();
        let transport = Transport::new();
        let ui = UiState::new();

        let mut samples = HashMap::new();

        let master = graph.add_node(DspNode::Gain(1.0));
        let s = graph.add_node(DspNode::Sampler(Sampler::new()));

        graph.connect(s, master, ConnectionKind::Default);
        graph.set_root(master);

        let mut steps = vec![false; transport.sequence_length];

        Self {
            graph,
            samples,
            steps,
            transport,
            ui,
        }
    }
}

fn process_audio(mut buffer: OutputBuffer<f32>, app: &Mutex<App>, sr: f64) {
    let mut app = app.lock().unwrap();

    let initial_frame = app.transport.frame;
    let frame_count = buffer.len() / 2;
    let mut next_cycle = app.transport.frame;

    if app.transport.playing {
        next_cycle += frame_count;
    } else {
        app.transport.next_step_frame = 0;
    }

    while app.transport.frame < next_cycle {
        if app.transport.frame == app.transport.next_step_frame {
            // The 4.0 here is the beats-per-bar.
            app.transport.next_step_frame +=
                (sr / (app.transport.tempo * 4.0 / 60.0)).round() as usize;

            app.transport.step();

            info!("schedule step here! {}", app.transport.step);
        }

        app.transport.tick();
    }

    let buffer: &mut [F] = buffer.to_frame_slice_mut().unwrap();

    if app.transport.playing {
        let frames = app.graph.compute(sr, initial_frame, frame_count);
        sample::slice::write(buffer, &frames);
    } else {
        sample::slice::equilibrium(buffer);
    }
}

fn main() -> Result<(), pa::Error> {
    let app = App::new();

    // Prepare graph for concurrency.
    let pair = Arc::new((Mutex::new(app), Condvar::new()));

    let engine_pair = Arc::clone(&pair);
    let ui_pair = Arc::clone(&pair);
    let input_pair = Arc::clone(&pair);

    let host = cpal::default_host();
    let event_loop = host.event_loop();

    let device = host
        .default_output_device()
        .expect("Could not load device.");

    let format = device
        .default_output_format()
        .expect("Could not load output format.");

    let stream = event_loop
        .build_output_stream(&device, &format)
        .expect("Could not build output stream.");

    event_loop
        .play_stream(stream)
        .expect("Could not play stream.");

    let sr = format.sample_rate.0 as f64;

    // Engine thread.
    thread::spawn(move || {
        event_loop.run(move |_stream_id, stream_result| {
            let stream_data = match stream_result {
                Ok(data) => data,
                Err(_) => panic!("Error in event loop!"),
            };

            let (app, _trigger) = &*engine_pair;

            match stream_data {
                StreamData::Output {
                    buffer: cpal::UnknownTypeOutputBuffer::F32(buffer),
                } => process_audio(buffer, app, sr),
                _ => panic!("Audio output not implemented for this format."),
            };
        });
    });

    let stdout = io::stdout().into_raw_mode().unwrap();
    let backend = TermionBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    // Clear and hide the cursor to start!
    terminal.clear().unwrap();
    terminal.hide_cursor().unwrap();

    // Input thread.
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || loop {
        let (_, trigger) = &*input_pair;
        let stdin = io::stdin();
        match stdin.keys().next().unwrap() {
            Ok(key) => {
                tx.send(key).unwrap();
                trigger.notify_all();
            }
            _ => {}
        }
    });

    // UI thread
    loop {
        let (app, trigger) = &*ui_pair;
        let mut app = app.lock().unwrap();
        let result = trigger
            // .wait(app)
            .wait_timeout(app, Duration::from_millis(32))
            .unwrap();

        app = result.0;
        // app = result;

        match rx.try_recv() {
            Ok(key) => match app.ui.mode {
                Mode::Command => match key {
                    Key::Esc => {
                        terminal.hide_cursor().unwrap();
                        app.ui.input = String::new();
                        app.ui.mode = Mode::Normal;
                    }
                    Key::Char('\n') => {
                        if app.ui.input == "q" {
                            terminal.clear().unwrap();
                            break;
                        }

                        terminal.hide_cursor().unwrap();
                        app.ui.input = String::new();
                        app.ui.mode = Mode::Normal;
                    }
                    Key::Char(c) => {
                        app.ui.input = format!("{}{}", app.ui.input, c);
                    }
                    Key::Backspace => {
                        app.ui.input.pop();
                    }
                    Key::Ctrl('u') => {
                        app.ui.input = String::new();
                    }
                    _ => {
                        info!("Unhandled key event: {:?}", key);
                    }
                },
                Mode::Insert => match key {
                    Key::Esc => {
                        terminal.hide_cursor().unwrap();
                        app.ui.input = String::new();
                        app.ui.mode = Mode::Normal;
                    }
                    _ => {
                        info!("Unhandled key event: {:?}", key);
                    }
                },
                Mode::Normal => match key {
                    Key::Char(' ') => {
                        app.transport.play_pause();
                    }
                    Key::Char('i') => {
                        terminal.show_cursor().unwrap();
                        app.ui.mode = Mode::Insert;
                    }
                    Key::Char('~') => {
                        app.ui.debug = !app.ui.debug;
                    }
                    Key::Char(':') => {
                        app.ui.input = String::new();
                        app.ui.mode = Mode::Command;
                    }
                    Key::Up => {
                        app.transport.step_by(-1);
                    }
                    Key::Down => {
                        app.transport.step_by(1);
                    }
                    _ => {
                        info!("Unhandled key event: {:?}", key);
                    }
                },
            },
            Err(_) => {}
        };

        let ui = app.ui.clone();
        let transport = app.transport.clone();
        let samples = app.samples.clone();
        let steps = app.steps.clone();

        // Drop the app reference so the audio thread can acquire a lock more quickly.
        drop(app);

        terminal
            .draw(|mut f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(
                        [
                            Constraint::Min(0),
                            Constraint::Length(1),
                            Constraint::Length(1),
                        ]
                        .as_ref(),
                    )
                    .split(f.size());

                let command_line = CommandLine {
                    mode: ui.mode,
                    input: ui.input,
                };

                let graph_nodes = Samples(samples);
                let lane = Lane(steps, transport.step);

                let lanes = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(
                        [
                            Constraint::Ratio(1, 8),
                            Constraint::Ratio(1, 8),
                            Constraint::Ratio(1, 8),
                            Constraint::Ratio(1, 8),
                            Constraint::Ratio(1, 8),
                            Constraint::Ratio(1, 8),
                            Constraint::Ratio(1, 8),
                            Constraint::Ratio(1, 8),
                        ]
                        .as_ref(),
                    )
                    .split(chunks[0]);

                f.render_widget(lane.clone(), lanes[0]);
                f.render_widget(lane.clone(), lanes[1]);
                f.render_widget(lane.clone(), lanes[2]);
                f.render_widget(lane.clone(), lanes[3]);
                f.render_widget(lane.clone(), lanes[4]);
                f.render_widget(lane.clone(), lanes[5]);
                f.render_widget(lane.clone(), lanes[6]);
                f.render_widget(lane.clone(), lanes[7]);
                // f.render_widget(graph_nodes, chunks[0]);
                f.render_widget(transport, chunks[1]);
                f.render_widget(command_line, chunks[2]);
            })
            .unwrap();

        terminal.autoresize().unwrap();
    }

    Ok(())
}

struct CommandLine {
    mode: Mode,
    input: String,
}

impl Widget for CommandLine {
    fn render(self, area: Rect, buffer: &mut Buffer) {
        let default = Style::default();
        let (text, style) = match self.mode {
            Mode::Command => {
                let text = format!(":{}", self.input);
                (text, default.bg(Color::Rgb(50, 50, 100)).fg(Color::White))
            }
            Mode::Insert => (
                "insert".to_string(),
                default.bg(Color::Rgb(0, 100, 0)).fg(Color::Black),
            ),
            Mode::Normal => (
                "normal".to_string(),
                default.bg(Color::Rgb(50, 50, 50)).fg(Color::White),
            ),
        };

        buffer.set_background(area, style.bg);
        buffer.set_string(area.x, area.y, text, style);
    }
}

#[derive(Clone)]
struct Lane(Vec<bool>, usize);

impl Widget for Lane {
    fn render(self, area: Rect, buffer: &mut Buffer) {
        let middle = (area.height / 2) - 1;
        let steps = self.0.len();

        for i in 0..area.height {
            let mut style = Style::default();

            let step_index = match (self.1 + i as usize).checked_sub(middle as usize) {
                Some(v) => v as usize,
                None => continue,
            };

            if step_index >= steps {
                continue;
            }

            if step_index % 4 == 0 {
                style = style.fg(Color::Rgb(150, 150, 150));
            }
            if step_index % 16 == 0 {
                style = style.fg(Color::Rgb(180, 250, 180));
            }

            if i == middle {
                style = style.fg(Color::Red);
            }

            buffer.set_string(
                area.x,
                area.y + i as u16,
                format!("{:2.}", step_index),
                style,
            );

            buffer.set_string(area.x + 3, area.y + i as u16, "-- -- --", style);
        }
    }
}

struct Samples(HashMap<String, NodeId>);

impl Widget for Samples {
    fn render(self, area: Rect, buffer: &mut Buffer) {
        let mut line_number = area.y;
        let style = Style::default();

        self.0.iter().for_each(|n| {
            buffer.set_string(area.x, line_number, format!("{}: {:?}", n.0, n.1), style);
            line_number += 1;
        });
    }
}

impl Widget for Transport {
    fn render(self, area: Rect, buffer: &mut Buffer) {
        let ticks = match self.step % 4 {
            0 => format!("{}   ", bar::FULL),
            1 => format!("{}{}  ", bar::FULL, bar::FULL),
            2 => format!("{}{}{} ", bar::FULL, bar::FULL, bar::FULL),
            3 => format!("{}{}{}{}", bar::FULL, bar::FULL, bar::FULL, bar::FULL),
            _ => unreachable!(),
        };

        let play_pause = match self.playing {
            true => "Playing",
            false => "Stopped",
        };

        let style = Style::default();
        let tick_style = if self.step % 4 == 0 {
            Style::default().fg(Color::Blue)
        } else {
            Style::default().fg(Color::Green)
        };

        buffer.set_string(area.width - 4, area.y, ticks, tick_style);
        buffer.set_string(
            area.x,
            area.y,
            format!(
                "{} T: {:.2} Step: {:2.} Frame: {} Seq: {}",
                play_pause, self.tempo, self.step, self.frame, self.sequence_length
            ),
            style,
        );
    }
}
