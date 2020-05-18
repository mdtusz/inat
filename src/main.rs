use std::collections::HashMap;
use std::fmt;
use std::io;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Error;
use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::StreamData;
use log::info;
use midir::{MidiInput, MidiOutput};
use sample::conv::ToFrameSliceMut;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use tui::backend::TermionBackend;
use tui::buffer::Buffer;
use tui::layout::{Constraint, Direction, Layout, Rect};
use tui::style::{Color, Modifier, Style};
use tui::symbols::{bar, block, line};
use tui::widgets::{Block, Borders, Paragraph, Sparkline, Text, Widget};
use tui::Terminal;

mod engine;

use crate::engine::{ConnectionKind, DspNode, Graph, NodeId};

#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    Command,
    Insert,
    Normal,
    Shutdown,
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Command => write!(f, "Command"),
            Self::Insert => write!(f, "Insert"),
            Self::Normal => write!(f, "Normal"),
            Self::Shutdown => write!(f, "Shutdown"),
        }
    }
}

#[derive(Clone, Debug)]
enum Message {
    NoteOn(u8, u8),
    CC(u8),
}

#[derive(Clone, Debug)]
enum Channel {
    Ch1,
    Ch2,
    Ch3,
    Ch4,
    Ch5,
    Ch6,
    Ch7,
    Ch8,
    Ch9,
    Ch10,
    Ch11,
    Ch12,
    Ch13,
    Ch14,
    Ch15,
    Ch16,
}

impl Into<Channel> for usize {
    fn into(self) -> Channel {
        match self {
            0 => Channel::Ch1,
            1 => Channel::Ch2,
            2 => Channel::Ch3,
            3 => Channel::Ch4,
            4 => Channel::Ch5,
            5 => Channel::Ch6,
            6 => Channel::Ch7,
            7 => Channel::Ch8,
            8 => Channel::Ch9,
            9 => Channel::Ch10,
            10 => Channel::Ch11,
            11 => Channel::Ch12,
            12 => Channel::Ch13,
            13 => Channel::Ch14,
            14 => Channel::Ch15,
            15 => Channel::Ch16,
            _ => panic!("Invalid channel!"),
        }
    }
}

impl Transport {
    fn new() -> Self {
        Self {
            frame: 0,
            next_step_frame: 0,
            playing: false,
            step: 0,
            sequence_length: 64,
            tempo: 120.0,
        }
    }

    fn play_pause(&mut self) {
        self.playing = !self.playing;

        if self.playing {
            self.next_step_frame = self.frame;
        }
    }

    fn on_step(&mut self) -> bool {
        self.frame == self.next_step_frame
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
        // Multiply tempo by 4 because we are using a 4/4 time signature.
        self.next_step_frame += (44_100.0 / (self.tempo * 4.0 / 60.0)).floor() as usize;
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
                "{} Tempo: {:.2} Step: {:2.} Frame: {} Seq: {}",
                play_pause, self.tempo, self.step, self.frame, self.sequence_length
            ),
            style,
        );
    }
}

struct Track {
    gain: NodeId,
    steps: Vec<Option<Step>>,
}

#[derive(Clone)]
struct Transport {
    frame: usize,
    next_step_frame: usize,
    playing: bool,
    step: usize,
    sequence_length: usize,
    tempo: f64,
}

#[derive(Clone)]
struct Step {
    instrument: u8,
    note: u8,
}

impl Step {
    fn new(instrument: u8, note: u8) -> Self {
        Self { instrument, note }
    }
}

struct Sample {
    start: usize,
    end: usize,
}

struct System {
    graph: Graph<DspNode>,
    cmd: String,
    midi_buffer: HashMap<usize, Vec<(Channel, Message)>>,
    mode: Mode,
    tracks: Vec<Track>,
    transport: Transport,
    audio_frame: usize,
}

impl System {
    fn new() -> Self {
        let mut graph = Graph::new();
        let mut tracks = Vec::new();

        let master = graph.add_node(DspNode::Gain(1.0));
        graph.set_root(master);

        let audio_frame = 0;
        let cmd = String::new();
        let midi_buffer = HashMap::new();
        let mode = Mode::Normal;
        let transport = Transport::new();

        for _ in 0..8 {
            let steps = vec![None; 64];
            let gain = graph.add_node(DspNode::Gain(1.0));
            graph.connect(gain, master, ConnectionKind::Default);
            tracks.push(Track { gain, steps });
        }

        Self {
            audio_frame,
            cmd,
            graph,
            midi_buffer,
            mode,
            tracks,
            transport,
        }
    }

    fn handle_input(&mut self, key: &Key) {
        match self.mode {
            Mode::Command => match key {
                Key::Esc => {
                    self.cmd = String::new();
                    self.mode = Mode::Normal;
                }
                Key::Char('\n') => {
                    if self.cmd == "q" {
                        self.mode = Mode::Shutdown;
                        return;
                    }

                    self.cmd = String::new();
                    self.mode = Mode::Normal;
                }
                Key::Char(c) => {
                    self.cmd = format!("{}{}", self.cmd, c);
                }
                Key::Backspace => {
                    self.cmd.pop();
                }
                Key::Ctrl('u') => {
                    self.cmd = String::new();
                }
                _ => {
                    info!("Unhandled key event: {:?}", key);
                }
            },
            Mode::Insert => match key {
                Key::Esc => {
                    self.cmd = String::new();
                    self.mode = Mode::Normal;
                }
                Key::Char('a') => {
                    let note_on = Message::NoteOn(65, 127);
                    let note_off = Message::NoteOn(0, 0);
                    self.midi_buffer
                        .insert(self.transport.frame, vec![(Channel::Ch1, note_on)]);
                    self.midi_buffer
                        .insert(self.transport.frame + 100, vec![(Channel::Ch1, note_off)]);
                }
                _ => {
                    info!("Unhandled key event: {:?}", key);
                }
            },
            Mode::Normal => match key {
                Key::Char(' ') => {
                    self.transport.play_pause();
                }
                Key::Char('i') => {
                    self.mode = Mode::Insert;
                }
                Key::Char(':') => {
                    self.cmd = String::new();
                    self.mode = Mode::Command;
                }
                _ => {
                    info!("Unhandled key event: {:?}", key);
                }
            },
            Mode::Shutdown => {}
        };
    }

    fn buffer_midi(&mut self) {
        let latency = 1024;

        while self.audio_frame + latency > self.transport.frame {
            if self.transport.playing && self.transport.on_step() {
                let mut messages = Vec::new();

                self.tracks.iter().enumerate().for_each(|(i, t)| {
                    let channel: Channel = i.into();

                    match &t.steps[self.transport.step] {
                        Some(step) => {
                            messages.push((channel.clone(), Message::CC(step.instrument)));
                            messages.push((channel.clone(), Message::NoteOn(step.note, 127)));
                        }
                        None => {}
                    };
                });

                if !messages.is_empty() {
                    self.midi_buffer.insert(self.transport.frame, messages);
                }

                self.transport.step()
            }

            self.transport.frame += 1;
        }
    }
}

fn main() -> Result<(), Error> {
    let system = System::new();

    // Prepare graph for concurrency.
    let pair = Arc::new((Mutex::new(system), Condvar::new()));

    let engine_pair = Arc::clone(&pair);
    let input_pair = Arc::clone(&pair);
    let sequence_pair = Arc::clone(&pair);
    let midi_pair = Arc::clone(&pair);
    let ui_pair = Arc::clone(&pair);

    let host = cpal::default_host();
    let event_loop = host.event_loop();

    let device = host.default_output_device().expect("No output device!");
    let format = device.default_output_format()?;
    let stream = event_loop.build_output_stream(&device, &format)?;

    event_loop.play_stream(stream)?;

    // Engine thread.
    thread::spawn(move || {
        // Monotonically increasing frame count. Risk of overflow is essentially zero, unless the
        // program runs for literally years.
        let sr = format.sample_rate.0 as f64;

        event_loop.run(move |_stream_id, stream_result| {
            let stream_data = stream_result.expect("No stream data!");
            let mut buffer = match stream_data {
                StreamData::Output {
                    buffer: cpal::UnknownTypeOutputBuffer::F32(buffer),
                } => buffer,
                _ => panic!("Audio output not implemented for this format."),
            };

            let frame_count = buffer.len() / format.channels as usize;
            let buffer = buffer.to_frame_slice_mut().unwrap();

            let (system, _trigger) = &*engine_pair;
            let mut system = system.lock().unwrap();

            let initial_frame = system.audio_frame;

            for i in 0..frame_count {
                let idx = initial_frame + i;
                match system.midi_buffer.remove(&idx) {
                    Some(msg) => {
                        println!("Hit! {} {:?}", idx, msg);
                    }
                    None => {
                        // println!("Miss! ({:?})", idx)
                    }
                };
            }

            let frames = system.graph.compute(sr, initial_frame, frame_count);
            sample::slice::write(buffer, &frames);

            system.audio_frame += frame_count;
        });
    });

    // Midi
    let midi_out = MidiOutput::new("midi out").expect("Could not construct midi out.");
    let midi_in = MidiInput::new("midi in").expect("Could not construct midi in.");

    let out_ports = midi_out.ports();
    let in_ports = midi_in.ports();

    let mut midi_out = midi_out.connect(&out_ports[0], "tracker internal").unwrap();
    let mut clock = Instant::now();
    let mut midi_offset = 0;

    let midi_in = midi_in
        .connect(
            &in_ports[0],
            "tracker internal",
            move |timestamp, midi_msg, _data| {
                match midi_msg {
                    &[0xFA] => {
                        clock = Instant::now();
                        midi_offset = timestamp;
                    }
                    _ => {}
                };

                let t = clock.elapsed().as_micros();
                let (system, _trigger) = &*midi_pair;
                let system = system.lock().unwrap();
                println!("MIDI Timestamp {}", timestamp);
                println!("MIDI Offset {}", midi_offset);
                println!("System Instant {}", t);
                println!("Frame {}", system.transport.frame);
                println!("");
            },
            (),
        )
        .unwrap();

    midi_in.close();
    midi_out.close();

    thread::spawn(move || loop {
        let (system, trigger) = &*sequence_pair;
        let mut system = system.lock().unwrap();
        let result = trigger
            .wait_timeout(system, Duration::from_millis(16))
            .unwrap();

        system = result.0;

        system.buffer_midi();
    });

    // Input thread.
    thread::spawn(move || loop {
        let stdin = io::stdin();

        match stdin.keys().next().unwrap() {
            Ok(key) => {
                let (system, trigger) = &*input_pair;
                let mut system = system.lock().unwrap();

                system.handle_input(&key);
                trigger.notify_all();
            }
            _ => panic!("Error while reading key input!"),
        }
    });

    let stdout = io::stdout().into_raw_mode().unwrap();
    let backend = TermionBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    // Clear and hide the cursor to start!
    terminal.clear().unwrap();
    terminal.hide_cursor().unwrap();

    // UI thread
    loop {
        let (system, trigger) = &*ui_pair;
        let mut system = system.lock().unwrap();
        let result = trigger
            .wait_timeout(system, Duration::from_millis(32))
            .unwrap();

        system = result.0;

        if system.mode == Mode::Shutdown {
            terminal.clear();
            terminal.show_cursor();
            break;
        }

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
                    mode: system.mode.clone(),
                    input: system.cmd.clone(),
                };

                let track_count = system.tracks.len() as u32;
                let track_constraint: Vec<Constraint> = system
                    .tracks
                    .iter()
                    .map(|_| Constraint::Ratio(1, track_count))
                    .collect();

                let tracks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(track_constraint)
                    .split(chunks[0]);

                for (i, t) in system.tracks.iter().enumerate() {
                    let track = TrackUI(t, system.transport.step);
                    f.render_widget(track, tracks[i]);
                }

                f.render_widget(system.transport.clone(), chunks[1]);
                f.render_widget(command_line, chunks[2]);
            })
            .unwrap();

        terminal.autoresize().unwrap();
    }

    Ok(())
}

struct TrackUI<'a>(&'a Track, usize);

impl<'a> Widget for TrackUI<'a> {
    fn render(self, area: Rect, buffer: &mut Buffer) {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(3)].as_ref())
            .split(area);

        let top = split[0];
        let bottom = split[1];

        let middle = (top.height / 2) - 1;
        let steps = self.0.steps.len();

        for i in 0..top.height {
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

            buffer.set_string(top.x, top.y + i as u16, format!("{:2.}", step_index), style);
            buffer.set_string(top.x + 3, top.y + i as u16, "-- -- --", style);
        }

        buffer.set_string(
            bottom.x,
            bottom.y + 1,
            format!("{:?}", self.0.gain),
            Style::default(),
        );
    }
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
                (text, default.bg(Color::Rgb(20, 80, 120)).fg(Color::White))
            }
            Mode::Insert => (
                "insert".to_string(),
                default.bg(Color::Rgb(0, 100, 0)).fg(Color::Black),
            ),
            Mode::Normal => (
                "normal".to_string(),
                default.bg(Color::Rgb(50, 50, 50)).fg(Color::White),
            ),
            Mode::Shutdown => (
                "goodbye".to_string(),
                default.bg(Color::Rgb(255, 50, 50)).fg(Color::White),
            ),
        };

        buffer.set_background(area, style.bg);
        buffer.set_string(area.x, area.y, text, style);
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
