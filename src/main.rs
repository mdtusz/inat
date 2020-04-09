use std::io;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use log::{debug, info, trace, warn, LevelFilter};
use petgraph::graph::NodeIndex;
use portaudio as pa;
use sample::conv::ToFrameSliceMut;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use tui::backend::TermionBackend;
use tui::buffer::Buffer;
use tui::layout::{Constraint, Direction, Layout, Rect};
use tui::style::{Color, Modifier, Style};
use tui::symbols::line::{VERTICAL_LEFT, VERTICAL_RIGHT};
use tui::widgets::{Block, Borders, List, Paragraph, SelectableList, Table, Text, Widget};
use tui::Terminal;
use tui_logger::{TuiLoggerSmartWidget, TuiLoggerWidget};

mod engine;
mod ui;

use crate::engine::{
    Adsr, ConnectionKind, DspNode, Frame as F, Gate, Graph, Oscillator, Output, Wave, CHANNELS,
    FRAMES, SAMPLE_HZ,
};

struct Transport {
    pub playing: bool,
}

impl Transport {
    fn new() -> Self {
        Self { playing: false }
    }
    fn play_pause(&mut self) {
        self.playing = !self.playing;
    }
}
struct App {
    graph: Graph<DspNode>,
    ui: UiState,
    trig: NodeIndex,
    transport: Transport,
}

#[derive(Clone, Debug)]
struct UiState {
    debug: bool,
    mode: Mode,
    step: usize,
    input: String,
}

impl UiState {
    fn new() -> Self {
        Self {
            debug: false,
            mode: Mode::Normal,
            step: 0,
            input: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
enum Mode {
    Command,
    Insert,
    Normal,
}

impl App {
    fn new() -> Self {
        let mut graph = Graph::new();
        let ui = UiState::new();

        let master = graph.add_node(DspNode::Gain(1.0));
        graph.set_root(master);

        let trig = graph.add_node(DspNode::Gate(Gate::default()));
        let osc = graph.add_node(DspNode::Oscillator(Oscillator::default()));
        let adsr = graph.add_node(DspNode::Adsr(Adsr::default()));

        graph.connect(trig, adsr, ConnectionKind::Trigger);
        graph.connect(osc, adsr, ConnectionKind::Default);
        graph.connect(adsr, master, ConnectionKind::Default);

        let transport = Transport::new();

        Self {
            graph,
            ui,
            transport,
            trig,
        }
    }
}

fn main() -> Result<(), pa::Error> {
    tui_logger::init_logger(LevelFilter::Info).unwrap();
    tui_logger::set_default_level(LevelFilter::Trace);

    let mut app = App::new();

    // Prepare graph for concurrency.
    let pair = Arc::new((Mutex::new(app), Condvar::new()));

    let audio_pair = Arc::clone(&pair);
    let ui_pair = Arc::clone(&pair);
    let input_pair = Arc::clone(&pair);

    let mut tempo: f64 = 100.0;
    let mut current_frame: usize = 0;

    // (frame, step) tuple.
    let mut next_step = (0, 0);

    let callback = move |pa::OutputStreamCallbackArgs { buffer, frames, .. }| {
        let buffer: &mut [F] = buffer.to_frame_slice_mut().unwrap();

        let (app, trigger) = &*audio_pair;
        let mut app = app.lock().unwrap();

        let initial_frame = current_frame;
        let next_cycle = current_frame + frames;

        while current_frame < next_cycle {
            if app.transport.playing {
                if current_frame == next_step.0 {
                    // The 4.0 here is the beats-per-bar.
                    next_step.0 += (SAMPLE_HZ / (tempo * 4.0 / 60.0)).round() as usize;
                    next_step.1 = (next_step.1 + 1) % 64;
                    app.ui.step = next_step.1;

                    // How to schedule? This is a bit of a struggle because we need to be able to
                    // struggle per-node. I'm unsure what's the best way to achieve this per-node
                    // though - we could pass a tuple of (frame, Change) or something, but that seems
                    // exceedingly difficult because of the strict constraints on rust.

                    info!("schedule step here! {}", next_step.1);
                    let t = app.trig;

                    if next_step.1 % 4 == 0 {
                        match app.graph.graph.node_weight_mut(t) {
                            Some(n) => match n {
                                DspNode::Gate(g) => {
                                    g.trigger(current_frame);
                                }
                                _ => {}
                            },
                            _ => {}
                        };
                    }
                }
            }

            current_frame += 1;
        }

        // Compute audio from graph.
        let frames = app.graph.compute(SAMPLE_HZ, initial_frame, frames);

        sample::slice::write(buffer, &frames);

        pa::Continue
    };

    let pa = pa::PortAudio::new()?;
    let settings =
        pa.default_output_stream_settings::<Output>(CHANNELS as i32, SAMPLE_HZ, FRAMES)?;
    let mut stream = pa.open_non_blocking_stream(settings, callback)?;
    // Audio thread
    stream.start()?;

    let stdout = io::stdout().into_raw_mode().unwrap();
    // let stdin = io::stdin();
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
                            terminal.clear();
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
                        terminal.show_cursor().unwrap();
                        app.ui.mode = Mode::Command;
                    }
                    _ => {
                        info!("Unhandled key event: {:?}", key);
                    }
                },
            },
            Err(_) => {}
        };

        let ui = app.ui.clone();

        // Drop the app reference so the audio thread can acquire a lock more quickly.
        drop(app);

        terminal.draw(|mut f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(0), Constraint::Length(1)].as_ref())
                .split(f.size());

            if ui.debug {
                TuiLoggerWidget::default()
                    .block(
                        Block::default()
                            .title("Logs ")
                            .title_style(Style::default().fg(Color::Blue))
                            .borders(Borders::TOP),
                    )
                    .style(Style::default().fg(Color::White).bg(Color::Black))
                    .render(&mut f, chunks[0]);
            } else {
                let main_view = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
                    .split(chunks[0]);

                Tracker::new()
                    .block(
                        Block::default()
                            .title(&make_title("Tracker"))
                            .borders(Borders::TOP),
                    )
                    .active(ui.step)
                    .render(&mut f, main_view[0]);

                Source::new()
                    .block(
                        Block::default()
                            .title(&make_title("Source"))
                            .borders(Borders::TOP),
                    )
                    .render(&mut f, main_view[1]);
            }

            // Command line.
            let default = Style::default();
            let (mode, style) = match ui.mode {
                Mode::Command => {
                    let text = format!(":{}", ui.input);
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

            Paragraph::new([Text::raw(mode)].iter())
                .style(style)
                .render(&mut f, chunks[1]);
        });
        terminal.autoresize().unwrap();
    }

    Ok(())
}

struct Source<'b> {
    block: Option<Block<'b>>,
}

impl<'b> Source<'b> {
    fn new() -> Self {
        Source { block: None }
    }

    fn block(&mut self, block: Block<'b>) -> &mut Self {
        self.block = Some(block);
        self
    }
}

impl<'b> Widget for Source<'b> {
    fn draw(&mut self, area: Rect, buffer: &mut Buffer) {
        let block_area = match self.block {
            Some(ref mut b) => {
                b.draw(area, buffer);
                b.inner(area)
            }
            None => area,
        };
    }
}

struct Tracker<'b> {
    block: Option<Block<'b>>,
    steps: [Step; 64],
    active: usize,
}

impl<'b> Tracker<'b> {
    fn new() -> Self {
        Tracker {
            block: None,
            steps: [Step::empty(); 64],
            active: 0,
        }
    }

    fn block(&mut self, block: Block<'b>) -> &mut Self {
        self.block = Some(block);
        self
    }

    fn active(&mut self, active: usize) -> &mut Self {
        self.active = active;
        self
    }
}

impl<'b> Widget for Tracker<'b> {
    fn draw(&mut self, area: Rect, buffer: &mut Buffer) {
        let block_area = match self.block {
            Some(ref mut b) => {
                b.draw(area, buffer);
                b.inner(area)
            }
            None => area,
        };

        let tracker_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Length(4),
                    Constraint::Length(4),
                    Constraint::Length(4),
                    Constraint::Length(4),
                ]
                .as_ref(),
            )
            .split(block_area);

        let notes = self
            .steps
            .iter()
            .map(|step| {
                step.note
                    .map(|n| n.to_string())
                    .unwrap_or("---".to_string())
            })
            .collect::<Vec<String>>();

        SelectableList::default()
            .items(&notes)
            .style(Style::default().fg(Color::Rgb(94, 255, 238)))
            .highlight_style(Style::default().bg(Color::Rgb(20, 50, 20)))
            .select(Some(self.active))
            .draw(tracker_cols[0], buffer);

        let instruments = self
            .steps
            .iter()
            .map(|step| {
                step.instrument
                    .map(|n| n.to_string())
                    .unwrap_or("---".to_string())
            })
            .collect::<Vec<String>>();

        SelectableList::default()
            .items(&instruments)
            .style(Style::default().fg(Color::Rgb(245, 230, 66)))
            .highlight_style(Style::default().bg(Color::Rgb(20, 50, 20)))
            .select(Some(self.active))
            .draw(tracker_cols[1], buffer);
    }
}

#[derive(Clone, Copy)]
struct Step {
    instrument: Option<u8>,
    note: Option<u8>,
}

impl Step {
    fn empty() -> Self {
        Self {
            instrument: None,
            note: None,
        }
    }
}

fn make_title(s: &str) -> String {
    format!("{} {} {}", VERTICAL_LEFT, s, VERTICAL_RIGHT)
}
