use std::io;
use std::io::Write;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use dsp::sample::ToFrameSliceMut;
use dsp::{Graph, Node, NodeIndex};
use log::{debug, info, trace, warn, LevelFilter};
use portaudio as pa;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use tui::backend::{Backend, TermionBackend};
use tui::layout::{Constraint, Direction, Layout};
use tui::style::{Color, Style};
use tui::widgets::{Block, Borders, List, Paragraph, Text, Widget};
use tui::Terminal;
use tui_logger::{TuiLoggerSmartWidget, TuiLoggerWidget};

mod engine;
mod ui;

use engine::{DspNode, Oscillator, Output, Wave, CHANNELS, FRAMES, SAMPLE_HZ};

#[derive(Debug, PartialEq)]
enum Message {
    Done,
}

struct App {
    graph: Graph<[Output; CHANNELS], DspNode>,
    ui: UiState,
}

#[derive(Clone, Debug)]
struct UiState {
    debug: bool,
    mode: Mode,
}

impl UiState {
    fn new() -> Self {
        Self {
            debug: false,
            mode: Mode::Normal,
        }
    }
}

#[derive(Clone, Debug)]
enum Mode {
    Normal,
    Insert,
}

impl App {
    fn new() -> Self {
        let graph = Graph::new();
        let ui = UiState::new();

        Self { graph, ui }
    }
}

fn main() -> Result<(), pa::Error> {
    tui_logger::init_logger(LevelFilter::Info).unwrap();
    tui_logger::set_default_level(LevelFilter::Trace);

    // let mut graph = Graph::new();
    let mut app = App::new();

    let master = app.graph.add_node(DspNode::Gain(1.0));
    app.graph.set_master(Some(master));

    let mut osc1 = Oscillator::new(Wave::Sine, 440.0, 0.0);
    let (_, osc1n) = app.graph.add_input(DspNode::Oscillator(osc1), master);

    // Prepare graph for concurrency.
    let graph = Arc::new(Mutex::new(app.graph));

    let callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
        let buffer: &mut [[Output; CHANNELS]] = buffer.to_frame_slice_mut().unwrap();

        // Insert silence to start.
        dsp::slice::equilibrium(buffer);

        // Compute audio from graph.
        graph.lock().unwrap().audio_requested(buffer, SAMPLE_HZ);

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

    terminal.clear();
    terminal.hide_cursor().unwrap();

    // Input thread.
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || loop {
        let stdin = io::stdin();
        match stdin.keys().next().unwrap() {
            Ok(key) => {
                tx.send(key).unwrap();
            }
            _ => {}
        }
    });

    // UI thread
    loop {
        match rx.try_recv() {
            Ok(key) => match key {
                Key::Char('q') => {
                    terminal.clear();
                    break;
                }
                Key::Char('i') => {
                    terminal.show_cursor().unwrap();
                    app.ui.mode = Mode::Insert;
                }
                Key::Char('~') => {
                    app.ui.debug = !app.ui.debug;
                }
                Key::Esc => {
                    terminal.hide_cursor().unwrap();
                    app.ui.mode = Mode::Normal;
                }
                _ => {
                    info!("Key event: {:?}", key);
                }
            },
            Err(_) => {}
        };

        let ui = app.ui.clone();

        terminal.draw(|mut f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(1)].as_ref())
                .split(f.size());

            if ui.debug {
                TuiLoggerWidget::default()
                    .block(
                        Block::default()
                            .title("Logs ")
                            .title_style(Style::default().fg(Color::Blue)),
                    )
                    .style(Style::default().fg(Color::White).bg(Color::Black))
                    .render(&mut f, chunks[0]);
            } else {
                Block::default().title("Tracker").render(&mut f, chunks[0]);
            }

            let mut default = Style::default();
            let (mode, style) = match ui.mode {
                Mode::Insert => ("insert", default.bg(Color::Green).fg(Color::Black)),
                Mode::Normal => ("normal", default.bg(Color::Black).fg(Color::White)),
            };
            Paragraph::new([Text::raw(mode)].iter())
                .style(style)
                .render(&mut f, chunks[1]);
        });
        terminal.autoresize().unwrap();
    }

    Ok(())
}
