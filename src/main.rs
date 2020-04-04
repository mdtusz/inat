use std::io;
use std::io::Write;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use dsp::sample::ToFrameSliceMut;
use dsp::{Graph, Node, NodeIndex};
use log::{debug, info, trace, warn, LevelFilter};
use portaudio as pa;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use tui::backend::{Backend, TermionBackend};
use tui::buffer::Buffer;
use tui::layout::{Constraint, Direction, Layout, Rect};
use tui::style::{Color, Modifier, Style};
use tui::symbols::line::{VERTICAL_LEFT, VERTICAL_RIGHT};
use tui::widgets::{Block, Borders, List, Paragraph, SelectableList, Text, Widget};
use tui::Terminal;
use tui_logger::{TuiLoggerSmartWidget, TuiLoggerWidget};

mod engine;
mod ui;

use engine::{DspNode, Oscillator, Output, Wave, CHANNELS, FRAMES, SAMPLE_HZ};

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
        let mut graph = Graph::new();
        let ui = UiState::new();

        let master = graph.add_node(DspNode::Gain(1.0));
        graph.set_master(Some(master));

        Self { graph, ui }
    }

    fn master(&self) -> &DspNode {
        let master_index = self.graph.master_index().expect("No master gain!");
        self.graph.node(master_index).unwrap()
    }
}

fn main() -> Result<(), pa::Error> {
    tui_logger::init_logger(LevelFilter::Info).unwrap();
    tui_logger::set_default_level(LevelFilter::Trace);

    // let mut graph = Graph::new();
    let mut app = App::new();

    // let mut osc1 = Oscillator::new(Wave::Sine, 440.0, 0.0);
    // let (_, osc1n) = app.graph.add_input(DspNode::Oscillator(osc1), master);

    // Prepare graph for concurrency.
    let pair = Arc::new((Mutex::new(app), Condvar::new()));

    let audio_pair = Arc::clone(&pair);
    let ui_pair = Arc::clone(&pair);
    let input_pair = Arc::clone(&pair);

    let callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
        let buffer: &mut [[Output; CHANNELS]] = buffer.to_frame_slice_mut().unwrap();

        // Insert silence to start.
        dsp::slice::equilibrium(buffer);

        let (app, trigger) = &*audio_pair;
        let mut app = app.lock().unwrap();

        // Compute audio from graph.
        app.graph.audio_requested(buffer, SAMPLE_HZ);

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
            .wait_timeout(app, Duration::from_millis(16))
            .unwrap();

        app = result.0;

        match rx.try_recv() {
            Ok(key) => match key {
                Key::Char('q') => {
                    terminal.clear();
                    break;
                }
                Key::Char('i') => match app.ui.mode {
                    Mode::Insert => {}
                    Mode::Normal => {
                        terminal.show_cursor().unwrap();
                        app.ui.mode = Mode::Insert;
                    }
                },
                Key::Char('~') => {
                    app.ui.debug = !app.ui.debug;
                }
                Key::Esc => {
                    terminal.hide_cursor().unwrap();
                    app.ui.mode = Mode::Normal;
                }
                _ => {
                    info!("Unhandled key event: {:?}", key);
                }
            },
            Err(_) => {}
        };

        let ui = app.ui.clone();
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
                    .render(&mut f, main_view[0]);

                Block::default()
                    .title(&make_title("Synth"))
                    .borders(Borders::TOP)
                    .render(&mut f, main_view[1]);
            }

            let mut default = Style::default();
            let (mode, style) = match ui.mode {
                Mode::Insert => ("insert", default.bg(Color::Green).fg(Color::Black)),
                Mode::Normal => (
                    "normal",
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

struct Tracker<'b> {
    block: Option<Block<'b>>,
}

impl<'b> Tracker<'b> {
    fn new() -> Self {
        Tracker { block: None }
    }

    fn block(&mut self, block: Block<'b>) -> &mut Self {
        self.block = Some(block);
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

        let x = [
            "C12 ", "---", "---", "---", "C12 ", "---", "---", "---", "C12 ", "---", "---", "---",
            "C12 ", "---", "---", "---",
        ];

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

        SelectableList::default()
            .items(&x)
            .style(Style::default().fg(Color::Rgb(94, 255, 238)))
            .highlight_style(Style::default().bg(Color::Rgb(20, 50, 20)))
            .select(Some(0))
            .draw(tracker_cols[0], buffer);

        SelectableList::default()
            .items(&x)
            .style(Style::default().fg(Color::Rgb(245, 230, 66)))
            .highlight_style(Style::default().bg(Color::Rgb(20, 50, 20)))
            .select(Some(0))
            .draw(tracker_cols[1], buffer);

        SelectableList::default()
            .items(&x)
            .style(Style::default().fg(Color::Rgb(255, 94, 236)))
            .highlight_style(Style::default().bg(Color::Rgb(20, 50, 20)))
            .select(Some(0))
            .draw(tracker_cols[2], buffer);

        SelectableList::default()
            .items(&x)
            .style(Style::default().fg(Color::Rgb(94, 190, 250)))
            .highlight_style(
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Rgb(50, 200, 50))
                    .modifier(Modifier::BOLD),
            )
            .select(Some(0))
            .draw(tracker_cols[3], buffer);
    }
}

fn make_title(s: &str) -> String {
    format!("{} {} {}", VERTICAL_LEFT, s, VERTICAL_RIGHT)
}
