use std::io;
use std::io::Write;
use std::sync::{Arc, Mutex};

use dsp::sample::ToFrameSliceMut;
use dsp::{Graph, Node, NodeIndex};
use log::{debug, info, trace, warn, LevelFilter};
use portaudio as pa;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use tui::backend::TermionBackend;
use tui::Terminal;

mod engine;
mod ui;

use engine::{DspNode, Oscillator, Output, Wave, CHANNELS, FRAMES, SAMPLE_HZ};

#[derive(Debug, PartialEq)]
enum Message {
    Done,
}

struct App {
    graph: Graph<[Output; CHANNELS], DspNode>,
}

impl App {
    fn new() -> Self {
        let mut graph = Graph::new();

        Self { graph }
    }
}

fn main() -> Result<(), pa::Error> {
    let stdout = io::stdout().into_raw_mode().unwrap();
    let stdin = io::stdin();
    let backend = TermionBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    log::set_max_level(LevelFilter::Info);

    // let mut graph = Graph::new();
    let mut app = App::new();

    let master = app.graph.add_node(DspNode::Gain(1.0));
    app.graph.set_master(Some(master));

    let mut osc1 = Oscillator::new(Wave::Sine, 440.0, 0.0);
    let (_, osc1n) = app.graph.add_input(DspNode::Oscillator(osc1), master);

    // Prepare graph for concurrency.
    let app = Arc::new(Mutex::new(app));
    let audio_app = Arc::clone(&app);

    let callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
        let buffer: &mut [[Output; CHANNELS]] = buffer.to_frame_slice_mut().unwrap();

        // Insert silence to start.
        dsp::slice::equilibrium(buffer);

        // Compute audio from graph.
        audio_app
            .lock()
            .unwrap()
            .graph
            .audio_requested(buffer, SAMPLE_HZ);

        pa::Continue
    };

    let pa = pa::PortAudio::new()?;
    let settings =
        pa.default_output_stream_settings::<Output>(CHANNELS as i32, SAMPLE_HZ, FRAMES)?;
    let mut stream = pa.open_non_blocking_stream(settings, callback)?;
    stream.start()?;

    loop {
        let stdin = io::stdin();
        for c in stdin.keys() {
            match c.unwrap() {
                Key::Char('q') => break,
                _ => (),
            }
        }
        break;
    }

    Ok(())
}
