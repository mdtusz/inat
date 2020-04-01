use std::sync::{mpsc, Arc, Mutex};

use cursive::align::{HAlign, VAlign};
use cursive::direction::Orientation;
use cursive::event::Key;
use cursive::menu::MenuTree;
use cursive::traits::*;
use cursive::views::*;
use cursive::Cursive;
use dsp::sample::ToFrameSliceMut;
use dsp::{Graph, Node, NodeIndex};
use log::{debug, info, trace, warn, LevelFilter};
use portaudio as pa;

mod engine;
mod ui;

use engine::{DspNode, Oscillator, Output, Wave, CHANNELS, FRAMES, SAMPLE_HZ};

#[derive(Debug, PartialEq)]
enum Message {
    Done,
}

fn main() -> Result<(), pa::Error> {
    cursive::logger::init();
    log::set_max_level(LevelFilter::Info);

    let mut siv = Cursive::default();
    let mut graph = Graph::new();

    let master = graph.add_node(DspNode::Gain(1.0));
    graph.set_master(Some(master));

    let mut osc1 = Oscillator::new(Wave::Sine, 440.0, 0.0);
    let (_, osc1n) = graph.add_input(DspNode::Oscillator(osc1), master);

    // Prepare graph for concurrency.
    let graph = Arc::new(Mutex::new(graph));
    let audio_graph = Arc::clone(&graph);

    // let osc2 = Oscillator::new(Wave::Ramp, 109.0, 0.5);
    // graph.add_input(DspNode::Oscillator(osc2), master);

    let callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
        let buffer: &mut [[Output; CHANNELS]] = buffer.to_frame_slice_mut().unwrap();

        // Insert silence to start.
        dsp::slice::equilibrium(buffer);

        // Compute audio from graph.
        audio_graph
            .lock()
            .unwrap()
            .audio_requested(buffer, SAMPLE_HZ);

        pa::Continue
    };

    let pa = pa::PortAudio::new()?;
    let settings =
        pa.default_output_stream_settings::<Output>(CHANNELS as i32, SAMPLE_HZ, FRAMES)?;
    let mut stream = pa.open_non_blocking_stream(settings, callback)?;
    stream.start()?;

    let theme = cursive::theme::load_theme_file("./theme.toml")
        .expect(&format!("Invalid theme file path: {}", "./theme.toml"));
    siv.set_theme(theme);

    siv.add_global_callback('q', |s| s.quit());
    siv.add_global_callback(Key::Esc, |s| s.select_menubar());
    siv.add_global_callback('~', |s| s.toggle_debug_console());

    siv.menubar().add_subtree(
        "File",
        MenuTree::new()
            .leaf("New Project", |s| s.add_layer(Dialog::info("New project.")))
            .leaf("Save", |s| s.add_layer(Dialog::info("Saved.")))
            .leaf("Save As", |s| s.add_layer(Dialog::info("Save as.")))
            .leaf("Quit", |s| s.quit()),
    );

    const TRACKS: usize = 8;

    let mut track_container = LinearLayout::horizontal();

    for i in 0..TRACKS {
        let track_select = wave_select(osc1n, Arc::clone(&graph));
        let freq_input = EditView::new().fixed_width(5);
        let mut_graph = Arc::clone(&graph);
        let mute = Checkbox::new().on_change(move |s, checked| {
            match mut_graph.lock().unwrap().node_mut(osc1n) {
                Some(node) => match node {
                    DspNode::Oscillator(o) => {
                        o.wave = Wave::Silence;
                    }
                    _ => {}
                },
                None => {}
            };
        });

        let l = LinearLayout::vertical()
            .child(track_select)
            .child(freq_input)
            .child(mute);

        track_container.add_child(l);
    }

    siv.add_layer(track_container.full_width().full_height());

    siv.run();

    Ok(())
}

fn wave_select(
    osc_node: NodeIndex,
    graph: Arc<Mutex<Graph<[Output; CHANNELS], DspNode>>>,
) -> SelectView<Wave> {
    let mut wave_select = SelectView::new().v_align(VAlign::Top);
    wave_select.add_item("Sine", Wave::Sine);
    wave_select.add_item("Saw", Wave::Saw);
    wave_select.add_item("Ramp", Wave::Ramp);
    wave_select.add_item("Square", Wave::Square);
    wave_select.add_item("Triangle", Wave::Triangle);
    wave_select.set_popup(false);

    wave_select.set_on_select(move |_, wave| {
        match graph.lock().unwrap().node_mut(osc_node) {
            Some(node) => match node {
                DspNode::Oscillator(o) => {
                    o.wave = *wave;
                    info!("Wave set to {:?}", wave);
                }
                _ => {}
            },
            None => {}
        };
    });

    wave_select
}

trait Mutate {}

impl Mutate for Graph<[Output; CHANNELS], DspNode> {}
