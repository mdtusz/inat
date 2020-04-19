use log::info;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use petgraph::Direction;

use hound::WavReader;
use sample::conv::FromSample;
use sample::frame::{Frame as F, Stereo};
use sample::{signal, Sample, Signal};

pub const CHANNELS: usize = 2;
pub const FRAMES: u32 = 256;
pub const SAMPLE_HZ: f64 = 44_100.0;

pub type Frequency = f64;
pub type Phase = f64;
pub type Gain = f32;
pub type Output = f32;
pub type Frame = Stereo<f32>;

/// General graph struct for storing and processing nodes within the DSP chain.
///
/// The graph consists of nodes implementing `Node`, accepting multiple input
/// connections and producing one output connection.
pub struct Graph<N: Node> {
    /// The root directed graph tree.
    ///
    /// TODO: This should not be public.
    pub graph: DiGraph<N, Connection>,

    /// The graph root index where final audio will be sourced from.
    /// This should be a master gain node.
    root_index: Option<NodeIndex>,
}

impl<N: Node> Graph<N> {
    /// Create a new and empty graph.
    pub fn new() -> Self {
        let graph = DiGraph::default();

        Self {
            graph,
            root_index: None,
        }
    }

    /// Sets the root index where all audio should be sourced from.
    pub fn set_root(&mut self, root: NodeIndex) {
        self.root_index = Some(root);
    }

    /// Adds a new node to the graph and returns it's index.
    pub fn add_node(&mut self, node: N) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Connects a source and destination together.
    pub fn connect(&mut self, src: NodeIndex, dest: NodeIndex, kind: ConnectionKind) {
        let conn = Connection::new(kind);
        self.graph.add_edge(src, dest, conn);
    }

    /// Computes the final audio from the root node.
    pub fn compute(&mut self, sample_rate: f64, start_frame: usize, frames: usize) -> Vec<Frame> {
        let mut traversal = Topo::new(&self.graph);

        while let Some(node_idx) = traversal.next(&self.graph) {
            let mut incoming_edges = self
                .graph
                .neighbors_directed(node_idx, Direction::Incoming)
                .detach();

            let mut inputs = HashMap::new();
            while let Some(edge_idx) = incoming_edges.next_edge(&self.graph) {
                let connection = &self.graph[edge_idx];
                inputs.insert(connection.kind.clone(), connection.signal.to_vec());
            }

            let node = &mut self.graph[node_idx];
            let node_signal = node.compute_signal(inputs, sample_rate, start_frame, frames);

            if node_idx == self.root_index.expect("No root node set!") {
                return node_signal;
            }

            let mut outgoing_edges = self
                .graph
                .neighbors_directed(node_idx, Direction::Outgoing)
                .detach();

            while let Some(edge_idx) = outgoing_edges.next_edge(&self.graph) {
                self.graph[edge_idx].signal = node_signal.clone();
            }
        }

        unreachable!("Error processing audio. No root node found!");
    }
}

pub trait Node {
    /// Compute the node signal for a given frame buffer.
    fn compute_signal(
        &mut self,
        inputs: HashMap<ConnectionKind, Vec<Frame>>,
        sample_rate: f64,
        start_frame: usize,
        frames: usize,
    ) -> Vec<Frame>;
}

/// The edge type linking graph nodes together.
pub struct Connection {
    /// The signal produced from upstream of this connection edge.
    signal: Vec<Frame>,

    /// The type of connection this edge makes with it's downstream neighbor.
    kind: ConnectionKind,
}

impl Connection {
    /// Creates a new connection of the specified kind.
    fn new(kind: ConnectionKind) -> Self {
        Self {
            signal: Vec::new(),
            kind: kind,
        }
    }
}

impl Default for Connection {
    fn default() -> Self {
        Self::new(ConnectionKind::Default)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ConnectionKind {
    /// Default connection type.
    Default,

    /// A trigger connection type.
    Trigger,
}

#[derive(Debug)]
pub enum DspNode {
    Adsr(Adsr),
    Gain(Gain),
    Gate(Gate),
    Oscillator(Oscillator),
    Sampler(Sampler),
}

impl Node for DspNode {
    fn compute_signal(
        &mut self,
        inputs: HashMap<ConnectionKind, Vec<Frame>>,
        sample_rate: f64,
        start_frame: usize,
        frames: usize,
    ) -> Vec<Frame> {
        match self {
            DspNode::Adsr(adsr) => adsr.compute_signal(inputs, sample_rate, start_frame, frames),
            DspNode::Gain(value) => {
                if let Some(input) = inputs.get(&ConnectionKind::Default) {
                    input
                        .iter()
                        .map(|frame| frame.map(|sample| sample.mul_amp(*value)))
                        .collect()
                } else {
                    vec![Frame::equilibrium(); frames]
                }
            }
            DspNode::Gate(gate) => gate.compute_signal(inputs, sample_rate, start_frame, frames),
            DspNode::Oscillator(osc) => {
                osc.compute_signal(inputs, sample_rate, start_frame, frames)
            }
            DspNode::Sampler(sampler) => {
                sampler.compute_signal(inputs, sample_rate, start_frame, frames)
            }
        }
    }
}

pub struct Sampler {
    sample: WavReader<BufReader<File>>,
    gain: f32,
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("type", &"Sample".to_string())
            .finish()
    }
}

impl Sampler {
    pub fn new() -> Self {
        let mut sample =
            WavReader::open("/home/miklos/Documents/audio/samples/chords/lofi_jazz_piano/16.wav")
                .expect("File not found!");

        // Find the max amplitude of the sample for automatic gain adjustment.
        let max_amp = sample
            .samples::<i32>()
            .max_by(|a, b| {
                a.as_ref()
                    .unwrap_or(&0)
                    .abs()
                    .cmp(&b.as_ref().unwrap_or(&0).abs())
            })
            .unwrap()
            .unwrap();

        // Reset the sample to the start position.
        sample.seek(0).unwrap();

        let gain = 1.0 / max_amp as f32;

        Self { sample, gain }
    }
}

impl Node for Sampler {
    fn compute_signal(
        &mut self,
        inputs: HashMap<ConnectionKind, Vec<Frame>>,
        sample_rate: f64,
        start_frame: usize,
        frames: usize,
    ) -> Vec<Frame> {
        let gain = self.gain;
        let samples = self
            .sample
            .samples::<i32>()
            .filter_map(Result::ok)
            .map(|f| (f as f32).mul_amp(gain));

        let mut sample_frames = signal::from_interleaved_samples_iter(samples).until_exhausted();

        let mut out = Vec::new();

        for _ in 0..frames {
            match sample_frames.next() {
                Some(f) => {
                    out.push(f);
                }
                None => {
                    out.push(F::equilibrium());
                }
            }
        }

        out
    }
}

/// Basic oscillator for generating wave signals.
#[derive(Clone, Debug)]
pub struct Oscillator {
    /// Oscillator frequency.
    frequency: Frequency,

    /// Wave phase.
    phase: Phase,

    /// Wave shape for the oscillator.
    wave: Wave,
}

impl Oscillator {
    /// Creates a new oscillator.
    pub fn new(wave: Wave, frequency: Frequency, phase: Phase) -> Self {
        Self {
            frequency,
            phase,
            wave,
        }
    }
}

impl Default for Oscillator {
    /// Creates a default sine wave oscillator with A440.
    fn default() -> Self {
        Self::new(Wave::Sine, 440.0, 0.0)
    }
}

impl Node for Oscillator {
    fn compute_signal(
        &mut self,
        inputs: HashMap<ConnectionKind, Vec<Frame>>,
        sample_rate: f64,
        start_frame: usize,
        frames: usize,
    ) -> Vec<Frame> {
        let mut buffer = vec![F::equilibrium(); frames as usize];
        let sr = sample_rate;
        match self.wave {
            Wave::Sine => {
                sample::slice::map_in_place(&mut buffer, |_| {
                    let val = sine(self.phase);
                    self.phase += self.frequency / sr;
                    F::from_fn(|_| val)
                });
                buffer.to_vec()
            }
            Wave::Triangle => {
                sample::slice::map_in_place(&mut buffer, |_| {
                    let val = triangle(self.phase);
                    self.phase += self.frequency / sr;
                    F::from_fn(|_| val)
                });
                buffer.to_vec()
            }
            Wave::Square => {
                sample::slice::map_in_place(&mut buffer, |_| {
                    let val = square(self.phase);
                    self.phase = (self.phase + self.frequency / sr) % 1.0;
                    F::from_fn(|_| val)
                });
                buffer.to_vec()
            }
            Wave::Saw => {
                sample::slice::map_in_place(&mut buffer, |_| {
                    let val = saw(self.phase);
                    self.phase = (self.phase + self.frequency / sr) % 1.0;
                    F::from_fn(|_| val)
                });
                buffer.to_vec()
            }
            Wave::Ramp => {
                sample::slice::map_in_place(&mut buffer, |_| {
                    let val = ramp(self.phase);
                    self.phase = (self.phase + self.frequency / sr) % 1.0;
                    F::from_fn(|_| val)
                });
                buffer.to_vec()
            }
            Wave::Noise => {
                let mut current_frame = start_frame;
                sample::slice::map_in_place(&mut buffer, |_| {
                    let val = noise(current_frame);
                    current_frame += 1;
                    F::from_fn(|_| val)
                });
                buffer.to_vec()
            }
        }
    }
}

/// ADSR envelope generator.
///
/// The ADSR node has a gate (or trigger) input, and outputs a buffer of frames which can be
/// routed to any other node - most commonly to a gain node to provide an envelope, similar
/// to routing an ADSR to a VCA in a modular synth.
#[derive(Clone, Debug)]
pub struct Adsr {
    /// Attack time in samples.
    attack: usize,

    /// Decay time in samples.
    decay: usize,

    /// Sustain gain level.
    sustain: Gain,

    /// Release time in samples.
    release: usize,

    /// Gate open frame.
    open_frame: usize,

    /// Gate close frame.
    close_frame: usize,

    /// Amp level.
    level: Gain,
}

impl Default for Adsr {
    fn default() -> Self {
        Self {
            attack: 22050,
            decay: 11025,
            sustain: 0.5,
            release: 22050,
            open_frame: 0,
            close_frame: 0,
            level: 0.0,
        }
    }
}

impl Node for Adsr {
    fn compute_signal(
        &mut self,
        inputs: HashMap<ConnectionKind, Vec<Frame>>,
        _sample_rate: f64,
        start_frame: usize,
        frames: usize,
    ) -> Vec<Frame> {
        let maybe_input = inputs.get(&ConnectionKind::Default);
        let maybe_trigger = inputs.get(&ConnectionKind::Trigger);

        match (maybe_input, maybe_trigger) {
            (Some(input), Some(trigger)) => {
                input
                    .iter()
                    .zip(trigger.iter())
                    .enumerate()
                    .map(|(i, (inp, t))| {
                        let current_frame = start_frame + i;
                        let gate = t.channel(0).unwrap();

                        if gate == &1.0 && self.close_frame < current_frame {
                            self.open_frame = current_frame;
                            self.close_frame = usize::max_value();
                        } else if gate == &0.0 && self.close_frame == usize::max_value() {
                            self.close_frame = current_frame;
                        }

                        let decay_start = (self.open_frame + self.attack).max(self.close_frame);
                        let sustain_start = decay_start + self.decay;
                        let release_start = decay_start.max(self.close_frame);
                        let release_end = release_start
                            .checked_add(self.release)
                            .unwrap_or(usize::max_value());

                        if self.open_frame == self.close_frame {
                            return F::equilibrium();
                        }

                        // Attack
                        if (self.open_frame..decay_start).contains(&current_frame) {
                            let delta = current_frame - self.open_frame;
                            let attack_level = (delta as f32 / self.attack as f32).min(1.0);
                            self.level = (self.level + attack_level).min(1.0);
                            return inp.scale_amp(self.level);
                        }

                        // Release
                        if (release_start..release_end).contains(&current_frame) {
                            let delta = current_frame - self.close_frame;
                            let level = 1.0 - (delta as f32 / self.release as f32).min(1.0);
                            self.level = self.sustain * level;
                            return inp.scale_amp(self.level);
                        }

                        // Decay
                        if (decay_start..sustain_start).contains(&current_frame) {
                            let delta = current_frame - decay_start;
                            self.level = 1.0 - (delta as f32 / self.decay as f32).min(1.0);
                            return inp.scale_amp(self.level);
                        }

                        // Sustain
                        if (sustain_start..release_start).contains(&current_frame) {
                            self.level = self.sustain;
                            return inp.scale_amp(self.level);
                        }

                        F::equilibrium()
                    })
                    .collect()
            }
            _ => vec![Frame::equilibrium(); frames],
        }
    }
}

#[derive(Debug)]
pub struct Gate {
    /// Whether or not the gate is currently open.
    open: bool,

    /// Queue of frames for when to open the gate.
    open_queue: VecDeque<usize>,

    /// Queue of frames for when to close the gate.
    close_queue: VecDeque<usize>,
}

impl Default for Gate {
    fn default() -> Self {
        Self {
            open: false,
            open_queue: VecDeque::new(),
            close_queue: VecDeque::new(),
        }
    }
}

impl Gate {
    /// Queue a gate open.
    pub fn open(&mut self, frame: usize) {
        self.open_queue.push_back(frame);
    }

    /// Queue a gate close.
    pub fn close(&mut self, frame: usize) {
        self.close_queue.push_back(frame);
    }

    /// Queue a gate trigger (1 frame total).
    pub fn trigger(&mut self, frame: usize) {
        self.open(frame);
        self.close(frame + 1);
    }

    /// Check whether the gate should open at the given frame and set if so.
    fn check_open(&mut self, frame: &usize) -> bool {
        if let Some(f) = self.open_queue.front() {
            if f == frame {
                self.open_queue.pop_front();
                self.open = true;
                return true;
            }
        }

        false
    }

    /// Check whether the gate should close at the given frame and set if so.
    fn check_closed(&mut self, frame: &usize) -> bool {
        if let Some(f) = self.close_queue.front() {
            if f == frame {
                self.close_queue.pop_front();
                self.open = false;
                return true;
            }
        }

        false
    }
}

impl Node for Gate {
    fn compute_signal(
        &mut self,
        inputs: HashMap<ConnectionKind, Vec<Frame>>,
        sample_rate: f64,
        start_frame: usize,
        frames: usize,
    ) -> Vec<Frame> {
        let mut current_frame = start_frame;
        let frames = vec![Frame::equilibrium(); frames];

        frames
            .iter()
            .map(|_f| {
                self.check_open(&current_frame);
                self.check_closed(&current_frame);

                current_frame += 1;

                if self.open {
                    Frame::from_fn(|_| 1.0)
                } else {
                    Frame::equilibrium()
                }
            })
            .collect()
    }
}

/// Primitive wave types.
#[derive(Clone, Copy, Debug)]
pub enum Wave {
    Noise,
    Ramp,
    Sine,
    Square,
    Saw,
    Triangle,
}

/// Ramp wave generator.
fn ramp<S: Sample>(phase: Phase) -> S
where
    S: Sample + FromSample<f32>,
{
    ((phase * 2.0 - 1.0) as f32).to_sample::<S>()
}

/// Saw wave generator.
fn saw<S: Sample>(phase: Phase) -> S
where
    S: Sample + FromSample<f32>,
{
    ((1.0 - phase * 2.0) as f32).to_sample::<S>()
}

/// Triangle wave generator.
fn triangle<S: Sample>(phase: Phase) -> S
where
    S: Sample + FromSample<f32>,
{
    ((phase * PI * 2.0).cos().asin() as f32).to_sample::<S>()
}

/// Sine wave generator.
fn sine<S: Sample>(phase: Phase) -> S
where
    S: Sample + FromSample<f32>,
{
    ((phase * PI * 2.0).sin() as f32).to_sample::<S>()
}

/// Square wave generator.
fn square<S: Sample>(phase: Phase) -> S
where
    S: Sample + FromSample<f32>,
{
    if phase < 0.5 {
        (-1.0).to_sample::<S>()
    } else {
        1.0.to_sample::<S>()
    }
}

/// Noise generator.
fn noise<S: Sample>(seed: usize) -> S
where
    S: Sample + FromSample<f32>,
{
    const PRIME_1: usize = 15_731;
    const PRIME_2: usize = 789_221;
    const PRIME_3: usize = 1_376_312_589;
    let x = (seed << 13) ^ seed;
    let noise = 1.0
        - (x.wrapping_mul(
            x.wrapping_mul(x)
                .wrapping_mul(PRIME_1)
                .wrapping_add(PRIME_2),
        )
        .wrapping_add(PRIME_3)
            & 0x7fffffff) as f32
            / 1_073_741_824.0;

    noise.to_sample::<S>()
}
