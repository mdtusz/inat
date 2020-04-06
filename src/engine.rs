use log::info;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use petgraph::Direction;

use sample::conv::FromSample;
use sample::frame::{Frame as F, Stereo};
use sample::Sample;

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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ConnectionKind {
    /// Default connection type.
    Default,

    /// A trigger connection type.
    Trigger,
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

impl<N: Node> Graph<N> {
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

#[derive(Debug)]
pub enum DspNode {
    Adsr(Adsr),
    Gain(Gain),
    Gate(Gate),
    Oscillator(Oscillator),
}

/// Primitive wave types.
#[derive(Clone, Copy, Debug)]
pub enum Wave {
    Sine,
    Square,
    Saw,
    Ramp,
    Triangle,
}

/// Basic oscillator for generating wave signals.
#[derive(Clone, Debug)]
pub struct Oscillator {
    frequency: Frequency,
    phase: Phase,
    pub wave: Wave,
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
    attack: u32,
    /// Decay time in samples.
    decay: u32,
    /// Sustain gain level.
    sustain: Gain,
    /// Release time in samples.
    release: u32,
    /// Current global frame.
    current_frame: usize,
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
            attack: 0,
            decay: 0,
            sustain: 1.0,
            release: 0,
            current_frame: 0,
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
        sample_rate: f64,
        start_frame: usize,
        frames: usize,
    ) -> Vec<Frame> {
        let maybe_input = inputs.get(&ConnectionKind::Default);
        let maybe_trigger = inputs.get(&ConnectionKind::Trigger);

        match (maybe_input, maybe_trigger) {
            (Some(input), Some(trigger)) => {
                self.current_frame = start_frame;

                input
                    .iter()
                    .zip(trigger.iter())
                    .map(|(i, t)| {
                        if t.channel(0).unwrap() > &0.0 {
                            self.level = 0.0;
                            self.open_frame = self.current_frame;
                            self.close_frame = usize::max_value();
                        } else {
                            self.close_frame = self.current_frame + self.release as usize;
                        }

                        let mut frame = Frame::equilibrium();

                        // ADSR
                        if (self.open_frame..self.close_frame).contains(&self.current_frame) {
                            frame = *i;
                        } else {
                        }

                        self.current_frame += 1;
                        frame
                    })
                    .collect()
            }
            _ => vec![Frame::equilibrium(); frames],
        }
    }
}

#[derive(Debug)]
pub struct Gate {
    open: bool,
    open_queue: VecDeque<usize>,
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
    pub fn open(&mut self, frame: usize) {
        self.open_queue.push_back(frame);
    }

    pub fn close(&mut self, frame: usize) {
        self.close_queue.push_back(frame);
    }

    pub fn trigger(&mut self, frame: usize) {
        self.open(frame);
        self.close(frame + 4000);
    }

    fn check_open(&mut self, frame: &usize) {
        if let Some(f) = self.open_queue.front() {
            if f == frame {
                self.open_queue.pop_front();
                self.open = true;
            }
        }
    }

    fn check_closed(&mut self, frame: &usize) {
        if let Some(f) = self.close_queue.front() {
            if f == frame {
                self.close_queue.pop_front();
                self.open = false;
            }
        }
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
            .map(|f| {
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
        }
    }
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
