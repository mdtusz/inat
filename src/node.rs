use log::info;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use petgraph::Direction;
use sample::frame::Stereo;
use std::collections::HashMap;

pub type Frame = Stereo<f32>;

pub struct Graph<N: Node> {
    pub graph: DiGraph<N, Connection>,
    root_index: Option<NodeIndex>,
}

pub trait Node {
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
    Default,
    Trigger,
}

pub struct Connection {
    signal: Vec<Frame>,
    kind: ConnectionKind,
}

impl Connection {
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

    pub fn set_root(&mut self, root: NodeIndex) {
        self.root_index = Some(root);
    }

    pub fn add_node(&mut self, node: N) -> NodeIndex {
        self.graph.add_node(node)
    }

    pub fn connect(&mut self, src: NodeIndex, dest: NodeIndex, kind: ConnectionKind) {
        let conn = Connection::new(kind);
        self.graph.add_edge(src, dest, conn);
    }

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
