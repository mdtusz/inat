use log::info;
use std::collections::HashMap;
use std::f64::consts::PI;

use sample::conv::FromSample;
use sample::frame::Frame as F;
use sample::Sample;

use crate::node::{ConnectionKind, Frame, Node};

pub const CHANNELS: usize = 2;
pub const FRAMES: u32 = 128;
pub const SAMPLE_HZ: f64 = 44_100.0;

pub type Frequency = f64;
pub type Phase = f64;
pub type Gain = f32;
pub type Output = f32;

#[derive(Debug)]
pub enum DspNode {
    Gain(Gain),
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

#[derive(Clone, Debug)]
pub struct Adsr {
    attack: u32,
    decay: u32,
    sustain: Gain,
    release: u32,
    current_frame: usize,
}

impl Default for Adsr {
    fn default() -> Self {
        Self {
            attack: 128,
            decay: 256,
            sustain: 1.0,
            release: 128,
            current_frame: 0,
        }
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

impl Node for DspNode {
    fn compute_signal(
        &mut self,
        inputs: HashMap<ConnectionKind, Vec<Frame>>,
        sample_rate: f64,
        start_frame: usize,
        frames: usize,
    ) -> Vec<Frame> {
        match self {
            DspNode::Oscillator(osc) => {
                osc.compute_signal(inputs, sample_rate, start_frame, frames)
            }
            DspNode::Gain(value) => {
                if let Some(input) = inputs.get(&ConnectionKind::Default) {
                    input
                        .iter()
                        .map(|frame| frame.map(|sample| sample.mul_amp(*value)))
                        .collect()
                } else {
                    vec![Frame::equilibrium(); frames as usize]
                }
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
