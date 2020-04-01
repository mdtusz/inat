use std::f64::consts::PI;

use dsp::sample::ToFrameSliceMut;
use dsp::signal::{ConstHz, Saw, Signal, Sine, Square};
use dsp::{Frame, FromSample, Graph, Node, Sample, Walker};

pub const CHANNELS: usize = 2;
pub const FRAMES: u32 = 512;
pub const SAMPLE_HZ: f64 = 44_100.0;

pub type Frequency = f64;
pub type Phase = f64;
pub type Gain = f32;
pub type Output = f32;

pub enum DspNode {
    Gain(Gain),
    Oscillator(Oscillator),
}

/// Primitive wave types.
#[derive(Clone, Copy, Debug)]
pub enum Wave {
    Silence,
    Sine,
    Square,
    Saw,
    Ramp,
    Triangle,
}

/// Basic oscillator for generating wave signals.
#[derive(Clone)]
pub struct Oscillator {
    frequency: Frequency,
    phase: Phase,
    pub wave: Wave,
}

impl Oscillator {
    /// Creates a new oscillator.
    pub fn new(wave: Wave, frequency: Frequency, phase: Phase) -> Self {
        let signal = dsp::signal::rate(SAMPLE_HZ).const_hz(frequency);
        let square = signal.clone().square();

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

impl Node<[Output; CHANNELS]> for Oscillator {
    fn audio_requested(&mut self, buffer: &mut [[Output; CHANNELS]], sample_hz: f64) {
        match self.wave {
            Wave::Silence => dsp::slice::equilibrium(buffer),
            Wave::Sine => {
                dsp::slice::map_in_place(buffer, |_| {
                    let val = sine(self.phase);
                    self.phase += self.frequency / sample_hz;
                    Frame::from_fn(|_| val)
                });
            }
            Wave::Triangle => {
                dsp::slice::map_in_place(buffer, |_| {
                    let val = triangle(self.phase);
                    self.phase += self.frequency / sample_hz;
                    Frame::from_fn(|_| val)
                });
            }
            Wave::Square => {
                dsp::slice::map_in_place(buffer, |_| {
                    let val = square(self.phase);
                    self.phase = (self.phase + self.frequency / sample_hz) % 1.0;
                    Frame::from_fn(|_| val)
                });
            }
            Wave::Saw => {
                dsp::slice::map_in_place(buffer, |_| {
                    let val = saw(self.phase);
                    self.phase = (self.phase + self.frequency / sample_hz) % 1.0;
                    Frame::from_fn(|_| val)
                });
            }
            Wave::Ramp => {
                dsp::slice::map_in_place(buffer, |_| {
                    let val = ramp(self.phase);
                    self.phase = (self.phase + self.frequency / sample_hz) % 1.0;
                    Frame::from_fn(|_| val)
                });
            }
        }
    }
}

impl Node<[Output; CHANNELS]> for DspNode {
    fn audio_requested(&mut self, buffer: &mut [[Output; CHANNELS]], sample_hz: f64) {
        match self {
            DspNode::Oscillator(osc) => osc.audio_requested(buffer, sample_hz),
            DspNode::Gain(value) => {
                dsp::slice::map_in_place(buffer, |frame| frame.map(|sample| sample.mul_amp(*value)))
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
