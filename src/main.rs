use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Error;
use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::StreamData;
use hound::WavReader;
use log::info;
use midir::{MidiInput, MidiOutput};
use sample::conv::{FromSample, ToFrameSliceMut};
use sample::frame::{Frame, Stereo};
use sample::{signal, Sample, Signal};
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use tui::backend::TermionBackend;
use tui::buffer::Buffer;
use tui::layout::{Constraint, Direction, Layout, Rect};
use tui::style::{Color, Modifier, Style};
use tui::symbols::{bar, block, line};
use tui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Widget};
use tui::Terminal;

pub struct TabsState<'a> {
    pub titles: Vec<&'a str>,
    pub index: usize,
}

impl<'a> TabsState<'a> {
    pub fn new(titles: Vec<&'a str>) -> Self {
        Self { titles, index: 0 }
    }

    pub fn next(&mut self) {
        self.index = (self.index + 1) % self.titles.len();
    }

    pub fn prev(&mut self) {
        if self.index > 0 {
            self.index = self.index - 1;
        } else {
            self.index = self.titles.len() - 1;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    Command,
    Insert,
    Normal,
    Shutdown,
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Command => write!(f, "Command"),
            Self::Insert => write!(f, "Insert"),
            Self::Normal => write!(f, "Normal"),
            Self::Shutdown => write!(f, "Shutdown"),
        }
    }
}

#[derive(Clone, Debug)]
enum Message {
    NoteOn(u8, u8),
    CC(u8),
    ProgramChange(u8),
}

impl Transport {
    fn new() -> Self {
        Self {
            frame: 0,
            next_step_frame: 0,
            playing: false,
            step: 0,
            sequence_length: 64,
            tempo: 120.0,
        }
    }

    fn play_pause(&mut self) {
        self.playing = !self.playing;

        if self.playing {
            self.next_step_frame = self.frame;
        }
    }

    fn on_step(&mut self) -> bool {
        self.frame == self.next_step_frame
    }

    fn step(&mut self) {
        self.step_by(1);
    }

    fn step_by(&mut self, step: i16) {
        let mut offset = step;

        while offset < 0 {
            offset += self.sequence_length as i16;
        }

        self.step = (self.step + offset as usize) % self.sequence_length;
        // Multiply tempo by 4 because we are using a 4/4 time signature.
        self.next_step_frame += (44_100.0 / (self.tempo * 4.0 / 60.0)).floor() as usize;
    }
}

impl Widget for Transport {
    fn render(self, area: Rect, buffer: &mut Buffer) {
        let ticks = match self.step % 4 {
            0 => format!("{}   ", bar::FULL),
            1 => format!("{}{}  ", bar::FULL, bar::FULL),
            2 => format!("{}{}{} ", bar::FULL, bar::FULL, bar::FULL),
            3 => format!("{}{}{}{}", bar::FULL, bar::FULL, bar::FULL, bar::FULL),
            _ => unreachable!(),
        };

        let play_pause = match self.playing {
            true => "Playing",
            false => "Stopped",
        };

        let style = Style::default();
        let tick_style = if self.step % 4 == 0 {
            Style::default().fg(Color::Blue)
        } else {
            Style::default().fg(Color::Green)
        };

        buffer.set_string(area.width - 4, area.y, ticks, tick_style);
        buffer.set_string(
            area.x,
            area.y,
            format!(
                "{} Tempo: {:.2} Step: {:2.} Frame: {} Seq: {}",
                play_pause, self.tempo, self.step, self.frame, self.sequence_length
            ),
            style,
        );
    }
}

struct Focus {
    track: usize,
    step: usize,
}

impl Default for Focus {
    fn default() -> Self {
        Self { track: 0, step: 0 }
    }
}

struct Track {
    steps: Vec<Option<Step>>,
    voice: Voice,
    vu: f32,
}

struct Voice {
    sample: Option<u8>,
    offset: usize,
    playing: bool,
}

impl Voice {
    fn note_on(&mut self) {
        self.offset = 0;
        self.playing = true;
    }

    fn note_off(&mut self) {
        self.playing = false;
    }

    fn get_frame(&mut self, samples: &HashMap<u8, SampleClip>) -> Stereo<f32> {
        if !self.playing {
            return Frame::equilibrium();
        }

        let frame = match self.sample {
            Some(sample) => {
                let sample = samples
                    .get(&sample)
                    .expect(&format!("No sample with address {}", sample));
                let frame = sample.get_frame(self.offset);

                frame
            }
            None => Frame::equilibrium(),
        };

        self.offset += 1;

        frame
    }
}

#[derive(Clone)]
struct Transport {
    frame: usize,
    next_step_frame: usize,
    playing: bool,
    step: usize,
    sequence_length: usize,
    tempo: f64,
}

#[derive(Clone, Debug)]
struct Step {
    instrument: u8,
    note: u8,
}

impl Step {
    fn new(instrument: u8, note: u8) -> Self {
        Self { instrument, note }
    }
}

struct SampleClip {
    samples: Vec<Stereo<f32>>,
    start: usize,
    end: usize,
}

impl SampleClip {
    fn new(path: &str) -> Self {
        let mut sample = WavReader::open(path).expect("File not found!");

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

        let gain = 1.0 / max_amp as f32;

        // Reset the sample to the start position.
        sample.seek(0).expect("Could not start sample at zero.");

        // Normalize the samples between 0 and 1.
        let frames = sample
            .samples::<i32>()
            .filter_map(Result::ok)
            .map(|f| (f as f32).mul_amp(gain));

        // Collect as interleaved frames.
        let samples = signal::from_interleaved_samples_iter(frames)
            .until_exhausted()
            .collect::<Vec<Stereo<f32>>>();

        let start = 0;
        let end = sample.len() as usize;

        Self {
            end: 21000,
            samples,
            start,
        }
    }

    fn get_frame(&self, offset: usize) -> Stereo<f32> {
        let idx = self.start + offset;

        if idx >= self.end {
            Frame::equilibrium()
        } else {
            *self.samples.get(idx).unwrap_or(&Frame::equilibrium())
        }
    }
}

struct System {
    cmd: String,
    midi_buffer: HashMap<usize, Vec<(u8, Message)>>,
    mode: Mode,
    tracks: Vec<Track>,
    samples: HashMap<u8, SampleClip>,
    transport: Transport,
    audio_frame: usize,
    focus: Focus,
}

impl System {
    fn new() -> Self {
        let audio_frame = 0;
        let cmd = String::new();
        let focus = Focus::default();
        let midi_buffer = HashMap::new();
        let mode = Mode::Normal;
        let transport = Transport::new();

        let mut tracks = Vec::new();
        let mut samples = HashMap::new();

        // Construct tracks.
        for i in 0..8 {
            let steps = vec![None; 64];

            let voice = Voice {
                sample: Some(i + 1),
                offset: 0,
                playing: false,
            };

            tracks.push(Track {
                steps,
                voice,
                vu: 0.0,
            });
        }

        // Create 10 dummy samples.
        for i in 1..=10 {
            let path = format!(
                "/home/miklos/Documents/audio/samples/chords/lofi_jazz_piano/{}.wav",
                i
            );
            let clip = SampleClip::new(&path);
            samples.insert(i, clip);
        }

        Self {
            audio_frame,
            cmd,
            focus,
            midi_buffer,
            mode,
            samples,
            tracks,
            transport,
        }
    }

    fn handle_input(&mut self, key: &Key) {
        match self.mode {
            Mode::Command => match key {
                Key::Esc => {
                    self.cmd = String::new();
                    self.mode = Mode::Normal;
                }
                Key::Char('\n') => {
                    if self.cmd == "q" {
                        self.mode = Mode::Shutdown;
                        return;
                    }

                    self.cmd = String::new();
                    self.mode = Mode::Normal;
                }
                Key::Char(c) => {
                    self.cmd = format!("{}{}", self.cmd, c);
                }
                Key::Backspace => {
                    self.cmd.pop();
                }
                Key::Ctrl('u') => {
                    self.cmd = String::new();
                }
                _ => {
                    info!("Unhandled key event: {:?}", key);
                }
            },
            Mode::Insert => match key {
                Key::Esc => {
                    self.cmd = String::new();
                    self.mode = Mode::Normal;
                }
                Key::Char('a') => {
                    self.midi_buffer.insert(
                        self.transport.frame,
                        vec![(self.focus.track as u8, Message::NoteOn(0, 127))],
                    );
                    self.midi_buffer.insert(
                        self.transport.frame + 100,
                        vec![(self.focus.track as u8, Message::NoteOn(0, 0))],
                    );
                }
                _ => {
                    info!("Unhandled key event: {:?}", key);
                }
            },
            Mode::Normal => match key {
                Key::Char(' ') => {
                    self.transport.play_pause();
                }
                Key::Char('h') => {
                    let track_count = self.tracks.len();
                    self.focus.track = (self.focus.track + track_count - 1) % track_count;
                }
                Key::Char('j') => {
                    let step_count = self.tracks[self.focus.track].steps.len();
                    self.focus.step = (self.focus.step + 1) % step_count;
                }
                Key::Char('k') => {
                    let step_count = self.tracks[self.focus.track].steps.len();
                    self.focus.step = (self.focus.step + step_count - 1) % step_count;
                }
                Key::Char('l') => {
                    self.focus.track = (self.focus.track + 1) % self.tracks.len();
                }
                Key::Char('i') => {
                    self.mode = Mode::Insert;
                }
                Key::Char(':') => {
                    self.cmd = String::new();
                    self.mode = Mode::Command;
                }
                Key::Char('\n') => {
                    self.tracks[self.focus.track].steps[self.focus.step] = Some(Step {
                        instrument: 0,
                        note: 0,
                    });
                    let step_count = self.tracks[self.focus.track].steps.len();
                    self.focus.step = (self.focus.step + 4) % step_count;
                }
                Key::Backspace => {
                    self.tracks[self.focus.track].steps[self.focus.step] = None;
                    let step_count = self.tracks[self.focus.track].steps.len();
                    self.focus.step = (self.focus.step + step_count - 4) % step_count;
                }
                _ => {
                    info!("Unhandled key event: {:?}", key);
                }
            },
            Mode::Shutdown => {}
        };
    }

    /// Schedules upcoming midi in a buffer to be consumed by the audio callback loop.
    fn buffer_midi(&mut self) {
        let latency = 2048;

        while self.audio_frame + latency >= self.transport.frame {
            if self.transport.playing && self.transport.on_step() {
                let mut messages = Vec::new();

                self.tracks.iter().enumerate().for_each(|(ch, track)| {
                    match &track.steps[self.transport.step] {
                        Some(step) => {
                            messages.push((ch as u8, Message::ProgramChange(step.instrument)));
                            messages.push((ch as u8, Message::NoteOn(step.note, 127)));
                        }
                        None => {}
                    };
                });

                if !messages.is_empty() {
                    self.midi_buffer.insert(self.transport.frame, messages);
                }

                self.transport.step()
            }

            self.transport.frame += 1;
        }
    }

    /// Computes audio frames for the engine callback to fill the stream.
    fn compute_frames(&mut self, count: usize) -> Vec<Stereo<f32>> {
        let mut frames = Vec::new();

        for i in 0..count {
            let offset = self.audio_frame + i;

            // Apply the midi for the current frame.
            match self.midi_buffer.remove(&offset) {
                Some(messages) => {
                    for message in messages {
                        match message {
                            (ch, Message::CC(cc)) => {}
                            (ch, Message::NoteOn(note, vel)) => {
                                self.tracks[ch as usize].voice.note_on();
                            }
                            _ => {}
                        }
                    }
                }
                None => {}
            };

            let mut frame: Stereo<f32> = Frame::equilibrium();

            // Compute the audio for each track.
            for track in self.tracks.iter_mut() {
                let track_frame = track.voice.get_frame(&self.samples);
                track.vu = (((track_frame[0] + track_frame[1]) / 2.0).abs() + track.vu) / 2.0;
                frame = frame.add_amp(track_frame);
            }

            // frame = frame.scale_amp(1.0 / self.tracks.len() as f32);
            frames.push(frame);
        }

        frames
    }
}

fn main() -> Result<(), Error> {
    let system = System::new();

    // Prepare graph for concurrency.
    let pair = Arc::new((Mutex::new(system), Condvar::new()));

    let engine_pair = Arc::clone(&pair);
    let input_pair = Arc::clone(&pair);
    let sequence_pair = Arc::clone(&pair);
    let midi_pair = Arc::clone(&pair);
    let ui_pair = Arc::clone(&pair);

    let host = cpal::default_host();
    let event_loop = host.event_loop();

    let device = host.default_output_device().expect("No output device!");
    let format = device.default_output_format()?;
    let stream = event_loop.build_output_stream(&device, &format)?;

    event_loop.play_stream(stream)?;

    // Engine thread.
    thread::spawn(move || {
        event_loop.run(move |_stream_id, stream_result| {
            let stream_data = stream_result.expect("No stream data!");
            let mut buffer = match stream_data {
                StreamData::Output {
                    buffer: cpal::UnknownTypeOutputBuffer::F32(buffer),
                } => buffer,
                _ => panic!("Audio output not implemented for this format."),
            };

            let frame_count = buffer.len() / format.channels as usize;
            let buffer = buffer
                .to_frame_slice_mut()
                .expect("Could not create frame slice!");

            let (system, _trigger) = &*engine_pair;
            let mut system = system
                .lock()
                .expect("Could not lock system for audio thread!");

            let frames = system.compute_frames(frame_count);
            sample::slice::write(buffer, &frames);

            system.audio_frame += frame_count;
        });
    });

    // Midi
    let midi_out = MidiOutput::new("midi out").expect("Could not construct midi out.");
    let midi_in = MidiInput::new("midi in").expect("Could not construct midi in.");

    let out_ports = midi_out.ports();
    let in_ports = midi_in.ports();

    let mut midi_out = midi_out.connect(&out_ports[0], "tracker internal").unwrap();
    let mut clock = Instant::now();
    let mut midi_offset = 0;

    let midi_in = midi_in
        .connect(
            &in_ports[0],
            "tracker internal",
            move |timestamp, midi_msg, _data| {
                match midi_msg {
                    &[0xFA] => {
                        clock = Instant::now();
                        midi_offset = timestamp;
                    }
                    _ => {}
                };

                let t = clock.elapsed().as_micros();
                let (system, _trigger) = &*midi_pair;
                let system = system.lock().unwrap();
                println!("MIDI Timestamp {}", timestamp);
                println!("MIDI Offset {}", midi_offset);
                println!("System Instant {}", t);
                println!("Frame {}", system.transport.frame);
                println!("");
            },
            (),
        )
        .unwrap();

    midi_in.close();
    midi_out.close();

    thread::spawn(move || loop {
        let (system, trigger) = &*sequence_pair;
        let mut system = system.lock().unwrap();
        let result = trigger
            .wait_timeout(system, Duration::from_millis(16))
            .expect("Could not get trigger result!");

        system = result.0;

        system.buffer_midi();
    });

    // Input thread.
    thread::spawn(move || loop {
        let stdin = io::stdin();

        match stdin.keys().next().unwrap() {
            Ok(key) => {
                let (system, trigger) = &*input_pair;
                let mut system = system.lock().unwrap();

                system.handle_input(&key);
                trigger.notify_all();
            }
            _ => panic!("Error while reading key input!"),
        }
    });

    let stdout = io::stdout().into_raw_mode().unwrap();
    let backend = TermionBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    // Clear and hide the cursor to start!
    terminal.clear().unwrap();
    terminal.hide_cursor().unwrap();

    // UI thread
    loop {
        let (system, trigger) = &*ui_pair;
        let mut system = system.lock().unwrap();
        let result = trigger
            .wait_timeout(system, Duration::from_millis(32))
            .unwrap();

        system = result.0;

        if system.mode == Mode::Shutdown {
            terminal.clear();
            terminal.show_cursor();
            break;
        }

        terminal
            .draw(|mut f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(
                        [
                            Constraint::Min(0),
                            Constraint::Length(1),
                            Constraint::Length(1),
                        ]
                        .as_ref(),
                    )
                    .split(f.size());

                let command_line = CommandLine {
                    mode: system.mode.clone(),
                    input: system.cmd.clone(),
                };

                let track_count = system.tracks.len() as u32;
                let track_constraint: Vec<Constraint> = system
                    .tracks
                    .iter()
                    .map(|_| Constraint::Ratio(1, track_count))
                    .collect();

                let tracks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(track_constraint)
                    .split(chunks[0]);

                for (i, t) in system.tracks.iter().enumerate() {
                    let mut track = TrackUI {
                        track: t,
                        current: system.transport.step,
                        focus: None,
                    };

                    if system.focus.track == i {
                        track.focus = Some(system.focus.step);
                    }

                    f.render_widget(track, tracks[i]);
                }

                f.render_widget(system.transport.clone(), chunks[1]);
                f.render_widget(command_line, chunks[2]);
            })
            .unwrap();

        terminal.autoresize().unwrap();
    }

    Ok(())
}

struct TrackUI<'a> {
    track: &'a Track,
    current: usize,
    focus: Option<usize>,
}

impl<'a> Widget for TrackUI<'a> {
    fn render(self, area: Rect, buffer: &mut Buffer) {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(3)].as_ref())
            .split(area);

        let top = split[0];
        let bottom = split[1];

        let middle = (top.height / 2) - 1;
        let steps = self.track.steps.len();

        for i in 0..top.height {
            let mut style = Style::default();

            let step_index = match (self.current + i as usize).checked_sub(middle as usize) {
                Some(v) => v as usize,
                None => continue,
            };

            if step_index >= steps {
                continue;
            }

            // Highlight each beat.
            if step_index % 4 == 0 {
                style = style.fg(Color::Rgb(150, 150, 150));
            }

            // Highlight each bar.
            if step_index % 16 == 0 {
                style = style.fg(Color::Rgb(180, 250, 180));
            }

            // Highlight current step.
            if i == middle {
                style = style.fg(Color::Red);
            }

            // Highlight focussed step.
            if let Some(focus) = self.focus {
                if step_index == focus {
                    style = style.fg(Color::Black).bg(Color::Red);
                }
            }

            // Step number.
            buffer.set_string(top.x, top.y + i as u16, format!("{:2.}", step_index), style);

            // Step data.
            let step = &self.track.steps[step_index];
            let text = match step {
                Some(s) => format!("{:02} {:02} --", s.instrument, s.note),
                None => "-- -- --".to_string(),
            };
            buffer.set_string(top.x + 3, top.y + i as u16, text, style);
        }

        let c = (self.track.vu * 255.0 * 2.0) as u8;
        let gain_color = Color::Rgb(c, 140, 140);

        let gain_gauge = Gauge::default()
            .percent((self.track.vu * 100.0) as u16)
            .label("Gain")
            .style(Style::default().fg(gain_color));

        gain_gauge.render(Rect::new(bottom.x, bottom.y, bottom.width, 1), buffer);

        buffer.set_string(
            bottom.x,
            bottom.y + 1,
            format!("{}", self.track.vu),
            Style::default().fg(gain_color),
        );
    }
}

/// VuMeter that displays a value from 0 to 1.
struct VuMeter(f32);

impl Widget for VuMeter {
    fn render(self, area: Rect, buffer: &mut Buffer) {
        let c = (self.0 * 255.0 * 2.0) as u8;
        let color = Color::Rgb(c, 100, 100);
        let style = Style::default().fg(color);
        let text = format!("{}", self.0);
        buffer.set_string(area.x, area.y, text, style);
    }
}

struct CommandLine {
    mode: Mode,
    input: String,
}

impl Widget for CommandLine {
    fn render(self, area: Rect, buffer: &mut Buffer) {
        let default = Style::default();
        let (text, style) = match self.mode {
            Mode::Command => {
                let text = format!(":{}", self.input);
                (text, default.bg(Color::Rgb(20, 80, 120)).fg(Color::White))
            }
            Mode::Insert => (
                "insert".to_string(),
                default.bg(Color::Rgb(0, 100, 0)).fg(Color::Black),
            ),
            Mode::Normal => (
                "normal".to_string(),
                default.bg(Color::Rgb(50, 50, 50)).fg(Color::White),
            ),
            Mode::Shutdown => (
                "goodbye".to_string(),
                default.bg(Color::Rgb(255, 50, 50)).fg(Color::White),
            ),
        };

        buffer.set_style(area, style);
        buffer.set_string(area.x, area.y, text, style);
    }
}
