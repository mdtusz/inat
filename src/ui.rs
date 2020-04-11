#[derive(Clone, Debug)]
pub enum Mode {
    Command,
    Insert,
    Normal,
}

#[derive(Clone, Debug)]
pub struct UiState {
    pub debug: bool,
    pub mode: Mode,
    pub step: usize,
    pub input: String,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            debug: false,
            mode: Mode::Normal,
            step: 0,
            input: String::new(),
        }
    }
}
