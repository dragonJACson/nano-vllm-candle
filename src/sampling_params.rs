#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub max_tokens: usize,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_tokens: 64,
            ignore_eos: false,
        }
    }
}

impl SamplingParams {
    pub fn new(temperature: f64, max_tokens: usize) -> Self {
        assert!(temperature > 1e-10, "temperature must be > 0 for sampling");
        Self {
            temperature,
            max_tokens,
            ignore_eos: false,
        }
    }

    pub fn with_ignore_eos(mut self, ignore_eos: bool) -> Self {
        self.ignore_eos = ignore_eos;
        self
    }
}
