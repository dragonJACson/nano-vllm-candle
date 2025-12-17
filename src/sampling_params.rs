#[derive(Debug, Clone, PartialEq)]
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

    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        assert!(temperature > 1e-10, "temperature must be > 0 for sampling");
        self.temperature = temperature;
        self
    }

    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    #[must_use]
    pub fn with_ignore_eos(mut self, ignore_eos: bool) -> Self {
        self.ignore_eos = ignore_eos;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let params = SamplingParams::default();
        assert!((params.temperature - 1.0).abs() < f64::EPSILON);
        assert_eq!(params.max_tokens, 64);
        assert!(!params.ignore_eos);
    }

    #[test]
    fn test_builder_chain() {
        let params = SamplingParams::default()
            .with_temperature(0.7)
            .with_max_tokens(128)
            .with_ignore_eos(true);

        assert!((params.temperature - 0.7).abs() < f64::EPSILON);
        assert_eq!(params.max_tokens, 128);
        assert!(params.ignore_eos);
    }

    #[test]
    #[should_panic(expected = "temperature must be > 0")]
    fn test_zero_temperature_panics() {
        SamplingParams::new(0.0, 64);
    }
}
