use std::env;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TPConfig {
    pub size: usize,
    pub rank: usize,
    pub dim: usize,
}

impl Default for TPConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl TPConfig {
    pub const fn new(size: usize, rank: usize, dim: usize) -> Self {
        Self { size, rank, dim }
    }

    pub fn from_env() -> Self {
        let size = env::var("TP_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);

        let rank = env::var("TP_RANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(|r: usize| if r >= size { 0 } else { r })
            .unwrap_or(0);

        Self { size, rank, dim: 0 }
    }

    pub const fn single() -> Self {
        Self::new(1, 0, 0)
    }

    #[must_use]
    pub const fn with_size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    #[must_use]
    pub const fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    #[must_use]
    pub const fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    pub const fn is_distributed(&self) -> bool {
        self.size > 1
    }

    pub const fn shard_size(&self, total: usize) -> usize {
        total / self.size
    }

    pub const fn shard_offset(&self, total: usize) -> usize {
        self.rank * self.shard_size(total)
    }
}

pub fn get_tp() -> TPConfig {
    TPConfig::from_env()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_config() {
        let cfg = TPConfig::single();
        assert_eq!(cfg.size, 1);
        assert_eq!(cfg.rank, 0);
        assert!(!cfg.is_distributed());
    }

    #[test]
    fn test_builder_pattern() {
        let cfg = TPConfig::single().with_size(4).with_rank(2).with_dim(1);
        assert_eq!(cfg.size, 4);
        assert_eq!(cfg.rank, 2);
        assert_eq!(cfg.dim, 1);
        assert!(cfg.is_distributed());
    }

    #[test]
    fn test_shard_calculation() {
        let cfg = TPConfig::new(4, 2, 0);
        assert_eq!(cfg.shard_size(100), 25);
        assert_eq!(cfg.shard_offset(100), 50);
    }
}
