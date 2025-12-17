use crate::engine::sequence::Sequence;

#[derive(Debug, Clone)]
pub struct BlockManagerConfig {
    pub num_blocks: usize,
    pub block_size: usize,
}

impl Default for BlockManagerConfig {
    fn default() -> Self {
        Self {
            num_blocks: 0,
            block_size: 256,
        }
    }
}

impl BlockManagerConfig {
    pub const fn new(num_blocks: usize, block_size: usize) -> Self {
        Self { num_blocks, block_size }
    }
}

/// Minimal block manager implementation.
///
/// This intentionally ignores real KV cache management and block
/// deduplication from `nano-vllm`. The public API mirrors the Python
/// version so that we can plug in an optimized implementation later
/// without touching the rest of the engine.
#[derive(Debug)]
pub struct BlockManager {
    config: BlockManagerConfig,
}

impl Default for BlockManager {
    fn default() -> Self {
        Self::new(BlockManagerConfig::default())
    }
}

impl From<BlockManagerConfig> for BlockManager {
    fn from(config: BlockManagerConfig) -> Self {
        Self::new(config)
    }
}

impl BlockManager {
    pub fn new(config: BlockManagerConfig) -> Self {
        Self { config }
    }

    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.config.num_blocks
    }

    /// Check whether we can allocate KV cache blocks for this sequence.
    ///
    /// The minimal implementation always returns `true` to avoid
    /// scheduling failures. Memory / capacity control can be added later.
    pub fn can_allocate(&self, _seq: &Sequence) -> bool {
        true
    }

    /// Allocate KV cache blocks for the given sequence.
    ///
    /// We only update book-keeping fields on `Sequence` to mimic
    /// successful allocation.
    pub fn allocate(&mut self, seq: &mut Sequence) {
        let num_blocks = seq.num_blocks();
        seq.block_table.clear();
        seq.block_table.extend(0..num_blocks);
        seq.num_cached_tokens = seq.len();
    }

    /// Deallocate all KV cache blocks associated with this sequence.
    pub fn deallocate(&mut self, seq: &mut Sequence) {
        seq.block_table.clear();
        seq.num_cached_tokens = 0;
    }

    /// Check if we can append one more token for this sequence.
    ///
    /// For now this is always `true`. A future implementation can
    /// enforce block capacity limits here.
    pub fn can_append(&self, _seq: &Sequence) -> bool {
        true
    }

    /// Update KV cache meta-data after appending tokens.
    ///
    /// The minimal implementation is a no-op. Once a real block manager
    /// is implemented, this is where we will extend the last block or
    /// allocate a new one.
    pub fn may_append(&mut self, _seq: &mut Sequence) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let mgr = BlockManager::default();
        assert_eq!(mgr.block_size(), 256);
        assert_eq!(mgr.num_blocks(), 0);
    }

    #[test]
    fn test_from_config() {
        let config = BlockManagerConfig::new(16, 128);
        let mgr = BlockManager::from(config);
        assert_eq!(mgr.num_blocks(), 16);
        assert_eq!(mgr.block_size(), 128);
    }
}
