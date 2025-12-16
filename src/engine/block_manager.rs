use crate::engine::sequence::Sequence;

/// Minimal block manager implementation.
///
/// This intentionally ignores real KV cache management and block
/// deduplication from `nano-vllm`. The public API mirrors the Python
/// version so that we can plug in an optimized implementation later
/// without touching the rest of the engine.
pub struct BlockManager {
    block_size: usize,
    /// Total number of blocks reserved for KV cache. Currently unused.
    num_blocks: usize,
}

impl BlockManager {
    /// Create a new block manager.
    ///
    /// `num_blocks` and `block_size` are kept for compatibility with
    /// `nano-vllm`, but the current minimal implementation only uses
    /// `block_size` to maintain `Sequence::num_cached_tokens`.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            block_size,
            num_blocks,
        }
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
