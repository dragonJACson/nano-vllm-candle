use std::sync::atomic::{AtomicUsize, Ordering};

use crate::sampling_params::SamplingParams;

static SEQ_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

#[derive(Debug)]
pub struct Sequence {
    pub seq_id: usize,
    block_size: usize,
    pub(crate) block_table: Vec<usize>,
    pub(crate) status: SequenceStatus,
    pub(crate) token_ids: Vec<usize>,
    pub(crate) last_token: usize,
    num_tokens: usize,
    num_prompt_tokens: usize,
    pub(crate) num_cached_tokens: usize,
    pub(crate) temperature: f64,
    pub(crate) max_tokens: usize,
    pub(crate) ignore_eos: bool,
}

impl Sequence {
    pub fn new(token_ids: &[usize], sampling_params: &SamplingParams) -> Self {
        let seq_id = SEQ_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            seq_id,
            block_size: 256,
            block_table: vec![],
            status: SequenceStatus::Waiting,
            token_ids: token_ids.to_vec(),
            last_token: *token_ids.last().unwrap_or(&0),
            num_tokens: token_ids.len(),
            num_prompt_tokens: token_ids.len(),
            num_cached_tokens: 0,
            temperature: sampling_params.temperature,
            max_tokens: sampling_params.max_tokens,
            ignore_eos: sampling_params.ignore_eos,
        }
    }

    pub fn len(&self) -> usize {
        self.num_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    pub fn num_completion_tokens(&self) -> usize {
        self.num_tokens - self.num_prompt_tokens
    }

    pub fn prompt_token_ids(&self) -> &[usize] {
        &self.token_ids[..self.num_prompt_tokens]
    }

    pub fn completion_token_ids(&self) -> &[usize] {
        &self.token_ids[self.num_prompt_tokens..]
    }

    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached_tokens / self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_tokens.div_ceil(self.block_size)
    }

    pub fn last_block_num_tokens(&self) -> usize {
        self.num_tokens - (self.num_blocks() - 1) * self.block_size
    }

    pub fn block(&self, block_id: usize) -> &[usize] {
        debug_assert!(block_id < self.num_blocks());

        let start = block_id * self.block_size;
        let end = ((block_id + 1) * self.block_size).min(self.num_tokens);
        &self.token_ids[start..end]
    }

    pub fn append_token(&mut self, token_id: usize) {
        self.token_ids.push(token_id);
        self.last_token = token_id;
        self.num_tokens += 1;
    }
}
