use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use log::{debug, trace};

use crate::engine::block_manager::{BlockManager, BlockManagerConfig};
use crate::engine::sequence::{Sequence, SequenceStatus};

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub eos: usize,
    pub num_kvcache_blocks: usize,
    pub kvcache_block_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 1,
            max_num_batched_tokens: 4096,
            eos: 0,
            num_kvcache_blocks: 0,
            kvcache_block_size: 256,
        }
    }
}

impl SchedulerConfig {
    #[must_use]
    pub fn with_max_num_seqs(mut self, max_num_seqs: usize) -> Self {
        self.max_num_seqs = max_num_seqs;
        self
    }

    #[must_use]
    pub fn with_max_num_batched_tokens(mut self, max_num_batched_tokens: usize) -> Self {
        self.max_num_batched_tokens = max_num_batched_tokens;
        self
    }

    #[must_use]
    pub fn with_eos(mut self, eos: usize) -> Self {
        self.eos = eos;
        self
    }

    #[must_use]
    pub fn with_kvcache(mut self, num_blocks: usize, block_size: usize) -> Self {
        self.num_kvcache_blocks = num_blocks;
        self.kvcache_block_size = block_size;
        self
    }
}

pub struct Scheduler {
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    block_manager: BlockManager,
    waiting: VecDeque<Rc<RefCell<Sequence>>>,
    running: VecDeque<Rc<RefCell<Sequence>>>,
    eos: usize,
}

impl From<SchedulerConfig> for Scheduler {
    fn from(config: SchedulerConfig) -> Self {
        Self::new(config)
    }
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        debug!(
            "Scheduler::new max_num_seqs={} max_num_batched_tokens={} eos={}",
            config.max_num_seqs, config.max_num_batched_tokens, config.eos
        );
        let block_manager = BlockManager::new(BlockManagerConfig::new(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        ));
        Self {
            max_num_seqs: config.max_num_seqs,
            max_num_batched_tokens: config.max_num_batched_tokens,
            block_manager,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            eos: config.eos,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub fn add(&mut self, seq: Rc<RefCell<Sequence>>) {
        let (seq_id, prompt_len) = {
            let s = seq.borrow();
            (s.seq_id, s.len())
        };
        debug!("Scheduler::add seq_id={} prompt_len={}", seq_id, prompt_len);
        self.waiting.push_back(seq);
    }

    pub fn schedule(&mut self) -> (Vec<Rc<RefCell<Sequence>>>, bool) {
        if let Some(result) = self.try_schedule_prefill() {
            return result;
        }
        self.schedule_decode()
    }

    fn try_schedule_prefill(&mut self) -> Option<(Vec<Rc<RefCell<Sequence>>>, bool)> {
        let mut scheduled_seqs = Vec::new();
        let mut num_seqs = 0usize;
        let mut num_batched_tokens = 0usize;

        while !self.waiting.is_empty() && num_seqs < self.max_num_seqs {
            let seq = self.waiting.front()?.clone();

            let can_schedule = {
                let seq_borrow = seq.borrow();
                num_batched_tokens + seq_borrow.len() <= self.max_num_batched_tokens
                    && self.block_manager.can_allocate(&seq_borrow)
            };

            if !can_schedule {
                break;
            }

            num_seqs += 1;
            self.block_manager.allocate(&mut seq.borrow_mut());

            let tokens_to_process = {
                let s = seq.borrow();
                s.len() - s.num_cached_tokens
            };
            num_batched_tokens += tokens_to_process;

            seq.borrow_mut().status = SequenceStatus::Running;

            self.waiting.pop_front();
            self.running.push_back(seq.clone());
            scheduled_seqs.push(seq);
        }

        if scheduled_seqs.is_empty() {
            return None;
        }

        trace!(
            "schedule prefill: {} seqs, {} tokens",
            scheduled_seqs.len(),
            num_batched_tokens
        );
        Some((scheduled_seqs, true))
    }

    fn schedule_decode(&mut self) -> (Vec<Rc<RefCell<Sequence>>>, bool) {
        let mut scheduled_seqs = Vec::new();
        let mut num_seqs = 0usize;

        while !self.running.is_empty() && num_seqs < self.max_num_seqs {
            let seq = self.running.pop_front().unwrap();

            let should_schedule = self.ensure_can_append(&seq);

            if should_schedule {
                num_seqs += 1;
                self.block_manager.may_append(&mut seq.borrow_mut());
                scheduled_seqs.push(seq);
            }
        }

        assert!(
            !scheduled_seqs.is_empty(),
            "decode stage should schedule at least one sequence"
        );

        for seq in scheduled_seqs.iter().rev() {
            self.running.push_front(seq.clone());
        }

        trace!("schedule decode: {} seqs", scheduled_seqs.len());
        (scheduled_seqs, false)
    }

    fn ensure_can_append(&mut self, seq: &Rc<RefCell<Sequence>>) -> bool {
        while !self.block_manager.can_append(&seq.borrow()) {
            if let Some(victim) = self.running.pop_back() {
                self.preempt(victim);
            } else {
                self.preempt(seq.clone());
                return false;
            }
        }
        true
    }

    pub fn preempt(&mut self, seq: Rc<RefCell<Sequence>>) {
        let seq_id = seq.borrow().seq_id;
        debug!("Scheduler::preempt seq_id={}", seq_id);

        {
            let mut seq_borrow = seq.borrow_mut();
            seq_borrow.status = SequenceStatus::Waiting;
            self.block_manager.deallocate(&mut seq_borrow);
        }

        self.waiting.push_front(seq);
    }

    pub fn post_process(&mut self, seqs: &[Rc<RefCell<Sequence>>], token_ids: &[usize]) {
        seqs.iter().zip(token_ids.iter()).for_each(|(seq, &token_id)| {
            self.process_single_sequence(seq, token_id);
        });
    }

    fn process_single_sequence(&mut self, seq: &Rc<RefCell<Sequence>>, token_id: usize) {
        let (seq_id, finished) = {
            let mut seq_borrow = seq.borrow_mut();
            seq_borrow.append_token(token_id);

            let is_eos = token_id == self.eos;
            let reached_max = seq_borrow.num_completion_tokens() >= seq_borrow.max_tokens;
            let finished = (!seq_borrow.ignore_eos && is_eos) || reached_max;

            debug!(
                "post_process seq_id={} token={} is_eos={} reached_max={} completion_tokens={}",
                seq_borrow.seq_id,
                token_id,
                is_eos,
                reached_max,
                seq_borrow.num_completion_tokens()
            );

            if finished {
                seq_borrow.status = SequenceStatus::Finished;
                self.block_manager.deallocate(&mut seq_borrow);
            }

            (seq_borrow.seq_id, finished)
        };

        if finished {
            debug!("post_process seq_id={} finished, removing from running queue", seq_id);
            self.running.retain(|s| s.borrow().seq_id != seq_id);
        }
    }
}
