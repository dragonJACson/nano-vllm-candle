use std::cell::RefCell;
use std::collections::VecDeque;
use std::iter::zip;
use std::rc::Rc;

use log::{debug, trace};

use crate::engine::block_manager::BlockManager;
use crate::engine::sequence::{Sequence, SequenceStatus};

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub eos: usize,
    pub num_kvcache_blocks: usize,
    pub kvcache_block_size: usize,
}

pub struct Scheduler {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub block_manager: BlockManager,
    waiting: VecDeque<Rc<RefCell<Sequence>>>,
    running: VecDeque<Rc<RefCell<Sequence>>>,
    eos: usize,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        debug!(
            "Scheduler::new max_num_seqs={} max_num_batched_tokens={} eos={}",
            config.max_num_seqs, config.max_num_batched_tokens, config.eos
        );
        let block_manager = BlockManager::new(config.num_kvcache_blocks, config.kvcache_block_size);
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
        let seq_id = seq.borrow().seq_id;
        let prompt_len = seq.borrow().len();
        debug!("Scheduler::add seq_id={} prompt_len={}", seq_id, prompt_len);
        self.waiting.push_back(seq);
    }

    pub fn schedule(&mut self) -> (Vec<Rc<RefCell<Sequence>>>, bool) {
        let mut scheduled_seqs: Vec<Rc<RefCell<Sequence>>> = vec![];
        let mut num_seqs = 0usize;
        let mut num_batched_tokens = 0usize;

        // Prefill stage: schedule sequences from waiting queue
        while !self.waiting.is_empty() && num_seqs < self.max_num_seqs {
            let seq = self.waiting.front().unwrap().clone();
            {
                let seq_borrow = seq.borrow();
                if num_batched_tokens + seq_borrow.len() > self.max_num_batched_tokens
                    || !self.block_manager.can_allocate(&seq_borrow)
                {
                    break;
                }
            }

            num_seqs += 1;

            self.block_manager.allocate(&mut seq.borrow_mut());
            num_batched_tokens += seq.borrow().len() - seq.borrow().num_cached_tokens;
            seq.borrow_mut().status = SequenceStatus::Running;

            self.waiting.pop_front();
            self.running.push_back(seq.clone());
            scheduled_seqs.push(seq);
        }

        if !scheduled_seqs.is_empty() {
            trace!(
                "schedule prefill: {} seqs, {} tokens",
                scheduled_seqs.len(),
                num_batched_tokens
            );
            return (scheduled_seqs, true);
        }

        // Decode stage: schedule sequences from running queue
        while !self.running.is_empty() && num_seqs < self.max_num_seqs {
            let seq = self.running.pop_front().unwrap();

            // Python: while not can_append: ... else: schedule
            // The else clause runs if the while loop completes normally (not via break)
            let mut should_schedule = true;

            while !self.block_manager.can_append(&seq.borrow()) {
                if !self.running.is_empty() {
                    let victim = self.running.pop_back().unwrap();
                    self.preempt(victim);
                } else {
                    self.preempt(seq.clone());
                    should_schedule = false;
                    break;
                }
            }

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

        // Python: self.running.extendleft(reversed(scheduled_seqs))
        for seq in scheduled_seqs.iter().rev() {
            self.running.push_front(seq.clone());
        }

        trace!("schedule decode: {} seqs", scheduled_seqs.len());
        (scheduled_seqs, false)
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
        for (seq, token_id) in zip(seqs, token_ids) {
            let (seq_id, finished) = {
                let mut seq_borrow = seq.borrow_mut();
                seq_borrow.append_token(*token_id);

                let is_eos = *token_id == self.eos;
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

            // Match nano-vllm: self.running.remove(seq) immediately
            if finished {
                debug!(
                    "post_process seq_id={} finished, removing from running queue",
                    seq_id
                );
                self.running.retain(|s| s.borrow().seq_id != seq_id);
            }
        }
    }
}
