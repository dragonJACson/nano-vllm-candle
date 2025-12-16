use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use log::{debug, info, trace};

use crate::engine::scheduler::Scheduler;
use crate::engine::sequence::Sequence;
use crate::models::qwen3::{Qwen3Config, Qwen3ForCausalLM};
use crate::sampling_params::SamplingParams;

pub trait ModelRunner {
    fn run(&mut self, seqs: &[Rc<RefCell<Sequence>>], is_prefill: bool) -> Vec<usize>;
}

pub struct DummyModelRunner;

impl DummyModelRunner {
    pub fn new() -> Self {
        debug!("DummyModelRunner::new");
        Self
    }
}

impl Default for DummyModelRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRunner for DummyModelRunner {
    fn run(&mut self, seqs: &[Rc<RefCell<Sequence>>], _is_prefill: bool) -> Vec<usize> {
        seqs.iter()
            .map(|seq_rc| {
                let seq = seq_rc.borrow();
                seq.last_token + 1
            })
            .collect()
    }
}

pub struct Qwen3ModelRunner {
    device: Device,
    model: Qwen3ForCausalLM,
    eos_id: usize,
}

impl Qwen3ModelRunner {
    pub fn new(device: Device, cfg: Qwen3Config, eos_id: usize) -> CandleResult<Self> {
        info!(
            "Qwen3ModelRunner::new device={:?} hidden_size={} layers={}",
            device, cfg.hidden_size, cfg.num_hidden_layers
        );
        let model = Qwen3ForCausalLM::new(cfg, &device)?;
        Ok(Self {
            device,
            model,
            eos_id,
        })
    }

    fn build_batch(&self, seqs: &[Rc<RefCell<Sequence>>]) -> CandleResult<(Tensor, Vec<usize>)> {
        let mut lens = Vec::with_capacity(seqs.len());
        let mut all_ids: Vec<Vec<usize>> = Vec::with_capacity(seqs.len());
        let mut max_len = 0usize;

        for s in seqs {
            let seq = s.borrow();
            let mut ids = Vec::with_capacity(seq.len());
            ids.extend_from_slice(seq.prompt_token_ids());
            ids.extend_from_slice(seq.completion_token_ids());
            max_len = max_len.max(ids.len());
            lens.push(ids.len());
            all_ids.push(ids);
        }

        if max_len == 0 {
            let input = Tensor::zeros((seqs.len(), 1), DType::U32, &self.device)?;
            return Ok((input, lens));
        }

        let pad_id = self.eos_id as u32;
        let mut flat: Vec<u32> = Vec::with_capacity(seqs.len() * max_len);
        for ids in &all_ids {
            for &id in ids {
                flat.push(id as u32);
            }
            for _ in ids.len()..max_len {
                flat.push(pad_id);
            }
        }

        let input = Tensor::from_vec(flat, (seqs.len(), max_len), &self.device)?;
        trace!("build_batch: batch_size={} max_len={}", seqs.len(), max_len);
        Ok((input, lens))
    }
}

impl ModelRunner for Qwen3ModelRunner {
    fn run(&mut self, seqs: &[Rc<RefCell<Sequence>>], is_prefill: bool) -> Vec<usize> {
        if seqs.is_empty() {
            return Vec::new();
        }

        trace!(
            "Qwen3ModelRunner::run seqs={} is_prefill={}",
            seqs.len(),
            is_prefill
        );

        let (input_ids, lens) = match self.build_batch(seqs) {
            Ok(v) => v,
            Err(e) => {
                debug!("build_batch error: {:?}", e);
                return vec![self.eos_id; seqs.len()];
            }
        };

        let hidden = match self.model.forward(&input_ids) {
            Ok(h) => h,
            Err(e) => {
                debug!("model.forward error: {:?}", e);
                return vec![self.eos_id; seqs.len()];
            }
        };

        let logits = match self.model.compute_logits(&hidden) {
            Ok(l) => l,
            Err(e) => {
                debug!("compute_logits error: {:?}", e);
                return vec![self.eos_id; seqs.len()];
            }
        };

        let mut next_ids = Vec::with_capacity(seqs.len());
        for (i, len) in lens.iter().enumerate() {
            let last_idx = if *len == 0 { 0 } else { *len - 1 };

            let logits_i = match logits.i(i) {
                Ok(t) => t,
                Err(_) => {
                    next_ids.push(self.eos_id);
                    continue;
                }
            };
            let logits_last = match logits_i.i(last_idx) {
                Ok(t) => t,
                Err(_) => {
                    next_ids.push(self.eos_id);
                    continue;
                }
            };

            let argmax = match logits_last.argmax(0) {
                Ok(t) => t,
                Err(_) => {
                    next_ids.push(self.eos_id);
                    continue;
                }
            };

            let id = argmax.to_scalar::<u32>().unwrap_or(self.eos_id as u32) as usize;
            next_ids.push(id);
        }

        next_ids
    }
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub seq_id: usize,
    pub token_ids: Vec<usize>,
    pub text: Option<String>,
}

pub struct LLMEngine<R: ModelRunner> {
    pub scheduler: Scheduler,
    pub model_runner: R,
}

impl<R: ModelRunner> LLMEngine<R> {
    pub fn new(scheduler: Scheduler, model_runner: R) -> Self {
        info!("LLMEngine::new");
        Self {
            scheduler,
            model_runner,
        }
    }

    pub fn add_request(&mut self, token_ids: Vec<usize>, sampling_params: &SamplingParams) {
        let seq = Rc::new(RefCell::new(Sequence::new(&token_ids, sampling_params)));
        debug!(
            "LLMEngine::add_request seq_id={} prompt_len={} max_tokens={}",
            seq.borrow().seq_id,
            token_ids.len(),
            sampling_params.max_tokens
        );
        self.scheduler.add(seq);
    }

    pub fn step(&mut self) -> (Vec<(usize, Vec<usize>)>, bool, isize) {
        let (seqs, is_prefill) = self.scheduler.schedule();
        let token_ids = self.model_runner.run(&seqs, is_prefill);
        self.scheduler.post_process(&seqs, &token_ids);

        let mut outputs = Vec::new();
        for seq_rc in &seqs {
            let seq = seq_rc.borrow();
            if seq.is_finished() {
                outputs.push((seq.seq_id, seq.completion_token_ids().to_vec()));
            }
        }

        let num_tokens = if is_prefill {
            seqs.iter().map(|s| s.borrow().len()).sum::<usize>() as isize
        } else {
            -(seqs.len() as isize)
        };

        (outputs, is_prefill, num_tokens)
    }

    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }

    pub fn generate(
        &mut self,
        prompts: Vec<Vec<usize>>,
        sampling_params: &SamplingParams,
    ) -> Vec<GenerationOutput> {
        info!(
            "LLMEngine::generate num_prompts={} max_tokens={}",
            prompts.len(),
            sampling_params.max_tokens
        );

        for prompt in &prompts {
            self.add_request(prompt.clone(), sampling_params);
        }

        let mut outputs: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut prefill_tokens = 0isize;
        let mut decode_tokens = 0isize;
        let start_time = Instant::now();

        while !self.is_finished() {
            let step_start = Instant::now();
            let (step_outputs, is_prefill, num_tokens) = self.step();

            if is_prefill {
                prefill_tokens += num_tokens;
            } else {
                decode_tokens += -num_tokens;
            }

            for (seq_id, token_ids) in step_outputs {
                debug!(
                    "generate: seq_id={} completed with {} tokens",
                    seq_id,
                    token_ids.len()
                );
                outputs.insert(seq_id, token_ids);
            }

            trace!(
                "step: is_prefill={} num_tokens={} elapsed={:?}",
                is_prefill,
                num_tokens,
                step_start.elapsed()
            );
        }

        let total_elapsed = start_time.elapsed();
        info!(
            "generate complete: {} prompts, prefill_tokens={}, decode_tokens={}, elapsed={:?}",
            prompts.len(),
            prefill_tokens,
            decode_tokens,
            total_elapsed
        );

        let mut sorted_ids: Vec<_> = outputs.keys().copied().collect();
        sorted_ids.sort();

        sorted_ids
            .into_iter()
            .map(|seq_id| GenerationOutput {
                seq_id,
                token_ids: outputs.remove(&seq_id).unwrap(),
                text: None,
            })
            .collect()
    }
}
