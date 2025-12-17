use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use log::{debug, error, info, trace};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::Distribution;

use crate::engine::scheduler::Scheduler;
use crate::engine::sequence::Sequence;
use crate::models::qwen3::{Qwen3Config, Qwen3ForCausalLM};
use crate::sampling_params::SamplingParams;

pub trait ModelRunner {
    fn run(&mut self, seqs: &[Rc<RefCell<Sequence>>], is_prefill: bool) -> Vec<usize>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DummyModelRunner;

impl DummyModelRunner {
    pub const fn new() -> Self {
        Self
    }
}

impl ModelRunner for DummyModelRunner {
    fn run(&mut self, seqs: &[Rc<RefCell<Sequence>>], _is_prefill: bool) -> Vec<usize> {
        seqs.iter().map(|seq_rc| seq_rc.borrow().last_token + 1).collect()
    }
}

pub struct Qwen3ModelRunner {
    device: Device,
    pub model: Qwen3ForCausalLM,
    pub cfg: Qwen3Config,
    pub eos_id: usize,
}

impl Qwen3ModelRunner {
    pub fn from_hf_dir(device: Device, model_dir: &str) -> anyhow::Result<Self> {
        info!(
            "Qwen3ModelRunner::from_hf_dir device={:?} model_dir={}",
            device, model_dir
        );
        let model = Qwen3ForCausalLM::from_hf_dir(model_dir, &device)?;
        let cfg = model.model.cfg.clone();
        let eos_id = cfg.eos_token_id;

        Ok(Self {
            device,
            model,
            cfg,
            eos_id,
        })
    }

    fn build_batch(&self, seqs: &[Rc<RefCell<Sequence>>]) -> anyhow::Result<(Tensor, Vec<usize>)> {
        let (all_ids, lens): (Vec<Vec<usize>>, Vec<usize>) = seqs
            .iter()
            .map(|s| {
                let seq = s.borrow();
                let mut ids = Vec::with_capacity(seq.len());
                ids.extend_from_slice(seq.prompt_token_ids());
                ids.extend_from_slice(seq.completion_token_ids());
                let len = ids.len();
                (ids, len)
            })
            .unzip();

        let max_len = lens.iter().copied().max().unwrap_or(0);

        if max_len == 0 {
            let input = Tensor::zeros((seqs.len(), 1), DType::U32, &self.device)?;
            return Ok((input, lens));
        }

        let pad_id = self.eos_id as u32;
        let flat: Vec<u32> = all_ids
            .iter()
            .flat_map(|ids| {
                ids.iter()
                    .map(|&id| id as u32)
                    .chain(std::iter::repeat(pad_id).take(max_len - ids.len()))
            })
            .collect();

        let input = Tensor::from_vec(flat, (seqs.len(), max_len), &self.device)?;
        trace!("build_batch: batch_size={} max_len={}", seqs.len(), max_len);
        debug!("positions: per-seq lengths={:?}", lens);

        Ok((input, lens))
    }

    fn sample_token(&self, logits: &Tensor, seq: &Rc<RefCell<Sequence>>, last_idx: usize) -> usize {
        let fallback = || self.eos_id;

        let logits_last = match logits.i(last_idx) {
            Ok(t) => t,
            Err(_) => return fallback(),
        };

        let temperature = seq.borrow().temperature.max(1e-6) as f32;
        let logits_vec: Vec<f32> = match logits_last.to_vec1() {
            Ok(v) => v,
            Err(_) => return fallback(),
        };

        let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let weights: Vec<f32> = logits_vec
            .iter()
            .map(|&l| ((l - max_logit) / temperature).exp())
            .collect();

        let sum: f32 = weights.iter().sum();

        if !sum.is_finite() || sum <= 0.0 {
            return self.argmax(&logits_vec);
        }

        let normalized: Vec<f32> = weights.iter().map(|w| w / sum).collect();

        match WeightedIndex::new(&normalized) {
            Ok(dist) => {
                let mut rng = rand::rng();
                dist.sample(&mut rng)
            },
            Err(_) => self.argmax(&logits_vec),
        }
    }

    fn argmax(&self, logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(self.eos_id)
    }
}

impl ModelRunner for Qwen3ModelRunner {
    fn run(&mut self, seqs: &[Rc<RefCell<Sequence>>], is_prefill: bool) -> Vec<usize> {
        if seqs.is_empty() {
            return Vec::new();
        }

        trace!("Qwen3ModelRunner::run seqs={} is_prefill={}", seqs.len(), is_prefill);

        let (input_ids, lens) = match self.build_batch(seqs) {
            Ok(v) => v,
            Err(e) => {
                error!("build_batch error: {:?}", e);
                return vec![self.eos_id; seqs.len()];
            },
        };

        let hidden = match self.model.forward(&input_ids) {
            Ok(h) => h,
            Err(e) => {
                error!("model.forward error: {:?}", e);
                return vec![self.eos_id; seqs.len()];
            },
        };

        let logits = match self.model.compute_logits(&hidden) {
            Ok(l) => l,
            Err(e) => {
                error!("compute_logits error: {:?}", e);
                return vec![self.eos_id; seqs.len()];
            },
        };

        seqs.iter()
            .zip(lens.iter())
            .enumerate()
            .map(|(i, (seq, &len))| {
                let last_idx = len.saturating_sub(1);
                match logits.i(i) {
                    Ok(seq_logits) => self.sample_token(&seq_logits, seq, last_idx),
                    Err(_) => self.eos_id,
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub seq_id: usize,
    pub token_ids: Vec<usize>,
    pub text: Option<String>,
}

impl GenerationOutput {
    pub fn new(seq_id: usize, token_ids: Vec<usize>) -> Self {
        Self {
            seq_id,
            token_ids,
            text: None,
        }
    }

    #[must_use]
    pub fn with_text(mut self, text: String) -> Self {
        self.text = Some(text);
        self
    }
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

    pub fn step(&mut self) -> StepOutput {
        let (seqs, is_prefill) = self.scheduler.schedule();
        let token_ids = self.model_runner.run(&seqs, is_prefill);
        self.scheduler.post_process(&seqs, &token_ids);

        let outputs: Vec<_> = seqs
            .iter()
            .filter_map(|seq_rc| {
                let seq = seq_rc.borrow();
                seq.is_finished()
                    .then(|| (seq.seq_id, seq.completion_token_ids().to_vec()))
            })
            .collect();

        let num_tokens = if is_prefill {
            seqs.iter().map(|s| s.borrow().len()).sum::<usize>() as isize
        } else {
            -(seqs.len() as isize)
        };

        StepOutput {
            outputs,
            is_prefill,
            num_tokens,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }

    pub fn generate(&mut self, prompts: Vec<Vec<usize>>, sampling_params: &SamplingParams) -> Vec<GenerationOutput> {
        info!(
            "LLMEngine::generate num_prompts={} max_tokens={}",
            prompts.len(),
            sampling_params.max_tokens
        );

        prompts
            .iter()
            .for_each(|prompt| self.add_request(prompt.clone(), sampling_params));

        let mut outputs: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut prefill_tokens = 0isize;
        let mut decode_tokens = 0isize;
        let start_time = Instant::now();

        while !self.is_finished() {
            let step_start = Instant::now();
            let step = self.step();

            if step.is_prefill {
                prefill_tokens += step.num_tokens;
            } else {
                decode_tokens += -step.num_tokens;
            }

            step.outputs.into_iter().for_each(|(seq_id, token_ids)| {
                debug!("generate: seq_id={} completed with {} tokens", seq_id, token_ids.len());
                outputs.insert(seq_id, token_ids);
            });

            trace!(
                "step: is_prefill={} num_tokens={} elapsed={:?}",
                step.is_prefill,
                step.num_tokens,
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
        sorted_ids.sort_unstable();

        sorted_ids
            .into_iter()
            .filter_map(|seq_id| outputs.remove(&seq_id).map(|ids| GenerationOutput::new(seq_id, ids)))
            .collect()
    }
}

#[derive(Debug)]
pub struct StepOutput {
    pub outputs: Vec<(usize, Vec<usize>)>,
    pub is_prefill: bool,
    pub num_tokens: isize,
}
