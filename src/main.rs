pub mod engine;
pub mod layers;
pub mod models;
pub mod sampling_params;

use std::error::Error;

use log::{LevelFilter, debug, info};

use crate::engine::llm_engine::{LLMEngine, Qwen3ModelRunner};
use crate::engine::scheduler::{Scheduler, SchedulerConfig};
use crate::models::qwen3::Qwen3Config;
use crate::sampling_params::SamplingParams;
use candle_core::Device;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .parse_default_env()
        .init();

    info!("candle-test inference service starting");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <tokenizer.json> <prompt>", args[0]);
        return Ok(());
    }

    let tokenizer_path = &args[1];
    let prompt = &args[2];

    info!("Loading tokenizer from: {}", tokenizer_path);
    let tokenizer_result = Tokenizer::from_file(tokenizer_path);

    let (input_ids, hf_tokenizer): (Vec<usize>, Option<Tokenizer>) = match tokenizer_result {
        Ok(tk) => {
            let encoding = tk.encode(prompt.as_str(), true)?;
            let ids: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
            info!(
                "Tokenized prompt: {} tokens from {} chars",
                ids.len(),
                prompt.len()
            );
            debug!("Token IDs: {:?}", &ids[..ids.len().min(20)]);
            (ids, Some(tk))
        }
        Err(err) => {
            info!(
                "Failed to load tokenizer from {}: {}. Falling back to byte-level tokenizer.",
                tokenizer_path, err
            );
            let ids: Vec<usize> = prompt.bytes().map(|b| b as usize).collect();
            (ids, None)
        }
    };

    let config = SchedulerConfig {
        max_num_seqs: 1,
        max_num_batched_tokens: 4096,
        eos: 151643,
        num_kvcache_blocks: 0,
        kvcache_block_size: 256,
    };

    let scheduler = Scheduler::new(config);

    let device = Device::new_cuda(0).unwrap_or_else(|_| {
        info!("CUDA device not available, falling back to CPU");
        Device::Cpu
    });
    info!("Using device: {:?}", device);

    let qwen_cfg = Qwen3Config::qwen3_0_6b();
    info!(
        "Qwen3 config: hidden_size={}, layers={}, heads={}",
        qwen_cfg.hidden_size, qwen_cfg.num_hidden_layers, qwen_cfg.num_attention_heads
    );

    let runner = Qwen3ModelRunner::new(device, qwen_cfg, 151643)?;
    let mut engine = LLMEngine::new(scheduler, runner);

    let sampling_params = SamplingParams::new(0.6, 16).with_ignore_eos(true);
    info!(
        "SamplingParams: temperature={}, max_tokens={}",
        sampling_params.temperature, sampling_params.max_tokens
    );

    let outputs = engine.generate(vec![input_ids], &sampling_params);

    for output in outputs {
        info!(
            "Generation complete: seq_id={}, {} tokens",
            output.seq_id,
            output.token_ids.len()
        );

        match &hf_tokenizer {
            Some(tk) => {
                let ids_u32: Vec<u32> = output.token_ids.iter().map(|&id| id as u32).collect();
                let text = tk.decode(&ids_u32, true)?;
                println!("Generated: {}", text);
            }
            None => {
                let bytes: Vec<u8> = output.token_ids.iter().map(|&id| id as u8).collect();
                match String::from_utf8(bytes) {
                    Ok(text) => println!("Generated (byte tokenizer): {}", text),
                    Err(_) => println!("Generated token ids: {:?}", output.token_ids),
                }
            }
        }
    }

    info!("Inference complete");
    Ok(())
}
