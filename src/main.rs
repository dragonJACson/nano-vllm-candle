pub mod engine;
pub mod layers;
pub mod models;
pub mod sampling_params;
pub mod tp;

use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use log::{debug, info};
use logforth::append;
use logforth::layout::TextLayout;
use logforth::record::LevelFilter;

use crate::engine::llm_engine::{LLMEngine, Qwen3ModelRunner};
use crate::engine::scheduler::{Scheduler, SchedulerConfig};
use crate::models::qwen3::Qwen3Config;
use crate::sampling_params::SamplingParams;
use candle_core::Device;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl From<LogLevel> for logforth::record::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => Self::Trace,
            LogLevel::Debug => Self::Debug,
            LogLevel::Info => Self::Info,
            LogLevel::Warn => Self::Warn,
            LogLevel::Error => Self::Error,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum DeviceType {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "metal")]
    Metal,
}

impl Default for DeviceType {
    fn default() -> Self {
        #[cfg(feature = "cuda")]
        {
            return Self::Cuda;
        }
        #[cfg(feature = "metal")]
        {
            return Self::Metal;
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            Self::Cpu
        }
    }
}

impl DeviceType {
    fn create_device(self) -> anyhow::Result<Device> {
        match self {
            DeviceType::Cpu => Ok(Device::Cpu),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => Device::new_cuda(0).context("failed to create CUDA device"),
            #[cfg(feature = "metal")]
            DeviceType::Metal => Device::new_metal(0).context("failed to create Metal device"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "nano-vllm-candle")]
#[command(about = "Candle-based LLM inference engine", long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        help = "Path to tokenizer.json (model directory is inferred from parent)"
    )]
    tokenizer: String,

    #[arg(short, long, help = "Prompt text to generate from")]
    prompt: String,

    #[arg(short = 'T', long, default_value_t = 0.6, help = "Sampling temperature")]
    temperature: f64,

    #[arg(short, long, default_value_t = 256, help = "Maximum tokens to generate")]
    max_tokens: usize,

    #[arg(long, default_value_t = false, help = "Ignore EOS token (for benchmarking)")]
    ignore_eos: bool,

    #[arg(short, long, value_enum, default_value_t = LogLevel::Info, help = "Log level")]
    log_level: LogLevel,

    #[arg(short, long, value_enum, default_value_t = DeviceType::Cuda, help = "Compute device")]
    device: DeviceType,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let layout = TextLayout::default();
    let log_level: logforth::record::Level = args.log_level.into();

    logforth::starter_log::builder()
        .dispatch(|d| {
            d.filter(LevelFilter::MoreSevereEqual(log_level))
                .append(append::Stdout::default().with_layout(layout))
        })
        .apply();

    info!("nano-vllm-candle inference service starting");

    let tokenizer_path = &args.tokenizer;
    let prompt = &args.prompt;

    let model_dir = Path::new(tokenizer_path)
        .parent()
        .expect("tokenizer.json must be inside model directory");
    let model_dir_str = model_dir.to_str().expect("model directory path must be valid UTF-8");

    info!("Loading tokenizer from: {}", tokenizer_path);
    let tokenizer_result = Tokenizer::from_file(tokenizer_path);

    let (input_ids, hf_tokenizer): (Vec<usize>, Option<Tokenizer>) = match tokenizer_result {
        Ok(tk) => {
            let chat_prompt = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt);
            let encoding = tk.encode(chat_prompt.as_str(), true).unwrap();
            let ids: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();
            info!(
                "Tokenized prompt: {} tokens from {} chars",
                ids.len(),
                chat_prompt.len()
            );
            debug!("Token IDs: {:?}", &ids[..ids.len().min(20)]);
            (ids, Some(tk))
        },
        Err(err) => {
            info!(
                "Failed to load tokenizer from {}: {}. Falling back to byte-level tokenizer.",
                tokenizer_path, err
            );
            let ids: Vec<usize> = prompt.bytes().map(|b| b as usize).collect();
            (ids, None)
        },
    };

    let device = args.device.create_device()?;
    info!("Using device: {:?}", device);
    let tp_cfg = crate::tp::get_tp();
    info!(
        "Tensor Parallel config: size={} rank={} dim={} (rank=1 by default)",
        tp_cfg.size, tp_cfg.rank, tp_cfg.dim
    );

    let qwen_cfg =
        Qwen3Config::from_hf_dir(model_dir_str).context("failed to load Qwen3 HF config from config.json")?;
    info!(
        "Qwen3 config: hidden_size={} layers={} heads={} kv_heads={} head_dim={} vocab_size={} eos={} bos={}",
        qwen_cfg.hidden_size,
        qwen_cfg.num_hidden_layers,
        qwen_cfg.num_attention_heads,
        qwen_cfg.num_key_value_heads,
        qwen_cfg.head_dim,
        qwen_cfg.vocab_size,
        qwen_cfg.eos_token_id,
        qwen_cfg.bos_token_id,
    );

    let config = SchedulerConfig {
        max_num_seqs: 1,
        max_num_batched_tokens: 4096,
        eos: qwen_cfg.eos_token_id,
        num_kvcache_blocks: 0,
        kvcache_block_size: 256,
    };

    let scheduler = Scheduler::new(config);

    let mut runner = Qwen3ModelRunner::from_hf_dir(device, model_dir_str)
        .context("failed to construct Qwen3ModelRunner from HF directory")?;
    runner.eos_id = qwen_cfg.eos_token_id;

    let mut engine = LLMEngine::new(scheduler, runner);

    let sampling_params = SamplingParams::new(args.temperature, args.max_tokens).with_ignore_eos(args.ignore_eos);
    info!(
        "SamplingParams: temperature={}, max_tokens={}, ignore_eos={}",
        sampling_params.temperature, sampling_params.max_tokens, sampling_params.ignore_eos
    );

    let mut prompt_ids = input_ids.clone();
    prompt_ids.insert(0, qwen_cfg.bos_token_id);
    let prompt_len = prompt_ids.len();

    let gen_start = Instant::now();
    let outputs = engine.generate(vec![prompt_ids], &sampling_params);
    let gen_elapsed = gen_start.elapsed();

    for output in &outputs {
        info!(
            "Generation complete: seq_id={}, {} tokens",
            output.seq_id,
            output.token_ids.len()
        );

        match &hf_tokenizer {
            Some(tk) => {
                let ids_u32: Vec<u32> = output.token_ids.iter().map(|&id| id as u32).collect();
                info!("Token IDs: {:?}", &ids_u32[..ids_u32.len().min(20)]);
                let text = tk.decode(&ids_u32, true).unwrap();
                println!("\n=== Generated Text ===\n{}", text);
            },
            None => {
                let bytes: Vec<u8> = output.token_ids.iter().map(|&id| id as u8).collect();
                match String::from_utf8(bytes) {
                    Ok(text) => println!("\n=== Generated Text (byte tokenizer) ===\n{}", text),
                    Err(_) => println!("\n=== Generated Token IDs ===\n{:?}", output.token_ids),
                }
            },
        }
    }

    let total_generated: usize = outputs.iter().map(|o| o.token_ids.len()).sum();
    let decode_tokens = total_generated.saturating_sub(prompt_len);
    let elapsed_secs = gen_elapsed.as_secs_f64();
    let throughput = if elapsed_secs > 0.0 {
        decode_tokens as f64 / elapsed_secs
    } else {
        0.0
    };

    println!("\n=== Performance Stats ===");
    println!(
        "Total: {}tok, Prompt: {}tok, Generated: {}tok",
        total_generated, prompt_len, decode_tokens
    );
    println!("Time: {:.2}s, Throughput: {:.2}tok/s", elapsed_secs, throughput);

    Ok(())
}
