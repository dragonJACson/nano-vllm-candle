# Nano-vLLM-Candle

A Rust port of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) using [Candle](https://github.com/huggingface/candle).

## Key Features

* 🦀 **Pure Rust** - Native Rust implementation with no Python dependencies
* 🚀 **Fast inference** - CUDA and Metal acceleration via Candle
* 📖 **Readable codebase** - Clean, idiomatic Rust with builder patterns and traits
* ⚡ **Optimization Suite** - Tensor Parallelism support, efficient memory management

## Installation

### Prerequisites

- Rust 1.75+ (edition 2024)
- CUDA Toolkit 12.x (for CUDA support)
- Or Xcode Command Line Tools (for Metal support on macOS)

### Build

```bash
# CPU only
cargo build --release

# With CUDA support
cargo build --release --features cuda

# With Metal support (macOS)
cargo build --release --features metal
```

## Model Download

Download the model weights using Hugging Face CLI:

```bash
hf download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
```

## Quick Start

```bash
cargo run --release --features cuda -- \
  -t ~/huggingface/Qwen3-0.6B/tokenizer.json \
  -p "Hello, Nano-vLLM-Candle." \
  -m 256 \
  -T 0.6
```

### CLI Options

```
Options:
  -t, --tokenizer <TOKENIZER>      Path to tokenizer.json
  -p, --prompt <PROMPT>            Prompt text to generate from
  -T, --temperature <TEMPERATURE>  Sampling temperature [default: 0.6]
  -m, --max-tokens <MAX_TOKENS>    Maximum tokens to generate [default: 256]
      --ignore-eos                 Ignore EOS token (for benchmarking)
  -l, --log-level <LOG_LEVEL>      Log level [default: info]
  -d, --device <DEVICE>            Compute device [default: cuda]
  -h, --help                       Print help
```

## Usage as Library

```rust
use nano_vllm_candle::{
    engine::llm_engine::{LLMEngine, Qwen3ModelRunner},
    engine::scheduler::{Scheduler, SchedulerConfig},
    sampling_params::SamplingParams,
};
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;

    let config = SchedulerConfig::default()
        .with_max_num_seqs(1)
        .with_max_num_batched_tokens(4096);

    let scheduler = Scheduler::from(config);
    let runner = Qwen3ModelRunner::from_hf_dir(device, "path/to/model")?;
    let mut engine = LLMEngine::new(scheduler, runner);

    let params = SamplingParams::default()
        .with_temperature(0.6)
        .with_max_tokens(256);

    let outputs = engine.generate(vec![prompt_ids], &params);
    Ok(())
}
```

## Benchmark

```bash
cargo run --release --features cuda -- \
  -t ~/huggingface/Qwen3-0.6B/tokenizer.json \
  -p "Benchmark test" \
  -m 133966 \
  --ignore-eos
```

## Acknowledgments

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) - Original Python implementation
- [Candle](https://github.com/huggingface/candle) - Minimalist ML framework for Rust
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
