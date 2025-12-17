use std::fs;
use std::path::Path;

use anyhow::Context;
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, ops::softmax};
use serde::Deserialize;

use crate::layers::activation::SiluAndMul;
use crate::layers::layernorm::RMSNorm;
use crate::layers::linear::{LinearBase, ParallelLinear, QKVParallelLinear, RowParallelLinear};
use crate::layers::rotary_embedding::RotaryEmbedding;

/// Minimal Qwen3 config used for a single-process 0.6B-style model.
///
/// NOTE: All fields and default values here are placeholders intended
/// to make the model structurally similar to Qwen3. They are **not**
/// guaranteed to match the official 0.6B configuration exactly.
#[derive(Clone, Debug)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub rope_theta: f64,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
}

#[derive(Debug, Deserialize)]
struct HfQwen3Config {
    vocab_size: usize,
    hidden_size: usize,
    head_dim: Option<usize>,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    hidden_act: String,
    #[serde(default)]
    rope_theta: Option<f64>,
    bos_token_id: usize,
    eos_token_id: usize,
}

impl Qwen3Config {
    /// Reasonable defaults for a small Qwen3-like model.
    pub fn qwen3_0_6b() -> Self {
        // 如果你要对齐真实的 Qwen3-0.6B，请根据官方配置修改这里的参数。
        Self {
            vocab_size: 151936,
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            // 为了简化实现，当前假设 kv heads == attention heads。
            num_key_value_heads: 16,
            head_dim: 2048 / 16,
            intermediate_size: 5632,
            max_position_embeddings: 4096 * 32,
            rms_norm_eps: 1e-6,
            hidden_act: "silu".to_string(),
            rope_theta: 1e6,
            bos_token_id: 151643,
            eos_token_id: 151645,
        }
    }

    /// Load config directly from HuggingFace `config.json` in the given model directory.
    pub fn from_hf_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let data = fs::read_to_string(&path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read HF config {}: {e}", path.display())))?;
        let hf_cfg: HfQwen3Config = serde_json::from_str(&data)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse HF config {}: {e}", path.display())))?;

        Ok(Self {
            vocab_size: hf_cfg.vocab_size,
            hidden_size: hf_cfg.hidden_size,
            num_hidden_layers: hf_cfg.num_hidden_layers,
            num_attention_heads: hf_cfg.num_attention_heads,
            num_key_value_heads: hf_cfg.num_key_value_heads,
            head_dim: hf_cfg
                .head_dim
                .unwrap_or(hf_cfg.hidden_size / hf_cfg.num_attention_heads),
            intermediate_size: hf_cfg.intermediate_size,
            max_position_embeddings: hf_cfg.max_position_embeddings,
            rms_norm_eps: hf_cfg.rms_norm_eps,
            hidden_act: hf_cfg.hidden_act,
            rope_theta: hf_cfg.rope_theta.unwrap_or(1e6),
            bos_token_id: hf_cfg.bos_token_id,
            eos_token_id: hf_cfg.eos_token_id,
        })
    }
}

struct Qwen3Attention {
    qkv_proj: QKVParallelLinear,
    o_proj: RowParallelLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    rope: RotaryEmbedding,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
}

impl Qwen3Attention {
    fn new(cfg: &Qwen3Config, device: &Device) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let qkv_proj = QKVParallelLinear::new(cfg.hidden_size, head_dim, num_heads, num_kv_heads, None);
        let w_o = Tensor::randn(0f32, 0.02, (cfg.hidden_size, num_heads * head_dim), device)?;
        let mut o_proj = RowParallelLinear::new(num_heads * head_dim, cfg.hidden_size, None);
        o_proj.load_weights(&w_o, None)?;
        let q_norm = RMSNorm::new(head_dim, Some(cfg.rms_norm_eps), device)?;
        let k_norm = RMSNorm::new(head_dim, Some(cfg.rms_norm_eps), device)?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope: RotaryEmbedding::new(head_dim, cfg.max_position_embeddings, cfg.rope_theta as f32, device),
            q_norm,
            k_norm,
        })
    }

    fn from_hf(cfg: &Qwen3Config, vb: &VarBuilder, layer_idx: usize, device: &Device) -> anyhow::Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let kv_heads = cfg.num_key_value_heads;

        let prefix = format!("model.layers.{}.self_attn", layer_idx);
        let w_q_raw: Tensor = vb
            .get(
                (num_heads * head_dim, cfg.hidden_size),
                format!("{}.q_proj.weight", prefix).as_str(),
            )
            .context(format!("load q_proj.weight for layer {}", layer_idx))?;
        let w_k_raw: Tensor = vb
            .get(
                (kv_heads * head_dim, cfg.hidden_size),
                format!("{}.k_proj.weight", prefix).as_str(),
            )
            .context(format!("load k_proj.weight for layer {}", layer_idx))?;
        let w_v_raw: Tensor = vb
            .get(
                (kv_heads * head_dim, cfg.hidden_size),
                format!("{}.v_proj.weight", prefix).as_str(),
            )
            .context(format!("load v_proj.weight for layer {}", layer_idx))?;
        let w_o_raw: Tensor = vb
            .get(
                (cfg.hidden_size, num_heads * head_dim),
                format!("{}.o_proj.weight", prefix).as_str(),
            )
            .context(format!("load o_proj.weight for layer {}", layer_idx))?;
        let w_qkv = Tensor::cat(&[w_q_raw, w_k_raw.clone(), w_v_raw.clone()], 0)?; // [(nh+2*kv)*hd, H]
        let mut qkv_proj = QKVParallelLinear::new(cfg.hidden_size, head_dim, num_heads, kv_heads, None);
        qkv_proj.linear.load_weights(&w_qkv.to_device(device)?, None)?;
        let mut o_proj = RowParallelLinear::new(num_heads * head_dim, cfg.hidden_size, None);
        o_proj.load_weights(&w_o_raw.to_device(device)?, None)?;

        let q_norm_w: Tensor = vb
            .get((head_dim,), format!("{}.q_norm.weight", prefix).as_str())
            .context(format!("load q_norm.weight for layer {}", layer_idx))?;
        let q_norm_w = q_norm_w.reshape((1, head_dim))?.to_device(device)?;
        let q_norm = RMSNorm::from_weight(q_norm_w, cfg.rms_norm_eps);

        let k_norm_w: Tensor = vb
            .get((head_dim,), format!("{}.k_norm.weight", prefix).as_str())
            .context(format!("load k_norm.weight for layer {}", layer_idx))?;
        let k_norm_w = k_norm_w.reshape((1, head_dim))?.to_device(device)?;
        let k_norm = RMSNorm::from_weight(k_norm_w, cfg.rms_norm_eps);

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads: kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope: RotaryEmbedding::new(head_dim, cfg.max_position_embeddings, cfg.rope_theta as f32, device),
            q_norm,
            k_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (b, t, _h) = x.dims3()?;
        let x2d = x.reshape((b * t, _h))?;
        let qkv = self.qkv_proj.forward(&x2d)?; // [B*T, (nh+2*kv)*hd]
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        let q2d = qkv.narrow(1, 0, q_size)?;
        let k2d = qkv.narrow(1, q_size, kv_size)?;
        let v2d = qkv.narrow(1, q_size + kv_size, kv_size)?;

        let q_bt = q2d.reshape((b, t, self.num_heads * self.head_dim))?;
        let k_bt = k2d.reshape((b, t, self.num_kv_heads * self.head_dim))?;
        let v_bt = v2d.reshape((b, t, self.num_kv_heads * self.head_dim))?;

        let q = q_bt.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?; // [B, nh, T, hd]
        let k = k_bt
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, kv, T, hd]
        let mut v = v_bt
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // q_norm / k_norm: normalize along head_dim BEFORE GQA repeat, matching nano-vllm.
        let q_shape = q.dims();
        let k_shape = k.dims();
        let q_flat = q.reshape((b * self.num_heads * t, self.head_dim))?;
        let k_flat = k.reshape((b * self.num_kv_heads * t, self.head_dim))?;
        let (q_normed, _) = self.q_norm.forward(&q_flat, None)?;
        let (k_normed, _) = self.k_norm.forward(&k_flat, None)?;
        let q = q_normed.reshape(q_shape)?;
        let k = k_normed.reshape(k_shape)?;

        let (q, k) = self.rope.apply(&q, &k)?;

        // GQA: repeat K/V heads AFTER norm and RoPE
        // Note: candle's repeat() is like numpy.tile, but we need numpy.repeat behavior
        // numpy.repeat([h0, h1], 2, axis=1) -> [h0, h0, h1, h1] (interleaved)
        // candle's repeat((1,2,1,1)) gives [h0, h1, h0, h1] (tiled) - WRONG for GQA
        let mut k = k;
        if self.num_kv_heads != self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            // Implement numpy.repeat by: unsqueeze -> expand -> reshape
            // k: [B, kv_heads, T, hd] -> [B, kv_heads, 1, T, hd] -> [B, kv_heads, repeat, T, hd] -> [B, kv_heads*repeat, T, hd]
            let (b, kv, t, hd) = k.dims4()?;
            k = k
                .unsqueeze(2)? // [B, kv_heads, 1, T, hd]
                .expand((b, kv, repeat, t, hd))? // [B, kv_heads, repeat, T, hd]
                .reshape((b, kv * repeat, t, hd))?; // [B, num_heads, T, hd]

            let (b, kv, t, hd) = v.dims4()?;
            v = v
                .unsqueeze(2)?
                .expand((b, kv, repeat, t, hd))?
                .reshape((b, kv * repeat, t, hd))?;
        }

        let kt = k.transpose(2, 3)?;
        let attn_scores = (q.matmul(&kt)? * self.scale as f64)?;
        let mut mask: Vec<f32> = Vec::with_capacity(b * self.num_heads * t * t);
        for _ in 0..b {
            for _ in 0..self.num_heads {
                for i in 0..t {
                    for j in 0..t {
                        mask.push(if j > i { -1e9 } else { 0.0 });
                    }
                }
            }
        }
        let mask = Tensor::from_vec(mask, (b, self.num_heads, t, t), &q.device())?;
        let attn_scores = (attn_scores + mask)?;
        let attn_scores = attn_scores.to_dtype(DType::F32)?;
        let attn_probs = softmax(&attn_scores, D::Minus1)?;

        let ctx = attn_probs.matmul(&v)?; // [B, nh, T, hd]
        let ctx_bt = ctx.transpose(1, 2)?; // [B, T, nh, hd]
        let ctx2d = ctx_bt.reshape((b * t, self.num_heads * self.head_dim))?; // [B*T, nh*hd]
        let out2d = self.o_proj.forward(&ctx2d)?; // [B*T, H]
        Ok(out2d.reshape((b, t, _h))?)
    }
}

struct Qwen3MLP {
    gate_up: Linear,
    down: Linear,
    act: SiluAndMul,
}

impl Qwen3MLP {
    fn new(cfg: &Qwen3Config, device: &Device) -> anyhow::Result<Self> {
        let w_gate_up = Tensor::randn(0f32, 0.02, (cfg.hidden_size, 2 * cfg.intermediate_size), device)?;
        let w_down = Tensor::randn(0f32, 0.02, (cfg.intermediate_size, cfg.hidden_size), device)?;
        Ok(Self {
            gate_up: Linear::new(w_gate_up, None),
            down: Linear::new(w_down, None),
            act: SiluAndMul,
        })
    }

    fn from_hf(cfg: &Qwen3Config, vb: &VarBuilder, layer_idx: usize, device: &Device) -> anyhow::Result<Self> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);
        let w_gate: Tensor = vb.get(
            (cfg.intermediate_size, cfg.hidden_size),
            format!("{}.gate_proj.weight", prefix).as_str(),
        )?;
        let w_up: Tensor = vb.get(
            (cfg.intermediate_size, cfg.hidden_size),
            format!("{}.up_proj.weight", prefix).as_str(),
        )?;
        let w_gate_up = Tensor::cat(&[w_gate, w_up], 0)?;
        let w_down: Tensor = vb.get(
            (cfg.hidden_size, cfg.intermediate_size),
            format!("{}.down_proj.weight", prefix).as_str(),
        )?;

        Ok(Self {
            gate_up: Linear::new(w_gate_up.to_device(device)?, None),
            down: Linear::new(w_down.to_device(device)?, None),
            act: SiluAndMul,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up.forward(x)?;
        let x = self.act.forward(&gate_up)?;
        self.down.forward(&x)
    }
}

struct Qwen3DecoderLayer {
    attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_ln: RMSNorm,
    post_attn_ln: RMSNorm,
}

impl Qwen3DecoderLayer {
    fn new(cfg: &Qwen3Config, device: &Device) -> anyhow::Result<Self> {
        Ok(Self {
            attn: Qwen3Attention::new(cfg, device)?,
            mlp: Qwen3MLP::new(cfg, device)?,
            input_ln: RMSNorm::new(cfg.hidden_size, Some(cfg.rms_norm_eps), device)?,
            post_attn_ln: RMSNorm::new(cfg.hidden_size, Some(cfg.rms_norm_eps), device)?,
        })
    }

    fn from_hf(cfg: &Qwen3Config, vb: &VarBuilder, layer_idx: usize, device: &Device) -> anyhow::Result<Self> {
        let attn = Qwen3Attention::from_hf(cfg, vb, layer_idx, device)?;
        let mlp = Qwen3MLP::from_hf(cfg, vb, layer_idx, device)?;

        let input_ln_w: Tensor = vb.get(
            (cfg.hidden_size,),
            format!("model.layers.{}.input_layernorm.weight", layer_idx).as_str(),
        )?;
        let input_ln_w = input_ln_w.reshape((1, cfg.hidden_size))?;
        let input_ln = RMSNorm::from_weight(input_ln_w.to_device(device)?, cfg.rms_norm_eps);

        let post_ln_w: Tensor = vb.get(
            (cfg.hidden_size,),
            format!("model.layers.{}.post_attention_layernorm.weight", layer_idx).as_str(),
        )?;
        let post_ln_w = post_ln_w.reshape((1, cfg.hidden_size))?;
        let post_attn_ln = RMSNorm::from_weight(post_ln_w.to_device(device)?, cfg.rms_norm_eps);

        Ok(Self {
            attn,
            mlp,
            input_ln,
            post_attn_ln,
        })
    }

    /// Forward one decoder layer, returning (hidden_states, residual) like nano-vllm.
    fn forward(&self, x: &Tensor, residual: Option<&Tensor>) -> anyhow::Result<(Tensor, Tensor)> {
        // Input RMSNorm with optional residual.
        let (x_norm, residual) = match residual {
            Some(r) => {
                let (normed, new_residual) = self.input_ln.forward(x, Some(r))?;
                let new_residual = new_residual.expect("RMSNorm with residual must return residual");
                (normed, new_residual)
            },
            None => {
                let (normed, _) = self.input_ln.forward(x, None)?;
                // nano-vllm: residual is original hidden_states when None.
                (normed, x.clone())
            },
        };

        // Self-attention uses normalized hidden states.
        let attn_out = self.attn.forward(&x_norm)?;

        // Post-attention RMSNorm adds residual internally and updates residual.
        let (h_norm, residual) = self.post_attn_ln.forward(&attn_out, Some(&residual))?;
        let residual = residual.expect("RMSNorm with residual must return residual");

        // MLP on normalized hidden states; residual is carried to next layer.
        let h = self.mlp.forward(&h_norm)?;
        Ok((h, residual))
    }
}

/// Qwen3 decoder-only transformer (简化版，仅支持单进程、无 KV cache)。
pub struct Qwen3Model {
    pub cfg: Qwen3Config,
    pub embed: Tensor, // [vocab, hidden]
    layers: Vec<Qwen3DecoderLayer>,
    norm: RMSNorm,
    device: Device,
}

impl Qwen3Model {
    pub fn new(cfg: Qwen3Config, device: &Device) -> anyhow::Result<Self> {
        let embed = Tensor::randn(0f32, 0.02, (cfg.vocab_size, cfg.hidden_size), device)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(&cfg, device)?);
        }
        let norm = RMSNorm::new(cfg.hidden_size, Some(cfg.rms_norm_eps), device)?;

        Ok(Self {
            cfg,
            embed,
            layers,
            norm,
            device: device.clone(),
        })
    }

    pub fn from_hf(cfg: Qwen3Config, vb: &VarBuilder, device: &Device) -> anyhow::Result<Self> {
        let embed: Tensor = vb
            .get((cfg.vocab_size, cfg.hidden_size), "model.embed_tokens.weight")
            .context("load model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(
                Qwen3DecoderLayer::from_hf(&cfg, vb, layer_idx, device)
                    .context(format!("build decoder layer {}", layer_idx))?,
            );
        }

        let norm_w: Tensor = vb
            .get((cfg.hidden_size,), "model.norm.weight")
            .context("load model.norm.weight")?;
        let norm_w = norm_w.reshape((1, cfg.hidden_size))?;
        let norm = RMSNorm::from_weight(norm_w.to_device(device)?, cfg.rms_norm_eps);

        Ok(Self {
            cfg,
            embed: embed.to_device(device)?,
            layers,
            norm,
            device: device.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> anyhow::Result<Tensor> {
        // input_ids: [B, T]
        let (b, t) = input_ids.dims2()?;

        log::debug!("embedding start: b={} t={} h={}", b, t, self.cfg.hidden_size);

        // 将 ids 展平为 1D，再做 embedding，然后 reshape 回 [B, T, H]。
        let ids = input_ids.to_dtype(DType::U32)?;
        let flat_ids = ids.flatten_all()?;
        let emb_flat = self.embed.embedding(&flat_ids)?; // [B*T, H]
        let mut h = emb_flat.reshape((b, t, self.cfg.hidden_size))?;

        log::debug!("decoder start: layers={}", self.cfg.num_hidden_layers);

        // Debug: check embedding stats
        if log::log_enabled!(log::Level::Trace) {
            let h_flat = h.flatten_all()?;
            let h_mean = h_flat.mean_all()?.to_vec0::<f32>()?;
            let h_var = h_flat.var(0)?.to_vec0::<f32>()?;
            log::trace!("embedding output: mean={:.6} var={:.6}", h_mean, h_var);
        }

        let mut residual: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let (new_h, new_residual) = layer.forward(&h, residual.as_ref())?;
            h = new_h;
            residual = Some(new_residual);

            // Debug: check layer output stats
            if log::log_enabled!(log::Level::Trace) && i < 2 {
                let h_flat = h.flatten_all()?;
                let h_mean = h_flat.mean_all()?.to_vec0::<f32>()?;
                let h_var = h_flat.var(0)?.to_vec0::<f32>()?;
                log::trace!("layer {} output: mean={:.6} var={:.6}", i, h_mean, h_var);
            }
        }

        log::debug!("final norm and output");

        let (h, _residual) = self.norm.forward(&h, residual.as_ref())?;
        Ok(h)
    }
}

/// Qwen3 For Causal LM: decoder + lm_head。
pub struct Qwen3ForCausalLM {
    pub model: Qwen3Model,
    lm_head: Tensor, // [hidden, vocab]
}

impl Qwen3ForCausalLM {
    pub fn new(cfg: Qwen3Config, device: &Device) -> anyhow::Result<Self> {
        let model = Qwen3Model::new(cfg.clone(), device)?;
        let lm_head = Tensor::randn(0f32, 0.02, (cfg.hidden_size, cfg.vocab_size), device)?;
        Ok(Self { model, lm_head })
    }

    pub fn from_hf_dir<P: AsRef<Path>>(model_dir: P, device: &Device) -> anyhow::Result<Self> {
        let cfg = Qwen3Config::from_hf_dir(&model_dir).context("load HF config.json")?;
        let model_path = model_dir.as_ref().join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.clone()], DType::F32, device)
                .context(format!("mmap {}", model_path.display()))?
        };

        let model = Qwen3Model::from_hf(cfg.clone(), &vb, device)?;

        let lm_head_w: Tensor = vb
            .get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
            .context("load lm_head.weight")?;
        let lm_head = lm_head_w.transpose(0, 1)?.to_device(device)?;
        log::debug!(
            "lm_head built: hidden_size={} vocab_size={}",
            cfg.hidden_size,
            cfg.vocab_size
        );

        Ok(Self { model, lm_head })
    }

    pub fn forward(&self, input_ids: &Tensor) -> anyhow::Result<Tensor> {
        self.model.forward(input_ids)
    }

    pub fn compute_logits(&self, hidden: &Tensor) -> anyhow::Result<Tensor> {
        // hidden: [B, T, H]
        let (b, t, h) = hidden.dims3()?;
        let hidden2d = hidden.reshape((b * t, h))?;
        log::debug!("compute_logits: b={} t={} h={}", b, t, h);

        let logits2d = hidden2d.matmul(&self.lm_head)?; // [B*T, V]
        Ok(logits2d.reshape((b, t, self.model.cfg.vocab_size))?)
    }
}
