use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::layers::activation::SiluAndMul;
use crate::layers::layernorm::RMSNorm;

/// Minimal Qwen3 config used for a single-process 0.6B-style model.
///
/// NOTE: All fields and default values here are placeholders intended
/// to make the model structurally similar to Qwen3. They are **not**
/// guaranteed to match the official 0.6B configuration exactly.
#[derive(Clone, Debug)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub rope_theta: f64,
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
            intermediate_size: 5632,
            max_position_embeddings: 4096 * 32,
            rms_norm_eps: 1e-6,
            hidden_act: "silu".to_string(),
            rope_theta: 1e6,
        }
    }
}

struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Qwen3Attention {
    fn new(cfg: &Qwen3Config, device: &Device) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;

        let w_q = Tensor::randn(0f32, 0.02, (cfg.hidden_size, num_heads * head_dim), device)?;
        let w_k = Tensor::randn(0f32, 0.02, (cfg.hidden_size, num_heads * head_dim), device)?;
        let w_v = Tensor::randn(0f32, 0.02, (cfg.hidden_size, num_heads * head_dim), device)?;
        let w_o = Tensor::randn(0f32, 0.02, (num_heads * head_dim, cfg.hidden_size), device)?;

        Ok(Self {
            q_proj: Linear::new(w_q, None),
            k_proj: Linear::new(w_k, None),
            v_proj: Linear::new(w_v, None),
            o_proj: Linear::new(w_o, None),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T, H]
        let (b, t, _h) = x.dims3()?;

        let q = self.q_proj.forward(x)?; // [B, T, nh*hd]
        let k = self.k_proj.forward(x)?; // [B, T, nh*hd]
        let v = self.v_proj.forward(x)?; // [B, T, nh*hd]

        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, nh, T, hd]
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, nh, T, hd]
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, nh, T, hd]

        let attn_scores = (q.matmul(&k.transpose(2, 3)?)? * self.scale as f64)?; // [B, nh, T, T]
        let attn_scores = attn_scores.to_dtype(DType::F32)?;
        let attn_probs = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;
        let ctx = attn_probs.matmul(&v)?; // [B, nh, T, hd]
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&ctx)
    }
}

struct Qwen3MLP {
    gate_up: Linear,
    down: Linear,
    act: SiluAndMul,
}

impl Qwen3MLP {
    fn new(cfg: &Qwen3Config, device: &Device) -> Result<Self> {
        let w_gate_up = Tensor::randn(
            0f32,
            0.02,
            (cfg.hidden_size, 2 * cfg.intermediate_size),
            device,
        )?;
        let w_down = Tensor::randn(0f32, 0.02, (cfg.intermediate_size, cfg.hidden_size), device)?;
        Ok(Self {
            gate_up: Linear::new(w_gate_up, None),
            down: Linear::new(w_down, None),
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
    fn new(cfg: &Qwen3Config, device: &Device) -> Result<Self> {
        Ok(Self {
            attn: Qwen3Attention::new(cfg, device)?,
            mlp: Qwen3MLP::new(cfg, device)?,
            input_ln: RMSNorm::new(cfg.hidden_size, Some(cfg.rms_norm_eps), device)?,
            post_attn_ln: RMSNorm::new(cfg.hidden_size, Some(cfg.rms_norm_eps), device)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 简化版：Pre-LN + 残差结构
        let (x_norm, _residual) = self.input_ln.forward(x, None)?;
        let h = self.attn.forward(&x_norm)?;
        let (h, _residual2) = self.post_attn_ln.forward(&h, Some(x))?;
        let h = self.mlp.forward(&h)?;
        Ok(h)
    }
}

/// Qwen3 decoder-only transformer (简化版，仅支持单进程、无 KV cache)。
pub struct Qwen3Model {
    pub cfg: Qwen3Config,
    embed: Tensor, // [vocab, hidden]
    layers: Vec<Qwen3DecoderLayer>,
    norm: RMSNorm,
    device: Device,
}

impl Qwen3Model {
    pub fn new(cfg: Qwen3Config, device: &Device) -> Result<Self> {
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

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // input_ids: [B, T]
        let (b, t) = input_ids.dims2()?;

        // 将 ids 展平为 1D，再做 embedding，然后 reshape 回 [B, T, H]。
        let ids = input_ids.to_dtype(DType::U32)?;
        let flat_ids = ids.flatten_all()?;
        let emb_flat = self.embed.embedding(&flat_ids)?; // [B*T, H]
        let mut h = emb_flat.reshape((b, t, self.cfg.hidden_size))?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
        }

        let (h, _residual) = self.norm.forward(&h, None)?;
        Ok(h)
    }
}

/// Qwen3 For Causal LM: decoder + lm_head。
pub struct Qwen3ForCausalLM {
    pub model: Qwen3Model,
    lm_head: Tensor, // [hidden, vocab]
}

impl Qwen3ForCausalLM {
    pub fn new(cfg: Qwen3Config, device: &Device) -> Result<Self> {
        let model = Qwen3Model::new(cfg.clone(), device)?;
        let lm_head = Tensor::randn(0f32, 0.02, (cfg.hidden_size, cfg.vocab_size), device)?;
        Ok(Self { model, lm_head })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids)
    }

    pub fn compute_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        // hidden: [B, T, H]
        let (b, t, h) = hidden.dims3()?;
        let hidden2d = hidden.reshape((b * t, h))?;
        let logits2d = hidden2d.matmul(&self.lm_head)?; // [B*T, V]
        logits2d.reshape((b, t, self.model.cfg.vocab_size))
    }
}
