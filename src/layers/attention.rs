use candle_core::{Result, Tensor};
use candle_nn::ops::sdpa;

pub struct Attention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub softcapping: f32,
}

impl Attention {
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let o = sdpa(q, k, v, self.scale, self.softcapping)?;
        Ok(o)
    }
}
