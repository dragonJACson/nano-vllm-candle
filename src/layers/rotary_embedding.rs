use candle_core::{D, Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RotaryEmbeddingConfig {
    pub head_size: usize,
    pub max_position: usize,
    pub base: f32,
}

impl RotaryEmbeddingConfig {
    pub const fn new(head_size: usize) -> Self {
        Self {
            head_size,
            max_position: 4096,
            base: 10000.0,
        }
    }

    #[must_use]
    pub const fn with_max_position(mut self, max_position: usize) -> Self {
        self.max_position = max_position;
        self
    }

    #[must_use]
    pub const fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }
}

pub struct RotaryEmbedding {
    config: RotaryEmbeddingConfig,
    device: Device,
}

impl RotaryEmbedding {
    pub fn new(head_size: usize, max_position: usize, base: f32, device: &Device) -> Self {
        Self {
            config: RotaryEmbeddingConfig {
                head_size,
                max_position,
                base,
            },
            device: device.clone(),
        }
    }

    pub fn from_config(config: RotaryEmbeddingConfig, device: &Device) -> Self {
        Self {
            config,
            device: device.clone(),
        }
    }

    fn build_cos_sin(&self, t: usize) -> Result<(Tensor, Tensor)> {
        let dim = self.config.head_size;
        let half = dim / 2;

        let inv_freq: Vec<f32> = (0..half)
            .map(|j| {
                let exponent = (2.0 * j as f32) / (dim as f32);
                1.0 / self.config.base.powf(exponent)
            })
            .collect();

        let (cos, sin): (Vec<f32>, Vec<f32>) = (0..t)
            .flat_map(|pos| {
                inv_freq.iter().map(move |&f| {
                    let ang = (pos as f32) * f;
                    (ang.cos(), ang.sin())
                })
            })
            .unzip();

        let cos = Tensor::from_vec(cos, (t, half), &self.device)?;
        let sin = Tensor::from_vec(sin, (t, half), &self.device)?;

        Ok((cos, sin))
    }

    fn apply_rotary(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let chunks = x.chunk(2, D::Minus1)?;
        let x1 = &chunks[0];
        let x2 = &chunks[1];

        let y1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
        let y2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;

        Tensor::cat(&[y1.as_ref(), y2.as_ref()], D::Minus1)
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let dims = q.dims();
        let t = dims[2];
        let hd = dims[3];
        let half = hd / 2;

        let (cos, sin) = self.build_cos_sin(t)?;
        let cos = cos.reshape((1, 1, t, half))?;
        let sin = sin.reshape((1, 1, t, half))?;

        let q_out = self.apply_rotary(q, &cos, &sin)?;
        let k_out = self.apply_rotary(k, &cos, &sin)?;

        Ok((q_out, k_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_norm_preserved() -> Result<()> {
        let device = Device::Cpu;
        let config = RotaryEmbeddingConfig::new(8).with_max_position(128).with_base(10000.0);
        let rope = RotaryEmbedding::from_config(config, &device);

        let (b, nh, t, head_dim) = (1, 2, 4, 8);
        let q = Tensor::randn(0f32, 0.01, (b, nh, t, head_dim), &device)?;
        let k = Tensor::randn(0f32, 0.01, (b, nh, t, head_dim), &device)?;

        let (qr, kr) = rope.apply(&q, &k)?;

        let qn = q.powf(2.0)?.sum_keepdim(D::Minus1)?;
        let qrn = qr.powf(2.0)?.sum_keepdim(D::Minus1)?;
        let kn = k.powf(2.0)?.sum_keepdim(D::Minus1)?;
        let krn = kr.powf(2.0)?.sum_keepdim(D::Minus1)?;

        let diff_q = (qn - qrn)?.abs()?.sum_all()?.to_vec0::<f32>()?;
        let diff_k = (kn - krn)?.abs()?.sum_all()?.to_vec0::<f32>()?;

        assert!(diff_q < 1e-5);
        assert!(diff_k < 1e-5);
        Ok(())
    }
}
