use candle_core::{D, Result, Tensor};
use candle_nn::Module;

#[derive(Debug, Clone, Copy, Default)]
pub struct SiluAndMul;

impl SiluAndMul {
    pub const fn new() -> Self {
        Self
    }
}

impl Module for SiluAndMul {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let chunks = x.chunk(2, D::Minus1)?;
        chunks[0].silu()?.mul(&chunks[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_silu_and_mul() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[0f32, 1f32, -1f32, 2f32]], &device)?;
        let layer = SiluAndMul::new();
        let out = layer.forward(&x)?;

        let v = out.to_vec2::<f32>()?;
        assert!((v[0][0] - 0.0).abs() < 1e-6);
        assert!((v[0][1] - 1.4621172).abs() < 1e-5);
        Ok(())
    }
}
