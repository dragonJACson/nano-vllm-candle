use candle_core::{D, Result, Tensor};
use candle_nn::Module;

pub struct SiluAndMul;

impl Module for SiluAndMul {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let chunks = x.chunk(2, D::Minus1)?;

        let x = &chunks[0];
        let y = &chunks[1];

        x.silu()?.mul(y)
    }
}
