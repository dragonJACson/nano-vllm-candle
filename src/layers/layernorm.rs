use candle_core::{D, DType, Result, Tensor};

#[derive(Debug)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: Option<f64>, device: &candle_core::Device) -> Result<Self> {
        let weight = Tensor::ones((1usize, hidden_size), DType::F32, device)?;
        Ok(Self {
            weight,
            eps: eps.unwrap_or(1e-6),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        residual: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let orig_dtype = x.dtype();

        let (x_float, new_residual) = match residual {
            Some(residual) => {
                let x_float = x.to_dtype(DType::F32)?;
                let residual_float = residual.to_dtype(DType::F32)?;
                let sum = (x_float + residual_float)?;

                (sum.clone(), Some(sum))
            }
            None => (x.to_dtype(DType::F32)?, None),
        };

        let var = x_float.sqr()?.mean_keepdim(D::Minus1)?;

        let x_norm = x_float.broadcast_mul(&(var + self.eps)?.sqrt()?.recip()?)?;

        let output = x_norm.to_dtype(orig_dtype)?.broadcast_mul(&self.weight)?;

        Ok((output, new_residual))
    }
}
