use candle_core::{D, DType, Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RMSNormConfig {
    pub hidden_size: usize,
    pub eps: f64,
}

impl RMSNormConfig {
    pub const fn new(hidden_size: usize) -> Self {
        Self { hidden_size, eps: 1e-6 }
    }

    #[must_use]
    pub const fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

#[derive(Debug)]
pub struct RMSNorm {
    pub(crate) weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: Option<f64>, device: &Device) -> Result<Self> {
        let weight = Tensor::ones((1usize, hidden_size), DType::F32, device)?;
        Ok(Self {
            weight,
            eps: eps.unwrap_or(1e-6),
        })
    }

    pub fn from_config(config: &RMSNormConfig, device: &Device) -> Result<Self> {
        Self::new(config.hidden_size, Some(config.eps), device)
    }

    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor, residual: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let orig_dtype = x.dtype();

        let (x_float, new_residual) = match residual {
            Some(res) => {
                let sum = (x.to_dtype(DType::F32)? + res.to_dtype(DType::F32)?)?;
                (sum.clone(), Some(sum))
            },
            None => (x.to_dtype(DType::F32)?, None),
        };

        let var = x_float.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x_float.broadcast_mul(&(var + self.eps)?.sqrt()?.recip()?)?;
        let output = x_norm.to_dtype(orig_dtype)?.broadcast_mul(&self.weight)?;

        Ok((output, new_residual))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_simple() -> Result<()> {
        let device = Device::Cpu;
        let config = RMSNormConfig::new(4).with_eps(1e-5);
        let rms_norm = RMSNorm::from_config(&config, &device)?;

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
        let x = Tensor::from_slice(&input_data, (2, 4), &device)?;

        let (output, residual) = rms_norm.forward(&x, None)?;

        assert!(residual.is_none());
        assert_eq!(output.dims(), &[2, 4]);

        let output_vec = output.to_vec2::<f32>()?;
        let row0 = &output_vec[0];

        let tolerance = 1e-4;
        assert!((row0[0] - 0.36515).abs() < tolerance);
        assert!((row0[3] - 1.46059).abs() < tolerance);

        Ok(())
    }

    #[test]
    fn test_rms_norm_with_residual() -> Result<()> {
        let device = Device::Cpu;
        let rms_norm = RMSNorm::new(3, None, &device)?;

        let x = Tensor::from_slice(&[0.5f32, 0.5, 0.5], (1, 3), &device)?;
        let res_input = Tensor::from_slice(&[0.5f32, 0.5, 0.5], (1, 3), &device)?;

        let (output, new_residual) = rms_norm.forward(&x, Some(&res_input))?;

        assert!(new_residual.is_some());
        let res_vec = new_residual.unwrap().to_vec2::<f32>()?;
        assert_eq!(res_vec[0], vec![1.0, 1.0, 1.0]);

        let out_vec = output.to_vec2::<f32>()?;
        assert!((out_vec[0][0] - 1.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_rms_norm_dtype_consistency() -> Result<()> {
        let device = Device::Cpu;
        let rms_norm = RMSNorm::new(2, None, &device)?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 2), &device)?;
        let (output, _) = rms_norm.forward(&x, None)?;

        assert_eq!(output.dtype(), DType::F32);
        Ok(())
    }
}
