use crate::tp;
use candle_core::{DType, Device, Module, Result, Tensor};

pub struct LinearBase {
    tp_dim: usize,
    tp_rank: usize,
    tp_size: usize,
    weight: Tensor,
}

impl LinearBase {
    pub fn new(tp_dim: usize, tp_rank: usize, tp_size: usize, input_size: usize, output_size: usize) -> Self {
        Self {
            tp_dim,
            tp_rank,
            tp_size,
            weight: Tensor::zeros((output_size, input_size), DType::F32, &Device::Cpu).unwrap(),
        }
    }
}

pub trait ParallelLinear: Module {
    fn load_weights(&mut self, weight: &Tensor, bias: Option<&Tensor>) -> Result<()>;
}

pub struct ReplicatedLinear {
    base: LinearBase,
    input_size: usize,
    output_size: usize,
    bias: Option<Tensor>,
}

impl Module for ReplicatedLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let layer = candle_nn::Linear::new(self.base.weight.clone(), self.bias.clone());
        layer.forward(x)
    }
}

impl ParallelLinear for ReplicatedLinear {
    fn load_weights(&mut self, weight: &Tensor, bias: Option<&Tensor>) -> Result<()> {
        self.base.weight = weight.clone();
        self.bias = bias.cloned();
        Ok(())
    }
}

pub struct ColumnParallelLinear {
    base: LinearBase,
    input_size: usize,
    output_size: usize,
    bias: Option<Tensor>,
}

impl ColumnParallelLinear {
    pub fn new(input_size: usize, output_size: usize, bias: Option<Tensor>) -> Self {
        let tp_cfg = tp::get_tp();
        Self {
            base: LinearBase {
                tp_dim: 0,
                tp_rank: tp_cfg.rank,
                tp_size: tp_cfg.size,
                weight: Tensor::zeros((output_size, input_size), DType::F32, &Device::Cpu).unwrap(),
            },
            input_size,
            output_size,
            bias,
        }
    }
}

impl Module for ColumnParallelLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let layer = candle_nn::Linear::new(self.base.weight.clone(), self.bias.clone());
        layer.forward(x)
    }
}

impl ParallelLinear for ColumnParallelLinear {
    fn load_weights(&mut self, weight: &Tensor, bias: Option<&Tensor>) -> Result<()> {
        let shard_size = weight.dim(self.base.tp_dim)?;
        let start_idx = self.base.tp_rank * shard_size;

        let loaded_weight = weight.narrow(self.base.tp_dim, start_idx, shard_size)?;

        self.base.weight = loaded_weight.clone();
        self.bias = bias.cloned();
        Ok(())
    }
}

pub struct MergedColumnParallelLinear {
    base: LinearBase,
    input_size: usize,
    output_size: Vec<usize>,
    bias: Option<Tensor>,
}

impl Module for MergedColumnParallelLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let layer = candle_nn::Linear::new(self.base.weight.clone(), self.bias.clone());
        layer.forward(x)
    }
}

impl MergedColumnParallelLinear {
    pub fn load_weights(&mut self, weight: &Tensor, _bias: Option<&Tensor>, loaded_shard_id: usize) -> Result<()> {
        let shard_offset = self.output_size[..loaded_shard_id].iter().sum::<usize>() / self.base.tp_size;
        let shard_size = self.output_size[loaded_shard_id] / self.base.tp_size;

        let loaded_weight = weight
            .narrow(self.base.tp_dim, shard_offset, shard_size)?
            .chunk(self.base.tp_size, self.base.tp_dim)?;

        self.base.weight = loaded_weight[self.base.tp_rank].clone();

        Ok(())
    }
}

pub struct QKVParallelLinear {
    pub linear: ColumnParallelLinear,
    hidden_size: usize,
    head_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
}

pub enum QKVShardId {
    Query,
    Key,
    Value,
}

impl QKVParallelLinear {
    pub fn new(
        hidden_size: usize, head_size: usize, num_heads: usize, num_kv_heads: usize, bias: Option<Tensor>,
    ) -> Self {
        let output_size = (num_heads + 2 * num_kv_heads) * head_size;

        Self {
            linear: ColumnParallelLinear::new(hidden_size, output_size, bias),
            hidden_size: output_size,
            head_size,
            num_heads,
            num_kv_heads,
        }
    }

    pub fn load_qkv_weight(&mut self, qkv_weight: &Tensor, shard_id: QKVShardId) -> Result<()> {
        let (shard_size, shard_offset) = match shard_id {
            QKVShardId::Query => (self.num_heads * self.head_size, 0),
            QKVShardId::Key => (self.num_kv_heads * self.head_size, self.num_heads * self.head_size),
            QKVShardId::Value => (
                self.num_kv_heads * self.head_size,
                (self.num_heads + self.num_kv_heads) * self.head_size,
            ),
        };

        let loaded_weight = qkv_weight
            .narrow(self.linear.base.tp_dim, shard_offset, shard_size)?
            .chunk(self.linear.base.tp_size, self.linear.base.tp_dim)?;

        self.linear
            .load_weights(&loaded_weight[self.linear.base.tp_rank], None)?;

        Ok(())
    }
}

impl Module for QKVParallelLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

pub struct RowParallelLinear {
    base: LinearBase,
    input_size: usize,
    output_size: usize,
    bias: Option<Tensor>,
}

impl Module for RowParallelLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let layer = candle_nn::Linear::new(
            self.base.weight.clone(),
            if self.base.tp_rank == 0 {
                self.bias.clone()
            } else {
                None
            },
        )
        .forward(x)?;

        Ok(layer)
    }
}

impl ParallelLinear for RowParallelLinear {
    fn load_weights(&mut self, weight: &Tensor, bias: Option<&Tensor>) -> Result<()> {
        let shard_size = weight.dim(self.base.tp_dim)?;
        let start_idx = self.base.tp_rank * shard_size;

        let loaded_weight = weight.narrow(self.base.tp_dim, start_idx, shard_size)?;

        self.base.weight = loaded_weight.clone();
        self.bias = bias.cloned();
        Ok(())
    }
}

impl RowParallelLinear {
    pub fn new(input_size: usize, output_size: usize, bias: Option<Tensor>) -> Self {
        let tp_cfg = tp::get_tp();
        Self {
            base: LinearBase::new(0, tp_cfg.rank, tp_cfg.size, input_size, output_size),
            input_size,
            output_size,
            bias,
        }
    }
}

mod test {
    use crate::layers::linear::{
        ColumnParallelLinear, LinearBase, ParallelLinear, QKVParallelLinear, ReplicatedLinear,
    };
    use candle_core::{DType, Device, Module, Result, Tensor};

    #[test]
    fn test_replicated_linear() -> Result<()> {
        let x = Tensor::new(&[[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]], &Device::Cpu)?;

        let weight = Tensor::new(
            &[
                [1.0, 0.0, -1.0, 2.0],
                [0.0, 1.0, 2.0, -1.0],
                [2.0, -1.0, 0.0, 1.0],
                [-2.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [3.0, 0.0, -2.0, 1.0],
            ],
            &Device::Cpu,
        )?;
        let bias = Tensor::new(&[1.0, -2.0, 0.0, 3.0, -1.0, 2.0], &Device::Cpu)?;

        let layer = ReplicatedLinear {
            base: LinearBase {
                tp_dim: 0,
                tp_rank: 0,
                tp_size: 1,
                weight: weight.clone(),
            },
            input_size: 4,
            output_size: 6,
            bias: Some(bias.clone()),
        };

        let output = layer.forward(&x)?;

        let expected_output = Tensor::new(
            &[[7.0, 2.0, 4.0, 6.0, 9.0, 3.0], [3.0, -2.0, 0.0, 6.0, 1.0, -1.0]],
            &Device::Cpu,
        )?;

        assert_eq!(output.to_vec2::<f64>()?, expected_output.to_vec2::<f64>()?);

        Ok(())
    }

    #[test]
    fn test_column_parallel_linear() -> Result<()> {
        let x = Tensor::new(&[[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]], &Device::Cpu)?;

        let weight_0 = Tensor::new(
            &[[1.0, 0.0, -1.0, 2.0], [0.0, 1.0, 2.0, -1.0], [2.0, -1.0, 0.0, 1.0]],
            &Device::Cpu,
        )?;
        let bias_0 = Tensor::new(&[1.0, -2.0, 0.0], &Device::Cpu)?;

        let weight_1 = Tensor::new(
            &[[-2.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [3.0, 0.0, -2.0, 1.0]],
            &Device::Cpu,
        )?;
        let bias_1 = Tensor::new(&[3.0, -1.0, 2.0], &Device::Cpu)?;

        let layer_0 = ColumnParallelLinear {
            base: LinearBase {
                tp_dim: 0,
                tp_rank: 0,
                tp_size: 2,
                weight: weight_0.clone(),
            },
            input_size: 4,
            output_size: 3,
            bias: Some(bias_0.clone()),
        };

        let layer_1 = ColumnParallelLinear {
            base: LinearBase {
                tp_dim: 0,
                tp_rank: 1,
                tp_size: 2,
                weight: weight_1.clone(),
            },
            input_size: 4,
            output_size: 3,
            bias: Some(bias_1.clone()),
        };

        let output_0 = layer_0.forward(&x)?;
        let output_1 = layer_1.forward(&x)?;

        let expected_output_0 = Tensor::new(&[[7.0, 2.0, 4.0], [3.0, -2.0, 0.0]], &Device::Cpu)?;
        let expected_output_1 = Tensor::new(&[[6.0, 9.0, 3.0], [6.0, 1.0, -1.0]], &Device::Cpu)?;

        assert_eq!(output_0.to_vec2::<f64>()?, expected_output_0.to_vec2::<f64>()?);
        assert_eq!(output_1.to_vec2::<f64>()?, expected_output_1.to_vec2::<f64>()?);

        Ok(())
    }

    #[test]
    fn test_qkv_parallel_linear_split() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 4;
        let head = 2;
        let num_heads = 2;
        let kv_heads = 1;
        let mut qkv = QKVParallelLinear::new(hidden, head, num_heads, kv_heads, None);

        let w_q = Tensor::new(&[[1.0, 0.0, -1.0, 2.0], [0.0, 1.0, 2.0, -1.0]], &device)?; // [2,4]
        let w_k = Tensor::new(&[[2.0, -1.0, 0.0, 1.0]], &device)?; // [1,4]
        let w_v = Tensor::new(&[[-2.0, 1.0, 1.0, 0.0]], &device)?; // [1,4]
        let qkv_cat = Tensor::cat(&[w_q.clone(), w_k.clone(), w_v.clone()], 0)?; // [4,4]
        qkv.linear.load_weights(&qkv_cat, None)?;

        let x = Tensor::new(&[[1.0, 2.0, 3.0, 4.0]], &device)?; // [1,4]
        let out = qkv.forward(&x)?; // [1, 2 + 1 + 1]

        let q_out = out.narrow(1, 0, 2)?;
        let k_out = out.narrow(1, 2, 1)?;
        let v_out = out.narrow(1, 3, 1)?;

        let expected_q = x.matmul(&w_q.transpose(0, 1)?)?;
        let expected_k = x.matmul(&w_k.transpose(0, 1)?)?;
        let expected_v = x.matmul(&w_v.transpose(0, 1)?)?;

        assert_eq!(q_out.to_vec2::<f64>()?, expected_q.to_vec2::<f64>()?);
        assert_eq!(k_out.to_vec2::<f64>()?, expected_k.to_vec2::<f64>()?);
        assert_eq!(v_out.to_vec2::<f64>()?, expected_v.to_vec2::<f64>()?);
        Ok(())
    }
}
