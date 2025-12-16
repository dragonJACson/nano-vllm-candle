use candle_core::{Module, Result, Tensor};

pub struct LinearBase {
    tp_dim: usize,
    tp_rank: usize,
    tp_size: usize,
    weight: Tensor,
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
    pub fn load_weights(
        &mut self,
        weight: &Tensor,
        _bias: Option<&Tensor>,
        loaded_shard_id: usize,
    ) -> Result<()> {
        let shard_offset =
            self.output_size[..loaded_shard_id].iter().sum::<usize>() / self.base.tp_size;
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

enum QKVShardId {
    Query,
    Key,
    Value,
}

impl QKVParallelLinear {
    pub fn load_qkv_weight(&mut self, qkv_weight: &Tensor, shard_id: QKVShardId) -> Result<()> {
        let (shard_size, shard_offset) = match shard_id {
            QKVShardId::Query => (self.num_heads * self.head_size, 0),
            QKVShardId::Key => (
                self.num_kv_heads * self.head_size,
                self.num_heads * self.head_size,
            ),
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

mod test {
    use crate::layers::linear::{
        ColumnParallelLinear, LinearBase, ParallelLinear, ReplicatedLinear,
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
            &[
                [7.0, 2.0, 4.0, 6.0, 9.0, 3.0],
                [3.0, -2.0, 0.0, 6.0, 1.0, -1.0],
            ],
            &Device::Cpu,
        )?;

        assert_eq!(output.to_vec2::<f64>()?, expected_output.to_vec2::<f64>()?);

        Ok(())
    }

    #[test]
    fn test_column_parallel_linear() -> Result<()> {
        let x = Tensor::new(&[[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]], &Device::Cpu)?;

        let weight_0 = Tensor::new(
            &[
                [1.0, 0.0, -1.0, 2.0],
                [0.0, 1.0, 2.0, -1.0],
                [2.0, -1.0, 0.0, 1.0],
            ],
            &Device::Cpu,
        )?;
        let bias_0 = Tensor::new(&[1.0, -2.0, 0.0], &Device::Cpu)?;

        let weight_1 = Tensor::new(
            &[
                [-2.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [3.0, 0.0, -2.0, 1.0],
            ],
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

        assert_eq!(
            output_0.to_vec2::<f64>()?,
            expected_output_0.to_vec2::<f64>()?
        );
        assert_eq!(
            output_1.to_vec2::<f64>()?,
            expected_output_1.to_vec2::<f64>()?
        );

        Ok(())
    }
}
