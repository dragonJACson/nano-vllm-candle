use candle_core::{Module, Result, Tensor};

pub struct VocabParallelEmbedding {
    weight: Tensor,
}

impl Module for VocabParallelEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

impl VocabParallelEmbedding {
    pub fn load_weight(&self, _weight: &Tensor) {}
}
