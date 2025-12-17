use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;

extern crate nano_vllm_candle;
use candle_nn::{Linear, Module};
use nano_vllm_candle::layers::layernorm::RMSNorm;
use nano_vllm_candle::layers::rotary_embedding::RotaryEmbedding;

/// Test candle's repeat vs numpy's repeat behavior for GQA
#[test]
fn test_repeat_behavior() {
    let device = Device::Cpu;

    // Create a tensor [1, 2, 1, 3] representing [B, kv_heads=2, T=1, head_dim=3]
    // Values: head0 = [1, 2, 3], head1 = [4, 5, 6]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::from_vec(data, (1, 2, 1, 3), &device).unwrap();

    println!(
        "Original tensor: {:?}",
        x.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    );

    // Repeat with factor 2 to get [1, 4, 1, 3]
    let repeated = x.repeat((1, 2, 1, 1)).unwrap();
    let repeated_vals: Vec<f32> = repeated.flatten_all().unwrap().to_vec1().unwrap();

    println!("Repeated tensor: {:?}", repeated_vals);
    println!("Expected for numpy.repeat (interleaved): [1,2,3, 1,2,3, 4,5,6, 4,5,6]");
    println!("Expected for tile (concatenated): [1,2,3, 4,5,6, 1,2,3, 4,5,6]");

    // Check what we got - numpy.repeat should give interleaved: [head0, head0, head1, head1]
    // while tile gives: [head0, head1, head0, head1]

    // For GQA, we want head0 to correspond to Q_heads 0,1 and head1 to correspond to Q_heads 2,3
    // This means we need interleaved: [head0, head0, head1, head1]

    let expected_interleaved: Vec<f32> = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0];
    let expected_tiled: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    if repeated_vals == expected_interleaved {
        println!("✓ Candle repeat behaves like numpy.repeat (interleaved) - CORRECT for GQA");
    } else if repeated_vals == expected_tiled {
        println!("✗ Candle repeat behaves like numpy.tile (concatenated) - WRONG for GQA!");
    } else {
        println!("? Unknown behavior");
    }

    // Now test the correct approach: unsqueeze -> expand -> reshape
    // x: [1, 2, 1, 3] -> unsqueeze(2) -> [1, 2, 1, 1, 3] -> expand [1, 2, 2, 1, 3] -> reshape [1, 4, 1, 3]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::from_vec(data, (1, 2, 1, 3), &device).unwrap();
    let repeat = 2usize;
    let (b, kv, t, hd) = x.dims4().unwrap();
    let x_fixed = x
        .unsqueeze(2)
        .unwrap() // [1, 2, 1, 1, 3]
        .expand((b, kv, repeat, t, hd))
        .unwrap() // [1, 2, 2, 1, 3]
        .reshape((b, kv * repeat, t, hd))
        .unwrap(); // [1, 4, 1, 3]

    let fixed_vals: Vec<f32> = x_fixed.flatten_all().unwrap().to_vec1().unwrap();
    println!("\nFixed GQA repeat: {:?}", fixed_vals);

    if fixed_vals == expected_interleaved {
        println!("✓ Fixed approach gives correct interleaved result!");
    } else {
        println!("✗ Fixed approach still incorrect: {:?}", fixed_vals);
    }
}

/// Manual layer-by-layer forward pass to debug numerical differences
#[test]
fn test_layer_by_layer_debug() {
    let device = Device::Cpu;
    let model_path = std::path::Path::new(r"G:\workspace\huggingface\Qwen3-0.6B\model.safetensors");

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path.to_path_buf()], DType::F32, &device)
            .expect("Failed to mmap safetensors")
    };

    // Config for Qwen3-0.6B
    let hidden_size = 1024;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;
    let intermediate_size = 3072;
    let eps = 1e-6f64;
    let rope_theta = 1_000_000.0f32;

    // Load embedding
    let embed_weight: Tensor = vb.get((151936, hidden_size), "model.embed_tokens.weight").unwrap();

    // Get embedding for "Hello" (token 9707)
    let input_embedding = embed_weight.narrow(0, 9707, 1).unwrap(); // [1, 1024]

    println!(
        "Embedding (first 5): {:?}",
        input_embedding.flatten_all().unwrap().to_vec1::<f32>().unwrap()[..5].to_vec()
    );

    // Process layer 0 and layer 1
    let mut hidden = input_embedding.clone();
    let mut residual: Option<Tensor> = None;

    for layer_idx in 0..3 {
        println!("\n=== Layer {} ===", layer_idx);

        // Load layer weights
        let prefix = format!("model.layers.{}", layer_idx);

        let input_ln_w: Tensor = vb
            .get((hidden_size,), format!("{}.input_layernorm.weight", prefix).as_str())
            .unwrap();
        let input_ln_w = input_ln_w.reshape((1, hidden_size)).unwrap();

        let q_proj_w: Tensor = vb
            .get(
                (num_heads * head_dim, hidden_size),
                format!("{}.self_attn.q_proj.weight", prefix).as_str(),
            )
            .unwrap();
        let k_proj_w: Tensor = vb
            .get(
                (num_kv_heads * head_dim, hidden_size),
                format!("{}.self_attn.k_proj.weight", prefix).as_str(),
            )
            .unwrap();
        let v_proj_w: Tensor = vb
            .get(
                (num_kv_heads * head_dim, hidden_size),
                format!("{}.self_attn.v_proj.weight", prefix).as_str(),
            )
            .unwrap();
        let o_proj_w: Tensor = vb
            .get(
                (hidden_size, num_heads * head_dim),
                format!("{}.self_attn.o_proj.weight", prefix).as_str(),
            )
            .unwrap();
        let q_norm_w: Tensor = vb
            .get((head_dim,), format!("{}.self_attn.q_norm.weight", prefix).as_str())
            .unwrap();
        let k_norm_w: Tensor = vb
            .get((head_dim,), format!("{}.self_attn.k_norm.weight", prefix).as_str())
            .unwrap();

        let post_ln_w: Tensor = vb
            .get(
                (hidden_size,),
                format!("{}.post_attention_layernorm.weight", prefix).as_str(),
            )
            .unwrap();
        let post_ln_w = post_ln_w.reshape((1, hidden_size)).unwrap();

        let gate_proj: Tensor = vb
            .get(
                (intermediate_size, hidden_size),
                format!("{}.mlp.gate_proj.weight", prefix).as_str(),
            )
            .unwrap();
        let up_proj: Tensor = vb
            .get(
                (intermediate_size, hidden_size),
                format!("{}.mlp.up_proj.weight", prefix).as_str(),
            )
            .unwrap();
        let down_proj: Tensor = vb
            .get(
                (hidden_size, intermediate_size),
                format!("{}.mlp.down_proj.weight", prefix).as_str(),
            )
            .unwrap();

        // Step 1: Input LayerNorm with residual
        let (h_norm, new_residual) = if let Some(ref res) = residual {
            // hidden + residual, then normalize
            let combined = hidden.broadcast_add(res).unwrap();
            let var = combined.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
            let x_norm = combined
                .broadcast_mul(
                    &(var
                        .broadcast_add(&Tensor::new(&[[eps as f32]], &device).unwrap())
                        .unwrap()
                        .sqrt()
                        .unwrap()
                        .recip()
                        .unwrap()),
                )
                .unwrap();
            let h_ln = x_norm.broadcast_mul(&input_ln_w).unwrap();
            (h_ln, combined)
        } else {
            // Just normalize, residual = original hidden
            let var = hidden.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
            let x_norm = hidden
                .broadcast_mul(
                    &(var
                        .broadcast_add(&Tensor::new(&[[eps as f32]], &device).unwrap())
                        .unwrap()
                        .sqrt()
                        .unwrap()
                        .recip()
                        .unwrap()),
                )
                .unwrap();
            let h_ln = x_norm.broadcast_mul(&input_ln_w).unwrap();
            (h_ln, hidden.clone())
        };

        let h_norm_vals: Vec<f32> = h_norm.flatten_all().unwrap().to_vec1().unwrap();
        println!("After input LN (first 5): {:?}", &h_norm_vals[..5]);

        // Step 2: QKV projection
        let q = h_norm.matmul(&q_proj_w.transpose(0, 1).unwrap()).unwrap(); // [1, 2048]
        let k = h_norm.matmul(&k_proj_w.transpose(0, 1).unwrap()).unwrap(); // [1, 1024]
        let v = h_norm.matmul(&v_proj_w.transpose(0, 1).unwrap()).unwrap(); // [1, 1024]

        if layer_idx == 0 {
            let q_vals: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
            println!("Q proj (first 10): {:?}", &q_vals[..10]);
            let k_vals: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
            println!("K proj (first 10): {:?}", &k_vals[..10]);
        }

        // Reshape for heads: [1, seq_len=1, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]
        let q = q.reshape((1, 1, num_heads, head_dim)).unwrap().transpose(1, 2).unwrap();
        let k = k
            .reshape((1, 1, num_kv_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v = v
            .reshape((1, 1, num_kv_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        // Step 3: Q/K norm (per-head RMSNorm)
        let q_flat = q.reshape((num_heads, head_dim)).unwrap();
        let k_flat = k.reshape((num_kv_heads, head_dim)).unwrap();

        // RMSNorm for Q
        let q_var = q_flat.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        let q_norm_tmp = q_flat
            .broadcast_mul(
                &(q_var
                    .broadcast_add(&Tensor::new(&[[eps as f32]], &device).unwrap().reshape((1, 1)).unwrap())
                    .unwrap()
                    .sqrt()
                    .unwrap()
                    .recip()
                    .unwrap()),
            )
            .unwrap();
        let q_norm_w_2d = q_norm_w.reshape((1, head_dim)).unwrap();
        let q_normed = q_norm_tmp.broadcast_mul(&q_norm_w_2d).unwrap();
        let q = q_normed.reshape((1, num_heads, 1, head_dim)).unwrap();

        if layer_idx == 0 {
            let q_normed_vals: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
            println!("Q after norm (head 0, first 10): {:?}", &q_normed_vals[..10]);
        }

        // RMSNorm for K
        let k_var = k_flat.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        let k_norm_tmp = k_flat
            .broadcast_mul(
                &(k_var
                    .broadcast_add(&Tensor::new(&[[eps as f32]], &device).unwrap().reshape((1, 1)).unwrap())
                    .unwrap()
                    .sqrt()
                    .unwrap()
                    .recip()
                    .unwrap()),
            )
            .unwrap();
        let k_norm_w_2d = k_norm_w.reshape((1, head_dim)).unwrap();
        let k_normed = k_norm_tmp.broadcast_mul(&k_norm_w_2d).unwrap();
        let k = k_normed.reshape((1, num_kv_heads, 1, head_dim)).unwrap();

        // Step 4: RoPE (position 0)
        let rope = RotaryEmbedding::new(head_dim, 4096, rope_theta, &device);
        let (q, k) = rope.apply(&q, &k).unwrap();

        if layer_idx == 0 {
            let q_rope_vals: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
            println!("Q after RoPE (head 0, first 10): {:?}", &q_rope_vals[..10]);
        }

        // Step 5: GQA repeat K/V
        let repeat = num_heads / num_kv_heads;
        let k = k.repeat((1, repeat, 1, 1)).unwrap();
        let v = v.repeat((1, repeat, 1, 1)).unwrap();

        // Step 6: Attention (for single token, output = V because softmax(scalar) = 1)
        // Actually we should compute properly even for single token
        let kt = k.transpose(2, 3).unwrap();
        let scale = (head_dim as f32).powf(-0.5);
        let attn_scores = (q.matmul(&kt).unwrap() * scale as f64).unwrap();
        let attn_probs = candle_nn::ops::softmax(&attn_scores, D::Minus1).unwrap();
        let ctx = attn_probs.matmul(&v).unwrap();

        if layer_idx == 0 {
            let ctx_vals: Vec<f32> = ctx.flatten_all().unwrap().to_vec1().unwrap();
            println!("Context (head 0, first 10): {:?}", &ctx_vals[..10]);
        }

        // Step 7: Output projection
        // ctx shape: [1, num_heads, 1, head_dim] -> transpose to [1, 1, num_heads, head_dim]
        let ctx_transposed = ctx.transpose(1, 2).unwrap();
        // Reshape to [1, num_heads * head_dim]
        let ctx_2d = ctx_transposed.reshape((1, num_heads * head_dim)).unwrap();

        if layer_idx == 0 {
            let ctx_2d_vals: Vec<f32> = ctx_2d.flatten_all().unwrap().to_vec1().unwrap();
            println!("Context 2D for o_proj (first 10): {:?}", &ctx_2d_vals[..10]);
            // Also print values at offset 128 (should be head 1's first values)
            println!("Context 2D at offset 128 (head 1): {:?}", &ctx_2d_vals[128..138]);
            println!("o_proj_w shape: {:?}", o_proj_w.dims());
        }

        let attn_out = ctx_2d.matmul(&o_proj_w.transpose(0, 1).unwrap()).unwrap();

        let attn_out_vals: Vec<f32> = attn_out.flatten_all().unwrap().to_vec1().unwrap();
        println!("Attn output (first 5): {:?}", &attn_out_vals[..5]);

        // Step 8: Post-attention LayerNorm with residual
        let combined = attn_out.broadcast_add(&new_residual).unwrap();
        let var = combined.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        let x_norm = combined
            .broadcast_mul(
                &(var
                    .broadcast_add(&Tensor::new(&[[eps as f32]], &device).unwrap())
                    .unwrap()
                    .sqrt()
                    .unwrap()
                    .recip()
                    .unwrap()),
            )
            .unwrap();
        let h_norm = x_norm.broadcast_mul(&post_ln_w).unwrap();
        let new_residual_post = combined.clone();

        // Step 9: MLP
        let gate = h_norm.matmul(&gate_proj.transpose(0, 1).unwrap()).unwrap();
        let up = h_norm.matmul(&up_proj.transpose(0, 1).unwrap()).unwrap();
        let activated = gate.silu().unwrap().mul(&up).unwrap();
        let mlp_out = activated.matmul(&down_proj.transpose(0, 1).unwrap()).unwrap();

        let mlp_out_vals: Vec<f32> = mlp_out.flatten_all().unwrap().to_vec1().unwrap();
        println!("MLP output (first 5): {:?}", &mlp_out_vals[..5]);

        // Update for next layer
        hidden = mlp_out;
        residual = Some(new_residual_post);

        let res_vals: Vec<f32> = residual.as_ref().unwrap().flatten_all().unwrap().to_vec1().unwrap();
        println!("Residual (first 5): {:?}", &res_vals[..5]);
    }
}
