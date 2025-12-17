use candle_core::{D, Device, Tensor};

// Import from main crate
extern crate nano_vllm_candle;
use nano_vllm_candle::layers::layernorm::RMSNorm;
use nano_vllm_candle::layers::rotary_embedding::RotaryEmbedding;
use nano_vllm_candle::models::qwen3::Qwen3ForCausalLM;

#[test]
fn test_simple_forward() {
    let device = Device::Cpu;
    let model_dir = r"G:\workspace\huggingface\Qwen3-0.6B";

    let model = Qwen3ForCausalLM::from_hf_dir(model_dir, &device).expect("Failed to load model");

    // Simple input: just token 1000
    let input_ids = Tensor::new(&[[1000u32]], &device).unwrap();

    let hidden = model.forward(&input_ids).expect("Forward failed");
    let logits = model.compute_logits(&hidden).expect("Logits failed");

    let logits_vec: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();

    // Get top-5 token predictions
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 predictions for token 1000:");
    for (idx, logit) in indexed.iter().take(5) {
        println!("  token {}: logit {:.4}", idx, logit);
    }

    // Verify shape
    assert_eq!(logits.dims(), &[1, 1, 151936]);
}

#[test]
fn test_multi_token_forward() {
    let device = Device::Cpu;
    let model_dir = r"G:\workspace\huggingface\Qwen3-0.6B";

    let model = Qwen3ForCausalLM::from_hf_dir(model_dir, &device).expect("Failed to load model");

    // Input: "Hello" - tokenized using simple tokens
    // Using tokens that should produce reasonable continuations
    let input_ids = Tensor::new(&[[1000u32, 2000, 3000]], &device).unwrap();

    let hidden = model.forward(&input_ids).expect("Forward failed");
    let logits = model.compute_logits(&hidden).expect("Logits failed");

    // Get predictions for last token
    let last_logits = logits.narrow(1, 2, 1).unwrap().squeeze(1).unwrap();
    let logits_vec: Vec<f32> = last_logits.flatten_all().unwrap().to_vec1().unwrap();

    // Get top-5 token predictions
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Multi-token test - Top 5 predictions for last position:");
    for (idx, logit) in indexed.iter().take(5) {
        println!("  token {}: logit {:.4}", idx, logit);
    }

    // Also check the variance of logits - should not be too low (uniform distribution would indicate broken model)
    let mean: f32 = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
    let var: f32 = logits_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / logits_vec.len() as f32;
    println!("Logits stats: mean={:.4} var={:.4}", mean, var);

    // Variance should be substantial for a working model
    assert!(var > 1.0, "Logit variance too low: {}", var);
}

#[test]
fn test_embedding_values() {
    let device = Device::Cpu;
    let model_dir = r"G:\workspace\huggingface\Qwen3-0.6B";

    let model = Qwen3ForCausalLM::from_hf_dir(model_dir, &device).expect("Failed to load model");

    // Check if embedding values look reasonable
    let embed_sample = model.model.embed.narrow(0, 0, 5).unwrap();
    let sample_vals: Vec<f32> = embed_sample.flatten_all().unwrap().to_vec1().unwrap();

    println!("First embedding row (first 10 values):");
    for v in &sample_vals[..10] {
        print!("{:.4} ", v);
    }
    println!();

    // Check embedding for "Hello" token 9707
    let hello_embed = model.model.embed.narrow(0, 9707, 1).unwrap();
    let hello_vals: Vec<f32> = hello_embed.flatten_all().unwrap().to_vec1().unwrap();
    println!("Embedding for 'Hello' (token 9707), first 10 values:");
    for v in &hello_vals[..10] {
        print!("{:.4} ", v);
    }
    println!();

    // Embeddings should not be all zeros or all the same
    let mean: f32 = sample_vals.iter().sum::<f32>() / sample_vals.len() as f32;
    let var: f32 = sample_vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / sample_vals.len() as f32;

    println!("Embedding stats: mean={:.6} var={:.6}", mean, var);
    assert!(
        var > 1e-6,
        "Embedding variance too low - weights may not be loaded correctly"
    );
}

#[test]
fn test_greedy_generation() {
    let device = Device::Cpu;
    let model_dir = r"G:\workspace\huggingface\Qwen3-0.6B";

    let model = Qwen3ForCausalLM::from_hf_dir(model_dir, &device).expect("Failed to load model");

    // "Hello, my name is" - real text that should produce meaningful continuation
    let mut input_ids: Vec<u32> = vec![9707, 11, 847, 829, 374];

    println!("Starting tokens: {:?}", input_ids);

    // Greedy generation for 10 tokens
    for step in 0..10 {
        let input = Tensor::from_vec(input_ids.clone(), (1, input_ids.len()), &device).unwrap();
        let hidden = model.forward(&input).expect("Forward failed");
        let logits = model.compute_logits(&hidden).expect("Logits failed");

        // Get logits for last position
        let seq_len = input_ids.len();
        let last_logits = logits.narrow(1, seq_len - 1, 1).unwrap().squeeze(1).unwrap();
        let logits_vec: Vec<f32> = last_logits.flatten_all().unwrap().to_vec1().unwrap();

        // Show top-5 predictions for first step
        if step == 0 {
            let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("Step 0 top-5:");
            for (idx, logit) in indexed.iter().take(5) {
                println!("  token {}: logit {:.4}", idx, logit);
            }
        }

        // Argmax for greedy decoding
        let (next_id, max_logit) = logits_vec
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        println!("Step {}: predicted token {} with logit {:.4}", step, next_id, max_logit);
        input_ids.push(next_id as u32);

        // EOS check
        if next_id == 151645 {
            println!("Hit EOS token");
            break;
        }
    }

    println!("Final sequence: {:?}", input_ids);
}

#[test]
fn test_single_position_logits() {
    // Test that logits at position 0 match when computed alone vs in sequence
    let device = Device::Cpu;
    let model_dir = r"G:\workspace\huggingface\Qwen3-0.6B";

    let model = Qwen3ForCausalLM::from_hf_dir(model_dir, &device).expect("Failed to load model");

    // Single token
    let single_input = Tensor::new(&[[9707u32]], &device).unwrap();
    let single_hidden = model.forward(&single_input).expect("Forward failed");
    let single_logits = model.compute_logits(&single_hidden).expect("Logits failed");
    let single_vec: Vec<f32> = single_logits.flatten_all().unwrap().to_vec1().unwrap();

    // Two tokens - check position 0 logits
    let double_input = Tensor::new(&[[9707u32, 11]], &device).unwrap();
    let double_hidden = model.forward(&double_input).expect("Forward failed");
    let double_logits = model.compute_logits(&double_hidden).expect("Logits failed");
    let pos0_logits = double_logits.narrow(1, 0, 1).unwrap().squeeze(1).unwrap();
    let double_vec: Vec<f32> = pos0_logits.flatten_all().unwrap().to_vec1().unwrap();

    // Compare - they should be identical due to causal masking
    let diff: f32 = single_vec
        .iter()
        .zip(double_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / single_vec.len() as f32;

    println!("Mean abs diff between single token and pos0 of double: {:.6}", diff);
    println!(
        "Single max: {:.4}, Double pos0 max: {:.4}",
        single_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        double_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // They should be nearly identical
    assert!(diff < 0.01, "Logits differ too much: {}", diff);
}

#[test]
fn test_layer_0_hidden_state() {
    // Compare intermediate values with Python reference
    use candle_core::DType;
    use candle_nn::VarBuilder;
    use candle_nn::ops::softmax;

    let device = Device::Cpu;
    let model_path = std::path::Path::new(r"G:\workspace\huggingface\Qwen3-0.6B\model.safetensors");

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path.to_path_buf()], DType::F32, &device)
            .expect("Failed to mmap safetensors")
    };

    // Load weights
    let embed_weight: Tensor = vb.get((151936, 1024), "model.embed_tokens.weight").unwrap();
    let input_ln_w: Tensor = vb.get((1024,), "model.layers.0.input_layernorm.weight").unwrap();
    let q_proj_w: Tensor = vb.get((2048, 1024), "model.layers.0.self_attn.q_proj.weight").unwrap();
    let k_proj_w: Tensor = vb.get((1024, 1024), "model.layers.0.self_attn.k_proj.weight").unwrap();
    let q_norm_w: Tensor = vb.get((128,), "model.layers.0.self_attn.q_norm.weight").unwrap();

    // Get embedding for "Hello" (token 9707)
    let hello_embed = embed_weight.narrow(0, 9707, 1).unwrap(); // [1, 1024]
    let x = hello_embed.reshape((1, 1, 1024)).unwrap(); // [B=1, T=1, H=1024]

    println!("Embedding shape: {:?}", x.dims());
    let x_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    println!("Embedding first 5: {:?}", &x_vals[..5]);

    // Step 1: RMSNorm
    let x_flat = x.reshape((1, 1024)).unwrap();
    let x2 = x_flat.sqr().unwrap();
    let var = x2.mean_keepdim(1).unwrap(); // [1, 1]
    let eps = 1e-6f64;
    let x_norm = x_flat
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
    let ln_w = input_ln_w.reshape((1, 1024)).unwrap();
    let x_normed = x_norm.broadcast_mul(&ln_w).unwrap();

    let x_normed_vals: Vec<f32> = x_normed.flatten_all().unwrap().to_vec1().unwrap();
    println!("After input RMSNorm first 10: {:?}", &x_normed_vals[..10]);

    // Python reference: [-0.005116, -0.39082798, 0.42305645, -0.10953392, 0.10066038, ...]
    let expected_normed = [-0.005116, -0.39082798, 0.42305645, -0.10953392, 0.10066038];
    println!("Expected: {:?}", expected_normed);

    for i in 0..5 {
        let diff = (x_normed_vals[i] - expected_normed[i]).abs();
        println!(
            "  idx {}: rust={:.6} python={:.6} diff={:.6}",
            i, x_normed_vals[i], expected_normed[i], diff
        );
    }

    // Step 2: Q projection
    // x_normed: [1, 1024] @ q_proj_w.T: [1024, 2048] -> [1, 2048]
    let q = x_normed.matmul(&q_proj_w.transpose(0, 1).unwrap()).unwrap();
    let q_vals: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
    println!("Q first 10: {:?}", &q_vals[..10]);

    // Python: [0.11234201, 0.17359026, 0.06209425, -0.12143741, 0.00986537, ...]
    let expected_q = [0.11234201, 0.17359026, 0.06209425, -0.12143741, 0.00986537];
    println!("Expected Q: {:?}", expected_q);

    // Check Q after Q-norm for head 0
    // q: [1, 2048] -> reshape to [1, 16, 128] -> take head 0 [1, 128]
    let q_reshaped = q.reshape((1, 16, 128)).unwrap();
    let q_head0 = q_reshaped.narrow(1, 0, 1).unwrap().squeeze(1).unwrap(); // [1, 128]

    // Apply RMSNorm with q_norm weight
    let q2 = q_head0.sqr().unwrap();
    let var = q2.mean_keepdim(1).unwrap();
    let q_norm_tmp = q_head0
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
    let qnw = q_norm_w.reshape((1, 128)).unwrap();
    let q_normed = q_norm_tmp.broadcast_mul(&qnw).unwrap();

    let q_normed_vals: Vec<f32> = q_normed.flatten_all().unwrap().to_vec1().unwrap();
    println!("Q after norm (head 0) first 10: {:?}", &q_normed_vals[..10]);

    // Python: [2.6336122, 1.1155888, -0.23591788, -1.0700169, 0.13238329, ...]
    let expected_qn = [2.6336122, 1.1155888, -0.23591788, -1.0700169, 0.13238329];
    println!("Expected Q normed: {:?}", expected_qn);

    for i in 0..5 {
        let diff = (q_normed_vals[i] - expected_qn[i]).abs();
        println!(
            "  Q_norm idx {}: rust={:.6} python={:.6} diff={:.6}",
            i, q_normed_vals[i], expected_qn[i], diff
        );
    }
}

#[test]
fn test_compare_with_nano_vllm() {
    // Test layer 0 output against Python reference
    use candle_core::DType;
    use candle_nn::{Module, VarBuilder};
    use nano_vllm_candle::layers::layernorm::RMSNorm;

    let device = Device::Cpu;
    let model_dir = r"G:\workspace\huggingface\Qwen3-0.6B";

    let model = Qwen3ForCausalLM::from_hf_dir(model_dir, &device).expect("Failed to load model");

    // Single token "Hello"
    let input_ids = Tensor::new(&[[9707u32]], &device).unwrap();

    // Get embedding
    let ids = input_ids.to_dtype(DType::U32).unwrap();
    let flat_ids = ids.flatten_all().unwrap();
    let emb_from_model = model.model.embed.embedding(&flat_ids).unwrap();
    let h = emb_from_model.reshape((1, 1, 1024)).unwrap();

    // Run layer 0 only (access via model.layers)
    // Note: We can't directly access layers due to privacy, so let's trace the full model

    // For now, let's verify the full model's first few values and compare with expected
    // Python layer 0 output:
    // hidden_states (first 5): [0.05532067, -0.03219886, 0.14347184, -0.32862923, -0.16485177]
    // residual (first 5): [-0.10749146, 0.24382538, -0.1059664, -0.849572, 0.10264889]

    // Run full forward
    let hidden = model.forward(&input_ids).expect("Forward failed");
    let hidden_vals: Vec<f32> = hidden.flatten_all().unwrap().to_vec1().unwrap();
    println!("Final hidden (first 5): {:?}", &hidden_vals[..5]);

    // The final hidden state should be the output after all 28 layers + final norm
    // Let's focus on checking that the computation is numerically stable

    // Check for NaN or Inf
    let has_nan = hidden_vals.iter().any(|&x| x.is_nan());
    let has_inf = hidden_vals.iter().any(|&x| x.is_infinite());
    assert!(!has_nan, "Hidden state contains NaN!");
    assert!(!has_inf, "Hidden state contains Inf!");

    println!("No NaN or Inf in hidden state");

    // Check the magnitude is reasonable (not exploding)
    let max_abs = hidden_vals.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!("Max absolute value in hidden: {:.4}", max_abs);

    // With 28 layers, values can grow, but should still be reasonable
    // The observed value of ~17 seems reasonable

    // Get logits and predictions
    let logits = model.compute_logits(&hidden).expect("Logits failed");
    let logits_vals: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();

    let mut indexed: Vec<(usize, f32)> = logits_vals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 predictions:");
    for (idx, logit) in indexed.iter().take(5) {
        println!("  token {}: logit {:.4}", idx, logit);
    }
}

#[test]
fn test_weight_loading() {
    // Verify that specific weight values match safetensors exactly
    use candle_core::DType;
    use candle_nn::VarBuilder;

    let device = Device::Cpu;
    let model_path = std::path::Path::new(r"G:\workspace\huggingface\Qwen3-0.6B\model.safetensors");

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path.to_path_buf()], DType::F32, &device)
            .expect("Failed to mmap safetensors")
    };

    // Check embedding for token 9707 ("Hello") - compare with Python reference
    // Python: [-0.00100708 -0.01477051  0.01916504 -0.00402832  0.01330566 ...]
    let embed_weight: Tensor = vb
        .get((151936, 1024), "model.embed_tokens.weight")
        .expect("Failed to load embed_tokens.weight");
    let hello_embed = embed_weight.narrow(0, 9707, 1).unwrap();
    let hello_vals: Vec<f32> = hello_embed.flatten_all().unwrap().to_vec1().unwrap();
    println!("Embedding token 9707 (Hello), first 10 values: {:?}", &hello_vals[..10]);

    // Expected from Python
    let expected = [-0.00100708, -0.01477051, 0.01916504, -0.00402832, 0.01330566];
    for i in 0..5 {
        let diff = (hello_vals[i] - expected[i]).abs();
        assert!(
            diff < 1e-4,
            "Embedding mismatch at {}: rust={}, python={}",
            i,
            hello_vals[i],
            expected[i]
        );
    }
    println!("Embedding values match Python reference!");

    // Check q_norm weight
    // Python: [ 4.53125 1.2421875 -0.734375 1.703125 2.59375 ...]
    let q_norm = vb
        .get((128,), "model.layers.0.self_attn.q_norm.weight")
        .expect("Failed to load q_norm");
    let q_norm_vals: Vec<f32> = q_norm.flatten_all().unwrap().to_vec1().unwrap();
    println!("Layer 0 q_norm weight, first 10 values: {:?}", &q_norm_vals[..10]);

    let expected_qnorm = [4.53125, 1.2421875, -0.734375, 1.703125, 2.59375];
    for i in 0..5 {
        let diff = (q_norm_vals[i] - expected_qnorm[i]).abs();
        assert!(
            diff < 1e-4,
            "q_norm mismatch at {}: rust={}, python={}",
            i,
            q_norm_vals[i],
            expected_qnorm[i]
        );
    }
    println!("q_norm weight values match Python reference!");
}

#[test]
fn test_rope_values() {
    let device = Device::Cpu;
    let head_dim = 128;
    let rope = RotaryEmbedding::new(head_dim, 4096, 1000000.0, &device);

    // Create test Q and K tensors [B=1, nh=2, T=4, hd=128]
    let q = Tensor::randn(0f32, 1.0, (1, 2, 4, head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (1, 2, 4, head_dim), &device).unwrap();

    let (q_rot, k_rot) = rope.apply(&q, &k).unwrap();

    // Check that norms are preserved (RoPE is a rotation)
    let q_norm_before = q.sqr().unwrap().sum_keepdim(D::Minus1).unwrap();
    let q_norm_after = q_rot.sqr().unwrap().sum_keepdim(D::Minus1).unwrap();
    let diff = (q_norm_before.sub(&q_norm_after).unwrap())
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();

    println!("Norm difference: {}", diff);
    assert!(diff < 1e-3, "RoPE should preserve norms, diff={}", diff);

    // Check that position 0 has different rotation than position 1
    let q0 = q_rot.narrow(2, 0, 1).unwrap(); // Position 0
    let q1 = q_rot.narrow(2, 1, 1).unwrap(); // Position 1

    // They should be different (unless original Q was identical)
    let q_orig0 = q.narrow(2, 0, 1).unwrap();
    let q_orig1 = q.narrow(2, 1, 1).unwrap();

    // If original Q[0] != Q[1], then after RoPE, Q_rot[0] != Q_rot[1]
    // More importantly, even if Q[0] == Q[1], after RoPE they should differ
    // because different positions get different rotations

    // Let's test with identical input at all positions
    let q_same = Tensor::ones((1, 2, 4, head_dim), candle_core::DType::F32, &device).unwrap();
    let k_same = Tensor::ones((1, 2, 4, head_dim), candle_core::DType::F32, &device).unwrap();

    let (q_rot_same, _) = rope.apply(&q_same, &k_same).unwrap();

    // Position 0 and position 1 should have different values after RoPE
    let qr0: Vec<f32> = q_rot_same
        .narrow(2, 0, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let qr1: Vec<f32> = q_rot_same
        .narrow(2, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let diff_01: f32 = qr0.iter().zip(qr1.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("Difference between position 0 and 1 (same input): {}", diff_01);
    assert!(diff_01 > 1.0, "Different positions should have different rotations");
}
