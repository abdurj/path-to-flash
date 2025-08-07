#include "src/phase1/naive_attention.hpp"
#include "src/phase1/matrix_ops.hpp"
#include "src/phase1/softmax.hpp"

#include "src/common/matrix_utils.hpp"

#include <chrono>

namespace ptflash {

absl::StatusOr<std::unique_ptr<NaiveAttention>> NaiveAttention::Create(
    int seq_len, int d_model, int num_heads) {
    return Attention::Create<NaiveAttention>(seq_len, d_model, num_heads);
}

NaiveAttention::NaiveAttention(int seq_len, int d_model, int num_heads)
    : Attention(seq_len, d_model, num_heads) {
    InitializeWeights();
}

void NaiveAttention::InitializeWeights() {
    // TODO: Initialize all weight matrices
    // Hint: Use MatrixUtils::random_matrix<float>(rows * cols, -0.1f, 0.1f)
    // 
    // Example:
    // w_q_ = MatrixUtils::random_matrix<float>(d_model() * d_model(), -0.1f, 0.1f);
    
    // TODO: Initialize w_q_, w_k_, w_v_, w_o_ here
    // Each should be of size [d_model * d_model] for flattened row-major storage
    
    // Optional: Initialize bias vectors (all zeros is fine to start)
    // b_q_ = std::vector<float>(d_model(), 0.0f);
}

absl::StatusOr<std::vector<float>> NaiveAttention::Forward(const std::vector<float>& input) {
    // Step 1: Validate input
    auto status = ValidateInput(input);
    if (!status.ok()) {
        return status;
    }
    
    // TODO: Implement the forward pass
    // Follow the algorithm in the README:
    //
    // 1. Linear projections: Q = input * W_q, K = input * W_k, V = input * W_v
    // 2. Reshape for multi-head processing  
    // 3. For each head: compute attention(Q_head, K_head, V_head)
    // 4. Concatenate all head outputs
    // 5. Final output projection: output = concat * W_o
    
    // Step 2: Linear projections
    // TODO: Compute Q, K, V using LinearProjection()
    // std::vector<float> Q = LinearProjection(input, w_q_, b_q_, seq_len(), d_model(), d_model());
    
    // Step 3: Multi-head attention
    // TODO: Split Q, K, V into heads and process each head
    // Hint: Each head processes dimensions [seq_len, head_dim] where head_dim = d_model/num_heads
    
    // Step 4: Concatenate heads and output projection
    // TODO: Combine all head outputs and apply final projection
    
    // Placeholder return - replace with actual computation
    return std::vector<float>(input.size(), 0.0f);
}

absl::StatusOr<double> NaiveAttention::Benchmark(int iterations) {
    if (iterations <= 0) {
        return absl::InvalidArgumentError("iterations must be positive");
    }
    
    // Create sample input
    auto input = MatrixUtils::random_matrix<float>(seq_len(), d_model());
    
    // Warm-up run
    auto warmup_result = Forward(input);
    if (!warmup_result.ok()) {
        return warmup_result.status();
    }
    
    // TODO: Implement timing logic
    // Use std::chrono::high_resolution_clock to measure execution time
    // Run the forward pass 'iterations' times and return average time in milliseconds
    
    // Placeholder - replace with actual timing
    return 1.0;
}

std::vector<float> NaiveAttention::LinearProjection(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    int rows, int cols, int out_dim) const {
    
    // TODO: Implement matrix multiplication + bias addition
    // output[i][j] = sum_k(input[i][k] * weights[k][j]) + bias[j]
    //
    // Remember: matrices are stored in flattened row-major format
    // - input[i * cols + k] accesses element at row i, column k
    // - weights[k * out_dim + j] accesses weight from input dim k to output dim j
    // - output[i * out_dim + j] accesses output at row i, column j
    
    std::vector<float> output(rows * out_dim, 0.0f);
    
    // TODO: Implement the matrix multiplication here
    
    return output;
}

std::vector<float> NaiveAttention::SingleHeadAttention(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V) const {
    
    // TODO: Implement single-head attention computation
    // 1. Compute attention scores: scores = Q * K^T / sqrt(head_dim)
    // 2. Apply softmax to get attention weights
    // 3. Compute output: output = attention_weights * V
    
    int head_dim = this->head_dim();
    int seq_len = this->seq_len();
    
    // Step 1: Compute Q * K^T / sqrt(head_dim)
    // TODO: Use MatrixOps::MatMul and MatrixOps::Transpose
    
    // Step 2: Apply softmax row-wise
    // TODO: Use Softmax::ApplySoftmax
    
    // Step 3: Multiply attention weights by V
    // TODO: Final matrix multiplication
    
    // Placeholder return - replace with actual computation
    return std::vector<float>(seq_len * head_dim, 0.0f);
}

}  // namespace ptflash
