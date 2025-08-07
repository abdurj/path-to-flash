#pragma once

#include "src/base/attention_base.hpp"
#include <vector>

namespace ptflash {

/**
 * Phase 1: Naive CPU Attention Implementation
 * 
 * This is an unoptimized, straightforward implementation of multi-head attention.
 * It prioritizes clarity and correctness over performance. This serves as our
 * baseline for future optimizations.
 * 
 * Implementation notes:
 * - Uses simple nested loops for matrix operations
 * - No SIMD or vectorization optimizations
 * - Memory layout is not optimized for cache efficiency
 * - Follows the mathematical definition closely
 */
class NaiveAttention : public Attention {
public:
    /**
     * Create a NaiveAttention instance.
     * 
     * @param seq_len Sequence length
     * @param d_model Model dimension  
     * @param num_heads Number of attention heads
     * @return StatusOr containing the created instance or error
     */
    static absl::StatusOr<std::unique_ptr<NaiveAttention>> Create(
        int seq_len, int d_model, int num_heads);

    /**
     * Forward pass through attention mechanism.
     * 
     * Computes: Attention(Q, K, V) = softmax(QK^T / √d_k) * V
     * 
     * @param input Input tensor [seq_len, d_model] (flattened row-major)
     * @return Output tensor [seq_len, d_model] or error status
     */
    absl::StatusOr<std::vector<float>> Forward(const std::vector<float>& input) override;

    /**
     * Benchmark the forward pass performance.
     * 
     * @param iterations Number of iterations to run
     * @return Average time per iteration in milliseconds
     */
    absl::StatusOr<double> Benchmark(int iterations = 100) override;

    /**
     * Get implementation name for logging/benchmarking.
     */
    std::string Name() const override { return "NaiveAttention"; }

private:
    friend class Attention;
    
    /**
     * Private constructor - use Create() instead.
     */
    NaiveAttention(int seq_len, int d_model, int num_heads);

    /**
     * Initialize weight matrices with random values.
     * Uses simple Gaussian initialization (mean=0, std=0.1).
     */
    void InitializeWeights();

    /**
     * Apply linear projection: output = input * weights + bias
     * 
     * @param input Input matrix [rows, cols] (flattened)
     * @param weights Weight matrix [cols, out_dim] (flattened)
     * @param bias Bias vector [out_dim] (can be empty)
     * @param rows Number of input rows
     * @param cols Number of input columns
     * @param out_dim Output dimension
     * @return Projected output [rows, out_dim] (flattened)
     */
    std::vector<float> LinearProjection(
        const std::vector<float>& input,
        const std::vector<float>& weights,
        const std::vector<float>& bias,
        int rows, int cols, int out_dim) const;

    /**
     * Compute attention for a single head.
     * 
     * @param Q Query matrix [seq_len, head_dim] (flattened)
     * @param K Key matrix [seq_len, head_dim] (flattened) 
     * @param V Value matrix [seq_len, head_dim] (flattened)
     * @return Attention output [seq_len, head_dim] (flattened)
     */
    std::vector<float> SingleHeadAttention(
        const std::vector<float>& Q,
        const std::vector<float>& K, 
        const std::vector<float>& V) const;

    // Weight matrices - TODO: Fill in the member variables
    // Hint: You need Q, K, V, and Output projection weights
    // Each projection matrix should be [d_model, d_model]
    // Consider: Do you need bias vectors?
    
    // TODO: Add weight matrix member variables here
    // std::vector<float> w_q_;    // Query projection weights [d_model * d_model]
    // std::vector<float> w_k_;    // Key projection weights [d_model * d_model]  
    // std::vector<float> w_v_;    // Value projection weights [d_model * d_model]
    // std::vector<float> w_o_;    // Output projection weights [d_model * d_model]
    
    // Optional: bias vectors (can start without these)
    // std::vector<float> b_q_;    // Query bias [d_model]
    // std::vector<float> b_k_;    // Key bias [d_model]
    // std::vector<float> b_v_;    // Value bias [d_model] 
    // std::vector<float> b_o_;    // Output bias [d_model]
};

}  // namespace ptflash
