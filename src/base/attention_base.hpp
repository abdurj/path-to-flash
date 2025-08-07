#pragma once

#include <vector>
#include <memory>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

namespace ptflash {

/**
 * Abstract base class for all attention implementations.
 * 
 * This class defines the common interface that all attention phases will implement.
 * Each phase (naive CPU, optimized CPU, CUDA, Flash Attention, etc.) will inherit
 * from this base class and provide their own implementation.
 * 
 * The interface is designed to be simple and consistent across all phases while
 * allowing for performance measurement and correctness verification.
 */
class Attention {
public:
    virtual ~Attention() = default;
    
    // Delete copy constructor and assignment operator
    Attention(const Attention&) = delete;
    Attention& operator=(const Attention&) = delete;
    
    // Allow move operations
    Attention(Attention&&) = default;
    Attention& operator=(Attention&&) = delete;

    /**
     * Forward pass through attention mechanism.
     * 
     * Computes: Attention(Q, K, V) = softmax(QK^T / √d_k) * V
     * For multi-head: Concat(head_1, ..., head_h) * W^O
     * 
     * @param input Input tensor of shape [seq_len, d_model]
     * @return StatusOr containing output tensor of shape [seq_len, d_model] or error
     */
    virtual absl::StatusOr<std::vector<float>> Forward(const std::vector<float>& input) = 0;
    
    /**
     * Run benchmark to measure performance.
     * 
     * @param iterations Number of iterations to run
     * @return StatusOr containing average time per iteration in milliseconds or error
     */
    virtual absl::StatusOr<double> Benchmark(int iterations = 100) = 0;
    
    // Getters for configuration
    int seq_len() const { return seq_len_; }
    int d_model() const { return d_model_; }
    int num_heads() const { return num_heads_; }
    int head_dim() const { return head_dim_; }
    
    /**
     * Get human-readable name for this implementation.
     * Used for benchmarking and logging.
     */
    virtual std::string Name() const = 0;

protected:
    /**
     * Protected constructor for derived classes.
     */
    Attention(int seq_len, int d_model, int num_heads);
    
    /**
     * Validate input dimensions (accessible to derived classes).
     */
    absl::Status ValidateInput(const std::vector<float>& input) const;
    
    /**
     * Validate construction parameters and create instance of derived class.
     * This handles all parameter validation in one place.
     */
    template<typename DerivedType>
    static absl::StatusOr<std::unique_ptr<DerivedType>> Create(
        int seq_len, int d_model, int num_heads) {
        
        // Validate constructor parameters
        if (seq_len <= 0) {
            return absl::InvalidArgumentError(
                absl::StrFormat("seq_len must be positive, got %d", seq_len));
        }
        
        if (d_model <= 0) {
            return absl::InvalidArgumentError(
                absl::StrFormat("d_model must be positive, got %d", d_model));
        }
        
        if (num_heads <= 0) {
            return absl::InvalidArgumentError(
                absl::StrFormat("num_heads must be positive, got %d", num_heads));
        }
        
        if (d_model % num_heads != 0) {
            return absl::InvalidArgumentError(
                absl::StrFormat("d_model (%d) must be divisible by num_heads (%d)", 
                               d_model, num_heads));
        }
        
        return std::unique_ptr<DerivedType>(
            new DerivedType(seq_len, d_model, num_heads));
    }

private:
    const int seq_len_;
    const int d_model_;
    const int num_heads_;
    const int head_dim_;
};

}  // namespace ptflash
