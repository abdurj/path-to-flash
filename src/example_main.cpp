#include "base/attention_base.hpp"
#include "common/matrix_utils.hpp"
#include <fmt/format.h>
#include <memory>

using namespace p2f;

// Example implementation that will be replaced with real phases later
class ExampleAttention : public AttentionBase {
public:
    static absl::StatusOr<std::unique_ptr<ExampleAttention>> Create(
        int seq_len, int d_model, int num_heads) {
        
        // Use base class validation
        auto base_result = AttentionBase::Create(seq_len, d_model, num_heads);
        if (!base_result.ok()) {
            return base_result.status();
        }
        
        return std::unique_ptr<ExampleAttention>(
            new ExampleAttention(seq_len, d_model, num_heads));
    }
    
    absl::StatusOr<std::vector<float>> Forward(const std::vector<float>& input) override {
        auto status = ValidateInput(input);
        if (!status.ok()) {
            return status;
        }
        
        // For now, just return the input (identity function)
        // This will be replaced with actual attention computation in Phase 1
        fmt::print("Running {} forward pass (seq_len={}, d_model={}, num_heads={})\n", 
                   Name(), seq_len(), d_model(), num_heads());
        
        return input;
    }
    
    absl::StatusOr<double> Benchmark(int iterations = 100) override {
        if (iterations <= 0) {
            return absl::InvalidArgumentError("iterations must be positive");
        }
        
        fmt::print("Benchmarking {} for {} iterations...\n", Name(), iterations);
        
        // Create sample input
        std::vector<float> input(seq_len() * d_model(), 0.5f);
        
        // Run forward pass multiple times (timing would be added later)
        for (int i = 0; i < iterations; ++i) {
            auto result = Forward(input);
            if (!result.ok()) {
                return result.status();
            }
        }
        
        // Mock timing result for now
        return 1.0; // 1ms per iteration
    }
    
    std::string Name() const override {
        return "ExampleAttention";
    }

private:
    ExampleAttention(int seq_len, int d_model, int num_heads)
        : AttentionBase(seq_len, d_model, num_heads) {}
};

int main() {
    fmt::print("Path To Flash - Attention Implementation\n");
    fmt::print("=======================================\n\n");
    
    // Configuration
    int seq_len = 512;
    int d_model = 768;
    int num_heads = 12;
    
    // Create attention instance
    auto attention_result = ExampleAttention::Create(seq_len, d_model, num_heads);
    if (!attention_result.ok()) {
        fmt::print("Error creating attention: {}\n", attention_result.status().message());
        return 1;
    }
    
    auto attention = std::move(attention_result.value());
    
    fmt::print("Created {} with configuration:\n", attention->Name());
    fmt::print("  - Sequence length: {}\n", attention->seq_len());
    fmt::print("  - Model dimension: {}\n", attention->d_model());
    fmt::print("  - Number of heads: {}\n", attention->num_heads());
    fmt::print("  - Head dimension: {}\n\n", attention->head_dim());
    
    // Create sample input using matrix utils
    auto input = MatrixUtils::random_matrix<float>(seq_len, d_model);
    fmt::print("Created random input matrix: {}x{}\n\n", seq_len, d_model);
    
    // Run forward pass
    auto output_result = attention->Forward(input);
    if (!output_result.ok()) {
        fmt::print("Error in forward pass: {}\n", output_result.status().message());
        return 1;
    }
    
    fmt::print("Forward pass completed. Output size: {}\n\n", output_result->size());
    
    // Run benchmark
    auto benchmark_result = attention->Benchmark(10); // Reduced iterations for demo
    if (!benchmark_result.ok()) {
        fmt::print("Error in benchmark: {}\n", benchmark_result.status().message());
        return 1;
    }
    
    fmt::print("Average time per iteration: {:.2f}ms\n", benchmark_result.value());
    
    return 0;
}
