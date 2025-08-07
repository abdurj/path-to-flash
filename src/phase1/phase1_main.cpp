#include "src/phase1/naive_attention.hpp"

#include <fmt/format.h>

#include "src/common/matrix_utils.hpp"
#include "src/common/macros.hpp"


using namespace ptflash;

int main() {
    fmt::print("Phase 1: Naive Attention Implementation\n");
    fmt::print("======================================\n\n");

    // Test configuration - start small for debugging
    int seq_len = 4;
    int d_model = 8;
    int num_heads = 2;

    fmt::print("Configuration:\n");
    fmt::print("  Sequence length: {}\n", seq_len);
    fmt::print("  Model dimension: {}\n", d_model);
    fmt::print("  Number of heads: {}\n", num_heads);
    fmt::print("  Head dimension: {}\n\n", d_model / num_heads);

    // Create attention instance
    std::unique_ptr<NaiveAttention> attention = EVAL_OR_ASSERT(NaiveAttention::Create(seq_len, d_model, num_heads));
    fmt::print("Created {} successfully!\n\n", attention->Name());

    // Create sample input - simple test pattern
    fmt::print("Creating sample input...\n");
    auto input = MatrixUtils::random_matrix<float>(seq_len, d_model);
    
    // Print first few values for debugging
    fmt::print("Input sample (first 8 values): ");
    for (int i = 0; i < std::min(8, (int)input.size()); ++i) {
        fmt::print("{:.3f} ", input[i]);
    }
    fmt::print("\n\n");

    // Run forward pass
    fmt::print("Running forward pass...\n");
    auto output = EVAL_OR_ASSERT(attention->Forward(input));
    fmt::print("Forward pass completed successfully!\n");
    fmt::print("Output size: {}\n", output.size());
    
    // Print first few output values
    fmt::print("Output sample (first 8 values): ");
    for (int i = 0; i < std::min(8, (int)output.size()); ++i) {
        fmt::print("{:.3f} ", output[i]);
    }
    fmt::print("\n\n");


    // Basic sanity checks
    bool passed_checks = true;
    
    // Check 1: Output size matches input size
    if (output.size() != input.size()) {
        fmt::print("❌ FAIL: Output size ({}) != Input size ({})\n", 
                   output.size(), input.size());
        passed_checks = false;
    } else {
        fmt::print("✅ PASS: Output size matches input size\n");
    }

    // Check 2: Output contains finite values (no NaN/inf)
    bool all_finite = true;
    for (float val : output) {
        if (!std::isfinite(val)) {
            all_finite = false;
            break;
        }
    }

    
    if (all_finite) {
        fmt::print("✅ PASS: All output values are finite\n");
    } else {
        fmt::print("❌ FAIL: Output contains NaN or infinity values\n");
        passed_checks = false;
    }


    // Check 3: Output is not identical to input (unless that's expected)
    bool identical = (output == input);
    if (identical) {
        fmt::print("⚠️  WARNING: Output is identical to input - is attention implemented?\n");
    } else {
        fmt::print("✅ PASS: Output differs from input\n");
    }
    fmt::print("\n");



    // Run benchmark
    fmt::print("Running benchmark (5 iterations)...\n");
    auto benchmark_result = attention->Benchmark(5);
    
    if (!benchmark_result.ok()) {
        fmt::print("Benchmark failed: {}\n", benchmark_result.status().message());
    } else {
        fmt::print("Average time per iteration: {:.3f} ms\n", benchmark_result.value());
    }
    fmt::print("\n");


    
    if (passed_checks) {
        fmt::print("🎉 Basic tests passed! Your implementation is working.\n");
        fmt::print("\nNext steps:\n");
        fmt::print("1. Implement the missing TODO sections\n"); 
        fmt::print("2. Run unit tests: bazel test //src/phase1:unit_tests\n");
        fmt::print("3. Verify attention weights sum to 1.0\n");
        fmt::print("4. Test with larger inputs (seq_len=64, d_model=256)\n");
        fmt::print("5. Move on to Phase 2 optimizations!\n");
    } else {
        fmt::print("❌ Some tests failed. Check your implementation.\n");
        return 1;
    }

    
    return 0;
}
