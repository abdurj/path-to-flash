#include "naive_attention.hpp"
#include "../common/matrix_utils.hpp"
#include <gtest/gtest.h>
#include "absl/log/check.h"

namespace ptflash {

class NaiveAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        seq_len_ = 4;
        d_model_ = 8; 
        num_heads_ = 2;
        
        auto result = NaiveAttention::Create(seq_len_, d_model_, num_heads_);
        CHECK_OK(result);
        attention_ = std::move(result.value());
    }
    
    int seq_len_;
    int d_model_;
    int num_heads_;
    std::unique_ptr<NaiveAttention> attention_;
};

TEST_F(NaiveAttentionTest, CreationSucceeds) {
    // Test that we can create the attention mechanism
    EXPECT_EQ(attention_->seq_len(), seq_len_);
    EXPECT_EQ(attention_->d_model(), d_model_);
    EXPECT_EQ(attention_->num_heads(), num_heads_);
    EXPECT_EQ(attention_->head_dim(), d_model_ / num_heads_);
    EXPECT_EQ(attention_->Name(), "NaiveAttention");
}

TEST_F(NaiveAttentionTest, ForwardPassShape) {
    // Test that forward pass returns correct output shape
    std::vector<float> input(seq_len_ * d_model_, 0.5f);
    
    auto result = attention_->Forward(input);
    CHECK_OK(result);
    
    EXPECT_EQ(result->size(), input.size());
}

TEST_F(NaiveAttentionTest, ForwardPassDeterministic) {
    // Test that multiple calls with same input produce same output
    // (assuming deterministic implementation)
    std::vector<float> input(seq_len_ * d_model_, 0.3f);
    
    auto result1 = attention_->Forward(input);
    auto result2 = attention_->Forward(input);
    
    CHECK_OK(result1);
    CHECK_OK(result2);
    
    EXPECT_EQ(*result1, *result2);
}

TEST_F(NaiveAttentionTest, InvalidInputSize) {
    // Test error handling for wrong input size
    std::vector<float> input(10, 1.0f);  // Wrong size
    
    auto result = attention_->Forward(input);
    CHECK(!result.ok());
}

TEST_F(NaiveAttentionTest, BenchmarkWorks) {
    // Test that benchmarking runs without error
    auto result = attention_->Benchmark(3);  // Small number for fast test
    
    CHECK_OK(result);
    EXPECT_GT(result.value(), 0.0);  // Should take some positive time
}

// TODO: Add more tests as you implement functionality
// 
// Suggested additional tests:
// - Test with different input values
// - Test attention weight properties (should sum to 1.0)
// - Test multi-head splitting and concatenation
// - Test with edge cases (very small/large values)
// - Test numerical stability
// 
// Example test for attention weights:
// TEST_F(NaiveAttentionTest, AttentionWeightsSumToOne) {
//     // This test requires access to intermediate attention weights
//     // You might need to add a debug method to expose them
// }

}  // namespace ptflash
