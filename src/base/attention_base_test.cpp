#include "attention_base.hpp"
#include <gtest/gtest.h>

namespace p2f {

// Simple mock implementation for testing
class MockAttention : public AttentionBase {
public:
    static absl::StatusOr<std::unique_ptr<MockAttention>> Create(
        int seq_len, int d_model, int num_heads) {
        
        // Use the base class validation
        auto base_result = AttentionBase::Create(seq_len, d_model, num_heads);
        if (!base_result.ok()) {
            return base_result.status();
        }
        
        return std::unique_ptr<MockAttention>(
            new MockAttention(seq_len, d_model, num_heads));
    }
    
    absl::StatusOr<std::vector<float>> Forward(const std::vector<float>& input) override {
        auto status = ValidateInput(input);
        if (!status.ok()) {
            return status;
        }
        
        // Return input unchanged (identity function)
        return input;
    }
    
    absl::StatusOr<double> Benchmark(int iterations = 100) override {
        if (iterations <= 0) {
            return absl::InvalidArgumentError("iterations must be positive");
        }
        return 1.0; // Mock 1ms per iteration
    }
    
    std::string Name() const override {
        return "MockAttention";
    }

private:
    MockAttention(int seq_len, int d_model, int num_heads)
        : AttentionBase(seq_len, d_model, num_heads) {}
};

class AttentionBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto result = MockAttention::Create(4, 8, 2);
        ASSERT_TRUE(result.ok());
        attention_ = std::move(result.value());
    }
    
    std::unique_ptr<MockAttention> attention_;
};

TEST_F(AttentionBaseTest, BasicForwardPass) {
    std::vector<float> input(4 * 8, 1.0f); // seq_len=4, d_model=8
    
    auto result = attention_->Forward(input);
    
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->size(), input.size());
    EXPECT_EQ(*result, input); // Mock returns input unchanged
}

TEST_F(AttentionBaseTest, InvalidInputSize) {
    std::vector<float> input(10, 1.0f); // Wrong size
    
    auto result = attention_->Forward(input);
    
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(AttentionBaseTest, BenchmarkWorks) {
    auto result = attention_->Benchmark(5);
    
    ASSERT_TRUE(result.ok());
    EXPECT_GT(result.value(), 0.0);
}

TEST_F(AttentionBaseTest, InvalidCreationParameters) {
    auto result = MockAttention::Create(-1, 8, 2); // Invalid seq_len
    EXPECT_FALSE(result.ok());
    
    result = MockAttention::Create(4, 0, 2); // Invalid d_model
    EXPECT_FALSE(result.ok());
    
    result = MockAttention::Create(4, 7, 2); // d_model not divisible by num_heads
    EXPECT_FALSE(result.ok());
}

} // namespace p2f
