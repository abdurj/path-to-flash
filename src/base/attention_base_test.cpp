#include "attention_base.hpp"
#include <gtest/gtest.h>
#include "absl/log/check.h"

namespace ptflash {

// Simple mock implementation for testing
class MockAttention : public Attention {
public:
    static absl::StatusOr<std::unique_ptr<MockAttention>> Create(
        int seq_len, int d_model, int num_heads) {
        return CreateValidated<MockAttention>(seq_len, d_model, num_heads);
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
    friend class Attention;  // Allow base class to access private constructor
    MockAttention(int seq_len, int d_model, int num_heads)
        : Attention(seq_len, d_model, num_heads) {}
};

class AttentionBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto result = MockAttention::Create(4, 8, 2);
        CHECK_OK(result);
        attention_ = std::move(result.value());
    }
    
    std::unique_ptr<MockAttention> attention_;
};

TEST_F(AttentionBaseTest, BasicForwardPass) {
    std::vector<float> input(4 * 8, 1.0f); // seq_len=4, d_model=8
    
    auto result = attention_->Forward(input);
    
    CHECK_OK(result);
    EXPECT_EQ(result->size(), input.size());
    EXPECT_EQ(*result, input); // Mock returns input unchanged
}

TEST_F(AttentionBaseTest, InvalidInputSize) {
    std::vector<float> input(10, 1.0f); // Wrong size
    
    auto result = attention_->Forward(input);
    
    CHECK(!result.ok());
    CHECK_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(AttentionBaseTest, BenchmarkWorks) {
    auto result = attention_->Benchmark(5);
    
    CHECK_OK(result);
    EXPECT_GT(result.value(), 0.0);
}

TEST_F(AttentionBaseTest, InvalidCreationParameters) {
    auto result = MockAttention::Create(-1, 8, 2); // Invalid seq_len
    CHECK(!result.ok());
    
    result = MockAttention::Create(4, 0, 2); // Invalid d_model
    CHECK(!result.ok());
    
    result = MockAttention::Create(4, 7, 2); // d_model not divisible by num_heads
    CHECK(!result.ok());
}

} // namespace ptflash
