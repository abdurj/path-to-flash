#include "matrix_ops.hpp"
#include "softmax.hpp" 
#include <gtest/gtest.h>
#include "absl/log/check.h"

namespace ptflash {

// Unit tests for matrix operations
class MatrixOpsTest : public ::testing::Test {};

TEST_F(MatrixOpsTest, MatMulBasic) {
    // Test basic 2x2 * 2x2 multiplication
    std::vector<float> A = {1, 2, 3, 4};  // [[1,2], [3,4]]
    std::vector<float> B = {5, 6, 7, 8};  // [[5,6], [7,8]]
    
    auto result = MatrixOps::MatMul(A, B, 2, 2, 2);
    
    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22], [43,50]]
    std::vector<float> expected = {19, 22, 43, 50};
    
    // TODO: Uncomment once MatMul is implemented
    // CHECK_OK(result);
    // EXPECT_EQ(*result, expected);
}

TEST_F(MatrixOpsTest, TransposeBasic) {
    // Test basic 2x3 transpose
    std::vector<float> A = {1, 2, 3, 4, 5, 6};  // [[1,2,3], [4,5,6]]
    
    auto result = MatrixOps::Transpose(A, 2, 3);
    
    // Expected: [[1,4], [2,5], [3,6]]
    std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    
    // TODO: Uncomment once Transpose is implemented  
    // EXPECT_EQ(result, expected);
}

// Unit tests for softmax
class SoftmaxTest : public ::testing::Test {};

TEST_F(SoftmaxTest, BasicSoftmax) {
    // Test softmax on simple vector
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    
    auto result = Softmax::ApplySoftmax(input);
    
    // Check that result sums to 1.0 (within floating point precision)
    float sum = 0.0f;
    for (float val : result) {
        sum += val;
        EXPECT_GT(val, 0.0f);  // All values should be positive
    }
    
    // TODO: Uncomment once ApplySoftmax is implemented
    // EXPECT_NEAR(sum, 1.0f, 1e-6);
}

TEST_F(SoftmaxTest, NumericalStability) {
    // Test with large values that would cause overflow without stability tricks
    std::vector<float> input = {1000.0f, 1001.0f, 1002.0f};
    
    auto result = Softmax::ApplySoftmax(input);
    
    // Should not contain NaN or infinity
    for (float val : result) {
        EXPECT_TRUE(std::isfinite(val));
        EXPECT_FALSE(std::isnan(val));
    }
}

// TODO: Add more comprehensive tests
//
// Suggested tests:
// - Matrix multiplication with non-square matrices
// - Edge cases (empty matrices, 1x1 matrices)
// - Softmax with all-zero input
// - Softmax with negative values
// - Performance tests for large matrices

}  // namespace ptflash
