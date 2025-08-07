#pragma once

#include <vector>

namespace ptflash {
namespace Softmax {

/**
 * Apply softmax function to matrix rows.
 * 
 * Uses numerically stable computation: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 * This prevents overflow that would occur with naive exp(x) computation.
 * 
 * @param matrix Input matrix [rows, cols] (flattened row-major)
 * @param rows Number of rows
 * @param cols Number of columns  
 * @return Matrix with softmax applied to each row independently
 */
std::vector<float> ApplySoftmax(
    const std::vector<float>& matrix,
    int rows, int cols);

/**
 * Apply softmax to a single vector.
 * 
 * @param vector Input vector
 * @return Vector with softmax applied
 */
std::vector<float> ApplySoftmax(const std::vector<float>& vector);

// TODO: Implement these functions in softmax.cpp
//
// Key points for implementation:
// 1. Numerical stability is crucial - always subtract the maximum value first
// 2. Handle edge cases like all-zero input or very large values
// 3. Ensure output sums to 1.0 (within floating point precision)
// 4. For matrix version, apply softmax to each row independently

}  // namespace Softmax
}  // namespace ptflash
