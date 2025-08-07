#pragma once

#include <vector>
#include "absl/status/statusor.h"

namespace ptflash {
namespace MatrixOps {

/**
 * Matrix multiplication: C = A * B
 * 
 * @param A Matrix A [rows_A, cols_A] (flattened row-major)
 * @param B Matrix B [cols_A, cols_B] (flattened row-major)  
 * @param rows_A Number of rows in A
 * @param cols_A Number of columns in A (must equal rows in B)
 * @param cols_B Number of columns in B
 * @return Result matrix C [rows_A, cols_B] (flattened row-major)
 */
absl::StatusOr<std::vector<float>> MatMul(
    const std::vector<float>& A,
    const std::vector<float>& B, 
    int rows_A, int cols_A, int cols_B);

/**
 * Matrix transpose: B = A^T
 * 
 * @param A Input matrix [rows, cols] (flattened row-major)
 * @param rows Number of rows in A
 * @param cols Number of columns in A
 * @return Transposed matrix [cols, rows] (flattened row-major)
 */
std::vector<float> Transpose(
    const std::vector<float>& A,
    int rows, int cols);

/**
 * Add bias vector to each row of a matrix.
 * 
 * @param matrix Input matrix [rows, cols] (flattened row-major)
 * @param bias Bias vector [cols]
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Matrix with bias added [rows, cols] (flattened row-major)
 */
std::vector<float> AddBias(
    const std::vector<float>& matrix,
    const std::vector<float>& bias,
    int rows, int cols);

/**
 * Scale matrix by a constant factor.
 * 
 * @param matrix Input matrix (flattened)
 * @param scale Scaling factor
 * @return Scaled matrix
 */
std::vector<float> Scale(
    const std::vector<float>& matrix,
    float scale);

// TODO: Implement these utility functions in matrix_ops.cpp
// 
// Implementation tips:
// - Remember matrices are stored in row-major order: element[i][j] = data[i * cols + j]  
// - Always validate input dimensions before computation
// - Use clear variable names and add comments
// - Consider edge cases (empty matrices, dimension mismatches)

}  // namespace MatrixOps  
}  // namespace ptflash
