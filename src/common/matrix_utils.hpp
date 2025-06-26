#pragma once

#include "absl/random/random.h"
#include "fmt/format.h"

#include <type_traits>
#include <vector>
namespace p2f {

// Simple concept for numeric types
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

class MatrixUtils {
public:
  template <Numeric T>
  static std::vector<T> random_matrix(size_t rows, size_t cols) {
    return random_matrix<T>(rows, cols, T(-1), T(1));
  }

  template <Numeric T>
  static std::vector<T> random_matrix(size_t rows, size_t cols, T min, T max) {
    std::vector<T> matrix(rows * cols);

    absl::BitGen bit_gen;

    for (auto &val : matrix) {
      val = absl::Uniform(bit_gen, min, max);
    }

    return matrix;
  }

  // Print matrix (first few elements)
  template <Numeric T>
  static void print_matrix(const std::vector<T> &matrix, int rows, int cols,
                           const std::string &name = "Matrix") {
    fmt::print("{}:\n", name);
    int print_rows = std::min(rows, 5);
    int print_cols = std::min(cols, 5);

    for (int i = 0; i < print_rows; ++i) {
      for (int j = 0; j < print_cols; ++j) {
        fmt::print("{:8.3f} ", matrix[i * cols + j]);
      }
      if (cols > 5)
        fmt::print("...");
      fmt::print("\n");
    }
    if (rows > 5)
      fmt::print("...\n");
    fmt::print("\n");
  }

  // Check if matrix are close
  template <Numeric T>
  static bool is_close(const std::vector<T> &a, const std::vector<T> &b,
                       T atol = 1e-6) {
    return std::equal(a.begin(), a.end(), b.begin(),
                      [atol](T x, T y) { return std::abs(x - y) < atol; });
  }
};

} // namespace p2f
