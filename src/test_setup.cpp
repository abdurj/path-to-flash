#include "common/benchmark_utils.hpp"
#include "common/matrix_utils.hpp"
#include "fmt/format.h"

using namespace ptflash;

int main(int argc, char **argv) {
  fmt::print("Testing Path to Flash Setup!\n");

  auto matrix = MatrixUtils::random_matrix<float>(10, 10);
  MatrixUtils::print_matrix(matrix, 10, 10);

  auto result = SimpleBenchmark::time_function(
      []() {
        auto matrix = MatrixUtils::random_matrix<float>(10, 10);
        volatile float sum = 0;
        for (size_t i = 0; i < matrix.size(); ++i) {
          sum += matrix[i];
        }
      },
      "Random Matrix", /*iterations=*/10, /*flops=*/100);

  result.print("Setup benchmark");

  fmt::print("✅ Setup working! Ready for attention implementation.\n");
  return 0;
}
