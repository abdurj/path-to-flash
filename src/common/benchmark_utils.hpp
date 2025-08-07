#pragma once

#include "absl/time/time.h"
#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <functional>
#include <string_view>

namespace ptflash {

/// @brief Result of a benchmark
struct BenchmarkResult {
  /// @brief Time in milliseconds
  double time_ms;
  /// @brief Floating point operations per second
  double gflops;

  /// @brief Print the result
  /// @param name The name of the benchmark
  void print(const std::string_view &name) const;
};

class SimpleBenchmark {
public:
  /// @brief Time a function and return the result
  /// @param func The function to time
  /// @param name The name of the function
  /// @param iterations The number of iterations to run
  /// @param flops The number of floating point operations to perform
  /// @return The result of the benchmark
  static BenchmarkResult time_function(const std::function<void()> &func,
                                       const std::string_view &name,
                                       int iterations = 100, size_t flops = 0);

  /// @brief Register a Google Benchmark function
  /// @tparam F The type of the function
  /// @param func The function to register
  /// @param name The name of the function
  template <typename F>
  static void register_benchmark(F func, const std::string_view &name) {
    benchmark::RegisterBenchmark(name, func);
  }
};

} // namespace ptflash
