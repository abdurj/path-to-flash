#include "benchmark_utils.hpp"
#include <chrono>

namespace ptflash {

void BenchmarkResult::print(const std::string_view &name) const {
  fmt::print("{}: {:.2f} ms", name, time_ms);
  if (gflops > 0) {
    fmt::print(" ({:.2f} GFLOPS)", gflops);
  }
  fmt::print("\n");
}

BenchmarkResult
SimpleBenchmark::time_function(const std::function<void()> &func,
                               const std::string_view &name, int iterations,
                               size_t flops) {
  // Warmup
  func();

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    func();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double time_ms = duration.count() / 1000.0 / iterations;
  double gflops = flops > 0 ? (flops / (time_ms / 1000.0)) / 1e9 : 0.0;

  return {time_ms, gflops};
}

} // namespace ptflash
