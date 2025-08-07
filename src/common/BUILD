package(default_visibility = ["//visibility:public"])

cc_library(
    name = "benchmark_utils",
    srcs = ["benchmark_utils.cpp"],
    hdrs = ["benchmark_utils.hpp"],
    deps = [
        "@abseil-cpp//absl/time",
        "@fmt",
        "@google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "matrix_utils",
    hdrs = ["matrix_utils.hpp"],
    deps = [
        ":benchmark_utils",
        "@abseil-cpp//absl/random",
        "@fmt",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.hpp"],
)
