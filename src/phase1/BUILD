package(default_visibility = ["//visibility:public"])

cc_library(
    name = "matrix_ops",
    srcs = ["matrix_ops.cpp"],
    hdrs = ["matrix_ops.hpp"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cpp"],
    hdrs = ["softmax.hpp"],
)

cc_library(
    name = "naive_attention",
    srcs = ["naive_attention.cpp"],
    hdrs = ["naive_attention.hpp"],
    deps = [
        ":matrix_ops",
        ":softmax",
        "//src/base:attention",
        "//src/common:matrix_utils",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_test(
    name = "naive_attention_test",
    srcs = ["naive_attention_test.cpp"],
    deps = [
        ":naive_attention",
        "//src/common:matrix_utils",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@abseil-cpp//absl/log:check",
    ],
)

cc_test(
    name = "unit_tests",
    srcs = ["unit_tests.cpp"],
    deps = [
        ":matrix_ops",
        ":softmax",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@abseil-cpp//absl/log:check",
    ],
)

cc_binary(
    name = "main",
    srcs = ["phase1_main.cpp"],
    deps = [
        ":naive_attention",
        "//src/common:macros",
        "//src/common:matrix_utils",
        "@fmt",
    ],
)
