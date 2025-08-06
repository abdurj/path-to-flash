package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "test_setup",
    srcs = ["test_setup.cpp"],
    deps = [
        "//src/common:benchmark_utils",
        "//src/common:matrix_utils",
        "@fmt",
    ],
)

# Example main demonstrating the base attention class
cc_binary(
    name = "example_main",
    srcs = ["example_main.cpp"],
    deps = [
        "//src/base:attention_base",
        "//src/common:matrix_utils",
        "@fmt",
    ],
)

# Test runner for all unit tests
cc_binary(
    name = "run_tests",
    srcs = [],
    deps = [
        "//src/base:attention_base_test",
    ],
)
