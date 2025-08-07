package(default_visibility = ["//visibility:public"])

cc_library(
    name = "attention",
    srcs = ["attention_base.cpp"],
    hdrs = ["attention_base.hpp"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_test(
    name = "attention_base_test",
    srcs = ["attention_base_test.cpp"],
    deps = [
        ":attention",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@abseil-cpp//absl/log:check",
    ],
)
