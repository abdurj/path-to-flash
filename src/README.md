# PathToFlash - Source Setup

Welcome to the PathToFlash codebase!

Before jumping into Phase 1, let’s make sure everything is working by running a quick test. This will confirm your development environment is properly set up.

## ⚙️ Prerequisites
Make sure you have:

A C++20-compatible compiler (e.g. GCC 10+, Clang 11+, MSVC 2019+)

Bazel installed (you can check with bazel --version)

## 🧪 Running the Setup Test
This project uses Bazel as its build system. If you're new to Bazel, don’t worry — you only need to know three basic commands:

```bash
bazel build <target>   # Build the target
bazel run <target>     # Build and run the target
bazel test <target>    # Build and run tests
```

A target is defined by a BUILD file. The top-level workspace folder is always referred to as //. Inside each BUILD file, targets are declared with names like cc_binary or cc_test.

In this folder, there is a BUILD file that defines a binary called `test_setup`. You can run it with:

```bash
bazel run //src:test_setup
```

If that runs successfully, awesome! You're all set. Go ahead and move on to Phase 1.
