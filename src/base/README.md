# Base Attention Architecture

This directory contains the foundational `Attention` class that defines the common interface for all attention implementations throughout the project.

## Overview

The `Attention` class serves as an abstract base class that all attention phases will inherit from. This design provides:

1. **Consistent Interface**: All implementations (naive CPU, optimized CPU, CUDA, Flash Attention) use the same API
2. **Built-in Validation**: Input validation and parameter checking are handled at the base level using `absl::StatusOr`
3. **Benchmarking Support**: Every implementation can be benchmarked using the same interface
4. **Error Handling**: Modern C++20 design with `absl::StatusOr` instead of exceptions

## Key Components

### `Attention` Class

The abstract base class defines:

```cpp
// Factory method for creation with validation
static absl::StatusOr<std::unique_ptr<AttentionBase>> Create(
    int seq_len, int d_model, int num_heads);

// Core interface
virtual absl::StatusOr<std::vector<float>> Forward(const std::vector<float>& input) = 0;
virtual absl::StatusOr<double> Benchmark(int iterations = 100) = 0;
virtual std::string Name() const = 0;

// Configuration getters
int seq_len() const;
int d_model() const; 
int num_heads() const;
int head_dim() const;
```

### Configuration Parameters

- **`seq_len`**: Sequence length (number of tokens)
- **`d_model`**: Model dimension (embedding size)
- **`num_heads`**: Number of attention heads
- **`head_dim`**: Dimension per head (automatically calculated as `d_model / num_heads`)

### Input/Output Format

- **Input**: Flat `std::vector<float>` representing a `[seq_len, d_model]` matrix in row-major order
- **Output**: `absl::StatusOr<std::vector<float>>` containing the result or error status

## Error Handling

This project uses `absl::StatusOr` for error handling instead of exceptions. All operations that can fail return `absl::StatusOr<T>` where `T` is the expected result type. This provides:

- **Explicit error handling**: You must check if operations succeeded
- **Rich error context**: Status objects contain detailed error messages
- **No hidden control flow**: No surprise exceptions

## Usage Example

```cpp
// Create an attention implementation
auto attention_result = YourAttentionImpl::Create(512, 768, 12);
if (!attention_result.ok()) {
    // Handle creation error
    return;
}
auto attention = std::move(attention_result.value());

// Create input data
auto input = MatrixUtils::random_matrix<float>(512, 768);

// Run forward pass  
auto output_result = attention->Forward(input);
if (!output_result.ok()) {
    // Handle forward pass error
    return;
}
auto output = std::move(output_result.value());

// Benchmark performance
auto benchmark_result = attention->Benchmark(100);
if (benchmark_result.ok()) {
    double avg_time = benchmark_result.value();
}
```

## Testing

The base class includes comprehensive unit tests covering:

- Constructor parameter validation
- Input dimension validation
- Interface contracts
- Edge cases and error conditions

Run tests with:
```bash
bazel test //src/base:attention_base_test
```

## Implementation Guidelines

When creating new attention phases:

1. **Inherit from `Attention`**
2. **Use `CreateValidated<YourClass>()` in your static `Create` method**
3. **Implement all pure virtual methods**
4. **Use `ValidateInput()` in your `Forward()` method**
5. **Return `absl::StatusOr` for all operations that can fail**
6. **Provide a descriptive `Name()` for benchmarking**
7. **Add comprehensive unit tests**

Example skeleton:

```cpp
class Phase1Attention : public Attention {
public:
    static absl::StatusOr<std::unique_ptr<Phase1Attention>> Create(
        int seq_len, int d_model, int num_heads) {
        // All validation is handled in the base class!
        return CreateValidated<Phase1Attention>(seq_len, d_model, num_heads);
    }
    
    absl::StatusOr<std::vector<float>> Forward(const std::vector<float>& input) override {
        auto status = ValidateInput(input);
        if (!status.ok()) {
            return status;
        }
        
        // Your implementation here
        return result;
    }
    
    absl::StatusOr<double> Benchmark(int iterations) override {
        if (iterations <= 0) {
            return absl::InvalidArgumentError("iterations must be positive");
        }
        // Timing logic here
        return avg_time_ms;
    }
    
    std::string Name() const override { return "Phase1NaiveAttention"; }

private:
    friend class Attention;  // Allow base class to access private constructor
    Phase1Attention(int seq_len, int d_model, int num_heads)
        : Attention(seq_len, d_model, num_heads) {}
};
```

This architecture ensures that as we progress through the project phases, each implementation maintains compatibility while allowing for extensive performance optimization and comparison.
