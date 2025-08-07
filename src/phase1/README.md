# Phase 1: Naive CPU Attention Implementation

Welcome to Phase 1 of Path to Flash! In this phase, you'll implement a basic, unoptimized attention mechanism from scratch. This serves as our baseline implementation that we'll optimize in later phases.

## Learning Objectives

By the end of this phase, you will:
- Understand the core attention computation: `Attention(Q, K, V) = softmax(QK^T / √d_k) * V`
- Implement multi-head attention from first principles
- Build a working CPU-based attention layer
- Establish performance baselines for future optimization

## What You'll Implement

### Core Components
1. **Matrix Operations**: Basic matrix multiplication and transpose
2. **Softmax Function**: Numerically stable softmax computation  
3. **Attention Computation**: The core attention formula
4. **Multi-Head Logic**: Splitting into heads and recombining
5. **Linear Projections**: Q, K, V, and output projections

## Step-by-Step Implementation Guide

### Step 1: Set Up the Class Structure

First, examine `naive_attention.hpp` - this is your Phase 1 attention implementation that inherits from the base `Attention` class.

**Your Task**: Complete the private member variables section.

**Hint**: You'll need weight matrices for:
- Query projection (Q)
- Key projection (K)  
- Value projection (V)
- Output projection (O)

Consider the dimensions: `[d_model, d_model]` for each projection.

### Step 2: Implement Matrix Utilities

Look at `matrix_ops.hpp` - these are helper functions for basic linear algebra.

**Your Task**: Implement the following functions:
- `MatMul()`: Matrix multiplication with proper dimension checking
- `Transpose()`: Matrix transpose operation
- `AddBias()`: Add bias vectors to matrices

**Mathematical Notes**:
- Matrix multiplication: `C[i][j] = Σ(A[i][k] * B[k][j])`
- Transpose: `B[j][i] = A[i][j]`

### Step 3: Implement Softmax

In `softmax.cpp`, implement numerically stable softmax.

**Your Task**: Complete the `Softmax()` function.

**Key Points**:
- Use the "subtract max" trick: `softmax(x) = softmax(x - max(x))`
- This prevents overflow while maintaining mathematical correctness
- Apply softmax row-wise to the attention scores

**Formula**:
```
softmax(x_i) = exp(x_i - max(x)) / Σ(exp(x_j - max(x)))
```

### Step 4: Implement Core Attention

Now for the main event! In `naive_attention.cpp`, implement the `Forward()` method.

**Your Task**: Follow this algorithm:

1. **Linear Projections**:
   ```cpp
   Q = input * W_q  // Shape: [seq_len, d_model]
   K = input * W_k  
   V = input * W_v
   ```

2. **Reshape for Multi-Head**:
   ```cpp
   // Reshape to [seq_len, num_heads, head_dim]
   // Then view as [num_heads, seq_len, head_dim] for easier processing
   ```

3. **Attention Computation** (for each head):
   ```cpp
   scores = Q * K^T / sqrt(head_dim)  // Shape: [seq_len, seq_len]
   attn_weights = softmax(scores)      // Row-wise softmax
   head_output = attn_weights * V     // Shape: [seq_len, head_dim]
   ```

4. **Concatenate Heads**:
   ```cpp
   // Reshape back to [seq_len, d_model]
   concat_output = concatenate_all_heads(head_outputs)
   ```

5. **Output Projection**:
   ```cpp
   output = concat_output * W_o
   ```

### Step 5: Weight Initialization

In the constructor, initialize all weight matrices.

**Your Task**: Use simple random initialization:
- Mean = 0, Standard deviation = 0.1
- Use the existing `MatrixUtils::random_matrix()` helper

**Professional Tip**: In real implementations, you'd use Xavier/Kaiming initialization, but simple random works for learning.

### Step 6: Implement Benchmarking

Complete the `Benchmark()` method.

**Your Task**: 
1. Create sample input data
2. Time multiple forward passes
3. Return average time per iteration
4. Use `std::chrono` for timing

## Testing Your Implementation

### Unit Tests
Run the provided tests to verify correctness:
```bash
bazel test //src/phase1:naive_attention_test
```

### Integration Test
Test with a simple example:
```bash
bazel run //src/phase1:phase1_main
```

### Expected Behavior
- Small input (4x8): Should complete in microseconds
- Typical input (512x768): Should complete in milliseconds
- Output should have same shape as input
- Attention weights should sum to 1.0 (check this in tests!)

## Debugging Tips

### Common Issues
1. **Dimension Mismatches**: Print matrix shapes at each step
2. **NaN Values**: Check for division by zero in softmax
3. **Wrong Results**: Verify attention weights sum to 1.0

### Debugging Helpers
Use the provided debugging utilities:
```cpp
MatrixUtils::print_matrix_shape(matrix, "Debug: Q matrix");
MatrixUtils::check_for_nan(matrix, "Attention weights");
```

## Performance Expectations

### Baseline Targets (Debug Build)
- **Small** (seq=64, d=256): ~0.1ms
- **Medium** (seq=256, d=512): ~2ms  
- **Large** (seq=512, d=768): ~10ms

Don't worry if you're slower - we'll optimize in later phases!

## Mathematical Deep Dive

### Why This Formula Works

The attention mechanism computes:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Intuition**:
1. `QK^T`: Measures similarity between queries and keys
2. `/√d_k`: Scales to prevent softmax saturation  
3. `softmax()`: Converts similarities to probabilities
4. `* V`: Weighted combination of values

### Multi-Head Benefits
- **Parallel Processing**: Each head can focus on different aspects
- **Representation Learning**: Different heads learn different patterns
- **Computational Efficiency**: Smaller matrices per head

## Next Steps

Once you complete Phase 1:
1. **Verify Correctness**: All tests should pass
2. **Measure Performance**: Record baseline timings
3. **Understand Bottlenecks**: Profile to see where time is spent
4. **Move to Phase 2**: CPU optimizations (SIMD, cache-friendly layouts)

## Common Questions

**Q: Why is this so slow?**
A: This is intentionally naive! We're prioritizing clarity over speed. Optimizations come later.

**Q: Can I use BLAS libraries?**
A: Not in Phase 1 - implement matrix operations from scratch to understand the computation.

**Q: My outputs don't match the reference?**
A: Check numerical precision, initialization seeds, and dimension handling.

## Files Overview

- `naive_attention.hpp/cpp`: Your main implementation
- `matrix_ops.hpp/cpp`: Linear algebra helpers  
- `softmax.hpp/cpp`: Numerically stable softmax
- `naive_attention_test.cpp`: Unit tests
- `phase1_main.cpp`: Integration test and demo
- `BUILD`: Bazel build configuration

Happy coding! 🚀

Remember: The goal is **understanding**, not speed. Make sure you understand each step before moving on.
