# Path To Flash (Attention)

Path To Flash is built as a hands on, end to end journey through implementing and optimizing attention mechanisms. 
The project will start out from first principles in naive C++ and the goal is to progress all the way to highly optimized Flash Attention CUDA kernels. 

This project is both a personal learning tool and a walk-along tutorial. It’s designed to help me (and hopefully others!) deeply understand how attention works at the system level, while also practicing clean, modern, production-quality C++.

Rather than hacking this together privately, I’m building this in public as a structured, readable, and extensible codebase — something anyone can clone, explore, and learn from.

### Goals
- Understand Attention: Implement it from first principles using basic CPU code
- Optimize Performance: Apply SIMD, memory layout tuning, blocking, and cache-friendly operations
- Scale with Parallelism: Use multithreading, work-stealing, and mixed precision
- GPU Acceleration: Port to CUDA, use shared memory and fuse operations
- Flash Attention Kernels: Implement streaming softmax and memory-efficient attention using block-wise processing

Hopefully by the end of this project you have a rough idea of how an optimized attention kernel is used in the real world!

## Prerequisites
This tutorial assumes you're starting from roughly the same place I was when I began this project. To get the most out of it, you'll need:

1. Proficiency in C++
    - This project is written entirely in C++ to help me challenge my systems programming skills. You should already be comfortable writing and reading C++ code.

2. Basic understanding of neural networks
    - You don’t need to be an expert in machine learning, but you should know what a neural network is and understand key concepts like linear layers, matrix multiplication, and activations.

3. (Optional) Familiarity with Bazel
    - I use Bazel as the build system for this project. You won’t need to edit any build files directly, just run the commands I provide.

        That said, Bazel is a powerful and widely used tool in ML infrastructure, and knowing how it works is a valuable skill. If you’ve never used it before, this is a great chance to learn it along the way.

## Quick Start

```bash
# Clone and setup
git clone <repo-url>

cd path-to-flash

```

**Note**: The project is structured in phases, each building on the previous:
- phase1/: Naive CPU implementation
- phase2/: Basic CPU optimizations
- phase3/: Parallel and SIMD enhancements
- phase4/: CUDA implementation
- phase5/: CUDA memory and performance tuning
- phase6/: Flash Attention kernels
- phase7/: Library integration, tuning, and deployment

## Author

Hi! I'm Abdur, a recent Software Engineering grad (Class of 2025) from the University of Waterloo. I’ve interned across the AI systems stack, working on compilers, training runtimes, model deployment, and low-level optimization. I love low-level software, something that challenges physical real-world limitations, and pushes the boundaries of what hardware is capable of.

As of writing this, I’m about to join the ML Training Runtime team at Tesla, working on their in-house training accelerator Dojo. This project is my way of preparing: building up deep understanding and strong implementation skills through real, performance-critical C++ and CUDA.

I'm developing this project primarily to:
- Learn by doing
- Push myself to write clean, testable, modular code
- Provide a tutorial others can follow, modify, and build on.

If you're getting into ML systems, compilers, or runtime engineering — my goal is to help you in your journey too.

## Contributing
This project is a personal challenge to grow as a software engineer. I'm using it to deepen my understanding of attention mechanisms, systems optimization, and modern C++ design.

If you see something that could be improved, have a better way of doing things, or just want to teach me something new, I'd geniunely be grateful! Your contributions, suggestions, and feedback will not only improve this project, but also help me (and hopefully others) become better engineers along the way :)

Feel free to open issues, submit pull requests, or just drop comments and ideas. I'm excited to learn from others and improve the project together.

## License
MIT

