# Zig Autodiff

A reverse-mode automatic differentiation DAG system written in Zig.

## Features

- **Operations**: add, sub, mul, div, pow, neg, exp, log, tanh, relu
- **Reverse-mode autodiff**: Topological sort + chain rule gradient propagation
- **Memory efficient**: Uses arena allocators for batch deallocation

## Requirements

- [Pixi](https://pixi.sh) (manages Zig toolchain)

## Quick Start

```bash
./run.sh
```

Or manually:

```bash
pixi run zig build run
```

## Examples

The project includes three demonstrations:

1. **Gradient computation**: `f = (x*y) + z²` with verified gradients
2. **Gradient descent**: Minimizing `f(x) = (x-3)²`
3. **Neuron training**: Learning OR function

## Benchmarks

Run benchmarks (compiled with ReleaseFast):

```bash
pixi run zig build bench
```

| Benchmark | Performance |
|-----------|-------------|
| Simple expr (f = x*y + z²) | ~17M ops/sec |
| Deep chain (depth=100) | ~316K ops/sec |
| Wide graph (width=100) | ~193K ops/sec |
| Gradient descent | ~21M steps/sec |

## Project Structure

```
├── build.zig          # Build configuration
├── pixi.toml          # Pixi config with Zig dependency
├── run.sh             # Build and run script
└── src/
    ├── ops.zig        # Operation enum
    ├── value.zig      # Core Value struct with forward operations
    ├── engine.zig     # Backward pass with topological sort
    ├── main.zig       # Example applications
    └── bench.zig      # Performance benchmarks
```

## Example Output

```
=== Example 1: Simple Expression ===
f = (x * y) + z^2 = 22
df/dx = 3 (expected 3)
df/dy = 2 (expected 2)
df/dz = 8 (expected 8)

=== Example 2: Gradient Descent ===
Minimizing f(x) = (x - 3)^2
Step 0: x = 0.0000, loss = 9.0000, grad = -6.0000
Final x = 2.9654 (target was 3.0)

=== Example 3: Simple Neuron ===
Training neuron to compute OR function
Epoch 0: loss = 1.4413
Epoch 80: loss = 0.0183
Predictions:
  (0, 0) -> -0.820 (target: -1)
  (0, 1) -> 0.889 (target: 1)
  (1, 0) -> 0.889 (target: 1)
  (1, 1) -> 0.999 (target: 1)
```
