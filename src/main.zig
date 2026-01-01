const std = @import("std");
const Value = @import("value.zig").Value;
const Engine = @import("engine.zig").Engine;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // ========================================
    // Example 1: Simple expression gradient
    // ========================================
    std.debug.print("\n=== Example 1: Simple Expression ===\n", .{});

    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_alloc = arena.allocator();

        // Compute: f = (x * y) + z^2
        // where x=2, y=3, z=4
        const x = try Value.init(arena_alloc, 2.0);
        const y = try Value.init(arena_alloc, 3.0);
        const z = try Value.init(arena_alloc, 4.0);

        // Build computational graph
        const xy = try x.mul(y); // 2 * 3 = 6
        const z2 = try z.pow(2.0); // 4^2 = 16
        const f = try xy.add(z2); // 6 + 16 = 22

        std.debug.print("f = (x * y) + z^2 = {d}\n", .{f.data});

        // Backward pass
        var engine = Engine.init(arena_alloc);
        try engine.backward(f);

        // Expected gradients:
        // df/dx = y = 3
        // df/dy = x = 2
        // df/dz = 2z = 8
        std.debug.print("df/dx = {d} (expected 3)\n", .{x.grad});
        std.debug.print("df/dy = {d} (expected 2)\n", .{y.grad});
        std.debug.print("df/dz = {d} (expected 8)\n", .{z.grad});
    }

    // ========================================
    // Example 2: Gradient Descent Optimization
    // ========================================
    std.debug.print("\n=== Example 2: Gradient Descent ===\n", .{});
    std.debug.print("Minimizing f(x) = (x - 3)^2\n", .{});

    {
        // Use arena for forward pass allocations
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        // Parameter to optimize
        var x_data: f64 = 0.0; // Starting value
        const learning_rate: f64 = 0.1;
        const target: f64 = 3.0;

        for (0..20) |step| {
            // Reset arena for each iteration
            _ = arena.reset(.retain_capacity);
            const arena_alloc = arena.allocator();

            // Build graph: f = (x - target)^2
            const x = try Value.init(arena_alloc, x_data);
            const t = try Value.init(arena_alloc, target);
            const diff = try x.sub(t);
            const loss = try diff.pow(2.0);

            // Backward pass
            var engine = Engine.init(arena_alloc);
            try engine.backward(loss);

            if (step % 5 == 0) {
                std.debug.print("Step {d}: x = {d:.4}, loss = {d:.4}, grad = {d:.4}\n", .{ step, x_data, loss.data, x.grad });
            }

            // Gradient descent update
            x_data -= learning_rate * x.grad;
        }

        std.debug.print("Final x = {d:.4} (target was 3.0)\n", .{x_data});
    }

    // ========================================
    // Example 3: Simple Neuron (OR gate)
    // ========================================
    std.debug.print("\n=== Example 3: Simple Neuron ===\n", .{});
    std.debug.print("Training neuron to compute OR function\n", .{});

    {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        // Neuron parameters: y = tanh(w1*x1 + w2*x2 + b)
        var w1: f64 = 0.1;
        var w2: f64 = 0.1;
        var b: f64 = -0.5;

        // Training data: OR function (learnable by single neuron)
        const inputs = [_][2]f64{
            .{ 0.0, 0.0 },
            .{ 0.0, 1.0 },
            .{ 1.0, 0.0 },
            .{ 1.0, 1.0 },
        };
        const targets = [_]f64{ -1.0, 1.0, 1.0, 1.0 }; // OR: only (0,0) is false

        const learning_rate: f64 = 0.5;

        for (0..100) |epoch| {
            var total_loss: f64 = 0.0;
            var grad_w1: f64 = 0.0;
            var grad_w2: f64 = 0.0;
            var grad_b: f64 = 0.0;

            for (inputs, targets) |inp, target| {
                _ = arena.reset(.retain_capacity);
                const arena_alloc = arena.allocator();

                // Build neuron graph
                const x1 = try Value.init(arena_alloc, inp[0]);
                const x2 = try Value.init(arena_alloc, inp[1]);
                const weight1 = try Value.init(arena_alloc, w1);
                const weight2 = try Value.init(arena_alloc, w2);
                const bias = try Value.init(arena_alloc, b);
                const tgt = try Value.init(arena_alloc, target);

                // Forward: y = tanh(w1*x1 + w2*x2 + b)
                const wx1 = try weight1.mul(x1);
                const wx2 = try weight2.mul(x2);
                const sum1 = try wx1.add(wx2);
                const sum2 = try sum1.add(bias);
                const y = try sum2.tanh_();

                // Loss: (y - target)^2
                const diff = try y.sub(tgt);
                const loss = try diff.pow(2.0);

                total_loss += loss.data;

                // Backward
                var engine = Engine.init(arena_alloc);
                try engine.backward(loss);

                // Accumulate gradients
                grad_w1 += weight1.grad;
                grad_w2 += weight2.grad;
                grad_b += bias.grad;
            }

            // Update parameters
            w1 -= learning_rate * grad_w1 / 4.0;
            w2 -= learning_rate * grad_w2 / 4.0;
            b -= learning_rate * grad_b / 4.0;

            if (epoch % 20 == 0) {
                std.debug.print("Epoch {d}: loss = {d:.4}\n", .{ epoch, total_loss / 4.0 });
            }
        }

        std.debug.print("\nFinal weights: w1={d:.3}, w2={d:.3}, b={d:.3}\n", .{ w1, w2, b });
        std.debug.print("Predictions:\n", .{});

        for (inputs, targets) |inp, target| {
            const pred = std.math.tanh(w1 * inp[0] + w2 * inp[1] + b);
            std.debug.print("  ({d}, {d}) -> {d:.3} (target: {d})\n", .{ inp[0], inp[1], pred, target });
        }
    }

    std.debug.print("\nDone!\n", .{});
}
