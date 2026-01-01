const std = @import("std");
const Value = @import("value.zig").Value;
const Engine = @import("engine.zig").Engine;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Autodiff Benchmark ===\n\n", .{});

    // Print CPU info
    printCpuInfo(allocator);

    // Benchmark 1: Simple expression (forward + backward)
    try benchSimpleExpr(allocator);

    // Benchmark 2: Deep chain (stress test backward pass)
    try benchDeepChain(allocator);

    // Benchmark 3: Wide graph (many parallel operations)
    try benchWideGraph(allocator);

    // Benchmark 4: Gradient descent iterations
    try benchGradientDescent(allocator);

    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}

fn benchSimpleExpr(allocator: std.mem.Allocator) !void {
    const iterations: usize = 100_000;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();

        const x = try Value.init(a, 2.0);
        const y = try Value.init(a, 3.0);
        const z = try Value.init(a, 4.0);

        const xy = try x.mul(y);
        const z2 = try z.pow(2.0);
        const f = try xy.add(z2);

        var engine = Engine.init(a);
        try engine.backward(f);
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);

    std.debug.print("Simple expr (f = x*y + z^2):\n", .{});
    std.debug.print("  {d} iterations in {d:.2} ms\n", .{ iterations, elapsed_ms });
    std.debug.print("  {d:.0} ops/sec\n\n", .{ops_per_sec});
}

fn benchDeepChain(allocator: std.mem.Allocator) !void {
    const depth: usize = 100;
    const iterations: usize = 10_000;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();

        // Build a deep chain: ((((x * 1.01) * 1.01) * 1.01) ... )
        var current = try Value.init(a, 1.0);
        const factor = try Value.init(a, 1.01);

        for (0..depth) |_| {
            current = try current.mul(factor);
        }

        // Apply tanh at the end
        const output = try current.tanh_();

        var engine = Engine.init(a);
        try engine.backward(output);
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);

    std.debug.print("Deep chain (depth={d}):\n", .{depth});
    std.debug.print("  {d} iterations in {d:.2} ms\n", .{ iterations, elapsed_ms });
    std.debug.print("  {d:.0} ops/sec\n\n", .{ops_per_sec});
}

fn benchWideGraph(allocator: std.mem.Allocator) !void {
    const width: usize = 100;
    const iterations: usize = 10_000;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        _ = arena.reset(.retain_capacity);
        const a = arena.allocator();

        // Create many inputs and sum them: x1 + x2 + x3 + ... + xN
        var sum = try Value.init(a, 0.0);

        for (0..width) |i| {
            const x = try Value.init(a, @as(f64, @floatFromInt(i)) * 0.01);
            const x_squared = try x.pow(2.0);
            sum = try sum.add(x_squared);
        }

        var engine = Engine.init(a);
        try engine.backward(sum);
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);

    std.debug.print("Wide graph (width={d}):\n", .{width});
    std.debug.print("  {d} iterations in {d:.2} ms\n", .{ iterations, elapsed_ms });
    std.debug.print("  {d:.0} ops/sec\n\n", .{ops_per_sec});
}

fn benchGradientDescent(allocator: std.mem.Allocator) !void {
    const steps: usize = 1000;
    const iterations: usize = 1000;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        var x_data: f64 = 0.0;
        const target: f64 = 3.0;
        const lr: f64 = 0.1;

        for (0..steps) |_| {
            _ = arena.reset(.retain_capacity);
            const a = arena.allocator();

            const x = try Value.init(a, x_data);
            const t = try Value.init(a, target);
            const diff = try x.sub(t);
            const loss = try diff.pow(2.0);

            var engine = Engine.init(a);
            try engine.backward(loss);

            x_data -= lr * x.grad;
        }
    }

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const total_steps = iterations * steps;
    const steps_per_sec = @as(f64, @floatFromInt(total_steps)) / (elapsed_ms / 1000.0);

    std.debug.print("Gradient descent ({d} steps x {d} runs):\n", .{ steps, iterations });
    std.debug.print("  {d} total steps in {d:.2} ms\n", .{ total_steps, elapsed_ms });
    std.debug.print("  {d:.0} steps/sec\n\n", .{steps_per_sec});
}

fn printCpuInfo(allocator: std.mem.Allocator) void {
    // Try to read CPU model from /proc/cpuinfo (Linux)
    const file = std.fs.openFileAbsolute("/proc/cpuinfo", .{}) catch {
        std.debug.print("CPU: unknown\n\n", .{});
        return;
    };
    defer file.close();

    const content = file.readToEndAlloc(allocator, 1024 * 1024) catch {
        std.debug.print("CPU: unknown\n\n", .{});
        return;
    };
    defer allocator.free(content);

    // Find "model name" line (x86) or "CPU architecture" (ARM)
    var lines = std.mem.splitScalar(u8, content, '\n');
    var cpu_arch: ?[]const u8 = null;
    var cpu_part: ?[]const u8 = null;

    while (lines.next()) |line| {
        // x86: model name
        if (std.mem.startsWith(u8, line, "model name")) {
            if (std.mem.indexOf(u8, line, ":")) |idx| {
                const model = std.mem.trim(u8, line[idx + 1 ..], " \t");
                std.debug.print("CPU: {s}\n\n", .{model});
                return;
            }
        }
        // ARM: CPU architecture
        if (std.mem.startsWith(u8, line, "CPU architecture")) {
            if (std.mem.indexOf(u8, line, ":")) |idx| {
                cpu_arch = std.mem.trim(u8, line[idx + 1 ..], " \t");
            }
        }
        // ARM: CPU part
        if (std.mem.startsWith(u8, line, "CPU part")) {
            if (std.mem.indexOf(u8, line, ":")) |idx| {
                cpu_part = std.mem.trim(u8, line[idx + 1 ..], " \t");
            }
        }
    }

    // ARM fallback
    if (cpu_arch) |arch| {
        if (cpu_part) |part| {
            std.debug.print("CPU: ARMv{s} (part {s})\n\n", .{ arch, part });
        } else {
            std.debug.print("CPU: ARMv{s}\n\n", .{arch});
        }
        return;
    }

    std.debug.print("CPU: unknown\n\n", .{});
}
