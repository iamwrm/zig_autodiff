const std = @import("std");
const Value = @import("value.zig").Value;
const Op = @import("ops.zig").Op;

/// Engine for computing gradients via reverse-mode autodiff
pub const Engine = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Engine {
        return .{ .allocator = allocator };
    }

    /// Perform backward pass starting from the output node
    /// Computes gradients for all nodes in the computational graph
    pub fn backward(self: *Engine, output: *Value) !void {
        // Step 1: Build topological order
        var topo: std.ArrayListUnmanaged(*Value) = .empty;
        defer topo.deinit(self.allocator);

        var visited: std.AutoHashMapUnmanaged(*Value, void) = .empty;
        defer visited.deinit(self.allocator);

        try buildTopo(self.allocator, output, &topo, &visited);

        // Step 2: Set output gradient to 1.0 (dL/dL = 1)
        output.grad = 1.0;

        // Step 3: Process nodes in reverse topological order
        var i: usize = topo.items.len;
        while (i > 0) {
            i -= 1;
            const node = topo.items[i];
            computeGradients(node);
        }
    }

    /// Recursively build topological ordering via DFS
    fn buildTopo(
        allocator: std.mem.Allocator,
        node: *Value,
        topo: *std.ArrayListUnmanaged(*Value),
        visited: *std.AutoHashMapUnmanaged(*Value, void),
    ) !void {
        if (visited.contains(node)) return;
        try visited.put(allocator, node, {});

        // Visit parents first (children in terms of data flow)
        for (node.parents) |maybe_parent| {
            if (maybe_parent) |parent| {
                try buildTopo(allocator, parent, topo, visited);
            }
        }

        // Add current node after all its dependencies
        try topo.append(allocator, node);
    }

    /// Compute gradients for parent nodes based on operation type
    fn computeGradients(node: *Value) void {
        const grad = node.grad;

        switch (node.op) {
            .none => {
                // Leaf node - no parents to propagate to
            },

            .add => {
                // c = a + b => da += dc, db += dc
                if (node.parents[0]) |a| a.grad += grad;
                if (node.parents[1]) |b| b.grad += grad;
            },

            .sub => {
                // c = a - b => da += dc, db += -dc
                if (node.parents[0]) |a| a.grad += grad;
                if (node.parents[1]) |b| b.grad -= grad;
            },

            .mul => {
                // c = a * b => da += b * dc, db += a * dc
                const a = node.parents[0].?;
                const b = node.parents[1].?;
                a.grad += b.data * grad;
                b.grad += a.data * grad;
            },

            .div => {
                // c = a / b => da += (1/b) * dc, db += (-a/b^2) * dc
                const a = node.parents[0].?;
                const b = node.parents[1].?;
                a.grad += (1.0 / b.data) * grad;
                b.grad += (-a.data / (b.data * b.data)) * grad;
            },

            .pow => {
                // c = a^n => da += n * a^(n-1) * dc
                const a = node.parents[0].?;
                const n = node.pow_exp;
                a.grad += n * std.math.pow(f64, a.data, n - 1.0) * grad;
            },

            .neg => {
                // c = -a => da += -dc
                if (node.parents[0]) |a| a.grad -= grad;
            },

            .exp => {
                // c = e^a => da += c * dc (since d(e^a)/da = e^a)
                if (node.parents[0]) |a| a.grad += node.data * grad;
            },

            .log => {
                // c = ln(a) => da += (1/a) * dc
                if (node.parents[0]) |a| a.grad += (1.0 / a.data) * grad;
            },

            .tanh => {
                // c = tanh(a) => da += (1 - c^2) * dc
                if (node.parents[0]) |a| a.grad += (1.0 - node.data * node.data) * grad;
            },

            .relu => {
                // c = relu(a) => da += (a > 0 ? 1 : 0) * dc
                if (node.parents[0]) |a| {
                    const gate: f64 = if (a.data > 0) 1.0 else 0.0;
                    a.grad += gate * grad;
                }
            },
        }
    }
};
