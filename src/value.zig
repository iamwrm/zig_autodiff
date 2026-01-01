const std = @import("std");
const Op = @import("ops.zig").Op;

/// A node in the computational graph for automatic differentiation
pub const Value = struct {
    const Self = @This();

    /// Forward pass value
    data: f64,

    /// Gradient (accumulated during backward pass)
    grad: f64,

    /// Operation that created this node
    op: Op,

    /// Parent nodes (up to 2 for binary ops)
    parents: [2]?*Self,

    /// For power operation, store the exponent
    pow_exp: f64,

    /// Allocator reference for memory management
    allocator: std.mem.Allocator,

    /// Create a leaf Value (input or parameter)
    pub fn init(allocator: std.mem.Allocator, data: f64) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .data = data,
            .grad = 0.0,
            .op = .none,
            .parents = .{ null, null },
            .pow_exp = 0.0,
            .allocator = allocator,
        };
        return self;
    }

    /// Create a Value from a unary operation
    fn initUnary(allocator: std.mem.Allocator, data: f64, op: Op, parent: *Self) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .data = data,
            .grad = 0.0,
            .op = op,
            .parents = .{ parent, null },
            .pow_exp = 0.0,
            .allocator = allocator,
        };
        return self;
    }

    /// Create a Value from a binary operation
    fn initBinary(allocator: std.mem.Allocator, data: f64, op: Op, left: *Self, right: *Self) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .data = data,
            .grad = 0.0,
            .op = op,
            .parents = .{ left, right },
            .pow_exp = 0.0,
            .allocator = allocator,
        };
        return self;
    }

    // ============ BINARY OPERATIONS ============

    /// Addition: c = a + b
    /// dc/da = 1, dc/db = 1
    pub fn add(self: *Self, other: *Self) !*Self {
        return initBinary(self.allocator, self.data + other.data, .add, self, other);
    }

    /// Subtraction: c = a - b
    /// dc/da = 1, dc/db = -1
    pub fn sub(self: *Self, other: *Self) !*Self {
        return initBinary(self.allocator, self.data - other.data, .sub, self, other);
    }

    /// Multiplication: c = a * b
    /// dc/da = b, dc/db = a
    pub fn mul(self: *Self, other: *Self) !*Self {
        return initBinary(self.allocator, self.data * other.data, .mul, self, other);
    }

    /// Division: c = a / b
    /// dc/da = 1/b, dc/db = -a/b^2
    pub fn div(self: *Self, other: *Self) !*Self {
        return initBinary(self.allocator, self.data / other.data, .div, self, other);
    }

    // ============ UNARY OPERATIONS ============

    /// Power: c = a^n (n is a constant)
    /// dc/da = n * a^(n-1)
    pub fn pow(self: *Self, exponent: f64) !*Self {
        const result = try initUnary(self.allocator, std.math.pow(f64, self.data, exponent), .pow, self);
        result.pow_exp = exponent;
        return result;
    }

    /// Negation: c = -a
    /// dc/da = -1
    pub fn neg(self: *Self) !*Self {
        return initUnary(self.allocator, -self.data, .neg, self);
    }

    /// Exponential: c = e^a
    /// dc/da = e^a = c
    pub fn exp_(self: *Self) !*Self {
        return initUnary(self.allocator, @exp(self.data), .exp, self);
    }

    /// Natural log: c = ln(a)
    /// dc/da = 1/a
    pub fn log_(self: *Self) !*Self {
        return initUnary(self.allocator, @log(self.data), .log, self);
    }

    /// Hyperbolic tangent: c = tanh(a)
    /// dc/da = 1 - tanh(a)^2 = 1 - c^2
    pub fn tanh_(self: *Self) !*Self {
        const t = std.math.tanh(self.data);
        return initUnary(self.allocator, t, .tanh, self);
    }

    /// ReLU: c = max(0, a)
    /// dc/da = 1 if a > 0 else 0
    pub fn relu(self: *Self) !*Self {
        const out = if (self.data > 0) self.data else 0.0;
        return initUnary(self.allocator, out, .relu, self);
    }

    /// Zero the gradient (useful before a new backward pass)
    pub fn zeroGrad(self: *Self) void {
        self.grad = 0.0;
    }
};
