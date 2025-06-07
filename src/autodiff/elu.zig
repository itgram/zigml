const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Variable = @import("variable.zig").Variable;

/// Exponential Linear Unit (ELU) activation function node.
/// The ELU function is used in neural networks to introduce non-linearity.
/// It is similar to ReLU but allows for negative values, which can help with learning and convergence.
/// The ELU function is defined as:
/// f(x) = x if x > 0 else α * (exp(x) - 1)
/// - For positive inputs: f(x) = x
/// - For negative inputs: f(x) = α * (exp(x) - 1)
/// where α is a small positive constant (default 0.01).
/// The ELU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
pub const ELU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    alpha: f64 = 0.01, // small slope for negative inputs

    /// Creates a new ELU node with the given input node and alpha value.
    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: f64) !*ELU {
        const self = try allocator.create(ELU);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
            .alpha = alpha,
        };

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *ELU) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the ELU function.
    /// The ELU function is defined as:
    /// f(x) = x if x > 0 else α * (exp(x) - 1)
    /// where α is a small positive constant (default 0.01).
    /// The ELU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
    pub fn eval(self: *ELU) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) xv else self.alpha * (math.exp(xv) - 1.0);
        }

        return self.value.?;
    }

    /// Compute the gradient of the ELU function.
    /// The gradient of the ELU function is defined as:
    /// ∂f/∂x = 1 if x > 0 else α * exp(x)
    /// where α is a small positive constant (default 0.01).
    /// The gradient of the ELU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *ELU, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, self.value.?.data, dval.data) |*v, xv, vv, dv| {
            v.* = if (xv > 0) dv else dv * (vv + self.alpha);
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *ELU) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this ELU node as a generic Node interface.
    pub fn node(self: *ELU) Node {
        return Node.init(self);
    }
};

test "elu basic" {
    const allocator = std.testing.allocator;

    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;
    xTensor.data[1] = 0.0;
    xTensor.data[2] = -1.0;
    xTensor.data[3] = -2.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    var elu_op = try ELU.init(allocator, x.node(), 0.01);
    defer elu_op.deinit();

    const result = try elu_op.eval();
    const expected = [_]f64{
        2.0, // x > 0
        0.0, // x == 0
        0.01 * (std.math.exp(-1.0) - 1.0), // x < 0
        0.01 * (std.math.exp(-2.0) - 1.0), // x < 0
    };
    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "elu gradient" {
    const allocator = std.testing.allocator;

    const xTensor = try Tensor.init(allocator, &[_]usize{3});
    defer xTensor.deinit();
    xTensor.data[0] = 1.0;
    xTensor.data[1] = 0.0;
    xTensor.data[2] = -1.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    var elu_op = try ELU.init(allocator, x.node(), 0.01);
    defer elu_op.deinit();

    const result = try elu_op.eval();
    const expected = [_]f64{
        1.0,
        0.0,
        0.01 * (std.math.exp(-1.0) - 1.0),
    };
    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Gradient wrt output
    const gradTensor = try Tensor.init(allocator, &[_]usize{3});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;

    try elu_op.diff(gradTensor);

    // For x > 0: grad = 1
    // For x == 0: grad = 1
    // For x < 0: grad = dv * (vv + alpha)
    const expected_grad = [_]f64{
        1.0, // x > 0
        0.01, // x == 0 (vv + alpha = alpha * (exp(0) - 1) + alpha = 0 + 0.01 = 0.01)
        gradTensor.data[2] * (result.data[2] + 0.01), // x < 0
    };
    try std.testing.expectApproxEqAbs(x.grad.data[0], expected_grad[0], 1e-6);
    try std.testing.expectApproxEqAbs(x.grad.data[1], expected_grad[1], 1e-6);
    try std.testing.expectApproxEqAbs(x.grad.data[2], expected_grad[2], 1e-6);
}

test "elu custom alpha" {
    const allocator = std.testing.allocator;

    const xTensor = try Tensor.init(allocator, &[_]usize{2});
    defer xTensor.deinit();
    xTensor.data[0] = -1.0;
    xTensor.data[1] = 2.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    var elu_op = try ELU.init(allocator, x.node(), 0.5);
    defer elu_op.deinit();

    const result = try elu_op.eval();
    const expected = [_]f64{
        0.5 * (std.math.exp(-1.0) - 1.0),
        2.0,
    };
    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "elu reset" {
    const allocator = std.testing.allocator;

    const xTensor = try Tensor.init(allocator, &[_]usize{2});
    defer xTensor.deinit();
    xTensor.data[0] = 1.0;
    xTensor.data[1] = -1.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    var elu_op = try ELU.init(allocator, x.node(), 0.01);
    defer elu_op.deinit();

    const result1 = try elu_op.eval();
    const expected1 = [_]f64{ 1.0, 0.01 * (std.math.exp(-1.0) - 1.0) };
    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
    elu_op.reset();
    const result2 = try elu_op.eval();
    const expected2 = [_]f64{ 1.0, 0.01 * (std.math.exp(-1.0) - 1.0) };
    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
