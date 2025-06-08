const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Leaky ReLU activation function node.
/// The Leaky ReLU function is used in neural networks to introduce non-linearity.
/// It allows a small, non-zero gradient when the input is negative, which helps to prevent dead neurons.
/// The Leaky ReLU function is defined as:
/// f(x) = x if x > 0 else α * x
/// - For positive inputs: f(x) = x
/// - For negative inputs: f(x) = α * x
/// where α is a small positive constant (default 0.01).
/// The Leaky ReLU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
pub const LeakyReLU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    alpha: f64 = 0.01, // small slope for negative inputs
    x: Node,

    /// Creates a new LeakyReLU node with the given input node and alpha value.
    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: f64) !*LeakyReLU {
        const self = try allocator.create(LeakyReLU);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .alpha = alpha,
            .x = x,
        };
        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *LeakyReLU) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the Leaky ReLU function.
    /// The Leaky ReLU function is defined as:
    /// f(x) = x if x > 0 else α * x
    /// where α is a small positive constant (default 0.01).
    /// The Leaky ReLU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
    pub fn eval(self: *LeakyReLU) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) xv else self.alpha * xv;
        }

        return self.value.?;
    }

    /// Compute the gradient of the Leaky ReLU function.
    /// The gradient of the Leaky ReLU function is defined as:
    /// ∂f/∂x = 1 if x > 0 else α
    /// where α is a small positive constant (default 0.01).
    /// The gradient of the Leaky ReLU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *LeakyReLU, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = if (xv > 0) dv else dv * self.alpha;
        }

        try self.x.diff(grad);
    }

    /// Returns this LeakyReLU node as a generic Node interface.
    pub fn node(self: *LeakyReLU) Node {
        return Node.init(self);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *LeakyReLU) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }
};

test "leaky_relu basic" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create leaky relu operation
    var leaky_relu_op = try LeakyReLU.init(allocator, x.node(), 0.01);
    defer leaky_relu_op.deinit();

    // First evaluate to cache the values
    const result = try leaky_relu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.01), // 0.01 * -1.0
        @as(f64, 0.0), // 0.0
        @as(f64, 1.0), // 1.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "leaky_relu gradient" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create leaky relu operation
    var leaky_relu_op = try LeakyReLU.init(allocator, x.node(), 0.01);
    defer leaky_relu_op.deinit();

    // First evaluate to cache the values
    const result = try leaky_relu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.01), // 0.01 * -1.0
        @as(f64, 0.0), // 0.0
        @as(f64, 1.0), // 1.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try leaky_relu_op.diff(gradTensor);

    // Expected gradients: 1 if x > 0 else alpha
    const expected_grad = [_]f64{
        @as(f64, 0.01), // alpha for x < 0
        @as(f64, 0.01), // alpha for x < 0
        @as(f64, 0.01), // alpha for x = 0
        @as(f64, 1.0), // 1 for x > 0
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "leaky_relu with different shapes" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create leaky relu operation
    var leaky_relu_op = try LeakyReLU.init(allocator, x.node(), 0.01);
    defer leaky_relu_op.deinit();

    // Evaluate
    const result = try leaky_relu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.01), // 0.01 * -1.0
        @as(f64, 0.0), // 0.0
        @as(f64, 1.0), // 1.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "leaky_relu reset" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create leaky relu operation
    var leaky_relu_op = try LeakyReLU.init(allocator, x.node(), 0.01);
    defer leaky_relu_op.deinit();

    // First evaluation
    const result1 = try leaky_relu_op.eval();
    const expected1 = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.01), // 0.01 * -1.0
        @as(f64, 0.0), // 0.0
        @as(f64, 1.0), // 1.0
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    leaky_relu_op.reset();
    const result2 = try leaky_relu_op.eval();
    const expected2 = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.01), // 0.01 * -1.0
        @as(f64, 0.0), // 0.0
        @as(f64, 1.0), // 1.0
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "leaky_relu custom alpha" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{2});
    defer xTensor.deinit();
    xTensor.data[0] = -1.0;
    xTensor.data[1] = 2.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create leaky relu operation with custom alpha
    var leaky_relu_op = try LeakyReLU.init(allocator, x.node(), 0.5);
    defer leaky_relu_op.deinit();

    const result = try leaky_relu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.5), // 0.5 * -1.0
        @as(f64, 2.0), // 2.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
