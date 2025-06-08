const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Swish function node.
/// The Swish function is a smooth, non-monotonic activation function.
/// The Swish function is often used in deep learning models as an activation function.
/// It has been shown to perform better than ReLU in some cases, especially in deeper networks.
/// The Swish function is differentiable everywhere, making it suitable for backpropagation in neural networks.
/// It is defined as:
/// f(x) = x * σ(x) = x / (1 + exp(-x))
/// where σ is the sigmoid function.
pub const Swish = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    const sqrt_2_over_pi: f32 = 0.79788456; // sqrt(2 / π)
    const coeff: f32 = 0.044715;

    /// Creates a new Swish node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Swish {
        const self = try allocator.create(Swish);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
        };
        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Swish) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the Swish function.
    /// The Swish function is defined as:
    /// f(x) = x * σ(x) = x / (1 + exp(-x))
    /// where σ is the sigmoid function.
    /// The Swish function is a smooth, non-monotonic activation function.
    /// The Swish function is often used in deep learning models as an activation function.
    /// It has been shown to perform better than ReLU in some cases, especially in deeper networks.
    pub fn eval(self: *Swish) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = xv / (1 + std.math.exp(-xv));
        }

        return self.value.?;
    }

    /// Compute the gradient of the Swish function.
    /// The gradient of the Swish function is defined as:
    /// ∂f/∂x = σ(x) + x * σ(x) * (1 - σ(x))
    /// where σ is the sigmoid function.
    /// The gradient of the Swish function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Swish, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            const sig = 1 / (1 + std.math.exp(-xv));
            v.* = dv * (sig + xv * sig * (1 - sig));
        }

        try self.x.diff(grad);
    }

    /// Returns this Swish node as a generic Node interface.
    pub fn node(self: *Swish) Node {
        return Node.init(self);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Swish) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }
};

test "swish basic" {
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

    // Create swish operation
    var swish_op = try Swish.init(allocator, x.node());
    defer swish_op.deinit();

    // First evaluate to cache the values
    const result = try swish_op.eval();
    const expected = [_]f64{
        @as(f64, -0.23840584404423514), // swish(-2.0)
        @as(f64, -0.2689414213699951), // swish(-1.0)
        @as(f64, 0.0), // swish(0.0)
        @as(f64, 0.7310585786300049), // swish(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "swish gradient" {
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

    // Create swish operation
    var swish_op = try Swish.init(allocator, x.node());
    defer swish_op.deinit();

    // First evaluate to cache the values
    const result = try swish_op.eval();
    const expected = [_]f64{
        @as(f64, -0.23840584404423514), // swish(-2.0)
        @as(f64, -0.2689414213699951), // swish(-1.0)
        @as(f64, 0.0), // swish(0.0)
        @as(f64, 0.7310585786300049), // swish(1.0)
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
    try swish_op.diff(gradTensor);

    // Expected gradients: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    const expected_grad = [_]f64{
        @as(f64, -0.09078424878489569), // sigmoid(-2.0) + (-2.0) * sigmoid(-2.0) * (1 - sigmoid(-2.0))
        @as(f64, 0.07232948812851325), // sigmoid(-1.0) + (-1.0) * sigmoid(-1.0) * (1 - sigmoid(-1.0))
        @as(f64, 0.5), // sigmoid(0.0) + 0.0 * sigmoid(0.0) * (1 - sigmoid(0.0))
        @as(f64, 0.9276705118714868), // sigmoid(1.0) + 1.0 * sigmoid(1.0) * (1 - sigmoid(1.0))
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "swish with different shapes" {
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

    // Create swish operation
    var swish_op = try Swish.init(allocator, x.node());
    defer swish_op.deinit();

    // Evaluate
    const result = try swish_op.eval();
    const expected = [_]f64{
        @as(f64, -0.23840584404423514), // swish(-2.0)
        @as(f64, -0.2689414213699951), // swish(-1.0)
        @as(f64, 0.0), // swish(0.0)
        @as(f64, 0.7310585786300049), // swish(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "swish reset" {
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

    // Create swish operation
    var swish_op = try Swish.init(allocator, x.node());
    defer swish_op.deinit();

    // First evaluation
    const result1 = try swish_op.eval();
    const expected1 = [_]f64{
        @as(f64, -0.23840584404423514), // swish(-2.0)
        @as(f64, -0.2689414213699951), // swish(-1.0)
        @as(f64, 0.0), // swish(0.0)
        @as(f64, 0.7310585786300049), // swish(1.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset
    swish_op.reset();

    // Second evaluation
    const result2 = try swish_op.eval();
    const expected2 = [_]f64{
        @as(f64, -0.23840584404423514), // swish(-2.0)
        @as(f64, -0.2689414213699951), // swish(-1.0)
        @as(f64, 0.0), // swish(0.0)
        @as(f64, 0.7310585786300049), // swish(1.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
