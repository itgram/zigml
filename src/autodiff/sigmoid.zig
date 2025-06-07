const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Variable = @import("variable.zig").Variable;

/// Sigmoid function node.
/// The Sigmoid function is defined as:
/// f(x) = σ(x) = 1 / (1 + exp(-x))
/// where σ is the sigmoid function.
/// The Sigmoid function maps any real-valued number to the (0, 1) interval.
/// The Sigmoid function is commonly used in neural networks as an activation function.
/// It is particularly useful for binary classification tasks.
/// The Sigmoid function is differentiable everywhere, making it suitable for backpropagation in neural networks.
pub const Sigmoid = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new sigmoid node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Sigmoid {
        const self = try allocator.create(Sigmoid);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
        };
        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Sigmoid) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the Sigmoid function.
    /// The Sigmoid function is defined as:
    /// f(x) = 1 / (1 + exp(-x))
    /// where x is the input tensor.
    /// The Sigmoid function maps any real-valued number to the (0, 1) interval.
    /// The Sigmoid function is commonly used in neural networks as an activation function.
    /// It is particularly useful for binary classification tasks.
    pub fn eval(self: *Sigmoid) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = 1.0 / (1.0 + math.exp(-xv));
        }

        return self.value.?;
    }

    /// Compute the gradient of the Sigmoid function.
    /// The gradient of the Sigmoid function is defined as:
    /// ∂f/∂x = σ(x) * (1 - σ(x))
    /// where x is the input tensor.
    /// The gradient of the Sigmoid function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Sigmoid, dval: *Tensor) !void {
        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, self.value.?.data, dval.data) |*v, vv, dv| {
            v.* = dv * vv * (1 - vv); // Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Sigmoid) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this sigmoid node as a generic Node interface.
    pub fn node(self: *Sigmoid) Node {
        return Node.init(self);
    }
};

test "sigmoid basic" {
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

    // Create sigmoid operation
    var sigmoid_op = try Sigmoid.init(allocator, x.node());
    defer sigmoid_op.deinit();

    // First evaluate to cache the values
    const result = try sigmoid_op.eval();
    const expected = [_]f64{
        @as(f64, 0.11920292202211755), // sigmoid(-2.0)
        @as(f64, 0.2689414213699951), // sigmoid(-1.0)
        @as(f64, 0.5), // sigmoid(0.0)
        @as(f64, 0.7310585786300049), // sigmoid(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "sigmoid gradient" {
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

    // Create sigmoid operation
    var sigmoid_op = try Sigmoid.init(allocator, x.node());
    defer sigmoid_op.deinit();

    // First evaluate to cache the values
    const result = try sigmoid_op.eval();
    const expected = [_]f64{
        @as(f64, 0.11920292202211755), // sigmoid(-2.0)
        @as(f64, 0.2689414213699951), // sigmoid(-1.0)
        @as(f64, 0.5), // sigmoid(0.0)
        @as(f64, 0.7310585786300049), // sigmoid(1.0)
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
    try sigmoid_op.diff(gradTensor);

    // Expected gradients: sigmoid(x) * (1 - sigmoid(x))
    const expected_grad = [_]f64{
        @as(f64, 0.10499358540350662), // sigmoid(-2.0) * (1 - sigmoid(-2.0))
        @as(f64, 0.19661193324148185), // sigmoid(-1.0) * (1 - sigmoid(-1.0))
        @as(f64, 0.25), // sigmoid(0.0) * (1 - sigmoid(0.0))
        @as(f64, 0.19661193324148185), // sigmoid(1.0) * (1 - sigmoid(1.0))
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "sigmoid with different shapes" {
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

    // Create sigmoid operation
    var sigmoid_op = try Sigmoid.init(allocator, x.node());
    defer sigmoid_op.deinit();

    // Evaluate
    const result = try sigmoid_op.eval();
    const expected = [_]f64{
        @as(f64, 0.11920292202211755), // sigmoid(-2.0)
        @as(f64, 0.2689414213699951), // sigmoid(-1.0)
        @as(f64, 0.5), // sigmoid(0.0)
        @as(f64, 0.7310585786300049), // sigmoid(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "sigmoid reset" {
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

    // Create sigmoid operation
    var sigmoid_op = try Sigmoid.init(allocator, x.node());
    defer sigmoid_op.deinit();

    // First evaluation
    const result1 = try sigmoid_op.eval();
    const expected1 = [_]f64{
        @as(f64, 0.11920292202211755), // sigmoid(-2.0)
        @as(f64, 0.2689414213699951), // sigmoid(-1.0)
        @as(f64, 0.5), // sigmoid(0.0)
        @as(f64, 0.7310585786300049), // sigmoid(1.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset
    sigmoid_op.reset();

    // Second evaluation
    const result2 = try sigmoid_op.eval();
    const expected2 = [_]f64{
        @as(f64, 0.11920292202211755), // sigmoid(-2.0)
        @as(f64, 0.2689414213699951), // sigmoid(-1.0)
        @as(f64, 0.5), // sigmoid(0.0)
        @as(f64, 0.7310585786300049), // sigmoid(1.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
