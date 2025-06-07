const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Variable = @import("variable.zig").Variable;

/// ReLU function node.
/// The ReLU (Rectified Linear Unit) activation function
/// It is commonly used in neural networks as an activation function.
/// It is defined as:
/// f(x) = x if x > 0 else 0
/// - For positive inputs: f(x) = x
/// - For negative inputs: f(x) = 0
/// where x is the input tensor.
/// The ReLU function is non-linear and allows for faster training of deep neural networks.
pub const ReLU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new ReLU node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*ReLU {
        const self = try allocator.create(ReLU);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
        };
        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *ReLU) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the ReLU function.
    /// The ReLU function is defined as:
    /// f(x) = x if x > 0 else 0
    /// where x is the input tensor.
    /// The ReLU function is non-linear and allows for faster training of deep neural networks.
    pub fn eval(self: *ReLU) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) xv else 0;
        }

        return self.value.?;
    }

    /// Compute the gradient of the ReLU function.
    /// The gradient of the ReLU function is defined as:
    /// ∂f/∂x = 1 if x > 0 else 0
    /// where x is the input tensor.
    /// The gradient of the ReLU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *ReLU, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = if (xv > 0) dv else 0;
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *ReLU) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this ReLU node as a generic Node interface.
    pub fn node(self: *ReLU) Node {
        return Node.init(self);
    }
};

test "relu basic" {
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

    // Create relu operation
    var relu_op = try ReLU.init(allocator, x.node());
    defer relu_op.deinit();

    // First evaluate to cache the values
    const result = try relu_op.eval();
    const expected = [_]f64{
        @as(f64, 0.0), // relu(-2.0)
        @as(f64, 0.0), // relu(-1.0)
        @as(f64, 0.0), // relu(0.0)
        @as(f64, 1.0), // relu(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "relu gradient" {
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

    // Create relu operation
    var relu_op = try ReLU.init(allocator, x.node());
    defer relu_op.deinit();

    // First evaluate to cache the values
    const result = try relu_op.eval();
    const expected = [_]f64{
        @as(f64, 0.0), // relu(-2.0)
        @as(f64, 0.0), // relu(-1.0)
        @as(f64, 0.0), // relu(0.0)
        @as(f64, 1.0), // relu(1.0)
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
    try relu_op.diff(gradTensor);

    // Expected gradients: 1.0 if x > 0, 0.0 otherwise
    const expected_grad = [_]f64{
        @as(f64, 0.0), // relu'(-2.0)
        @as(f64, 0.0), // relu'(-1.0)
        @as(f64, 0.0), // relu'(0.0)
        @as(f64, 1.0), // relu'(1.0)
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "relu with different shapes" {
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

    // Create relu operation
    var relu_op = try ReLU.init(allocator, x.node());
    defer relu_op.deinit();

    // Evaluate
    const result = try relu_op.eval();
    const expected = [_]f64{
        @as(f64, 0.0), // relu(-2.0)
        @as(f64, 0.0), // relu(-1.0)
        @as(f64, 0.0), // relu(0.0)
        @as(f64, 1.0), // relu(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "relu reset" {
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

    // Create relu operation
    var relu_op = try ReLU.init(allocator, x.node());
    defer relu_op.deinit();

    // First evaluation
    const result1 = try relu_op.eval();
    const expected1 = [_]f64{
        @as(f64, 0.0), // relu(-2.0)
        @as(f64, 0.0), // relu(-1.0)
        @as(f64, 0.0), // relu(0.0)
        @as(f64, 1.0), // relu(1.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset
    relu_op.reset();

    // Second evaluation
    const result2 = try relu_op.eval();
    const expected2 = [_]f64{
        @as(f64, 0.0), // relu(-2.0)
        @as(f64, 0.0), // relu(-1.0)
        @as(f64, 0.0), // relu(0.0)
        @as(f64, 1.0), // relu(1.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
