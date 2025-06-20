const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// PReLU (Parametric ReLU) function node.
/// PReLU is a variant of the ReLU activation function that allows for a small, learnable slope for negative inputs.
/// The PReLU function is defined as:
/// f(x) = x if x > 0 else α * x
/// - For positive inputs: f(x) = x
/// - For negative inputs: f(x) = α * x
/// where α is a learnable parameter tensor that must have the same shape as the input.
/// The alpha parameter is a learnable parameter that can be trained during the optimization process.
/// The PReLU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
/// The PReLU function is particularly useful in deep neural networks where the ReLU function may lead to dead neurons.
/// It allows the model to learn a small slope for negative inputs, which can help improve gradient flow during training.
pub const PReLU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    grad: *Tensor, // gradient of alpha
    alpha: *Tensor, // learnable parameter (trainable)
    x: Node,

    /// Creates a new PReLU node with the given input node and alpha value.
    /// The alpha tensor must have the same shape as the input tensor.
    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: *Tensor) !*PReLU {
        const grad = try Tensor.init(allocator, alpha.shape);
        errdefer grad.deinit();

        const self = try allocator.create(PReLU);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .value = null,
            .grad = grad,
            .alpha = alpha,
            .x = x,
        };
        self.grad.zeros();

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *PReLU) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.grad.deinit();
        self.allocator.destroy(self);
    }

    /// Evaluate the PReLU function.
    /// The PReLU function is defined as:
    /// f(x) = x if x > 0 else α * x
    /// where α is a learnable parameter (default 0.01).
    /// The PReLU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
    /// The PReLU function is particularly useful in deep neural networks where the ReLU function may lead to dead neurons.
    /// It allows the model to learn a small slope for negative inputs, which can help improve gradient flow during training.
    pub fn eval(self: *PReLU) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data, 0..) |*v, xv, i| {
            const alpha = self.alpha.data[i % self.alpha.data.len];
            v.* = if (xv > 0) xv else alpha * xv;
        }

        return self.value.?;
    }

    /// Compute the gradient of the PReLU function.
    /// The gradient of the PReLU function is defined as:
    /// ∂f/∂x = 1 if x > 0 else α
    /// ∂f/∂α = x if x > 0 else 0
    /// where α is a learnable parameter (default 0.01).
    /// The gradient of the PReLU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *PReLU, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data, 0..) |*v, xv, dv, i| {
            const alpha = self.alpha.data[i % self.alpha.data.len];
            v.* = if (xv > 0) dv else dv * alpha;
            self.grad.data[i % self.grad.data.len] += if (xv > 0) 0 else dv * xv;
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *PReLU) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this PReLU node as a generic Node interface.
    pub fn node(self: *PReLU) Node {
        return Node.init(self);
    }
};

test "prelu basic" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create alpha tensor with same shape as input
    const alphaTensor = try Tensor.init(allocator, &[_]usize{4}); // same shape as input
    defer alphaTensor.deinit();
    alphaTensor.data[0] = 0.01;
    alphaTensor.data[1] = 0.02;
    alphaTensor.data[2] = 0.03;
    alphaTensor.data[3] = 0.04;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create prelu operation
    var prelu_op = try PReLU.init(allocator, x.node(), alphaTensor);
    defer prelu_op.deinit();

    // First evaluate to cache the values
    const result = try prelu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.02), // 0.02 * -1.0
        @as(f64, 0.0), // 0.0
        @as(f64, 1.0), // 1.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "prelu gradient" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create alpha tensor with same shape as input
    const alphaTensor = try Tensor.init(allocator, &[_]usize{4}); // same shape as input
    defer alphaTensor.deinit();
    alphaTensor.data[0] = 0.01;
    alphaTensor.data[1] = 0.02;
    alphaTensor.data[2] = 0.03;
    alphaTensor.data[3] = 0.04;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create prelu operation
    var prelu_op = try PReLU.init(allocator, x.node(), alphaTensor);
    defer prelu_op.deinit();

    // First evaluate to cache the values
    const result = try prelu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.02), // 0.02 * -1.0
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
    try prelu_op.diff(gradTensor);

    // Expected gradients for x: 1 if x > 0 else alpha
    const expected_grad_x = [_]f64{
        @as(f64, 0.01), // alpha for x < 0
        @as(f64, 0.02), // alpha for x < 0
        @as(f64, 0.03), // alpha for x = 0
        @as(f64, 1.0), // 1 for x > 0
    };

    // Expected gradients for alpha: x if x < 0 else 0
    const expected_grad_alpha = [_]f64{
        @as(f64, -2.0), // -2.0 for x < 0
        @as(f64, -1.0), // -1.0 for x < 0
        @as(f64, 0.0), // 0.0 for x = 0
        @as(f64, 0.0), // 0.0 for x > 0
    };

    // Check gradients for x
    for (x.grad.data, expected_grad_x) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Check gradients for alpha
    for (prelu_op.grad.data, expected_grad_alpha) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "prelu with different shapes" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create alpha tensor with same shape as input
    const alphaTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 }); // same shape as input
    defer alphaTensor.deinit();
    alphaTensor.data[0] = 0.01;
    alphaTensor.data[1] = 0.02;
    alphaTensor.data[2] = 0.03;
    alphaTensor.data[3] = 0.04;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create prelu operation
    var prelu_op = try PReLU.init(allocator, x.node(), alphaTensor);
    defer prelu_op.deinit();

    // Evaluate
    const result = try prelu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.02), // 0.02 * -1.0
        @as(f64, 0.0), // 0.0
        @as(f64, 1.0), // 1.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "prelu reset" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create alpha tensor with same shape as input
    const alphaTensor = try Tensor.init(allocator, &[_]usize{4}); // same shape as input
    defer alphaTensor.deinit();
    alphaTensor.data[0] = 0.01;
    alphaTensor.data[1] = 0.02;
    alphaTensor.data[2] = 0.03;
    alphaTensor.data[3] = 0.04;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create prelu operation
    var prelu_op = try PReLU.init(allocator, x.node(), alphaTensor);
    defer prelu_op.deinit();

    // First evaluation
    const result1 = try prelu_op.eval();
    const expected1 = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.02), // 0.02 * -1.0
        @as(f64, 0.0), // 0.0
        @as(f64, 1.0), // 1.0
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    prelu_op.reset();
    const result2 = try prelu_op.eval();
    const expected2 = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0
        @as(f64, -0.02), // 0.02 * -1.0
        @as(f64, 0.0), // 0.0
        @as(f64, 1.0), // 1.0
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "prelu alpha learning" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{2});
    defer xTensor.deinit();
    xTensor.data[0] = -1.0;
    xTensor.data[1] = 2.0;

    // Create alpha tensor with same shape as input
    const alphaTensor = try Tensor.init(allocator, &[_]usize{2}); // same shape as input
    defer alphaTensor.deinit();
    alphaTensor.data[0] = 0.01;
    alphaTensor.data[1] = 0.02;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create prelu operation
    var prelu_op = try PReLU.init(allocator, x.node(), alphaTensor);
    defer prelu_op.deinit();

    // First evaluation
    const result = try prelu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.01), // 0.01 * -1.0
        @as(f64, 2.0), // 2.0 (positive input)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{2});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;

    // Compute gradients
    try prelu_op.diff(gradTensor);

    // Expected gradients for x: 1 if x > 0 else alpha
    const expected_grad_x = [_]f64{
        @as(f64, 0.01), // alpha for x < 0
        @as(f64, 1.0), // 1 for x > 0
    };

    // Expected gradients for alpha: x if x < 0 else 0
    const expected_grad_alpha = [_]f64{
        @as(f64, -1.0), // -1.0 for x < 0
        @as(f64, 0.0), // 0.0 for x > 0
    };

    // Check gradients for x
    for (x.grad.data, expected_grad_x) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Check gradients for alpha
    for (prelu_op.grad.data, expected_grad_alpha) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Simulate a learning step by updating alpha
    const learning_rate = 0.1;
    for (alphaTensor.data, prelu_op.grad.data) |*alpha, grad| {
        alpha.* -= learning_rate * grad;
    }

    // Verify alpha was updated correctly
    const expected_alpha = [_]f64{
        @as(f64, 0.11), // 0.01 - 0.1 * (-1.0)
        @as(f64, 0.02), // 0.02 - 0.1 * 0.0 (no change for positive input)
    };

    for (alphaTensor.data, expected_alpha) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Evaluate again with updated alpha
    prelu_op.reset();
    const result2 = try prelu_op.eval();
    const expected2 = [_]f64{
        @as(f64, -0.11), // 0.11 * -1.0
        @as(f64, 2.0), // 2.0 (positive input)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "prelu shape mismatch" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create alpha tensor with different shape
    const alphaTensor = try Tensor.init(allocator, &[_]usize{2}); // different shape from input
    defer alphaTensor.deinit();
    alphaTensor.data[0] = 0.01;
    alphaTensor.data[1] = 0.02;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create prelu operation - should now work with different shapes
    var prelu_op = try PReLU.init(allocator, x.node(), alphaTensor);
    defer prelu_op.deinit();

    // Evaluate and verify results
    const result = try prelu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0 (using alpha[0])
        @as(f64, -0.02), // 0.02 * -1.0 (using alpha[1])
        @as(f64, 0.0), // 0.0 (using alpha[0])
        @as(f64, 1.0), // 1.0 (using alpha[1])
    };

    // Verify the results
    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Test gradient computation
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    try prelu_op.diff(gradTensor);

    // Expected gradients for x: 1 if x > 0 else alpha
    const expected_grad_x = [_]f64{
        @as(f64, 0.01), // alpha[0] for x < 0
        @as(f64, 0.02), // alpha[1] for x < 0
        @as(f64, 0.01), // alpha[0] for x = 0
        @as(f64, 1.0), // 1 for x > 0
    };

    // Expected gradients for alpha: x if x < 0 else 0
    const expected_grad_alpha = [_]f64{
        @as(f64, -2.0), // -2.0 for x < 0 using alpha[0]
        @as(f64, -1.0), // -1.0 for x < 0 using alpha[1]
    };

    // Check gradients for x
    for (x.grad.data, expected_grad_x) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Check gradients for alpha
    for (prelu_op.grad.data, expected_grad_alpha) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "prelu with 2d shapes" {
    const allocator = std.testing.allocator;

    // Create input tensor with shape [2, 2]
    const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // [0,0]
    xTensor.data[1] = -1.0; // [0,1]
    xTensor.data[2] = 0.0; // [1,0]
    xTensor.data[3] = 1.0; // [1,1]

    // Create alpha tensor with same shape [2, 2]
    const alphaTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer alphaTensor.deinit();
    alphaTensor.data[0] = 0.01; // [0,0]
    alphaTensor.data[1] = 0.02; // [0,1]
    alphaTensor.data[2] = 0.03; // [1,0]
    alphaTensor.data[3] = 0.04; // [1,1]

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create prelu operation
    var prelu_op = try PReLU.init(allocator, x.node(), alphaTensor);
    defer prelu_op.deinit();

    // Evaluate forward pass
    const result = try prelu_op.eval();
    const expected = [_]f64{
        @as(f64, -0.02), // 0.01 * -2.0 at [0,0]
        @as(f64, -0.02), // 0.02 * -1.0 at [0,1]
        @as(f64, 0.0), // 0.0 at [1,0]
        @as(f64, 1.0), // 1.0 at [1,1]
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Test gradient computation
    const gradTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0; // [0,0]
    gradTensor.data[1] = 1.0; // [0,1]
    gradTensor.data[2] = 1.0; // [1,0]
    gradTensor.data[3] = 1.0; // [1,1]

    try prelu_op.diff(gradTensor);

    // Expected gradients for x: 1 if x > 0 else alpha
    const expected_grad_x = [_]f64{
        @as(f64, 0.01), // alpha for x < 0 at [0,0]
        @as(f64, 0.02), // alpha for x < 0 at [0,1]
        @as(f64, 0.03), // alpha for x = 0 at [1,0]
        @as(f64, 1.0), // 1 for x > 0 at [1,1]
    };

    // Expected gradients for alpha: x if x < 0 else 0
    const expected_grad_alpha = [_]f64{
        @as(f64, -2.0), // -2.0 for x < 0 at [0,0]
        @as(f64, -1.0), // -1.0 for x < 0 at [0,1]
        @as(f64, 0.0), // 0.0 for x = 0 at [1,0]
        @as(f64, 0.0), // 0.0 for x > 0 at [1,1]
    };

    // Check gradients for x
    for (x.grad.data, expected_grad_x) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Check gradients for alpha
    for (prelu_op.grad.data, expected_grad_alpha) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Simulate a learning step
    const learning_rate = 0.1;
    for (alphaTensor.data, prelu_op.grad.data) |*alpha, grad| {
        alpha.* -= learning_rate * grad;
    }

    // Verify alpha was updated correctly
    const expected_alpha = [_]f64{
        @as(f64, 0.21), // 0.01 - 0.1 * (-2.0) at [0,0]
        @as(f64, 0.12), // 0.02 - 0.1 * (-1.0) at [0,1]
        @as(f64, 0.03), // 0.03 - 0.1 * 0.0 at [1,0]
        @as(f64, 0.04), // 0.04 - 0.1 * 0.0 at [1,1]
    };

    for (alphaTensor.data, expected_alpha) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Evaluate again with updated alpha
    prelu_op.reset();
    const result2 = try prelu_op.eval();
    const expected2 = [_]f64{
        @as(f64, -0.42), // 0.21 * -2.0 at [0,0]
        @as(f64, -0.12), // 0.12 * -1.0 at [0,1]
        @as(f64, 0.0), // 0.0 at [1,0]
        @as(f64, 1.0), // 1.0 at [1,1]
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "prelu allocation failure" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create alpha tensor with same shape as input
    const alphaTensor = try Tensor.init(allocator, &[_]usize{4});
    defer alphaTensor.deinit();
    alphaTensor.data[0] = 0.01;
    alphaTensor.data[1] = 0.02;
    alphaTensor.data[2] = 0.03;
    alphaTensor.data[3] = 0.04;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Test PReLU struct allocation failure
    {
        var failing_allocator = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });

        const result = PReLU.init(failing_allocator.allocator(), x.node(), alphaTensor);
        try std.testing.expectError(error.OutOfMemory, result);
    }

    // Test gradient tensor allocation failure
    {
        var failing_allocator = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 1 });

        const result = PReLU.init(failing_allocator.allocator(), x.node(), alphaTensor);
        try std.testing.expectError(error.OutOfMemory, result);
    }

    // Test value tensor allocation failure during eval
    {
        var prelu_op = try PReLU.init(allocator, x.node(), alphaTensor);
        defer prelu_op.deinit();

        var failing_allocator = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
        prelu_op.allocator = failing_allocator.allocator();

        const result = prelu_op.eval();
        try std.testing.expectError(error.OutOfMemory, result);
    }

    // Test successful allocation after failures
    {
        var prelu_op = try PReLU.init(allocator, x.node(), alphaTensor);
        defer prelu_op.deinit();

        const result = try prelu_op.eval();
        try std.testing.expectEqual(@as(usize, 4), result.shape[0]);
    }
}
