const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Variable node.
/// The Variable node represents a variable in the computation graph.
/// It holds a tensor value and its gradient.
/// The Variable node is used to store parameters in machine learning models.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// Variable(name, value)
/// where `name` is the name of the variable and `value` is the tensor value.
/// The Variable node is commonly used in neural networks to represent weights and biases.
/// It is also used in optimization algorithms to update parameters during training.
/// The Variable node is differentiable, allowing gradients to be computed for backpropagation.
/// The Variable node can be evaluated to get the current value of the tensor.
/// The Variable node can compute the gradient with respect to the value tensor.
/// The Variable node can be used in a computation graph to represent trainable parameters.
/// It is useful for implementing optimization algorithms like gradient descent.
/// The Variable node can be used in conjunction with other nodes like Sigmoid, Cos, Sin, etc.
/// The Variable node is a fundamental building block in machine learning frameworks.
pub const Variable = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    value: *Tensor,
    grad: *Tensor,

    /// Creates a new variable node with the given tensor value.
    pub fn init(allocator: std.mem.Allocator, name: []const u8, value: *Tensor) !*Variable {
        const self = try allocator.create(Variable);
        self.* = .{
            .allocator = allocator,
            .name = name,
            .value = value,
            .grad = try Tensor.init(allocator, value.shape),
        };
        self.grad.zero();

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Variable) void {
        self.grad.deinit();
        self.allocator.destroy(self);
    }

    /// Evaluate the variable.
    /// The variable is defined as:
    /// f(x) = x
    /// where x is the input tensor.
    /// The variable is often used in neural networks to represent weights and biases.
    /// It is also used in optimization algorithms to update parameters during training.
    /// The variable is differentiable, allowing gradients to be computed for backpropagation.
    pub fn eval(self: *Variable) !*Tensor {
        return self.value;
    }

    /// Compute the gradient of the variable.
    /// The gradient of the variable is defined as:
    /// ∂f/∂x = 1
    /// where x is the input tensor.
    /// The gradient of the variable is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Variable, dval: *Tensor) void {
        for (self.grad.data, dval.data) |*g, dv| {
            g.* += dv;
        }
    }

    /// Resets the node's state by clearing accumulated gradients.
    /// This is useful when you want to start a new gradient computation.
    pub fn reset(self: *Variable) void {
        self.grad.zero();
    }

    /// Returns this variable node as a generic Node interface.
    pub fn node(self: *Variable) Node {
        return Node.init(self);
    }
};

test "variable basic" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input
    xTensor.data[4] = 2.0; // positive input
    xTensor.data[5] = 3.0; // positive input

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Evaluate forward pass
    const result = try x.eval();

    // Expected values for each input:
    // f(x) = x
    const expected = [_]f64{
        -2.0, // x = -2.0
        -1.0, // x = -1.0
        0.0, // x = 0.0
        1.0, // x = 1.0
        2.0, // x = 2.0
        3.0, // x = 3.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Verify tensor shape is preserved
    try std.testing.expectEqual(@as(usize, 6), result.shape[0]);
}

test "variable gradient" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input
    xTensor.data[4] = 2.0; // positive input
    xTensor.data[5] = 3.0; // positive input

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // First evaluate to cache the values
    _ = try x.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{6});
    defer gradTensor.deinit();
    for (gradTensor.data) |*v| {
        v.* = 1.0;
    }

    // Compute gradients
    x.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = 1
    const expected_grad = [_]f64{
        1.0, // ∂f/∂x = 1
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "variable with multiple shapes" {
    const allocator = std.testing.allocator;

    // Test 1: 2D shape [2, 2]
    {
        // Create input tensor with shape [2, 2]
        const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -2.0; // [0,0]
        xTensor.data[1] = -1.0; // [0,1]
        xTensor.data[2] = 0.0; // [1,0]
        xTensor.data[3] = 1.0; // [1,1]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Evaluate forward pass
        const result = try x.eval();

        // Expected values for each input:
        // f(x) = x
        const expected = [_]f64{
            -2.0, // x = -2.0
            -1.0, // x = -1.0
            0.0, // x = 0.0
            1.0, // x = 1.0
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Test gradient computation
        const gradTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer gradTensor.deinit();
        for (gradTensor.data) |*v| {
            v.* = 1.0;
        }

        x.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1
        const expected_grad = [_]f64{
            1.0, // ∂f/∂x = 1
            1.0,
            1.0,
            1.0,
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }

    // Test 2: 3D shape [2, 2, 2]
    {
        // Create input tensor with shape [2, 2, 2]
        const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -2.0; // [0,0,0]
        xTensor.data[1] = -1.0; // [0,0,1]
        xTensor.data[2] = 0.0; // [0,1,0]
        xTensor.data[3] = 1.0; // [0,1,1]
        xTensor.data[4] = 2.0; // [1,0,0]
        xTensor.data[5] = 3.0; // [1,0,1]
        xTensor.data[6] = 4.0; // [1,1,0]
        xTensor.data[7] = 5.0; // [1,1,1]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Evaluate forward pass
        const result = try x.eval();

        // Expected values for each input:
        // f(x) = x
        const expected = [_]f64{
            -2.0, // x = -2.0
            -1.0, // x = -1.0
            0.0, // x = 0.0
            1.0, // x = 1.0
            2.0, // x = 2.0
            3.0, // x = 3.0
            4.0, // x = 4.0
            5.0, // x = 5.0
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Test gradient computation
        const gradTensor = try Tensor.init(allocator, &[_]usize{ 2, 2, 2 });
        defer gradTensor.deinit();
        for (gradTensor.data) |*v| {
            v.* = 1.0;
        }

        x.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1
        const expected_grad = [_]f64{
            1.0, // ∂f/∂x = 1
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "variable reset" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // First evaluation
    const result1 = try x.eval();

    // Expected values for each input:
    // f(x) = x
    const expected1 = [_]f64{
        -2.0, // x = -2.0
        -1.0, // x = -1.0
        0.0, // x = 0.0
        1.0, // x = 1.0
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    x.reset();
    const result2 = try x.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        -2.0, // x = -2.0
        -1.0, // x = -1.0
        0.0, // x = 0.0
        1.0, // x = 1.0
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "variable gradient accumulation" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // First evaluate to cache the values
    _ = try x.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    for (gradTensor.data) |*v| {
        v.* = 1.0;
    }

    // Compute gradients multiple times to test accumulation
    x.diff(gradTensor);
    x.diff(gradTensor);
    x.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = 1, accumulated 3 times
    const expected_grad = [_]f64{
        3.0, // ∂f/∂x = 1, accumulated 3 times
        3.0,
        3.0,
        3.0,
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset gradients and verify they are zeroed
    x.reset();
    for (x.grad.data) |g| {
        try std.testing.expectApproxEqAbs(0.0, g, 1e-6);
    }
}
