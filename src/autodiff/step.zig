const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Step function node.
/// where threshold is a configurable value (default is 0.0).
/// The Step function is often used in binary classification tasks and as an activation function in neural networks.
/// It is not differentiable at the threshold, but it can be used in contexts where a hard thresholding is required.
/// The Step function is useful for creating binary outputs from continuous inputs.
/// It is defined as:
/// f(x) = 1 if x >= threshold, else 0
/// where x is the input tensor and threshold is a configurable value (default is 0.0).
pub const Step = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    threshold: f64 = 0.0, // Default threshold value

    /// Creates a new step node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node, threshold: f64) !*Step {
        const self = try allocator.create(Step);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
            .threshold = threshold,
        };

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Step) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the step function.
    /// The step function is defined as:
    /// f(x) = 1 if x >= threshold, else 0
    /// where x is the input tensor and threshold is a configurable value (default is 0.0).
    /// The step function is often used in binary classification tasks and as an activation function in neural networks.
    /// It is not differentiable at the threshold, but it can be used in contexts where a hard thresholding is required.
    /// The step function is useful for creating binary outputs from continuous inputs.
    pub fn eval(self: *Step) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv >= self.threshold) 1.0 else 0.0;
        }

        return self.value.?;
    }

    /// Compute the gradient of the step function.
    /// The gradient of the step function is defined as:
    /// ∂f/∂x = 0
    /// where x is the input tensor.
    /// The gradient of the step function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Step, dval: *Tensor) !void {
        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data) |*v| {
            v.* = 0;
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Step) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this step node as a generic Node interface.
    pub fn node(self: *Step) Node {
        return Node.init(self);
    }
};

test "step basic" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // below threshold
    xTensor.data[1] = -1.0; // below threshold
    xTensor.data[2] = -0.1; // below threshold
    xTensor.data[3] = 0.0; // at threshold
    xTensor.data[4] = 0.1; // above threshold
    xTensor.data[5] = 1.0; // above threshold

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create step operation with default threshold (0.0)
    var step_op = try Step.init(allocator, x.node(), 0.0);
    defer step_op.deinit();

    // Evaluate forward pass
    const result = try step_op.eval();

    // Expected values for each input:
    // f(x) = 1 if x >= threshold, else 0
    const expected = [_]f64{
        0.0, // -2.0 < 0.0
        0.0, // -1.0 < 0.0
        0.0, // -0.1 < 0.0
        1.0, // 0.0 = 0.0
        1.0, // 0.1 > 0.0
        1.0, // 1.0 > 0.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Verify tensor shape is preserved
    try std.testing.expectEqual(@as(usize, 6), result.shape[0]);
}

test "step with custom threshold" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // below threshold
    xTensor.data[1] = -1.0; // below threshold
    xTensor.data[2] = -0.1; // below threshold
    xTensor.data[3] = 0.0; // below threshold
    xTensor.data[4] = 0.1; // below threshold
    xTensor.data[5] = 1.0; // above threshold

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create step operation with custom threshold (0.5)
    var step_op = try Step.init(allocator, x.node(), 0.5);
    defer step_op.deinit();

    // Evaluate forward pass
    const result = try step_op.eval();

    // Expected values for each input:
    // f(x) = 1 if x >= threshold, else 0
    const expected = [_]f64{
        0.0, // -2.0 < 0.5
        0.0, // -1.0 < 0.5
        0.0, // -0.1 < 0.5
        0.0, // 0.0 < 0.5
        0.0, // 0.1 < 0.5
        1.0, // 1.0 > 0.5
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "step gradient" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // below threshold
    xTensor.data[1] = -1.0; // below threshold
    xTensor.data[2] = -0.1; // below threshold
    xTensor.data[3] = 0.0; // at threshold
    xTensor.data[4] = 0.1; // above threshold
    xTensor.data[5] = 1.0; // above threshold

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create step operation
    var step_op = try Step.init(allocator, x.node(), 0.0);
    defer step_op.deinit();

    // First evaluate to cache the values
    _ = try step_op.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{6});
    defer gradTensor.deinit();
    for (gradTensor.data) |*v| {
        v.* = 1.0;
    }

    // Compute gradients
    try step_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = 0 (step function is not differentiable at threshold)
    const expected_grad = [_]f64{
        0.0, // gradient is always 0
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "step with multiple shapes" {
    const allocator = std.testing.allocator;

    // Test 1: 2D shape [2, 2]
    {
        // Create input tensor with shape [2, 2]
        const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -1.0; // [0,0]
        xTensor.data[1] = 0.0; // [0,1]
        xTensor.data[2] = 0.5; // [1,0]
        xTensor.data[3] = 1.0; // [1,1]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Create step operation
        var step_op = try Step.init(allocator, x.node(), 0.0);
        defer step_op.deinit();

        // Evaluate forward pass
        const result = try step_op.eval();

        // Expected values for each input:
        // f(x) = 1 if x >= threshold, else 0
        const expected = [_]f64{
            0.0, // -1.0 < 0.0
            1.0, // 0.0 = 0.0
            1.0, // 0.5 > 0.0
            1.0, // 1.0 > 0.0
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

        try step_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 0
        const expected_grad = [_]f64{
            0.0, // gradient is always 0
            0.0,
            0.0,
            0.0,
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
        xTensor.data[3] = 0.5; // [0,1,1]
        xTensor.data[4] = 1.0; // [1,0,0]
        xTensor.data[5] = 1.5; // [1,0,1]
        xTensor.data[6] = 2.0; // [1,1,0]
        xTensor.data[7] = 2.5; // [1,1,1]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Create step operation
        var step_op = try Step.init(allocator, x.node(), 0.0);
        defer step_op.deinit();

        // Evaluate forward pass
        const result = try step_op.eval();

        // Expected values for each input:
        // f(x) = 1 if x >= threshold, else 0
        const expected = [_]f64{
            0.0, // -2.0 < 0.0
            0.0, // -1.0 < 0.0
            1.0, // 0.0 = 0.0
            1.0, // 0.5 > 0.0
            1.0, // 1.0 > 0.0
            1.0, // 1.5 > 0.0
            1.0, // 2.0 > 0.0
            1.0, // 2.5 > 0.0
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

        try step_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 0
        const expected_grad = [_]f64{
            0.0, // gradient is always 0
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "step reset" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -1.0; // below threshold
    xTensor.data[1] = 0.0; // at threshold
    xTensor.data[2] = 0.5; // above threshold
    xTensor.data[3] = 1.0; // above threshold

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create step operation
    var step_op = try Step.init(allocator, x.node(), 0.0);
    defer step_op.deinit();

    // First evaluation
    const result1 = try step_op.eval();

    // Expected values for each input:
    // f(x) = 1 if x >= threshold, else 0
    const expected1 = [_]f64{
        0.0, // -1.0 < 0.0
        1.0, // 0.0 = 0.0
        1.0, // 0.5 > 0.0
        1.0, // 1.0 > 0.0
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    step_op.reset();
    const result2 = try step_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        0.0, // -1.0 < 0.0
        1.0, // 0.0 = 0.0
        1.0, // 0.5 > 0.0
        1.0, // 1.0 > 0.0
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
