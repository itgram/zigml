const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Subtract function node.
/// The Subtract node represents the subtraction operation between two tensors.
/// It computes the element-wise difference between the two input tensors.
/// The Subtract node is used in neural networks and mathematical computations where subtraction is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x, y) = x - y
/// where x and y are the input tensors.
/// The Subtract function is often used in loss functions and optimization algorithms.
pub const Subtract = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    /// Creates a new subtraction node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Subtract {
        const self = try allocator.create(Subtract);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
            .y = y,
        };

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Subtract) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the subtract function.
    /// The subtract function is defined as:
    /// f(x, y) = x - y
    /// where x and y are the input tensors.
    /// The subtract function is often used in loss functions and optimization algorithms.
    pub fn eval(self: *Subtract) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();
        const y = try self.y.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = xv - yv;
        }

        return self.value.?;
    }

    /// Compute the gradient of the subtract function.
    /// The gradient of the subtract function is defined as:
    /// ∂f/∂x = 1
    /// ∂f/∂y = -1
    /// where x and y are the input tensors.
    /// The gradient of the subtract function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Subtract, dval: *Tensor) !void {
        const grad_x = try Tensor.init(self.allocator, dval.shape);
        defer grad_x.deinit();
        const grad_y = try Tensor.init(self.allocator, dval.shape);
        defer grad_y.deinit();

        for (grad_x.data, grad_y.data, dval.data) |*gx, *gy, dv| {
            gx.* = dv;
            gy.* = -dv;
        }

        try self.x.diff(grad_x);
        try self.y.diff(grad_y);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Subtract) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this subtract node as a generic Node interface.
    pub fn node(self: *Subtract) Node {
        return Node.init(self);
    }
};

test "subtract basic" {
    const allocator = std.testing.allocator;

    // Create input tensors with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input
    xTensor.data[4] = 2.0; // positive input
    xTensor.data[5] = 3.0; // positive input

    const yTensor = try Tensor.init(allocator, &[_]usize{6});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0; // positive input
    yTensor.data[1] = 0.0; // zero input
    yTensor.data[2] = 1.0; // positive input
    yTensor.data[3] = -1.0; // negative input
    yTensor.data[4] = 2.0; // positive input
    yTensor.data[5] = -2.0; // negative input

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // Create subtract operation
    var sub_op = try Subtract.init(allocator, x.node(), y.node());
    defer sub_op.deinit();

    // Evaluate forward pass
    const result = try sub_op.eval();

    // Expected values for each input pair:
    // f(x, y) = x - y
    const expected = [_]f64{
        -2.0 - 3.0, // (-2.0) - 3.0 = -5.0
        -1.0 - 0.0, // (-1.0) - 0.0 = -1.0
        0.0 - 1.0, // 0.0 - 1.0 = -1.0
        1.0 - -1.0, // 1.0 - (-1.0) = 2.0
        2.0 - 2.0, // 2.0 - 2.0 = 0.0
        3.0 - -2.0, // 3.0 - (-2.0) = 5.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Verify tensor shape is preserved
    try std.testing.expectEqual(@as(usize, 6), result.shape[0]);
}

test "subtract gradient" {
    const allocator = std.testing.allocator;

    // Create input tensors with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input
    xTensor.data[4] = 2.0; // positive input
    xTensor.data[5] = 3.0; // positive input

    const yTensor = try Tensor.init(allocator, &[_]usize{6});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0; // positive input
    yTensor.data[1] = 0.0; // zero input
    yTensor.data[2] = 1.0; // positive input
    yTensor.data[3] = -1.0; // negative input
    yTensor.data[4] = 2.0; // positive input
    yTensor.data[5] = -2.0; // negative input

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // Create subtract operation
    var sub_op = try Subtract.init(allocator, x.node(), y.node());
    defer sub_op.deinit();

    // First evaluate to cache the values
    _ = try sub_op.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{6});
    defer gradTensor.deinit();
    for (gradTensor.data) |*v| {
        v.* = 1.0;
    }

    // Compute gradients
    try sub_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = 1
    // ∂f/∂y = -1
    const expected_x_grad = [_]f64{
        1.0, // ∂f/∂x = 1
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    };

    const expected_y_grad = [_]f64{
        -1.0, // ∂f/∂y = -1
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    };

    for (x.grad.data, expected_x_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    for (y.grad.data, expected_y_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "subtract with multiple shapes" {
    const allocator = std.testing.allocator;

    // Test 1: 2D shape [2, 2]
    {
        // Create input tensors with shape [2, 2]
        const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -2.0; // [0,0]
        xTensor.data[1] = -1.0; // [0,1]
        xTensor.data[2] = 0.0; // [1,0]
        xTensor.data[3] = 1.0; // [1,1]

        const yTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer yTensor.deinit();
        yTensor.data[0] = 3.0; // [0,0]
        yTensor.data[1] = 0.0; // [0,1]
        yTensor.data[2] = 1.0; // [1,0]
        yTensor.data[3] = -1.0; // [1,1]

        // Create variables
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();
        var y = try Variable.init(allocator, "y", yTensor);
        defer y.deinit();

        // Create subtract operation
        var sub_op = try Subtract.init(allocator, x.node(), y.node());
        defer sub_op.deinit();

        // Evaluate forward pass
        const result = try sub_op.eval();

        // Expected values for each input pair:
        // f(x, y) = x - y
        const expected = [_]f64{
            -2.0 - 3.0, // (-2.0) - 3.0 = -5.0
            -1.0 - 0.0, // (-1.0) - 0.0 = -1.0
            0.0 - 1.0, // 0.0 - 1.0 = -1.0
            1.0 - -1.0, // 1.0 - (-1.0) = 2.0
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

        try sub_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1
        // ∂f/∂y = -1
        const expected_x_grad = [_]f64{
            1.0, // ∂f/∂x = 1
            1.0,
            1.0,
            1.0,
        };

        const expected_y_grad = [_]f64{
            -1.0, // ∂f/∂y = -1
            -1.0,
            -1.0,
            -1.0,
        };

        for (x.grad.data, expected_x_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        for (y.grad.data, expected_y_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }

    // Test 2: 3D shape [2, 2, 2]
    {
        // Create input tensors with shape [2, 2, 2]
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

        const yTensor = try Tensor.init(allocator, &[_]usize{ 2, 2, 2 });
        defer yTensor.deinit();
        yTensor.data[0] = 3.0; // [0,0,0]
        yTensor.data[1] = 0.0; // [0,0,1]
        yTensor.data[2] = 1.0; // [0,1,0]
        yTensor.data[3] = -1.0; // [0,1,1]
        yTensor.data[4] = 2.0; // [1,0,0]
        yTensor.data[5] = -2.0; // [1,0,1]
        yTensor.data[6] = 3.0; // [1,1,0]
        yTensor.data[7] = -3.0; // [1,1,1]

        // Create variables
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();
        var y = try Variable.init(allocator, "y", yTensor);
        defer y.deinit();

        // Create subtract operation
        var sub_op = try Subtract.init(allocator, x.node(), y.node());
        defer sub_op.deinit();

        // Evaluate forward pass
        const result = try sub_op.eval();

        // Expected values for each input pair:
        // f(x, y) = x - y
        const expected = [_]f64{
            -2.0 - 3.0, // (-2.0) - 3.0 = -5.0
            -1.0 - 0.0, // (-1.0) - 0.0 = -1.0
            0.0 - 1.0, // 0.0 - 1.0 = -1.0
            1.0 - -1.0, // 1.0 - (-1.0) = 2.0
            2.0 - 2.0, // 2.0 - 2.0 = 0.0
            3.0 - -2.0, // 3.0 - (-2.0) = 5.0
            4.0 - 3.0, // 4.0 - 3.0 = 1.0
            5.0 - -3.0, // 5.0 - (-3.0) = 8.0
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

        try sub_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1
        // ∂f/∂y = -1
        const expected_x_grad = [_]f64{
            1.0, // ∂f/∂x = 1
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        };

        const expected_y_grad = [_]f64{
            -1.0, // ∂f/∂y = -1
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
        };

        for (x.grad.data, expected_x_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        for (y.grad.data, expected_y_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "subtract reset" {
    const allocator = std.testing.allocator;

    // Create input tensors with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input

    const yTensor = try Tensor.init(allocator, &[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0; // positive input
    yTensor.data[1] = 0.0; // zero input
    yTensor.data[2] = 1.0; // positive input
    yTensor.data[3] = -1.0; // negative input

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // Create subtract operation
    var sub_op = try Subtract.init(allocator, x.node(), y.node());
    defer sub_op.deinit();

    // First evaluation
    const result1 = try sub_op.eval();

    // Expected values for each input pair:
    // f(x, y) = x - y
    const expected1 = [_]f64{
        -2.0 - 3.0, // (-2.0) - 3.0 = -5.0
        -1.0 - 0.0, // (-1.0) - 0.0 = -1.0
        0.0 - 1.0, // 0.0 - 1.0 = -1.0
        1.0 - -1.0, // 1.0 - (-1.0) = 2.0
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    sub_op.reset();
    const result2 = try sub_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        -2.0 - 3.0, // (-2.0) - 3.0 = -5.0
        -1.0 - 0.0, // (-1.0) - 0.0 = -1.0
        0.0 - 1.0, // 0.0 - 1.0 = -1.0
        1.0 - -1.0, // 1.0 - (-1.0) = 2.0
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
