const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

/// Multiply function node.
/// The Multiply node represents the element-wise multiplication of two tensors.
/// It computes the product of each corresponding element in the input tensors.
/// The Multiply node is used in various mathematical computations and neural networks where multiplication is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Multiply node is defined as:
/// f(x, y) = x * y
/// where x and y are the input tensors.
/// The Multiply node is typically used in conjunction with other nodes to build complex computation graphs.
/// It is commonly used in neural networks for operations such as weight updates and loss calculations.
pub const Multiply = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    /// Creates a new multiplication node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Multiply {
        const self = try allocator.create(Multiply);
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
    pub fn deinit(self: *Multiply) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the multiply function.
    /// The multiply function is defined as:
    /// f(x, y) = x * y
    /// where x and y are the input tensors.
    /// The multiply function is often used in conjunction with other nodes to build complex computation graphs.
    pub fn eval(self: *Multiply) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();
        const y = try self.y.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = xv * yv;
        }

        std.debug.print("Multiply-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the multiply function.
    /// The gradient of the multiply function is defined as:
    /// ∂f/∂x = y
    /// ∂f/∂y = x
    /// where x and y are the input tensors.
    /// The gradient of the multiply function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Multiply, dval: *Tensor) !void {
        const x = try self.x.eval();
        const y = try self.y.eval();

        const grad_x = try Tensor.init(self.allocator, dval.shape);
        defer grad_x.deinit();
        const grad_y = try Tensor.init(self.allocator, dval.shape);
        defer grad_y.deinit();

        for (grad_x.data, grad_y.data, x.data, y.data, dval.data) |*gx, *gy, xv, yv, dv| {
            gx.* = dv * yv;
            gy.* = dv * xv;
        }

        try self.x.diff(grad_x);
        try self.y.diff(grad_y);

        std.debug.print("Multiply-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Multiply) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this multiply node as a generic Node interface.
    pub fn node(self: *Multiply) Node {
        return Node.init(self);
    }
};

test "multiply basic" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensors with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 2.0; // positive input

    const yTensor = try graph.tensor(&[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0; // positive input
    yTensor.data[1] = 0.0; // zero input
    yTensor.data[2] = 5.0; // positive input
    yTensor.data[3] = -2.0; // negative input

    // Create variables
    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // Create multiply operation
    var mul_op = try graph.multiply(x.node(), y.node());
    defer mul_op.deinit();

    // Evaluate forward pass
    const result = try mul_op.eval();

    // Expected values for each input pair:
    // f(x, y) = x * y
    const expected = [_]f64{
        -2.0 * 3.0, // (-2.0) * 3.0 = -6.0
        -1.0 * 0.0, // (-1.0) * 0.0 = 0.0
        0.0 * 5.0, // 0.0 * 5.0 = 0.0
        2.0 * -2.0, // 2.0 * (-2.0) = -4.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Verify tensor shape is preserved
    try std.testing.expectEqual(@as(usize, 4), result.shape[0]);
}

test "multiply gradient" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensors with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 2.0; // positive input

    const yTensor = try graph.tensor(&[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0; // positive input
    yTensor.data[1] = 0.0; // zero input
    yTensor.data[2] = 5.0; // positive input
    yTensor.data[3] = -2.0; // negative input

    // Create variables
    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // Create multiply operation
    var mul_op = try graph.multiply(x.node(), y.node());
    defer mul_op.deinit();

    // First evaluate to cache the values
    _ = try mul_op.eval();

    // Create gradient tensor
    const gradTensor = try graph.tensor(&[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try mul_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = y
    // ∂f/∂y = x
    const expected_x_grad = [_]f64{
        3.0, // ∂f/∂x = y = 3.0
        0.0, // ∂f/∂x = y = 0.0
        5.0, // ∂f/∂x = y = 5.0
        -2.0, // ∂f/∂x = y = -2.0
    };

    const expected_y_grad = [_]f64{
        -2.0, // ∂f/∂y = x = -2.0
        -1.0, // ∂f/∂y = x = -1.0
        0.0, // ∂f/∂y = x = 0.0
        2.0, // ∂f/∂y = x = 2.0
    };

    for (x.grad.data, expected_x_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    for (y.grad.data, expected_y_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "multiply with multiple shapes" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test 1: 2D shape [2, 2]
    {
        // Create input tensors with shape [2, 2]
        const xTensor = try graph.tensor(&[_]usize{ 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -2.0; // [0,0]
        xTensor.data[1] = -1.0; // [0,1]
        xTensor.data[2] = 0.0; // [1,0]
        xTensor.data[3] = 2.0; // [1,1]

        const yTensor = try graph.tensor(&[_]usize{ 2, 2 });
        defer yTensor.deinit();
        yTensor.data[0] = 3.0; // [0,0]
        yTensor.data[1] = 0.0; // [0,1]
        yTensor.data[2] = 5.0; // [1,0]
        yTensor.data[3] = -2.0; // [1,1]

        // Create variables
        var x = try graph.variable("x", xTensor);
        defer x.deinit();
        var y = try graph.variable("y", yTensor);
        defer y.deinit();

        // Create multiply operation
        var mul_op = try graph.multiply(x.node(), y.node());
        defer mul_op.deinit();

        // Evaluate forward pass
        const result = try mul_op.eval();

        // Expected values for each input pair:
        // f(x, y) = x * y
        const expected = [_]f64{
            -2.0 * 3.0, // (-2.0) * 3.0 = -6.0
            -1.0 * 0.0, // (-1.0) * 0.0 = 0.0
            0.0 * 5.0, // 0.0 * 5.0 = 0.0
            2.0 * -2.0, // 2.0 * (-2.0) = -4.0
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Test gradient computation
        const gradTensor = try graph.tensor(&[_]usize{ 2, 2 });
        defer gradTensor.deinit();
        gradTensor.data[0] = 1.0;
        gradTensor.data[1] = 1.0;
        gradTensor.data[2] = 1.0;
        gradTensor.data[3] = 1.0;

        try mul_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = y
        // ∂f/∂y = x
        const expected_x_grad = [_]f64{
            3.0, // ∂f/∂x = y = 3.0
            0.0, // ∂f/∂x = y = 0.0
            5.0, // ∂f/∂x = y = 5.0
            -2.0, // ∂f/∂x = y = -2.0
        };

        const expected_y_grad = [_]f64{
            -2.0, // ∂f/∂y = x = -2.0
            -1.0, // ∂f/∂y = x = -1.0
            0.0, // ∂f/∂y = x = 0.0
            2.0, // ∂f/∂y = x = 2.0
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
        const xTensor = try graph.tensor(&[_]usize{ 2, 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -2.0; // [0,0,0]
        xTensor.data[1] = -1.0; // [0,0,1]
        xTensor.data[2] = 0.0; // [0,1,0]
        xTensor.data[3] = 2.0; // [0,1,1]
        xTensor.data[4] = -1.5; // [1,0,0]
        xTensor.data[5] = -0.5; // [1,0,1]
        xTensor.data[6] = 0.5; // [1,1,0]
        xTensor.data[7] = 1.5; // [1,1,1]

        const yTensor = try graph.tensor(&[_]usize{ 2, 2, 2 });
        defer yTensor.deinit();
        yTensor.data[0] = 3.0; // [0,0,0]
        yTensor.data[1] = 0.0; // [0,0,1]
        yTensor.data[2] = 5.0; // [0,1,0]
        yTensor.data[3] = -2.0; // [0,1,1]
        yTensor.data[4] = 2.0; // [1,0,0]
        yTensor.data[5] = -1.0; // [1,0,1]
        yTensor.data[6] = 4.0; // [1,1,0]
        yTensor.data[7] = -3.0; // [1,1,1]

        // Create variables
        var x = try graph.variable("x", xTensor);
        defer x.deinit();
        var y = try graph.variable("y", yTensor);
        defer y.deinit();

        // Create multiply operation
        var mul_op = try graph.multiply(x.node(), y.node());
        defer mul_op.deinit();

        // Evaluate forward pass
        const result = try mul_op.eval();

        // Expected values for each input pair:
        // f(x, y) = x * y
        const expected = [_]f64{
            -2.0 * 3.0, // (-2.0) * 3.0 = -6.0
            -1.0 * 0.0, // (-1.0) * 0.0 = 0.0
            0.0 * 5.0, // 0.0 * 5.0 = 0.0
            2.0 * -2.0, // 2.0 * (-2.0) = -4.0
            -1.5 * 2.0, // (-1.5) * 2.0 = -3.0
            -0.5 * -1.0, // (-0.5) * (-1.0) = 0.5
            0.5 * 4.0, // 0.5 * 4.0 = 2.0
            1.5 * -3.0, // 1.5 * (-3.0) = -4.5
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Test gradient computation
        const gradTensor = try graph.tensor(&[_]usize{ 2, 2, 2 });
        defer gradTensor.deinit();
        for (gradTensor.data) |*v| {
            v.* = 1.0;
        }

        try mul_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = y
        // ∂f/∂y = x
        const expected_x_grad = [_]f64{
            3.0, // ∂f/∂x = y = 3.0
            0.0, // ∂f/∂x = y = 0.0
            5.0, // ∂f/∂x = y = 5.0
            -2.0, // ∂f/∂x = y = -2.0
            2.0, // ∂f/∂x = y = 2.0
            -1.0, // ∂f/∂x = y = -1.0
            4.0, // ∂f/∂x = y = 4.0
            -3.0, // ∂f/∂x = y = -3.0
        };

        const expected_y_grad = [_]f64{
            -2.0, // ∂f/∂y = x = -2.0
            -1.0, // ∂f/∂y = x = -1.0
            0.0, // ∂f/∂y = x = 0.0
            2.0, // ∂f/∂y = x = 2.0
            -1.5, // ∂f/∂y = x = -1.5
            -0.5, // ∂f/∂y = x = -0.5
            0.5, // ∂f/∂y = x = 0.5
            1.5, // ∂f/∂y = x = 1.5
        };

        for (x.grad.data, expected_x_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        for (y.grad.data, expected_y_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "multiply reset" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensors with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 2.0; // positive input

    const yTensor = try graph.tensor(&[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0; // positive input
    yTensor.data[1] = 0.0; // zero input
    yTensor.data[2] = 5.0; // positive input
    yTensor.data[3] = -2.0; // negative input

    // Create variables
    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // Create multiply operation
    var mul_op = try graph.multiply(x.node(), y.node());
    defer mul_op.deinit();

    // First evaluation
    const result1 = try mul_op.eval();

    // Expected values for each input pair:
    // f(x, y) = x * y
    const expected1 = [_]f64{
        -2.0 * 3.0, // (-2.0) * 3.0 = -6.0
        -1.0 * 0.0, // (-1.0) * 0.0 = 0.0
        0.0 * 5.0, // 0.0 * 5.0 = 0.0
        2.0 * -2.0, // 2.0 * (-2.0) = -4.0
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    mul_op.reset();
    const result2 = try mul_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        -2.0 * 3.0, // (-2.0) * 3.0 = -6.0
        -1.0 * 0.0, // (-1.0) * 0.0 = 0.0
        0.0 * 5.0, // 0.0 * 5.0 = 0.0
        2.0 * -2.0, // 2.0 * (-2.0) = -4.0
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
