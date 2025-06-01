const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

/// Linear function node.
/// The Linear node represents a linear transformation of the input tensor.
/// It simply passes the input tensor through without any modification.
/// This is often used as a baseline or identity function in neural networks.
/// The Linear node is useful for testing and debugging purposes, as it does not introduce any non-linearity.
/// It can also be used in conjunction with other activation functions to create more complex models.
/// The Linear node supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = x
/// where x is the input tensor.
/// The Linear function is often used in the output layer of neural networks for regression tasks.
/// It is also used in the hidden layers of neural networks when no activation function is applied.
pub const Linear = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new linear node with the given input, weight, and bias nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Linear {
        const ptr = try allocator.create(Linear);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Linear) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the linear function.
    /// The linear function is defined as:
    /// f(x) = x
    /// where x is the input tensor.
    /// The linear function is often used in the output layer of neural networks for regression tasks.
    /// It is also used in the hidden layers of neural networks when no activation function is applied.
    pub fn eval(self: *Linear) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = xv;
        }

        std.debug.print("Linear-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the linear function.
    /// The gradient of the linear function is defined as:
    /// ∂f/∂x = 1
    /// where x is the input tensor.
    /// The gradient of the linear function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Linear, dval: *Tensor) !void {
        try self.x.diff(dval);

        std.debug.print("Linear-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Linear) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this linear node as a generic Node interface.
    pub fn node(self: *Linear) Node {
        return Node.init(self);
    }
};

test "linear basic" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create linear operation
    var linear_op = try graph.linear(x.node());
    defer linear_op.deinit();

    // Evaluate forward pass
    const result = try linear_op.eval();

    // Expected values for each input:
    // f(x) = x
    const expected = [_]f64{
        -2.0, // linear(-2.0)
        -1.0, // linear(-1.0)
        0.0, // linear(0.0)
        1.0, // linear(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "linear gradient" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create linear operation
    var linear_op = try graph.linear(x.node());
    defer linear_op.deinit();

    // First evaluate to cache the values
    _ = try linear_op.eval();

    // Create gradient tensor
    const gradTensor = try graph.tensor(&[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try linear_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = 1
    const expected_grad = [_]f64{
        1.0, // linear'(-2.0)
        1.0, // linear'(-1.0)
        1.0, // linear'(0.0)
        1.0, // linear'(1.0)
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "linear with multiple shapes" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test 1: 2D shape [2, 2]
    {
        // Create input tensor with shape [2, 2]
        const xTensor = try graph.tensor(&[_]usize{ 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -2.0; // [0,0]
        xTensor.data[1] = -1.0; // [0,1]
        xTensor.data[2] = 0.0; // [1,0]
        xTensor.data[3] = 1.0; // [1,1]

        // Create variable
        var x = try graph.variable("x", xTensor);
        defer x.deinit();

        // Create linear operation
        var linear_op = try graph.linear(x.node());
        defer linear_op.deinit();

        // Evaluate forward pass
        const result = try linear_op.eval();

        // Expected values for each input:
        // f(x) = x
        const expected = [_]f64{
            -2.0, // linear(-2.0)
            -1.0, // linear(-1.0)
            0.0, // linear(0.0)
            1.0, // linear(1.0)
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

        try linear_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1
        const expected_grad = [_]f64{
            1.0, // linear'(-2.0)
            1.0, // linear'(-1.0)
            1.0, // linear'(0.0)
            1.0, // linear'(1.0)
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }

    // Test 2: 3D shape [2, 2, 2]
    {
        // Create input tensor with shape [2, 2, 2]
        const xTensor = try graph.tensor(&[_]usize{ 2, 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -2.0; // [0,0,0]
        xTensor.data[1] = -1.0; // [0,0,1]
        xTensor.data[2] = 0.0; // [0,1,0]
        xTensor.data[3] = 1.0; // [0,1,1]
        xTensor.data[4] = -1.5; // [1,0,0]
        xTensor.data[5] = -0.5; // [1,0,1]
        xTensor.data[6] = 0.5; // [1,1,0]
        xTensor.data[7] = 1.5; // [1,1,1]

        // Create variable
        var x = try graph.variable("x", xTensor);
        defer x.deinit();

        // Create linear operation
        var linear_op = try graph.linear(x.node());
        defer linear_op.deinit();

        // Evaluate forward pass
        const result = try linear_op.eval();

        // Expected values for each input:
        // f(x) = x
        const expected = [_]f64{
            -2.0, // linear(-2.0)
            -1.0, // linear(-1.0)
            0.0, // linear(0.0)
            1.0, // linear(1.0)
            -1.5, // linear(-1.5)
            -0.5, // linear(-0.5)
            0.5, // linear(0.5)
            1.5, // linear(1.5)
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

        try linear_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1
        const expected_grad = [_]f64{
            1.0, // linear'(-2.0)
            1.0, // linear'(-1.0)
            1.0, // linear'(0.0)
            1.0, // linear'(1.0)
            1.0, // linear'(-1.5)
            1.0, // linear'(-0.5)
            1.0, // linear'(0.5)
            1.0, // linear'(1.5)
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "linear reset" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create linear operation
    var linear_op = try graph.linear(x.node());
    defer linear_op.deinit();

    // First evaluation
    const result1 = try linear_op.eval();

    // Expected values for each input:
    // f(x) = x
    const expected1 = [_]f64{
        -2.0, // linear(-2.0)
        -1.0, // linear(-1.0)
        0.0, // linear(0.0)
        1.0, // linear(1.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    linear_op.reset();
    const result2 = try linear_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        -2.0, // linear(-2.0)
        -1.0, // linear(-1.0)
        0.0, // linear(0.0)
        1.0, // linear(1.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
