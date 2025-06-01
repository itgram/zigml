const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

/// Ln function node.
/// The natural logarithm function, which is the logarithm to the base e.
/// It is defined as the inverse of the exponential function.
/// The Ln node computes the natural logarithm of each element in the input tensor.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Ln node is commonly used in various mathematical computations and neural networks.
/// It is defined as:
/// f(x) = ln(x)
/// where x is the input tensor.
/// The natural logarithm is often used in optimization problems and loss functions.
pub const Ln = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new natural logarithm node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Ln {
        const ptr = try allocator.create(Ln);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Ln) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the natural logarithm function.
    /// The natural logarithm function is defined as:
    /// f(x) = ln(x)
    /// where x is the input tensor.
    /// The natural logarithm function is often used in optimization problems and loss functions.
    pub fn eval(self: *Ln) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = @log(xv);
        }

        std.debug.print("Ln-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the natural logarithm function.
    /// The gradient of the natural logarithm function is defined as:
    /// ∂f/∂x = 1 / x
    /// where x is the input tensor.
    /// The gradient of the natural logarithm function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Ln, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv / xv;
        }

        try self.x.diff(grad);

        std.debug.print("Ln-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Ln) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this natural logarithm node as a generic Node interface.
    pub fn node(self: *Ln) Node {
        return Node.init(self);
    }
};

test "ln basic" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.1; // small positive input
    xTensor.data[1] = 0.5; // small positive input
    xTensor.data[2] = 1.0; // unit input
    xTensor.data[3] = 2.0; // positive input

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create ln operation
    var ln_op = try graph.ln(x.node());
    defer ln_op.deinit();

    // Evaluate forward pass
    const result = try ln_op.eval();

    // Expected values for each input:
    // f(x) = ln(x)
    const expected = [_]f64{
        @as(f64, @log(0.1)), // ln(0.1)
        @as(f64, @log(0.5)), // ln(0.5)
        @as(f64, @log(1.0)), // ln(1.0)
        @as(f64, @log(2.0)), // ln(2.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "ln gradient" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.1; // small positive input
    xTensor.data[1] = 0.5; // small positive input
    xTensor.data[2] = 1.0; // unit input
    xTensor.data[3] = 2.0; // positive input

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create ln operation
    var ln_op = try graph.ln(x.node());
    defer ln_op.deinit();

    // First evaluate to cache the values
    _ = try ln_op.eval();

    // Create gradient tensor
    const gradTensor = try graph.tensor(&[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try ln_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = 1/x
    const expected_grad = [_]f64{
        1.0 / 0.1, // ln'(0.1)
        1.0 / 0.5, // ln'(0.5)
        1.0 / 1.0, // ln'(1.0)
        1.0 / 2.0, // ln'(2.0)
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "ln with multiple shapes" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test 1: 2D shape [2, 2]
    {
        // Create input tensor with shape [2, 2]
        const xTensor = try graph.tensor(&[_]usize{ 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = 0.1; // [0,0]
        xTensor.data[1] = 0.5; // [0,1]
        xTensor.data[2] = 1.0; // [1,0]
        xTensor.data[3] = 2.0; // [1,1]

        // Create variable
        var x = try graph.variable("x", xTensor);
        defer x.deinit();

        // Create ln operation
        var ln_op = try graph.ln(x.node());
        defer ln_op.deinit();

        // Evaluate forward pass
        const result = try ln_op.eval();

        // Expected values for each input:
        // f(x) = ln(x)
        const expected = [_]f64{
            @as(f64, @log(0.1)), // ln(0.1)
            @as(f64, @log(0.5)), // ln(0.5)
            @as(f64, @log(1.0)), // ln(1.0)
            @as(f64, @log(2.0)), // ln(2.0)
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

        try ln_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1/x
        const expected_grad = [_]f64{
            1.0 / 0.1, // ln'(0.1)
            1.0 / 0.5, // ln'(0.5)
            1.0 / 1.0, // ln'(1.0)
            1.0 / 2.0, // ln'(2.0)
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
        xTensor.data[0] = 0.1; // [0,0,0]
        xTensor.data[1] = 0.5; // [0,0,1]
        xTensor.data[2] = 1.0; // [0,1,0]
        xTensor.data[3] = 2.0; // [0,1,1]
        xTensor.data[4] = 0.2; // [1,0,0]
        xTensor.data[5] = 0.8; // [1,0,1]
        xTensor.data[6] = 1.5; // [1,1,0]
        xTensor.data[7] = 3.0; // [1,1,1]

        // Create variable
        var x = try graph.variable("x", xTensor);
        defer x.deinit();

        // Create ln operation
        var ln_op = try graph.ln(x.node());
        defer ln_op.deinit();

        // Evaluate forward pass
        const result = try ln_op.eval();

        // Expected values for each input:
        // f(x) = ln(x)
        const expected = [_]f64{
            @as(f64, @log(0.1)), // ln(0.1)
            @as(f64, @log(0.5)), // ln(0.5)
            @as(f64, @log(1.0)), // ln(1.0)
            @as(f64, @log(2.0)), // ln(2.0)
            @as(f64, @log(0.2)), // ln(0.2)
            @as(f64, @log(0.8)), // ln(0.8)
            @as(f64, @log(1.5)), // ln(1.5)
            @as(f64, @log(3.0)), // ln(3.0)
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

        try ln_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1/x
        const expected_grad = [_]f64{
            1.0 / 0.1, // ln'(0.1)
            1.0 / 0.5, // ln'(0.5)
            1.0 / 1.0, // ln'(1.0)
            1.0 / 2.0, // ln'(2.0)
            1.0 / 0.2, // ln'(0.2)
            1.0 / 0.8, // ln'(0.8)
            1.0 / 1.5, // ln'(1.5)
            1.0 / 3.0, // ln'(3.0)
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "ln reset" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.1; // small positive input
    xTensor.data[1] = 0.5; // small positive input
    xTensor.data[2] = 1.0; // unit input
    xTensor.data[3] = 2.0; // positive input

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create ln operation
    var ln_op = try graph.ln(x.node());
    defer ln_op.deinit();

    // First evaluation
    const result1 = try ln_op.eval();

    // Expected values for each input:
    // f(x) = ln(x)
    const expected1 = [_]f64{
        @as(f64, @log(0.1)), // ln(0.1)
        @as(f64, @log(0.5)), // ln(0.5)
        @as(f64, @log(1.0)), // ln(1.0)
        @as(f64, @log(2.0)), // ln(2.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    ln_op.reset();
    const result2 = try ln_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        @as(f64, @log(0.1)), // ln(0.1)
        @as(f64, @log(0.5)), // ln(0.5)
        @as(f64, @log(1.0)), // ln(1.0)
        @as(f64, @log(2.0)), // ln(2.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
