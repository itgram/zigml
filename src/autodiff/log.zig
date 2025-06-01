const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

/// Log function node.
/// The Log node represents the logarithm function applied to a tensor.
/// It computes the logarithm of each element in the input tensor.
/// The Log node is used in neural networks and mathematical computations where the logarithm function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = log10(x)
/// where x is the input tensor.
/// The logarithm function is often used in optimization problems and loss functions.
/// The Log node is typically used in conjunction with other nodes to build complex computation graphs.
pub const Log = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new logarithm node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Log {
        const ptr = try allocator.create(Log);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Log) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the logarithm function.
    /// The logarithm function is defined as:
    /// f(x) = log10(x)
    /// where x is the input tensor.
    /// The logarithm function is often used in optimization problems and loss functions.
    pub fn eval(self: *Log) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.log10(xv);
        }

        std.debug.print("Log-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the logarithm function.
    /// The gradient of the logarithm function is defined as:
    /// ∂f/∂x = 1 / (x * ln(10))
    /// where x is the input tensor.
    /// The gradient of the logarithm function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Log, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv / (xv * math.ln10);
        }

        try self.x.diff(grad);

        std.debug.print("Log-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Log) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this logarithm node as a generic Node interface.
    pub fn node(self: *Log) Node {
        return Node.init(self);
    }
};

test "log basic" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor with test values
    const xTensor = try graph.tensor(&[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = 0.01; // very small positive input
    xTensor.data[1] = 0.1; // small positive input
    xTensor.data[2] = 0.5; // small positive input
    xTensor.data[3] = 1.0; // unit input
    xTensor.data[4] = 10.0; // positive input
    xTensor.data[5] = 100.0; // large positive input

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create log operation
    var log_op = try graph.log(x.node());
    defer log_op.deinit();

    // Evaluate forward pass
    const result = try log_op.eval();

    // Expected values for each input:
    // f(x) = log₁₀(x)
    const expected = [_]f64{
        @as(f64, math.log10(0.01)), // log₁₀(0.01) = -2
        @as(f64, math.log10(0.1)), // log₁₀(0.1) = -1
        @as(f64, math.log10(0.5)), // log₁₀(0.5) ≈ -0.3010
        @as(f64, math.log10(1.0)), // log₁₀(1.0) = 0
        @as(f64, math.log10(10.0)), // log₁₀(10.0) = 1
        @as(f64, math.log10(100.0)), // log₁₀(100.0) = 2
    };

    // Verify results with appropriate tolerance
    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Verify tensor shape is preserved
    try std.testing.expectEqual(@as(usize, 6), result.shape[0]);
}

test "log gradient" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.1; // small positive input
    xTensor.data[1] = 0.5; // small positive input
    xTensor.data[2] = 1.0; // unit input
    xTensor.data[3] = 10.0; // positive input

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create log operation
    var log_op = try graph.log(x.node());
    defer log_op.deinit();

    // First evaluate to cache the values
    _ = try log_op.eval();

    // Create gradient tensor
    const gradTensor = try graph.tensor(&[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try log_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = 1/(x * ln(10))
    const ln10 = math.ln10;
    const expected_grad = [_]f64{
        1.0 / (0.1 * ln10), // log'(0.1)
        1.0 / (0.5 * ln10), // log'(0.5)
        1.0 / (1.0 * ln10), // log'(1.0)
        1.0 / (10.0 * ln10), // log'(10.0)
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "log with multiple shapes" {
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
        xTensor.data[3] = 10.0; // [1,1]

        // Create variable
        var x = try graph.variable("x", xTensor);
        defer x.deinit();

        // Create log operation
        var log_op = try graph.log(x.node());
        defer log_op.deinit();

        // Evaluate forward pass
        const result = try log_op.eval();

        // Expected values for each input:
        // f(x) = log₁₀(x)
        const expected = [_]f64{
            @as(f64, math.log10(0.1)), // log₁₀(0.1)
            @as(f64, math.log10(0.5)), // log₁₀(0.5)
            @as(f64, math.log10(1.0)), // log₁₀(1.0)
            @as(f64, math.log10(10.0)), // log₁₀(10.0)
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

        try log_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1/(x * ln(10))
        const ln10 = math.ln10;
        const expected_grad = [_]f64{
            1.0 / (0.1 * ln10), // log'(0.1)
            1.0 / (0.5 * ln10), // log'(0.5)
            1.0 / (1.0 * ln10), // log'(1.0)
            1.0 / (10.0 * ln10), // log'(10.0)
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
        xTensor.data[3] = 10.0; // [0,1,1]
        xTensor.data[4] = 0.2; // [1,0,0]
        xTensor.data[5] = 0.8; // [1,0,1]
        xTensor.data[6] = 1.5; // [1,1,0]
        xTensor.data[7] = 20.0; // [1,1,1]

        // Create variable
        var x = try graph.variable("x", xTensor);
        defer x.deinit();

        // Create log operation
        var log_op = try graph.log(x.node());
        defer log_op.deinit();

        // Evaluate forward pass
        const result = try log_op.eval();

        // Expected values for each input:
        // f(x) = log₁₀(x)
        const expected = [_]f64{
            @as(f64, math.log10(0.1)), // log₁₀(0.1)
            @as(f64, math.log10(0.5)), // log₁₀(0.5)
            @as(f64, math.log10(1.0)), // log₁₀(1.0)
            @as(f64, math.log10(10.0)), // log₁₀(10.0)
            @as(f64, math.log10(0.2)), // log₁₀(0.2)
            @as(f64, math.log10(0.8)), // log₁₀(0.8)
            @as(f64, math.log10(1.5)), // log₁₀(1.5)
            @as(f64, math.log10(20.0)), // log₁₀(20.0)
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

        try log_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 1/(x * ln(10))
        const ln10 = math.ln10;
        const expected_grad = [_]f64{
            1.0 / (0.1 * ln10), // log'(0.1)
            1.0 / (0.5 * ln10), // log'(0.5)
            1.0 / (1.0 * ln10), // log'(1.0)
            1.0 / (10.0 * ln10), // log'(10.0)
            1.0 / (0.2 * ln10), // log'(0.2)
            1.0 / (0.8 * ln10), // log'(0.8)
            1.0 / (1.5 * ln10), // log'(1.5)
            1.0 / (20.0 * ln10), // log'(20.0)
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "log reset" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.1; // small positive input
    xTensor.data[1] = 0.5; // small positive input
    xTensor.data[2] = 1.0; // unit input
    xTensor.data[3] = 10.0; // positive input

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create log operation
    var log_op = try graph.log(x.node());
    defer log_op.deinit();

    // First evaluation
    const result1 = try log_op.eval();

    // Expected values for each input:
    // f(x) = log₁₀(x)
    const expected1 = [_]f64{
        @as(f64, math.log10(0.1)), // log₁₀(0.1)
        @as(f64, math.log10(0.5)), // log₁₀(0.5)
        @as(f64, math.log10(1.0)), // log₁₀(1.0)
        @as(f64, math.log10(10.0)), // log₁₀(10.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    log_op.reset();
    const result2 = try log_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        @as(f64, math.log10(0.1)), // log₁₀(0.1)
        @as(f64, math.log10(0.5)), // log₁₀(0.5)
        @as(f64, math.log10(1.0)), // log₁₀(1.0)
        @as(f64, math.log10(10.0)), // log₁₀(10.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
