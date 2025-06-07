const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Variable = @import("variable.zig").Variable;

/// Exp function node.
/// The Exp node represents the exponential function applied to a tensor.
/// It computes the exponential of each element in the input tensor.
/// The Exp node is used in neural networks and mathematical computations where the exponential function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = e^x
/// where x is the input tensor.
/// The exponential function is often used in activation functions and probability distributions.
pub const Exp = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new exponential node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Exp {
        const self = try allocator.create(Exp);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
        };

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Exp) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the exponential function.
    /// The exponential function is defined as:
    /// f(x) = e^x
    /// where x is the input tensor.
    /// The exponential function is often used in activation functions and probability distributions.
    pub fn eval(self: *Exp) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.exp(xv);
        }

        return self.value.?;
    }

    /// Compute the gradient of the exponential function.
    /// The gradient of the exponential function is defined as:
    /// ∂f/∂x = e^x
    /// where x is the input tensor.
    /// The gradient of the exponential function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Exp, dval: *Tensor) !void {
        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, self.value.?.data, dval.data) |*v, vv, dv| {
            v.* = dv * vv;
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Exp) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this exponential node as a generic Node interface.
    pub fn node(self: *Exp) Node {
        return Node.init(self);
    }
};

test "exp basic" {
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

    // Create exp operation
    var exp_op = try Exp.init(allocator, x.node());
    defer exp_op.deinit();

    // Evaluate forward pass
    const result = try exp_op.eval();

    // Expected values for each input:
    // f(x) = e^x
    const expected = [_]f64{
        @as(f64, math.exp(-2.0)), // exp(-2.0)
        @as(f64, math.exp(-1.0)), // exp(-1.0)
        @as(f64, math.exp(0.0)), // exp(0.0)
        @as(f64, math.exp(1.0)), // exp(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "exp gradient" {
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

    // Create exp operation
    var exp_op = try Exp.init(allocator, x.node());
    defer exp_op.deinit();

    // First evaluate to cache the values
    _ = try exp_op.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try exp_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = e^x
    const expected_grad = [_]f64{
        @as(f64, math.exp(-2.0)), // exp'(-2.0)
        @as(f64, math.exp(-1.0)), // exp'(-1.0)
        @as(f64, math.exp(0.0)), // exp'(0.0)
        @as(f64, math.exp(1.0)), // exp'(1.0)
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "exp with multiple shapes" {
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

        // Create exp operation
        var exp_op = try Exp.init(allocator, x.node());
        defer exp_op.deinit();

        // Evaluate forward pass
        const result = try exp_op.eval();

        // Expected values for each input:
        // f(x) = e^x
        const expected = [_]f64{
            @as(f64, math.exp(-2.0)), // exp(-2.0)
            @as(f64, math.exp(-1.0)), // exp(-1.0)
            @as(f64, math.exp(0.0)), // exp(0.0)
            @as(f64, math.exp(1.0)), // exp(1.0)
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Test gradient computation
        const gradTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer gradTensor.deinit();
        gradTensor.data[0] = 1.0;
        gradTensor.data[1] = 1.0;
        gradTensor.data[2] = 1.0;
        gradTensor.data[3] = 1.0;

        try exp_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = e^x
        const expected_grad = [_]f64{
            @as(f64, math.exp(-2.0)), // exp'(-2.0)
            @as(f64, math.exp(-1.0)), // exp'(-1.0)
            @as(f64, math.exp(0.0)), // exp'(0.0)
            @as(f64, math.exp(1.0)), // exp'(1.0)
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
        xTensor.data[4] = -1.5; // [1,0,0]
        xTensor.data[5] = -0.5; // [1,0,1]
        xTensor.data[6] = 0.5; // [1,1,0]
        xTensor.data[7] = 1.5; // [1,1,1]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Create exp operation
        var exp_op = try Exp.init(allocator, x.node());
        defer exp_op.deinit();

        // Evaluate forward pass
        const result = try exp_op.eval();

        // Expected values for each input:
        // f(x) = e^x
        const expected = [_]f64{
            @as(f64, math.exp(-2.0)), // exp(-2.0)
            @as(f64, math.exp(-1.0)), // exp(-1.0)
            @as(f64, math.exp(0.0)), // exp(0.0)
            @as(f64, math.exp(1.0)), // exp(1.0)
            @as(f64, math.exp(-1.5)), // exp(-1.5)
            @as(f64, math.exp(-0.5)), // exp(-0.5)
            @as(f64, math.exp(0.5)), // exp(0.5)
            @as(f64, math.exp(1.5)), // exp(1.5)
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

        try exp_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = e^x
        const expected_grad = [_]f64{
            @as(f64, math.exp(-2.0)), // exp'(-2.0)
            @as(f64, math.exp(-1.0)), // exp'(-1.0)
            @as(f64, math.exp(0.0)), // exp'(0.0)
            @as(f64, math.exp(1.0)), // exp'(1.0)
            @as(f64, math.exp(-1.5)), // exp'(-1.5)
            @as(f64, math.exp(-0.5)), // exp'(-0.5)
            @as(f64, math.exp(0.5)), // exp'(0.5)
            @as(f64, math.exp(1.5)), // exp'(1.5)
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "exp reset" {
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

    // Create exp operation
    var exp_op = try Exp.init(allocator, x.node());
    defer exp_op.deinit();

    // First evaluation
    const result1 = try exp_op.eval();

    // Expected values for each input:
    // f(x) = e^x
    const expected1 = [_]f64{
        @as(f64, math.exp(-2.0)), // exp(-2.0)
        @as(f64, math.exp(-1.0)), // exp(-1.0)
        @as(f64, math.exp(0.0)), // exp(0.0)
        @as(f64, math.exp(1.0)), // exp(1.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    exp_op.reset();
    const result2 = try exp_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        @as(f64, math.exp(-2.0)), // exp(-2.0)
        @as(f64, math.exp(-1.0)), // exp(-1.0)
        @as(f64, math.exp(0.0)), // exp(0.0)
        @as(f64, math.exp(1.0)), // exp(1.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
