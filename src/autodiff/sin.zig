const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Sin function node.
/// The Sin node represents the sine function applied to a tensor.
/// It computes the sine of each element in the input tensor.
/// The Sin node is used in neural networks and mathematical computations where the sine function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = sin(x)
/// where x is the input tensor.
/// The Sine function is a periodic function that oscillates between -1 and 1.
/// The sine function is often used in trigonometric calculations and periodic functions.
pub const Sin = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new sine node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Sin {
        const self = try allocator.create(Sin);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
        };
        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Sin) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the sine function.
    /// The sine function is defined as:
    /// f(x) = sin(x)
    /// where x is the input tensor.
    /// The sine function is a periodic function that oscillates between -1 and 1.
    /// The sine function is often used in trigonometric calculations and periodic functions.
    pub fn eval(self: *Sin) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = std.math.sin(xv);
        }

        return self.value.?;
    }

    /// Compute the gradient of the sine function.
    /// The gradient of the sine function is defined as:
    /// ∂f/∂x = cos(x)
    /// where x is the input tensor.
    /// The gradient of the sine function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Sin, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv * std.math.cos(xv);
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Sin) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this sine node as a generic Node interface.
    pub fn node(self: *Sin) Node {
        return Node.init(self);
    }
};

test "sin basic" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = -std.math.pi; // -π
    xTensor.data[1] = -std.math.pi / 2.0; // -π/2
    xTensor.data[2] = 0.0; // 0
    xTensor.data[3] = std.math.pi / 2.0; // π/2
    xTensor.data[4] = std.math.pi; // π
    xTensor.data[5] = 2.0 * std.math.pi; // 2π

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create sin operation
    var sin_op = try Sin.init(allocator, x.node());
    defer sin_op.deinit();

    // Evaluate forward pass
    const result = try sin_op.eval();

    // Expected values for each input:
    // f(x) = sin(x)
    const expected = [_]f64{
        std.math.sin(-std.math.pi), // sin(-π) = 0
        std.math.sin(-std.math.pi / 2.0), // sin(-π/2) = -1
        std.math.sin(0.0), // sin(0) = 0
        std.math.sin(std.math.pi / 2.0), // sin(π/2) = 1
        std.math.sin(std.math.pi), // sin(π) = 0
        std.math.sin(2 * std.math.pi), // sin(2π) = 0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Verify tensor shape is preserved
    try std.testing.expectEqual(@as(usize, 6), result.shape[0]);
}

test "sin gradient" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{6});
    defer xTensor.deinit();
    xTensor.data[0] = -std.math.pi; // -π
    xTensor.data[1] = -std.math.pi / 2.0; // -π/2
    xTensor.data[2] = 0.0; // 0
    xTensor.data[3] = std.math.pi / 2.0; // π/2
    xTensor.data[4] = std.math.pi; // π
    xTensor.data[5] = 2.0 * std.math.pi; // 2π

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create sin operation
    var sin_op = try Sin.init(allocator, x.node());
    defer sin_op.deinit();

    // First evaluate to cache the values
    _ = try sin_op.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{6});
    defer gradTensor.deinit();
    for (gradTensor.data) |*v| {
        v.* = 1.0;
    }

    // Compute gradients
    try sin_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = cos(x)
    const expected_grad = [_]f64{
        std.math.cos(-std.math.pi), // cos(-π) = -1
        std.math.cos(-std.math.pi / 2.0), // cos(-π/2) = 0
        std.math.cos(0.0), // cos(0) = 1
        std.math.cos(std.math.pi / 2.0), // cos(π/2) = 0
        std.math.cos(std.math.pi), // cos(π) = -1
        std.math.cos(2.0 * std.math.pi), // cos(2π) = 1
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "sin with multiple shapes" {
    const allocator = std.testing.allocator;

    // Test 1: 2D shape [2, 2]
    {
        // Create input tensor with shape [2, 2]
        const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -std.math.pi / 2.0; // [0,0]
        xTensor.data[1] = 0.0; // [0,1]
        xTensor.data[2] = std.math.pi / 2.0; // [1,0]
        xTensor.data[3] = std.math.pi; // [1,1]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Create sin operation
        var sin_op = try Sin.init(allocator, x.node());
        defer sin_op.deinit();

        // Evaluate forward pass
        const result = try sin_op.eval();

        // Expected values for each input:
        // f(x) = sin(x)
        const expected = [_]f64{
            std.math.sin(-std.math.pi / 2.0), // sin(-π/2) = -1
            std.math.sin(0.0), // sin(0) = 0
            std.math.sin(std.math.pi / 2.0), // sin(π/2) = 1
            std.math.sin(std.math.pi), // sin(π) = 0
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

        try sin_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = cos(x)
        const expected_grad = [_]f64{
            std.math.cos(-std.math.pi / 2.0), // cos(-π/2) = 0
            std.math.cos(0.0), // cos(0) = 1
            std.math.cos(std.math.pi / 2.0), // cos(π/2) = 0
            std.math.cos(std.math.pi), // cos(π) = -1
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
        xTensor.data[0] = -std.math.pi; // [0,0,0]
        xTensor.data[1] = -std.math.pi / 2.0; // [0,0,1]
        xTensor.data[2] = 0.0; // [0,1,0]
        xTensor.data[3] = std.math.pi / 2.0; // [0,1,1]
        xTensor.data[4] = std.math.pi; // [1,0,0]
        xTensor.data[5] = 3.0 * std.math.pi / 2.0; // [1,0,1]
        xTensor.data[6] = 2.0 * std.math.pi; // [1,1,0]
        xTensor.data[7] = 5.0 * std.math.pi / 2.0; // [1,1,1]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Create sin operation
        var sin_op = try Sin.init(allocator, x.node());
        defer sin_op.deinit();

        // Evaluate forward pass
        const result = try sin_op.eval();

        // Expected values for each input:
        // f(x) = sin(x)
        const expected = [_]f64{
            std.math.sin(-std.math.pi), // sin(-π) = 0
            std.math.sin(-std.math.pi / 2.0), // sin(-π/2) = -1
            std.math.sin(0.0), // sin(0) = 0
            std.math.sin(std.math.pi / 2.0), // sin(π/2) = 1
            std.math.sin(std.math.pi), // sin(π) = 0
            std.math.sin(3.0 * std.math.pi / 2.0), // sin(3π/2) = -1
            std.math.sin(2.0 * std.math.pi), // sin(2π) = 0
            std.math.sin(5.0 * std.math.pi / 2.0), // sin(5π/2) = 1
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

        try sin_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = cos(x)
        const expected_grad = [_]f64{
            std.math.cos(-std.math.pi), // cos(-π) = -1
            std.math.cos(-std.math.pi / 2.0), // cos(-π/2) = 0
            std.math.cos(0.0), // cos(0) = 1
            std.math.cos(std.math.pi / 2.0), // cos(π/2) = 0
            std.math.cos(std.math.pi), // cos(π) = -1
            std.math.cos(3.0 * std.math.pi / 2.0), // cos(3π/2) = 0
            std.math.cos(2.0 * std.math.pi), // cos(2π) = 1
            std.math.cos(5.0 * std.math.pi / 2.0), // cos(5π/2) = 0
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "sin reset" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -std.math.pi / 2.0; // -π/2
    xTensor.data[1] = 0.0; // 0
    xTensor.data[2] = std.math.pi / 2.0; // π/2
    xTensor.data[3] = std.math.pi; // π

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create sin operation
    var sin_op = try Sin.init(allocator, x.node());
    defer sin_op.deinit();

    // First evaluation
    const result1 = try sin_op.eval();

    // Expected values for each input:
    // f(x) = sin(x)
    const expected1 = [_]f64{
        std.math.sin(-std.math.pi / 2.0), // sin(-π/2) = -1
        std.math.sin(0.0), // sin(0) = 0
        std.math.sin(std.math.pi / 2.0), // sin(π/2) = 1
        std.math.sin(std.math.pi), // sin(π) = 0
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    sin_op.reset();
    const result2 = try sin_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        std.math.sin(-std.math.pi / 2.0), // sin(-π/2) = -1
        std.math.sin(0.0), // sin(0) = 0
        std.math.sin(std.math.pi / 2.0), // sin(π/2) = 1
        std.math.sin(std.math.pi), // sin(π) = 0
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
