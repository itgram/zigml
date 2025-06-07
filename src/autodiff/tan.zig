const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Tan function node.
/// The Tan node represents the tangent function applied to a tensor.
/// It computes the tangent of each element in the input tensor.
/// The Tan node is used in neural networks and mathematical computations where the tangent function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = tan(x)
/// where x is the input tensor.
/// The tangent function is a periodic function that oscillates between -∞ and +∞.
/// The tangent function is often used in trigonometric calculations and periodic functions.
pub const Tan = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new tangent node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Tan {
        const self = try allocator.create(Tan);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
        };

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Tan) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the tangent function.
    /// The tangent function is defined as:
    /// f(x) = tan(x)
    /// where x is the input tensor.
    /// The tangent function is a periodic function that oscillates between -∞ and +∞.
    /// The tangent function is often used in trigonometric calculations and periodic functions.
    pub fn eval(self: *Tan) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = std.math.tan(xv);
        }

        return self.value.?;
    }

    /// Compute the gradient of the tangent function.
    /// The gradient of the tangent function is defined as:
    /// ∂f/∂x = sec^2(x)
    /// where x is the input tensor.
    /// The gradient of the tangent function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Tan, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            const sec2 = 1.0 / std.math.cos(xv);
            v.* = dv * sec2 * sec2;
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Tan) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this tangent node as a generic Node interface.
    pub fn node(self: *Tan) Node {
        return Node.init(self);
    }
};

test "tan basic" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.0;
    xTensor.data[1] = std.math.pi / 4.0;
    xTensor.data[2] = std.math.pi / 3.0;
    xTensor.data[3] = std.math.pi / 6.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create tan operation
    var tan_op = try Tan.init(allocator, x.node());
    defer tan_op.deinit();

    // Evaluate
    const result = try tan_op.eval();
    const expected = [_]f64{
        std.math.tan(0.0),
        std.math.tan(std.math.pi / 4.0),
        std.math.tan(std.math.pi / 3.0),
        std.math.tan(std.math.pi / 6.0),
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "tan gradient" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.0;
    xTensor.data[1] = std.math.pi / 4.0;
    xTensor.data[2] = std.math.pi / 3.0;
    xTensor.data[3] = std.math.pi / 6.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create tan operation
    var tan_op = try Tan.init(allocator, x.node());
    defer tan_op.deinit();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try tan_op.diff(gradTensor);

    // Expected gradients: 1 / cos^2(x)
    const expected_grad = [_]f64{
        1.0 / (std.math.cos(0.0) * std.math.cos(0.0)),
        1.0 / (std.math.cos(std.math.pi / 4.0) * std.math.cos(std.math.pi / 4.0)),
        1.0 / (std.math.cos(std.math.pi / 3.0) * std.math.cos(std.math.pi / 3.0)),
        1.0 / (std.math.cos(std.math.pi / 6.0) * std.math.cos(std.math.pi / 6.0)),
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "tan with different shapes" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer xTensor.deinit();
    xTensor.data[0] = 0.0;
    xTensor.data[1] = std.math.pi / 4.0;
    xTensor.data[2] = std.math.pi / 3.0;
    xTensor.data[3] = std.math.pi / 6.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create tan operation
    var tan_op = try Tan.init(allocator, x.node());
    defer tan_op.deinit();

    // Evaluate
    const result = try tan_op.eval();
    const expected = [_]f64{
        std.math.tan(0.0),
        std.math.tan(std.math.pi / 4.0),
        std.math.tan(std.math.pi / 3.0),
        std.math.tan(std.math.pi / 6.0),
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "tan reset" {
    const allocator = std.testing.allocator;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.0;
    xTensor.data[1] = std.math.pi / 4.0;
    xTensor.data[2] = std.math.pi / 3.0;
    xTensor.data[3] = std.math.pi / 6.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create tan operation
    var tan_op = try Tan.init(allocator, x.node());
    defer tan_op.deinit();

    // First evaluation
    const result1 = try tan_op.eval();
    const expected1 = [_]f64{
        std.math.tan(0.0),
        std.math.tan(std.math.pi / 4.0),
        std.math.tan(std.math.pi / 3.0),
        std.math.tan(std.math.pi / 6.0),
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset
    tan_op.reset();

    // Second evaluation
    const result2 = try tan_op.eval();
    const expected2 = [_]f64{
        std.math.tan(0.0),
        std.math.tan(std.math.pi / 4.0),
        std.math.tan(std.math.pi / 3.0),
        std.math.tan(std.math.pi / 6.0),
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
