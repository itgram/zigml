const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

/// Cos function node.
/// The Cos node represents the cosine function applied to a tensor.
/// It computes the cosine of each element in the input tensor.
/// The Cos node is used in neural networks and mathematical computations where the cosine function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = cos(x)
/// where x is the input tensor.
/// The Cosine function is a periodic function that oscillates between -1 and 1.
/// The cosine function is often used in trigonometric calculations and periodic functions.
pub const Cos = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new cosine node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Cos {
        const self = try allocator.create(Cos);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
        };

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Cos) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the cosine function.
    /// The cosine function is defined as:
    /// f(x) = cos(x)
    /// where x is the input tensor.
    /// The cosine function is a periodic function that oscillates between -1 and 1.
    /// The cosine function is often used in trigonometric calculations and periodic functions.
    pub fn eval(self: *Cos) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.cos(xv);
        }

        return self.value.?;
    }

    /// Compute the gradient of the cosine function.
    /// The gradient of the cosine function is defined as:
    /// ∂f/∂x = -sin(x)
    /// where x is the input tensor.
    /// The gradient of the cosine function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Cos, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = -dv * math.sin(xv);
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Cos) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this cosine node as a generic Node interface.
    pub fn node(self: *Cos) Node {
        return Node.init(self);
    }
};

test "cos basic" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.0;
    xTensor.data[1] = std.math.pi / 4.0;
    xTensor.data[2] = std.math.pi / 2.0;
    xTensor.data[3] = std.math.pi;

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create cos operation
    var cos_op = try graph.cos(x.node());
    defer cos_op.deinit();

    // Evaluate
    const result = try cos_op.eval();
    const expected = [_]f64{
        math.cos(0.0),
        math.cos(std.math.pi / 4.0),
        math.cos(std.math.pi / 2.0),
        math.cos(std.math.pi),
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "cos gradient" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.0;
    xTensor.data[1] = std.math.pi / 4.0;
    xTensor.data[2] = std.math.pi / 2.0;
    xTensor.data[3] = std.math.pi;

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create cos operation
    var cos_op = try graph.cos(x.node());
    defer cos_op.deinit();

    // Create gradient tensor
    const gradTensor = try graph.tensor(&[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try cos_op.diff(gradTensor);

    // Expected gradients
    const expected_grad = [_]f64{
        -math.sin(0.0),
        -math.sin(std.math.pi / 4.0),
        -math.sin(std.math.pi / 2.0),
        -math.sin(std.math.pi),
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "cos with different shapes" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor
    const xTensor = try graph.tensor(&[_]usize{ 2, 2 });
    defer xTensor.deinit();
    xTensor.data[0] = 0.0;
    xTensor.data[1] = std.math.pi / 4.0;
    xTensor.data[2] = std.math.pi / 2.0;
    xTensor.data[3] = std.math.pi;

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create cos operation
    var cos_op = try graph.cos(x.node());
    defer cos_op.deinit();

    // Evaluate
    const result = try cos_op.eval();
    const expected = [_]f64{
        math.cos(0.0),
        math.cos(std.math.pi / 4.0),
        math.cos(std.math.pi / 2.0),
        math.cos(std.math.pi),
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "cos reset" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 0.0;
    xTensor.data[1] = std.math.pi / 4.0;
    xTensor.data[2] = std.math.pi / 2.0;
    xTensor.data[3] = std.math.pi;

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create cos operation
    var cos_op = try graph.cos(x.node());
    defer cos_op.deinit();

    // First evaluation
    const result1 = try cos_op.eval();
    const expected1 = [_]f64{
        math.cos(0.0),
        math.cos(std.math.pi / 4.0),
        math.cos(std.math.pi / 2.0),
        math.cos(std.math.pi),
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset
    cos_op.reset();

    // Second evaluation
    const result2 = try cos_op.eval();
    const expected2 = [_]f64{
        math.cos(0.0),
        math.cos(std.math.pi / 4.0),
        math.cos(std.math.pi / 2.0),
        math.cos(std.math.pi),
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
