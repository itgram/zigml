const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

/// Divide two nodes
/// where x and y are nodes that evaluate to tensors.
/// The Divide node computes the element-wise division of the tensors produced by its two input nodes.
/// It is used to represent division operations in the computation graph.
/// The Divide node is a fundamental operation in many neural networks and mathematical computations.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Divide node is defined as:
/// f(x, y) = x / y
/// where x is the numerator tensor and y is the denominator tensor.
/// The Divide node is typically used in conjunction with other nodes to build complex computation graphs.
pub const Divide = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    /// Creates a new division node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Divide {
        const self = try allocator.create(Divide);
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
    pub fn deinit(self: *Divide) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the divide function.
    /// The divide function is defined as:
    /// f(x, y) = x / y
    /// where x and y are the input tensors.
    /// The divide function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn eval(self: *Divide) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();
        const y = try self.y.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = xv / yv;
        }

        std.debug.print("Divide-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the divide function.
    /// The gradient of the divide function is defined as:
    /// ∂f/∂x = 1 / y
    /// ∂f/∂y = -x / (y * y)
    /// where x and y are the input tensors.
    /// The gradient of the divide function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Divide, dval: *Tensor) !void {
        const x = try self.x.eval();
        const y = try self.y.eval();

        const grad_x = try Tensor.init(self.allocator, dval.shape);
        defer grad_x.deinit();
        const grad_y = try Tensor.init(self.allocator, dval.shape);
        defer grad_y.deinit();

        for (grad_x.data, grad_y.data, x.data, y.data, dval.data) |*gx, *gy, xv, yv, dv| {
            gx.* = dv / yv;
            gy.* = -dv * xv / (yv * yv);
        }

        try self.x.diff(grad_x);
        try self.y.diff(grad_y);

        std.debug.print("Divide-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Divide) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this division node as a generic Node interface.
    pub fn node(self: *Divide) Node {
        return Node.init(self);
    }
};

test "divide basic" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Numerator and denominator
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 4.0;
    xTensor.data[1] = 9.0;
    xTensor.data[2] = -6.0;
    xTensor.data[3] = 0.0;

    const yTensor = try graph.tensor(&[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 2.0;
    yTensor.data[1] = -3.0;
    yTensor.data[2] = 2.0;
    yTensor.data[3] = 1.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    var div_op = try graph.divide(x.node(), y.node());
    defer div_op.deinit();

    const result = try div_op.eval();
    const expected = [_]f64{ 2.0, -3.0, -3.0, 0.0 };
    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "divide gradient" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    const xTensor = try graph.tensor(&[_]usize{2});
    defer xTensor.deinit();
    xTensor.data[0] = 6.0;
    xTensor.data[1] = 2.0;
    const yTensor = try graph.tensor(&[_]usize{2});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;
    yTensor.data[1] = 4.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    var div_op = try graph.divide(x.node(), y.node());
    defer div_op.deinit();

    const result = try div_op.eval();
    const expected = [_]f64{ 2.0, 0.5 };
    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Gradient wrt output
    const gradTensor = try graph.tensor(&[_]usize{2});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;

    try div_op.diff(gradTensor);

    // dx = dval / y, dy = -dval * x / (y*y)
    const expected_grad_x = [_]f64{ 1.0 / 3.0, 1.0 / 4.0 };
    const expected_grad_y = [_]f64{ -6.0 / (3.0 * 3.0), -2.0 / (4.0 * 4.0) };
    for (x.grad.data, expected_grad_x) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
    for (y.grad.data, expected_grad_y) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "divide edge cases" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Division by negative and zero
    const xTensor = try graph.tensor(&[_]usize{3});
    defer xTensor.deinit();
    xTensor.data[0] = 1.0;
    xTensor.data[1] = -2.0;
    xTensor.data[2] = 0.0;
    const yTensor = try graph.tensor(&[_]usize{3});
    defer yTensor.deinit();
    yTensor.data[0] = -1.0;
    yTensor.data[1] = 0.0;
    yTensor.data[2] = 5.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    var div_op = try graph.divide(x.node(), y.node());
    defer div_op.deinit();

    const result = try div_op.eval();
    // Division by zero will result in +/-inf or nan
    try std.testing.expect(result.data[0] == -1.0);
    try std.testing.expect(std.math.isNan(result.data[1]) or std.math.isInf(result.data[1]));
    try std.testing.expect(result.data[2] == 0.0);
}

test "divide reset" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    const xTensor = try graph.tensor(&[_]usize{2});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;
    xTensor.data[1] = 4.0;
    const yTensor = try graph.tensor(&[_]usize{2});
    defer yTensor.deinit();
    yTensor.data[0] = 2.0;
    yTensor.data[1] = 2.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    var div_op = try graph.divide(x.node(), y.node());
    defer div_op.deinit();

    const result1 = try div_op.eval();
    const expected1 = [_]f64{ 1.0, 2.0 };
    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
    div_op.reset();
    const result2 = try div_op.eval();
    const expected2 = [_]f64{ 1.0, 2.0 };
    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
