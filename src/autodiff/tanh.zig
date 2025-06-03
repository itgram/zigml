const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

/// Tanh function node.
/// The Tanh (hyperbolic tangent) function.
/// The Tanh function maps any real-valued number to the (-1, 1) interval.
/// The Tanh function is commonly used in neural networks as an activation function.
/// It is particularly useful for hidden layers in neural networks.
/// The Tanh function is differentiable everywhere, making it suitable for backpropagation in neural networks.
/// It is defined as:
/// f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// where e is the base of the natural logarithm.
/// and x is the input tensor.
/// The Tanh function is a smooth, continuous function that is symmetric around the origin.
pub const Tanh = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new tanh node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Tanh {
        const ptr = try allocator.create(Tanh);
        ptr.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
        };
        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Tanh) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the hyperbolic tangent function.
    /// The hyperbolic tangent function is defined as:
    /// f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    /// where e is the base of the natural logarithm.
    /// and x is the input tensor.
    /// The hyperbolic tangent function is a smooth, continuous function that is symmetric around the origin.
    pub fn eval(self: *Tanh) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.tanh(xv);
        }

        std.debug.print("Tanh-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the hyperbolic tangent function.
    /// The gradient of the hyperbolic tangent function is defined as:
    /// ∂f/∂x = 1 - tanh^2(x)
    /// where x is the input tensor.
    /// The gradient of the hyperbolic tangent function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Tanh, dval: *Tensor) !void {
        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, self.value.?.data, dval.data) |*v, vv, dv| {
            v.* = dv * (1 - vv * vv);
        }

        try self.x.diff(grad);

        std.debug.print("Tanh-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Tanh) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this hyperbolic tangent node as a generic Node interface.
    pub fn node(self: *Tanh) Node {
        return Node.init(self);
    }
};

test "tanh basic" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create tanh operation
    var tanh_op = try graph.tanh(x.node());
    defer tanh_op.deinit();

    // First evaluate to cache the values
    const result = try tanh_op.eval();
    const expected = [_]f64{
        @as(f64, -0.9640275800758169), // tanh(-2.0)
        @as(f64, -0.7615941559557649), // tanh(-1.0)
        @as(f64, 0.0), // tanh(0.0)
        @as(f64, 0.7615941559557649), // tanh(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "tanh gradient" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create tanh operation
    var tanh_op = try graph.tanh(x.node());
    defer tanh_op.deinit();

    // First evaluate to cache the values
    const result = try tanh_op.eval();
    const expected = [_]f64{
        @as(f64, -0.9640275800758169), // tanh(-2.0)
        @as(f64, -0.7615941559557649), // tanh(-1.0)
        @as(f64, 0.0), // tanh(0.0)
        @as(f64, 0.7615941559557649), // tanh(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Create gradient tensor
    const gradTensor = try graph.tensor(&[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try tanh_op.diff(gradTensor);

    // Expected gradients: 1 - tanh(x)^2
    const expected_grad = [_]f64{
        @as(f64, 0.07065082485316443), // 1 - tanh(-2.0)^2
        @as(f64, 0.41997434161402614), // 1 - tanh(-1.0)^2
        @as(f64, 1.0), // 1 - tanh(0.0)^2
        @as(f64, 0.41997434161402614), // 1 - tanh(1.0)^2
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "tanh with different shapes" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor
    const xTensor = try graph.tensor(&[_]usize{ 2, 2 });
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create tanh operation
    var tanh_op = try graph.tanh(x.node());
    defer tanh_op.deinit();

    // Evaluate
    const result = try tanh_op.eval();
    const expected = [_]f64{
        @as(f64, -0.9640275800758169), // tanh(-2.0)
        @as(f64, -0.7615941559557649), // tanh(-1.0)
        @as(f64, 0.0), // tanh(0.0)
        @as(f64, 0.7615941559557649), // tanh(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "tanh reset" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensor
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create variable
    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    // Create tanh operation
    var tanh_op = try graph.tanh(x.node());
    defer tanh_op.deinit();

    // First evaluation
    const result1 = try tanh_op.eval();
    const expected1 = [_]f64{
        @as(f64, -0.9640275800758169), // tanh(-2.0)
        @as(f64, -0.7615941559557649), // tanh(-1.0)
        @as(f64, 0.0), // tanh(0.0)
        @as(f64, 0.7615941559557649), // tanh(1.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset
    tanh_op.reset();

    // Second evaluation
    const result2 = try tanh_op.eval();
    const expected2 = [_]f64{
        @as(f64, -0.9640275800758169), // tanh(-2.0)
        @as(f64, -0.7615941559557649), // tanh(-1.0)
        @as(f64, 0.0), // tanh(0.0)
        @as(f64, 0.7615941559557649), // tanh(1.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
