const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

/// Mean Squared Error function node.
/// The MSE node represents the mean squared error function applied to a tensor.
/// It is defined as:
/// f(x, y) = mean((x - y)²)
/// where x is the input tensor and y is the target tensor.
/// The MSE function is used in regression problems to measure the average squared difference between the predicted and actual values.
/// The MSE function is differentiable and can be used in gradient descent algorithms.
pub const MSE = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node, // predictions
    y: Node, // targets

    /// Creates a new MSE node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*MSE {
        const self = try allocator.create(MSE);
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
    pub fn deinit(self: *MSE) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluates the MSE function.
    /// The MSE function is defined as:
    /// f(x, y) = mean((x - y)²)
    /// where x is the input tensor and y is the target tensor.
    /// The MSE function is used in regression problems to measure the average squared difference between the predicted and actual values.
    pub fn eval(self: *MSE) !*Tensor {
        const x = try self.x.eval();
        const y = try self.y.eval();

        // Validate shapes
        if (!std.mem.eql(usize, x.shape, y.shape)) {
            return error.ShapeMismatch;
        }

        var sum: f64 = 0;
        for (x.data, y.data) |xv, yv| {
            const diff_val = xv - yv;
            sum += diff_val * diff_val;
        }

        self.value = try Tensor.init(self.allocator, &[_]usize{1});
        self.value.?.data[0] = sum / @as(f64, @floatFromInt(x.size));

        return self.value.?;
    }

    /// Compute the gradient of the MSE function.
    /// The gradient of MSE is defined as:
    /// ∂f/∂x = 2(x - y) / n
    /// where n is the number of elements in the tensor.
    pub fn diff(self: *MSE, dval: *Tensor) !void {
        const x = try self.x.eval();
        const y = try self.y.eval();

        // Validate shapes
        if (!std.mem.eql(usize, x.shape, y.shape)) {
            return error.ShapeMismatch;
        }

        const grad_x = try Tensor.init(self.allocator, x.shape);
        defer grad_x.deinit();
        const grad_y = try Tensor.init(self.allocator, y.shape);
        defer grad_y.deinit();

        const n = @as(f64, @floatFromInt(x.size));
        for (grad_x.data, grad_y.data, x.data, y.data) |*gx, *gy, xv, yv| {
            const diff_val = xv - yv;
            const scaled_diff = 2.0 * dval.data[0] * diff_val / n;

            gx.* = scaled_diff;
            gy.* = -scaled_diff; // y gradient is opposite of x gradient
        }

        try self.x.diff(grad_x);
        try self.y.diff(grad_y);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *MSE) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this MSE node as a generic Node interface.
    pub fn node(self: *MSE) Node {
        return Node.init(self);
    }
};

test "mse basic evaluation" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [1, 2, 3], y = [2, 2, 2]
    // Expected MSE = ((1-2)² + (2-2)² + (3-2)²) / 3 = (1 + 0 + 1) / 3 = 2/3
    const x_tensor = try graph.tensor(&[_]usize{3});
    defer x_tensor.deinit();
    x_tensor.data[0] = 1.0;
    x_tensor.data[1] = 2.0;
    x_tensor.data[2] = 3.0;

    const y_tensor = try graph.tensor(&[_]usize{3});
    defer y_tensor.deinit();
    y_tensor.data[0] = 2.0;
    y_tensor.data[1] = 2.0;
    y_tensor.data[2] = 2.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var mse = try graph.mse(x.node(), y.node());
    defer mse.deinit();

    const result = try mse.eval();
    try std.testing.expectEqual(@as(f64, 2.0 / 3.0), result.data[0]);
}

test "mse gradient computation" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [1, 2, 3], y = [2, 2, 2]
    // Expected gradients:
    // ∂f/∂x = [2(1-2), 2(2-2), 2(3-2)] / 3 = [-2, 0, 2] / 3
    const x_tensor = try graph.tensor(&[_]usize{3});
    defer x_tensor.deinit();
    x_tensor.data[0] = 1.0;
    x_tensor.data[1] = 2.0;
    x_tensor.data[2] = 3.0;

    const y_tensor = try graph.tensor(&[_]usize{3});
    defer y_tensor.deinit();
    y_tensor.data[0] = 2.0;
    y_tensor.data[1] = 2.0;
    y_tensor.data[2] = 2.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var mse = try graph.mse(x.node(), y.node());
    defer mse.deinit();

    // First compute the forward pass
    const result = try mse.eval();
    try std.testing.expectEqual(@as(f64, 2.0 / 3.0), result.data[0]);

    // Then compute gradients
    const df_tensor = try graph.tensor(&[_]usize{1});
    defer df_tensor.deinit();
    df_tensor.data[0] = 1.0;

    try mse.diff(df_tensor);

    // Check gradients for x
    try std.testing.expectEqual(@as(f64, -2.0 / 3.0), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 0.0), x.grad.data[1]);
    try std.testing.expectEqual(@as(f64, 2.0 / 3.0), x.grad.data[2]);

    // Check gradients for y (should be opposite of x)
    try std.testing.expectEqual(@as(f64, 2.0 / 3.0), y.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 0.0), y.grad.data[1]);
    try std.testing.expectEqual(@as(f64, -2.0 / 3.0), y.grad.data[2]);

    // Reset gradients before testing scaling
    x.reset();
    y.reset();

    // Test gradient scaling
    df_tensor.data[0] = 2.0;
    try mse.diff(df_tensor);

    // Gradients should be scaled by 2
    try std.testing.expectEqual(@as(f64, -4.0 / 3.0), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 0.0), x.grad.data[1]);
    try std.testing.expectEqual(@as(f64, 4.0 / 3.0), x.grad.data[2]);

    try std.testing.expectEqual(@as(f64, 4.0 / 3.0), y.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 0.0), y.grad.data[1]);
    try std.testing.expectEqual(@as(f64, -4.0 / 3.0), y.grad.data[2]);
}

test "mse shape mismatch error" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [1, 2, 3], y = [2, 2] (different shapes)
    const x_tensor = try graph.tensor(&[_]usize{3});
    defer x_tensor.deinit();
    x_tensor.data[0] = 1.0;
    x_tensor.data[1] = 2.0;
    x_tensor.data[2] = 3.0;

    const y_tensor = try graph.tensor(&[_]usize{2});
    defer y_tensor.deinit();
    y_tensor.data[0] = 2.0;
    y_tensor.data[1] = 2.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var mse = try graph.mse(x.node(), y.node());
    defer mse.deinit();

    // Should return ShapeMismatch error
    try std.testing.expectError(error.ShapeMismatch, mse.eval());
}

test "mse with negative values" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [-1, -2, -3], y = [-2, -2, -2]
    // Expected MSE = ((-1-(-2))² + (-2-(-2))² + (-3-(-2))²) / 3 = (1 + 0 + 1) / 3 = 2/3
    const x_tensor = try graph.tensor(&[_]usize{3});
    defer x_tensor.deinit();
    x_tensor.data[0] = -1.0;
    x_tensor.data[1] = -2.0;
    x_tensor.data[2] = -3.0;

    const y_tensor = try graph.tensor(&[_]usize{3});
    defer y_tensor.deinit();
    y_tensor.data[0] = -2.0;
    y_tensor.data[1] = -2.0;
    y_tensor.data[2] = -2.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var mse = try graph.mse(x.node(), y.node());
    defer mse.deinit();

    const result = try mse.eval();
    try std.testing.expectEqual(@as(f64, 2.0 / 3.0), result.data[0]);
}

test "mse with zero values" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [0, 0, 0], y = [0, 0, 0]
    // Expected MSE = 0
    const x_tensor = try graph.tensor(&[_]usize{3});
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.0;
    x_tensor.data[1] = 0.0;
    x_tensor.data[2] = 0.0;

    const y_tensor = try graph.tensor(&[_]usize{3});
    defer y_tensor.deinit();
    y_tensor.data[0] = 0.0;
    y_tensor.data[1] = 0.0;
    y_tensor.data[2] = 0.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var mse = try graph.mse(x.node(), y.node());
    defer mse.deinit();

    const result = try mse.eval();
    try std.testing.expectEqual(@as(f64, 0.0), result.data[0]);
}
