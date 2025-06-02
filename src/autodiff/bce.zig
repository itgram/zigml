const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

const epsilon = 1e-7; // Small value to prevent log(0), matching PyTorch's BCE implementation

/// Binary Cross-Entropy loss function node.
/// The BCE node represents the binary cross-entropy loss function applied to a tensor.
/// It is defined as:
/// f(x, y) = -mean(y * log(x) + (1 - y) * log(1 - x))
/// where x is the input tensor (probabilities) and y is the target tensor (binary labels).
/// The BCE function is used in binary classification problems to measure the difference between predicted probabilities and actual binary labels.
/// The BCE function is differentiable and can be used in gradient descent algorithms.
pub const BCE = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node, // probabilities
    y: Node, // binary labels

    /// Creates a new BCE node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*BCE {
        const ptr = try allocator.create(BCE);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.y = y;
        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *BCE) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluates the BCE function.
    /// The BCE function is defined as:
    /// f(x, y) = -mean(y * log(x) + (1 - y) * log(1 - x))
    /// where x is the input tensor (probabilities) and y is the target tensor (binary labels).
    pub fn eval(self: *BCE) !*Tensor {
        const x = try self.x.eval();
        const y = try self.y.eval();

        // Validate shapes
        if (!std.mem.eql(usize, x.shape, y.shape)) {
            return error.ShapeMismatch;
        }

        var sum: f64 = 0;
        for (x.data, y.data) |xv, yv| {
            sum += -(yv * @log(xv + epsilon) + (1.0 - yv) * @log(1.0 - xv + epsilon));
        }

        self.value = try Tensor.init(self.allocator, &[_]usize{1});
        self.value.?.data[0] = sum / @as(f64, @floatFromInt(x.size));

        return self.value.?;
    }

    /// Compute the gradient of the BCE function.
    /// The gradient of BCE is defined as:
    /// ∂f/∂x = -(y/x - (1-y)/(1-x)) / n
    /// ∂f/∂y = -(log(x) - log(1-x)) / n
    /// where n is the number of elements in the tensor.
    pub fn diff(self: *BCE, dval: *Tensor) !void {
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
            // Gradient for x: -(y/x - (1-y)/(1-x)) / n
            gx.* = -dval.data[0] * (yv / (xv + epsilon) - (1.0 - yv) / (1.0 - xv + epsilon)) / n;

            // Gradient for y: -(log(x) - log(1-x)) / n
            gy.* = -dval.data[0] * (@log(xv + epsilon) - @log(1.0 - xv + epsilon)) / n;
        }

        try self.x.diff(grad_x);
        try self.y.diff(grad_y);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *BCE) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this BCE node as a generic Node interface.
    pub fn node(self: *BCE) Node {
        return Node.init(self);
    }
};

test "bce basic evaluation" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [0.5, 0.5, 0.5], y = [0, 0.5, 1]
    // Expected BCE = -mean(y * log(x + epsilon) + (1 - y) * log(1 - x + epsilon))
    // For x=0.5, y=0: -log(0.5 + 1e-7) ≈ 0.693147
    // For x=0.5, y=0.5: -0.5*log(0.5 + 1e-7) - 0.5*log(0.5 + 1e-7) ≈ 0.693147
    // For x=0.5, y=1: -log(0.5 + 1e-7) ≈ 0.693147
    // Mean: (0.693147 + 0.693147 + 0.693147) / 3 ≈ 0.693147
    const x_tensor = try graph.tensor(&[_]usize{3});
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.5;
    x_tensor.data[1] = 0.5;
    x_tensor.data[2] = 0.5;

    const y_tensor = try graph.tensor(&[_]usize{3});
    defer y_tensor.deinit();
    y_tensor.data[0] = 0.0;
    y_tensor.data[1] = 0.5;
    y_tensor.data[2] = 1.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var bce = try graph.bce(x.node(), y.node());
    defer bce.deinit();

    const result = try bce.eval();
    try std.testing.expectApproxEqAbs(@as(f64, 0.6931471805599453), result.data[0], 1e-6);
}

test "bce gradient computation" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [0.5, 0.5, 0.5], y = [0, 0.5, 1]
    // Expected gradients:
    // ∂f/∂x = -(y/(x + epsilon) - (1-y)/(1-x + epsilon)) / n
    // ∂f/∂y = -(log(x + epsilon) - log(1-x + epsilon)) / n
    const x_tensor = try graph.tensor(&[_]usize{3});
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.5;
    x_tensor.data[1] = 0.5;
    x_tensor.data[2] = 0.5;

    const y_tensor = try graph.tensor(&[_]usize{3});
    defer y_tensor.deinit();
    y_tensor.data[0] = 0.0;
    y_tensor.data[1] = 0.5;
    y_tensor.data[2] = 1.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var bce = try graph.bce(x.node(), y.node());
    defer bce.deinit();

    // First compute the forward pass
    const result = try bce.eval();
    try std.testing.expectApproxEqAbs(@as(f64, 0.6931471805599453), result.data[0], 1e-6);

    // Then compute gradients
    const df_tensor = try graph.tensor(&[_]usize{1});
    defer df_tensor.deinit();
    df_tensor.data[0] = 1.0;

    try bce.diff(df_tensor);

    // Check gradients for x
    try std.testing.expectApproxEqAbs(@as(f64, 0.6666666666666666), x.grad.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, -0.6666666666666666), x.grad.data[2], 1e-6);

    // Check gradients for y
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), y.grad.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), y.grad.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), y.grad.data[2], 1e-6);

    // Reset gradients before testing scaling
    x.reset();
    y.reset();

    // Test gradient scaling
    df_tensor.data[0] = 2.0;
    try bce.diff(df_tensor);

    // Gradients should be scaled by 2
    try std.testing.expectApproxEqAbs(@as(f64, 1.3333333333333333), x.grad.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, -1.3333333333333333), x.grad.data[2], 1e-6);

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), y.grad.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), y.grad.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), y.grad.data[2], 1e-6);
}

test "bce shape mismatch error" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [0.5, 0.5, 0.5], y = [0, 0.5] (different shapes)
    const x_tensor = try graph.tensor(&[_]usize{3});
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.5;
    x_tensor.data[1] = 0.5;
    x_tensor.data[2] = 0.5;

    const y_tensor = try graph.tensor(&[_]usize{2});
    defer y_tensor.deinit();
    y_tensor.data[0] = 0.0;
    y_tensor.data[1] = 0.5;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var bce = try graph.bce(x.node(), y.node());
    defer bce.deinit();

    // Should return ShapeMismatch error
    try std.testing.expectError(error.ShapeMismatch, bce.eval());
}

test "bce with extreme values" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [0.999, 0.001, 0.5], y = [1, 0, 0.5]
    // Testing with probabilities very close to 0 and 1
    const x_tensor = try graph.tensor(&[_]usize{3});
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.999;
    x_tensor.data[1] = 0.001;
    x_tensor.data[2] = 0.5;

    const y_tensor = try graph.tensor(&[_]usize{3});
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0;
    y_tensor.data[1] = 0.0;
    y_tensor.data[2] = 0.5;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var bce = try graph.bce(x.node(), y.node());
    defer bce.deinit();

    const result = try bce.eval();
    // Expected: -mean(y * log(x + epsilon) + (1-y) * log(1-x + epsilon))
    // For x=0.999, y=1: -log(0.999 + 1e-7) ≈ 0.0010005
    // For x=0.001, y=0: -log(0.999 + 1e-7) ≈ 0.0010005
    // For x=0.5, y=0.5: -0.5*log(0.5 + 1e-7) - 0.5*log(0.5 + 1e-7) ≈ 0.693147
    // Mean: (0.0010005 + 0.0010005 + 0.693147) / 3 ≈ 0.231716
    try std.testing.expectApproxEqAbs(@as(f64, 0.231716), result.data[0], 1e-6);
}

test "bce with perfect prediction" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [0.99, 0.01], y = [1, 0]
    // Perfect prediction should result in very small loss
    const x_tensor = try graph.tensor(&[_]usize{2});
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.99;
    x_tensor.data[1] = 0.01;

    const y_tensor = try graph.tensor(&[_]usize{2});
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0;
    y_tensor.data[1] = 0.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var bce = try graph.bce(x.node(), y.node());
    defer bce.deinit();

    const result = try bce.eval();
    // Expected: -mean(y * log(x + epsilon) + (1-y) * log(1-x + epsilon))
    // For x=0.99, y=1: -log(0.99 + 1e-7) ≈ 0.010050
    // For x=0.01, y=0: -log(0.99 + 1e-7) ≈ 0.010050
    // Mean: (0.010050 + 0.010050) / 2 ≈ 0.010050
    try std.testing.expectApproxEqAbs(@as(f64, 0.01005033585350145), result.data[0], 1e-6);
}
