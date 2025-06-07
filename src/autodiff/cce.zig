const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

const epsilon = 1e-12; // Small value to prevent log(0), matching PyTorch's CCE implementation

/// Categorical Cross-Entropy loss function node.
/// The CCE node represents the categorical cross-entropy loss function applied to a tensor.
/// It is defined as:
/// f(x, y) = -mean(sum(y * log(x)))
/// where x is the input tensor (probabilities) and y is the target tensor (one-hot encoded labels).
/// The CCE function is used in multi-class classification problems to measure the difference between predicted probabilities and actual class labels.
/// The CCE function is differentiable and can be used in gradient descent algorithms.
pub const CCE = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node, // probabilities
    y: Node, // one-hot encoded labels

    /// Creates a new CCE node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*CCE {
        const self = try allocator.create(CCE);
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
    pub fn deinit(self: *CCE) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluates the CCE function.
    /// The CCE function is defined as:
    /// f(x, y) = -mean(sum(y * log(x)))
    /// where x is the input tensor (probabilities) and y is the target tensor (one-hot encoded labels).
    pub fn eval(self: *CCE) !*Tensor {
        const x = try self.x.eval();
        const y = try self.y.eval();

        // Validate shapes
        if (!std.mem.eql(usize, x.shape, y.shape)) {
            return error.ShapeMismatch;
        }

        var sum: f64 = 0;
        for (x.data, y.data) |xv, yv| {
            sum += -(yv * @log(xv + epsilon));
        }

        self.value = try Tensor.init(self.allocator, &[_]usize{1});
        // Average over number of samples (first dimension)
        const num_samples = x.shape[0];
        self.value.?.data[0] = sum / @as(f64, @floatFromInt(num_samples));

        return self.value.?;
    }

    /// Compute the gradient of the CCE function.
    /// The gradient of CCE is defined as:
    /// ∂f/∂x = -y/x / n
    /// ∂f/∂y = -log(x) / n
    /// where n is the number of samples.
    pub fn diff(self: *CCE, dval: *Tensor) !void {
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

        const num_samples = x.shape[0];
        const n = @as(f64, @floatFromInt(num_samples));

        for (grad_x.data, grad_y.data, x.data, y.data) |*gx, *gy, xv, yv| {
            // Gradient for x: -y/x / n
            gx.* = -dval.data[0] * (yv / (xv + epsilon)) / n;

            // Gradient for y: -log(x) / n
            gy.* = -dval.data[0] * @log(xv + epsilon) / n;
        }

        try self.x.diff(grad_x);
        try self.y.diff(grad_y);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *CCE) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this CCE node as a generic Node interface.
    pub fn node(self: *CCE) Node {
        return Node.init(self);
    }
};

test "cce basic evaluation" {
    const allocator = std.testing.allocator;

    // Test case: 3 classes, 2 samples
    // x = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]] (valid probability distributions)
    // y = [[1, 0, 0], [0, 1, 0]] (one-hot encoded labels)
    const x_tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.7; // sample 1, class 1
    x_tensor.data[1] = 0.2; // sample 1, class 2
    x_tensor.data[2] = 0.1; // sample 1, class 3
    x_tensor.data[3] = 0.1; // sample 2, class 1
    x_tensor.data[4] = 0.8; // sample 2, class 2
    x_tensor.data[5] = 0.1; // sample 2, class 3

    const y_tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0; // sample 1, class 1
    y_tensor.data[1] = 0.0; // sample 1, class 2
    y_tensor.data[2] = 0.0; // sample 1, class 3
    y_tensor.data[3] = 0.0; // sample 2, class 1
    y_tensor.data[4] = 1.0; // sample 2, class 2
    y_tensor.data[5] = 0.0; // sample 2, class 3

    var x = try Variable.init(allocator, "x", x_tensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", y_tensor);
    defer y.deinit();

    var cce = try CCE.init(allocator, x.node(), y.node());
    defer cce.deinit();

    const result = try cce.eval();
    // Expected: -mean(sum(y * log(x + epsilon)))
    // Sample 1: -(1*log(0.7 + 1e-12) + 0*log(0.2 + 1e-12) + 0*log(0.1 + 1e-12)) = -log(0.7 + 1e-12) ≈ 0.356675
    // Sample 2: -(0*log(0.1 + 1e-12) + 1*log(0.8 + 1e-12) + 0*log(0.1 + 1e-12)) = -log(0.8 + 1e-12) ≈ 0.223144
    // Mean: (0.356675 + 0.223144) / 2 ≈ 0.289909
    try std.testing.expectApproxEqAbs(@as(f64, 0.289909), result.data[0], 1e-6);
}

test "cce gradient computation" {
    const allocator = std.testing.allocator;

    // Test case: 3 classes, 2 samples
    // x = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]] (valid probability distributions)
    // y = [[1, 0, 0], [0, 1, 0]] (one-hot encoded labels)
    const x_tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.7;
    x_tensor.data[1] = 0.2;
    x_tensor.data[2] = 0.1;
    x_tensor.data[3] = 0.1;
    x_tensor.data[4] = 0.8;
    x_tensor.data[5] = 0.1;

    const y_tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0;
    y_tensor.data[1] = 0.0;
    y_tensor.data[2] = 0.0;
    y_tensor.data[3] = 0.0;
    y_tensor.data[4] = 1.0;
    y_tensor.data[5] = 0.0;

    var x = try Variable.init(allocator, "x", x_tensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", y_tensor);
    defer y.deinit();

    var cce = try CCE.init(allocator, x.node(), y.node());
    defer cce.deinit();

    // First compute the forward pass
    const result = try cce.eval();
    try std.testing.expectApproxEqAbs(@as(f64, 0.289909), result.data[0], 1e-6);

    // Reset gradients before computing them
    x.reset();
    y.reset();

    // Then compute gradients
    const df_tensor = try Tensor.init(allocator, &[_]usize{1});
    defer df_tensor.deinit();
    df_tensor.data[0] = 1.0;

    try cce.diff(df_tensor);

    // Check gradients for x
    // ∂f/∂x = -y/(x + epsilon) / n, where n is the number of samples (2)
    try std.testing.expectApproxEqAbs(@as(f64, -0.7142857142857143), x.grad.data[0], 1e-6); // -1/(0.7 + 1e-12)/2
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, -0.6250000000000001), x.grad.data[4], 1e-6); // -1/(0.8 + 1e-12)/2
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[5], 1e-6);

    // Check gradients for y
    // ∂f/∂y = -log(x + epsilon) / n, where n is the number of samples (2)
    try std.testing.expectApproxEqAbs(@as(f64, 0.17833747196936622), y.grad.data[0], 1e-6); // -log(0.7 + 1e-12)/2
    try std.testing.expectApproxEqAbs(@as(f64, 0.8047189562170501), y.grad.data[1], 1e-6); // -log(0.2 + 1e-12)/2
    try std.testing.expectApproxEqAbs(@as(f64, 1.1512925464970228), y.grad.data[2], 1e-6); // -log(0.1 + 1e-12)/2
    try std.testing.expectApproxEqAbs(@as(f64, 1.1512925464970228), y.grad.data[3], 1e-6); // -log(0.1 + 1e-12)/2
    try std.testing.expectApproxEqAbs(@as(f64, 0.11157177565710488), y.grad.data[4], 1e-6); // -log(0.8 + 1e-12)/2
    try std.testing.expectApproxEqAbs(@as(f64, 1.1512925464970228), y.grad.data[5], 1e-6); // -log(0.1 + 1e-12)/2

    // Reset gradients before testing scaling
    x.reset();
    y.reset();

    // Test gradient scaling
    df_tensor.data[0] = 2.0;
    try cce.diff(df_tensor);

    // Gradients should be scaled by 2
    try std.testing.expectApproxEqAbs(@as(f64, -1.4285714285714286), x.grad.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, -1.25), x.grad.data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), x.grad.data[5], 1e-6);

    try std.testing.expectApproxEqAbs(@as(f64, 0.35667494393873244), y.grad.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.6094379124341002), y.grad.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 2.3025850929940456), y.grad.data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 2.3025850929940456), y.grad.data[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.22314355131420976), y.grad.data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 2.3025850929940456), y.grad.data[5], 1e-6);
}

test "cce shape mismatch error" {
    const allocator = std.testing.allocator;

    // Test case: x = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], y = [[1, 0], [0, 1]] (different shapes)
    const x_tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.7;
    x_tensor.data[1] = 0.2;
    x_tensor.data[2] = 0.1;
    x_tensor.data[3] = 0.1;
    x_tensor.data[4] = 0.8;
    x_tensor.data[5] = 0.1;

    const y_tensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0;
    y_tensor.data[1] = 0.0;
    y_tensor.data[2] = 0.0;
    y_tensor.data[3] = 1.0;

    var x = try Variable.init(allocator, "x", x_tensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", y_tensor);
    defer y.deinit();

    var cce = try CCE.init(allocator, x.node(), y.node());
    defer cce.deinit();

    // Should return ShapeMismatch error
    try std.testing.expectError(error.ShapeMismatch, cce.eval());
}

test "cce with perfect prediction" {
    const allocator = std.testing.allocator;

    // Test case: 3 classes, 2 samples with perfect predictions
    // x = [[0.99, 0.005, 0.005], [0.005, 0.99, 0.005]] (valid probability distributions)
    // y = [[1, 0, 0], [0, 1, 0]] (one-hot encoded labels)
    const x_tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.99;
    x_tensor.data[1] = 0.005;
    x_tensor.data[2] = 0.005;
    x_tensor.data[3] = 0.005;
    x_tensor.data[4] = 0.99;
    x_tensor.data[5] = 0.005;

    const y_tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0;
    y_tensor.data[1] = 0.0;
    y_tensor.data[2] = 0.0;
    y_tensor.data[3] = 0.0;
    y_tensor.data[4] = 1.0;
    y_tensor.data[5] = 0.0;

    var x = try Variable.init(allocator, "x", x_tensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", y_tensor);
    defer y.deinit();

    var cce = try CCE.init(allocator, x.node(), y.node());
    defer cce.deinit();

    const result = try cce.eval();
    // Expected: -mean(sum(y * log(x)))
    // Sample 1: -(1*log(0.99) + 0*log(0.005) + 0*log(0.005)) = -log(0.99) ≈ 0.010050
    // Sample 2: -(0*log(0.005) + 1*log(0.99) + 0*log(0.005)) = -log(0.99) ≈ 0.010050
    // Mean: (0.010050 + 0.010050) / 2 ≈ 0.010050
    try std.testing.expectApproxEqAbs(@as(f64, 0.010050), result.data[0], 1e-6);
}

test "cce with uniform distribution" {
    const allocator = std.testing.allocator;

    // Test case: 3 classes, 2 samples with uniform predictions
    // x = [[0.33, 0.33, 0.34], [0.33, 0.34, 0.33]] (valid probability distributions)
    // y = [[1, 0, 0], [0, 1, 0]] (one-hot encoded labels)
    const x_tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer x_tensor.deinit();
    x_tensor.data[0] = 0.33;
    x_tensor.data[1] = 0.33;
    x_tensor.data[2] = 0.34;
    x_tensor.data[3] = 0.33;
    x_tensor.data[4] = 0.34;
    x_tensor.data[5] = 0.33;

    const y_tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0;
    y_tensor.data[1] = 0.0;
    y_tensor.data[2] = 0.0;
    y_tensor.data[3] = 0.0;
    y_tensor.data[4] = 1.0;
    y_tensor.data[5] = 0.0;

    var x = try Variable.init(allocator, "x", x_tensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", y_tensor);
    defer y.deinit();

    var cce = try CCE.init(allocator, x.node(), y.node());
    defer cce.deinit();

    const result = try cce.eval();
    // Expected: -mean(sum(y * log(x)))
    // Sample 1: -(1*log(0.33) + 0*log(0.33) + 0*log(0.34)) = -log(0.33) ≈ 1.108663
    // Sample 2: -(0*log(0.33) + 1*log(0.34) + 0*log(0.33)) = -log(0.34) ≈ 1.078810
    // Mean: (1.108663 + 1.078810) / 2 ≈ 1.093737
    try std.testing.expectApproxEqAbs(@as(f64, 1.093737), result.data[0], 1e-6);
}
