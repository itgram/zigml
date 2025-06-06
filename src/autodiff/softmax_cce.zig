const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

const epsilon = 1e-12; // Small value to prevent log(0), matching PyTorch's CCE implementation

/// Combined Softmax and Categorical Cross-Entropy loss function node.
/// This node fuses the softmax and CCE operations for better performance.
/// It computes log_softmax directly and then applies the CCE loss.
/// The gradient computation is also optimized by combining both operations.
pub const SoftmaxCCE = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node, // logits
    y: Node, // one-hot encoded labels
    axis: usize, // Axis along which to compute the softmax. Default is 0.

    /// Creates a new SoftmaxCCE node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node, axis: usize) !*SoftmaxCCE {
        const self = try allocator.create(SoftmaxCCE);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
            .y = y,
            .axis = axis,
        };
        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    pub fn deinit(self: *SoftmaxCCE) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluates the combined softmax and CCE function.
    /// Computes log_softmax directly and then applies CCE loss.
    pub fn eval(self: *SoftmaxCCE) !*Tensor {
        const x = try self.x.eval();
        const y = try self.y.eval();

        // Validate shapes
        if (!std.mem.eql(usize, x.shape, y.shape)) {
            return error.ShapeMismatch;
        }

        const shape = x.shape;
        const axis = self.axis;
        const axis_dim = shape[axis];
        const outer = Tensor.sizeOf(shape[0..axis]);
        const inner = Tensor.sizeOf(shape[axis + 1 ..]);

        var sum: f64 = 0;
        const num_samples = x.shape[0];

        for (0..outer) |outer_idx| {
            for (0..inner) |inner_idx| {
                const base = outer_idx * axis_dim * inner + inner_idx;

                // 1. Find max for numerical stability
                var maxVal: f64 = x.data[base];
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    if (x.data[idx] > maxVal) maxVal = x.data[idx];
                }

                // 2. Compute sum of exp
                var sumExp: f64 = 0;
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    sumExp += math.exp(x.data[idx] - maxVal);
                }

                // 3. Compute loss directly using log_softmax
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    const log_prob = x.data[idx] - maxVal - @log(sumExp);
                    sum += -(y.data[idx] * log_prob);
                }
            }
        }

        self.value = try Tensor.init(self.allocator, &[_]usize{1});
        self.value.?.data[0] = sum / @as(f64, @floatFromInt(num_samples));

        return self.value.?;
    }

    /// Compute the gradient of the combined function.
    /// The gradient is computed more efficiently by combining both operations.
    pub fn diff(self: *SoftmaxCCE, dval: *Tensor) !void {
        const x = try self.x.eval();
        const y = try self.y.eval();

        // Validate shapes
        if (!std.mem.eql(usize, x.shape, y.shape)) {
            return error.ShapeMismatch;
        }

        const grad_x = try Tensor.init(self.allocator, x.shape);
        defer grad_x.deinit();

        const shape = x.shape;
        const axis = self.axis;
        const axis_dim = shape[axis];
        const outer = Tensor.sizeOf(shape[0..axis]);
        const inner = Tensor.sizeOf(shape[axis + 1 ..]);
        const num_samples = x.shape[0];
        const n = @as(f64, @floatFromInt(num_samples));

        for (0..outer) |outer_idx| {
            for (0..inner) |inner_idx| {
                const base = outer_idx * axis_dim * inner + inner_idx;

                // 1. Find max for numerical stability
                var maxVal: f64 = x.data[base];
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    if (x.data[idx] > maxVal) maxVal = x.data[idx];
                }

                // 2. Compute sum of exp
                var sumExp: f64 = 0;
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    sumExp += math.exp(x.data[idx] - maxVal);
                }

                // 3. Compute gradients directly
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    const exp_val = math.exp(x.data[idx] - maxVal);
                    const softmax_val = exp_val / sumExp;
                    grad_x.data[idx] = dval.data[0] * (softmax_val - y.data[idx]) / n;
                }
            }
        }

        try self.x.diff(grad_x);
    }

    /// Resets the node's state by clearing cached values.
    pub fn reset(self: *SoftmaxCCE) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this SoftmaxCCE node as a generic Node interface.
    pub fn node(self: *SoftmaxCCE) Node {
        return Node.init(self);
    }
};

test "softmax_cce basic evaluation" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: 3 classes, 2 samples
    // x = [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]] (logits)
    // y = [[1, 0, 0], [0, 1, 0]] (one-hot encoded labels)
    const x_tensor = try graph.tensor(&[_]usize{ 2, 3 });
    defer x_tensor.deinit();
    x_tensor.data[0] = 2.0;
    x_tensor.data[1] = 1.0;
    x_tensor.data[2] = 0.0;
    x_tensor.data[3] = 0.0;
    x_tensor.data[4] = 2.0;
    x_tensor.data[5] = 1.0;

    const y_tensor = try graph.tensor(&[_]usize{ 2, 3 });
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0;
    y_tensor.data[1] = 0.0;
    y_tensor.data[2] = 0.0;
    y_tensor.data[3] = 0.0;
    y_tensor.data[4] = 1.0;
    y_tensor.data[5] = 0.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var softmax_cce = try graph.softmax_cce(x.node(), y.node(), 1);
    defer softmax_cce.deinit();

    const result = try softmax_cce.eval();
    // Expected: -mean(sum(y * log_softmax(x)))
    // Sample 1: -(1*log_softmax(2.0) + 0*log_softmax(1.0) + 0*log_softmax(0.0)) ≈ 0.407606
    // Sample 2: -(0*log_softmax(0.0) + 1*log_softmax(2.0) + 0*log_softmax(1.0)) ≈ 0.407606
    // Mean: (0.407606 + 0.407606) / 2 ≈ 0.407606
    try std.testing.expectApproxEqAbs(@as(f64, 0.407606), result.data[0], 1e-6);
}

test "softmax_cce gradient computation" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: 3 classes, 2 samples
    // x = [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]] (logits)
    // y = [[1, 0, 0], [0, 1, 0]] (one-hot encoded labels)
    const x_tensor = try graph.tensor(&[_]usize{ 2, 3 });
    defer x_tensor.deinit();
    x_tensor.data[0] = 2.0;
    x_tensor.data[1] = 1.0;
    x_tensor.data[2] = 0.0;
    x_tensor.data[3] = 0.0;
    x_tensor.data[4] = 2.0;
    x_tensor.data[5] = 1.0;

    const y_tensor = try graph.tensor(&[_]usize{ 2, 3 });
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0;
    y_tensor.data[1] = 0.0;
    y_tensor.data[2] = 0.0;
    y_tensor.data[3] = 0.0;
    y_tensor.data[4] = 1.0;
    y_tensor.data[5] = 0.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var softmax_cce = try graph.softmax_cce(x.node(), y.node(), 1);
    defer softmax_cce.deinit();

    // First compute the forward pass
    const result = try softmax_cce.eval();
    try std.testing.expectApproxEqAbs(@as(f64, 0.407606), result.data[0], 1e-6);

    // Reset gradients before computing them
    x.reset();

    // Then compute gradients
    const df_tensor = try graph.tensor(&[_]usize{1});
    defer df_tensor.deinit();
    df_tensor.data[0] = 1.0;

    try softmax_cce.diff(df_tensor);

    // Check gradients for x
    // First sample gradients
    try std.testing.expectApproxEqAbs(@as(f64, -0.1673795), x.grad.data[0], 1e-6); // First class
    try std.testing.expectApproxEqAbs(@as(f64, 0.122364), x.grad.data[1], 1e-6); // Second class
    try std.testing.expectApproxEqAbs(@as(f64, 0.0450155), x.grad.data[2], 1e-6); // Third class

    // Second sample gradients
    try std.testing.expectApproxEqAbs(@as(f64, 0.0450155), x.grad.data[3], 1e-6); // First class
    try std.testing.expectApproxEqAbs(@as(f64, -0.1673795), x.grad.data[4], 1e-6); // Second class
    try std.testing.expectApproxEqAbs(@as(f64, 0.122364), x.grad.data[5], 1e-6); // Third class
}

test "softmax_cce shape mismatch error" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test case: x = [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]], y = [[1, 0], [0, 1]] (different shapes)
    const x_tensor = try graph.tensor(&[_]usize{ 2, 3 });
    defer x_tensor.deinit();
    x_tensor.data[0] = 2.0;
    x_tensor.data[1] = 1.0;
    x_tensor.data[2] = 0.0;
    x_tensor.data[3] = 0.0;
    x_tensor.data[4] = 2.0;
    x_tensor.data[5] = 1.0;

    const y_tensor = try graph.tensor(&[_]usize{ 2, 2 });
    defer y_tensor.deinit();
    y_tensor.data[0] = 1.0;
    y_tensor.data[1] = 0.0;
    y_tensor.data[2] = 0.0;
    y_tensor.data[3] = 1.0;

    var x = try graph.variable("x", x_tensor);
    defer x.deinit();
    var y = try graph.variable("y", y_tensor);
    defer y.deinit();

    var softmax_cce = try graph.softmax_cce(x.node(), y.node(), 1);
    defer softmax_cce.deinit();

    // Should return ShapeMismatch error
    try std.testing.expectError(error.ShapeMismatch, softmax_cce.eval());
}
