const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Softmax function node.
/// The Softmax function is commonly used in neural networks, especially in the output layer for multi-class classification tasks.
/// It converts a vector of real numbers into a probability distribution, where the sum of the probabilities is 1.
/// The Softmax function is often used in conjunction with the cross-entropy loss function for training neural networks.
/// The Softmax function is differentiable, making it suitable for backpropagation in neural networks.
/// It is defined as:
/// f(x_i) = exp(x_i) / sum(exp(x_j)) for j in [1, ..., n]
/// where x_i is the i-th element of the input tensor and n is the number of elements in the tensor.
pub const Softmax = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    axis: usize, // Axis along which to compute the softmax. Default is 0.

    /// Creates a new softmax node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node, axis: usize) !*Softmax {
        const self = try allocator.create(Softmax);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
            .axis = axis,
        };
        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Softmax) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the softmax function.
    /// The softmax function is defined as:
    /// f(x_i) = exp(x_i) / sum(exp(x_j)) for j in [1, ..., n]
    /// where x_i is the i-th element of the input tensor and n is the number of elements in the tensor.
    pub fn eval(self: *Softmax) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        const shape = x.shape;
        const axis = self.axis;
        const axis_dim = shape[axis];
        const outer = Tensor.sizeOf(shape[0..axis]);
        const inner = Tensor.sizeOf(shape[axis + 1 ..]);

        for (0..outer) |outer_idx| {
            for (0..inner) |inner_idx| {
                // Compute the base offset for this slice
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
                    sumExp += std.math.exp(x.data[idx] - maxVal);
                }

                // 3. Write softmax values
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    self.value.?.data[idx] = std.math.exp(x.data[idx] - maxVal) / sumExp;
                }
            }
        }

        return self.value.?;
    }

    /// Compute the gradient of the softmax function.
    /// The gradient of the softmax function is defined as:
    /// ∂Si / ∂Xj =
    ///  Si  * (1 - Si),   if i = j
    ///  -Si * Sj,         if i ≠ j
    /// Where:
    /// - S is the softmax output.
    pub fn diff(self: *Softmax, dval: *Tensor) !void {
        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        const shape = dval.shape;
        const axis = self.axis;
        const axis_dim = shape[axis];
        const outer = Tensor.sizeOf(shape[0..axis]);
        const inner = Tensor.sizeOf(shape[axis + 1 ..]);

        for (0..outer) |outer_idx| {
            for (0..inner) |inner_idx| {
                const base = outer_idx * axis_dim * inner + inner_idx;

                // Compute gradient for this slice
                for (0..axis_dim) |i| {
                    const idx_i = base + i * inner;
                    grad.data[idx_i] = 0;
                    const Si = self.value.?.data[idx_i];
                    const dvi = dval.data[idx_i];

                    for (0..axis_dim) |j| {
                        const idx_j = base + j * inner;
                        const Sj = self.value.?.data[idx_j];
                        if (i == j) {
                            grad.data[idx_i] += dvi * Si * (1 - Si);
                        } else {
                            grad.data[idx_i] -= dvi * Si * Sj;
                        }
                    }
                }
            }
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Softmax) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this softmax node as a generic Node interface.
    pub fn node(self: *Softmax) Node {
        return Node.init(self);
    }
};

test "softmax basic" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0; // high value
    xTensor.data[1] = 1.0; // medium value
    xTensor.data[2] = 0.0; // zero value
    xTensor.data[3] = -1.0; // negative value

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create softmax operation along axis 0
    var softmax_op = try Softmax.init(allocator, x.node(), 0);
    defer softmax_op.deinit();

    // Evaluate forward pass
    const result = try softmax_op.eval();

    // Compute expected values manually
    const exp_vals = [_]f64{
        std.math.exp(2.0),
        std.math.exp(1.0),
        std.math.exp(0.0),
        std.math.exp(-1.0),
    };
    var sum_exp: f64 = 0;
    for (exp_vals) |v| {
        sum_exp += v;
    }

    const expected = [_]f64{
        exp_vals[0] / sum_exp, // exp(2.0) / sum
        exp_vals[1] / sum_exp, // exp(1.0) / sum
        exp_vals[2] / sum_exp, // exp(0.0) / sum
        exp_vals[3] / sum_exp, // exp(-1.0) / sum
    };

    // Verify results
    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Verify sum of probabilities is 1
    var sum: f64 = 0;
    for (result.data) |v| {
        sum += v;
    }
    try std.testing.expectApproxEqAbs(1.0, sum, 1e-6);

    // Verify tensor shape is preserved
    try std.testing.expectEqual(@as(usize, 4), result.shape[0]);
}

test "softmax gradient" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{3});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0; // high value
    xTensor.data[1] = 1.0; // medium value
    xTensor.data[2] = 0.0; // zero value

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create softmax operation
    var softmax_op = try Softmax.init(allocator, x.node(), 0);
    defer softmax_op.deinit();

    // First evaluate to cache the values
    const result = try softmax_op.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{3});
    defer gradTensor.deinit();
    for (gradTensor.data) |*v| {
        v.* = 1.0;
    }

    // Compute gradients
    try softmax_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂Si/∂Xj = Si * (1 - Si) if i = j, else -Si * Sj
    const S = result.data;
    const expected_grad = [_]f64{
        S[0] * (1 - S[0]) - S[0] * S[1] - S[0] * S[2], // ∂S0/∂X0 - ∂S1/∂X0 - ∂S2/∂X0
        -S[1] * S[0] + S[1] * (1 - S[1]) - S[1] * S[2], // ∂S0/∂X1 + ∂S1/∂X1 - ∂S2/∂X1
        -S[2] * S[0] - S[2] * S[1] + S[2] * (1 - S[2]), // ∂S0/∂X2 + ∂S1/∂X2 + ∂S2/∂X2
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "softmax with multiple shapes" {
    const allocator = std.testing.allocator;

    // Test 1: 2D shape [2, 3] with axis 0
    {
        // Create input tensor with shape [2, 3]
        const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
        defer xTensor.deinit();
        xTensor.data[0] = 2.0; // [0,0]
        xTensor.data[1] = 1.0; // [0,1]
        xTensor.data[2] = 0.0; // [0,2]
        xTensor.data[3] = 1.0; // [1,0]
        xTensor.data[4] = 0.0; // [1,1]
        xTensor.data[5] = -1.0; // [1,2]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Create softmax operation along axis 0
        var softmax_op = try Softmax.init(allocator, x.node(), 0);
        defer softmax_op.deinit();

        // Evaluate forward pass
        const result = try softmax_op.eval();

        // Compute expected values manually for each column
        const exp_vals = [_]f64{
            std.math.exp(2.0), std.math.exp(1.0), std.math.exp(0.0),
            std.math.exp(1.0), std.math.exp(0.0), std.math.exp(-1.0),
        };
        var sum_exp = [_]f64{0} ** 3;
        for (0..2) |i| {
            for (0..3) |j| {
                sum_exp[j] += exp_vals[i * 3 + j];
            }
        }

        const expected = [_]f64{
            exp_vals[0] / sum_exp[0], // exp(2.0) / sum
            exp_vals[1] / sum_exp[1], // exp(1.0) / sum
            exp_vals[2] / sum_exp[2], // exp(0.0) / sum
            exp_vals[3] / sum_exp[0], // exp(1.0) / sum
            exp_vals[4] / sum_exp[1], // exp(0.0) / sum
            exp_vals[5] / sum_exp[2], // exp(-1.0) / sum
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Verify sum of probabilities is 1 for each column
        for (0..3) |j| {
            var sum: f64 = 0;
            for (0..2) |i| {
                sum += result.data[i * 3 + j];
            }
            try std.testing.expectApproxEqAbs(1.0, sum, 1e-6);
        }
    }

    // Test 2: 2D shape [2, 3] with axis 1
    {
        // Create input tensor with shape [2, 3]
        const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
        defer xTensor.deinit();
        xTensor.data[0] = 2.0; // [0,0]
        xTensor.data[1] = 1.0; // [0,1]
        xTensor.data[2] = 0.0; // [0,2]
        xTensor.data[3] = 1.0; // [1,0]
        xTensor.data[4] = 0.0; // [1,1]
        xTensor.data[5] = -1.0; // [1,2]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Create softmax operation along axis 1
        var softmax_op = try Softmax.init(allocator, x.node(), 1);
        defer softmax_op.deinit();

        // Evaluate forward pass
        const result = try softmax_op.eval();

        // Compute expected values manually for each row
        const exp_vals = [_]f64{
            std.math.exp(2.0), std.math.exp(1.0), std.math.exp(0.0),
            std.math.exp(1.0), std.math.exp(0.0), std.math.exp(-1.0),
        };
        var sum_exp = [_]f64{0} ** 2;
        for (0..2) |i| {
            for (0..3) |j| {
                sum_exp[i] += exp_vals[i * 3 + j];
            }
        }

        const expected = [_]f64{
            exp_vals[0] / sum_exp[0], // exp(2.0) / sum
            exp_vals[1] / sum_exp[0], // exp(1.0) / sum
            exp_vals[2] / sum_exp[0], // exp(0.0) / sum
            exp_vals[3] / sum_exp[1], // exp(1.0) / sum
            exp_vals[4] / sum_exp[1], // exp(0.0) / sum
            exp_vals[5] / sum_exp[1], // exp(-1.0) / sum
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Verify sum of probabilities is 1 for each row
        for (0..2) |i| {
            var sum: f64 = 0;
            for (0..3) |j| {
                sum += result.data[i * 3 + j];
            }
            try std.testing.expectApproxEqAbs(1.0, sum, 1e-6);
        }
    }
}

test "softmax reset" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{3});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0; // high value
    xTensor.data[1] = 1.0; // medium value
    xTensor.data[2] = 0.0; // zero value

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create softmax operation
    var softmax_op = try Softmax.init(allocator, x.node(), 0);
    defer softmax_op.deinit();

    // First evaluation
    const result1 = try softmax_op.eval();

    // Compute expected values manually
    const exp_vals = [_]f64{
        std.math.exp(2.0),
        std.math.exp(1.0),
        std.math.exp(0.0),
    };
    var sum_exp: f64 = 0;
    for (exp_vals) |v| {
        sum_exp += v;
    }

    const expected1 = [_]f64{
        exp_vals[0] / sum_exp, // exp(2.0) / sum
        exp_vals[1] / sum_exp, // exp(1.0) / sum
        exp_vals[2] / sum_exp, // exp(0.0) / sum
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    softmax_op.reset();
    const result2 = try softmax_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        exp_vals[0] / sum_exp, // exp(2.0) / sum
        exp_vals[1] / sum_exp, // exp(1.0) / sum
        exp_vals[2] / sum_exp, // exp(0.0) / sum
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "softmax numerical stability" {
    const allocator = std.testing.allocator;

    // Create input tensor with large values that could cause overflow
    const xTensor = try Tensor.init(allocator, &[_]usize{3});
    defer xTensor.deinit();
    xTensor.data[0] = 1000.0; // very large value
    xTensor.data[1] = 1001.0; // very large value
    xTensor.data[2] = 1002.0; // very large value

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create softmax operation
    var softmax_op = try Softmax.init(allocator, x.node(), 0);
    defer softmax_op.deinit();

    // Evaluate forward pass
    const result = try softmax_op.eval();

    // Verify results are valid probabilities
    for (result.data) |v| {
        try std.testing.expect(v >= 0.0 and v <= 1.0);
    }

    // Verify sum of probabilities is 1
    var sum: f64 = 0;
    for (result.data) |v| {
        sum += v;
    }
    try std.testing.expectApproxEqAbs(1.0, sum, 1e-6);

    // Verify the largest input has the highest probability
    try std.testing.expect(result.data[2] > result.data[1]);
    try std.testing.expect(result.data[1] > result.data[0]);
}
