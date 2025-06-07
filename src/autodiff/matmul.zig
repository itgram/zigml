const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Matrix multiplication node.
/// The MatMul node represents the matrix multiplication of two tensors.
/// It computes the product of two matrices following the rules of matrix multiplication.
/// The MatMul node is used in various neural network operations where matrix multiplication is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The MatMul node is defined as:
/// C = X @ Y
/// where X and Y are the input matrices.
/// The MatMul node is typically used in conjunction with other nodes to build complex computation graphs.
pub const MatMul = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    /// Creates a new matrix multiplication node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*MatMul {
        const self = try allocator.create(MatMul);
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
    pub fn deinit(self: *MatMul) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the matrix multiplication.
    /// The matrix multiplication is defined as:
    /// C = X @ Y
    /// where X and Y are the input matrices.
    ///
    /// TODO: Optimize matrix multiplication using:
    /// 1. Strassen's algorithm for large matrices (O(n^2.807) complexity)
    /// 2. Block-based (tiled) multiplication for better cache utilization
    /// 3. SIMD instructions for parallel computation
    /// 4. Loop unrolling for better instruction pipelining
    /// 5. Memory access pattern optimization
    pub fn eval(self: *MatMul) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();
        const y = try self.y.eval();

        // Validate shapes for matrix multiplication
        if (x.shape.len != 2 or y.shape.len != 2) {
            return error.InvalidShape;
        }
        if (x.shape[1] != y.shape[0]) {
            return error.ShapeMismatch;
        }

        // Create output tensor with shape [x.shape[0], y.shape[1]]
        const output_shape = [_]usize{ x.shape[0], y.shape[1] };
        self.value = try Tensor.init(self.allocator, &output_shape);

        // Perform matrix multiplication
        const m = x.shape[0];
        const n = y.shape[1];
        const k = x.shape[1];

        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f64 = 0;
                for (0..k) |l| {
                    sum += x.data[i * k + l] * y.data[l * n + j];
                }
                self.value.?.data[i * n + j] = sum;
            }
        }

        return self.value.?;
    }

    /// Compute the gradient of the matrix multiplication.
    /// The gradient of matrix multiplication is defined as:
    /// ∂C/∂X = dC @ Y^T
    /// ∂C/∂Y = X^T @ dC
    /// where dC is the gradient of the output.
    pub fn diff(self: *MatMul, dval: *Tensor) !void {
        const x = try self.x.eval();
        const y = try self.y.eval();

        // Compute gradients
        const grad_x = try Tensor.init(self.allocator, x.shape);
        defer grad_x.deinit();
        const grad_y = try Tensor.init(self.allocator, y.shape);
        defer grad_y.deinit();

        const m = x.shape[0];
        const n = y.shape[1];
        const k = x.shape[1];

        // Compute ∂C/∂X = dC @ Y^T
        for (0..m) |i| {
            for (0..k) |l| {
                var sum: f64 = 0;
                for (0..n) |j| {
                    sum += dval.data[i * n + j] * y.data[l * n + j];
                }
                grad_x.data[i * k + l] = sum;
            }
        }

        // Compute ∂C/∂Y = X^T @ dC
        for (0..k) |l| {
            for (0..n) |j| {
                var sum: f64 = 0;
                for (0..m) |i| {
                    sum += x.data[i * k + l] * dval.data[i * n + j];
                }
                grad_y.data[l * n + j] = sum;
            }
        }

        try self.x.diff(grad_x);
        try self.y.diff(grad_y);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *MatMul) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this matrix multiplication node as a generic Node interface.
    pub fn node(self: *MatMul) Node {
        return Node.init(self);
    }
};

test "matmul basic" {
    const allocator = std.testing.allocator;

    // Create input tensors
    const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer xTensor.deinit();
    xTensor.data[0] = 1.0; // [0,0]
    xTensor.data[1] = 2.0; // [0,1]
    xTensor.data[2] = 3.0; // [0,2]
    xTensor.data[3] = 4.0; // [1,0]
    xTensor.data[4] = 5.0; // [1,1]
    xTensor.data[5] = 6.0; // [1,2]

    const yTensor = try Tensor.init(allocator, &[_]usize{ 3, 2 });
    defer yTensor.deinit();
    yTensor.data[0] = 7.0; // [0,0]
    yTensor.data[1] = 8.0; // [0,1]
    yTensor.data[2] = 9.0; // [1,0]
    yTensor.data[3] = 10.0; // [1,1]
    yTensor.data[4] = 11.0; // [2,0]
    yTensor.data[5] = 12.0; // [2,1]

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // Create matrix multiplication operation
    var matmul_op = try MatMul.init(allocator, x.node(), y.node());
    defer matmul_op.deinit();

    // Evaluate forward pass
    const result = try matmul_op.eval();

    // Expected output:
    // [1 2 3]   [7  8 ]   [1*7 + 2*9 + 3*11   1*8 + 2*10 + 3*12]
    // [4 5 6] @ [9  10] = [4*7 + 5*9 + 6*11   4*8 + 5*10 + 6*12]
    //           [11 12]
    const expected = [_]f64{
        58.0, 64.0, // First row
        139.0, 154.0, // Second row
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "matmul gradient" {
    const allocator = std.testing.allocator;

    // Create input tensors
    const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer xTensor.deinit();
    xTensor.data[0] = 1.0; // [0,0]
    xTensor.data[1] = 2.0; // [0,1]
    xTensor.data[2] = 3.0; // [0,2]
    xTensor.data[3] = 4.0; // [1,0]
    xTensor.data[4] = 5.0; // [1,1]
    xTensor.data[5] = 6.0; // [1,2]

    const yTensor = try Tensor.init(allocator, &[_]usize{ 3, 2 });
    defer yTensor.deinit();
    yTensor.data[0] = 7.0; // [0,0]
    yTensor.data[1] = 8.0; // [0,1]
    yTensor.data[2] = 9.0; // [1,0]
    yTensor.data[3] = 10.0; // [1,1]
    yTensor.data[4] = 11.0; // [2,0]
    yTensor.data[5] = 12.0; // [2,1]

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // Create matrix multiplication operation
    var matmul_op = try MatMul.init(allocator, x.node(), y.node());
    defer matmul_op.deinit();

    // First evaluate to cache the values
    _ = try matmul_op.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0; // [0,0]
    gradTensor.data[1] = 1.0; // [0,1]
    gradTensor.data[2] = 1.0; // [1,0]
    gradTensor.data[3] = 1.0; // [1,1]

    // Compute gradients
    try matmul_op.diff(gradTensor);

    // Expected gradients:
    // For X: dC @ Y^T
    // [1 1]   [7  9  11]   [1*7 + 1*8   1*9 + 1*10   1*11 + 1*12]
    // [1 1] @ [8  10 12] = [1*7 + 1*8   1*9 + 1*10   1*11 + 1*12]
    const expected_x_grad = [_]f64{
        15.0, 19.0, 23.0, // First row
        15.0, 19.0, 23.0, // Second row
    };

    // For Y: X^T @ dC
    // [1 4]   [1 1]   [1*1 + 4*1   1*1 + 4*1]
    // [2 5] @ [1 1] = [2*1 + 5*1   2*1 + 5*1]
    // [3 6]           [3*1 + 6*1   3*1 + 6*1]
    const expected_y_grad = [_]f64{
        5.0, 5.0, // First row
        7.0, 7.0, // Second row
        9.0, 9.0, // Third row
    };

    for (x.grad.data, expected_x_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    for (y.grad.data, expected_y_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
