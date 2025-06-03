const std = @import("std");
const math = std.math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Graph = @import("graph.zig").Graph;

const epsilon = 1e-7; // Small value to prevent log(0), matching PyTorch's implementation

/// Power function node.
/// where x and y are nodes representing tensors.
/// The Power node is used to compute the element-wise power of two tensors.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Power node is defined as:
/// f(x, y) = x^y
/// where x is the base tensor and y is the exponent tensor.
/// The Power node is typically used in neural networks for operations such as exponentiation and activation functions.
pub const Power = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    /// Creates a new power node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Power {
        const self = try allocator.create(Power);
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
    pub fn deinit(self: *Power) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the power function.
    /// The power function is defined as:
    /// f(x, y) = x^y
    /// where x and y are the input tensors.
    pub fn eval(self: *Power) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();
        const y = try self.y.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = math.pow(f64, xv + epsilon, yv);
        }

        std.debug.print("Power-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the power function.
    /// The gradient of the power function is defined as:
    /// ∂f/∂x = y * x^(y-1)
    /// ∂f/∂y = x^y * ln(x)
    /// where x and y are the input tensors.
    /// The gradient of the power function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Power, dval: *Tensor) !void {
        const x = try self.x.eval();
        const y = try self.y.eval();

        const grad_x = try Tensor.init(self.allocator, dval.shape);
        defer grad_x.deinit();
        const grad_y = try Tensor.init(self.allocator, dval.shape);
        defer grad_y.deinit();

        for (grad_x.data, grad_y.data, x.data, y.data, dval.data) |*gx, *gy, xv, yv, dv| {
            gx.* = dv * yv * math.pow(f64, xv + epsilon, yv - 1);
            gy.* = dv * math.pow(f64, xv + epsilon, yv) * @log(xv + epsilon);
        }

        try self.x.diff(grad_x);
        try self.y.diff(grad_y);

        std.debug.print("Power-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Power) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this power node as a generic Node interface.
    pub fn node(self: *Power) Node {
        return Node.init(self);
    }
};

test "power basic" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensors with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0; // positive base
    xTensor.data[1] = 3.0; // positive base
    xTensor.data[2] = 0.5; // fractional base
    xTensor.data[3] = 4.0; // positive base

    const yTensor = try graph.tensor(&[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 2.0; // positive exponent
    yTensor.data[1] = 0.0; // zero exponent
    yTensor.data[2] = -1.0; // negative exponent
    yTensor.data[3] = 0.5; // fractional exponent

    // Create variables
    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // Create power operation
    var pow_op = try graph.power(x.node(), y.node());
    defer pow_op.deinit();

    // Evaluate forward pass
    const result = try pow_op.eval();

    // Expected values for each input pair:
    // f(x, y) = (x + epsilon)^y
    const expected = [_]f64{
        math.pow(f64, 2.0 + 1e-7, 2.0), // (2.0 + 1e-7)^2.0 = 4.0
        math.pow(f64, 3.0 + 1e-7, 0.0), // (3.0 + 1e-7)^0.0 = 1.0
        math.pow(f64, 0.5 + 1e-7, -1.0), // (0.5 + 1e-7)^-1.0 = 2.0
        math.pow(f64, 4.0 + 1e-7, 0.5), // (4.0 + 1e-7)^0.5 = 2.0
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Verify tensor shape is preserved
    try std.testing.expectEqual(@as(usize, 4), result.shape[0]);
}

test "power gradient" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensors with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0; // positive base
    xTensor.data[1] = 3.0; // positive base
    xTensor.data[2] = 0.5; // fractional base
    xTensor.data[3] = 4.0; // positive base

    const yTensor = try graph.tensor(&[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 2.0; // positive exponent
    yTensor.data[1] = 0.0; // zero exponent
    yTensor.data[2] = -1.0; // negative exponent
    yTensor.data[3] = 0.5; // fractional exponent

    // Create variables
    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // Create power operation
    var pow_op = try graph.power(x.node(), y.node());
    defer pow_op.deinit();

    // First evaluate to cache the values
    _ = try pow_op.eval();

    // Create gradient tensor
    const gradTensor = try graph.tensor(&[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try pow_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = y * (x + epsilon)^(y-1)
    // ∂f/∂y = (x + epsilon)^y * ln(x + epsilon)
    const expected_x_grad = [_]f64{
        2.0 * math.pow(f64, 2.0 + 1e-7, 1.0), // 2.0 * (2.0 + 1e-7)^1.0 = 4.0
        0.0 * math.pow(f64, 3.0 + 1e-7, -1.0), // 0.0 * (3.0 + 1e-7)^-1.0 = 0.0
        -1.0 * math.pow(f64, 0.5 + 1e-7, -2.0), // -1.0 * (0.5 + 1e-7)^-2.0 = -4.0
        0.5 * math.pow(f64, 4.0 + 1e-7, -0.5), // 0.5 * (4.0 + 1e-7)^-0.5 = 0.25
    };

    const expected_y_grad = [_]f64{
        math.pow(f64, 2.0 + 1e-7, 2.0) * @log(2.0 + 1e-7), // (2.0 + 1e-7)^2.0 * ln(2.0 + 1e-7) ≈ 2.7726
        math.pow(f64, 3.0 + 1e-7, 0.0) * @log(3.0 + 1e-7), // (3.0 + 1e-7)^0.0 * ln(3.0 + 1e-7) ≈ 1.0986
        math.pow(f64, 0.5 + 1e-7, -1.0) * @log(0.5 + 1e-7), // (0.5 + 1e-7)^-1.0 * ln(0.5 + 1e-7) ≈ -1.3863
        math.pow(f64, 4.0 + 1e-7, 0.5) * @log(4.0 + 1e-7), // (4.0 + 1e-7)^0.5 * ln(4.0 + 1e-7) ≈ 2.7726
    };

    for (x.grad.data, expected_x_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    for (y.grad.data, expected_y_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "power with multiple shapes" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Test 1: 2D shape [2, 2]
    {
        // Create input tensors with shape [2, 2]
        const xTensor = try graph.tensor(&[_]usize{ 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = 2.0; // [0,0]
        xTensor.data[1] = 3.0; // [0,1]
        xTensor.data[2] = 0.5; // [1,0]
        xTensor.data[3] = 4.0; // [1,1]

        const yTensor = try graph.tensor(&[_]usize{ 2, 2 });
        defer yTensor.deinit();
        yTensor.data[0] = 2.0; // [0,0]
        yTensor.data[1] = 0.0; // [0,1]
        yTensor.data[2] = -1.0; // [1,0]
        yTensor.data[3] = 0.5; // [1,1]

        // Create variables
        var x = try graph.variable("x", xTensor);
        defer x.deinit();
        var y = try graph.variable("y", yTensor);
        defer y.deinit();

        // Create power operation
        var pow_op = try graph.power(x.node(), y.node());
        defer pow_op.deinit();

        // Evaluate forward pass
        const result = try pow_op.eval();

        // Expected values for each input pair:
        // f(x, y) = (x + epsilon)^y
        const expected = [_]f64{
            math.pow(f64, 2.0 + 1e-7, 2.0), // (2.0 + 1e-7)^2.0 = 4.0
            math.pow(f64, 3.0 + 1e-7, 0.0), // (3.0 + 1e-7)^0.0 = 1.0
            math.pow(f64, 0.5 + 1e-7, -1.0), // (0.5 + 1e-7)^-1.0 = 2.0
            math.pow(f64, 4.0 + 1e-7, 0.5), // (4.0 + 1e-7)^0.5 = 2.0
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Test gradient computation
        const gradTensor = try graph.tensor(&[_]usize{ 2, 2 });
        defer gradTensor.deinit();
        gradTensor.data[0] = 1.0;
        gradTensor.data[1] = 1.0;
        gradTensor.data[2] = 1.0;
        gradTensor.data[3] = 1.0;

        try pow_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = y * (x + epsilon)^(y-1)
        // ∂f/∂y = (x + epsilon)^y * ln(x + epsilon)
        const expected_x_grad = [_]f64{
            2.0 * math.pow(f64, 2.0 + 1e-7, 1.0), // 2.0 * (2.0 + 1e-7)^1.0 = 4.0
            0.0 * math.pow(f64, 3.0 + 1e-7, -1.0), // 0.0 * (3.0 + 1e-7)^-1.0 = 0.0
            -1.0 * math.pow(f64, 0.5 + 1e-7, -2.0), // -1.0 * (0.5 + 1e-7)^-2.0 = -4.0
            0.5 * math.pow(f64, 4.0 + 1e-7, -0.5), // 0.5 * (4.0 + 1e-7)^-0.5 = 0.25
        };

        const expected_y_grad = [_]f64{
            math.pow(f64, 2.0 + 1e-7, 2.0) * @log(2.0 + 1e-7), // (2.0 + 1e-7)^2.0 * ln(2.0 + 1e-7) ≈ 2.7726
            math.pow(f64, 3.0 + 1e-7, 0.0) * @log(3.0 + 1e-7), // (3.0 + 1e-7)^0.0 * ln(3.0 + 1e-7) ≈ 1.0986
            math.pow(f64, 0.5 + 1e-7, -1.0) * @log(0.5 + 1e-7), // (0.5 + 1e-7)^-1.0 * ln(0.5 + 1e-7) ≈ -1.3863
            math.pow(f64, 4.0 + 1e-7, 0.5) * @log(4.0 + 1e-7), // (4.0 + 1e-7)^0.5 * ln(4.0 + 1e-7) ≈ 2.7726
        };

        for (x.grad.data, expected_x_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        for (y.grad.data, expected_y_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }

    // Test 2: 3D shape [2, 2, 2]
    {
        // Create input tensors with shape [2, 2, 2]
        const xTensor = try graph.tensor(&[_]usize{ 2, 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = 2.0; // [0,0,0]
        xTensor.data[1] = 3.0; // [0,0,1]
        xTensor.data[2] = 0.5; // [0,1,0]
        xTensor.data[3] = 4.0; // [0,1,1]
        xTensor.data[4] = 1.5; // [1,0,0]
        xTensor.data[5] = 2.5; // [1,0,1]
        xTensor.data[6] = 3.5; // [1,1,0]
        xTensor.data[7] = 4.5; // [1,1,1]

        const yTensor = try graph.tensor(&[_]usize{ 2, 2, 2 });
        defer yTensor.deinit();
        yTensor.data[0] = 2.0; // [0,0,0]
        yTensor.data[1] = 0.0; // [0,0,1]
        yTensor.data[2] = -1.0; // [0,1,0]
        yTensor.data[3] = 0.5; // [0,1,1]
        yTensor.data[4] = -2.0; // [1,0,0]
        yTensor.data[5] = 1.0; // [1,0,1]
        yTensor.data[6] = 2.0; // [1,1,0]
        yTensor.data[7] = 0.5; // [1,1,1]

        // Create variables
        var x = try graph.variable("x", xTensor);
        defer x.deinit();
        var y = try graph.variable("y", yTensor);
        defer y.deinit();

        // Create power operation
        var pow_op = try graph.power(x.node(), y.node());
        defer pow_op.deinit();

        // Evaluate forward pass
        const result = try pow_op.eval();

        // Expected values for each input pair:
        // f(x, y) = (x + epsilon)^y
        const expected = [_]f64{
            math.pow(f64, 2.0 + 1e-7, 2.0), // (2.0 + 1e-7)^2.0 = 4.0
            math.pow(f64, 3.0 + 1e-7, 0.0), // (3.0 + 1e-7)^0.0 = 1.0
            math.pow(f64, 0.5 + 1e-7, -1.0), // (0.5 + 1e-7)^-1.0 = 2.0
            math.pow(f64, 4.0 + 1e-7, 0.5), // (4.0 + 1e-7)^0.5 = 2.0
            math.pow(f64, 1.5 + 1e-7, -2.0), // (1.5 + 1e-7)^-2.0 ≈ 0.4444
            math.pow(f64, 2.5 + 1e-7, 1.0), // (2.5 + 1e-7)^1.0 = 2.5
            math.pow(f64, 3.5 + 1e-7, 2.0), // (3.5 + 1e-7)^2.0 = 12.25
            math.pow(f64, 4.5 + 1e-7, 0.5), // (4.5 + 1e-7)^0.5 ≈ 2.1213
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Test gradient computation
        const gradTensor = try graph.tensor(&[_]usize{ 2, 2, 2 });
        defer gradTensor.deinit();
        for (gradTensor.data) |*v| {
            v.* = 1.0;
        }

        try pow_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = y * (x + epsilon)^(y-1)
        // ∂f/∂y = (x + epsilon)^y * ln(x + epsilon)
        const expected_x_grad = [_]f64{
            2.0 * math.pow(f64, 2.0 + 1e-7, 1.0), // 2.0 * (2.0 + 1e-7)^1.0 = 4.0
            0.0 * math.pow(f64, 3.0 + 1e-7, -1.0), // 0.0 * (3.0 + 1e-7)^-1.0 = 0.0
            -1.0 * math.pow(f64, 0.5 + 1e-7, -2.0), // -1.0 * (0.5 + 1e-7)^-2.0 = -4.0
            0.5 * math.pow(f64, 4.0 + 1e-7, -0.5), // 0.5 * (4.0 + 1e-7)^-0.5 = 0.25
            -2.0 * math.pow(f64, 1.5 + 1e-7, -3.0), // -2.0 * (1.5 + 1e-7)^-3.0 ≈ -0.5926
            1.0 * math.pow(f64, 2.5 + 1e-7, 0.0), // 1.0 * (2.5 + 1e-7)^0.0 = 1.0
            2.0 * math.pow(f64, 3.5 + 1e-7, 1.0), // 2.0 * (3.5 + 1e-7)^1.0 = 7.0
            0.5 * math.pow(f64, 4.5 + 1e-7, -0.5), // 0.5 * (4.5 + 1e-7)^-0.5 ≈ 0.2357
        };

        const expected_y_grad = [_]f64{
            math.pow(f64, 2.0 + 1e-7, 2.0) * @log(2.0 + 1e-7), // (2.0 + 1e-7)^2.0 * ln(2.0 + 1e-7) ≈ 2.7726
            math.pow(f64, 3.0 + 1e-7, 0.0) * @log(3.0 + 1e-7), // (3.0 + 1e-7)^0.0 * ln(3.0 + 1e-7) ≈ 1.0986
            math.pow(f64, 0.5 + 1e-7, -1.0) * @log(0.5 + 1e-7), // (0.5 + 1e-7)^-1.0 * ln(0.5 + 1e-7) ≈ -1.3863
            math.pow(f64, 4.0 + 1e-7, 0.5) * @log(4.0 + 1e-7), // (4.0 + 1e-7)^0.5 * ln(4.0 + 1e-7) ≈ 2.7726
            math.pow(f64, 1.5 + 1e-7, -2.0) * @log(1.5 + 1e-7), // (1.5 + 1e-7)^-2.0 * ln(1.5 + 1e-7) ≈ -0.4055
            math.pow(f64, 2.5 + 1e-7, 1.0) * @log(2.5 + 1e-7), // (2.5 + 1e-7)^1.0 * ln(2.5 + 1e-7) ≈ 2.2907
            math.pow(f64, 3.5 + 1e-7, 2.0) * @log(3.5 + 1e-7), // (3.5 + 1e-7)^2.0 * ln(3.5 + 1e-7) ≈ 15.7526
            math.pow(f64, 4.5 + 1e-7, 0.5) * @log(4.5 + 1e-7), // (4.5 + 1e-7)^0.5 * ln(4.5 + 1e-7) ≈ 3.2958
        };

        for (x.grad.data, expected_x_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        for (y.grad.data, expected_y_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "power reset" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // Create input tensors with test values
    const xTensor = try graph.tensor(&[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0; // positive base
    xTensor.data[1] = 3.0; // positive base
    xTensor.data[2] = 0.5; // fractional base
    xTensor.data[3] = 4.0; // positive base

    const yTensor = try graph.tensor(&[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 2.0; // positive exponent
    yTensor.data[1] = 0.0; // zero exponent
    yTensor.data[2] = -1.0; // negative exponent
    yTensor.data[3] = 0.5; // fractional exponent

    // Create variables
    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // Create power operation
    var pow_op = try graph.power(x.node(), y.node());
    defer pow_op.deinit();

    // First evaluation
    const result1 = try pow_op.eval();

    // Expected values for each input pair:
    // f(x, y) = (x + epsilon)^y
    const expected1 = [_]f64{
        math.pow(f64, 2.0 + 1e-7, 2.0), // (2.0 + 1e-7)^2.0 = 4.0
        math.pow(f64, 3.0 + 1e-7, 0.0), // (3.0 + 1e-7)^0.0 = 1.0
        math.pow(f64, 0.5 + 1e-7, -1.0), // (0.5 + 1e-7)^-1.0 = 2.0
        math.pow(f64, 4.0 + 1e-7, 0.5), // (4.0 + 1e-7)^0.5 = 2.0
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    pow_op.reset();
    const result2 = try pow_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        math.pow(f64, 2.0 + 1e-7, 2.0), // (2.0 + 1e-7)^2.0 = 4.0
        math.pow(f64, 3.0 + 1e-7, 0.0), // (3.0 + 1e-7)^0.0 = 1.0
        math.pow(f64, 0.5 + 1e-7, -1.0), // (0.5 + 1e-7)^-1.0 = 2.0
        math.pow(f64, 4.0 + 1e-7, 0.5), // (4.0 + 1e-7)^0.5 = 2.0
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
