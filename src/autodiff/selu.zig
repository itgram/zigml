const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// SELU function node.
/// The Scaled Exponential Linear Unit (SELU) activation function.
/// It is defined as:
/// f(x) = λ * x if x > 0 else λ * α * (exp(x) - 1)
/// - For positive inputs: f(x) = λ * x
/// - For negative inputs: f(x) = λ * α * (exp(x) - 1)
/// where λ is a scaling factor (default 1.0507009873554804934193349852946)
/// and α is a small positive constant (default 1.6732632423543772848170429916717).
/// The SELU function is designed to self-normalize, meaning it helps maintain a mean of 0 and variance of 1 across layers.
pub const SELU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    alpha: f64 = 1.6732632423543772848170429916717, // small slope for negative inputs
    lambda: f64 = 1.0507009873554804934193349852946, // scaling factor for positive inputs

    /// Creates a new SELU node with the given input node and optional alpha and lambda values.
    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: f64, lambda: f64) !*SELU {
        const self = try allocator.create(SELU);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
            .alpha = alpha,
            .lambda = lambda,
        };
        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *SELU) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the SELU function.
    /// The SELU function is defined as:
    /// f(x) = λ * x if x > 0 else λ * α * (exp(x) - 1)
    /// where λ is a scaling factor (default 1.0507009873554804934193349852946)
    /// and α is a small positive constant (default 1.6732632423543772848170429916717).
    /// The SELU function is designed to self-normalize, meaning it helps maintain a mean of 0 and variance of 1 across layers.
    pub fn eval(self: *SELU) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) self.lambda * xv else self.lambda * self.alpha * (std.math.exp(xv) - 1);
        }

        return self.value.?;
    }

    /// Compute the gradient of the SELU function.
    /// The gradient of the SELU function is defined as:
    /// ∂f/∂x = λ if x > 0 else λ * α * exp(x)
    /// where λ is a scaling factor (default 1.0507009873554804934193349852946)
    /// and α is a small positive constant (default 1.6732632423543772848170429916717).
    /// The gradient of the SELU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *SELU, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, self.value.?.data, dval.data) |*v, xv, vv, dv| {
            v.* = if (xv > 0) dv * self.lambda else dv * (vv + self.lambda * self.alpha);
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *SELU) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this SELU node as a generic Node interface.
    pub fn node(self: *SELU) Node {
        return Node.init(self);
    }
};

test "selu basic" {
    const allocator = std.testing.allocator;

    // Default SELU parameters
    const default_alpha = 1.6732632423543772848170429916717;
    const default_lambda = 1.0507009873554804934193349852946;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create selu operation with default parameters
    var selu_op = try SELU.init(allocator, x.node(), default_alpha, default_lambda);
    defer selu_op.deinit();

    // Evaluate forward pass
    const result = try selu_op.eval();

    // Expected values for each input using default parameters:
    // For x > 0: f(x) = λ * x
    // For x ≤ 0: f(x) = λ * α * (exp(x) - 1)
    const expected = [_]f64{
        @as(f64, default_lambda * default_alpha * (std.math.exp(-2.0) - 1)), // selu(-2.0)
        @as(f64, default_lambda * default_alpha * (std.math.exp(-1.0) - 1)), // selu(-1.0)
        @as(f64, 0.0), // selu(0.0)
        @as(f64, default_lambda), // selu(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "selu gradient" {
    const allocator = std.testing.allocator;

    // Default SELU parameters
    const default_alpha = 1.6732632423543772848170429916717;
    const default_lambda = 1.0507009873554804934193349852946;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create selu operation with default parameters
    var selu_op = try SELU.init(allocator, x.node(), default_alpha, default_lambda);
    defer selu_op.deinit();

    // First evaluate to cache the values
    _ = try selu_op.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try selu_op.diff(gradTensor);

    // Expected gradients for each input using default parameters:
    // For x > 0: ∂f/∂x = λ
    // For x ≤ 0: ∂f/∂x = λ * α * exp(x)
    const expected_grad = [_]f64{
        @as(f64, default_lambda * default_alpha * std.math.exp(-2.0)), // selu'(-2.0)
        @as(f64, default_lambda * default_alpha * std.math.exp(-1.0)), // selu'(-1.0)
        @as(f64, default_lambda * default_alpha), // selu'(0.0)
        @as(f64, default_lambda), // selu'(1.0)
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "selu with 2d shapes" {
    const allocator = std.testing.allocator;

    // Default SELU parameters
    const default_alpha = 1.6732632423543772848170429916717;
    const default_lambda = 1.0507009873554804934193349852946;

    // Create input tensor with shape [2, 2]
    const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // [0,0]
    xTensor.data[1] = -1.0; // [0,1]
    xTensor.data[2] = 0.0; // [1,0]
    xTensor.data[3] = 1.0; // [1,1]

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create selu operation with default parameters
    var selu_op = try SELU.init(allocator, x.node(), default_alpha, default_lambda);
    defer selu_op.deinit();

    // Evaluate forward pass
    const result = try selu_op.eval();

    // Expected values for each input using default parameters:
    // For x > 0: f(x) = λ * x
    // For x ≤ 0: f(x) = λ * α * (exp(x) - 1)
    const expected = [_]f64{
        @as(f64, default_lambda * default_alpha * (std.math.exp(-2.0) - 1)), // selu(-2.0)
        @as(f64, default_lambda * default_alpha * (std.math.exp(-1.0) - 1)), // selu(-1.0)
        @as(f64, 0.0), // selu(0.0)
        @as(f64, default_lambda), // selu(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Test gradient computation
    const gradTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    try selu_op.diff(gradTensor);

    // Expected gradients for each position using default parameters:
    // For x > 0: ∂f/∂x = λ
    // For x ≤ 0: ∂f/∂x = λ * α * exp(x)
    const expected_grad = [_]f64{
        @as(f64, default_lambda * default_alpha * std.math.exp(-2.0)), // selu'(-2.0)
        @as(f64, default_lambda * default_alpha * std.math.exp(-1.0)), // selu'(-1.0)
        @as(f64, default_lambda * default_alpha), // selu'(0.0)
        @as(f64, default_lambda), // selu'(1.0)
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "selu reset" {
    const allocator = std.testing.allocator;

    // Default SELU parameters
    const default_alpha = 1.6732632423543772848170429916717;
    const default_lambda = 1.0507009873554804934193349852946;

    // Create input tensor
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0;
    xTensor.data[1] = -1.0;
    xTensor.data[2] = 0.0;
    xTensor.data[3] = 1.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create selu operation with default parameters
    var selu_op = try SELU.init(allocator, x.node(), default_alpha, default_lambda);
    defer selu_op.deinit();

    // First evaluation
    const result1 = try selu_op.eval();

    // Expected values for each input using default parameters:
    // For x > 0: f(x) = λ * x
    // For x ≤ 0: f(x) = λ * α * (exp(x) - 1)
    const expected1 = [_]f64{
        @as(f64, default_lambda * default_alpha * (std.math.exp(-2.0) - 1)), // selu(-2.0)
        @as(f64, default_lambda * default_alpha * (std.math.exp(-1.0) - 1)), // selu(-1.0)
        @as(f64, 0.0), // selu(0.0)
        @as(f64, default_lambda), // selu(1.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    selu_op.reset();
    const result2 = try selu_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        @as(f64, default_lambda * default_alpha * (std.math.exp(-2.0) - 1)), // selu(-2.0)
        @as(f64, default_lambda * default_alpha * (std.math.exp(-1.0) - 1)), // selu(-1.0)
        @as(f64, 0.0), // selu(0.0)
        @as(f64, default_lambda), // selu(1.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "selu custom parameters" {
    const allocator = std.testing.allocator;

    // Create input tensor with test values
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = -2.0; // negative input
    xTensor.data[1] = -1.0; // negative input
    xTensor.data[2] = 0.0; // zero input
    xTensor.data[3] = 1.0; // positive input

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Create selu operation with custom parameters
    const custom_alpha = 2.0;
    const custom_lambda = 1.5;
    var selu_op = try SELU.init(allocator, x.node(), custom_alpha, custom_lambda);
    defer selu_op.deinit();

    // Evaluate forward pass
    const result = try selu_op.eval();

    // Expected values for each input with custom parameters:
    // -2.0: λ * α * (exp(-2.0) - 1)
    // -1.0: λ * α * (exp(-1.0) - 1)
    //  0.0: 0
    //  1.0: λ * 1.0
    const expected = [_]f64{
        @as(f64, custom_lambda * custom_alpha * (std.math.exp(-2.0) - 1)), // selu(-2.0)
        @as(f64, custom_lambda * custom_alpha * (std.math.exp(-1.0) - 1)), // selu(-1.0)
        @as(f64, 0.0), // selu(0.0)
        @as(f64, custom_lambda), // selu(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Test gradient computation
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    try selu_op.diff(gradTensor);

    // Expected gradients for each input with custom parameters:
    // -2.0: λ * α * exp(-2.0)
    // -1.0: λ * α * exp(-1.0)
    //  0.0: λ * α * exp(0.0)
    //  1.0: λ
    const expected_grad = [_]f64{
        @as(f64, custom_lambda * custom_alpha * std.math.exp(-2.0)), // selu'(-2.0)
        @as(f64, custom_lambda * custom_alpha * std.math.exp(-1.0)), // selu'(-1.0)
        @as(f64, custom_lambda * custom_alpha), // selu'(0.0)
        @as(f64, custom_lambda), // selu'(1.0)
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
