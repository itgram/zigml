const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// GELU function node.
/// The GELU (Gaussian Error Linear Unit) activation function is a smooth approximation of the ReLU function.
/// The GELU function is often used in neural networks as an activation function.
/// It is particularly popular in transformer models and has been shown to perform well in various tasks.
/// The GELU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
/// it is defined as:
/// f(x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))
/// where x is the input tensor.
/// The GELU function is a smooth approximation of the ReLU function.
pub const GELU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    const sqrt_2_over_pi: f32 = 0.79788456; // sqrt(2 / π)
    const coeff: f32 = 0.044715;

    /// Creates a new GELU node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*GELU {
        const self = try allocator.create(GELU);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
        };

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *GELU) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the GELU function.
    /// The GELU function is defined as:
    /// f(x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))
    /// where x is the input tensor.
    /// The GELU function is a smooth approximation of the ReLU function.
    pub fn eval(self: *GELU) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = 0.5 * xv * (1 + std.math.tanh(sqrt_2_over_pi * (xv + coeff * xv * xv * xv)));
        }

        return self.value.?;
    }

    /// Compute the gradient of the GELU function.
    /// The gradient of the GELU function is defined as:
    /// ∂f/∂x = 0.5 * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3))) + 0.5 * x * (1 - tanh(sqrt(2 / π) * (x + 0.044715 * x^3))^2) * sqrt(2 / π) * (1 + 3 * 0.044715 * x^2)
    /// where x is the input tensor.
    /// The gradient of the GELU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *GELU, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            const tanhPart = std.math.tanh(sqrt_2_over_pi * (xv + coeff * xv * xv * xv));
            const derivative = 0.5 * (1 + tanhPart) + 0.5 * xv * (1 - tanhPart * tanhPart) * sqrt_2_over_pi * (1 + 3 * coeff * xv * xv);

            v.* = dv * derivative;
        }

        try self.x.diff(grad);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *GELU) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this GELU node as a generic Node interface.
    pub fn node(self: *GELU) Node {
        return Node.init(self);
    }
};

test "gelu basic" {
    const allocator = std.testing.allocator;

    // GELU constants
    const sqrt_2_over_pi: f32 = 0.79788456; // sqrt(2 / π)
    const coeff: f32 = 0.044715;

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

    // Create gelu operation
    var gelu_op = try GELU.init(allocator, x.node());
    defer gelu_op.deinit();

    // Evaluate forward pass
    const result = try gelu_op.eval();

    // Expected values for each input:
    // f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const expected = [_]f64{
        @as(f64, 0.5 * -2.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0)))), // gelu(-2.0)
        @as(f64, 0.5 * -1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0)))), // gelu(-1.0)
        @as(f64, 0.0), // gelu(0.0)
        @as(f64, 0.5 * 1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0)))), // gelu(1.0)
    };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "gelu gradient" {
    const allocator = std.testing.allocator;

    // GELU constants
    const sqrt_2_over_pi: f32 = 0.79788456; // sqrt(2 / π)
    const coeff: f32 = 0.044715;

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

    // Create gelu operation
    var gelu_op = try GELU.init(allocator, x.node());
    defer gelu_op.deinit();

    // First evaluate to cache the values
    _ = try gelu_op.eval();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 1.0;
    gradTensor.data[2] = 1.0;
    gradTensor.data[3] = 1.0;

    // Compute gradients
    try gelu_op.diff(gradTensor);

    // Expected gradients for each input:
    // ∂f/∂x = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) +
    //         0.5 * x * (1 - tanh(sqrt(2/π) * (x + 0.044715 * x^3))^2) * sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
    const expected_grad = [_]f64{
        @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0))) +
            0.5 * -2.0 * (1 - std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0)) *
                std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0))) *
                sqrt_2_over_pi * (1 + 3 * coeff * -2.0 * -2.0)), // gelu'(-2.0)
        @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0))) +
            0.5 * -1.0 * (1 - std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0)) *
                std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0))) *
                sqrt_2_over_pi * (1 + 3 * coeff * -1.0 * -1.0)), // gelu'(-1.0)
        @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (0.0 + coeff * 0.0 * 0.0 * 0.0))) +
            0.5 * 0.0 * (1 - std.math.tanh(sqrt_2_over_pi * (0.0 + coeff * 0.0 * 0.0 * 0.0)) *
                std.math.tanh(sqrt_2_over_pi * (0.0 + coeff * 0.0 * 0.0 * 0.0))) *
                sqrt_2_over_pi * (1 + 3 * coeff * 0.0 * 0.0)), // gelu'(0.0)
        @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0))) +
            0.5 * 1.0 * (1 - std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0)) *
                std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0))) *
                sqrt_2_over_pi * (1 + 3 * coeff * 1.0 * 1.0)), // gelu'(1.0)
    };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "gelu with multiple shapes" {
    const allocator = std.testing.allocator;

    // GELU constants
    const sqrt_2_over_pi: f32 = 0.79788456; // sqrt(2 / π)
    const coeff: f32 = 0.044715;

    // Test 1: 2D shape [2, 2]
    {
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

        // Create gelu operation
        var gelu_op = try GELU.init(allocator, x.node());
        defer gelu_op.deinit();

        // Evaluate forward pass
        const result = try gelu_op.eval();

        // Expected values for each input:
        // f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const expected = [_]f64{
            @as(f64, 0.5 * -2.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0)))), // gelu(-2.0)
            @as(f64, 0.5 * -1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0)))), // gelu(-1.0)
            @as(f64, 0.0), // gelu(0.0)
            @as(f64, 0.5 * 1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0)))), // gelu(1.0)
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

        try gelu_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) +
        //         0.5 * x * (1 - tanh(sqrt(2/π) * (x + 0.044715 * x^3))^2) * sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
        const expected_grad = [_]f64{
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0))) +
                0.5 * -2.0 * (1 - std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0)) *
                    std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * -2.0 * -2.0)), // gelu'(-2.0)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0))) +
                0.5 * -1.0 * (1 - std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0)) *
                    std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * -1.0 * -1.0)), // gelu'(-1.0)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (0.0 + coeff * 0.0 * 0.0 * 0.0))) +
                0.5 * 0.0 * (1 - std.math.tanh(sqrt_2_over_pi * (0.0 + coeff * 0.0 * 0.0 * 0.0)) *
                    std.math.tanh(sqrt_2_over_pi * (0.0 + coeff * 0.0 * 0.0 * 0.0))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * 0.0 * 0.0)), // gelu'(0.0)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0))) +
                0.5 * 1.0 * (1 - std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0)) *
                    std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * 1.0 * 1.0)), // gelu'(1.0)
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }

    // Test 2: 3D shape [2, 2, 2]
    {
        // Create input tensor with shape [2, 2, 2]
        const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = -2.0; // [0,0,0]
        xTensor.data[1] = -1.0; // [0,0,1]
        xTensor.data[2] = 0.0; // [0,1,0]
        xTensor.data[3] = 1.0; // [0,1,1]
        xTensor.data[4] = -1.5; // [1,0,0]
        xTensor.data[5] = -0.5; // [1,0,1]
        xTensor.data[6] = 0.5; // [1,1,0]
        xTensor.data[7] = 1.5; // [1,1,1]

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Create gelu operation
        var gelu_op = try GELU.init(allocator, x.node());
        defer gelu_op.deinit();

        // Evaluate forward pass
        const result = try gelu_op.eval();

        // Expected values for each input:
        // f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const expected = [_]f64{
            @as(f64, 0.5 * -2.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0)))), // gelu(-2.0)
            @as(f64, 0.5 * -1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0)))), // gelu(-1.0)
            @as(f64, 0.0), // gelu(0.0)
            @as(f64, 0.5 * 1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0)))), // gelu(1.0)
            @as(f64, 0.5 * -1.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.5 + coeff * -1.5 * -1.5 * -1.5)))), // gelu(-1.5)
            @as(f64, 0.5 * -0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-0.5 + coeff * -0.5 * -0.5 * -0.5)))), // gelu(-0.5)
            @as(f64, 0.5 * 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (0.5 + coeff * 0.5 * 0.5 * 0.5)))), // gelu(0.5)
            @as(f64, 0.5 * 1.5 * (1 + std.math.tanh(sqrt_2_over_pi * (1.5 + coeff * 1.5 * 1.5 * 1.5)))), // gelu(1.5)
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Test gradient computation
        const gradTensor = try Tensor.init(allocator, &[_]usize{ 2, 2, 2 });
        defer gradTensor.deinit();
        for (gradTensor.data) |*v| {
            v.* = 1.0;
        }

        try gelu_op.diff(gradTensor);

        // Expected gradients for each position:
        // ∂f/∂x = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) +
        //         0.5 * x * (1 - tanh(sqrt(2/π) * (x + 0.044715 * x^3))^2) * sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
        const expected_grad = [_]f64{
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0))) +
                0.5 * -2.0 * (1 - std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0)) *
                    std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * -2.0 * -2.0)), // gelu'(-2.0)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0))) +
                0.5 * -1.0 * (1 - std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0)) *
                    std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * -1.0 * -1.0)), // gelu'(-1.0)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (0.0 + coeff * 0.0 * 0.0 * 0.0))) +
                0.5 * 0.0 * (1 - std.math.tanh(sqrt_2_over_pi * (0.0 + coeff * 0.0 * 0.0 * 0.0)) *
                    std.math.tanh(sqrt_2_over_pi * (0.0 + coeff * 0.0 * 0.0 * 0.0))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * 0.0 * 0.0)), // gelu'(0.0)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0))) +
                0.5 * 1.0 * (1 - std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0)) *
                    std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * 1.0 * 1.0)), // gelu'(1.0)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.5 + coeff * -1.5 * -1.5 * -1.5))) +
                0.5 * -1.5 * (1 - std.math.tanh(sqrt_2_over_pi * (-1.5 + coeff * -1.5 * -1.5 * -1.5)) *
                    std.math.tanh(sqrt_2_over_pi * (-1.5 + coeff * -1.5 * -1.5 * -1.5))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * -1.5 * -1.5)), // gelu'(-1.5)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (-0.5 + coeff * -0.5 * -0.5 * -0.5))) +
                0.5 * -0.5 * (1 - std.math.tanh(sqrt_2_over_pi * (-0.5 + coeff * -0.5 * -0.5 * -0.5)) *
                    std.math.tanh(sqrt_2_over_pi * (-0.5 + coeff * -0.5 * -0.5 * -0.5))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * -0.5 * -0.5)), // gelu'(-0.5)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (0.5 + coeff * 0.5 * 0.5 * 0.5))) +
                0.5 * 0.5 * (1 - std.math.tanh(sqrt_2_over_pi * (0.5 + coeff * 0.5 * 0.5 * 0.5)) *
                    std.math.tanh(sqrt_2_over_pi * (0.5 + coeff * 0.5 * 0.5 * 0.5))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * 0.5 * 0.5)), // gelu'(0.5)
            @as(f64, 0.5 * (1 + std.math.tanh(sqrt_2_over_pi * (1.5 + coeff * 1.5 * 1.5 * 1.5))) +
                0.5 * 1.5 * (1 - std.math.tanh(sqrt_2_over_pi * (1.5 + coeff * 1.5 * 1.5 * 1.5)) *
                    std.math.tanh(sqrt_2_over_pi * (1.5 + coeff * 1.5 * 1.5 * 1.5))) *
                    sqrt_2_over_pi * (1 + 3 * coeff * 1.5 * 1.5)), // gelu'(1.5)
        };

        for (x.grad.data, expected_grad) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }
    }
}

test "gelu reset" {
    const allocator = std.testing.allocator;

    // GELU constants
    const sqrt_2_over_pi: f32 = 0.79788456; // sqrt(2 / π)
    const coeff: f32 = 0.044715;

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

    // Create gelu operation
    var gelu_op = try GELU.init(allocator, x.node());
    defer gelu_op.deinit();

    // First evaluation
    const result1 = try gelu_op.eval();

    // Expected values for each input:
    // f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const expected1 = [_]f64{
        @as(f64, 0.5 * -2.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0)))), // gelu(-2.0)
        @as(f64, 0.5 * -1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0)))), // gelu(-1.0)
        @as(f64, 0.0), // gelu(0.0)
        @as(f64, 0.5 * 1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0)))), // gelu(1.0)
    };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset and evaluate again
    gelu_op.reset();
    const result2 = try gelu_op.eval();

    // Expected values should be the same after reset
    const expected2 = [_]f64{
        @as(f64, 0.5 * -2.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-2.0 + coeff * -2.0 * -2.0 * -2.0)))), // gelu(-2.0)
        @as(f64, 0.5 * -1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (-1.0 + coeff * -1.0 * -1.0 * -1.0)))), // gelu(-1.0)
        @as(f64, 0.0), // gelu(0.0)
        @as(f64, 0.5 * 1.0 * (1 + std.math.tanh(sqrt_2_over_pi * (1.0 + coeff * 1.0 * 1.0 * 1.0)))), // gelu(1.0)
    };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
