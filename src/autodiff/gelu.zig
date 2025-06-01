const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

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
        const ptr = try allocator.create(GELU);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *GELU) void {
        if (self.value) |v| {
            v.deinit();
            self.allocator.destroy(v);
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the GELU function.
    /// The GELU function is defined as:
    /// f(x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))
    /// where x is the input tensor.
    /// The GELU function is a smooth approximation of the ReLU function.
    pub fn eval(self: *GELU) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = 0.5 * xv * (1 + math.tanh(sqrt_2_over_pi * (xv + coeff * xv * xv * xv)));
        }

        std.debug.print("GELU-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the GELU function.
    /// The gradient of the GELU function is defined as:
    /// ∂f/∂x = 0.5 * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3))) + 0.5 * x * (1 - tanh(sqrt(2 / π) * (x + 0.044715 * x^3))^2) * sqrt(2 / π) * (1 + 3 * 0.044715 * x^2)
    /// where x is the input tensor.
    /// The gradient of the GELU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *GELU, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            const tanhPart = math.tanh(sqrt_2_over_pi * (xv + coeff * xv * xv * xv));
            const derivative = 0.5 * (1 + tanhPart) + 0.5 * xv * (1 - tanhPart * tanhPart) * sqrt_2_over_pi * (1 + 3 * coeff * xv * xv);

            v.* = dv * derivative;
        }

        self.x.diff(grad);

        std.debug.print("GELU-diff: value: {?}, dval: {}\n", .{ self.value, dval });
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
