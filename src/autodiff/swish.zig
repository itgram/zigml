const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Swish function node.
/// The Swish function is a smooth, non-monotonic activation function.
/// The Swish function is often used in deep learning models as an activation function.
/// It has been shown to perform better than ReLU in some cases, especially in deeper networks.
/// The Swish function is differentiable everywhere, making it suitable for backpropagation in neural networks.
/// It is defined as:
/// f(x) = x * σ(x) = x / (1 + exp(-x))
/// where σ is the sigmoid function.
pub const Swish = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    const sqrt_2_over_pi: f32 = 0.79788456; // sqrt(2 / π)
    const coeff: f32 = 0.044715;

    /// Creates a new Swish node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Swish {
        const ptr = try allocator.create(Swish);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Swish) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the Swish function.
    /// The Swish function is defined as:
    /// f(x) = x * σ(x) = x / (1 + exp(-x))
    /// where σ is the sigmoid function.
    /// The Swish function is a smooth, non-monotonic activation function.
    /// The Swish function is often used in deep learning models as an activation function.
    /// It has been shown to perform better than ReLU in some cases, especially in deeper networks.
    pub fn eval(self: *Swish) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = xv / (1 + math.exp(-xv));
        }

        std.debug.print("Swish-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the Swish function.
    /// The gradient of the Swish function is defined as:
    /// ∂f/∂x = σ(x) + x * σ(x) * (1 - σ(x))
    /// where σ is the sigmoid function.
    /// The gradient of the Swish function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Swish, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            const sig = 1 / (1 + std.math.exp(-v));
            v.* = dv * (sig + xv * sig * (1 - sig));
        }

        self.x.diff(grad);

        std.debug.print("Swish-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Returns this Swish node as a generic Node interface.
    pub fn node(self: *Swish) Node {
        return Node.init(self);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Swish) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }
};
