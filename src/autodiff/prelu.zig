const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// PReLU (Parametric ReLU) function node.
/// PReLU is a variant of the ReLU activation function that allows for a small, learnable slope for negative inputs.
/// The PReLU function is defined as:
/// f(x) = x if x > 0 else α * x
/// - For positive inputs: f(x) = x
/// - For negative inputs: f(x) = α * x
/// where α is a learnable parameter (default 0.01).
/// The alpha parameter is a learnable parameter that can be trained during the optimization process.
/// The PReLU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
/// The PReLU function is particularly useful in deep neural networks where the ReLU function may lead to dead neurons.
/// It allows the model to learn a small slope for negative inputs, which can help improve gradient flow during training.
pub const PReLU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    grad: *Tensor, // gradient of alpha
    x: Node,
    alpha: *Tensor = 0.01, // learnable parameter (trainable)

    /// Creates a new PReLU node with the given input node and alpha value.
    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: *Tensor) !*PReLU {
        const ptr = try allocator.create(PReLU);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.grad = try Tensor.init(allocator, alpha.shape);
        ptr.x = x;
        ptr.alpha = alpha;

        ptr.grad.zero();

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *PReLU) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the PReLU function.
    /// The PReLU function is defined as:
    /// f(x) = x if x > 0 else α * x
    /// where α is a learnable parameter (default 0.01).
    /// The PReLU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
    /// The PReLU function is particularly useful in deep neural networks where the ReLU function may lead to dead neurons.
    /// It allows the model to learn a small slope for negative inputs, which can help improve gradient flow during training.
    pub fn eval(self: *PReLU) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data, self.alpha.data) |*v, xv, alpha| {
            v.* = if (xv > 0) xv else alpha * xv;
        }

        std.debug.print("PReLU-eval: value: {?}, alpha: {}\n", .{ self.value, self.alpha });

        return self.value.?;
    }

    /// Compute the gradient of the PReLU function.
    /// The gradient of the PReLU function is defined as:
    /// ∂f/∂x = 1 if x > 0 else α
    /// ∂f / ∂α = x if x > 0 else 0
    /// where α is a learnable parameter (default 0.01).
    /// The gradient of the PReLU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *PReLU, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, self.grad.data, x.data, dval.data) |*v, *ag, xv, dv| {
            v.* = if (xv > 0) dv else dv * self.alpha;
            ag.* += if (xv > 0) 0 else dv * xv;
        }

        try self.x.diff(grad);

        std.debug.print("PReLU-diff: value: {?}, alpha-grad: {}, dval: {}\n", .{ self.value, self.grad, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *PReLU) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this PReLU node as a generic Node interface.
    pub fn node(self: *PReLU) Node {
        return Node.init(self);
    }
};
