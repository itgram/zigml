const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Exponential Linear Unit (ELU) activation function node.
/// The ELU function is used in neural networks to introduce non-linearity.
/// It is similar to ReLU but allows for negative values, which can help with learning and convergence.
/// The ELU function is defined as:
/// f(x) = x if x > 0 else α * (exp(x) - 1)
/// - For positive inputs: f(x) = x
/// - For negative inputs: f(x) = α * (exp(x) - 1)
/// where α is a small positive constant (default 0.01).
/// The ELU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
pub const ELU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    alpha: f64 = 0.01, // small slope for negative inputs

    /// Creates a new ELU node with the given input node and alpha value.
    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: f64) !*ELU {
        const ptr = try allocator.create(ELU);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.alpha = alpha;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *ELU) void {
        if (self.value) |v| {
            v.deinit();
            self.allocator.destroy(v);
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the ELU function.
    /// The ELU function is defined as:
    /// f(x) = x if x > 0 else α * (exp(x) - 1)
    /// where α is a small positive constant (default 0.01).
    /// The ELU function is differentiable everywhere, making it suitable for backpropagation in neural networks.
    pub fn eval(self: *ELU) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) xv else self.alpha * (math.exp(xv) - 1);
        }

        std.debug.print("ELU-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the ELU function.
    /// The gradient of the ELU function is defined as:
    /// ∂f/∂x = 1 if x > 0 else α * exp(x)
    /// where α is a small positive constant (default 0.01).
    /// The gradient of the ELU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *ELU, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, self.value.?.data, dval.data) |*v, xv, vv, dv| {
            v.* = if (xv > 0) dv else dv * (vv + self.alpha);
        }

        self.x.diff(grad);

        std.debug.print("ELU-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *ELU) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this ELU node as a generic Node interface.
    pub fn node(self: *ELU) Node {
        return Node.init(self);
    }
};
