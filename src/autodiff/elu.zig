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
pub const Elu = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    alpha: f64 = 0.01, // small slope for negative inputs

    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: f64) !*Elu {
        const ptr = try allocator.create(Elu);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.alpha = alpha;

        return ptr;
    }

    pub fn eval(self: *Elu) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) xv else self.alpha * (math.exp(xv) - 1);
        }

        std.debug.print("Elu-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Elu, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, self.value.?.data, dval.data) |*v, xv, vv, dv| {
            v.* = if (xv > 0) dv else dv * (vv + self.alpha);
        }

        self.x.diff(grad);

        std.debug.print("Elu-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Elu) Node {
        return Node.init(self);
    }
};
