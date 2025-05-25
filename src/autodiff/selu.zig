const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Selu Scaled Exponential Linear Unit function node.
/// f(x, alpha, lambda) = lambda * x if x > 0 else lambda * alpha * (exp(x) - 1),
/// where alpha is a small constant (e.g., 0.01) and lambda is a scaling factor (e.g., 1.0507)
/// This activation function is designed to keep the mean and variance of the inputs close to zero and one, respectively.
/// It is often used in deep neural networks to help with convergence and stability.
pub const Selu = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    alpha: f64 = 1.6732632423543772848170429916717, // small slope for negative inputs
    lambda: f64 = 1.0507009873554804934193349852946, // scaling factor for positive inputs

    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: f64, lambda: f64) !*Selu {
        const ptr = try allocator.create(Selu);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.alpha = alpha;
        ptr.lambda = lambda;

        return ptr;
    }

    pub fn eval(self: *Selu) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) self.lambda * xv else self.lambda * self.alpha * (@exp(xv) - 1);
        }

        std.debug.print("Selu-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Selu, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, self.value.?.data, dval.data) |*v, xv, vv, dv| {
            v.* = if (xv > 0) dv * self.lambda else dv * (vv + self.lambda * self.alpha);
        }

        self.x.diff(grad);

        std.debug.print("Selu-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Selu) Node {
        return Node.init(self);
    }
};
