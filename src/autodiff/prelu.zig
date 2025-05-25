const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// PRelu Parametric ReLU activation function node.
/// f(x, alpha) = x if x > 0 else alpha * x, where alpha is a learnable parameter
pub const PRelu = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    grad: *Tensor, // gradient of alpha
    x: Node,
    alpha: *Tensor = 0.01, // learnable parameter (trainable)

    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: *Tensor) !*PRelu {
        const ptr = try allocator.create(PRelu);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.grad = try Tensor.init(allocator, alpha.shape);
        ptr.x = x;
        ptr.alpha = alpha;

        ptr.grad.zero();

        return ptr;
    }

    pub fn eval(self: *PRelu) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data, self.alpha.data) |*v, xv, alpha| {
            v.* = if (xv > 0) xv else alpha * xv;
        }

        std.debug.print("PRelu-eval: value: {?}, alpha: {}\n", .{ self.value, self.alpha });

        return self.value.?;
    }

    pub fn diff(self: *PRelu, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, self.grad.data, x.data, dval.data) |*v, *ag, xv, dv| {
            v.* = if (xv > 0) dv else dv * self.alpha;
            ag.* += if (xv > 0) 0 else dv * xv;
        }

        self.x.diff(grad);

        std.debug.print("PRelu-diff: value: {?}, alpha-grad: {}, dval: {}\n", .{ self.value, self.grad, dval });
    }

    pub fn node(self: *PRelu) Node {
        return Node.init(self);
    }
};
