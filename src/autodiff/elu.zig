const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Elu Exponential Linear Unit function node.
/// f(x, alpha) = x if x > 0 else alpha * (exp(x) - 1), where alpha is a small constant (e.g., 0.01)
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
            v.* = if (xv > 0) xv else self.alpha * (@exp(xv) - 1);
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
