const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// LeakyReLU function node.
/// f(x, alpha) = x if x > 0 else alpha * x, where alpha is a small constant (e.g., 0.01)
pub const LeakyReLU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    alpha: f64 = 0.01, // small slope for negative inputs

    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: f64) !*LeakyReLU {
        const ptr = try allocator.create(LeakyReLU);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.alpha = alpha;

        return ptr;
    }

    pub fn eval(self: *LeakyReLU) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) xv else self.alpha * xv;
        }

        std.debug.print("LeakyReLU-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *LeakyReLU, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = if (xv > 0) dv else dv * self.alpha;
        }

        self.x.diff(grad);

        std.debug.print("LeakyReLU-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *LeakyReLU) Node {
        return Node.init(self);
    }
};
