const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Exp node
/// f = e ^ x
pub const Exp = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Exp {
        const ptr = try allocator.create(Exp);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Exp) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.pow(math.e, xv);
        }

        std.debug.print("Exp-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Exp, dval: *Tensor) void {
        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, self.value.?.data, dval.data) |*v, vv, dv| {
            v.* = dv * vv;
        }
        self.x.diff(grad);

        std.debug.print("Exp-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Exp) Node {
        return Node.init(self);
    }
};
