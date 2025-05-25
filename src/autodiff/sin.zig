const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Sine function node.
/// f = sin(x)
pub const Sin = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Sin {
        const ptr = try allocator.create(Sin);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Sin) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.sin(xv);
        }

        std.debug.print("Sin-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Sin, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv * math.cos(xv);
        }

        self.x.diff(grad);

        std.debug.print("Sin-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Sin) Node {
        return Node.init(self);
    }
};
