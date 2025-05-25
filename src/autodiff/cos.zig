const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Cos function node.
/// f = cos(x)
pub const Cos = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Cos {
        const ptr = try allocator.create(Cos);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Cos) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.cos(xv);
        }

        std.debug.print("Cos-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Cos, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv * -math.sin(xv); // derivative of cos is -sin
        }

        self.x.diff(grad);

        std.debug.print("Cos-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Cos) Node {
        return Node.init(self);
    }
};
