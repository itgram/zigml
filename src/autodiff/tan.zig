const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Tangent function node.
/// f = tan(x)
pub const Tan = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Tan {
        const ptr = try allocator.create(Tan);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Tan) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.tan(xv);
        }

        std.debug.print("Tan-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Tan, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            const sec2 = 1.0 / math.cos(xv);
            v.* = dv * sec2 * sec2;
        }

        self.x.diff(grad);

        std.debug.print("Tan-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Tan) Node {
        return Node.init(self);
    }
};
