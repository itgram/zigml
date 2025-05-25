const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Linear function node.
/// f(x) = x
pub const Linear = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Linear {
        const ptr = try allocator.create(Linear);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Linear) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = xv;
        }

        std.debug.print("Linear-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Linear, dval: *Tensor) void {
        self.x.diff(dval);

        std.debug.print("Linear-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Linear) Node {
        return Node.init(self);
    }
};
