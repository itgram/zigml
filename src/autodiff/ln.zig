const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Natural Logarithm node
/// f = ln(x)
pub const Ln = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Ln {
        const ptr = try allocator.create(Ln);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Ln) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.log(math.e, xv);
        }

        std.debug.print("Ln-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Ln, dval: *Tensor) void {
        const x = self.x.eval();

        const ndval = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (ndval.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv / xv;
        }

        self.x.diff(ndval);

        std.debug.print("Ln-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Ln) Node {
        return Node.init(self);
    }
};
