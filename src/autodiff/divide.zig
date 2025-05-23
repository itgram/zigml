const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Divide two nodes
/// f = a / b
pub const Divide = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Divide {
        const ptr = try allocator.create(Divide);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Divide) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const a = self.a.eval();
        const b = self.b.eval();

        self.value = Tensor.init(self.allocator, a.shape) catch null;

        for (self.value.?.data, a.data, b.data) |*v, av, bv| {
            v.* = av / bv;
        }

        std.debug.print("Divide-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Divide, dval: *Tensor) void {
        const a = self.a.eval();
        const b = self.b.eval();

        const ndval = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (ndval.data, b.data, dval.data) |*v, bv, dv| {
            v.* = dv / bv;
        }
        self.a.diff(ndval);

        for (ndval.data, a.data, b.data, dval.data) |*v, av, bv, dv| {
            v.* = -(dv * av) / (bv * bv);
        }
        self.b.diff(ndval);

        std.debug.print("Divide-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Divide) Node {
        return Node.init(self);
    }
};
