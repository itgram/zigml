const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Multiply two nodes together.
/// f = a * b
pub const Multiply = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Multiply {
        const ptr = try allocator.create(Multiply);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Multiply) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const a = self.a.eval();
        const b = self.b.eval();

        self.value = Tensor.init(self.allocator, a.shape) catch null;

        for (self.value.?.data, a.data, b.data) |*v, av, bv| {
            v.* = av * bv;
        }

        std.debug.print("Multiply-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Multiply, dval: *Tensor) void {
        const a = self.a.eval();
        const b = self.b.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, b.data, dval.data) |*v, bv, dv| {
            v.* = dv * bv;
        }
        self.a.diff(grad);

        for (grad.data, a.data, dval.data) |*v, av, dv| {
            v.* = dv * av;
        }
        self.b.diff(grad);

        std.debug.print("Multiply-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Multiply) Node {
        return Node.init(self);
    }
};
