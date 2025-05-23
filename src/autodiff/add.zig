const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Add two nodes
/// f = a + b
pub const Add = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Add {
        const ptr = try allocator.create(Add);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Add) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const a = self.a.eval();
        const b = self.b.eval();

        self.value = Tensor.init(self.allocator, a.shape) catch null;

        for (self.value.?.data, a.data, b.data) |*v, av, bv| {
            v.* = av + bv;
        }

        std.debug.print("Add-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Add, dval: *Tensor) void {
        self.a.diff(dval);
        self.b.diff(dval);

        std.debug.print("Add-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Add) Node {
        return Node.init(self);
    }
};
