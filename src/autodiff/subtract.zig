const std = @import("std");
const Node = @import("node.zig").Node;

/// Subtract two nodes
/// f = a - b
pub const Subtract = struct {
    value: ?f64,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Subtract {
        const ptr = try allocator.create(Subtract);
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Subtract) f64 {
        if (self.value) |v| {
            return v;
        }

        self.value = self.a.eval() - self.b.eval();

        std.debug.print("Subtract-eval: value: {?d}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Subtract, dval: f64) void {
        self.a.diff(dval);
        self.b.diff(dval * -1.0);

        std.debug.print("Subtract-diff: value: {?d}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Subtract) Node {
        return Node.init(self);
    }
};
