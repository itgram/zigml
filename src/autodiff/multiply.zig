const std = @import("std");
const Node = @import("node.zig").Node;

/// Multiply two nodes together.
/// f = a * b
pub const Multiply = struct {
    value: ?f64,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Multiply {
        const ptr = try allocator.create(Multiply);
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Multiply) f64 {
        if (self.value) |v| {
            return v;
        }

        self.value = self.a.eval() * self.b.eval();

        std.debug.print("Multiply-eval: value: {?d}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Multiply, dval: f64) void {
        self.a.diff(dval * self.b.eval());
        self.b.diff(dval * self.a.eval());

        std.debug.print("Multiply-diff: value: {?d}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Multiply) Node {
        return Node.init(self);
    }
};
