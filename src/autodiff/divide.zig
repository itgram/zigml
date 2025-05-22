const std = @import("std");
const Node = @import("node.zig").Node;

/// Divide two nodes
/// f = a / b
pub const Divide = struct {
    value: ?f64,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Divide {
        const ptr = try allocator.create(Divide);
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Divide) f64 {
        if (self.value) |v| {
            return v;
        }

        self.value = self.a.eval() / self.b.eval();

        std.debug.print("Divide-eval: value: {?d}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Divide, dval: f64) void {
        const a = self.a.eval();
        const b = self.b.eval();

        self.a.diff(dval * (1 / b));
        self.b.diff(dval * (-a / (b * b)));

        std.debug.print("Divide-diff: value: {?d}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Divide) Node {
        return Node.init(self);
    }
};
