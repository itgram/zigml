const std = @import("std");
const Node = @import("node.zig").Node;

/// Power node
/// f = a ^ b
pub const Power = struct {
    value: ?f64,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Power {
        const ptr = try allocator.create(Power);
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Power) f64 {
        if (self.value) |v| {
            return v;
        }

        self.value = std.math.pow(self.a.eval(), self.b.eval());

        std.debug.print("Power-eval: value: {?d}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Power, dval: f64) void {
        const a = self.a.eval();
        const b = self.b.eval();

        const c: f64 = if (a == std.math.e) 1 else std.math.log(std.math.e, a);

        self.a.diff(dval * (b * self.value / a));
        self.b.diff(dval * (self.value * c));

        std.debug.print("Power-diff: value: {?d}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Power) Node {
        return Node.init(self);
    }
};
