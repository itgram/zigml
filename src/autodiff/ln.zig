const std = @import("std");
const Node = @import("node.zig").Node;

/// Natural Logarithm node
/// f = ln(x)
pub const Ln = struct {
    value: ?f64,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Ln {
        const ptr = try allocator.create(Ln);
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Ln) f64 {
        if (self.value) |v| {
            return v;
        }

        self.value = std.math.log(std.math.e, self.x.eval());

        std.debug.print("Ln-eval: value: {?d}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Ln, dval: f64) void {
        self.x.diff(dval * (1 / self.x.eval()));

        std.debug.print("Ln-diff: value: {?d}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Ln) Node {
        return Node.init(self);
    }
};
