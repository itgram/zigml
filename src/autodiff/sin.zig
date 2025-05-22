const std = @import("std");
const Node = @import("node.zig").Node;

/// Sine function node.
/// f = sin(x)
pub const Sin = struct {
    value: ?f64,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Sin {
        const ptr = try allocator.create(Sin);
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Sin) f64 {
        if (self.value) |v| {
            return v;
        }

        self.value = std.math.sin(self.x.eval());

        std.debug.print("Sin-eval: value: {?d}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Sin, dval: f64) void {
        self.x.diff(dval * std.math.cos(self.x.eval()));

        std.debug.print("Sin-diff: value: {?d}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Sin) Node {
        return Node.init(self);
    }
};
