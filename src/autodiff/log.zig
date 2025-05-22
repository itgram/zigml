const std = @import("std");
const Node = @import("node.zig").Node;

/// Logarithm node
/// f = log(x)
pub const Log = struct {
    value: ?f64,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Log {
        const ptr = try allocator.create(Log);
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Log) f64 {
        if (self.value) |v| {
            return v;
        }

        self.value = std.math.log(10, self.x.eval());

        std.debug.print("Log-eval: value: {?d}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Log, dval: f64) void {
        self.x.diff(dval * 1 / (self.x.eval() * std.math.ln10));

        std.debug.print("Log-diff: value: {?d}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Log) Node {
        return Node.init(self);
    }
};
