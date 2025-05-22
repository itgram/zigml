const std = @import("std");
const Node = @import("node.zig").Node;

/// Add two nodes
/// f = a + b
pub const Add = struct {
    value: ?f64,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Add {
        const ptr = try allocator.create(Add);
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Add) f64 {
        if (self.value) |v| {
            return v;
        }

        self.value = self.a.eval() + self.b.eval();

        std.debug.print("Add-eval: value: {?d}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Add, dval: f64) void {
        self.a.diff(dval);
        self.b.diff(dval);

        std.debug.print("Add-diff: value: {?d}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Add) Node {
        return Node.init(self);
    }
};
