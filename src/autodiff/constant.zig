const std = @import("std");
const Node = @import("node.zig").Node;

pub const Constant = struct {
    value: f64,

    pub fn init(allocator: std.mem.Allocator, value: f64) !*Constant {
        const ptr = try allocator.create(Constant);
        ptr.value = value;

        return ptr;
    }

    pub fn eval(self: *Constant) f64 {
        std.debug.print("Constant-eval: {d}\n", .{self.value});

        return self.value;
    }

    pub fn diff(self: *Constant, dval: f64) void {
        std.debug.print("Constant-diff: {d}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Constant) Node {
        return Node.init(self);
    }
};
