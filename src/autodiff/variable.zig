const std = @import("std");
const Node = @import("node.zig").Node;

pub const Variable = struct {
    name: []const u8,
    value: f64,
    grad: f64,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, value: f64) !*Variable {
        const ptr = try allocator.create(Variable);
        ptr.name = name;
        ptr.value = value;
        ptr.grad = 0.0;

        return ptr;
    }

    pub fn eval(self: *Variable) f64 {
        std.debug.print("Variable-eval: {s}, value: {d}, grad: {?d}\n", .{ self.name, self.value, self.grad });

        return self.value;
    }

    pub fn diff(self: *Variable, dval: f64) void {
        self.grad += dval;

        std.debug.print("Variable-diff: {s}, value: {d}, grad: {?d}, dval: {d}\n", .{ self.name, self.value, self.grad, dval });
    }

    pub fn node(self: *Variable) Node {
        return Node.init(self);
    }
};
