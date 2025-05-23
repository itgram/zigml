const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

pub const Variable = struct {
    name: []const u8,
    value: *Tensor,
    grad: *Tensor,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, value: *Tensor) !*Variable {
        const ptr = try allocator.create(Variable);
        ptr.name = name;
        ptr.value = value;
        ptr.grad = try Tensor.init(allocator, value.shape);
        ptr.grad.zero();

        return ptr;
    }

    pub fn eval(self: *Variable) *Tensor {
        std.debug.print("Variable-eval: {s}, value: {}, grad: {}\n", .{ self.name, self.value, self.grad });

        return self.value;
    }

    pub fn diff(self: *Variable, dval: *Tensor) void {
        for (self.grad.data, dval.data) |*g, dv| g.* += dv;

        std.debug.print("Variable-diff: {s}, value: {}, grad: {}, dval: {}\n", .{ self.name, self.value, self.grad, dval });
    }

    pub fn node(self: *Variable) Node {
        return Node.init(self);
    }
};
