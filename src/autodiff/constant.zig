const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

pub const Constant = struct {
    value: *Tensor,

    pub fn init(allocator: std.mem.Allocator, value: *Tensor) !*Constant {
        const ptr = try allocator.create(Constant);
        ptr.value = value;

        return ptr;
    }

    pub fn eval(self: *Constant) *Tensor {
        std.debug.print("Constant-eval: {}\n", .{self.value});

        return self.value;
    }

    pub fn diff(self: *Constant, dval: *Tensor) void {
        std.debug.print("Constant-diff: {}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Constant) Node {
        return Node.init(self);
    }
};
