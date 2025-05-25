const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Constant node.
/// Represents a constant value in the computation graph.
/// The Constant node is used to hold fixed values that do not change during the computation.
/// It is typically used to represent constants in mathematical expressions or parameters in neural networks.
/// The Constant node does not have any learnable parameters and does not require gradients.
/// It is a leaf node in the computation graph, meaning it does not have any dependencies on other nodes.
/// The Constant node is useful for representing fixed values that are used in computations,
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
