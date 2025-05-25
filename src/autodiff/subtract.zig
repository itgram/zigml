const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Subtract function node.
/// The Subtract node represents the subtraction operation between two tensors.
/// It computes the element-wise difference between the two input tensors.
/// The Subtract node is used in neural networks and mathematical computations where subtraction is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(a, b) = a - b
/// where a and b are the input tensors.
/// The Subtract function is often used in loss functions and optimization algorithms.
pub const Subtract = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Subtract {
        const ptr = try allocator.create(Subtract);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Subtract) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const a = self.a.eval();
        const b = self.b.eval();

        self.value = Tensor.init(self.allocator, a.shape) catch null;

        for (self.value.?.data, a.data, b.data) |*v, av, bv| {
            v.* = av - bv;
        }

        std.debug.print("Subtract-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Subtract, dval: *Tensor) void {
        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;
        for (grad.data, dval.data) |*v, dv| {
            v.* = -dv;
        }

        self.a.diff(dval);
        self.b.diff(grad);

        std.debug.print("Subtract-diff: value: {?}, dval: {d}\n", .{ self.value, dval });
    }

    pub fn node(self: *Subtract) Node {
        return Node.init(self);
    }
};
