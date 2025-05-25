const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Divide function node.
/// where a and b are nodes that evaluate to tensors.
/// The Divide node computes the element-wise division of the tensors produced by its two input nodes.
/// It is used to represent division operations in the computation graph.
/// The Divide node is a fundamental operation in many neural networks and mathematical computations.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Divide node is defined as:
/// f(a, b) = a / b
/// where a is the numerator tensor and b is the denominator tensor.
/// The Divide node is typically used in conjunction with other nodes to build complex computation graphs.
pub const Divide = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Divide {
        const ptr = try allocator.create(Divide);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    pub fn eval(self: *Divide) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const a = self.a.eval();
        const b = self.b.eval();

        self.value = Tensor.init(self.allocator, a.shape) catch null;

        for (self.value.?.data, a.data, b.data) |*v, av, bv| {
            v.* = av / bv;
        }

        std.debug.print("Divide-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Divide, dval: *Tensor) void {
        const a = self.a.eval();
        const b = self.b.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, b.data, dval.data) |*v, bv, dv| {
            v.* = dv / bv;
        }
        self.a.diff(grad);

        for (grad.data, a.data, b.data, dval.data) |*v, av, bv, dv| {
            v.* = -(dv * av) / (bv * bv);
        }
        self.b.diff(grad);

        std.debug.print("Divide-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Divide) Node {
        return Node.init(self);
    }
};
