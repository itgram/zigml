const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Multiply function node.
/// The Multiply node represents the element-wise multiplication of two tensors.
/// It computes the product of each corresponding element in the input tensors.
/// The Multiply node is used in various mathematical computations and neural networks where multiplication is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Multiply node is defined as:
/// f(x, y) = x * y
/// where x and y are the input tensors.
/// The Multiply node is typically used in conjunction with other nodes to build complex computation graphs.
/// It is commonly used in neural networks for operations such as weight updates and loss calculations.
pub const Multiply = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Multiply {
        const ptr = try allocator.create(Multiply);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.y = y;

        return ptr;
    }

    /// Evaluate the multiply function.
    /// The multiply function is defined as:
    /// f(x, y) = x * y
    /// where x and y are the input tensors.
    /// The multiply function is often used in conjunction with other nodes to build complex computation graphs.
    pub fn eval(self: *Multiply) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();
        const y = self.y.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = xv * yv;
        }

        std.debug.print("Multiply-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the multiply function.
    /// The gradient of the multiply function is defined as:
    /// ∂f / ∂x = y
    /// ∂f / ∂y = x
    /// where x and y are the input tensors.
    /// The gradient of the multiply function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Multiply, dval: *Tensor) void {
        const x = self.x.eval();
        const y = self.y.eval();

        const grad_x = Tensor.init(self.allocator, dval.shape) catch unreachable;
        const grad_y = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad_x.data, grad_y.data, x.data, y.data, dval.data) |*gx, *gy, xv, yv, dv| {
            gx.* = dv * yv;
            gy.* = dv * xv;
        }

        self.x.diff(grad_x);
        self.y.diff(grad_y);

        std.debug.print("Multiply-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Multiply) Node {
        return Node.init(self);
    }
};
