const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Subtract function node.
/// The Subtract node represents the subtraction operation between two tensors.
/// It computes the element-wise difference between the two input tensors.
/// The Subtract node is used in neural networks and mathematical computations where subtraction is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x, y) = x - y
/// where x and y are the input tensors.
/// The Subtract function is often used in loss functions and optimization algorithms.
pub const Subtract = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    /// Creates a new subtraction node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Subtract {
        const ptr = try allocator.create(Subtract);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.y = y;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Subtract) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the subtract function.
    /// The subtract function is defined as:
    /// f(x, y) = x - y
    /// where x and y are the input tensors.
    /// The subtract function is often used in loss functions and optimization algorithms.
    pub fn eval(self: *Subtract) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();
        const y = self.y.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = xv - yv;
        }

        std.debug.print("Subtract-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the subtract function.
    /// The gradient of the subtract function is defined as:
    /// ∂f/∂x = 1
    /// ∂f/∂y = -1
    /// where x and y are the input tensors.
    /// The gradient of the subtract function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Subtract, dval: *Tensor) void {
        const grad_x = Tensor.init(self.allocator, dval.shape) catch unreachable;
        defer grad_x.deinit();
        const grad_y = Tensor.init(self.allocator, dval.shape) catch unreachable;
        defer grad_y.deinit();

        for (grad_x.data, grad_y.data, dval.data) |*gx, *gy, dv| {
            gx.* = dv;
            gy.* = -dv;
        }

        self.x.diff(grad_x);
        self.y.diff(grad_y);

        std.debug.print("Subtract-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Subtract) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this subtract node as a generic Node interface.
    pub fn node(self: *Subtract) Node {
        return Node.init(self);
    }
};
