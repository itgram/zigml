const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Divide function node.
/// where x and y are nodes that evaluate to tensors.
/// The Divide node computes the element-wise division of the tensors produced by its two input nodes.
/// It is used to represent division operations in the computation graph.
/// The Divide node is a fundamental operation in many neural networks and mathematical computations.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Divide node is defined as:
/// f(x, y) = x / y
/// where x is the numerator tensor and y is the denominator tensor.
/// The Divide node is typically used in conjunction with other nodes to build complex computation graphs.
pub const Divide = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    /// Creates a new division node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Divide {
        const ptr = try allocator.create(Divide);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.y = y;

        return ptr;
    }

    /// Evaluate the divide function.
    /// The divide function is defined as:
    /// f(x, y) = x / y
    /// where x and y are the input tensors.
    /// The divide function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn eval(self: *Divide) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();
        const y = self.y.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = xv / yv;
        }

        std.debug.print("Divide-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the divide function.
    /// The gradient of the divide function is defined as:
    /// ∂f/∂x = 1 / y
    /// ∂f/∂y = -x / (y * y)
    /// where x and y are the input tensors.
    /// The gradient of the divide function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Divide, dval: *Tensor) void {
        const x = self.x.eval();
        const y = self.y.eval();

        const grad_x = Tensor.init(self.allocator, dval.shape) catch unreachable;
        const grad_y = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad_x.data, grad_y.data, x.data, y.data, dval.data) |*gx, *gy, xv, yv, dv| {
            gx.* = dv / yv;
            gy.* = -dv * xv / (yv * yv);
        }

        self.x.diff(grad_x);
        self.y.diff(grad_y);

        std.debug.print("Divide-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Divide) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this divide node as a generic Node interface.
    pub fn node(self: *Divide) Node {
        return Node.init(self);
    }
};
