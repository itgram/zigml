const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Cos function node.
/// The Cos node represents the cosine function applied to a tensor.
/// It computes the cosine of each element in the input tensor.
/// The Cos node is used in neural networks and mathematical computations where the cosine function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = cos(x)
/// where x is the input tensor.
/// The Cosine function is a periodic function that oscillates between -1 and 1.
/// The cosine function is often used in trigonometric calculations and periodic functions.
pub const Cos = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new cosine node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Cos {
        const ptr = try allocator.create(Cos);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Evaluate the cosine function.
    /// The cosine function is defined as:
    /// f(x) = cos(x)
    /// where x is the input tensor.
    /// The cosine function is a periodic function that oscillates between -1 and 1.
    /// The cosine function is often used in trigonometric calculations and periodic functions.
    pub fn eval(self: *Cos) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.cos(xv);
        }

        std.debug.print("Cos-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the cosine function.
    /// The gradient of the cosine function is defined as:
    /// ∂f/∂x = -sin(x)
    /// where x is the input tensor.
    /// The gradient of the cosine function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Cos, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv * -math.sin(xv); // derivative of cos is -sin
        }

        self.x.diff(grad);

        std.debug.print("Cos-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Returns this cosine node as a generic Node interface.
    pub fn node(self: *Cos) Node {
        return Node.init(self);
    }
};
