const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Tan function node.
/// The Tan node represents the tangent function applied to a tensor.
/// It computes the tangent of each element in the input tensor.
/// The Tan node is used in neural networks and mathematical computations where the tangent function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = tan(x)
/// where x is the input tensor.
/// The tangent function is a periodic function that oscillates between -∞ and +∞.
/// The tangent function is often used in trigonometric calculations and periodic functions.
pub const Tan = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new tangent node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Tan {
        const ptr = try allocator.create(Tan);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Tan) void {
        if (self.value) |v| {
            v.deinit();
            self.allocator.destroy(v);
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the tangent function.
    /// The tangent function is defined as:
    /// f(x) = tan(x)
    /// where x is the input tensor.
    /// The tangent function is a periodic function that oscillates between -∞ and +∞.
    /// The tangent function is often used in trigonometric calculations and periodic functions.
    pub fn eval(self: *Tan) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.tan(xv);
        }

        std.debug.print("Tan-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the tangent function.
    /// The gradient of the tangent function is defined as:
    /// ∂f/∂x = sec^2(x)
    /// where x is the input tensor.
    /// The gradient of the tangent function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Tan, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            const sec2 = 1.0 / math.cos(xv);
            v.* = dv * sec2 * sec2;
        }

        self.x.diff(grad);

        std.debug.print("Tan-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Tan) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this tangent node as a generic Node interface.
    pub fn node(self: *Tan) Node {
        return Node.init(self);
    }
};
