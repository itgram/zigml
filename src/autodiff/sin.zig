const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Sin function node.
/// The Sin node represents the sine function applied to a tensor.
/// It computes the sine of each element in the input tensor.
/// The Sin node is used in neural networks and mathematical computations where the sine function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = sin(x)
/// where x is the input tensor.
/// The Sine function is a periodic function that oscillates between -1 and 1.
/// The sine function is often used in trigonometric calculations and periodic functions.
pub const Sin = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new sine node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Sin {
        const ptr = try allocator.create(Sin);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Evaluate the sine function.
    /// The sine function is defined as:
    /// f(x) = sin(x)
    /// where x is the input tensor.
    /// The sine function is a periodic function that oscillates between -1 and 1.
    /// The sine function is often used in trigonometric calculations and periodic functions.
    pub fn eval(self: *Sin) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.sin(xv);
        }

        std.debug.print("Sin-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the sine function.
    /// The gradient of the sine function is defined as:
    /// ∂f/∂x = cos(x)
    /// where x is the input tensor.
    /// The gradient of the sine function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Sin, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv * math.cos(xv);
        }

        self.x.diff(grad);

        std.debug.print("Sin-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Returns this sine node as a generic Node interface.
    pub fn node(self: *Sin) Node {
        return Node.init(self);
    }
};
