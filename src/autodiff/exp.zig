const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Exp function node.
/// The Exp node represents the exponential function applied to a tensor.
/// It computes the exponential of each element in the input tensor.
/// The Exp node is used in neural networks and mathematical computations where the exponential function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = e^x
/// where x is the input tensor.
/// The exponential function is often used in activation functions and probability distributions.
pub const Exp = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new exponential node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Exp {
        const ptr = try allocator.create(Exp);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Evaluate the exponential function.
    /// The exponential function is defined as:
    /// f(x) = e^x
    /// where x is the input tensor.
    /// The exponential function is often used in activation functions and probability distributions.
    pub fn eval(self: *Exp) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.pow(math.e, xv);
        }

        std.debug.print("Exp-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the exponential function.
    /// The gradient of the exponential function is defined as:
    /// ∂f/∂x = e^x
    /// where x is the input tensor.
    /// The gradient of the exponential function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Exp, dval: *Tensor) void {
        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, self.value.?.data, dval.data) |*v, vv, dv| {
            v.* = dv * vv;
        }
        self.x.diff(grad);

        std.debug.print("Exp-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Exp) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this exponential node as a generic Node interface.
    pub fn node(self: *Exp) Node {
        return Node.init(self);
    }
};
