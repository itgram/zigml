const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Ln function node.
/// The natural logarithm function, which is the logarithm to the base e.
/// It is defined as the inverse of the exponential function.
/// The Ln node computes the natural logarithm of each element in the input tensor.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Ln node is commonly used in various mathematical computations and neural networks.
/// It is defined as:
/// f(x) = ln(x)
/// where x is the input tensor.
/// The natural logarithm is often used in optimization problems and loss functions.
pub const Ln = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new natural logarithm node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Ln {
        const ptr = try allocator.create(Ln);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Ln) void {
        if (self.value) |v| {
            v.deinit();
            self.allocator.destroy(v);
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the natural logarithm function.
    /// The natural logarithm function is defined as:
    /// f(x) = ln(x)
    /// where x is the input tensor.
    /// The natural logarithm is often used in optimization problems and loss functions.
    pub fn eval(self: *Ln) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.log(math.e, xv);
        }

        std.debug.print("Ln-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// The gradient of the natural logarithm function is defined as:
    /// ∂f/∂x = 1 / x
    /// where x is the input tensor.
    /// The gradient of the natural logarithm function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Ln, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv / xv;
        }

        self.x.diff(grad);

        std.debug.print("Ln-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Ln) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this natural logarithm node as a generic Node interface.
    pub fn node(self: *Ln) Node {
        return Node.init(self);
    }
};
