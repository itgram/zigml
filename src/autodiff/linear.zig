const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Linear function node.
/// The Linear node represents a linear transformation of the input tensor.
/// It simply passes the input tensor through without any modification.
/// This is often used as a baseline or identity function in neural networks.
/// The Linear node is useful for testing and debugging purposes, as it does not introduce any non-linearity.
/// It can also be used in conjunction with other activation functions to create more complex models.
/// The Linear node supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = x
/// where x is the input tensor.
/// The Linear function is often used in the output layer of neural networks for regression tasks.
/// It is also used in the hidden layers of neural networks when no activation function is applied.
pub const Linear = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new linear node with the given input, weight, and bias nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Linear {
        const ptr = try allocator.create(Linear);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Linear) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the linear function.
    /// The linear function is defined as:
    /// f(x) = x
    /// where x is the input tensor.
    /// The linear function is often used in the output layer of neural networks for regression tasks.
    /// It is also used in the hidden layers of neural networks when no activation function is applied.
    pub fn eval(self: *Linear) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = xv;
        }

        std.debug.print("Linear-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the linear function.
    /// The gradient of the linear function is defined as:
    /// ∂f/∂x = 1
    /// where x is the input tensor.
    /// The gradient of the linear function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Linear, dval: *Tensor) void {
        self.x.diff(dval);

        std.debug.print("Linear-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Linear) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this linear node as a generic Node interface.
    pub fn node(self: *Linear) Node {
        return Node.init(self);
    }
};
