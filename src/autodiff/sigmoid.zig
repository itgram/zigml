const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Sigmoid function node.
/// The Sigmoid function is defined as:
/// f(x) = σ(x) = 1 / (1 + exp(-x))
/// where σ is the sigmoid function.
/// The Sigmoid function maps any real-valued number to the (0, 1) interval.
/// The Sigmoid function is commonly used in neural networks as an activation function.
/// It is particularly useful for binary classification tasks.
/// The Sigmoid function is differentiable everywhere, making it suitable for backpropagation in neural networks.
pub const Sigmoid = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new sigmoid node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Sigmoid {
        const ptr = try allocator.create(Sigmoid);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Sigmoid) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the Sigmoid function.
    /// The Sigmoid function is defined as:
    /// f(x) = 1 / (1 + exp(-x))
    /// where x is the input tensor.
    /// The Sigmoid function maps any real-valued number to the (0, 1) interval.
    /// The Sigmoid function is commonly used in neural networks as an activation function.
    /// It is particularly useful for binary classification tasks.
    pub fn eval(self: *Sigmoid) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = 1.0 / (1.0 + math.exp(-xv));
        }

        std.debug.print("Sigmoid-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the Sigmoid function.
    /// The gradient of the Sigmoid function is defined as:
    /// ∂f/∂x = σ(x) * (1 - σ(x))
    /// where x is the input tensor.
    /// The gradient of the Sigmoid function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Sigmoid, dval: *Tensor) !void {
        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, self.value.?.data, dval.data) |*v, vv, dv| {
            v.* = dv * vv * (1 - vv); // Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
        }

        try self.x.diff(grad);

        std.debug.print("Sigmoid-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Sigmoid) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this sigmoid node as a generic Node interface.
    pub fn node(self: *Sigmoid) Node {
        return Node.init(self);
    }
};
