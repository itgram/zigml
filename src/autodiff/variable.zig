const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Variable node.
/// The Variable node represents a variable in the computation graph.
/// It holds a tensor value and its gradient.
/// The Variable node is used to store parameters in machine learning models.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// Variable(name, value)
/// where `name` is the name of the variable and `value` is the tensor value.
/// The Variable node is commonly used in neural networks to represent weights and biases.
/// It is also used in optimization algorithms to update parameters during training.
/// The Variable node is differentiable, allowing gradients to be computed for backpropagation.
/// The Variable node can be evaluated to get the current value of the tensor.
/// The Variable node can compute the gradient with respect to the value tensor.
/// The Variable node can be used in a computation graph to represent trainable parameters.
/// It is useful for implementing optimization algorithms like gradient descent.
/// The Variable node can be used in conjunction with other nodes like Sigmoid, Cos, Sin, etc.
/// The Variable node is a fundamental building block in machine learning frameworks.
pub const Variable = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    value: *Tensor,
    grad: *Tensor,

    /// Creates a new variable node with the given tensor value.
    pub fn init(allocator: std.mem.Allocator, name: []const u8, value: *Tensor) !*Variable {
        const ptr = try allocator.create(Variable);
        ptr.allocator = allocator;
        ptr.name = name;
        ptr.value = value;
        ptr.grad = try Tensor.init(allocator, value.shape);
        ptr.grad.zero();

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Variable) void {
        self.grad.deinit();
        self.allocator.destroy(self);
    }

    /// Evaluate the variable.
    /// The variable is defined as:
    /// f(x) = x
    /// where x is the input tensor.
    /// The variable is often used in neural networks to represent weights and biases.
    /// It is also used in optimization algorithms to update parameters during training.
    /// The variable is differentiable, allowing gradients to be computed for backpropagation.
    pub fn eval(self: *Variable) !*Tensor {
        std.debug.print("Variable-eval: {s}, value: {}, grad: {}\n", .{ self.name, self.value, self.grad });

        return self.value;
    }

    /// Compute the gradient of the variable.
    /// The gradient of the variable is defined as:
    /// ∂f/∂x = 1
    /// where x is the input tensor.
    /// The gradient of the variable is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Variable, dval: *Tensor) void {
        for (self.grad.data, dval.data) |*g, dv| {
            g.* += dv;
        }

        std.debug.print("Variable-diff: {s}, value: {}, grad: {}, dval: {}\n", .{ self.name, self.value, self.grad, dval });
    }

    /// Resets the node's state by clearing accumulated gradients.
    /// This is useful when you want to start a new gradient computation.
    pub fn reset(self: *Variable) void {
        self.grad.zero();
    }

    /// Returns this variable node as a generic Node interface.
    pub fn node(self: *Variable) Node {
        return Node.init(self);
    }
};
