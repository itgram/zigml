const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Constant node.
/// Represents a constant value in the computation graph.
/// The Constant node is used to hold fixed values that do not change during the computation.
/// It is typically used to represent constants in mathematical expressions or parameters in neural networks.
/// The Constant node does not have any learnable parameters and does not require gradients.
/// It is a leaf node in the computation graph, meaning it does not have any dependencies on other nodes.
/// The Constant node is useful for representing fixed values that are used in computations,
pub const Constant = struct {
    value: *Tensor,

    /// Creates a new constant node with the given tensor value.
    pub fn init(allocator: std.mem.Allocator, value: *Tensor) !*Constant {
        const ptr = try allocator.create(Constant);
        ptr.value = value;

        return ptr;
    }

    /// Evaluate the constant function.
    /// The constant function is defined as:
    /// f(x) = value
    /// where x is the input tensor.
    /// The constant function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn eval(self: *Constant) *Tensor {
        std.debug.print("Constant-eval: {}\n", .{self.value});

        return self.value;
    }

    /// Compute the gradient of the constant function.
    /// The gradient of the constant function is defined as:
    /// ∂f/∂x = 0
    /// where x is the input tensor.
    /// The gradient of the constant function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Constant, dval: *Tensor) void {
        std.debug.print("Constant-diff: {}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state.
    /// For constant nodes, this is a no-op since they don't have any cached values.
    pub fn reset(_: *Constant) void {
        // Constants don't need to be reset as they don't have any cached values
    }

    /// Returns this constant node as a generic Node interface.
    pub fn node(self: *Constant) Node {
        return Node.init(self);
    }
};
