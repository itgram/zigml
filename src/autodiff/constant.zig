const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;
const Variable = @import("variable.zig").Variable;

/// Constant node.
/// Represents a constant value in the computation graph.
/// The Constant node is used to hold fixed values that do not change during the computation.
/// It is typically used to represent constants in mathematical expressions or parameters in neural networks.
/// The Constant node does not have any learnable parameters and does not require gradients.
/// It is a leaf node in the computation graph, meaning it does not have any dependencies on other nodes.
/// The Constant node is useful for representing fixed values that are used in computations,
pub const Constant = struct {
    allocator: std.mem.Allocator,
    value: *Tensor,

    /// Creates a new constant node with the given tensor value.
    pub fn init(allocator: std.mem.Allocator, value: *Tensor) !*Constant {
        const self = try allocator.create(Constant);
        self.* = .{
            .allocator = allocator,
            .value = value,
        };

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Constant) void {
        self.allocator.destroy(self);
    }

    /// Evaluate the constant function.
    /// The constant function is defined as:
    /// f(x) = value
    /// where x is the input tensor.
    /// The constant function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn eval(self: *Constant) !*Tensor {
        return self.value;
    }

    /// Compute the gradient of the constant function.
    /// The gradient of the constant function is defined as:
    /// ∂f/∂x = 0
    /// where x is the input tensor.
    /// The gradient of the constant function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(_: *Constant, _: *Tensor) !void {}

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

test "constant basic" {
    const allocator = std.testing.allocator;

    // Create constant tensor
    const valueTensor = try Tensor.init(allocator, &[_]usize{4});
    defer valueTensor.deinit();
    valueTensor.data[0] = 1.0;
    valueTensor.data[1] = 2.0;
    valueTensor.data[2] = 3.0;
    valueTensor.data[3] = 4.0;

    // Create constant node
    var constant = try Constant.init(allocator, valueTensor);
    defer constant.deinit();

    // Evaluate
    const result = try constant.eval();
    const expected = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "constant gradient" {
    const allocator = std.testing.allocator;

    // Create constant tensor
    const valueTensor = try Tensor.init(allocator, &[_]usize{4});
    defer valueTensor.deinit();
    valueTensor.data[0] = 1.0;
    valueTensor.data[1] = 2.0;
    valueTensor.data[2] = 3.0;
    valueTensor.data[3] = 4.0;

    // Create constant node
    var constant = try Constant.init(allocator, valueTensor);
    defer constant.deinit();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 2.0;
    gradTensor.data[2] = 3.0;
    gradTensor.data[3] = 4.0;

    // Compute gradients
    try constant.diff(gradTensor);

    // Verify that the constant node's value remains unchanged after gradient computation
    const result = try constant.eval();
    const expected = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "constant with different shapes" {
    const allocator = std.testing.allocator;

    // Create constant tensor
    const valueTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer valueTensor.deinit();
    valueTensor.data[0] = 1.0;
    valueTensor.data[1] = 2.0;
    valueTensor.data[2] = 3.0;
    valueTensor.data[3] = 4.0;

    // Create constant node
    var constant = try Constant.init(allocator, valueTensor);
    defer constant.deinit();

    // Evaluate
    const result = try constant.eval();
    const expected = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "constant reset" {
    const allocator = std.testing.allocator;

    // Create constant tensor
    const valueTensor = try Tensor.init(allocator, &[_]usize{4});
    defer valueTensor.deinit();
    valueTensor.data[0] = 1.0;
    valueTensor.data[1] = 2.0;
    valueTensor.data[2] = 3.0;
    valueTensor.data[3] = 4.0;

    // Create constant node
    var constant = try Constant.init(allocator, valueTensor);
    defer constant.deinit();

    // First evaluation
    const result1 = try constant.eval();
    const expected1 = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset (should be a no-op for constants)
    constant.reset();

    // Second evaluation
    const result2 = try constant.eval();
    const expected2 = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
