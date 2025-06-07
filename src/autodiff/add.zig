const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Add two nodes
/// where x and y are nodes that evaluate to tensors.
/// The Add node computes the element-wise sum of the tensors produced by its two input nodes.
/// It is used to represent addition operations in the computation graph.
/// The Add node is a fundamental operation in many neural networks and mathematical computations.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Add node is defined as:
/// f(x, y) = x + y
/// where x and y are the input tensors.
/// The Add node is typically used in conjunction with other nodes to build complex computation graphs.
pub const Add = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    /// Creates a new addition node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Add {
        const self = try allocator.create(Add);
        self.* = .{
            .allocator = allocator,
            .value = null,
            .x = x,
            .y = y,
        };

        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Add) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the add function.
    /// The add function is defined as:
    /// f(x, y) = x + y
    /// where x and y are the input tensors.
    /// The add function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn eval(self: *Add) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();
        const y = try self.y.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = xv + yv;
        }

        return self.value.?;
    }

    /// Compute the gradient of the add function.
    /// The gradient of the add function is defined as:
    /// ∂f/∂x = 1
    /// ∂f/∂y = 1
    /// where x and y are the input tensors.
    /// The gradient of the add function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Add, dval: *Tensor) !void {
        const grad_x = try Tensor.init(self.allocator, dval.shape);
        defer grad_x.deinit();
        const grad_y = try Tensor.init(self.allocator, dval.shape);
        defer grad_y.deinit();

        for (grad_x.data, grad_y.data, dval.data) |*gx, *gy, dv| {
            gx.* = dv;
            gy.* = dv;
        }

        try self.x.diff(grad_x);
        try self.y.diff(grad_y);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Add) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this addition node as a generic Node interface.
    pub fn node(self: *Add) Node {
        return Node.init(self);
    }
};

test "add basic" {
    const allocator = std.testing.allocator;

    // Create input tensors
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 1.0;
    xTensor.data[1] = 2.0;
    xTensor.data[2] = 3.0;
    xTensor.data[3] = 4.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 5.0;
    yTensor.data[1] = 6.0;
    yTensor.data[2] = 7.0;
    yTensor.data[3] = 8.0;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // Create add operation
    var add = try Add.init(allocator, x.node(), y.node());
    defer add.deinit();

    // Evaluate
    const result = try add.eval();
    const expected = [_]f64{ 6.0, 8.0, 10.0, 12.0 };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "add gradient" {
    const allocator = std.testing.allocator;

    // Create input tensors
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 1.0;
    xTensor.data[1] = 2.0;
    xTensor.data[2] = 3.0;
    xTensor.data[3] = 4.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 5.0;
    yTensor.data[1] = 6.0;
    yTensor.data[2] = 7.0;
    yTensor.data[3] = 8.0;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // Create add operation
    var add = try Add.init(allocator, x.node(), y.node());
    defer add.deinit();

    // Create gradient tensor
    const gradTensor = try Tensor.init(allocator, &[_]usize{4});
    defer gradTensor.deinit();
    gradTensor.data[0] = 1.0;
    gradTensor.data[1] = 2.0;
    gradTensor.data[2] = 3.0;
    gradTensor.data[3] = 4.0;

    // Compute gradients
    try add.diff(gradTensor);

    // Expected gradients for x and y
    const expected_grad = [_]f64{ 1.0, 2.0, 3.0, 4.0 };

    for (x.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    for (y.grad.data, expected_grad) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "add with different shapes" {
    const allocator = std.testing.allocator;

    // Create input tensors
    const xTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer xTensor.deinit();
    xTensor.data[0] = 1.0;
    xTensor.data[1] = 2.0;
    xTensor.data[2] = 3.0;
    xTensor.data[3] = 4.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer yTensor.deinit();
    yTensor.data[0] = 5.0;
    yTensor.data[1] = 6.0;
    yTensor.data[2] = 7.0;
    yTensor.data[3] = 8.0;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // Create add operation
    var add = try Add.init(allocator, x.node(), y.node());
    defer add.deinit();

    // Evaluate
    const result = try add.eval();
    const expected = [_]f64{ 6.0, 8.0, 10.0, 12.0 };

    for (result.data, expected) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}

test "add reset" {
    const allocator = std.testing.allocator;

    // Create input tensors
    const xTensor = try Tensor.init(allocator, &[_]usize{4});
    defer xTensor.deinit();
    xTensor.data[0] = 1.0;
    xTensor.data[1] = 2.0;
    xTensor.data[2] = 3.0;
    xTensor.data[3] = 4.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{4});
    defer yTensor.deinit();
    yTensor.data[0] = 5.0;
    yTensor.data[1] = 6.0;
    yTensor.data[2] = 7.0;
    yTensor.data[3] = 8.0;

    // Create variables
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // Create add operation
    var add = try Add.init(allocator, x.node(), y.node());
    defer add.deinit();

    // First evaluation
    const result1 = try add.eval();
    const expected1 = [_]f64{ 6.0, 8.0, 10.0, 12.0 };

    for (result1.data, expected1) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }

    // Reset
    add.reset();

    // Second evaluation
    const result2 = try add.eval();
    const expected2 = [_]f64{ 6.0, 8.0, 10.0, 12.0 };

    for (result2.data, expected2) |actual, exp| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
    }
}
