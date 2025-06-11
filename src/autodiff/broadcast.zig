const std = @import("std");
const autodiff = @import("autodiff.zig");
const Node = autodiff.Node;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

/// Error type for broadcast operations.
const BroadcastError = error{
    /// The shapes of the input and target are not compatible for broadcasting.
    IncompatibleShapes,
};

/// Broadcast node.
/// The Broadcast node expands a tensor to match the shape of another tensor.
/// It is defined as:
/// Broadcast(x, shape)
/// where `x` is the input tensor and `shape` is the desired output shape.
/// The Broadcast node is commonly used in neural networks to make tensors compatible for arithmetic operations.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
pub const Broadcast = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    shape: []const usize, // target shape
    x: Node,

    /// Creates a new broadcast node with the given input node and shape.
    pub fn init(allocator: std.mem.Allocator, x: Node, shape: []const usize) !*Broadcast {
        const shape_copy = try allocator.alloc(usize, shape.len);
        errdefer allocator.free(shape_copy);

        std.mem.copyForwards(usize, shape_copy, shape);

        const self = try allocator.create(Broadcast);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .value = null,
            .shape = shape_copy,
            .x = x,
        };
        return self;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Broadcast) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.free(self.shape);
        self.allocator.destroy(self);
    }

    /// Evaluates the broadcast operation.
    /// The broadcast operation expands the input tensor to match the target shape.
    /// The broadcasting rules are:
    /// 1. If the shapes have different lengths, pad the shorter shape with 1s
    /// 2. For each dimension, if the sizes are equal, keep them as is
    /// 3. If one size is 1, expand it to match the other size
    /// 4. If neither size is 1 and they are not equal, raise an error
    pub fn eval(self: *Broadcast) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        // Check if the shapes are compatible for broadcasting
        if (!isBroadcastCompatible(x.shape, self.shape)) {
            return error.IncompatibleShapes;
        }

        // Create output tensor with target shape
        self.value = try Tensor.init(self.allocator, self.shape);

        // Broadcast the input tensor to the output shape
        var x_indices = try self.allocator.alloc(usize, x.shape.len);
        defer self.allocator.free(x_indices);
        @memset(x_indices, 0);

        var out_indices = try self.allocator.alloc(usize, self.shape.len);
        defer self.allocator.free(out_indices);
        @memset(out_indices, 0);

        // Iterate through all elements of the output tensor
        var out_idx: usize = 0;
        while (out_idx < self.value.?.size) : (out_idx += 1) {
            // Compute input tensor index
            var x_idx: usize = 0;
            for (x_indices, x.strides) |idx, stride| {
                x_idx += idx * stride;
            }

            // Copy value from input to output
            self.value.?.data[out_idx] = x.data[x_idx];

            // Update indices
            var i: usize = self.shape.len;
            while (i > 0) {
                i -= 1;
                out_indices[i] += 1;
                if (out_indices[i] < self.shape[i]) {
                    break;
                }
                out_indices[i] = 0;
            }

            // Update input indices based on broadcasting rules
            i = x.shape.len;
            while (i > 0) {
                i -= 1;
                if (i >= self.shape.len) {
                    x_indices[i] = 0;
                } else if (x.shape[i] == 1) {
                    x_indices[i] = 0;
                } else {
                    x_indices[i] = out_indices[i];
                }
            }
        }

        return self.value.?;
    }

    /// Compute the gradient of the broadcast operation.
    /// The gradient is computed by summing the gradients of the broadcasted elements.
    pub fn diff(self: *Broadcast, dval: *Tensor) !void {
        const x = try self.x.eval();

        // Create gradient tensor with input shape
        const grad_x = try Tensor.init(self.allocator, x.shape);
        defer grad_x.deinit();
        grad_x.zeros();

        // Accumulate gradients from output to input
        var x_indices = try self.allocator.alloc(usize, x.shape.len);
        defer self.allocator.free(x_indices);
        @memset(x_indices, 0);

        var out_indices = try self.allocator.alloc(usize, self.shape.len);
        defer self.allocator.free(out_indices);
        @memset(out_indices, 0);

        // Iterate through all elements of the output tensor
        var out_idx: usize = 0;
        while (out_idx < dval.size) : (out_idx += 1) {
            // Compute input tensor index
            var x_idx: usize = 0;
            for (x_indices, x.strides) |idx, stride| {
                x_idx += idx * stride;
            }

            // Accumulate gradient
            grad_x.data[x_idx] += dval.data[out_idx];

            // Update indices
            var i: usize = self.shape.len;
            while (i > 0) {
                i -= 1;
                out_indices[i] += 1;
                if (out_indices[i] < self.shape[i]) {
                    break;
                }
                out_indices[i] = 0;
            }

            // Update input indices based on broadcasting rules
            i = x.shape.len;
            while (i > 0) {
                i -= 1;
                if (i >= self.shape.len) {
                    x_indices[i] = 0;
                } else if (x.shape[i] == 1) {
                    x_indices[i] = 0;
                } else {
                    x_indices[i] = out_indices[i];
                }
            }
        }

        try self.x.diff(grad_x);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Broadcast) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this broadcast node as a generic Node interface.
    pub fn node(self: *Broadcast) Node {
        return Node.init(self);
    }
};

fn isBroadcastCompatible(input: []const usize, target: []const usize) bool {
    const in_len = input.len;
    const tgt_len = target.len;
    const pad = tgt_len -| in_len;

    for (0..tgt_len) |i| {
        const in_dim = if (i < pad) 1 else input[i - pad];
        const tgt_dim = target[i];

        if (in_dim != tgt_dim and in_dim != 1) return false;
    }

    return true;
}

test "broadcast basic" {
    const allocator = std.testing.allocator;

    // Test case: invalid broadcast [2] to [2, 3] should raise error
    {
        const xTensor = try Tensor.init(allocator, &[_]usize{2});
        defer xTensor.deinit();
        xTensor.data[0] = 1.0;
        xTensor.data[1] = 2.0;

        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        var broadcast_op = try Broadcast.init(allocator, x.node(), &[_]usize{ 2, 3 });
        defer broadcast_op.deinit();

        const result = broadcast_op.eval();
        try std.testing.expectError(error.IncompatibleShapes, result);
    }

    // Test case: broadcast [1, 2] to [2, 2]
    {
        // Create input tensor with shape [1, 2]
        const xTensor = try Tensor.init(allocator, &[_]usize{ 1, 2 });
        defer xTensor.deinit();
        xTensor.data[0] = 1.0;
        xTensor.data[1] = 2.0;

        // Create variable
        var x = try Variable.init(allocator, "x", xTensor);
        defer x.deinit();

        // Create broadcast operation to shape [2, 2]
        var broadcast_op = try Broadcast.init(allocator, x.node(), &[_]usize{ 2, 2 });
        defer broadcast_op.deinit();

        // Evaluate forward pass
        const result = try broadcast_op.eval();

        // Expected values:
        // [[1.0, 2.0],
        //  [1.0, 2.0]]
        const expected = [_]f64{
            1.0, 2.0,
            1.0, 2.0,
        };

        for (result.data, expected) |actual, exp| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-6);
        }

        // Test gradient computation
        const gradTensor = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer gradTensor.deinit();
        for (gradTensor.data) |*v| {
            v.* = 1.0;
        }

        try broadcast_op.diff(gradTensor);

        // Expected gradients for input:
        // [[2.0, 2.0]] (sum of gradients for each broadcasted element)
        try std.testing.expectApproxEqAbs(2.0, x.grad.data[0], 1e-6);
        try std.testing.expectApproxEqAbs(2.0, x.grad.data[1], 1e-6);
    }
}

test "broadcast allocation failure" {
    const allocator = std.testing.allocator;

    // Create input tensor with shape [2]
    const xTensor = try Tensor.init(allocator, &[_]usize{2});
    defer xTensor.deinit();
    xTensor.data[0] = 1.0;
    xTensor.data[1] = 2.0;

    // Create variable
    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    // Test Broadcast struct allocation failure
    {
        var failing_allocator = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });

        const result = Broadcast.init(failing_allocator.allocator(), x.node(), &[_]usize{ 2, 3 });
        try std.testing.expectError(error.OutOfMemory, result);
    }

    // Test target shape allocation failure
    {
        var failing_allocator = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 1 });

        const result = Broadcast.init(failing_allocator.allocator(), x.node(), &[_]usize{ 2, 3 });
        try std.testing.expectError(error.OutOfMemory, result);
    }
}
