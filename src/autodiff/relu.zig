const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// ReLU function node.
/// The ReLU (Rectified Linear Unit) activation function
/// It is commonly used in neural networks as an activation function.
/// It is defined as:
/// f(x) = x if x > 0 else 0
/// - For positive inputs: f(x) = x
/// - For negative inputs: f(x) = 0
/// where x is the input tensor.
/// The ReLU function is non-linear and allows for faster training of deep neural networks.
pub const ReLU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new ReLU node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*ReLU {
        const ptr = try allocator.create(ReLU);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *ReLU) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the ReLU function.
    /// The ReLU function is defined as:
    /// f(x) = x if x > 0 else 0
    /// where x is the input tensor.
    /// The ReLU function is non-linear and allows for faster training of deep neural networks.
    pub fn eval(self: *ReLU) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) xv else 0;
        }

        std.debug.print("ReLU-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the ReLU function.
    /// The gradient of the ReLU function is defined as:
    /// ∂f/∂x = 1 if x > 0 else 0
    /// where x is the input tensor.
    /// The gradient of the ReLU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *ReLU, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = if (xv > 0) dv else 0;
        }

        try self.x.diff(grad);

        std.debug.print("ReLU-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *ReLU) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this ReLU node as a generic Node interface.
    pub fn node(self: *ReLU) Node {
        return Node.init(self);
    }
};
