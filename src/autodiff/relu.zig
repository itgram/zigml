const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Relu function node.
/// The ReLU (Rectified Linear Unit) activation function
/// It is commonly used in neural networks as an activation function.
/// It is defined as:
/// f(x) = x if x > 0 else 0
/// - For positive inputs: f(x) = x
/// - For negative inputs: f(x) = 0
/// where x is the input tensor.
/// The ReLU function is non-linear and allows for faster training of deep neural networks.
pub const Relu = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Relu {
        const ptr = try allocator.create(Relu);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Evaluate the ReLU function.
    /// The ReLU function is defined as:
    /// f(x) = x if x > 0 else 0
    /// where x is the input tensor.
    /// The ReLU function is non-linear and allows for faster training of deep neural networks.
    pub fn eval(self: *Relu) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) xv else 0;
        }

        std.debug.print("Relu-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the ReLU function.
    /// The gradient of the ReLU function is defined as:
    /// ∂f / ∂x = 1 if x > 0 else 0
    /// where x is the input tensor.
    /// The gradient of the ReLU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Relu, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = if (xv > 0) dv else 0;
        }

        self.x.diff(grad);

        std.debug.print("Relu-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Relu) Node {
        return Node.init(self);
    }
};
