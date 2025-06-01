const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Tanh function node.
/// The Tanh (hyperbolic tangent) function.
/// The Tanh function maps any real-valued number to the (-1, 1) interval.
/// The Tanh function is commonly used in neural networks as an activation function.
/// It is particularly useful for hidden layers in neural networks.
/// The Tanh function is differentiable everywhere, making it suitable for backpropagation in neural networks.
/// It is defined as:
/// f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// where e is the base of the natural logarithm.
/// and x is the input tensor.
/// The Tanh function is a smooth, continuous function that is symmetric around the origin.
pub const Tanh = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new tanh node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Tanh {
        const ptr = try allocator.create(Tanh);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Tanh) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the hyperbolic tangent function.
    /// The hyperbolic tangent function is defined as:
    /// f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    /// where e is the base of the natural logarithm.
    /// and x is the input tensor.
    /// The hyperbolic tangent function is a smooth, continuous function that is symmetric around the origin.
    pub fn eval(self: *Tanh) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.tanh(xv);
        }

        std.debug.print("Tanh-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the hyperbolic tangent function.
    /// The gradient of the hyperbolic tangent function is defined as:
    /// ∂f/∂x = 1 - tanh^2(x)
    /// where x is the input tensor.
    /// The gradient of the hyperbolic tangent function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Tanh, dval: *Tensor) void {
        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;
        defer grad.deinit();

        for (grad.data, self.value.?.data, dval.data) |*v, vv, dv| {
            v.* = dv * (1 - vv * vv);
        }
        self.x.diff(grad);

        std.debug.print("Tanh-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Tanh) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this tanh node as a generic Node interface.
    pub fn node(self: *Tanh) Node {
        return Node.init(self);
    }
};
