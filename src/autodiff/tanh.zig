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

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Tanh {
        const ptr = try allocator.create(Tanh);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

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

    pub fn diff(self: *Tanh, dval: *Tensor) void {
        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, self.data, dval.data) |*v, vv, dv| {
            v.* = dv * (1 - vv * vv); // derivative of tanh is 1 - tanh^2
        }

        self.x.diff(grad);

        std.debug.print("Tanh-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Tanh) Node {
        return Node.init(self);
    }
};
