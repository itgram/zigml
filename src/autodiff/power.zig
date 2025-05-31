const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Power function node.
/// where a and b are nodes representing tensors.
/// The Power node is used to compute the element-wise power of two tensors.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Power node is defined as:
/// f(a, b) = a^b
/// where a is the base tensor and b is the exponent tensor.
/// The Power node is typically used in neural networks for operations such as exponentiation and activation functions.
pub const Power = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    a: Node,
    b: Node,

    pub fn init(allocator: std.mem.Allocator, a: Node, b: Node) !*Power {
        const ptr = try allocator.create(Power);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.a = a;
        ptr.b = b;

        return ptr;
    }

    /// Evaluate the power function.
    /// The power function is defined as:
    /// f(a, b) = a^b
    /// where a is the base tensor and b is the exponent tensor.
    /// The power function is often used in neural networks for operations such as exponentiation and activation functions.
    pub fn eval(self: *Power) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const a = self.a.eval();
        const b = self.b.eval();

        self.value = Tensor.init(self.allocator, a.shape) catch null;

        for (self.value.?.data, a.data, b.data) |*v, av, bv| {
            v.* = math.pow(av, bv);
        }

        std.debug.print("Power-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the power function.
    /// The gradient of the power function is defined as:
    /// ∂f / ∂a = (∂f / ∂x) * (∂x / ∂a)
    /// ∂f / ∂b = (∂f / ∂x) * (∂x / ∂b)
    /// where x is the input tensor.
    /// The gradient of the power function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Power, dval: *Tensor) void {
        const a = self.a.eval();
        const b = self.b.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, a.data, b.data, self.value.?.data, dval.data) |*v, av, bv, vv, dv| {
            v.* = (dv * bv * vv) / av;
        }
        self.a.diff(grad);

        for (grad.data, a.data, self.value.?.data, dval.data) |*v, av, vv, dv| {
            v.* = dv * vv * math.log(math.e, av);
        }
        self.b.diff(grad);

        std.debug.print("Power-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Power) Node {
        return Node.init(self);
    }
};
