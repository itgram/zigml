const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// SELU function node.
/// The Scaled Exponential Linear Unit (SELU) activation function.
/// It is defined as:
/// f(x) = λ * x if x > 0 else λ * α * (exp(x) - 1)
/// - For positive inputs: f(x) = λ * x
/// - For negative inputs: f(x) = λ * α * (exp(x) - 1)
/// where λ is a scaling factor (default 1.0507009873554804934193349852946)
/// and α is a small positive constant (default 1.6732632423543772848170429916717).
/// The SELU function is designed to self-normalize, meaning it helps maintain a mean of 0 and variance of 1 across layers.
pub const SELU = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    alpha: f64 = 1.6732632423543772848170429916717, // small slope for negative inputs
    lambda: f64 = 1.0507009873554804934193349852946, // scaling factor for positive inputs

    pub fn init(allocator: std.mem.Allocator, x: Node, alpha: f64, lambda: f64) !*SELU {
        const ptr = try allocator.create(SELU);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.alpha = alpha;
        ptr.lambda = lambda;

        return ptr;
    }

    /// Evaluate the SELU function.
    /// The SELU function is defined as:
    /// f(x) = λ * x if x > 0 else λ * α * (exp(x) - 1)
    /// where λ is a scaling factor (default 1.0507009873554804934193349852946)
    /// and α is a small positive constant (default 1.6732632423543772848170429916717).
    /// The SELU function is designed to self-normalize, meaning it helps maintain a mean of 0 and variance of 1 across layers.
    pub fn eval(self: *SELU) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv > 0) self.lambda * xv else self.lambda * self.alpha * (@exp(xv) - 1);
        }

        std.debug.print("SELU-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the SELU function.
    /// The gradient of the SELU function is defined as:
    /// ∂f / ∂x = λ if x > 0 else λ * α * exp(x)
    /// where λ is a scaling factor (default 1.0507009873554804934193349852946)
    /// and α is a small positive constant (default 1.6732632423543772848170429916717).
    /// The gradient of the SELU function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *SELU, dval: *Tensor) void {
        const x = self.x.eval();

        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, x.data, self.value.?.data, dval.data) |*v, xv, vv, dv| {
            v.* = if (xv > 0) dv * self.lambda else dv * (vv + self.lambda * self.alpha);
        }

        self.x.diff(grad);

        std.debug.print("SELU-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *SELU) Node {
        return Node.init(self);
    }
};
