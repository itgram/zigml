const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Power function node.
/// where x and y are nodes representing tensors.
/// The Power node is used to compute the element-wise power of two tensors.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Power node is defined as:
/// f(x, y) = x^y
/// where x is the base tensor and y is the exponent tensor.
/// The Power node is typically used in neural networks for operations such as exponentiation and activation functions.
pub const Power = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    /// Creates a new power node with the given input nodes.
    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Power {
        const ptr = try allocator.create(Power);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.y = y;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Power) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the power function.
    /// The power function is defined as:
    /// f(x, y) = x^y
    /// where x and y are the input tensors.
    pub fn eval(self: *Power) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();
        const y = self.y.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = math.pow(xv, yv);
        }

        std.debug.print("Power-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the power function.
    /// The gradient of the power function is defined as:
    /// ∂f/∂x = y * x^(y-1)
    /// ∂f/∂y = x^y * ln(x)
    /// where x and y are the input tensors.
    /// The gradient of the power function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Power, dval: *Tensor) void {
        const x = self.x.eval();
        const y = self.y.eval();

        const grad_x = Tensor.init(self.allocator, dval.shape) catch unreachable;
        defer grad_x.deinit();
        const grad_y = Tensor.init(self.allocator, dval.shape) catch unreachable;
        defer grad_y.deinit();

        for (grad_x.data, grad_y.data, x.data, y.data, dval.data) |*gx, *gy, xv, yv, dv| {
            gx.* = dv * yv * math.pow(xv, yv - 1);
            gy.* = dv * math.pow(xv, yv) * math.ln(xv);
        }

        self.x.diff(grad_x);
        self.y.diff(grad_y);

        std.debug.print("Power-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Power) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
        self.y.reset();
    }

    /// Returns this power node as a generic Node interface.
    pub fn node(self: *Power) Node {
        return Node.init(self);
    }
};
