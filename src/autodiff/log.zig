const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Log function node.
/// The Log node represents the logarithm function applied to a tensor.
/// It computes the logarithm of each element in the input tensor.
/// The Log node is used in neural networks and mathematical computations where the logarithm function is required.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// It is defined as:
/// f(x) = log10(x)
/// where x is the input tensor.
/// The logarithm function is often used in optimization problems and loss functions.
/// The Log node is typically used in conjunction with other nodes to build complex computation graphs.
pub const Log = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    /// Creates a new logarithm node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node) !*Log {
        const ptr = try allocator.create(Log);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    /// Deinitializes the node and frees all allocated resources.
    /// This should be called when the node is no longer needed.
    pub fn deinit(self: *Log) void {
        if (self.value) |v| {
            v.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Evaluate the logarithm function.
    /// The logarithm function is defined as:
    /// f(x) = log10(x)
    /// where x is the input tensor.
    /// The logarithm function is often used in optimization problems and loss functions.
    pub fn eval(self: *Log) !*Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = try self.x.eval();

        self.value = try Tensor.init(self.allocator, x.shape);

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.log(10, xv);
        }

        std.debug.print("Log-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the logarithm function.
    /// The gradient of the logarithm function is defined as:
    /// ∂f/∂x = 1 / (x * ln(10))
    /// where x is the input tensor.
    /// The gradient of the logarithm function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Log, dval: *Tensor) !void {
        const x = try self.x.eval();

        const grad = try Tensor.init(self.allocator, dval.shape);
        defer grad.deinit();

        for (grad.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv / (xv * math.ln10);
        }

        try self.x.diff(grad);

        std.debug.print("Log-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Log) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this logarithm node as a generic Node interface.
    pub fn node(self: *Log) Node {
        return Node.init(self);
    }
};
