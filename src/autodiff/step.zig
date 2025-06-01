const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Step function node.
/// where threshold is a configurable value (default is 0.0).
/// The Step function is often used in binary classification tasks and as an activation function in neural networks.
/// It is not differentiable at the threshold, but it can be used in contexts where a hard thresholding is required.
/// The Step function is useful for creating binary outputs from continuous inputs.
/// It is defined as:
/// f(x) = 1 if x >= threshold, else 0
/// where x is the input tensor and threshold is a configurable value (default is 0.0).
pub const Step = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    threshold: f64 = 0.0, // Default threshold value

    /// Creates a new step node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node, threshold: f64) !*Step {
        const ptr = try allocator.create(Step);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.threshold = threshold;

        return ptr;
    }

    /// Evaluate the step function.
    /// The step function is defined as:
    /// f(x) = 1 if x >= threshold, else 0
    /// where x is the input tensor and threshold is a configurable value (default is 0.0).
    /// The step function is often used in binary classification tasks and as an activation function in neural networks.
    /// It is not differentiable at the threshold, but it can be used in contexts where a hard thresholding is required.
    /// The step function is useful for creating binary outputs from continuous inputs.
    pub fn eval(self: *Step) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = if (xv >= self.threshold) 1 else 0;
        }

        std.debug.print("Step-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the step function.
    /// The gradient of the step function is defined as:
    /// ∂f/∂x = 0
    /// where x is the input tensor.
    /// The gradient of the step function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Step, dval: *Tensor) void {
        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data) |*v| {
            v.* = 0;
        }

        self.x.diff(grad);

        std.debug.print("Step-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: *Step) void {
        if (self.value) |v| {
            v.deinit();
            self.value = null;
        }
        self.x.reset();
    }

    /// Returns this step node as a generic Node interface.
    pub fn node(self: *Step) Node {
        return Node.init(self);
    }
};
