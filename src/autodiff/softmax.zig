const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Softmax function node.
/// The Softmax function is commonly used in neural networks, especially in the output layer for multi-class classification tasks.
/// It converts a vector of real numbers into a probability distribution, where the sum of the probabilities is 1.
/// The Softmax function is often used in conjunction with the cross-entropy loss function for training neural networks.
/// The Softmax function is differentiable, making it suitable for backpropagation in neural networks.
/// It is defined as:
/// f(x_i) = exp(x_i) / sum(exp(x_j)) for j in [1, ..., n]
/// where x_i is the i-th element of the input tensor and n is the number of elements in the tensor.
pub const Softmax = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Softmax {
        const ptr = try allocator.create(Softmax);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Softmax) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        var maxVal: f64 = x.data[0];
        for (x.data) |v| {
            if (v > maxVal) maxVal = v;
        }

        var sumExp: f64 = 0;
        for (x.data) |xv| {
            sumExp += math.exp(xv - maxVal); // for numerical stability
        }

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.exp(xv - maxVal) / sumExp; // Softmax formula
        }

        std.debug.print("Softmax-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Softmax, dval: *Tensor) void {
        // Formula: ∂Si / ∂Xj =
        //  Si  * (1 - Si),   if i = j
        //  -Si * Sj,         if i ≠ j
        //
        // Where:
        // - S is the softmax output.
        // - X is the input to the softmax function.
        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data, self.value.?.data, dval.data, 0..) |*v, iv, dv, i| {
            v.* = 0; // Initialize gradient to zero

            for (self.value.?.data, 0..) |jv, j| {
                if (i == j) {
                    v.* += dv * iv * (1 - iv); // Diagonal element
                } else {
                    v.* -= dv * iv * jv; // Off-diagonal element
                }
            }
        }

        self.x.diff(grad);

        std.debug.print("Softmax-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Softmax) Node {
        return Node.init(self);
    }
};
