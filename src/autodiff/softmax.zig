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
    axis: usize, // Axis along which to compute the softmax. Default is 0.

    /// Creates a new softmax node with the given input node.
    pub fn init(allocator: std.mem.Allocator, x: Node, axis: usize) !*Softmax {
        const ptr = try allocator.create(Softmax);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.axis = axis;

        return ptr;
    }

    /// Evaluate the softmax function.
    /// The softmax function is defined as:
    /// f(x_i) = exp(x_i) / sum(exp(x_j)) for j in [1, ..., n]
    /// where x_i is the i-th element of the input tensor and n is the number of elements in the tensor.
    pub fn eval(self: *Softmax) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;
        const shape = x.shape;
        const axis = self.axis;
        const axis_dim = shape[axis];
        const outer = Tensor.stride(shape[0..axis]);
        const inner = Tensor.stride(shape[axis + 1 ..]);

        for (0..outer) |outer_idx| {
            for (0..inner) |inner_idx| {
                // Compute the base offset for this slice
                const base = outer_idx * axis_dim * inner + inner_idx;

                // 1. Find max for numerical stability
                var maxVal: f64 = x.data[base];
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    if (x.data[idx] > maxVal) maxVal = x.data[idx];
                }

                // 2. Compute sum of exp
                var sumExp: f64 = 0;
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    sumExp += math.exp(x.data[idx] - maxVal);
                }

                // 3. Write softmax values
                for (0..axis_dim) |i| {
                    const idx = base + i * inner;
                    self.value.?.data[idx] = math.exp(x.data[idx] - maxVal) / sumExp;
                }
            }
        }

        std.debug.print("Softmax-eval: value: {?}\n", .{self.value});
        return self.value.?;
    }

    /// Compute the gradient of the softmax function.
    /// The gradient of the softmax function is defined as:
    /// ∂Si / ∂Xj =
    ///  Si  * (1 - Si),   if i = j
    ///  -Si * Sj,         if i ≠ j
    /// Where:
    /// - S is the softmax output.
    pub fn diff(self: *Softmax, dval: *Tensor) void {
        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;
        const shape = dval.shape;
        const axis = self.axis;
        const axis_dim = shape[axis];
        const outer = Tensor.stride(shape[0..axis]);
        const inner = Tensor.stride(shape[axis + 1 ..]);

        for (0..outer) |outer_idx| {
            for (0..inner) |inner_idx| {
                const base = outer_idx * axis_dim * inner + inner_idx;

                // Compute gradient for this slice
                for (0..axis_dim) |i| {
                    const idx_i = base + i * inner;
                    grad.data[idx_i] = 0;
                    const Si = self.value.?.data[idx_i];
                    const dvi = dval.data[idx_i];

                    for (0..axis_dim) |j| {
                        const idx_j = base + j * inner;
                        const Sj = self.value.?.data[idx_j];
                        if (i == j) {
                            grad.data[idx_i] += dvi * Si * (1 - Si);
                        } else {
                            grad.data[idx_i] -= dvi * Si * Sj;
                        }
                    }
                }
            }
        }
        self.x.diff(grad);
        std.debug.print("Softmax-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    /// Returns this softmax node as a generic Node interface.
    pub fn node(self: *Softmax) Node {
        return Node.init(self);
    }
};
