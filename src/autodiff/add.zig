const std = @import("std");
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Add two nodes
/// where x and y are nodes that evaluate to tensors.
/// The Add node computes the element-wise sum of the tensors produced by its two input nodes.
/// It is used to represent addition operations in the computation graph.
/// The Add node is a fundamental operation in many neural networks and mathematical computations.
/// It supports automatic differentiation, allowing gradients to be computed for backpropagation.
/// The Add node is defined as:
/// f(x, y) = x + y
/// where x and y are the input tensors.
/// The Add node is typically used in conjunction with other nodes to build complex computation graphs.
pub const Add = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    y: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node, y: Node) !*Add {
        const ptr = try allocator.create(Add);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.y = y;

        return ptr;
    }

    /// Evaluate the add function.
    /// The add function is defined as:
    /// f(x, y) = x + y
    /// where x and y are the input tensors.
    /// The add function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn eval(self: *Add) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();
        const y = self.y.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data, y.data) |*v, xv, yv| {
            v.* = xv + yv;
        }

        std.debug.print("Add-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    /// Compute the gradient of the add function.
    /// The gradient of the add function is defined as:
    /// ∂f / ∂x = 1
    /// ∂f / ∂y = 1
    /// where x and y are the input tensors.
    /// The gradient of the add function is typically used in conjunction with other nodes to build complex computation graphs.
    pub fn diff(self: *Add, dval: *Tensor) void {
        self.x.diff(dval);
        self.y.diff(dval);

        std.debug.print("Add-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Add) Node {
        return Node.init(self);
    }
};
