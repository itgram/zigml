const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Step function node
/// f(x) = 1 if x >= threshold else 0, where threshold is a constant value (default 0)
pub const Step = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,
    threshold: f64 = 0.0, // Default threshold value

    pub fn init(allocator: std.mem.Allocator, x: Node, threshold: f64) !*Step {
        const ptr = try allocator.create(Step);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;
        ptr.threshold = threshold;

        return ptr;
    }

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

    pub fn diff(self: *Step, dval: *Tensor) void {
        const grad = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (grad.data) |*v| {
            v.* = 0;
        }

        self.x.diff(grad);

        std.debug.print("Step-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Step) Node {
        return Node.init(self);
    }
};
