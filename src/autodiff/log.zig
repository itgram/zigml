const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Logarithm node
/// f = log(x)
pub const Log = struct {
    allocator: std.mem.Allocator,
    value: ?*Tensor,
    x: Node,

    pub fn init(allocator: std.mem.Allocator, x: Node) !*Log {
        const ptr = try allocator.create(Log);
        ptr.allocator = allocator;
        ptr.value = null;
        ptr.x = x;

        return ptr;
    }

    pub fn eval(self: *Log) *Tensor {
        if (self.value) |v| {
            return v;
        }

        const x = self.x.eval();

        self.value = Tensor.init(self.allocator, x.shape) catch null;

        for (self.value.?.data, x.data) |*v, xv| {
            v.* = math.log(10, xv);
        }

        std.debug.print("Log-eval: value: {?}\n", .{self.value});

        return self.value.?;
    }

    pub fn diff(self: *Log, dval: *Tensor) void {
        const x = self.x.eval();

        const ndval = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (ndval.data, x.data, dval.data) |*v, xv, dv| {
            v.* = dv / (xv * math.ln10);
        }

        self.x.diff(ndval);

        std.debug.print("Log-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Log) Node {
        return Node.init(self);
    }
};
