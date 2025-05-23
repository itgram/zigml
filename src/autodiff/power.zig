const std = @import("std");
const math = @import("std").math;
const Node = @import("node.zig").Node;
const Tensor = @import("tensor.zig").Tensor;

/// Power node
/// f = a ^ b
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

    pub fn diff(self: *Power, dval: *Tensor) void {
        const a = self.a.eval();
        const b = self.b.eval();

        const c: f64 = if (a == math.e) 1 else math.log(math.e, a);

        const ndval = Tensor.init(self.allocator, dval.shape) catch unreachable;

        for (ndval.data, a.data, b.data, self.value.?.data, dval.data) |*v, av, bv, vv, dv| {
            v.* = (dv * bv * vv) / av;
        }
        self.a.diff(ndval);

        for (ndval.data, self.value.?.data, dval.data) |*v, vv, dv| {
            v.* = dv * vv * c;
        }
        self.b.diff(ndval);

        std.debug.print("Power-diff: value: {?}, dval: {}\n", .{ self.value, dval });
    }

    pub fn node(self: *Power) Node {
        return Node.init(self);
    }
};
