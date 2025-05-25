const std = @import("std");

pub const Tensor = struct {
    allocator: std.mem.Allocator,
    shape: []const usize,
    data: []f64,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !*Tensor {
        const size = shapeProduct(shape);

        const ptr = try allocator.create(Tensor);
        ptr.allocator = allocator;
        ptr.shape = shape;
        ptr.data = try allocator.alloc(f64, size);

        return ptr;
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
    }

    pub fn zero(self: *Tensor) void {
        for (self.data) |*d| d.* = 0.0;
    }

    pub fn format(self: Tensor, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Tensor(shape={any}, data={d})", .{ self.shape, self.data });
    }
};

fn shapeProduct(shape: []const usize) usize {
    var total: usize = 1;
    for (shape) |dim| {
        if (dim == 0) @panic("Invalid shape: zero dimension");
        total *= dim;
    }

    return total;
}
