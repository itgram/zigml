const std = @import("std");

/// Tensor structure for representing multi-dimensional arrays.
/// The Tensor structure is often used in machine learning and scientific computing.
/// It is defined as:
/// Tensor(shape=[dim1, dim2, ...], data=[value1, value2, ...])
/// where shape is an array of dimensions and data is an array of values.
/// The Tensor structure is a fundamental building block for many machine learning frameworks.
/// It is used to represent inputs, outputs, and intermediate values in neural networks.
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
