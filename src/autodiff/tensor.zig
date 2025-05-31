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
    strides: []usize,
    size: usize,
    data: []f64,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !*Tensor {
        const ptr = try allocator.create(Tensor);
        ptr.allocator = allocator;
        ptr.shape = shape;
        ptr.strides = try computeStrides(allocator, shape);
        ptr.size = numel(shape);
        ptr.data = try allocator.alloc(f64, ptr.size);

        return ptr;
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
    }

    pub fn zero(self: *Tensor) void {
        for (self.data) |*d| d.* = 0.0;
    }

    pub fn format(self: Tensor, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Tensor(shape={any}, data={d}, strides={d}, size={d})", .{ self.shape, self.data, self.strides, self.size });
    }

    /// Compute strides for a given shape (row-major layout).
    fn computeStrides(allocator: std.mem.Allocator, shape: []const usize) ![]usize {
        const rank = shape.len;
        const strides = try allocator.alloc(usize, rank);

        var acc: usize = 1;
        for (0..rank) |i| {
            acc *= shape[i];
            strides[i] = acc;
        }

        return strides;
    }

    /// Get flat index offset of a slice at `axis` with slice index `i`.
    pub fn sliceOffset(self: *Tensor, axis: usize, slice_index: usize) usize {
        return slice_index * self.strides[axis];
    }

    /// Get the stride to jump between elements along `axis`.
    pub fn axisStride(self: *Tensor, axis: usize) usize {
        return self.strides[axis];
    }

    /// Reshapes the tensor to a new shape. Returns a new Tensor struct with shared data.
    pub fn reshape(self: *Tensor, new_shape: []const usize) !*Tensor {
        const old_numel = Tensor.numel(self.shape);
        const new_numel = Tensor.numel(new_shape);

        if (old_numel != new_numel) {
            return error.ShapeMismatch;
        }

        const new_shape_copy = try self.allocator.dupe(usize, new_shape);

        const reshaped = try self.allocator.create(Tensor);
        reshaped.* = Tensor{
            .allocator = self.allocator,
            .shape = new_shape_copy,
            .data = self.data,
        };

        return reshaped;
    }

    pub fn stride(shape: []const usize) usize {
        var s: usize = 1;
        for (shape) |dim| {
            s *= dim;
        }

        return s;
    }

    pub fn numel(shape: []const usize) usize {
        var total: usize = 1;
        for (shape) |dim| {
            total *= dim;
        }

        return total;
    }
};
