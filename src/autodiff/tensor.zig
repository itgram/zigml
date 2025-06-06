const std = @import("std");

/// Error types for the Tensor structure.
pub const TensorError = error{
    /// Error when the shape of the tensor does not match the expected shape.
    ShapeMismatch,

    /// Error when the indices are out of bounds.
    IndexOutOfBounds,

    /// Error when the shape of the tensor does not match the expected shape.
    InvalidShape,
};

/// Tensor structure for representing multi-dimensional arrays.
/// The Tensor structure is often used in machine learning and scientific computing.
/// It is defined as:
/// Tensor(shape=[dim1, dim2, ...], data=[value1, value2, ...])
/// where shape is an array of dimensions and data is an array of values.
/// The Tensor structure is a fundamental building block for many machine learning frameworks.
/// It is used to represent inputs, outputs, and intermediate values in neural networks.
pub const Tensor = struct {
    allocator: std.mem.Allocator,
    size: usize,
    shape: []const usize,
    strides: []usize,
    data: []f64,

    /// Initialize a new tensor.
    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !*Tensor {
        // Compute the number of elements in the tensor.
        const size = sizeOf(shape);

        // Copy the shape to avoid modifying the original shape.
        const rank = shape.len;
        const shapeCopy = try allocator.alloc(usize, rank);
        std.mem.copyForwards(usize, shapeCopy, shape);

        // Compute the strides for the tensor.
        const strides = try allocator.alloc(usize, rank);

        var stride: usize = 1;
        var i: usize = rank;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= shape[i];
        }

        const self = try allocator.create(Tensor);
        self.* = .{
            .allocator = allocator,
            .size = size,
            .shape = shapeCopy,
            .strides = strides,
            .data = try allocator.alloc(f64, size),
        };

        return self;
    }

    /// Deinitialize a tensor.
    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
        self.allocator.free(self.data);
        self.allocator.destroy(self);
    }

    /// Set all elements of the tensor to 0.
    pub fn zero(self: *Tensor) void {
        for (self.data) |*d| d.* = 0.0;
    }

    /// Get the value of the tensor at the given indices.
    pub fn get(self: *Tensor, indices: []const usize) !f64 {
        const idx = try self.indexOf(indices);
        return self.data[idx];
    }

    /// Set the value of the tensor at the given indices.
    pub fn set(self: *Tensor, indices: []const usize, value: f64) !void {
        const idx = try self.indexOf(indices);
        self.data[idx] = value;
    }

    /// Get the index of the tensor at the given indices.
    pub fn indexOf(self: *Tensor, indices: []const usize) !usize {
        if (indices.len != self.shape.len) return error.InvalidShape;

        // Check bounds for each dimension
        for (indices, self.shape) |idx, dim| {
            if (idx >= dim) return error.IndexOutOfBounds;
        }

        var idx: usize = 0;
        for (indices, self.strides) |i, stride| {
            idx += i * stride;
        }

        return idx;
    }

    /// Format the tensor as a string.
    pub fn format(self: Tensor, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Tensor(size={d}, shape={any}, strides={any}, data={any})", .{ self.size, self.shape, self.strides, self.data });
    }

    /// Compute the number of elements in a tensor.
    pub fn sizeOf(shape: []const usize) usize {
        var total: usize = 1;
        for (shape) |dim| {
            total *= dim;
        }

        return total;
    }
};

test "tensor initialization" {
    const allocator = std.testing.allocator;

    // Test 1D tensor
    {
        const shape = &[_]usize{3};
        const tensor = try Tensor.init(allocator, shape);
        defer tensor.deinit();

        try std.testing.expectEqual(@as(usize, 3), tensor.size);
        try std.testing.expectEqualSlices(usize, shape, tensor.shape);
        try std.testing.expectEqual(@as(usize, 1), tensor.strides[0]);
        try std.testing.expectEqual(@as(usize, 3), tensor.data.len);
    }

    // Test 2D tensor
    {
        const shape = &[_]usize{ 2, 3 };
        const tensor = try Tensor.init(allocator, shape);
        defer tensor.deinit();

        try std.testing.expectEqual(@as(usize, 6), tensor.size);
        try std.testing.expectEqualSlices(usize, shape, tensor.shape);
        try std.testing.expectEqual(@as(usize, 3), tensor.strides[0]);
        try std.testing.expectEqual(@as(usize, 1), tensor.strides[1]);
        try std.testing.expectEqual(@as(usize, 6), tensor.data.len);
    }

    // Test 3D tensor
    {
        const shape = &[_]usize{ 2, 3, 4 };
        const tensor = try Tensor.init(allocator, shape);
        defer tensor.deinit();

        try std.testing.expectEqual(@as(usize, 24), tensor.size);
        try std.testing.expectEqualSlices(usize, shape, tensor.shape);
        try std.testing.expectEqual(@as(usize, 12), tensor.strides[0]);
        try std.testing.expectEqual(@as(usize, 4), tensor.strides[1]);
        try std.testing.expectEqual(@as(usize, 1), tensor.strides[2]);
        try std.testing.expectEqual(@as(usize, 24), tensor.data.len);
    }

    // Test empty tensor
    {
        const shape = &[_]usize{};
        const tensor = try Tensor.init(allocator, shape);
        defer tensor.deinit();

        try std.testing.expectEqual(@as(usize, 1), tensor.size);
        try std.testing.expectEqualSlices(usize, shape, tensor.shape);
        try std.testing.expectEqual(@as(usize, 1), tensor.data.len);
    }
}

test "tensor indexing" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 2, 3 };
    const tensor = try Tensor.init(allocator, shape);
    defer tensor.deinit();

    // Test setting and getting values
    {
        // Set values in row-major order
        try tensor.set(&[_]usize{ 0, 0 }, 1.0);
        try tensor.set(&[_]usize{ 0, 1 }, 2.0);
        try tensor.set(&[_]usize{ 0, 2 }, 3.0);
        try tensor.set(&[_]usize{ 1, 0 }, 4.0);
        try tensor.set(&[_]usize{ 1, 1 }, 5.0);
        try tensor.set(&[_]usize{ 1, 2 }, 6.0);

        // Verify values
        try std.testing.expectEqual(@as(f64, 1.0), try tensor.get(&[_]usize{ 0, 0 }));
        try std.testing.expectEqual(@as(f64, 2.0), try tensor.get(&[_]usize{ 0, 1 }));
        try std.testing.expectEqual(@as(f64, 3.0), try tensor.get(&[_]usize{ 0, 2 }));
        try std.testing.expectEqual(@as(f64, 4.0), try tensor.get(&[_]usize{ 1, 0 }));
        try std.testing.expectEqual(@as(f64, 5.0), try tensor.get(&[_]usize{ 1, 1 }));
        try std.testing.expectEqual(@as(f64, 6.0), try tensor.get(&[_]usize{ 1, 2 }));

        // Verify data array layout
        try std.testing.expectEqualSlices(f64, &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, tensor.data);
    }

    // Test error cases
    {
        // Test invalid shape
        try std.testing.expectError(error.InvalidShape, tensor.get(&[_]usize{0}));
        try std.testing.expectError(error.InvalidShape, tensor.get(&[_]usize{ 0, 0, 0 }));

        // Test out of bounds
        try std.testing.expectError(error.IndexOutOfBounds, tensor.get(&[_]usize{ 2, 0 }));
        try std.testing.expectError(error.IndexOutOfBounds, tensor.get(&[_]usize{ 0, 3 }));
        try std.testing.expectError(error.IndexOutOfBounds, tensor.get(&[_]usize{ 1, 3 }));
        try std.testing.expectError(error.IndexOutOfBounds, tensor.get(&[_]usize{ 2, 2 }));
    }
}

test "tensor zero" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 2, 3 };
    const tensor = try Tensor.init(allocator, shape);
    defer tensor.deinit();

    // Set some non-zero values
    try tensor.set(&[_]usize{ 0, 0 }, 1.0);
    try tensor.set(&[_]usize{ 0, 1 }, 2.0);
    try tensor.set(&[_]usize{ 0, 2 }, 3.0);
    try tensor.set(&[_]usize{ 1, 0 }, 4.0);
    try tensor.set(&[_]usize{ 1, 1 }, 5.0);
    try tensor.set(&[_]usize{ 1, 2 }, 6.0);

    // Verify initial values
    try std.testing.expectEqualSlices(f64, &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, tensor.data);

    // Zero the tensor
    tensor.zero();

    // Check all values are zero
    try std.testing.expectEqualSlices(f64, &[_]f64{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, tensor.data);
}

test "tensor format" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 2, 2 };
    const tensor = try Tensor.init(allocator, shape);
    defer tensor.deinit();

    // Set some values
    try tensor.set(&[_]usize{ 0, 0 }, 1.0);
    try tensor.set(&[_]usize{ 0, 1 }, 2.0);
    try tensor.set(&[_]usize{ 1, 0 }, 3.0);
    try tensor.set(&[_]usize{ 1, 1 }, 4.0);

    // Test string representation
    var buffer: [256]u8 = undefined;
    const str = try std.fmt.bufPrint(&buffer, "{any}", .{tensor});
    try std.testing.expectEqualStrings("Tensor(size=4, shape={ 2, 2 }, strides={ 2, 1 }, data={ 1e0, 2e0, 3e0, 4e0 })", str);
}

test "tensor sizeOf" {
    // Test 1D tensor
    try std.testing.expectEqual(@as(usize, 5), Tensor.sizeOf(&[_]usize{5}));

    // Test 2D tensor
    try std.testing.expectEqual(@as(usize, 6), Tensor.sizeOf(&[_]usize{ 2, 3 }));

    // Test 3D tensor
    try std.testing.expectEqual(@as(usize, 24), Tensor.sizeOf(&[_]usize{ 2, 3, 4 }));

    // Test empty tensor
    try std.testing.expectEqual(@as(usize, 1), Tensor.sizeOf(&[_]usize{}));

    // Test large tensor
    try std.testing.expectEqual(@as(usize, 1000), Tensor.sizeOf(&[_]usize{ 10, 10, 10 }));
}

test "tensor memory management" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 2, 3 };
    var tensor = try Tensor.init(allocator, shape);

    // Set some values
    try tensor.set(&[_]usize{ 0, 0 }, 1.0);
    try tensor.set(&[_]usize{ 0, 1 }, 2.0);
    try tensor.set(&[_]usize{ 0, 2 }, 3.0);

    // Deinitialize and verify no memory leaks
    tensor.deinit();

    // Create a new tensor to verify allocator is still working
    const new_tensor = try Tensor.init(allocator, shape);
    defer new_tensor.deinit();
    try std.testing.expectEqual(@as(usize, 6), new_tensor.size);
}
