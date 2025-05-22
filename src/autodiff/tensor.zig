const std = @import("std");

const TensorError = error{
    ShapeMismatch,
};

pub const Tensor = struct {
    allocator: std.mem.Allocator,
    shape: []const usize,
    data: []f64,
    grad: []f64,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        const size = shapeProduct(shape);

        return Tensor{
            .allocator = allocator,
            .shape = shape,
            .data = try allocator.alloc(f64, size),
            .grad = try allocator.alloc(f64, size),
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.grad);
    }

    pub fn zeroGrad(self: *Tensor) void {
        for (self.grad) |*g| g.* = 0.0;
    }

    pub fn print(self: *Tensor) void {
        std.debug.print("Tensor(shape={any}, data={any}, grad={any})\n", .{ self.shape, self.data, self.grad });
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

pub fn add(a: *Tensor, b: *Tensor) !Tensor {
    if (a.data.len != b.data.len) return error.ShapeMismatch;

    const out = try Tensor.init(a.allocator, a.shape);
    for (a.data, b.data, out.data) |va, vb, *vo| vo.* = va + vb;
    return out;
}

pub fn mul(a: *Tensor, b: *Tensor) !Tensor {
    if (a.data.len != b.data.len) return error.ShapeMismatch;

    const out = try Tensor.init(a.allocator, a.shape);
    for (a.data, b.data, out.data) |va, vb, *vo| vo.* = va * vb;
    return out;
}

// pub fn relu(x: *Tensor) !Tensor {
//     const out = try Tensor.init(x.allocator, x.shape);
//     for (x.data, out.data) |vx, *vo| vo.* = if (vx > 0.0) vx else 0.0;

//     return out;
// }

// pub fn scalar(allocator: std.mem.Allocator, x: f64) !Tensor {
//     var t = try Tensor.init(allocator, &[_]usize{1});
//     t.data[0] = x;

//     return t;
// }
