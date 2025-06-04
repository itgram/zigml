const std = @import("std");
const autodiff = @import("autodiff");
const Tensor = autodiff.Tensor;
const math = std.math;

/// Error types for optimizers
const OptimizerError = error{
    /// Error when the shape of the tensor is invalid
    InvalidShape,
    /// Error when the shapes of tensors don't match
    ShapeMismatch,
};

/// Generic optimizer interface
pub fn GenericOptimizer(
    comptime Context: type,
    comptime stepFn: fn (context: Context, params: *Tensor, grads: *Tensor) OptimizerError!void,
) type {
    return struct {
        context: Context,

        const Self = @This();

        pub inline fn step(self: Self, params: *Tensor, grads: *Tensor) OptimizerError!void {
            return stepFn(self.context, params, grads);
        }
    };
}

/// Stochastic Gradient Descent optimizer
pub const SGD = struct {
    learning_rate: f64,

    pub fn init(allocator: std.mem.Allocator, learning_rate: f64) !*SGD {
        const self = try allocator.create(SGD);
        self.* = .{
            .learning_rate = learning_rate,
        };
        return self;
    }

    pub fn deinit(self: *SGD, allocator: std.mem.Allocator) void {
        allocator.destroy(self);
    }

    pub fn step(self: *SGD, params: *Tensor, grads: *Tensor) OptimizerError!void {
        if (params.shape.len != grads.shape.len) {
            return error.ShapeMismatch;
        }
        for (params.shape, 0..) |dim, i| {
            if (dim != grads.shape[i]) {
                return error.ShapeMismatch;
            }
        }

        // Update parameters: param = param - lr * grad
        for (params.data, grads.data) |*param, grad| {
            param.* -= self.learning_rate * grad;
        }
    }

    pub fn optimizer(self: *SGD) GenericOptimizer(*SGD, step) {
        return .{ .context = self };
    }
};

/// AdaGrad optimizer
pub const AdaGrad = struct {
    learning_rate: f64,
    epsilon: f64,
    squared_grads: *Tensor,

    pub fn init(allocator: std.mem.Allocator, learning_rate: f64, shape: []const usize) !*AdaGrad {
        const self = try allocator.create(AdaGrad);
        self.* = .{
            .learning_rate = learning_rate,
            .epsilon = 1e-8,
            .squared_grads = try Tensor.init(allocator, shape),
        };
        // Initialize squared_grads with zeros
        for (self.squared_grads.data) |*val| {
            val.* = 0;
        }
        return self;
    }

    pub fn deinit(self: *AdaGrad, allocator: std.mem.Allocator) void {
        self.squared_grads.deinit();
        allocator.destroy(self);
    }

    pub fn step(self: *AdaGrad, params: *Tensor, grads: *Tensor) OptimizerError!void {
        if (params.shape.len != grads.shape.len or params.shape.len != self.squared_grads.shape.len) {
            return error.ShapeMismatch;
        }
        for (params.shape, 0..) |dim, i| {
            if (dim != grads.shape[i] or dim != self.squared_grads.shape[i]) {
                return error.ShapeMismatch;
            }
        }

        // Update squared gradients: s = s + grad^2
        for (self.squared_grads.data, grads.data) |*s, grad| {
            s.* += grad * grad;
        }

        // Update parameters: param = param - lr * grad / sqrt(s + epsilon)
        for (params.data, grads.data, self.squared_grads.data) |*param, grad, s| {
            param.* -= self.learning_rate * grad / @sqrt(s + self.epsilon);
        }
    }

    pub fn optimizer(self: *AdaGrad) GenericOptimizer(*AdaGrad, step) {
        return .{ .context = self };
    }
};

/// RMSprop optimizer
pub const RMSprop = struct {
    learning_rate: f64,
    beta: f64,
    epsilon: f64,
    squared_grads: *Tensor,

    pub fn init(allocator: std.mem.Allocator, learning_rate: f64, shape: []const usize) !*RMSprop {
        const self = try allocator.create(RMSprop);
        self.* = .{
            .learning_rate = learning_rate,
            .beta = 0.999,
            .epsilon = 1e-8,
            .squared_grads = try Tensor.init(allocator, shape),
        };
        // Initialize squared_grads with zeros
        for (self.squared_grads.data) |*val| {
            val.* = 0;
        }
        return self;
    }

    pub fn deinit(self: *RMSprop, allocator: std.mem.Allocator) void {
        self.squared_grads.deinit();
        allocator.destroy(self);
    }

    pub fn step(self: *RMSprop, params: *Tensor, grads: *Tensor) OptimizerError!void {
        if (params.shape.len != grads.shape.len or params.shape.len != self.squared_grads.shape.len) {
            return error.ShapeMismatch;
        }
        for (params.shape, 0..) |dim, i| {
            if (dim != grads.shape[i] or dim != self.squared_grads.shape[i]) {
                return error.ShapeMismatch;
            }
        }

        // Update squared gradients: s = beta * s + (1 - beta) * grad^2
        for (self.squared_grads.data, grads.data) |*s, grad| {
            s.* = self.beta * s.* + (1 - self.beta) * grad * grad;
        }

        // Update parameters: param = param - lr * grad / sqrt(s + epsilon)
        for (params.data, grads.data, self.squared_grads.data) |*param, grad, s| {
            param.* -= self.learning_rate * grad / @sqrt(s + self.epsilon);
        }
    }

    pub fn optimizer(self: *RMSprop) GenericOptimizer(*RMSprop, step) {
        return .{ .context = self };
    }
};

/// Adam optimizer
pub const Adam = struct {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
    m: *Tensor,
    v: *Tensor,

    pub fn init(allocator: std.mem.Allocator, learning_rate: f64, shape: []const usize) !*Adam {
        const self = try allocator.create(Adam);
        self.* = .{
            .learning_rate = learning_rate,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .epsilon = 1e-8,
            .t = 0,
            .m = try Tensor.init(allocator, shape),
            .v = try Tensor.init(allocator, shape),
        };
        // Initialize m and v with zeros
        for (self.m.data) |*val| val.* = 0;
        for (self.v.data) |*val| val.* = 0;
        return self;
    }

    pub fn deinit(self: *Adam, allocator: std.mem.Allocator) void {
        self.m.deinit();
        self.v.deinit();
        allocator.destroy(self);
    }

    pub fn step(self: *Adam, params: *Tensor, grads: *Tensor) OptimizerError!void {
        if (params.shape.len != grads.shape.len or params.shape.len != self.m.shape.len or params.shape.len != self.v.shape.len) {
            return error.ShapeMismatch;
        }
        for (params.shape, 0..) |dim, i| {
            if (dim != grads.shape[i] or dim != self.m.shape[i] or dim != self.v.shape[i]) {
                return error.ShapeMismatch;
            }
        }

        self.t += 1;
        const t = @as(f64, @floatFromInt(self.t));

        // Update first moment: m = beta1 * m + (1 - beta1) * grad
        for (self.m.data, grads.data) |*m, grad| {
            m.* = self.beta1 * m.* + (1 - self.beta1) * grad;
        }

        // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
        for (self.v.data, grads.data) |*v, grad| {
            v.* = self.beta2 * v.* + (1 - self.beta2) * grad * grad;
        }

        // Compute bias-corrected moments
        const m_hat = 1.0 / (1.0 - std.math.pow(f64, self.beta1, t));
        const v_hat = 1.0 / (1.0 - std.math.pow(f64, self.beta2, t));

        // Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
        for (params.data, self.m.data, self.v.data) |*param, m, v| {
            param.* -= self.learning_rate * m_hat * m / (@sqrt(v_hat * v) + self.epsilon);
        }
    }

    pub fn optimizer(self: *Adam) GenericOptimizer(*Adam, step) {
        return .{ .context = self };
    }
};

test "optimizers" {
    const allocator = std.testing.allocator;

    // Test SGD
    {
        const sgd = try SGD.init(allocator, 0.1);
        defer sgd.deinit(allocator);
        const optimizer = sgd.optimizer();

        const params = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer params.deinit();
        const grads = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer grads.deinit();

        // Initialize params and grads
        params.data[0] = 1.0;
        params.data[1] = 2.0;
        params.data[2] = 3.0;
        params.data[3] = 4.0;

        grads.data[0] = 0.1;
        grads.data[1] = 0.2;
        grads.data[2] = 0.3;
        grads.data[3] = 0.4;

        try optimizer.step(params, grads);

        try std.testing.expectApproxEqAbs(@as(f64, 0.99), params.data[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f64, 1.98), params.data[1], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f64, 2.97), params.data[2], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f64, 3.96), params.data[3], 1e-6);
    }

    // Test AdaGrad
    {
        const adagrad = try AdaGrad.init(allocator, 0.1, &[_]usize{ 2, 2 });
        defer adagrad.deinit(allocator);
        const optimizer = adagrad.optimizer();

        const params = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer params.deinit();
        const grads = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer grads.deinit();

        // Initialize params and grads
        params.data[0] = 1.0;
        params.data[1] = 2.0;
        params.data[2] = 3.0;
        params.data[3] = 4.0;

        grads.data[0] = 0.1;
        grads.data[1] = 0.2;
        grads.data[2] = 0.3;
        grads.data[3] = 0.4;

        try optimizer.step(params, grads);

        // Check that parameters were updated
        try std.testing.expect(params.data[0] != 1.0);
        try std.testing.expect(params.data[1] != 2.0);
        try std.testing.expect(params.data[2] != 3.0);
        try std.testing.expect(params.data[3] != 4.0);
    }

    // Test RMSprop
    {
        const rmsprop = try RMSprop.init(allocator, 0.1, &[_]usize{ 2, 2 });
        defer rmsprop.deinit(allocator);
        const optimizer = rmsprop.optimizer();

        const params = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer params.deinit();
        const grads = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer grads.deinit();

        // Initialize params and grads
        params.data[0] = 1.0;
        params.data[1] = 2.0;
        params.data[2] = 3.0;
        params.data[3] = 4.0;

        grads.data[0] = 0.1;
        grads.data[1] = 0.2;
        grads.data[2] = 0.3;
        grads.data[3] = 0.4;

        try optimizer.step(params, grads);

        // Check that parameters were updated
        try std.testing.expect(params.data[0] != 1.0);
        try std.testing.expect(params.data[1] != 2.0);
        try std.testing.expect(params.data[2] != 3.0);
        try std.testing.expect(params.data[3] != 4.0);
    }

    // Test Adam
    {
        const adam = try Adam.init(allocator, 0.1, &[_]usize{ 2, 2 });
        defer adam.deinit(allocator);
        const optimizer = adam.optimizer();

        const params = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer params.deinit();
        const grads = try Tensor.init(allocator, &[_]usize{ 2, 2 });
        defer grads.deinit();

        // Initialize params and grads
        params.data[0] = 1.0;
        params.data[1] = 2.0;
        params.data[2] = 3.0;
        params.data[3] = 4.0;

        grads.data[0] = 0.1;
        grads.data[1] = 0.2;
        grads.data[2] = 0.3;
        grads.data[3] = 0.4;

        try optimizer.step(params, grads);

        // Check that parameters were updated
        try std.testing.expect(params.data[0] != 1.0);
        try std.testing.expect(params.data[1] != 2.0);
        try std.testing.expect(params.data[2] != 3.0);
        try std.testing.expect(params.data[3] != 4.0);
    }
}
