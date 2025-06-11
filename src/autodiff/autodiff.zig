const std = @import("std");

pub const Add = @import("add.zig").Add;
pub const BCE = @import("bce.zig").BCE;
pub const Broadcast = @import("broadcast.zig").Broadcast;
pub const CCE = @import("cce.zig").CCE;
pub const Constant = @import("constant.zig").Constant;
pub const Cos = @import("cos.zig").Cos;
pub const Divide = @import("divide.zig").Divide;
pub const ELU = @import("elu.zig").ELU;
pub const Exp = @import("exp.zig").Exp;
pub const GELU = @import("gelu.zig").GELU;
pub const LeakyReLU = @import("leaky_relu.zig").LeakyReLU;
pub const Linear = @import("linear.zig").Linear;
pub const Ln = @import("ln.zig").Ln;
pub const Log = @import("log.zig").Log;
pub const MAE = @import("mae.zig").MAE;
pub const MatMul = @import("matmul.zig").MatMul;
pub const MSE = @import("mse.zig").MSE;
pub const Multiply = @import("multiply.zig").Multiply;
pub const Power = @import("power.zig").Power;
pub const PReLU = @import("prelu.zig").PReLU;
pub const ReLU = @import("relu.zig").ReLU;
pub const SELU = @import("selu.zig").SELU;
pub const Sigmoid = @import("sigmoid.zig").Sigmoid;
pub const Sin = @import("sin.zig").Sin;
pub const SoftmaxCCE = @import("softmax_cce.zig").SoftmaxCCE;
pub const Softmax = @import("softmax.zig").Softmax;
pub const Step = @import("step.zig").Step;
pub const Subtract = @import("subtract.zig").Subtract;
pub const Swish = @import("swish.zig").Swish;
pub const Tan = @import("tan.zig").Tan;
pub const Tanh = @import("tanh.zig").Tanh;
pub const Tensor = @import("tensor.zig").Tensor;
pub const Variable = @import("variable.zig").Variable;

/// Node interface for autodiff.
/// The Node interface defines the structure for nodes in the computation graph.
/// Each node must implement the `eval`, `diff`, and `reset` methods.
pub const Node = union(enum) {
    add: *Add,
    bce: *BCE,
    broadcast: *Broadcast,
    cce: *CCE,
    constant: *Constant,
    cos: *Cos,
    divide: *Divide,
    elu: *ELU,
    exp: *Exp,
    gelu: *GELU,
    leaky_relu: *LeakyReLU,
    linear: *Linear,
    ln: *Ln,
    log: *Log,
    mae: *MAE,
    matmul: *MatMul,
    mse: *MSE,
    multiply: *Multiply,
    power: *Power,
    prelu: *PReLU,
    relu: *ReLU,
    selu: *SELU,
    sigmoid: *Sigmoid,
    sin: *Sin,
    softmax_cce: *SoftmaxCCE,
    softmax: *Softmax,
    step: *Step,
    subtract: *Subtract,
    swish: *Swish,
    tan: *Tan,
    tanh: *Tanh,
    variable: *Variable,

    /// Creates a new node from a concrete implementation.
    pub fn init(pointer: anytype) Node {
        const T = @TypeOf(pointer);
        const ptrInfo = @typeInfo(T);

        std.debug.assert(ptrInfo == .pointer); // Must be a pointer
        std.debug.assert(ptrInfo.pointer.size == .one); // Must be a single-item pointer
        std.debug.assert(@typeInfo(ptrInfo.pointer.child) == .@"struct"); // Must point to a struct

        inline for (std.meta.fields(Node)) |field| {
            if (field.type == T) {
                return @unionInit(Node, field.name, pointer);
            }
        }
        @compileError("Invalid node type: " ++ @typeName(T));
    }

    /// Evaluates the node and returns the result tensor.
    pub fn eval(self: Node) anyerror!*Tensor {
        return switch (self) {
            inline else => |n| try n.eval(),
        };
    }

    /// Computes the gradient of the node with respect to its inputs.
    pub fn diff(self: Node, dval: *Tensor) anyerror!void {
        try switch (self) {
            inline else => |n| n.diff(dval),
        };
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: Node) void {
        switch (self) {
            inline else => |n| n.reset(),
        }
    }
};

// Include all test files
test {
    _ = @import("add.zig");
    _ = @import("bce.zig");
    _ = @import("broadcast.zig");
    _ = @import("cce.zig");
    _ = @import("constant.zig");
    _ = @import("cos.zig");
    _ = @import("divide.zig");
    _ = @import("elu.zig");
    _ = @import("exp.zig");
    _ = @import("gelu.zig");
    _ = @import("leaky_relu.zig");
    _ = @import("linear.zig");
    _ = @import("ln.zig");
    _ = @import("log.zig");
    _ = @import("mae.zig");
    _ = @import("matmul.zig");
    _ = @import("mse.zig");
    _ = @import("multiply.zig");
    _ = @import("power.zig");
    _ = @import("prelu.zig");
    _ = @import("relu.zig");
    _ = @import("selu.zig");
    _ = @import("sigmoid.zig");
    _ = @import("sin.zig");
    _ = @import("softmax_cce.zig");
    _ = @import("softmax.zig");
    _ = @import("step.zig");
    _ = @import("subtract.zig");
    _ = @import("swish.zig");
    _ = @import("tan.zig");
    _ = @import("tanh.zig");
    _ = @import("tensor.zig");
    _ = @import("variable.zig");
}

test "add operation eval and diff" {
    const allocator = std.testing.allocator;

    // f = x + y, where x = 2, y = 3
    // ∂f/∂x = 1
    // ∂f/∂y = 1
    const xTensor = try Tensor.init(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // f = x + y
    var f = try Add.init(allocator, x.node(), y.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 5.0), result.data[0]);

    const dfTensor = try Tensor.init(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 1.0), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 1.0), y.grad.data[0]);
}

test "multiply operation eval and diff" {
    const allocator = std.testing.allocator;

    // f = x * y, where x = 2, y = 3
    // ∂f/∂x = y
    // ∂f/∂y = x
    const xTensor = try Tensor.init(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // f = x * y
    var f = try Multiply.init(allocator, x.node(), y.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 6.0), result.data[0]);

    const dfTensor = try Tensor.init(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 3.0), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 2.0), y.grad.data[0]);
}

test "multiply operation with subtract eval and diff" {
    const allocator = std.testing.allocator;

    // f = x*x - y*y, where x = 2, y = 3
    // ∂f/∂x = 2*x
    // ∂f/∂y = -2*y
    const xTensor = try Tensor.init(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // v1 = x * x
    var v1 = try Multiply.init(allocator, x.node(), x.node());
    defer v1.deinit();

    // v2 = y * y
    var v2 = try Multiply.init(allocator, y.node(), y.node());
    defer v2.deinit();

    // f = v1 - v2
    var f = try Subtract.init(allocator, v1.node(), v2.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, -5.0), result.data[0]);

    const dfTensor = try Tensor.init(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 4.0), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, -6.0), y.grad.data[0]);
}

test "sin operation with other operations eval and diff" {
    const allocator = std.testing.allocator;

    // f = x * sin(y + 5) + (y + 5) * (y + 5) * x, where x = 2, y = 3
    // ∂f/∂x = sin(y + 5) + (y + 5) * (y + 5)
    // ∂f/∂y = x * cos(y + 5) + 2 * (y + 5) * x
    const xTensor = try Tensor.init(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    const cTensor = try Tensor.init(allocator, &[_]usize{1});
    defer cTensor.deinit();
    cTensor.data[0] = 5.0;

    var c = try Constant.init(allocator, cTensor);
    defer c.deinit();

    // v1 = y + c
    var v1 = try Add.init(allocator, y.node(), c.node());
    defer v1.deinit();

    // v2 = v1 * v1
    var v2 = try Multiply.init(allocator, v1.node(), v1.node());
    defer v2.deinit();

    // v3 = v2 * x
    const v3 = try Multiply.init(allocator, v2.node(), x.node());
    defer v3.deinit();

    // v4 = sin(v1)
    var v4 = try Sin.init(allocator, v1.node());
    defer v4.deinit();

    // v5 = x * v4
    var v5 = try Multiply.init(allocator, x.node(), v4.node());
    defer v5.deinit();

    // f = v5 + v3
    var f = try Add.init(allocator, v5.node(), v3.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 129.97871649324676), result.data[0]);

    const dfTensor = try Tensor.init(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 64.98935824662338), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 31.708999932382774), y.grad.data[0]);
}

test "duplicate input eval and diff" {
    const allocator = std.testing.allocator;

    // f = (x + 2) * sin(x), where x = 2
    // ∂f/∂x = sin(x) + (x + 2) * cos(x)
    const xTensor = try Tensor.init(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();

    const cTensor = try Tensor.init(allocator, &[_]usize{1});
    defer cTensor.deinit();
    cTensor.data[0] = 2.0;

    var c = try Constant.init(allocator, cTensor);
    defer c.deinit();

    // v1 = x + c
    var v1 = try Add.init(allocator, x.node(), c.node());
    defer v1.deinit();

    // v2 = sin(x)
    var v2 = try Sin.init(allocator, x.node());
    defer v2.deinit();

    // f = v1 * v2
    const f = try Multiply.init(allocator, v1.node(), v2.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 3.637189707302727), result.data[0]);

    const dfTensor = try Tensor.init(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, -0.7552899193628879), x.grad.data[0]);
}

test "shared input eval and diff" {
    const allocator = std.testing.allocator;

    // f = (x + y) + (y + z), where x = 2, y = 3, z = 4
    // ∂f/∂x = 1
    // ∂f/∂y = 2
    // ∂f/∂z = 1
    const xTensor = try Tensor.init(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    const zTensor = try Tensor.init(allocator, &[_]usize{1});
    defer zTensor.deinit();
    zTensor.data[0] = 4.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();
    var z = try Variable.init(allocator, "z", zTensor);
    defer z.deinit();

    // v1 = x + y
    var v1 = try Add.init(allocator, x.node(), y.node());
    defer v1.deinit();

    // v2 = y + z
    var v2 = try Add.init(allocator, y.node(), z.node());
    defer v2.deinit();

    // f = v1 + v2
    const f = try Add.init(allocator, v1.node(), v2.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 12), result.data[0]);

    const dfTensor = try Tensor.init(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 1), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 2), y.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 1), z.grad.data[0]);
}

test "relu eval and diff" {
    const allocator = std.testing.allocator;

    // f = relu(x + y) + (y + z), where x = 2, y = 3, z = 4
    // ∂f/∂x = 1 if x + y > 0 else 0
    // ∂f/∂y = (1 if x + y > 0 else 0) + 1
    // ∂f/∂z = 1
    const xTensor = try Tensor.init(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    const zTensor = try Tensor.init(allocator, &[_]usize{1});
    defer zTensor.deinit();
    zTensor.data[0] = 4.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();
    var z = try Variable.init(allocator, "z", zTensor);
    defer z.deinit();

    // v1 = x + y
    var v1 = try Add.init(allocator, x.node(), y.node());
    defer v1.deinit();

    // v2 = relu(v1)
    var v2 = try ReLU.init(allocator, v1.node());
    defer v2.deinit();

    // v3 = y + z
    var v3 = try Add.init(allocator, y.node(), z.node());
    defer v3.deinit();

    // f = v2 + v3
    const f = try Add.init(allocator, v2.node(), v3.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 12), result.data[0]);

    const dfTensor = try Tensor.init(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 1), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 2), y.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 1), z.grad.data[0]);
}

test "leaky relu eval and diff" {
    const allocator = std.testing.allocator;

    // f = leakyReLU(x + y, alpha) + (y + z), where x = 2, y = -3, z = 4, alpha = 0.01
    // ∂f/∂x = alpha if x + y <= 0 else 1
    // ∂f/∂y = (alpha if x + y <= 0 else 1) + 1
    // ∂f/∂z = 1
    const xTensor = try Tensor.init(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = -3.0;

    const zTensor = try Tensor.init(allocator, &[_]usize{1});
    defer zTensor.deinit();
    zTensor.data[0] = 4.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();
    var z = try Variable.init(allocator, "z", zTensor);
    defer z.deinit();

    const alpha = 0.01;

    // v1 = x + y
    var v1 = try Add.init(allocator, x.node(), y.node());
    defer v1.deinit();

    // v2 = leakyReLU(v1)
    var v2 = try LeakyReLU.init(allocator, v1.node(), alpha);
    defer v2.deinit();

    // v3 = y + z
    var v3 = try Add.init(allocator, y.node(), z.node());
    defer v3.deinit();

    // f = v2 + v3
    const f = try Add.init(allocator, v2.node(), v3.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 0.99), result.data[0]);

    const dfTensor = try Tensor.init(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, alpha), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, alpha + 1), y.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 1), z.grad.data[0]);
}

test "sigmoid eval and diff" {
    const allocator = std.testing.allocator;

    // f = sigmoid(x + y), where x = 2, y = 3
    // ∂f/∂x = sigmoid(x + y) * (1 - sigmoid(x + y))
    // ∂f/∂y = sigmoid(x + y) * (1 - sigmoid(x + y))
    const xTensor = try Tensor.init(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try Tensor.init(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try Variable.init(allocator, "x", xTensor);
    defer x.deinit();
    var y = try Variable.init(allocator, "y", yTensor);
    defer y.deinit();

    // v1 = x + y
    var v1 = try Add.init(allocator, x.node(), y.node());
    defer v1.deinit();

    // f = sigmoid(v1)
    const f = try Sigmoid.init(allocator, v1.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 0.9933071490757153), result.data[0]);

    const dfTensor = try Tensor.init(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 0.006648056670790033), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 0.006648056670790033), y.grad.data[0]);
}
