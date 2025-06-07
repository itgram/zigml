pub const Add = @import("add.zig").Add;
pub const BCE = @import("bce.zig").BCE;
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
pub const Node = @import("node.zig").Node;
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

const std = @import("std");

/// Create an addition node
pub fn add(allocator: std.mem.Allocator, x: Node, y: Node) !*Add {
    return try Add.init(allocator, x, y);
}

/// Create a binary cross-entropy loss node
pub fn bce(allocator: std.mem.Allocator, predictions: Node, targets: Node) !*BCE {
    return try BCE.init(allocator, predictions, targets);
}

/// Create a categorical cross-entropy loss node
pub fn cce(allocator: std.mem.Allocator, predictions: Node, targets: Node) !*CCE {
    return try CCE.init(allocator, predictions, targets);
}

/// Create a constant node
pub fn constant(allocator: std.mem.Allocator, value: *Tensor) !*Constant {
    return try Constant.init(allocator, value);
}

/// Create a cosine node
pub fn cos(allocator: std.mem.Allocator, x: Node) !*Cos {
    return try Cos.init(allocator, x);
}

/// Create a division node
pub fn divide(allocator: std.mem.Allocator, x: Node, y: Node) !*Divide {
    return try Divide.init(allocator, x, y);
}

/// Create an elu node
pub fn elu(allocator: std.mem.Allocator, x: Node, alpha: f64) !*ELU {
    return try ELU.init(allocator, x, alpha);
}

/// Create an exponential node
pub fn exp(allocator: std.mem.Allocator, x: Node) !*Exp {
    return try Exp.init(allocator, x);
}

/// Create a gelu node
pub fn gelu(allocator: std.mem.Allocator, x: Node) !*GELU {
    return try GELU.init(allocator, x);
}

/// Create a leaky relu node
pub fn leakyReLU(allocator: std.mem.Allocator, x: Node, alpha: f64) !*LeakyReLU {
    return try LeakyReLU.init(allocator, x, alpha);
}

/// Create a linear node
pub fn linear(allocator: std.mem.Allocator, x: Node) !*Linear {
    return try Linear.init(allocator, x);
}

/// Create a natural logarithm node
pub fn ln(allocator: std.mem.Allocator, x: Node) !*Ln {
    return try Ln.init(allocator, x);
}

/// Create a logarithm node
pub fn log(allocator: std.mem.Allocator, x: Node) !*Log {
    return try Log.init(allocator, x);
}

/// Create a mean absolute error node
pub fn mae(allocator: std.mem.Allocator, predictions: Node, targets: Node) !*MAE {
    return try MAE.init(allocator, predictions, targets);
}

/// Create a matrix multiplication node
pub fn matmul(allocator: std.mem.Allocator, x: Node, y: Node) !*MatMul {
    return try MatMul.init(allocator, x, y);
}

/// Create a mean squared error node
pub fn mse(allocator: std.mem.Allocator, predictions: Node, targets: Node) !*MSE {
    return try MSE.init(allocator, predictions, targets);
}

/// Create a multiplication node
pub fn multiply(allocator: std.mem.Allocator, x: Node, y: Node) !*Multiply {
    return try Multiply.init(allocator, x, y);
}

/// Create a power node
pub fn power(allocator: std.mem.Allocator, x: Node, y: Node) !*Power {
    return try Power.init(allocator, x, y);
}

/// Create a parametric relu node
pub fn prelu(allocator: std.mem.Allocator, x: Node, alpha: *Tensor) !*PReLU {
    return try PReLU.init(allocator, x, alpha);
}

/// Create a relu node
pub fn relu(allocator: std.mem.Allocator, x: Node) !*ReLU {
    return try ReLU.init(allocator, x);
}

/// Create a selu node
pub fn selu(allocator: std.mem.Allocator, x: Node, alpha: f64, lambda: f64) !*SELU {
    return try SELU.init(allocator, x, alpha, lambda);
}

/// Create a sigmoid node
pub fn sigmoid(allocator: std.mem.Allocator, x: Node) !*Sigmoid {
    return try Sigmoid.init(allocator, x);
}

/// Create a sine node
pub fn sin(allocator: std.mem.Allocator, x: Node) !*Sin {
    return try Sin.init(allocator, x);
}

/// Create a softmax_cce node
pub fn softmax_cce(allocator: std.mem.Allocator, x: Node, y: Node, axis: usize) !*SoftmaxCCE {
    return try SoftmaxCCE.init(allocator, x, y, axis);
}

/// Create a softmax node
pub fn softmax(allocator: std.mem.Allocator, x: Node, axis: usize) !*Softmax {
    return try Softmax.init(allocator, x, axis);
}

/// Create a step node
pub fn step(allocator: std.mem.Allocator, x: Node, threshold: f64) !*Step {
    return try Step.init(allocator, x, threshold);
}

/// Create a subtraction node
pub fn subtract(allocator: std.mem.Allocator, x: Node, y: Node) !*Subtract {
    return try Subtract.init(allocator, x, y);
}

/// Create a swish node
pub fn swish(allocator: std.mem.Allocator, x: Node) !*Swish {
    return try Swish.init(allocator, x);
}

/// Create a tangent node
pub fn tan(allocator: std.mem.Allocator, x: Node) !*Tan {
    return try Tan.init(allocator, x);
}

/// Create a hyperbolic tangent node
pub fn tanh(allocator: std.mem.Allocator, x: Node) !*Tanh {
    return try Tanh.init(allocator, x);
}

/// Create a tensor
pub fn tensor(allocator: std.mem.Allocator, shape: []const usize) !*Tensor {
    return try Tensor.init(allocator, shape);
}

/// Create a variable node
pub fn variable(allocator: std.mem.Allocator, name: []const u8, value: *Tensor) !*Variable {
    return try Variable.init(allocator, name, value);
}

// Include all test files
test {
    _ = @import("add.zig");
    _ = @import("bce.zig");
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
    _ = @import("node.zig");
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
    const xTensor = try tensor(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try tensor(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try variable(allocator, "x", xTensor);
    defer x.deinit();
    var y = try variable(allocator, "y", yTensor);
    defer y.deinit();

    // f = x + y
    var f = try add(allocator, x.node(), y.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 5.0), result.data[0]);

    const dfTensor = try tensor(allocator, &[_]usize{1});
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
    const xTensor = try tensor(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try tensor(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try variable(allocator, "x", xTensor);
    defer x.deinit();
    var y = try variable(allocator, "y", yTensor);
    defer y.deinit();

    // f = x * y
    var f = try multiply(allocator, x.node(), y.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 6.0), result.data[0]);

    const dfTensor = try tensor(allocator, &[_]usize{1});
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
    const xTensor = try tensor(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try tensor(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try variable(allocator, "x", xTensor);
    defer x.deinit();
    var y = try variable(allocator, "y", yTensor);
    defer y.deinit();

    // v1 = x * x
    var v1 = try multiply(allocator, x.node(), x.node());
    defer v1.deinit();

    // v2 = y * y
    var v2 = try multiply(allocator, y.node(), y.node());
    defer v2.deinit();

    // f = v1 - v2
    var f = try subtract(allocator, v1.node(), v2.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, -5.0), result.data[0]);

    const dfTensor = try tensor(allocator, &[_]usize{1});
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
    const xTensor = try tensor(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try tensor(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try variable(allocator, "x", xTensor);
    defer x.deinit();
    var y = try variable(allocator, "y", yTensor);
    defer y.deinit();

    const cTensor = try tensor(allocator, &[_]usize{1});
    defer cTensor.deinit();
    cTensor.data[0] = 5.0;

    var c = try constant(allocator, cTensor);
    defer c.deinit();

    // v1 = y + c
    var v1 = try add(allocator, y.node(), c.node());
    defer v1.deinit();

    // v2 = v1 * v1
    var v2 = try multiply(allocator, v1.node(), v1.node());
    defer v2.deinit();

    // v3 = v2 * x
    const v3 = try multiply(allocator, v2.node(), x.node());
    defer v3.deinit();

    // v4 = sin(v1)
    var v4 = try sin(allocator, v1.node());
    defer v4.deinit();

    // v5 = x * v4
    var v5 = try multiply(allocator, x.node(), v4.node());
    defer v5.deinit();

    // f = v5 + v3
    var f = try add(allocator, v5.node(), v3.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 129.97871649324676), result.data[0]);

    const dfTensor = try tensor(allocator, &[_]usize{1});
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
    const xTensor = try tensor(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    var x = try variable(allocator, "x", xTensor);
    defer x.deinit();

    const cTensor = try tensor(allocator, &[_]usize{1});
    defer cTensor.deinit();
    cTensor.data[0] = 2.0;

    var c = try constant(allocator, cTensor);
    defer c.deinit();

    // v1 = x + c
    var v1 = try add(allocator, x.node(), c.node());
    defer v1.deinit();

    // v2 = sin(x)
    var v2 = try sin(allocator, x.node());
    defer v2.deinit();

    // f = v1 * v2
    const f = try multiply(allocator, v1.node(), v2.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 3.637189707302727), result.data[0]);

    const dfTensor = try tensor(allocator, &[_]usize{1});
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
    const xTensor = try tensor(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try tensor(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    const zTensor = try tensor(allocator, &[_]usize{1});
    defer zTensor.deinit();
    zTensor.data[0] = 4.0;

    var x = try variable(allocator, "x", xTensor);
    defer x.deinit();
    var y = try variable(allocator, "y", yTensor);
    defer y.deinit();
    var z = try variable(allocator, "z", zTensor);
    defer z.deinit();

    // v1 = x + y
    var v1 = try add(allocator, x.node(), y.node());
    defer v1.deinit();

    // v2 = y + z
    var v2 = try add(allocator, y.node(), z.node());
    defer v2.deinit();

    // f = v1 + v2
    const f = try add(allocator, v1.node(), v2.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 12), result.data[0]);

    const dfTensor = try tensor(allocator, &[_]usize{1});
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
    const xTensor = try tensor(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try tensor(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    const zTensor = try tensor(allocator, &[_]usize{1});
    defer zTensor.deinit();
    zTensor.data[0] = 4.0;

    var x = try variable(allocator, "x", xTensor);
    defer x.deinit();
    var y = try variable(allocator, "y", yTensor);
    defer y.deinit();
    var z = try variable(allocator, "z", zTensor);
    defer z.deinit();

    // v1 = x + y
    var v1 = try add(allocator, x.node(), y.node());
    defer v1.deinit();

    // v2 = relu(v1)
    var v2 = try relu(allocator, v1.node());
    defer v2.deinit();

    // v3 = y + z
    var v3 = try add(allocator, y.node(), z.node());
    defer v3.deinit();

    // f = v2 + v3
    const f = try add(allocator, v2.node(), v3.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 12), result.data[0]);

    const dfTensor = try tensor(allocator, &[_]usize{1});
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
    const xTensor = try tensor(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try tensor(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = -3.0;

    const zTensor = try tensor(allocator, &[_]usize{1});
    defer zTensor.deinit();
    zTensor.data[0] = 4.0;

    var x = try variable(allocator, "x", xTensor);
    defer x.deinit();
    var y = try variable(allocator, "y", yTensor);
    defer y.deinit();
    var z = try variable(allocator, "z", zTensor);
    defer z.deinit();

    const alpha = 0.01;

    // v1 = x + y
    var v1 = try add(allocator, x.node(), y.node());
    defer v1.deinit();

    // v2 = leakyReLU(v1)
    var v2 = try leakyReLU(allocator, v1.node(), alpha);
    defer v2.deinit();

    // v3 = y + z
    var v3 = try add(allocator, y.node(), z.node());
    defer v3.deinit();

    // f = v2 + v3
    const f = try add(allocator, v2.node(), v3.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 0.99), result.data[0]);

    const dfTensor = try tensor(allocator, &[_]usize{1});
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
    const xTensor = try tensor(allocator, &[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try tensor(allocator, &[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try variable(allocator, "x", xTensor);
    defer x.deinit();
    var y = try variable(allocator, "y", yTensor);
    defer y.deinit();

    // v1 = x + y
    var v1 = try add(allocator, x.node(), y.node());
    defer v1.deinit();

    // f = sigmoid(v1)
    const f = try sigmoid(allocator, v1.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 0.9933071490757153), result.data[0]);

    const dfTensor = try tensor(allocator, &[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 0.006648056670790033), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 0.006648056670790033), y.grad.data[0]);
}
