const std = @import("std");

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

/// Graph structure for building computational graphs.
/// The Graph struct provides methods to create and manipulate nodes in a computational graph.
/// It allows the user to define operations such as addition, multiplication, and activation functions.
/// The Graph struct is designed to be used with an allocator for memory management.
pub const Graph = struct {
    allocator: std.mem.Allocator,

    /// Initialize a new Graph with the given allocator.
    pub fn init(allocator: std.mem.Allocator) Graph {
        return Graph{ .allocator = allocator };
    }

    /// Create an addition node
    pub fn add(self: *Graph, x: Node, y: Node) !*Add {
        return try Add.init(self.allocator, x, y);
    }

    /// Create a binary cross-entropy loss node
    pub fn bce(self: *Graph, predictions: Node, targets: Node) !*BCE {
        return try BCE.init(self.allocator, predictions, targets);
    }

    /// Create a categorical cross-entropy loss node
    pub fn cce(self: *Graph, predictions: Node, targets: Node) !*CCE {
        return try CCE.init(self.allocator, predictions, targets);
    }

    /// Create a constant node
    pub fn constant(self: *Graph, value: *Tensor) !*Constant {
        return try Constant.init(self.allocator, value);
    }

    /// Create a cosine node
    pub fn cos(self: *Graph, x: Node) !*Cos {
        return try Cos.init(self.allocator, x);
    }

    /// Create a division node
    pub fn divide(self: *Graph, x: Node, y: Node) !*Divide {
        return try Divide.init(self.allocator, x, y);
    }

    /// Create an elu node
    pub fn elu(self: *Graph, x: Node, alpha: f64) !*ELU {
        return try ELU.init(self.allocator, x, alpha);
    }

    /// Create an exponential node
    pub fn exp(self: *Graph, x: Node) !*Exp {
        return try Exp.init(self.allocator, x);
    }

    /// Create a gelu node
    pub fn gelu(self: *Graph, x: Node) !*GELU {
        return try GELU.init(self.allocator, x);
    }

    /// Create a leaky relu node
    pub fn leakyReLU(self: *Graph, x: Node, alpha: f64) !*LeakyReLU {
        return try LeakyReLU.init(self.allocator, x, alpha);
    }

    /// Create a linear node
    pub fn linear(self: *Graph, x: Node) !*Linear {
        return try Linear.init(self.allocator, x);
    }

    /// Create a natural logarithm node
    pub fn ln(self: *Graph, x: Node) !*Ln {
        return try Ln.init(self.allocator, x);
    }

    /// Create a logarithm node
    pub fn log(self: *Graph, x: Node) !*Log {
        return try Log.init(self.allocator, x);
    }

    /// Create a mean absolute error node
    pub fn mae(self: *Graph, predictions: Node, targets: Node) !*MAE {
        return try MAE.init(self.allocator, predictions, targets);
    }

    /// Create a mean squared error node
    pub fn mse(self: *Graph, predictions: Node, targets: Node) !*MSE {
        return try MSE.init(self.allocator, predictions, targets);
    }

    /// Create a multiplication node
    pub fn multiply(self: *Graph, x: Node, y: Node) !*Multiply {
        return try Multiply.init(self.allocator, x, y);
    }

    /// Create a power node
    pub fn power(self: *Graph, x: Node, y: Node) !*Power {
        return try Power.init(self.allocator, x, y);
    }

    /// Create a parametric relu node
    pub fn prelu(self: *Graph, x: Node, alpha: *Tensor) !*PReLU {
        return try PReLU.init(self.allocator, x, alpha);
    }

    /// Create a relu node
    pub fn relu(self: *Graph, x: Node) !*ReLU {
        return try ReLU.init(self.allocator, x);
    }

    /// Create a selu node
    pub fn selu(self: *Graph, x: Node, alpha: f64, lambda: f64) !*SELU {
        return try SELU.init(self.allocator, x, alpha, lambda);
    }

    /// Create a sigmoid node
    pub fn sigmoid(self: *Graph, x: Node) !*Sigmoid {
        return try Sigmoid.init(self.allocator, x);
    }

    /// Create a sine node
    pub fn sin(self: *Graph, x: Node) !*Sin {
        return try Sin.init(self.allocator, x);
    }

    /// Create a softmax_cce node
    pub fn softmax_cce(self: *Graph, x: Node, y: Node, axis: usize) !*SoftmaxCCE {
        return try SoftmaxCCE.init(self.allocator, x, y, axis);
    }

    /// Create a softmax node
    pub fn softmax(self: *Graph, x: Node, axis: usize) !*Softmax {
        return try Softmax.init(self.allocator, x, axis);
    }

    /// Create a step node
    pub fn step(self: *Graph, x: Node, threshold: f64) !*Step {
        return try Step.init(self.allocator, x, threshold);
    }

    /// Create a subtraction node
    pub fn subtract(self: *Graph, x: Node, y: Node) !*Subtract {
        return try Subtract.init(self.allocator, x, y);
    }

    /// Create a swish node
    pub fn swish(self: *Graph, x: Node) !*Swish {
        return try Swish.init(self.allocator, x);
    }

    /// Create a tensor with the given shape
    pub fn tensor(self: *Graph, shape: []const usize) !*Tensor {
        return try Tensor.init(self.allocator, shape);
    }

    /// Create a tangent node
    pub fn tan(self: *Graph, x: Node) !*Tan {
        return try Tan.init(self.allocator, x);
    }

    /// Create a hyperbolic tangent node
    pub fn tanh(self: *Graph, x: Node) !*Tanh {
        return try Tanh.init(self.allocator, x);
    }

    /// Create an input variable node
    pub fn variable(self: *Graph, name: []const u8, value: *Tensor) !*Variable {
        return try Variable.init(self.allocator, name, value);
    }
};

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
    _ = @import("graph.zig");
    _ = @import("leaky_relu.zig");
    _ = @import("linear.zig");
    _ = @import("ln.zig");
    _ = @import("log.zig");
    _ = @import("mae.zig");
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
    var graph = Graph.init(allocator);

    // f = x + y, where x = 2, y = 3
    // ∂f/∂x = 1
    // ∂f/∂y = 1
    const xTensor = try graph.tensor(&[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try graph.tensor(&[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // f = x + y
    var f = try graph.add(x.node(), y.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 5.0), result.data[0]);

    const dfTensor = try graph.tensor(&[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 1.0), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 1.0), y.grad.data[0]);
}

test "multiply operation eval and diff" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // f = x * y, where x = 2, y = 3
    // ∂f/∂x = y
    // ∂f/∂y = x
    const xTensor = try graph.tensor(&[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try graph.tensor(&[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // f = x * y
    var f = try graph.multiply(x.node(), y.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 6.0), result.data[0]);

    const dfTensor = try graph.tensor(&[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 3.0), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 2.0), y.grad.data[0]);
}

test "multiply operation with subtract eval and diff" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // f = x*x - y*y, where x = 2, y = 3
    // ∂f/∂x = 2*x
    // ∂f/∂y = -2*y
    const xTensor = try graph.tensor(&[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try graph.tensor(&[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // v1 = x * x
    var v1 = try graph.multiply(x.node(), x.node());
    defer v1.deinit();

    // v2 = y * y
    var v2 = try graph.multiply(y.node(), y.node());
    defer v2.deinit();

    // f = v1 - v2
    var f = try graph.subtract(v1.node(), v2.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, -5.0), result.data[0]);

    const dfTensor = try graph.tensor(&[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 4.0), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, -6.0), y.grad.data[0]);
}

test "sin operation with other operations eval and diff" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // f = x * sin(y + 5) + (y + 5) * (y + 5) * x, where x = 2, y = 3
    // ∂f/∂x = sin(y + 5) + (y + 5) * (y + 5)
    // ∂f/∂y = x * cos(y + 5) + 2 * (y + 5) * x
    const xTensor = try graph.tensor(&[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try graph.tensor(&[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    const cTensor = try graph.tensor(&[_]usize{1});
    defer cTensor.deinit();
    cTensor.data[0] = 5.0;

    var c = try graph.constant(cTensor);
    defer c.deinit();

    // v1 = y + c
    var v1 = try graph.add(y.node(), c.node());
    defer v1.deinit();

    // v2 = v1 * v1
    var v2 = try graph.multiply(v1.node(), v1.node());
    defer v2.deinit();

    // v3 = v2 * x
    const v3 = try graph.multiply(v2.node(), x.node());
    defer v3.deinit();

    // v4 = sin(v1)
    var v4 = try graph.sin(v1.node());
    defer v4.deinit();

    // v5 = x * v4
    var v5 = try graph.multiply(x.node(), v4.node());
    defer v5.deinit();

    // f = v5 + v3
    var f = try graph.add(v5.node(), v3.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 129.97871649324676), result.data[0]);

    const dfTensor = try graph.tensor(&[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 64.98935824662338), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 31.708999932382774), y.grad.data[0]);
}

test "duplicate input eval and diff" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // f = (x + 2) * sin(x), where x = 2
    // ∂f/∂x = sin(x) + (x + 2) * cos(x)
    const xTensor = try graph.tensor(&[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();

    const cTensor = try graph.tensor(&[_]usize{1});
    defer cTensor.deinit();
    cTensor.data[0] = 2.0;

    var c = try graph.constant(cTensor);
    defer c.deinit();

    // v1 = x + c
    var v1 = try graph.add(x.node(), c.node());
    defer v1.deinit();

    // v2 = sin(x)
    var v2 = try graph.sin(x.node());
    defer v2.deinit();

    // f = v1 * v2
    const f = try graph.multiply(v1.node(), v2.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 3.637189707302727), result.data[0]);

    const dfTensor = try graph.tensor(&[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, -0.7552899193628879), x.grad.data[0]);
}

test "shared input eval and diff" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // f = (x + y) + (y + z), where x = 2, y = 3, z = 4
    // ∂f/∂x = 1
    // ∂f/∂y = 2
    // ∂f/∂z = 1
    const xTensor = try graph.tensor(&[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try graph.tensor(&[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    const zTensor = try graph.tensor(&[_]usize{1});
    defer zTensor.deinit();
    zTensor.data[0] = 4.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();
    var z = try graph.variable("z", zTensor);
    defer z.deinit();

    // v1 = x + y
    var v1 = try graph.add(x.node(), y.node());
    defer v1.deinit();

    // v2 = y + z
    var v2 = try graph.add(y.node(), z.node());
    defer v2.deinit();

    // f = v1 + v2
    const f = try graph.add(v1.node(), v2.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 12), result.data[0]);

    const dfTensor = try graph.tensor(&[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 1), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 2), y.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 1), z.grad.data[0]);
}

test "relu eval and diff" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // f = relu(x + y) + (y + z), where x = 2, y = 3, z = 4
    // ∂f/∂x = 1 if x + y > 0 else 0
    // ∂f/∂y = (1 if x + y > 0 else 0) + 1
    // ∂f/∂z = 1
    const xTensor = try graph.tensor(&[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try graph.tensor(&[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    const zTensor = try graph.tensor(&[_]usize{1});
    defer zTensor.deinit();
    zTensor.data[0] = 4.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();
    var z = try graph.variable("z", zTensor);
    defer z.deinit();

    // v1 = x + y
    var v1 = try graph.add(x.node(), y.node());
    defer v1.deinit();

    // v2 = relu(v1)
    var v2 = try graph.relu(v1.node());
    defer v2.deinit();

    // v3 = y + z
    var v3 = try graph.add(y.node(), z.node());
    defer v3.deinit();

    // f = v2 + v3
    const f = try graph.add(v2.node(), v3.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 12), result.data[0]);

    const dfTensor = try graph.tensor(&[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 1), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 2), y.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 1), z.grad.data[0]);
}

test "leaky relu eval and diff" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // f = leakyReLU(x + y, alpha) + (y + z), where x = 2, y = -3, z = 4, alpha = 0.01
    // ∂f/∂x = alpha if x + y <= 0 else 1
    // ∂f/∂y = (alpha if x + y <= 0 else 1) + 1
    // ∂f/∂z = 1
    const xTensor = try graph.tensor(&[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try graph.tensor(&[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = -3.0;

    const zTensor = try graph.tensor(&[_]usize{1});
    defer zTensor.deinit();
    zTensor.data[0] = 4.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();
    var z = try graph.variable("z", zTensor);
    defer z.deinit();

    const alpha = 0.01;

    // v1 = x + y
    var v1 = try graph.add(x.node(), y.node());
    defer v1.deinit();

    // v2 = leakyReLU(v1)
    var v2 = try graph.leakyReLU(v1.node(), alpha);
    defer v2.deinit();

    // v3 = y + z
    var v3 = try graph.add(y.node(), z.node());
    defer v3.deinit();

    // f = v2 + v3
    const f = try graph.add(v2.node(), v3.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 0.99), result.data[0]);

    const dfTensor = try graph.tensor(&[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, alpha), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, alpha + 1), y.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 1), z.grad.data[0]);
}

test "sigmoid eval and diff" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);

    // f = sigmoid(x + y), where x = 2, y = 3
    // ∂f/∂x = sigmoid(x + y) * (1 - sigmoid(x + y))
    // ∂f/∂y = sigmoid(x + y) * (1 - sigmoid(x + y))
    const xTensor = try graph.tensor(&[_]usize{1});
    defer xTensor.deinit();
    xTensor.data[0] = 2.0;

    const yTensor = try graph.tensor(&[_]usize{1});
    defer yTensor.deinit();
    yTensor.data[0] = 3.0;

    var x = try graph.variable("x", xTensor);
    defer x.deinit();
    var y = try graph.variable("y", yTensor);
    defer y.deinit();

    // v1 = x + y
    var v1 = try graph.add(x.node(), y.node());
    defer v1.deinit();

    // f = sigmoid(v1)
    const f = try graph.sigmoid(v1.node());
    defer f.deinit();

    const result = try f.eval();
    try std.testing.expectEqual(@as(f64, 0.9933071490757153), result.data[0]);

    const dfTensor = try graph.tensor(&[_]usize{1});
    defer dfTensor.deinit();
    dfTensor.data[0] = 1.0;

    try f.diff(dfTensor);
    try std.testing.expectEqual(@as(f64, 0.006648056670790033), x.grad.data[0]);
    try std.testing.expectEqual(@as(f64, 0.006648056670790033), y.grad.data[0]);
}
