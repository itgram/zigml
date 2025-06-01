const std = @import("std");
const Graph = @import("autodiff/graph.zig").Graph;

pub fn main() !void {}

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
