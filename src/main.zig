const std = @import("std");
const Graph = @import("autodiff/graph.zig").Graph;

pub fn main() !void {
    try f1();
    try f2();
    try f3();
    try f4();
}

fn f1() !void {
    const allocator = std.heap.page_allocator;
    var graph = Graph.init(allocator);

    // f = x + y, where x = 2, y = 3
    // ∂f/∂x = 1
    // ∂f/∂y = 1
    const x_tensor = try graph.tensor(&[_]usize{1});
    x_tensor.data[0] = 2.0;

    const y_tensor = try graph.tensor(&[_]usize{1});
    y_tensor.data[0] = 3.0;

    var x = try graph.input("x", x_tensor);
    var y = try graph.input("y", y_tensor);

    // f = x + y
    var f = try graph.add(x.node(), y.node());

    const fVal = f.eval();
    std.debug.print("f = {}\n", .{fVal});

    const df_tensor = try graph.tensor(&[_]usize{1});
    df_tensor.data[0] = 1.0;

    f.diff(df_tensor);
    std.debug.print("∂f/∂x = {?}\n", .{x.grad});
    std.debug.print("∂f/∂y = {?}\n", .{y.grad});

    std.debug.print("-------------------------------\n", .{});
    std.debug.print("\n", .{});
}

fn f2() !void {
    const allocator = std.heap.page_allocator;
    var graph = Graph.init(allocator);

    // f = x * y, where x = 2, y = 3
    // ∂f/∂x = y
    // ∂f/∂y = x
    const x_tensor = try graph.tensor(&[_]usize{1});
    x_tensor.data[0] = 2.0;

    const y_tensor = try graph.tensor(&[_]usize{1});
    y_tensor.data[0] = 3.0;

    var x = try graph.input("x", x_tensor);
    var y = try graph.input("y", y_tensor);

    // f = x + y
    var f = try graph.multiply(x.node(), y.node());

    const fVal = f.eval();
    std.debug.print("f = {d}\n", .{fVal});

    const df_tensor = try graph.tensor(&[_]usize{1});
    df_tensor.data[0] = 1.0;

    f.diff(df_tensor);
    std.debug.print("∂f/∂x = {?d}\n", .{x.grad});
    std.debug.print("∂f/∂y = {?d}\n", .{y.grad});

    std.debug.print("-------------------------------\n", .{});
    std.debug.print("\n", .{});
}

fn f3() !void {
    const allocator = std.heap.page_allocator;
    var graph = Graph.init(allocator);

    // f = x*x - y*y, where x = 2, y = 3
    // ∂f/∂x = 2*x
    // ∂f/∂y = -2*y
    const x_tensor = try graph.tensor(&[_]usize{1});
    x_tensor.data[0] = 2.0;

    const y_tensor = try graph.tensor(&[_]usize{1});
    y_tensor.data[0] = 3.0;

    var x = try graph.input("x", x_tensor);
    var y = try graph.input("y", y_tensor);

    // v1 = x * x
    var v1 = try graph.multiply(x.node(), x.node());

    // v2 = y * y
    var v2 = try graph.multiply(y.node(), y.node());

    // f = v1 - v2
    var f = try graph.subtract(v1.node(), v2.node());

    const fVal = f.eval();
    std.debug.print("f = {d}\n", .{fVal});

    const df_tensor = try graph.tensor(&[_]usize{1});
    df_tensor.data[0] = 1.0;

    f.diff(df_tensor);
    std.debug.print("∂f/∂x = {?d}\n", .{x.grad});
    std.debug.print("∂f/∂y = {?d}\n", .{y.grad});

    std.debug.print("-------------------------------\n", .{});
    std.debug.print("\n", .{});
}

fn f4() !void {
    const allocator = std.heap.page_allocator;
    var graph = Graph.init(allocator);

    // f = x * sin(y + 5) + (y + 5) * (y + 5) * x, where x = 2, y = 3
    // ∂f/∂x = sin(y + 5) + (y + 5) * (y + 5)
    // ∂f/∂y = x * cos(y + 5) + 2 * (y + 5) * x
    const x_tensor = try graph.tensor(&[_]usize{1});
    x_tensor.data[0] = 2.0;

    const y_tensor = try graph.tensor(&[_]usize{1});
    y_tensor.data[0] = 3.0;

    var x = try graph.input("x", x_tensor);
    var y = try graph.input("y", y_tensor);

    const c_tensor = try graph.tensor(&[_]usize{1});
    c_tensor.data[0] = 5.0;

    var c = try graph.constant(c_tensor);

    // v1 = y + c
    var v1 = try graph.add(y.node(), c.node());

    // v2 = v1 * v1
    var v2 = try graph.multiply(v1.node(), v1.node());

    // v3 = v2 * x
    const v3 = try graph.multiply(v2.node(), x.node());

    // v4 = sin(v1)
    var v4 = try graph.sin(v1.node());

    // v5 = x * v4
    var v5 = try graph.multiply(x.node(), v4.node());

    // f = v5 + v3
    var f = try graph.add(v5.node(), v3.node());

    const fVal = f.eval();
    std.debug.print("f = {d}\n", .{fVal});

    const df_tensor = try graph.tensor(&[_]usize{1});
    df_tensor.data[0] = 1.0;

    f.diff(df_tensor);
    std.debug.print("∂f/∂x = {?d}\n", .{x.grad});
    std.debug.print("∂f/∂y = {?d}\n", .{y.grad});

    std.debug.print("-------------------------------\n", .{});
    std.debug.print("\n", .{});
}
