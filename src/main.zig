const std = @import("std");
const autodiff = @import("autodiff/autodiff.zig");

pub fn main() !void {
    try f1();
    try f2();
    try f3();
    try f4();
}

fn f1() !void {
    const allocator = std.heap.page_allocator;

    // f = x + y, where x = 2, y = 3
    // ∂f/∂x = 1
    // ∂f/∂y = 1
    var x = try autodiff.Variable.init(allocator, "x", 2.0);
    var y = try autodiff.Variable.init(allocator, "y", 3.0);

    // f = x + y
    var f = try autodiff.Add.init(allocator, x.node(), y.node());

    const fVal = f.eval();
    std.debug.print("f = {d}\n", .{fVal});

    f.diff(1.0);
    std.debug.print("∂f/∂x = {?d}\n", .{x.grad});
    std.debug.print("∂f/∂y = {?d}\n", .{y.grad});

    std.debug.print("-------------------------------\n", .{});
    std.debug.print("\n", .{});
}

fn f2() !void {
    const allocator = std.heap.page_allocator;

    // f = x * y, where x = 2, y = 3
    // ∂f/∂x = y
    // ∂f/∂y = x
    var x = try autodiff.Variable.init(allocator, "x", 2.0);
    var y = try autodiff.Variable.init(allocator, "y", 3.0);

    // f = x + y
    var f = try autodiff.Multiply.init(allocator, x.node(), y.node());

    const fVal = f.eval();
    std.debug.print("f = {d}\n", .{fVal});

    f.diff(1.0);
    std.debug.print("∂f/∂x = {?d}\n", .{x.grad});
    std.debug.print("∂f/∂y = {?d}\n", .{y.grad});

    std.debug.print("-------------------------------\n", .{});
    std.debug.print("\n", .{});
}

fn f3() !void {
    const allocator = std.heap.page_allocator;

    // f = x*x - y*y, where x = 2, y = 3
    // ∂f/∂x = 2*x
    // ∂f/∂y = -2*y
    var x = try autodiff.Variable.init(allocator, "x", 2.0);
    var y = try autodiff.Variable.init(allocator, "y", 3.0);

    // v1 = x * x
    var v1 = try autodiff.Multiply.init(allocator, x.node(), x.node());

    // v2 = y * y
    var v2 = try autodiff.Multiply.init(allocator, y.node(), y.node());

    // f = v1 - v2
    var f = try autodiff.Subtract.init(allocator, v1.node(), v2.node());

    const fVal = f.eval();
    std.debug.print("f = {d}\n", .{fVal});

    f.diff(1.0);
    std.debug.print("∂f/∂x = {?d}\n", .{x.grad});
    std.debug.print("∂f/∂y = {?d}\n", .{y.grad});

    std.debug.print("-------------------------------\n", .{});
    std.debug.print("\n", .{});
}

fn f4() !void {
    const allocator = std.heap.page_allocator;

    // f = x * sin(y + 5) + (y + 5) * (y + 5) * x, where x = 2, y = 3
    // ∂f/∂x = sin(y + 5) + (y + 5) * (y + 5)
    // ∂f/∂y = x * cos(y + 5) + 2 * (y + 5) * x
    var x = try autodiff.Variable.init(allocator, "x", 2.0);
    var y = try autodiff.Variable.init(allocator, "y", 3.0);
    var c = try autodiff.Constant.init(allocator, 5.0);

    // v1 = y + c
    var v1 = try autodiff.Add.init(allocator, y.node(), c.node());

    // v2 = v1 * v1
    var v2 = try autodiff.Multiply.init(allocator, v1.node(), v1.node());

    // v3 = v2 * x
    const v3 = try autodiff.Multiply.init(allocator, v2.node(), x.node());

    // v4 = sin(v1)
    var v4 = try autodiff.Sin.init(allocator, v1.node());

    // v5 = x * v4
    var v5 = try autodiff.Multiply.init(allocator, x.node(), v4.node());

    // f = v5 + v3
    var f = try autodiff.Add.init(allocator, v5.node(), v3.node());

    const fVal = f.eval();
    std.debug.print("f = {d}\n", .{fVal});

    f.diff(1.0);
    std.debug.print("∂f/∂x = {?d}\n", .{x.grad});
    std.debug.print("∂f/∂y = {?d}\n", .{y.grad});

    std.debug.print("-------------------------------\n", .{});
    std.debug.print("\n", .{});
}
