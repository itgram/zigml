const std = @import("std");

pub const Add = @import("add.zig").Add;
pub const Constant = @import("constant.zig").Constant;
pub const Divide = @import("divide.zig").Divide;
pub const Ln = @import("ln.zig").Ln;
pub const Log = @import("log.zig").Log;
pub const Multiply = @import("multiply.zig").Multiply;
pub const Node = @import("node.zig").Node;
pub const Power = @import("power.zig").Power;
pub const Sin = @import("sin.zig").Sin;
pub const Subtract = @import("subtract.zig").Subtract;
pub const Tensor = @import("tensor.zig").Tensor;
pub const Variable = @import("variable.zig").Variable;

/// Graph abstraction to manage node creation and memory
pub const Graph = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Graph {
        return Graph{ .allocator = allocator };
    }

    /// Add two nodes
    pub fn add(self: *Graph, a: Node, b: Node) !*Add {
        return try Add.init(self.allocator, a, b);
    }

    /// Create an constant node
    pub fn constant(self: *Graph, value: *Tensor) !*Constant {
        return try Constant.init(self.allocator, value);
    }

    /// Divide two nodes
    pub fn divide(self: *Graph, a: Node, b: Node) !*Divide {
        return try Divide.init(self.allocator, a, b);
    }

    /// Ln of a node
    pub fn ln(self: *Graph, x: Node) !*Ln {
        return try Ln.init(self.allocator, x);
    }

    /// Log of a node
    pub fn log(self: *Graph, x: Node) !*Log {
        return try Log.init(self.allocator, x);
    }

    /// Multiply two nodes
    pub fn multiply(self: *Graph, a: Node, b: Node) !*Multiply {
        return try Multiply.init(self.allocator, a, b);
    }

    /// Power two nodes
    pub fn power(self: *Graph, a: Node, b: Node) !*Power {
        return try Power.init(self.allocator, a, b);
    }

    /// Sine of a node
    pub fn sin(self: *Graph, x: Node) !*Sin {
        return try Sin.init(self.allocator, x);
    }

    /// Subtract two nodes
    pub fn subtract(self: *Graph, a: Node, b: Node) !*Subtract {
        return try Subtract.init(self.allocator, a, b);
    }

    /// Create an tensor
    pub fn tensor(self: *Graph, shape: []const usize) !*Tensor {
        return try Tensor.init(self.allocator, shape);
    }

    /// Create an input variable node
    pub fn input(self: *Graph, name: []const u8, value: *Tensor) !*Variable {
        return try Variable.init(self.allocator, name, value);
    }
};
