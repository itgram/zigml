const std = @import("std");

pub const Add = @import("add.zig").Add;
pub const Constant = @import("constant.zig").Constant;
pub const Cos = @import("cos.zig").Cos;
pub const Divide = @import("divide.zig").Divide;
pub const Elu = @import("elu.zig").Elu;
pub const Exp = @import("exp.zig").Exp;
pub const LeakyReLU = @import("leaky_relu.zig").LeakyReLU;
pub const Linear = @import("linear.zig").Linear;
pub const Ln = @import("ln.zig").Ln;
pub const Log = @import("log.zig").Log;
pub const Multiply = @import("multiply.zig").Multiply;
pub const Node = @import("node.zig").Node;
pub const Power = @import("power.zig").Power;
pub const PRelu = @import("prelu.zig").PRelu;
pub const Relu = @import("relu.zig").Relu;
pub const Selu = @import("selu.zig").Selu;
pub const Sigmoid = @import("sigmoid.zig").Sigmoid;
pub const Sin = @import("sin.zig").Sin;
pub const Step = @import("step.zig").Step;
pub const Subtract = @import("subtract.zig").Subtract;
pub const Tan = @import("tan.zig").Tan;
pub const Tanh = @import("tanh.zig").Tanh;
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

    /// Create a cos node
    pub fn cos(self: *Graph, x: Node) !*Cos {
        return try Cos.init(self.allocator, x);
    }

    /// Divide two nodes
    pub fn divide(self: *Graph, a: Node, b: Node) !*Divide {
        return try Divide.init(self.allocator, a, b);
    }

    /// Create an elu node
    pub fn elu(self: *Graph, x: Node, alpha: f64) !*Elu {
        return try Elu.init(self.allocator, x, alpha);
    }

    /// Exponential of a node
    pub fn exp(self: *Graph, x: Node) !*Exp {
        return try Exp.init(self.allocator, x);
    }

    /// Create a leaky relu node
    pub fn leakyReLU(self: *Graph, x: Node, alpha: f64) !*LeakyReLU {
        return try LeakyReLU.init(self.allocator, x, alpha);
    }

    /// Create a linear node
    pub fn linear(self: *Graph, x: Node) !*Linear {
        return try Linear.init(self.allocator, x);
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

    /// Create a parametric relu node
    pub fn prelu(self: *Graph, x: Node, alpha: *Tensor) !*PRelu {
        return try PRelu.init(self.allocator, x, alpha);
    }

    /// Create a relu node
    pub fn relu(self: *Graph, x: Node) !*Relu {
        return try Relu.init(self.allocator, x);
    }

    /// Create a selu node
    pub fn selu(self: *Graph, x: Node, alpha: f64, lambda: f64) !*Selu {
        return try Selu.init(self.allocator, x, alpha, lambda);
    }

    /// Sigmoid of a node
    pub fn sigmoid(self: *Graph, x: Node) !*Sigmoid {
        return try Sigmoid.init(self.allocator, x);
    }

    /// Sin of a node
    pub fn sin(self: *Graph, x: Node) !*Sin {
        return try Sin.init(self.allocator, x);
    }

    /// Step function of a node
    pub fn step(self: *Graph, x: Node, threshold: f64) !*Step {
        return try Step.init(self.allocator, x, threshold);
    }

    /// Subtract two nodes
    pub fn subtract(self: *Graph, a: Node, b: Node) !*Subtract {
        return try Subtract.init(self.allocator, a, b);
    }

    /// Create an tensor
    pub fn tensor(self: *Graph, shape: []const usize) !*Tensor {
        return try Tensor.init(self.allocator, shape);
    }

    /// Tangent of a node
    pub fn tan(self: *Graph, x: Node) !*Tan {
        return try Tan.init(self.allocator, x);
    }

    /// Hyperbolic tangent of a node
    pub fn tanh(self: *Graph, x: Node) !*Tanh {
        return try Tanh.init(self.allocator, x);
    }

    /// Create an input variable node
    pub fn input(self: *Graph, name: []const u8, value: *Tensor) !*Variable {
        return try Variable.init(self.allocator, name, value);
    }
};
