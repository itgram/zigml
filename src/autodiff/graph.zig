const std = @import("std");

pub const Add = @import("add.zig").Add;
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
pub const Multiply = @import("multiply.zig").Multiply;
pub const Node = @import("node.zig").Node;
pub const Power = @import("power.zig").Power;
pub const PReLU = @import("prelu.zig").PReLU;
pub const ReLU = @import("relu.zig").ReLU;
pub const SELU = @import("selu.zig").SELU;
pub const Sigmoid = @import("sigmoid.zig").Sigmoid;
pub const Sin = @import("sin.zig").Sin;
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
