const std = @import("std");
const autodiff = @import("autodiff");
const Add = autodiff.Add;
const Broadcast = autodiff.Broadcast;
const MatMul = autodiff.MatMul;
const Node = autodiff.Node;
const ReLU = autodiff.ReLU;
const Tensor = autodiff.Tensor;
const Variable = autodiff.Variable;

pub const DenseLayerError = error{
    OutOfMemory,
    InvalidShape,
    NoForwardPass,
};

/// Weight initialization types for neural network layers
pub const WeightInit = enum {
    /// Xavier/Glorot initialization: scale = sqrt(2.0 / (fan_in + fan_out)), best for tanh/linear activations
    xavier,
    /// He initialization: scale = sqrt(2.0 / fan_in), best for ReLU activations
    he,
    /// LeCun initialization: scale = sqrt(1.0 / fan_in), best for sigmoid, tanh activations
    lecun,
    /// Standard normal distribution (mean=0, std=1)
    normal,
};

/// A dense (fully connected) layer in a neural network
pub const DenseLayer = struct {
    allocator: std.mem.Allocator,
    weight: *Variable,
    bias: *Variable,
    matmul: ?*MatMul,
    broadcast: ?*Broadcast,
    add: ?*Add,
    relu: ?*ReLU,

    /// Initialize a new dense layer with random weights
    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, init_type: WeightInit) !*DenseLayer {
        // Scale the weights based on initialization type
        const scale = switch (init_type) {
            .xavier => std.math.sqrt(2.0 / @as(f64, @floatFromInt(input_size + output_size))),
            .he => std.math.sqrt(2.0 / @as(f64, @floatFromInt(input_size))),
            .lecun => std.math.sqrt(1.0 / @as(f64, @floatFromInt(input_size))),
            .normal => 1.0,
        };

        // Initialize weights
        var w_tensor = try Tensor.init(allocator, &[_]usize{ output_size, input_size });
        errdefer w_tensor.deinit();
        w_tensor.randn(scale);

        // Initialize bias
        var b_tensor = try Tensor.init(allocator, &[_]usize{output_size});
        errdefer b_tensor.deinit();
        b_tensor.zeros();

        // Initialize weight variable
        const weight = try Variable.init(allocator, "weight", w_tensor);
        errdefer weight.deinit();

        // Initialize bias variable
        const bias = try Variable.init(allocator, "bias", b_tensor);
        errdefer bias.deinit();

        // Create the layer struct last
        const self = try allocator.create(DenseLayer);
        self.* = DenseLayer{
            .allocator = allocator,
            .weight = weight,
            .bias = bias,
            .matmul = null,
            .broadcast = null,
            .add = null,
            .relu = null,
        };
        return self;
    }

    /// Free the layer's resources
    pub fn deinit(self: *DenseLayer) void {
        // Deinitialize operations created during forward pass
        if (self.relu) |relu| relu.deinit();
        if (self.add) |add| add.deinit();
        if (self.broadcast) |broadcast| broadcast.deinit();
        if (self.matmul) |matmul| matmul.deinit();

        // Deinitialize the tensors owned by the variables
        self.weight.value.deinit();
        self.bias.value.deinit();

        // Deinitialize the variables
        self.weight.deinit();
        self.bias.deinit();

        // Free the layer struct
        self.allocator.destroy(self);
    }

    /// Forward pass through the dense layer
    /// Performs: ReLU(input @ weight + broadcast(bias))
    pub fn forward(self: *DenseLayer, input: Node) !Node {
        // Create new matmul operation for each forward pass since input changes
        const matmul = try MatMul.init(self.allocator, input, self.weight.node());
        errdefer matmul.deinit();

        // Create new broadcast operation
        const broadcast = try Broadcast.init(self.allocator, self.bias.node(), matmul.node());
        errdefer broadcast.deinit();

        // Create new add operation
        const add = try Add.init(self.allocator, matmul.node(), broadcast.node());
        errdefer add.deinit();

        // Create new relu operation
        const relu = try ReLU.init(self.allocator, add.node());
        errdefer relu.deinit();

        if (self.matmul) |old_matmul| old_matmul.deinit();
        if (self.broadcast) |old_broadcast| old_broadcast.deinit();
        if (self.add) |old_add| old_add.deinit();
        if (self.relu) |old_relu| old_relu.deinit();

        self.matmul = matmul;
        self.broadcast = broadcast;
        self.add = add;
        self.relu = relu;

        return relu.node();
    }

    /// Backward pass through the dense layer
    /// Computes gradients for weights, bias, and input
    pub fn backward(self: *DenseLayer, grad: *Tensor) !void {
        // Use the operations created during forward pass
        if (self.relu == null or self.add == null or self.broadcast == null or self.matmul == null) {
            return error.NoForwardPass;
        }

        try self.relu.?.diff(grad);
    }

    /// Get the list of trainable parameters
    pub fn parameters(self: *DenseLayer) []const *Variable {
        return &[_]*Variable{ self.weight, self.bias };
    }
};
