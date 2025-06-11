const std = @import("std");
const autodiff = @import("autodiff");
const Tensor = autodiff.Tensor;

/// Error types for metrics
const MetricsError = error{
    /// Error when the shape of the tensor is invalid for the metric
    InvalidShape,
    /// Error when the shapes of tensors don't match
    ShapeMismatch,
};

/// Accuracy metrics for neural network evaluation
pub const Metrics = struct {
    /// Calculates binary classification accuracy.
    /// Args:
    ///   predictions: Predicted probabilities of shape [batch_size, 1]
    ///   targets: True labels of shape [batch_size, 1] (0 or 1)
    ///   threshold: Optional threshold for converting probabilities to binary predictions (default: 0.5)
    /// Returns:
    ///   Accuracy as a float between 0 and 1
    pub fn binaryAccuracy(predictions: *Tensor, targets: *Tensor, threshold: f64) MetricsError!f64 {
        if (predictions.shape.len != 2 or targets.shape.len != 2) {
            return error.InvalidShape;
        }
        if (predictions.shape[0] != targets.shape[0] or predictions.shape[1] != 1 or targets.shape[1] != 1) {
            return error.ShapeMismatch;
        }

        var correct: usize = 0;
        for (predictions.data, targets.data) |pred, target| {
            const predicted_class = if (pred > threshold) @as(f64, 1.0) else @as(f64, 0.0);
            if (predicted_class == target) {
                correct += 1;
            }
        }

        return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(predictions.size));
    }

    /// Calculates regression accuracy using a precision threshold derived from standard deviation.
    /// Args:
    ///   predictions: Predicted values of shape [batch_size, 1]
    ///   targets: True values of shape [batch_size, 1]
    ///   threshold_factor: Factor to multiply with standard deviation to get threshold (default: 0.1)
    /// Returns:
    ///   Accuracy as a float between 0 and 1, where predictions within threshold are considered correct
    pub fn regressionAccuracy(predictions: *Tensor, targets: *Tensor, threshold_factor: f64) MetricsError!f64 {
        if (predictions.shape.len != 2 or targets.shape.len != 2) {
            return error.InvalidShape;
        }
        if (predictions.shape[0] != targets.shape[0] or predictions.shape[1] != 1 or targets.shape[1] != 1) {
            return error.ShapeMismatch;
        }

        // Calculate mean of targets
        var mean: f64 = 0;
        for (targets.data) |target| {
            mean += target;
        }
        mean /= @as(f64, @floatFromInt(predictions.size));

        // Calculate standard deviation of targets
        var variance: f64 = 0;
        for (targets.data) |target| {
            const diff = target - mean;
            variance += diff * diff;
        }
        variance /= @as(f64, @floatFromInt(predictions.size - 1)); // Use n-1 for sample standard deviation
        const std_dev = @sqrt(variance);

        // Use threshold_factor * std_dev as the precision threshold
        const threshold = threshold_factor * std_dev;

        // Count predictions within threshold
        var correct: usize = 0;
        for (predictions.data, targets.data) |pred, target| {
            const diff = @abs(pred - target);
            if (diff <= threshold) {
                correct += 1;
            }
        }

        return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(predictions.size));
    }

    /// Calculates categorical classification accuracy.
    /// Args:
    ///   predictions: Predicted probabilities of shape [batch_size, num_classes]
    ///   targets: True labels of shape [batch_size, num_classes] (one-hot encoded)
    /// Returns:
    ///   Accuracy as a float between 0 and 1
    pub fn categoricalAccuracy(predictions: *Tensor, targets: *Tensor) MetricsError!f64 {
        if (predictions.shape.len != 2 or targets.shape.len != 2) {
            return error.InvalidShape;
        }
        if (predictions.shape[0] != targets.shape[0] or predictions.shape[1] != targets.shape[1]) {
            return error.ShapeMismatch;
        }

        var correct: usize = 0;
        const batch_size = predictions.shape[0];
        const num_classes = predictions.shape[1];

        for (0..batch_size) |i| {
            var max_pred_idx: usize = 0;
            var max_pred_val: f64 = predictions.data[i * num_classes];
            var max_target_idx: usize = 0;
            var max_target_val: f64 = targets.data[i * num_classes];

            // Find predicted class
            for (1..num_classes) |j| {
                const pred_val = predictions.data[i * num_classes + j];
                if (pred_val > max_pred_val) {
                    max_pred_val = pred_val;
                    max_pred_idx = j;
                }
            }

            // Find true class
            for (1..num_classes) |j| {
                const target_val = targets.data[i * num_classes + j];
                if (target_val > max_target_val) {
                    max_target_val = target_val;
                    max_target_idx = j;
                }
            }

            if (max_pred_idx == max_target_idx) {
                correct += 1;
            }
        }

        return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(batch_size));
    }
};

test "binary accuracy" {
    const allocator = std.testing.allocator;

    // Test case 1: Perfect predictions
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer targets.deinit();

        // Set the data after initialization
        predictions.data[0] = 0.9;
        predictions.data[1] = 0.1;
        predictions.data[2] = 0.8;
        predictions.data[3] = 0.2;

        targets.data[0] = 1.0;
        targets.data[1] = 0.0;
        targets.data[2] = 1.0;
        targets.data[3] = 0.0;

        const accuracy = try Metrics.binaryAccuracy(predictions, targets, 0.5);
        try std.testing.expectEqual(@as(f64, 1.0), accuracy);
    }

    // Test case 2: Some incorrect predictions
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer targets.deinit();

        // Set the data after initialization
        predictions.data[0] = 0.9;
        predictions.data[1] = 0.1;
        predictions.data[2] = 0.3;
        predictions.data[3] = 0.8;

        targets.data[0] = 1.0;
        targets.data[1] = 0.0;
        targets.data[2] = 1.0;
        targets.data[3] = 0.0;

        const accuracy = try Metrics.binaryAccuracy(predictions, targets, 0.5);
        try std.testing.expectEqual(@as(f64, 0.5), accuracy);
    }

    // Test case 3: Invalid shape
    {
        const predictions = try Tensor.init(allocator, &[_]usize{2});
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{2});
        defer targets.deinit();

        try std.testing.expectError(error.InvalidShape, Metrics.binaryAccuracy(predictions, targets, 0.5));
    }

    // Test case 4: Shape mismatch
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 2, 1 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 1, 1 });
        defer targets.deinit();

        try std.testing.expectError(error.ShapeMismatch, Metrics.binaryAccuracy(predictions, targets, 0.5));
    }
}

test "regression accuracy" {
    const allocator = std.testing.allocator;

    // Test case 1: Perfect predictions
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer targets.deinit();

        // Set the data after initialization
        predictions.data[0] = 1.0;
        predictions.data[1] = 2.0;
        predictions.data[2] = 3.0;
        predictions.data[3] = 4.0;

        targets.data[0] = 1.0;
        targets.data[1] = 2.0;
        targets.data[2] = 3.0;
        targets.data[3] = 4.0;

        const accuracy = try Metrics.regressionAccuracy(predictions, targets, 0.1);
        try std.testing.expectEqual(@as(f64, 1.0), accuracy);
    }

    // Test case 2: Some predictions within threshold
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer targets.deinit();

        // Set the data after initialization
        predictions.data[0] = 1.0;
        predictions.data[1] = 2.05; // Small difference
        predictions.data[2] = 3.0;
        predictions.data[3] = 4.05; // Small difference

        targets.data[0] = 1.0;
        targets.data[1] = 2.0;
        targets.data[2] = 3.0;
        targets.data[3] = 4.0;

        const accuracy = try Metrics.regressionAccuracy(predictions, targets, 0.1);
        try std.testing.expectEqual(@as(f64, 1.0), accuracy);
    }

    // Test case 3: Some predictions outside threshold
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer targets.deinit();

        // Set the data after initialization
        predictions.data[0] = 1.0;
        predictions.data[1] = 5.0; // Much larger difference
        predictions.data[2] = 3.0;
        predictions.data[3] = 4.0;

        targets.data[0] = 1.0;
        targets.data[1] = 2.0;
        targets.data[2] = 3.0;
        targets.data[3] = 4.0;

        const accuracy = try Metrics.regressionAccuracy(predictions, targets, 0.1);
        try std.testing.expectEqual(@as(f64, 0.75), accuracy);
    }

    // Test case 4: Different threshold factor
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 4, 1 });
        defer targets.deinit();

        // Set the data after initialization
        predictions.data[0] = 1.0;
        predictions.data[1] = 2.5; // Outside threshold with 0.1, but within with 0.2
        predictions.data[2] = 3.0;
        predictions.data[3] = 4.0;

        targets.data[0] = 1.0;
        targets.data[1] = 2.0;
        targets.data[2] = 3.0;
        targets.data[3] = 4.0;

        const accuracy = try Metrics.regressionAccuracy(predictions, targets, 0.2);
        try std.testing.expectEqual(@as(f64, 0.75), accuracy);
    }

    // Test case 5: Invalid shape
    {
        const predictions = try Tensor.init(allocator, &[_]usize{2});
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{2});
        defer targets.deinit();

        try std.testing.expectError(error.InvalidShape, Metrics.regressionAccuracy(predictions, targets, 0.1));
    }

    // Test case 6: Shape mismatch
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 1, 2 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 1, 3 });
        defer targets.deinit();

        try std.testing.expectError(error.ShapeMismatch, Metrics.regressionAccuracy(predictions, targets, 0.1));
    }
}

test "categorical accuracy" {
    const allocator = std.testing.allocator;

    // Test case 1: Perfect predictions (3 classes)
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 3, 3 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 3, 3 });
        defer targets.deinit();

        // Set the data after initialization
        // Class 0
        predictions.data[0] = 0.9;
        predictions.data[1] = 0.1;
        predictions.data[2] = 0.1;

        // Class 1
        predictions.data[3] = 0.1;
        predictions.data[4] = 0.8;
        predictions.data[5] = 0.1;

        // Class 2
        predictions.data[6] = 0.1;
        predictions.data[7] = 0.1;
        predictions.data[8] = 0.9;

        // Targets
        // Class 0
        targets.data[0] = 1.0;
        targets.data[1] = 0.0;
        targets.data[2] = 0.0;

        // Class 1
        targets.data[3] = 0.0;
        targets.data[4] = 1.0;
        targets.data[5] = 0.0;

        // Class 2
        targets.data[6] = 0.0;
        targets.data[7] = 0.0;
        targets.data[8] = 1.0;

        const accuracy = try Metrics.categoricalAccuracy(predictions, targets);
        try std.testing.expectEqual(@as(f64, 1.0), accuracy);
    }

    // Test case 2: Some incorrect predictions
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 3, 3 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 3, 3 });
        defer targets.deinit();

        // Set the data after initialization
        // Class 0
        predictions.data[0] = 0.9;
        predictions.data[1] = 0.1;
        predictions.data[2] = 0.1;

        // Class 1
        predictions.data[3] = 0.1;
        predictions.data[4] = 0.8;
        predictions.data[5] = 0.1;

        // Class 2 (wrong prediction)
        predictions.data[6] = 0.8;
        predictions.data[7] = 0.1;
        predictions.data[8] = 0.1;

        // Targets
        // Class 0
        targets.data[0] = 1.0;
        targets.data[1] = 0.0;
        targets.data[2] = 0.0;

        // Class 1
        targets.data[3] = 0.0;
        targets.data[4] = 1.0;
        targets.data[5] = 0.0;

        // Class 2
        targets.data[6] = 0.0;
        targets.data[7] = 0.0;
        targets.data[8] = 1.0;

        const accuracy = try Metrics.categoricalAccuracy(predictions, targets);
        try std.testing.expectEqual(@as(f64, 2.0 / 3.0), accuracy);
    }

    // Test case 3: Invalid shape
    {
        const predictions = try Tensor.init(allocator, &[_]usize{2});
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{2});
        defer targets.deinit();

        try std.testing.expectError(error.InvalidShape, Metrics.categoricalAccuracy(predictions, targets));
    }

    // Test case 4: Shape mismatch
    {
        const predictions = try Tensor.init(allocator, &[_]usize{ 1, 2 });
        defer predictions.deinit();
        const targets = try Tensor.init(allocator, &[_]usize{ 1, 3 });
        defer targets.deinit();

        try std.testing.expectError(error.ShapeMismatch, Metrics.categoricalAccuracy(predictions, targets));
    }
}
