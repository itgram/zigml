const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// Node interface for autodiff.
/// The Node interface defines the structure for nodes in the computation graph.
/// Each node must implement the `eval`, `diff`, and `reset` methods.
pub const Node = struct {
    ptr: *anyopaque,
    vtab: *const VTab, //ptr to vtab

    const VTab = struct {
        evalFn: *const fn (ptr: *anyopaque) *Tensor,
        diffFn: *const fn (ptr: *anyopaque, dval: *Tensor) void,
        resetFn: *const fn (ptr: *anyopaque) void,
    };

    // cast concrete implementation types/objs to interface
    pub fn init(pointer: anytype) Node {
        const T = @TypeOf(pointer);
        const ptrInfo = @typeInfo(T);

        std.debug.assert(ptrInfo == .pointer); // Must be a pointer
        std.debug.assert(ptrInfo.pointer.size == .one); // Must be a single-item pointer
        std.debug.assert(@typeInfo(ptrInfo.pointer.child) == .@"struct"); // Must point to a struct

        const impl = struct {
            fn eval(ptr: *anyopaque) *Tensor {
                const self: T = @ptrCast(@alignCast(ptr));
                return self.eval();
            }
            fn diff(ptr: *anyopaque, dval: *Tensor) void {
                const self: T = @ptrCast(@alignCast(ptr));
                self.diff(dval);
            }
            fn reset(ptr: *anyopaque) void {
                const self: T = @ptrCast(@alignCast(ptr));
                self.reset();
            }
        };

        return .{
            .ptr = pointer,
            .vtab = &.{
                .evalFn = impl.eval,
                .diffFn = impl.diff,
                .resetFn = impl.reset,
            },
        };
    }

    pub fn eval(self: Node) *Tensor {
        return self.vtab.evalFn(self.ptr);
    }

    pub fn diff(self: Node, dval: *Tensor) void {
        self.vtab.diffFn(self.ptr, dval);
    }

    /// Resets the node's state by clearing cached values.
    /// This is useful when you want to recompute values in the computation graph.
    pub fn reset(self: Node) void {
        self.vtab.resetFn(self.ptr);
    }
};
