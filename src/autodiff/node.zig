const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

pub const Node = struct {
    ptr: *anyopaque,
    vtab: *const VTab, //ptr to vtab

    const VTab = struct {
        evalFn: *const fn (ptr: *anyopaque) *Tensor,
        diffFn: *const fn (ptr: *anyopaque, dval: *Tensor) void,
    };

    // cast concrete implementation types/objs to interface
    pub fn init(obj: anytype) Node {
        const T = @TypeOf(obj);
        const ptrInfo = @typeInfo(T);

        std.debug.assert(ptrInfo == .pointer); // Must be a pointer
        std.debug.assert(ptrInfo.pointer.size == .one); // Must be a single-item pointer
        std.debug.assert(@typeInfo(ptrInfo.pointer.child) == .@"struct"); // Must point to a struct

        const impl = struct {
            fn eval(pointer: *anyopaque) *Tensor {
                const self: T = @ptrCast(@alignCast(pointer));
                return self.eval();
            }
            fn diff(pointer: *anyopaque, dval: *Tensor) void {
                const self: T = @ptrCast(@alignCast(pointer));
                self.diff(dval);
            }
        };

        return .{
            .ptr = obj,
            .vtab = &.{
                .evalFn = impl.eval,
                .diffFn = impl.diff,
            },
        };
    }

    pub fn eval(self: Node) *Tensor {
        return self.vtab.evalFn(self.ptr);
    }

    pub fn diff(self: Node, dval: *Tensor) void {
        self.vtab.diffFn(self.ptr, dval);
    }
};
