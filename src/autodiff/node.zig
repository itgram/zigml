const std = @import("std");

pub const Node = struct {
    ptr: *anyopaque,
    vtab: *const VTab, //ptr to vtab

    const VTab = struct {
        evalFn: *const fn (ptr: *anyopaque) f64,
        diffFn: *const fn (ptr: *anyopaque, dval: f64) void,
    };

    // cast concrete implementation types/objs to interface
    pub fn init(obj: anytype) Node {
        const T = @TypeOf(obj);
        const ptrInfo = @typeInfo(T);

        std.debug.assert(ptrInfo == .pointer); // Must be a pointer
        std.debug.assert(ptrInfo.pointer.size == .one); // Must be a single-item pointer
        std.debug.assert(@typeInfo(ptrInfo.pointer.child) == .@"struct"); // Must point to a struct

        const impl = struct {
            fn eval(pointer: *anyopaque) f64 {
                const self: T = @ptrCast(@alignCast(pointer));
                return self.eval();
            }
            fn diff(pointer: *anyopaque, dval: f64) void {
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

    pub fn eval(self: Node) f64 {
        return self.vtab.evalFn(self.ptr);
    }

    pub fn diff(self: Node, dval: f64) void {
        self.vtab.diffFn(self.ptr, dval);
    }
};
