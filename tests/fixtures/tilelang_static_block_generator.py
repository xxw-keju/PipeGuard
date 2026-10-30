from pathlib import Path

import tilelang
from tilelang import tvm
from tvm import tirx

global_i32_pointer = tvm.ir.PointerType(tvm.ir.PrimType("int32"), "global")
source_handle = tirx.Var("source", global_i32_pointer)
destination_handle = tirx.Var("destination", global_i32_pointer)
source = tirx.decl_buffer(
    (4, 4),
    "int32",
    name="source",
    data=source_handle,
    strides=(8, 2),
    elem_offset=1,
    scope="global",
)
destination = tirx.decl_buffer(
    (4, 4),
    "int32",
    name="destination",
    data=destination_handle,
    strides=(8, 2),
    elem_offset=1,
    scope="global",
)
block_var = tirx.Var("block_index", "int32")
block_iter = tirx.IterVar(tvm.ir.Range(0, 4), block_var, tirx.IterVar.DataPar)
load = tirx.BufferLoad(source, [block_var, tirx.IntImm("int32", 2)])
store = tirx.BufferStore(
    destination,
    load + tirx.IntImm("int32", 1),
    [block_var, tirx.IntImm("int32", 2)],
)
block = tirx.SBlock(
    [block_iter],
    [],
    [],
    "static_block",
    store,
)
realize = tirx.SBlockRealize([tirx.IntImm("int32", 1)], True, block)
false_store = tirx.BufferStore(
    destination,
    tirx.IntImm("int32", 0),
    [tirx.IntImm("int32", 99), tirx.IntImm("int32", 2)],
)
false_block = tirx.SBlock([], [], [], "statically_false_block", false_store)
false_realize = tirx.SBlockRealize([], False, false_block)
reduction_var = tirx.Var("reduction_index", "int32")
reduction_iter = tirx.IterVar(
    tvm.ir.Range(0, 2),
    reduction_var,
    tirx.IterVar.CommReduce,
)
reduction_init = tirx.BufferStore(
    destination,
    tirx.IntImm("int32", 0),
    [tirx.IntImm("int32", 0), tirx.IntImm("int32", 0)],
)
reduction_body = tirx.BufferStore(
    destination,
    tirx.IntImm("int32", 1),
    [reduction_var, tirx.IntImm("int32", 1)],
)
reduction_block = tirx.SBlock(
    [reduction_iter],
    [],
    [],
    "static_reduction",
    reduction_body,
    init=reduction_init,
)
reduction_zero = tirx.SBlockRealize([tirx.IntImm("int32", 0)], True, reduction_block)
reduction_one = tirx.SBlockRealize([tirx.IntImm("int32", 1)], True, reduction_block)
thread_var = tirx.Var("thread_index", "int32")
thread_iter = tirx.IterVar(
    tvm.ir.Range(0, 1),
    thread_var,
    tirx.IterVar.ThreadIndex,
    "threadIdx.x",
)
body = tirx.AttrStmt(
    thread_iter,
    "thread_extent",
    tirx.IntImm("int32", 1),
    tirx.SeqStmt([realize, false_realize, reduction_zero, reduction_one]),
)
function = tirx.PrimFunc(
    [source_handle, destination_handle],
    body,
    buffer_map={source_handle: source, destination_handle: destination},
)
function = function.with_attr("target", tvm.target.Target({
    "kind": "cuda",
    "arch": "sm_90a",
    "thread_warp_size": 32,
    "max_num_threads": 1024,
}))
function = function.with_attr("thread_extent", {"threadIdx.x": 1})
function = function.with_attr("tirx.noalias", True)
module = tvm.IRModule({"main": function})
Path("tilelang_static_block_device.tvm.json").write_text(tvm.ir.save_json(module))
blackwell = function.with_attr("target", tvm.target.Target({
    "kind": "cuda",
    "arch": "sm_100",
    "thread_warp_size": 32,
    "max_num_threads": 1024,
}))
Path("tilelang_blackwell_device.tvm.json").write_text(
    tvm.ir.save_json(tvm.IRModule({"main": blackwell}))
)
amd = function.with_attr("target", tvm.target.Target({
    "kind": "rocm",
    "mcpu": "gfx942",
    "thread_warp_size": 64,
    "max_num_threads": 1024,
}))
Path("tilelang_amd_device.tvm.json").write_text(
    tvm.ir.save_json(tvm.IRModule({"main": amd}))
)
alias_unresolved = function.with_attr("tirx.noalias", False)
Path("tilelang_alias_unresolved_device.tvm.json").write_text(
    tvm.ir.save_json(tvm.IRModule({"main": alias_unresolved}))
)
opaque_store = tirx.BufferStore(
    destination,
    tirx.call_intrin("int32", "tirx.ptx_commit_group"),
    [tirx.IntImm("int32", 0), tirx.IntImm("int32", 0)],
)
opaque_body = tirx.AttrStmt(
    thread_iter,
    "thread_extent",
    tirx.IntImm("int32", 1),
    opaque_store,
)
opaque_function = tirx.PrimFunc(
    [source_handle, destination_handle],
    opaque_body,
    buffer_map={source_handle: source, destination_handle: destination},
)
for key, value in dict(function.attrs).items():
    opaque_function = opaque_function.with_attr(key, value)
Path("tilelang_nested_effect_device.tvm.json").write_text(
    tvm.ir.save_json(tvm.IRModule({"main": opaque_function}))
)
counter = tirx.Var("counter", "int32")
while_store = tirx.BufferStore(
    destination,
    tirx.IntImm("int32", 1),
    [counter, tirx.IntImm("int32", 0)],
)
while_loop = tirx.While(
    counter < tirx.IntImm("int32", 3),
    tirx.SeqStmt([
        while_store,
        tirx.Bind(counter, counter + tirx.IntImm("int32", 1)),
    ]),
)
while_body = tirx.AttrStmt(
    thread_iter,
    "thread_extent",
    tirx.IntImm("int32", 1),
    tirx.SeqStmt([
        tirx.Bind(counter, tirx.IntImm("int32", 0)),
        while_loop,
    ]),
)
while_function = tirx.PrimFunc(
    [source_handle, destination_handle],
    while_body,
    buffer_map={source_handle: source, destination_handle: destination},
)
for key, value in dict(function.attrs).items():
    while_function = while_function.with_attr(key, value)
Path("tilelang_static_while_device.tvm.json").write_text(
    tvm.ir.save_json(tvm.IRModule({"main": while_function}))
)
