from pathlib import Path
import tilelang
from tilelang import tvm
T=tvm.tirx
out=Path(__file__).resolve().parent
for mutable in [False,True]:
 I=T.decl_buffer((2,),'int32',name='I');A=T.decl_buffer((32,),'int32',name='A');tx=T.Var('tx','int32')
 stmts=[]
 if mutable:stmts.append(T.BufferStore(I,T.IntImm('int32',0),[0]))
 stmts.append(T.IfThenElse(tx<T.BufferLoad(I,[0]),T.BufferStore(A,T.IntImm('int32',1),[tx]),None))
 body=T.AttrStmt(T.IterVar(tvm.ir.Range(0,32),tx,T.IterVar.ThreadIndex,'threadIdx.x'),'thread_extent',32,(T.SeqStmt(stmts) if len(stmts)>1 else stmts[0]))
 f=T.PrimFunc([I.data,A.data],body,buffer_map={I.data:I,A.data:A}).with_attr('global_symbol','kernel').with_attr('target',tvm.target.Target({'kind':'cuda','arch':'sm_90'})).with_attr('tirx.noalias',True).with_attr('thread_extent',{'threadIdx.x':32})
 (out/('tilelang_constant_input_'+('mutable' if mutable else 'readonly')+'.tvm.json')).write_text(tvm.ir.save_json(tvm.IRModule({'kernel':f})))

# Signed negative upper bounds must keep the skipped body's effects absent;
# the otherwise identical positive loop exposes an unguarded write race.
for extent in [-1, 1]:
    A=T.decl_buffer((1,), 'int32', name='A');i=T.Var('i', 'int32');tx=T.Var('tx', 'int32')
    loop=T.For(i, 0, extent, T.ForKind.SERIAL, T.BufferStore(A, 1, [0]))
    body=T.AttrStmt(T.IterVar(tvm.ir.Range(0,32),tx,T.IterVar.ThreadIndex,'threadIdx.x'),'thread_extent',32,loop)
    f=T.PrimFunc([A.data], body, buffer_map={A.data:A}).with_attr('global_symbol','kernel').with_attr('target',tvm.target.Target({'kind':'cuda','arch':'sm_90'})).with_attr('tirx.noalias',True).with_attr('thread_extent',{'threadIdx.x':32})
    (out/('tilelang_signed_loop_'+('negative' if extent<0 else 'positive')+'.tvm.json')).write_text(tvm.ir.save_json(tvm.IRModule({'kernel':f})))
