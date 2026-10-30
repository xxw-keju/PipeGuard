"""CPU-only TIR fixtures: multidimensional ownership, reduction workspace reuse, explicit-width memory and CTA votes."""
import pathlib,tilelang
from tilelang import tvm
T=tvm.tirx
out=pathlib.Path(__file__).resolve().parent
def save(name,body,buf,threads):
 f=T.PrimFunc([buf.data],body,buffer_map={buf.data:buf}).with_attr('target',tvm.target.Target({'kind':'cuda','arch':'sm_90','thread_warp_size':32})).with_attr('tirx.noalias',True).with_attr('thread_extent',threads).with_attr('global_symbol','main_kernel')
 (out/(name+'.tvm.json')).write_text(tvm.ir.save_json(tvm.IRModule({'main_kernel':f})))
def bind(tag,size,body,var):return T.AttrStmt(T.IterVar(tvm.ir.Range(0,size),var,T.IterVar.ThreadIndex,tag),'thread_extent',size,body)
for unsafe in [False,True]:
 A=T.decl_buffer((32,),'int32',name='A');x=T.Var('tx','int32');y=T.Var('ty','int32')
 body=T.BufferStore(A,T.IntImm('int32',1),[x if unsafe else x+8*y]);body=bind('threadIdx.y',4,body,y);body=bind('threadIdx.x',8,body,x)
 save('tilelang_multidim_'+('race' if unsafe else 'safe'),body,A,{'threadIdx.x':8,'threadIdx.y':4})
for sync in [False,True]:
 A=T.decl_buffer((64,),'float32',name='A');ws=T.decl_buffer((64,),'float32',name='workspace',scope='shared');v=T.decl_buffer((1,),'float32',name='v',scope='local');tx=T.Var('tx','int32')
 call=T.call_extern('handle','tl::AllReduce<tl::SumOp, 64, 1, 0, tl::NamedBarrier<64>, 1, 64>::run_batch',v.data,ws.access_ptr('rw'))
 stmts=[T.AllocBuffer(ws),T.AllocBuffer(v),T.BufferStore(v,T.FloatImm('float32',1),[0]),T.Evaluate(call)]
 if sync:stmts.append(T.Evaluate(T.call_intrin('int32','tirx.tvm_storage_sync','shared')))
 stmts.extend([T.BufferStore(ws,T.FloatImm('float32',0),[tx]),T.BufferStore(A,T.BufferLoad(v,[0]),[tx])]);body=bind('threadIdx.x',64,T.SeqStmt(stmts),tx)
 save('tilelang_batch_reduce_'+('safe' if sync else 'reuse_race'),body,A,{'threadIdx.x':64})
for unsafe in [False,True]:
 A=T.decl_buffer((128,),'float32',name='A');tx=T.Var('tx','int32')
 load=T.Call('uint32x4',tvm.ir.Op.get('tl.ldg128'),[A.access_ptr('r',offset=tx*4)])
 store=T.Call('handle',tvm.ir.Op.get('tl.stg128'),[A.access_ptr('w',offset=tx if unsafe else tx*4),load])
 save('tilelang_explicit128_'+('race' if unsafe else 'safe'),bind('threadIdx.x',32,T.Evaluate(store),tx),A,{'threadIdx.x':32})
for divergent in [False,True]:
 A=T.decl_buffer((64,),'int32',name='A');S=T.decl_buffer((64,),'int32',name='S',scope='shared');tx=T.Var('tx','int32');vote=T.Call('int32',tvm.ir.Op.get('tl.syncthreads_count'),[tx<32])
 body=T.BufferStore(A,vote,[tx])
 if divergent:body=T.IfThenElse(tx<32,body,None)
 save('tilelang_cta_vote_'+('divergent' if divergent else 'safe'),bind('threadIdx.x',64,body,tx),A,{'threadIdx.x':64})
