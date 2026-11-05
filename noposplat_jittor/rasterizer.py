"""Native Jittor custom gradient around the author's pose-aware CUDA kernels.

No PyTorch backend. Linux .so built by scripts/build_rasterizer.sh is required.
Call release_states() AFTER evaluating/backpropagating every render group.
"""
from pathlib import Path
import ctypes
import os
import numpy as np
import jittor as jt

ROOT=Path(__file__).resolve().parents[1]
LIB=Path(os.environ.get('NPS_RASTER_LIB',ROOT/'native/build/libnps_raster.so')).resolve()
_library=None


def ensure_native():
    global _library
    if not jt.flags.use_cuda:raise RuntimeError('Native rasterizer requires jt.flags.use_cuda=1')
    if not LIB.exists():raise RuntimeError(f'Build native rasterizer first: {LIB}')
    if _library is None:
        _library=ctypes.CDLL(str(LIB))
        _library.nps_abi_version.restype=ctypes.c_int
        _library.nps_release_all.restype=None
        if _library.nps_abi_version()!=1:raise RuntimeError('Raster ABI mismatch')


def options(outputs):
    # Jittor's compilation-cache parser treats bare path arguments as input
    # files. Bind the runtime directory to -rpath=, as its own compiler does.
    flags=f'FLAGS: -I"{ROOT / "native"}" -L"{LIB.parent}" -lnps_raster -Xlinker -rpath="{LIB.parent}"'
    for x in outputs:x.compile_options={flags:1}
    return outputs


HEADER='#include "bridge.h"\n#include <cuda_runtime.h>\n#include <stdexcept>\n'


class Rasterize(jt.Function):
    def __init__(self,height,width,tanx,tany,degree):
        super().__init__();self.h=height;self.w=width;self.tx=tanx;self.ty=tany;self.degree=degree
    def execute(self,means,cov,sh,opacity,view,proj,rawproj,cam,bg,theta,rho):
        ensure_native()
        if means.dtype!='float32' or cov.dtype!='float32':raise TypeError('Rasterizer expects FP32')
        self.fov=jt.array(np.asarray([self.tx,self.ty],dtype=np.float32)).stop_grad()
        self.inputs=(means,cov,sh,opacity,view,proj,rawproj,cam,bg,theta,rho,self.fov)
        self.p,self.m=means.shape[0],sh.shape[1]
        if self.p<1:raise ValueError('Empty Gaussian set is not supported by this bridge')
        if tuple(cov.shape)!=(self.p,6) or tuple(sh.shape)!=(self.p,self.m,3):raise ValueError('Packed covariance/SH layout mismatch')
        code=f'''
float fov[2];
cudaError_t copied=cudaMemcpy(fov,in11_p,sizeof(fov),cudaMemcpyDeviceToHost);
if(copied!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(copied));
int error=nps_forward({self.p},{self.m},{self.degree},{self.h},{self.w},fov[0],fov[1],
 in0_p,in1_p,in2_p,in3_p,in4_p,in5_p,in7_p,in8_p,out0_p,out1_p,out2_p,reinterpret_cast<std::int64_t*>(out3_p));
if(error) throw std::runtime_error(nps_last_error());
'''
        # theta/rho carry the analytical camera Jacobian; zero deltas do not
        # modify the forward, exactly as in the original extension.
        outs=jt.code([(3,self.h,self.w),(1,self.h,self.w),(self.p,),(1,)],['float32','float32','int32','int64'],list(self.inputs),cuda_header=HEADER,cuda_src=code)
        self.color,self.depth,self.radii,self.handle=options(outs)
        return self.color,self.depth
    def grad(self,gcolor,gdepth):
        gcolor=jt.zeros_like(self.color) if gcolor is None else gcolor
        gdepth=jt.zeros_like(self.depth) if gdepth is None else gdepth
        inp=list(self.inputs[:9])+[self.radii,self.handle,gcolor,gdepth,self.fov]
        code=f'''
std::int64_t id; cudaMemcpy(&id,in10_p,sizeof(id),cudaMemcpyDeviceToHost);
float fov[2];
cudaError_t copied=cudaMemcpy(fov,in13_p,sizeof(fov),cudaMemcpyDeviceToHost);
if(copied!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(copied));
int error=nps_backward(id,{self.p},{self.m},{self.degree},{self.h},{self.w},fov[0],fov[1],
 in0_p,in1_p,in2_p,in4_p,in5_p,in6_p,in7_p,in8_p,in9_p,in11_p,in12_p,out0_p,out1_p,out2_p,out3_p,out4_p);
if(error) throw std::runtime_error(nps_last_error());
'''
        outs=jt.code([(self.p,3),(self.p,6),(self.p,self.m,3),(self.p,1),(self.p,6)],['float32']*5,inp,cuda_header=HEADER,cuda_src=code)
        gm,gcov,gsh,gopacity,gtau=options(outs)
        tau=gtau.sum(0)
        return gm,gcov,gsh,gopacity,None,None,None,None,None,tau[3:],tau[:3]


def release_states():
    if _library is not None:
        jt.sync_all()
        _library.nps_release_all()

