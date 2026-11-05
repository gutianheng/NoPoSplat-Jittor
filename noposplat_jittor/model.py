# Copyright (C) 2022/2024-present Naver Corporation. All rights reserved.
# CroCo/DUSt3R/DPT-derived parts: CC BY-NC-SA 4.0 (non-commercial use only).
# NoPoSplat organization: Copyright (c) 2024 Botao Ye, original MIT license retained.
"""Native Jittor implementation of NoPoSplat's DEFAULT two-view architecture.

Derived from the supplied NoPoSplat/CroCo/DUSt3R source; see THIRD_PARTY.md.
No torch import, NumPy detour, or PyTorch execution in model forward/backward.
"""
from dataclasses import dataclass
import math
import numpy as np
import jittor as jt
from jittor import nn
from .spherical_harmonics import rsh_cart_2,rsh_cart_4,rsh_cart_8


class LayerList(nn.Module):
    """Numerically named children, preserving official checkpoint path components."""
    def __init__(self, layers):
        super().__init__()
        self.length = len(layers)
        for i, layer in enumerate(layers):setattr(self,str(i),layer)
    def __iter__(self):
        return (getattr(self,str(i)) for i in range(self.length))
    def __getitem__(self, index):return getattr(self,str(index))
    def execute(self,x):
        for layer in self:x=layer(x)
        return x


class LayerNorm(nn.Module):
    """Centered variance rather than E[x²]-E[x]²; PyTorch epsilon convention."""
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps=eps
        self.weight=jt.ones((width,))
        self.bias=jt.zeros((width,))
    def execute(self,x):
        centered=x-x.mean(-1,keepdims=True)
        return centered / jt.sqrt((centered*centered).mean(-1,keepdims=True)+self.eps)*self.weight+self.bias


class RoPE2D(nn.Module):
    def __init__(self,base=100.):super().__init__();self.base=base
    def execute(self,tokens,positions):
        dim=tokens.shape[-1]//2
        if tokens.shape[-1]%4:raise ValueError("RoPE2D head width must be divisible by four")
        inv=self.base ** (-jt.arange(0,dim,2).float32()/dim)
        result=[]
        for axis in range(2):
            x=tokens[...,axis*dim:(axis+1)*dim]
            f=positions[:,:,axis].float32().unsqueeze(-1)*inv.reshape(1,1,-1)
            f=jt.concat((f,f),dim=-1).unsqueeze(1)
            rotated=jt.concat((-x[...,dim//2:],x[...,:dim//2]),dim=-1)
            result.append(x*jt.cos(f)+rotated*jt.sin(f))
        return jt.concat(result,dim=-1)


class Mlp(nn.Module):
    def __init__(self,dim,ratio):
        super().__init__();self.fc1=nn.Linear(dim,int(dim*ratio));self.fc2=nn.Linear(int(dim*ratio),dim)
    def execute(self,x):
        x=self.fc1(x)
        return self.fc2(.5*x*(1+jt.erf(x/math.sqrt(2))))


class Attention(nn.Module):
    def __init__(self,dim,heads,rope):
        super().__init__();self.heads=heads;self.scale=(dim//heads)**-.5
        self.qkv=nn.Linear(dim,3*dim);self.proj=nn.Linear(dim,dim);self.rope=rope
    def execute(self,x,pos):
        b,n,c=x.shape
        z=self.qkv(x).reshape(b,n,3,self.heads,c//self.heads).permute(0,3,2,1,4)
        q,k,v=z[:,:,0],z[:,:,1],z[:,:,2]
        q,k=self.rope(q,pos),self.rope(k,pos)
        a=nn.softmax(jt.matmul(q,k.permute(0,1,3,2))*self.scale,dim=-1)
        return self.proj(jt.matmul(a,v).permute(0,2,1,3).reshape(b,n,c))


class CrossAttention(nn.Module):
    def __init__(self,dim,heads,rope):
        super().__init__();self.heads=heads;self.scale=(dim//heads)**-.5;self.rope=rope
        self.projq=nn.Linear(dim,dim);self.projk=nn.Linear(dim,dim);self.projv=nn.Linear(dim,dim);self.proj=nn.Linear(dim,dim)
    def execute(self,x,y,px,py):
        b,n,c=x.shape;m=y.shape[1]
        q=self.projq(x).reshape(b,n,self.heads,c//self.heads).permute(0,2,1,3)
        k=self.projk(y).reshape(b,m,self.heads,c//self.heads).permute(0,2,1,3)
        v=self.projv(y).reshape(b,m,self.heads,c//self.heads).permute(0,2,1,3)
        q,k=self.rope(q,px),self.rope(k,py)
        a=nn.softmax(jt.matmul(q,k.permute(0,1,3,2))*self.scale,dim=-1)
        return self.proj(jt.matmul(a,v).permute(0,2,1,3).reshape(b,n,c))


class Block(nn.Module):
    def __init__(self,dim,heads,ratio,eps,rope):
        super().__init__();self.norm1=LayerNorm(dim,eps);self.norm2=LayerNorm(dim,eps)
        self.attn=Attention(dim,heads,rope);self.mlp=Mlp(dim,ratio)
    def execute(self,x,pos):
        x=x+self.attn(self.norm1(x),pos)
        return x+self.mlp(self.norm2(x))


class DecoderBlock(nn.Module):
    def __init__(self,dim,heads,ratio,eps,rope):
        super().__init__()
        self.norm1=LayerNorm(dim,eps);self.norm2=LayerNorm(dim,eps);self.norm3=LayerNorm(dim,eps);self.norm_y=LayerNorm(dim,eps)
        self.attn=Attention(dim,heads,rope);self.cross_attn=CrossAttention(dim,heads,rope);self.mlp=Mlp(dim,ratio)
    def execute(self,x,y,px,py):
        x=x+self.attn(self.norm1(x),px)
        x=x+self.cross_attn(self.norm2(x),self.norm_y(y),px,py)
        return x+self.mlp(self.norm3(x))


class PatchEmbed(nn.Module):
    def __init__(self,dim,patch,channels=3):
        super().__init__();self.patch=patch;self.proj=nn.Conv2d(channels,dim,patch,stride=patch)
    def execute(self,x):
        z=self.proj(x);b,c,h,w=z.shape
        # y-major/x-minor Cartesian positions, then a distinct intrinsics-token row.
        y=jt.arange(h).reshape(h,1).broadcast((h,w))
        x=jt.arange(w).reshape(1,w).broadcast((h,w))
        p=jt.stack((y,x),dim=-1).reshape(1,h*w,2).broadcast((b,h*w,2))
        return z.reshape(b,c,h*w).permute(0,2,1),p


def dense_intrinsic_rays(k,h,w):
    """Jittor-only normalized local rays; pixel centers follow the author grid."""
    n=k.shape[0];a,b,c,d,e,f,g,z,i=[k.reshape(n,9)[:,j] for j in range(9)]
    values=(e*i-f*z,c*z-b*i,b*f-c*e,f*g-d*i,a*i-c*g,c*d-a*f,d*z-e*g,b*g-a*z,a*e-b*d)
    determinant=a*values[0]+b*values[3]+c*values[6]
    inv=jt.stack(values,dim=-1).reshape(n,3,3)/determinant.reshape(n,1,1)
    yy=((jt.arange(h).float32()+.5)/h).reshape(h,1).broadcast((h,w))
    xx=((jt.arange(w).float32()+.5)/w).reshape(1,w).broadcast((h,w))
    pixels=jt.stack((xx,yy,jt.ones((h,w))),dim=-1).reshape(h*w,3)
    rays=jt.matmul(inv,pixels.permute(1,0)).permute(0,2,1).reshape(n,h,w,3).permute(0,3,1,2)
    return rays/jt.sqrt((rays*rays).sum(1,keepdims=True))


class AsymmetricCroCo(nn.Module):
    def __init__(self,cfg):
        super().__init__();e,d=cfg['enc_dim'],cfg['dec_dim'];eps=cfg['layer_norm_eps'];r=cfg['mlp_ratio']
        self.intrinsics_mode=cfg.get('intrinsics_mode','token')
        if self.intrinsics_mode not in ('token','none','linear','pixelwise'):raise ValueError('Unsupported intrinsics mode')
        self.intrinsics_degree=cfg.get('intrinsics_degree',4)
        if self.intrinsics_degree not in (0,2,4,8):raise ValueError('Unsupported author intrinsic SH degree')
        channels=3 if self.intrinsics_degree==0 else (self.intrinsics_degree+1)**2
        rope=RoPE2D(cfg['rope_base']);self.patch_embed=PatchEmbed(e,cfg['patch_size'],3+channels if self.intrinsics_mode=='pixelwise' else 3)
        self.enc_blocks=LayerList([Block(e,cfg['enc_heads'],r,eps,rope) for _ in range(cfg['enc_depth'])])
        self.enc_norm=LayerNorm(e,eps);self.decoder_embed=nn.Linear(e,d)
        self.dec_blocks=LayerList([DecoderBlock(d,cfg['dec_heads'],r,eps,rope) for _ in range(cfg['dec_depth'])])
        self.dec_blocks2=LayerList([DecoderBlock(d,cfg['dec_heads'],r,eps,rope) for _ in range(cfg['dec_depth'])])
        # Independent parameters. Full official NPZ initializes each branch explicitly.
        self.dec_norm=LayerNorm(d,eps)
        if self.intrinsics_mode!='pixelwise':self.intrinsic_encoder=nn.Linear(9,e)
        self.mask_token=jt.zeros((1,1,d))  # retained for checkpoint compatibility; not used.
    def execute(self,context,trace=None):
        image=context['image'];b,v,_,h,w=image.shape
        if v!=2 or h%16 or w%16 or h!=w:
            raise ValueError('Two-view backbone needs two square views divisible by 16')
        # Same ordering as the official cat((view1,view2),dim=0), not a B,V flatten.
        inp=jt.concat((image[:,0],image[:,1]),dim=0)
        k=jt.concat((context['intrinsics'][:,0],context['intrinsics'][:,1]),dim=0)
        x,pos=self.encode_input(inp,k,h,w)
        if trace is not None:trace['tokens_with_intrinsics']=x
        for block in self.enc_blocks:x=block(x,pos)
        x=self.enc_norm(x);f1,f2=x[:b],x[b:];p1,p2=pos[:b],pos[b:]
        out1,out2=[f1],[f2]
        f1,f2=self.decoder_embed(f1),self.decoder_embed(f2)
        for left,right in zip(self.dec_blocks,self.dec_blocks2):
            old1,old2=f1,f2
            f1=left(old1,old2,p1,p2);f2=right(old2,old1,p2,p1)
            out1.append(f1);out2.append(f2)
        out1[-1]=self.dec_norm(out1[-1]);out2[-1]=self.dec_norm(out2[-1])
        if self.intrinsics_mode=='token':out1=[x[:,:-1] for x in out1];out2=[x[:,:-1] for x in out2]
        if trace is not None:
            for i,(a,z) in enumerate(zip(out1,out2)):trace[f'dec1_{i}']=a;trace[f'dec2_{i}']=z
        return out1,out2

    def encode_input(self,image,k,h,w):
        n=image.shape[0]
        if self.intrinsics_mode=='pixelwise':
            rays=dense_intrinsic_rays(k,h,w)
            if self.intrinsics_degree:
                function={2:rsh_cart_2,4:rsh_cart_4,8:rsh_cart_8}[self.intrinsics_degree]
                rays=function(rays.permute(0,2,3,1)).permute(0,3,1,2)
            image=jt.concat((image,rays),dim=1)
        x,pos=self.patch_embed(image)
        if self.intrinsics_mode in ('token','linear'):
            embedding=self.intrinsic_encoder(k.reshape(n,9)).unsqueeze(1)
            if self.intrinsics_mode=='linear':x=x+embedding
            else:
                x=jt.concat((x,embedding),dim=1)
                extra=jt.stack((jt.full((n,1),h//16,dtype='int32'),jt.zeros((n,1),dtype='int32')),dim=-1)
                pos=jt.concat((pos,extra),dim=1)
        return x,pos


def other_view_tokens(x):
    """Ascending non-self view order, then flatten (other_view, token)."""
    b,v,l,c=x.shape
    return jt.stack([jt.concat([x[:,j] for j in range(v) if j!=i],dim=1) for i in range(v)],dim=1)


class AsymmetricCroCoMulti(AsymmetricCroCo):
    def execute(self,context,trace=None):
        image=context['image'];b,v,_,h,w=image.shape
        if not 2<=v<=6 or h!=w or h%16:raise ValueError('Multi-view backbone currently supports 2..6 square patch16 views')
        x,pos=self.encode_input(image.reshape(b*v,3,h,w),context['intrinsics'].reshape(b*v,3,3),h,w)
        if trace is not None:trace['tokens_with_intrinsics']=x
        for block in self.enc_blocks:x=block(x,pos)
        x=self.enc_norm(x);l=x.shape[1];x=x.reshape(b,v,l,-1);pos=pos.reshape(b,v,l,2)
        result=[x];current=self.decoder_embed(x.reshape(b*v,l,-1)).reshape(b,v,l,-1);other_pos=other_view_tokens(pos)
        for left,right in zip(self.dec_blocks,self.dec_blocks2):
            others=other_view_tokens(current);width=current.shape[-1]
            first=left(current[:,0],others[:,0],pos[:,0],other_pos[:,0]).unsqueeze(1)
            rest=right(current[:,1:].reshape(b*(v-1),l,width),others[:,1:].reshape(b*(v-1),(v-1)*l,width),
                pos[:,1:].reshape(b*(v-1),l,2),other_pos[:,1:].reshape(b*(v-1),(v-1)*l,2)).reshape(b,v-1,l,width)
            current=jt.concat((first,rest),dim=1);result.append(current)
        result[-1]=self.dec_norm(result[-1].reshape(b*v,l,-1)).reshape(b,v,l,-1)
        if self.intrinsics_mode=='token':result=[item[:,:,:-1] for item in result]
        return result


class Interpolate(nn.Module):
    def execute(self,x):return nn.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)


class ResidualConvUnit(nn.Module):
    def __init__(self,dim):
        super().__init__();self.conv1=nn.Conv2d(dim,dim,3,padding=1);self.conv2=nn.Conv2d(dim,dim,3,padding=1)
    def execute(self,x):return x+self.conv2(nn.relu(self.conv1(nn.relu(x))))


class Refine(nn.Module):
    def __init__(self,dim):
        super().__init__();self.out_conv=nn.Conv2d(dim,dim,1)
        self.resConfUnit1=ResidualConvUnit(dim);self.resConfUnit2=ResidualConvUnit(dim)
    def execute(self,x,skip=None):
        if skip is not None:x=x+self.resConfUnit1(skip)
        return self.out_conv(nn.interpolate(self.resConfUnit2(x),scale_factor=2,mode='bilinear',align_corners=True))


class Scratch(nn.Module):
    def __init__(self,dims,feature):
        super().__init__()
        for i,d in enumerate(dims,1):
            setattr(self,f'layer{i}_rn',nn.Conv2d(d,feature,3,padding=1,bias=False))
            setattr(self,f'refinenet{i}',Refine(feature))


class DPTCore(nn.Module):
    def __init__(self,cfg,channels,gs=False):
        super().__init__();self.gs=gs;self.patch=cfg['patch_size']
        depth=cfg['dec_depth'];self.hooks=[0,depth*2//4,depth*3//4,depth]
        dims=cfg['layer_dims'];feature=cfg['feature_dim'];e,d=cfg['enc_dim'],cfg['dec_dim']
        self.act_postprocess=LayerList([
            LayerList([nn.Conv2d(e,dims[0],1),nn.ConvTranspose2d(dims[0],dims[0],4,stride=4)]),
            LayerList([nn.Conv2d(d,dims[1],1),nn.ConvTranspose2d(dims[1],dims[1],2,stride=2)]),
            LayerList([nn.Conv2d(d,dims[2],1)]),
            LayerList([nn.Conv2d(d,dims[3],1),nn.Conv2d(dims[3],dims[3],3,stride=2,padding=1)])])
        self.scratch=Scratch(dims,feature)
        if gs:
            self.input_merger=LayerList([nn.Conv2d(3,feature,7,padding=3),nn.ReLU()])
            self.head=LayerList([nn.Conv2d(feature,feature,3,padding=1,bias=False),nn.Identity(),nn.ReLU(),nn.Dropout(.1),nn.Conv2d(feature,channels,1)])
        else:
            self.head=LayerList([nn.Conv2d(feature,feature//2,3,padding=1),Interpolate(),nn.Conv2d(feature//2,feature//2,3,padding=1),nn.ReLU(),nn.Conv2d(feature//2,channels,1)])
    def execute(self,tokens,image):
        b,_,h,w=image.shape;nh,nw=h//self.patch,w//self.patch
        maps=[]
        for i,hook in enumerate(self.hooks):
            x=tokens[hook].permute(0,2,1).reshape(b,-1,nh,nw)
            x=self.act_postprocess[i](x)
            maps.append(getattr(self.scratch,f'layer{i+1}_rn')(x))
        x=self.scratch.refinenet4(maps[3])[:,:,:maps[2].shape[2],:maps[2].shape[3]]
        x=self.scratch.refinenet3(x,maps[2]);x=self.scratch.refinenet2(x,maps[1]);x=self.scratch.refinenet1(x,maps[0])
        if self.gs:x=nn.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)+self.input_merger(image)
        return self.head(x)


class DPTTask(nn.Module):
    def __init__(self,cfg,channels,gs=False):super().__init__();self.dpt=DPTCore(cfg,channels,gs)
    def execute(self,tokens,image):return self.dpt(tokens,image)


def expm1_stable(x):
    """FP32 expm1 with its exact first derivative; no assumed jt.expm1 API.

    Subtracting 1 from a float32 exp loses precision near zero. These Jittor
    code ops call the C++/CUDA math primitive directly and register d/dx=exp(x).
    """
    if str(x.dtype)!='float32':raise TypeError('Center-head expm1 expects FP32')
    return jt.code(x.shape,x.dtype,[x],
        cpu_header='#include <cmath>',
        cpu_src='for (int64_t i=0;i<in0->num;++i) out0_p[i]=std::expm1(in0_p[i]);',
        cpu_grad_src=['for (int64_t i=0;i<in0->num;++i) out0_p[i]=in1_p[i]*std::exp(in0_p[i]);'],
        cuda_header='#include <math.h>',
        cuda_src='''
__global__ void nps_expm1_forward(const float* x,float* y,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=expm1f(x[i]);
}
nps_expm1_forward<<<(in0->num+255)/256,256>>>(in0_p,out0_p,in0->num);
''',
        cuda_grad_src=['''
__global__ void nps_expm1_backward(const float* x,const float* g,float* y,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=g[i]*expf(x[i]);
}
nps_expm1_backward<<<(in0->num+255)/256,256>>>(in0_p,in1_p,out0_p,in0->num);
'''])


def radial_exp(raw):
    # Center head outputs BCHW, exp applies to RADIUS rather than coordinates.
    x=raw.permute(0,2,3,1)
    squared=(x*x).sum(-1,keepdims=True)
    d=jt.sqrt(jt.maximum(squared,1e-30))*(squared>0).float32()
    return x/jt.maximum(d,1e-8)*expm1_stable(d)


def covariance_from_xyzw(scales,q):
    i,j,k,r=[q[...,n] for n in range(4)]
    s=2/((q*q).sum(-1)+1e-8)
    R=jt.stack((1-s*(j*j+k*k),s*(i*j-k*r),s*(i*k+j*r),s*(i*j+k*r),1-s*(i*i+k*k),s*(j*k-i*r),s*(i*k-j*r),s*(j*k+i*r),1-s*(i*i+j*j)),dim=-1).reshape(*q.shape[:-1],3,3)
    scaled=R*scales.unsqueeze(-2)
    order=list(range(scaled.ndim));order[-1],order[-2]=order[-2],order[-1]
    return jt.matmul(scaled,scaled.permute(order))


def map_opacity(opacity,exponent):
    # Preserve the author's forward expression. Jittor's generic floating
    # power differentiates p**1 through p**0; its CUDA implementation can
    # produce NaN at p=0 (including 1-sigmoid(x)=0 for saturated logits).
    # For this exact exponent, the two powers are identities, including at
    # both endpoints. No clamp, epsilon, detached gradient or skipped update.
    if exponent==1:
        return .5*(1-(1-opacity)+opacity)
    return .5*(1-(1-opacity)**exponent+opacity**(1/exponent))


@dataclass
class Gaussians:
    means: jt.Var
    covariances: jt.Var
    harmonics: jt.Var
    opacities: jt.Var


class NoPoSplat(nn.Module):
    def __init__(self,cfg):
        if cfg['patch_size']!=16 or cfg['dec_depth']<4:
            raise ValueError('Supported DPT contract: patch16 and at least four decoder layers')
        for dim,heads in ((cfg['enc_dim'],cfg['enc_heads']),(cfg['dec_dim'],cfg['dec_heads'])):
            if dim%heads or (dim//heads)%4:raise ValueError('Attention head width must be divisible by four for 2D RoPE')
        super().__init__();self.cfg=cfg;self.multi_view=cfg.get('multi_view',False)
        self.backbone=(AsymmetricCroCoMulti if self.multi_view else AsymmetricCroCo)(cfg)
        raw=1+7+3*(cfg['sh_degree']+1)**2
        self.downstream_head1=DPTTask(cfg,3);self.downstream_head2=DPTTask(cfg,3)
        head=cfg.get('gs_head','dpt_gs')
        if head not in ('dpt_gs','dpt'):raise ValueError('Only author DPT/DPT-GS heads implemented')
        self.gaussian_param_head=DPTTask(cfg,raw,head=='dpt_gs');self.gaussian_param_head2=DPTTask(cfg,raw,head=='dpt_gs')
    def execute(self,context,global_step=0,trace=None):
        # context images MUST already be normalized to [-1,1].
        if self.multi_view:return self.execute_multiview(context,global_step,trace)
        a,b=self.backbone(context,trace)
        im=context['image'];batch,_,_,h,w=im.shape
        rawp=[self.downstream_head1(a,im[:,0]),self.downstream_head2(b,im[:,1])]
        rawgs=[self.gaussian_param_head(a,im[:,0]),self.gaussian_param_head2(b,im[:,1])]
        pts=[radial_exp(x) for x in rawp]
        means=jt.stack(pts,dim=1).reshape(batch,2*h*w,3)
        raw=jt.stack(rawgs,dim=1).permute(0,1,3,4,2).reshape(batch,2*h*w,-1)
        opacity=jt.sigmoid(raw[...,0])
        c=self.cfg;x=c['opacity_initial']+min(global_step/max(c['opacity_warmup'],1),1)*(c['opacity_final']-c['opacity_initial'])
        exponent=2**x;opacity=map_opacity(opacity,exponent)
        scales=jt.minimum(nn.softplus(raw[...,1:4])*.001,.3)
        q=raw[...,4:8];q=q/(jt.sqrt((q*q).sum(-1,keepdims=True))+1e-8)
        cov=covariance_from_xyzw(scales,q)
        degree=c['sh_degree'];dsh=(degree+1)**2
        mask=np.ones(dsh,np.float32)
        for l in range(1,degree+1):mask[l*l:(l+1)**2]=.1*.25**l
        sh=raw[...,8:].reshape(batch,2*h*w,3,dsh)*jt.array(mask).stop_grad()
        if trace is not None:
            trace.update(raw_pts1=rawp[0],raw_pts2=rawp[1],pts1=pts[0],pts2=pts[1],raw_gs1=rawgs[0],raw_gs2=rawgs[1],scales=scales,rotations=q,means=means,covariances=cov,harmonics=sh,opacities=opacity)
        return Gaussians(means,cov,sh,opacity)

    def execute_multiview(self,context,global_step=0,trace=None):
        if self.cfg.get('gs_head','dpt_gs')!='dpt_gs':raise ValueError('Author multi-view path only implements DPT-GS')
        features=self.backbone(context,trace);images=context['image'];batch,v,_,h,w=images.shape
        rawp=[];rawgs=[];pts=[]
        for view in range(v):
            tokens=[x[:,view] for x in features]
            center=self.downstream_head1 if view==0 else self.downstream_head2
            attribute=self.gaussian_param_head if view==0 else self.gaussian_param_head2
            point=center(tokens,images[:,view]);other=attribute(tokens,images[:,view]);rawp.append(point);rawgs.append(other);pts.append(radial_exp(point))
            if trace is not None:trace.update({f'raw_pts{view+1}':point,f'raw_gs{view+1}':other,f'pts{view+1}':pts[-1]})
        means=jt.stack(pts,dim=1).reshape(batch,v*h*w,3)
        raw=jt.stack(rawgs,dim=1).permute(0,1,3,4,2).reshape(batch,v*h*w,-1)
        opacity=jt.sigmoid(raw[...,0]);c=self.cfg
        x=c['opacity_initial']+min(global_step/max(c['opacity_warmup'],1),1)*(c['opacity_final']-c['opacity_initial']);exponent=2**x
        opacity=map_opacity(opacity,exponent);scales=jt.minimum(nn.softplus(raw[...,1:4])*.001,.3)
        q=raw[...,4:8];q=q/(jt.sqrt((q*q).sum(-1,keepdims=True))+1e-8);cov=covariance_from_xyzw(scales,q)
        degree=c['sh_degree'];dsh=(degree+1)**2;mask=np.ones(dsh,np.float32)
        for l in range(1,degree+1):mask[l*l:(l+1)**2]=.1*.25**l
        sh=raw[...,8:].reshape(batch,v*h*w,3,dsh)*jt.array(mask).stop_grad()
        if trace is not None:trace.update(scales=scales,rotations=q,means=means,covariances=cov,harmonics=sh,opacities=opacity)
        return Gaussians(means,cov,sh,opacity)
