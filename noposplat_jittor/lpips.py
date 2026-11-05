"""Frozen VGG LPIPS v0.1, with input gradients; weights exported separately.

Model structure matches richzhang/PerceptualSimilarity LPIPS(net='vgg').
No automatic weights download and no random fallback if weights are missing.
"""
import numpy as np
import jittor as jt
from jittor import nn
from .model import LayerList


class LPIPSVGG(nn.Module):
    def __init__(self,weights_path):
        super().__init__()
        layers=[];c=3
        for channels,count in ((64,2),(128,2),(256,3),(512,3),(512,3)):
            for _ in range(count):layers.extend([nn.Conv2d(c,channels,3,padding=1),nn.ReLU()]);c=channels
            layers.append(nn.MaxPool2d(2,stride=2))
        self.features=LayerList(layers[:30])
        self.linear=[]
        with np.load(weights_path,allow_pickle=False) as archive:
            if str(archive['__format__'].item())!='lpips_vgg_0.1':raise ValueError('Expected LPIPS VGG v0.1 export')
            for key,p in self.features.state_dict().items():
                name='features.'+key
                if name not in archive or tuple(p.shape)!=archive[name].shape:raise ValueError(f'Bad LPIPS weight: {name}')
                p.assign(jt.array(archive[name]));p.stop_grad()
            for i,c in enumerate((64,128,256,512,512)):
                a=archive[f'linear.{i}']
                if a.size!=c:raise ValueError('LPIPS calibration width mismatch')
                self.linear.append(jt.array(a.reshape(1,c,1,1)).stop_grad())
        self.eval()
    def _features(self,x):
        x=2*x-1
        shift=jt.array(np.array([-.030,-.088,-.188],np.float32).reshape(1,3,1,1)).stop_grad()
        scale=jt.array(np.array([.458,.448,.450],np.float32).reshape(1,3,1,1)).stop_grad()
        x=(x-shift)/scale;out=[]
        for i,layer in enumerate(self.features):
            x=layer(x)
            if i in (3,8,15,22,29):out.append(x)
        return out
    def execute(self,x,y):
        # Do not use no_grad here: the frozen network must differentiate w.r.t. x.
        result=0
        for i,(a,b) in enumerate(zip(self._features(x),self._features(y))):
            norm=lambda z:z/(jt.sqrt((z*z).sum(1,keepdims=True))+1e-10)
            result=result+(((norm(a)-norm(b))**2)*self.linear[i]).sum(1).mean((1,2))
        return result

