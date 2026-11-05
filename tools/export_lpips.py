"""Run in a separate PyTorch reference environment; may download VGG weights."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    import torch
    from lpips import LPIPS
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    model=LPIPS(net='vgg',version='0.1').eval()
    arrays={'__format__':np.asarray('lpips_vgg_0.1')}
    for i in range(1,6):
        for name,layer in getattr(model.net,f'slice{i}').named_children():
            if isinstance(layer,torch.nn.Conv2d):
                arrays[f'features.{name}.weight']=layer.weight.detach().cpu().numpy()
                arrays[f'features.{name}.bias']=layer.bias.detach().cpu().numpy()
    for i in range(5):
        conv=next(x for x in getattr(model,f'lin{i}').model if isinstance(x,torch.nn.Conv2d))
        arrays[f'linear.{i}']=conv.weight.detach().cpu().numpy()
    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**arrays)
    # A deterministic forward/input-gradient fixture for independent Jittor checks.
    rng=np.random.default_rng(42)
    x=torch.tensor(rng.random((1,3,32,32),dtype=np.float32),requires_grad=True)
    y=torch.tensor(rng.random((1,3,32,32),dtype=np.float32))
    loss=model(x,y,normalize=True).mean();loss.backward()
    np.savez(path.with_name('lpips_reference.npz'),x=x.detach().numpy(),y=y.numpy(),loss=loss.detach().numpy(),grad=x.grad.numpy())
    print(path)


if __name__=='__main__':main()
