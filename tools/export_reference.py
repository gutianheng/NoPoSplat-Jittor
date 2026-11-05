"""Export original PyTorch model/batch/intermediate arrays for real parity checks.

Run ONLY in the original project's working Python environment, not Jittor's.
Outputs contain real computed values, never preset pass/results numbers.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official',required=True);p.add_argument('--experiment',default='re10k',choices=['re10k','acid'])
    p.add_argument('--checkpoint',help='Full NoPoSplat checkpoint')
    p.add_argument('--mast3r',help='Instead, initialize with official MASt3R checkpoint_filter_fn')
    p.add_argument('--batch',help='Framework-neutral batch .npz from tools/prepare_batch.py')
    p.add_argument('--output',required=True);p.add_argument('--grad',action='store_true');p.add_argument('--cpu',action='store_true')
    p.add_argument('--trusted-input',action='store_true')
    p.add_argument('--photometric',action='store_true',help='Use one actual target render and MSE+LPIPS instead of the algebraic probe')
    p.add_argument('--lpips-weights',default='weights/lpips_vgg.npz')
    p.add_argument('--seed',type=int,default=123456,help='Constructor seed; author default training uses 111123. Export all newly initialized heads too.')
    p.add_argument('--override',action='append',default=[],help='Additional explicit original Hydra override; recorded in initialization provenance')
    p.add_argument('--math-mode',choices=['fp32','tf32'],default='fp32')
    args=p.parse_args()
    if not args.trusted_input:p.error('Acknowledge trusted checkpoint loading with --trusted-input')
    if args.checkpoint and args.mast3r:p.error('Choose full checkpoint or MASt3R initialization, not both')
    if args.photometric and (not args.batch or args.cpu):p.error('Photometric reference needs a frozen batch and CUDA')
    root=Path(args.official).resolve();out=Path(args.output).resolve()
    if out.exists():raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True)
    sys.path.insert(0,str(root));sys.path.append(str(Path(__file__).resolve().parents[1]))
    import torch
    from hydra import initialize_config_dir,compose
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder
    torch.manual_seed(args.seed)
    with initialize_config_dir(version_base=None,config_dir=str(root/'config')):
        raw=compose(config_name='main',overrides=[f'+experiment={args.experiment}',*args.override])
    set_cfg(raw);cfg=load_typed_root_config(raw);encoder,_=get_encoder(cfg.model.encoder)
    init_info={}
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False);state=state.get('state_dict',state)
        state={k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')} if any(k.startswith('encoder.') for k in state) else state
        encoder.load_state_dict(state,strict=True)
    elif args.mast3r:
        from src.misc.weight_modify import checkpoint_filter_fn
        state=torch.load(args.mast3r,map_location='cpu',weights_only=False)
        result=encoder.load_state_dict(checkpoint_filter_fn(state['model'],encoder),strict=False)
        init_info={'missing':result.missing_keys,'unexpected':result.unexpected_keys}
    else:init_info={'warning':'Random official initialization for numerical tests only; not pretrained inference.'}
    init_info.update(torch_seed=args.seed,initialization_source='full_checkpoint' if args.checkpoint else 'mast3r_author_filter' if args.mast3r else 'random_diagnostic',original_overrides=args.override)
    encoder.eval()
    device='cpu' if args.cpu else 'cuda';encoder.to(device)
    # CroCo enables TF32; strict initial tests explicitly disable it after imports.
    torch.backends.cuda.matmul.allow_tf32=args.math_mode=='tf32';torch.backends.cudnn.allow_tf32=args.math_mode=='tf32'
    init_info.update(torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),math_mode=args.math_mode,
        allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
    from noposplat_jittor.weights import canonical_key
    arrays={}
    for k,v in encoder.state_dict().items():
        key=canonical_key(k);a=v.detach().float().cpu().numpy()
        if key in arrays and not np.array_equal(arrays[key],a):raise ValueError('Conflicting duplicate weight aliases')
        arrays[key]=a
    np.savez(out/'encoder.npz',**arrays)
    init_info['loss_mode']='photometric_one_target' if args.photometric else 'algebraic_gaussian_probe'
    (out/'initialization.json').write_text(json.dumps(init_info,indent=2),encoding='utf-8')
    if not args.batch:
        print('Exported initialized encoder only; no forward was run.');return
    archive=np.load(args.batch,allow_pickle=False)
    context={k.split('.',1)[1]:torch.tensor(archive[k],device=device) for k in archive.files if k.startswith('context.')}
    trace={};handles=[]
    multiview=cfg.model.encoder.name=='noposplat_multi';hook_counts={}
    def hook(name):
        def callback(module,inputs,output):
            actual=name
            if multiview and name in ('raw_pts2','raw_gs2'):
                offset=hook_counts.get(name,0);hook_counts[name]=offset+1;actual=name[:-1]+str(2+offset)
            trace[actual]=output
        return callback
    for name,module in [('raw_pts1',encoder.downstream_head1.dpt),('raw_pts2',encoder.downstream_head2.dpt),('raw_gs1',encoder.gaussian_param_head.dpt),('raw_gs2',encoder.gaussian_param_head2.dpt)]:handles.append(module.register_forward_hook(hook(name)))
    def before_encoder(module,inputs):trace['tokens_with_intrinsics']=inputs[0]
    handles.append(encoder.backbone.enc_blocks[0].register_forward_pre_hook(before_encoder))
    viz={}
    with torch.set_grad_enabled(args.grad):
        gaussians=encoder(context,visualization_dump=viz)
        # The visualization adds a singleton surface/sample axis. Export the
        # point-head trace in canonical BHWC layout without changing values.
        trace.update(means=gaussians.means,covariances=gaussians.covariances,harmonics=gaussians.harmonics,opacities=gaussians.opacities,scales=viz['scales'],rotations=viz['rotations'])
        for i in range(viz['means'].shape[1]):trace[f'pts{i+1}']=viz['means'][:,i].squeeze(-2)
        if args.photometric:
            from src.model.decoder.cuda_splatting import render_cuda
            from tools.export_pose_reference import load_lpips
            perceptual=load_lpips(args.lpips_weights)
            target={k.split('.',1)[1]:torch.tensor(archive[k][:,0],device=device) for k in archive.files if k.startswith('target.')}
            color,_=render_cuda(target['extrinsics'],target['intrinsics'],target['near'],target['far'],tuple(target['image'].shape[-2:]),
                torch.zeros((len(target['image']),3),device=device),
                gaussians.means,gaussians.covariances,gaussians.harmonics,gaussians.opacities,scale_invariant=True)
            mse=((color-target['image'])**2).mean();lpips_loss=perceptual(color,target['image'],normalize=True).mean()
            loss=mse+.05*lpips_loss
            trace.update(training_color=color,mse_loss=mse,lpips_loss=lpips_loss)
        else:loss=sum((getattr(gaussians,k)**2).mean() for k in ('means','covariances','harmonics','opacities'))
        gradients={}
        if args.grad:
            if args.photometric:
                for key in ('means','covariances','harmonics','opacities'):getattr(gaussians,key).retain_grad()
            loss.backward()
            if args.photometric:
                np.savez(out/'gaussian_gradients.npz',**{key:getattr(gaussians,key).grad.detach().cpu().numpy() for key in ('means','covariances','harmonics','opacities')})
            for k,p in encoder.named_parameters():
                if p.grad is not None:gradients[canonical_key(k)]=p.grad.detach().cpu().numpy()
            np.savez(out/'gradients.npz',**gradients)
    np.savez(out/'forward.npz',**{k:v.detach().float().cpu().numpy() for k,v in trace.items()},probe_loss=np.asarray(loss.detach().cpu()))
    for handle in handles:handle.remove()
    np.savez(out/'batch.npz',**{k:archive[k] for k in archive.files})
    print('Exported original-model forward'+(' and gradients' if args.grad else '')+'.')


if __name__=='__main__':main()

