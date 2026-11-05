"""Execute the author's small geometry functions on CPU without model imports.

This diagnoses FP32 operation-order differences. It does not establish CUDA
equivalence or exercise the full network, and never edits the reference source.
"""
from pathlib import Path
import argparse,ast,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def original_functions(path,names,namespace):
    source=Path(path).read_text(encoding='utf-8');tree=ast.parse(source)
    found=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in found}!=set(names):raise ValueError('Missing reference functions')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,*found],type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)
    return hashlib.sha256(source.encode()).hexdigest()


def error(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return {'max_abs':float(np.max(np.abs(a-b))),'mean_abs':float(np.mean(np.abs(a-b)))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--official',required=True);p.add_argument('--batch',action='append',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    import torch
    from einops import einsum
    from noposplat_jittor.geometry_numpy import projection_matrices,retract_pose
    namespace={'torch':torch,'einsum':einsum}
    original=Path(args.official)
    hashes={relative:original_functions(original/relative,names,namespace) for relative,names in (
        ('src/geometry/projection.py',['get_fov']),
        ('src/model/decoder/cuda_splatting.py',['get_projection_matrix']),
        ('src/misc/cam_utils.py',['skew_sym_mat','SO3_exp','V','SE3_exp','update_pose']))}
    cameras=[];poses=[]
    for filename in args.batch:
        with np.load(filename,allow_pickle=False) as batch:
            for view in range(batch['target.extrinsics'].shape[1]):
                c2w=batch['target.extrinsics'][0,view].copy();k=batch['target.intrinsics'][0,view].copy()
                near=np.float32(batch['target.near'][0,view]);far=np.float32(batch['target.far'][0,view])
                scale=np.float32(1)/near;scaled=c2w.copy();scaled[:3,3]*=scale
                actual=projection_matrices(scaled,k,np.float32(near*scale),np.float32(far*scale))
                c=torch.tensor(c2w[None]);intr=torch.tensor(k[None]);n=torch.tensor([near]);f=torch.tensor([far]);s=1/n
                c[:,:3,3]*=s[:,None];fov=namespace['get_fov'](intr);tan=(.5*fov).tan()
                raw=namespace['get_projection_matrix'](n*s,f*s,fov[:,0],fov[:,1]).transpose(1,2)
                v=c.inverse().transpose(1,2);full=v@raw
                ref=[v[0].numpy(),full[0].numpy(),raw[0].numpy(),tan[0,0].item(),tan[0,1].item()]
                cameras.append({'batch':Path(filename).name,'target':view,'differences':{name:error(a,b) for name,a,b in zip(('view','full_projection','raw_projection','tanx','tany'),ref,actual)}})
                poses.append(c2w)
    rng=np.random.default_rng(548);retractions=[]
    for angle in (0.,5e-6,1e-5,1e-4,1e-3,8e-3,.1):
        vector=rng.normal(size=3);theta=(vector/np.linalg.norm(vector)*angle).astype(np.float32);rho=rng.normal(0,.005,3).astype(np.float32)
        c2w=poses[len(retractions)%len(poses)]
        expected=namespace['update_pose'](torch.tensor(rho[None]),torch.tensor(theta[None]),torch.tensor(c2w[None]))[0].numpy()
        retractions.append({'angle':angle,'differences':error(expected,retract_pose(c2w,rho,theta))})
    report={'scope':__doc__,'torch':torch.__version__,'device':'cpu','source_sha256':hashes,'camera_cases':cameras,'pose_cases':retractions}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'cameras':len(cameras),'camera_max_abs':max(v['max_abs'] for row in cameras for v in row['differences'].values()),'pose_max_abs':max(x['differences']['max_abs'] for x in retractions),'report':str(out)},ensure_ascii=True))


if __name__=='__main__':main()
