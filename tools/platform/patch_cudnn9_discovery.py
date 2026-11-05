"""Version-specific cuDNN9 library discovery adaptation from the executed setup.
This patches an isolated Jittor installation, not model mathematics.
"""
from pathlib import Path
import argparse,hashlib,json,os

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',default=os.environ.get('NPS_BLACKWELL_PREFIX','/root/nps-blackwell'))
    args=p.parse_args();root=Path(args.prefix)
    path=root/'venv/lib/python3.10/site-packages/jittor/compile_extern.py'
    before=path.read_text(encoding='utf-8')
    needle='                libs = ["libcudnn_ops_infer.so", "libcudnn_ops_train.so", "libcudnn_cnn_infer.so", "libcudnn_cnn_train.so"]'
    replacement='                # NPS isolated Blackwell compatibility patch: cuDNN 9 uses\n                # consolidated component names (upstream Jittor issue #666).\n                version_header = os.path.join(os.path.dirname(cuda_include_name), "cudnn_version.h")\n                major = 8\n                if os.path.isfile(version_header):\n                    with open(version_header) as version_file:\n                        match = re.search(r"#define\\s+CUDNN_MAJOR\\s+(\\d+)", version_file.read())\n                    if match: major = int(match.group(1))\n                libs = (["libcudnn_graph.so", "libcudnn_ops.so", "libcudnn_cnn.so", "libcudnn_adv.so"] if major >= 9 else\n                        ["libcudnn_ops_infer.so", "libcudnn_ops_train.so", "libcudnn_cnn_infer.so", "libcudnn_cnn_train.so"])'
    if before.count(needle)!=1:raise ValueError('Unexpected Jittor source or already patched; no blind replacement')
    after=before.replace(needle,replacement);compile(after,str(path),'exec')
    backup=path.with_suffix('.py.before_nps_cudnn9')
    if backup.exists():raise FileExistsError('Preserve the original backup; inspect installed state first')
    backup.write_bytes(path.read_bytes());path.write_text(after,encoding='utf-8')
    print(json.dumps({'source_backup':str(backup),'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'fresh_gpu_qualification':False}))

if __name__=='__main__':main()
