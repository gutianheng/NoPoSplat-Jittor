"""Compile ONLY the official cache parser to reproduce a flag classification bug.

No Jittor/GPU execution. --source must be the verified official cache_compile.cc.
The caller records the provenance of that source separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();source=Path(args.source).read_text(encoding='utf-8')
    function=source[source.index('void find_names('):source.index('size_t skip_comments(')]
    before='nvcc "/tmp/operator.cc" -I"/tmp/nps/native" -L"/tmp/nps/native/build" -lnps_raster -Xlinker -rpath -Xlinker "/tmp/nps/native/build" -o "/tmp/operator.so"'
    after=before.replace('-Xlinker -rpath -Xlinker "/tmp/nps/native/build"','-Xlinker -rpath="/tmp/nps/native/build"')
    prefix='''#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <stdexcept>
using namespace std;
struct Check { bool ok; explicit Check(bool v):ok(v){};
template<class T> Check& operator<<(const T&){return *this;}
~Check() noexcept(false){if(!ok)throw runtime_error("parser assertion");}};
#define CHECK(x) Check(bool(x))
#define ASSERT(x) Check(bool(x))
'''
    body=prefix+function+'\nint main(){\n'
    for label,command in [('before',before),('after',after)]:
        body+='{vector<string> names; string output; map<string,vector<string>> extra;\n'
        body+='find_names('+json.dumps(command)+',names,output,extra);\n'
        body+='cout<<"'+label+'"; for(auto& n:names)cout<<"\\t"<<n; cout<<"\\n";}\n'
    body+='}\n'
    compiler=shutil.which('g++')
    if not compiler:raise RuntimeError('g++ not found')
    temp_root=ROOT/'.test_tmp';temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as temp:
        cpp=Path(temp)/'parser.cpp';exe=Path(temp)/'parser.exe'
        cpp.write_text(body,encoding='utf-8',newline='\n')
        subprocess.run([compiler,'-std=c++17','-static-libstdc++','-static-libgcc',str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
        result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    parsed={parts[0]:parts[1:] for parts in (line.split('\t') for line in result.stdout.splitlines())}
    passed=parsed=={'before':['/tmp/operator.cc','/tmp/nps/native/build'],'after':['/tmp/operator.cc']}
    report={'scope':'Actual official C++ argument parser only; NOT NVCC compilation or native rendering',
            'source_sha256':hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
            'compiler':compiler,'classified_input_files':parsed,'pass':passed}
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
