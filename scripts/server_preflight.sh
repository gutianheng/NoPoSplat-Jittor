#!/usr/bin/env bash
# Read-only Linux hardware/toolchain probe. No pip, downloads or training.
set -u
cd "$(dirname "$0")/.." || exit 1
if [[ "$(uname -s)" != Linux ]]; then
  printf '%s\n' 'This probe targets the Linux rental instance.' >&2
  exit 1
fi
task_missing=0
for task_command in nvidia-smi nvcc g++ cmake "${NPS_PYTHON_BIN:-python3.10}"; do
  if command -v "$task_command" >/dev/null 2>&1; then
    printf 'FOUND %s: %s\n' "$task_command" "$(command -v "$task_command")"
  else
    printf 'MISSING %s\n' "$task_command" >&2
    task_missing=1
  fi
done
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || task_missing=1
  # Some older drivers do not support this field; do not guess architecture.
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader || true
fi
if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi
if command -v g++ >/dev/null 2>&1; then g++ --version; fi
if command -v cmake >/dev/null 2>&1; then cmake --version; fi
if command -v "${NPS_PYTHON_BIN:-python3.10}" >/dev/null 2>&1; then
  "${NPS_PYTHON_BIN:-python3.10}" - <<'PY'
import json,platform,shutil,sys,sysconfig,os
from pathlib import Path
limits={}
for name in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
    p=Path(name)
    if p.is_file():limits[name]=p.read_text().strip()
config_candidates=[Path(os.environ['python_config_path'])] if os.environ.get('python_config_path') else []
config_candidates += [Path(sys.base_prefix)/'bin'/n for n in ('python3.10-config','python3-config','python-config')]
config_candidates += [Path('/usr/bin/python3.10-config'),Path('/usr/local/bin/python3.10-config')]
configs=[str(p) for p in config_candidates if p.is_file() and os.access(p,os.X_OK)]
header=Path(sysconfig.get_path('include'))/'Python.h'
print(json.dumps({'python':sys.version,'platform':platform.platform(),
                  'python_header':str(header),'python_header_exists':header.is_file(),
                  'python_config_candidates_found':configs,
                  'working_disk_free_gib':shutil.disk_usage('.').free/1024**3,
                  'container_memory_limits':limits},indent=2))
if sys.version_info[:2]!=(3,10):raise SystemExit('This run package targets Python 3.10; select the intended interpreter before installing.')
if not header.is_file() or not configs:raise SystemExit('Python development header/config missing. Resolve before first Jittor import; use matching base python_config_path or install python3.10-dev.')
PY
  if [[ $? -ne 0 ]]; then task_missing=1; fi
fi
printf '%s\n' 'Probe only: DOES NOT verify Jittor/CUDA correctness, paper coverage, price, or cloud shutdown.'
exit "$task_missing"
