"""Explicit math permission; never use implicit FP16 conversion for TF32."""
from pathlib import Path
import os


def configure_math(jt,mode='fp32'):
    if mode not in ('fp32','tf32'):raise ValueError('Math mode must be fp32 or tf32')
    jt.flags.amp_level=0
    if mode=='fp32':jt.flags.use_tensorcore=0;return
    if os.environ.get('NVIDIA_TF32_OVERRIDE')=='0':raise ValueError('TF32 requested but NVIDIA_TF32_OVERRIDE=0; launch an explicitly configured process')
    directory=Path(jt.__file__).resolve().parent/'extern/cuda/cudnn/ops'
    expected='use_tensorcore == 1 ? CUDNN_DEFAULT_MATH : CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION'
    for name in ('cudnn_conv_op.cc','cudnn_conv_backward_x_op.cc','cudnn_conv_backward_w_op.cc'):
        if expected not in (directory/name).read_text():raise RuntimeError('TF32 requires the audited cuDNN DEFAULT_MATH compatibility patch; stock mode also permits FP16 down-conversion')
    jt.flags.use_tensorcore=1
