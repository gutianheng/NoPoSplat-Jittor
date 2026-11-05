# Executed experimental protocol

The report distinguishes official-weight transfer from independent reduced training.

| Setting | Executed value |
|---|---|
| Input | 2 views, 256×256, normalised intrinsics; no input extrinsics to encoder |
| Model | Default CroCo/DPT path, SH degree4, pixelwise Gaussian prediction |
| Training | RE10K only; 3000 fixed candidate scenes,2819 actually sampled |
| Initialisation | Author-filtered MASt3R, seed111123; not official NoPoSplat finetuning |
| Variants | default / no intrinsic embedding / no RGB shortcut; independent Torch and Jittor runs |
| Budget | 3000 optimiser updates each, batch4 × accumulate2;24000 sample occurrences |
| Optimisation | AdamW, lr2e-4, backbone multiplier0.1, betas(0.9,0.95), wd0.05, clip0.5 |
| Schedule | warmup125;5000-update LR horizon,2500-update view-gap curriculum; stopped at3000 |
| Testing | RE10K500 and ACID500,3 targets/scene,100 target-pose steps,lr0.005 |
| Precision | FP32; AMP/TF32 disabled |

Two frameworks and three variants have identical6000 microbatch input records. Training and test scene IDs do not overlap. ACID was evaluated only with official weights, not independently trained. Small/medium/large refer to view-overlap groups, not model sizes.

RE10K overlap counts:125/229/146; ACID:93/240/167. The500-scene sets contain previously observed100-scene subsets; the additional400 are reported separately in results/summary.json. Integrity failures were excluded by fixed candidate/replacement order without consulting model scores. Some detailed CRC exclusion rules were expanded after encountering an upstream archive defect; they are not retrospectively called fully preregistered.

## Target-pose alignment and input-target coincidence

Pose alignment freezes Gaussians and uses target ground-truth images to optimise the target camera. It is an evaluation protocol and not pure feed-forward deployment. Each model optimises its own target cameras.

An additional audit found43/1500 RE10K and32/1500 ACID target indices coinciding with input indices in the frozen evaluation subsets. The original aggregate metrics and indices are retained; these coincidences are disclosed rather than silently changing the benchmark. Qualitative presentation targets exclude them. Adjacent but distinct frames can still be easy novel views.

## Paper comparison and limits

Paper full-test numbers: RE10K25.033/0.838/0.160; ACID25.961/0.781/0.189. Our500-scene subset differs from that full benchmark. Each short-trained model has24000 sample occurrences, versus roughly2.4million in the author's training recipes. Batch size, LR schedule, curriculum and data coverage also differ. Therefore short training is not full-paper convergence, and changing steps alone cannot explain every difference.

## Local numerical repair

The first extended Jittor attempt produced nonfinite parameters. The failure was isolated to a floating-point exponent-one opacity derivative at a boundary. For exactly exponent1, the two powers were replaced with identities while retaining the rest of the author's expression. Boundary forward/gradient and update/recovery checks passed; failed runs were excluded and valid Jittor models were retrained. No nan_to_num, skipped batches or tolerance relaxation was used.
