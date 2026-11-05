# Known limits and outstanding numerical qualification

1. **Historical strict gradients are not all aligned:**629/1129 failed full-chain checks, maximum gradient relativeL2 about3.17%; the Torch self-repeat passed1129. No claim is made that subsequent aggregate metrics eliminate this issue.
2. Default reduced-training PSNR mean difference+0.342768 exceeds the original0.3dB tolerance. Other two metrics pass. Four other paired cases meet all three aggregate tolerances, which does not imply per-scene agreement.
3. Implemented scope: default two-view square patch16, encoder intrinsic-token, DPT/DPT-GS, one Gaussian per pixel. Not full three/multiview network, dense/additive intrinsic variants, all paper ablations, mixed training, independentPnP/AUC or512-resolution experiments.
4. One training seed and reduced data/sample exposure; no full-paper convergence claim. Host/GPU-driver migration history prevents attributing all training differences solely to the framework.
5. Target-pose optimisation uses target ground truth; no unmeasured FPS or deployment-time claims. Bridge synchronisation is conservative for correctness and can affect performance.
6. Data were obtained from a fixed public mirror revision with full chunkSHA checks; an independent whole-library identity proof against original release data is not available. Frozen subset target-input coincidences are disclosed in EXPERIMENTS.md.
7. Third-party components and weights have separate licence conditions; the aggregate is not all-MIT.

Suggested next controls: freeze the same cameras and weights, isolate network/render/loss gradients; report non-coincident targets separately; then use multiple seeds and progressively larger training exposure. These are planned, not completed experiments.
