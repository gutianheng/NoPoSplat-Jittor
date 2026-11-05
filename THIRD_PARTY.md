# 来源、署名与使用边界

本仓库是非官方 Jittor 迁移工作，不代表作者或 Jittor 官方实现；不能将未跑通的结果描述为论文复现成功。

## NoPoSplat

- 论文：Botao Ye et al., *No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images*, ICLR 2025（Oral）。
- 原项目以用户提供的 `the author source snapshot` 为基准；原项目 MIT 许可完整保留在 `vendor/licenses/NoPoSplat-LICENSE`。
- 官方项目：https://github.com/cvg/NoPoSplat
- 模型组织、规范高斯、内参token与数据/优化器约定参考原实现。不是将原 PyTorch 模型包装后冒充 Jittor。

## CroCo / DUSt3R / DPT

- `model.py` 中注意力、RoPE、双路解码与DPT为参考原文件的Jittor改写。
- 原文件声明：Copyright (C) 2022/2024-present Naver Corporation. All rights reserved. Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
- 对这些派生部分保留署名，并按 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 使用；**原项目顶层MIT不取消嵌入组件的非商业限制**。
- 修改包括：forward→execute、Jittor张量算子、显式中心方差LayerNorm、无CUDA依赖的RoPE、去除未选用分支、保留checkpoint层级。
- 来源文件哈希记录在 `vendor/source_manifest.json`，并不冒称ZIP具有可核实Git commit。

## 可微高斯光栅化器

- 来自用户本地 `NoPoSplat_diff_gaussian`，上游 https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose 。
- 该组件基于 Inria GRAPHDECO 3DGS 光栅化器。源头与许可已在 `vendor/rasterizer/LICENSE.md` 及源码头中保留。
- 仅复制CUDA内核/头文件/GLM，不复制PyTorch C++绑定，不依赖libtorch。
- `native/bridge.cu` 和 `rasterizer.py` 为本迁移新增的C ABI与Jittor Function桥接，保留原颜色、深度和相机Jacobian计算。商业使用需另外核对原组件许可。
- GLM的许可见 `vendor/rasterizer/third_party/glm` 下的对应文件。

## Jittor 与 LPIPS

- Jittor：https://github.com/Jittor/jittor ，依赖而非本仓库所有代码的许可证。
- LPIPS：https://github.com/richzhang/PerceptualSimilarity ，这里实现VGG/v0.1计算并提供独立导出与输入梯度对齐；权重需要按上游许可获取，不随运行包分发。

研究复现、发布GitHub前，应保留本文件及各上游许可证；不要把本包统一改标为“全部MIT、可自由商用”。

