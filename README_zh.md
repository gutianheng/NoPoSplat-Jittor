# NoPoSplat的Jittor迁移与配对复现实验

[English README](README.md) · [环境与完整命令](docs/RUNNING.md) · [Blackwell适配](docs/BLACKWELL.md) · [实验协议](docs/EXPERIMENTS.md) · [未通过项](docs/LIMITATIONS.md)

这是顾天恒整理的**非官方研究迁移**，对应ICLR2025 Oral论文《No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images》。采用作者默认双视图256×256路径，在Jittor中实现ViT/DPT、高斯参数后处理、可微渲染接口、MSE+LPIPS、AdamW与训练/测试循环。原CUDA数学内核保留，框架桥接不依赖ATen或libtorch；PyTorch只用于离线转换和独立参考。

## 已完成与尚未完成

- 官方RE10K、ACID权重各500场景测试，每场景3目标、100步目标相机对齐。
- 默认、去内参嵌入、去RGB捷径三配置分别在Torch/Jittor训练3000更新，共六模型；每模型24000次样本出现，实际涉及2819个不同训练场景。
- 两框架及三配置的6000个微批次输入记录相同；训练与测试场景ID不重合。
- 五组配对中，四组满足全部预设均值差目标。**默认短训PSNR差+0.342768dB超过原0.3dB阈值；历史严格梯度检查仍有629/1129项未通过。**
- 不宣称完整论文训练收敛、完整官方测试集、全部位姿AUC/多视图/512模型或所有消融已经复现。

## 如何运行

1. Linux配置Python3.10、Jittor1.3.11.0、C++/CMake/nvcc开发环境。Ada早期使用CUDA11.8；Blackwell扩展使用CUDA12.8并需文档中的兼容适配。
2. 按作者说明获取合法数据和官方权重，在**独立Torch环境**转换成不依赖Torch运行的NPZ。训练使用MASt3R过滤初始化，不把官方已训练NoPoSplat权重当作从头初始化。
3. 默认与去内参使用默认初始化；去RGB需要匹配的DPT属性头初始化。转换、目录、源计划与固定候选索引见RUNNING.md。
4. 完成tiny和原生渲染/梯度检查后，再执行完整网络；tiny不是论文模型。

```bash
bash scripts/train_reduced.sh jittor default
bash scripts/train_reduced.sh jittor no_intrinsics
bash scripts/train_reduced.sh jittor no_rgb
bash scripts/test_official.sh re10k
bash scripts/test_official.sh acid
```

这些入口需要先准备数据、权重和正确环境；完整参数与Torch参考命令见[运行指南](docs/RUNNING.md)。配置保留5000更新学习率周期，但显式在3000停止，不能把配置上限记为实际完成5000步。

## 从结果到证据

- [README三指标总表与实际曲线](README.md#measured-results)
- [六份训练日志](results/train)与[十组评估记录](results/evaluation)
- [公开核心源码SHA](docs/executed_core_sha256.json)：26份核心数学/桥接文件与实际服务器源码字节一致。
- [CPU发布检查](results/release_cpu_validation.json)：74项中71项通过；3项需要可选Torch档案测试依赖，在当前CPU解释器中明确跳过。它不替代GPU数值检查。

定性样例为事后高质量展示，不替代500场景总体指标。原始分辨率256，无锐化/超分；官方与自训训练预算不同；每模型独立优化目标相机。原固定子集存在RE10K43/1500、ACID32/1500个目标索引与输入重合，原结果保留并披露，展示目标排除这些重合项。

## 署名与使用边界

保留NoPoSplat、CroCo/DUSt3R/DPT、原可微高斯光栅化器、GLM和LPIPS的来源。项目**不是统一MIT，也不提供无限制商用授权**；派生组件的非商业条款和渲染器独立许可见[THIRD_PARTY.md](THIRD_PARTY.md)。原始数据、第三方预训练权重与大优化器断点不放在Git仓库中。

下一步优先固定相同权重与相机，分段定位网络、渲染与损失梯度差异，再做多种子和更大训练暴露；这些是后续计划，不作为当前已完成成果。
