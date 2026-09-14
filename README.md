# Paper Research

大模型 / 视频生成 / 物理 AI 论文调研笔记（中文），内容包括架构分析、创新点解读与论文配图。

## 目录

### 1. [MAGI-2 Preview](magi2-preview/)

Sand.ai 开源的全球首个千亿级（~114B）MoE 视频生成模型调研：

- [MAGI-2-Preview-架构与创新点详解.md](magi2-preview/MAGI-2-Preview-架构与创新点详解.md)
- 内容：超细粒度 MoE（MagiMoE）、Multi-Head LatentMoE + Head Parallel、分层 Head Parallel、MagiMuon 优化器、数据过滤陷阱
- 来源：官方技术博客 [*MAGI-2 Preview: Scaling Video Generation Models Efficiently*](https://sand.ai/blog/magi-2-preview)
- 配图：`magi2-preview/assets/`

### 2. [Cosmos 3](cosmos3/)

NVIDIA 面向物理 AI 的全模态世界模型（Omnimodal World Model）调研：

- [Cosmos-3-架构与创新点详解.md](cosmos3/Cosmos-3-架构与创新点详解.md)
- 内容：Mixture-of-Transformers（MoT）双塔架构、AR/DM 双子序列、3D MRoPE 绝对时间调制、统一动作表示、整流流多阶段训练
- 来源：论文 [*Cosmos 3: Omnimodal World Models for Physical AI*](https://arxiv.org/abs/2606.02800)（arXiv:2606.02800）与 [NVIDIA 官方博客](https://blogs.nvidia.cn/blog/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai/)
- 配图：`cosmos3/assets/`

### 3. [LingBot-World](lingbot-world/)

Robbyant 开源的交互式世界模拟器（源自视频生成）调研：

- [LingBot-World-架构与创新点详解.md](lingbot-world/LingBot-World-架构与创新点详解.md)
- 内容：三阶段演化训练（预训练→中期训练→后期训练）、按去噪步路由的双专家 MoE（28B/步激活 14B）、块因果注意力 + KV cache 流式生成、DMD+对抗蒸馏、涌现记忆能力、LingBot-World 2.0（Infinity）
- 来源：技术报告 [*Advancing Open-source World Models*](https://arxiv.org/abs/2601.20540)（arXiv:2601.20540）与 [*Infinite Worlds with Versatile Interactions*](https://arxiv.org/abs/2607.07534)
- 配图：`lingbot-world/assets/`

### 4. [LingBot-Video](lingbot-video/)

Robbyant 面向具身智能的稀疏 MoE 视频基础模型调研：

- [LingBot-Video-架构与创新点详解.md](lingbot-video/LingBot-Video-架构与创新点详解.md)
- 内容：任务统一单流 Diffusion Transformer、DeepSeekMoE 风格稀疏 MoE（13B/1.4B 激活，可扩至 120B）、级联 Refiner、六维奖励 GRPO、负样本感知微调、动作条件世界模型 A2V
- 来源：论文 [*Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence*](https://arxiv.org/abs/2607.07675)（arXiv:2607.07675）
- 配图：`lingbot-video/assets/`

### 5. [PDD](pdd/)

NVIDIA GenAI Research 的轨迹式蒸馏方法（并行解码蒸馏）调研：

- [PDD-架构与创新点详解.md](pdd/PDD-架构与创新点详解.md)
- 内容：并行解码（一次评估推进 L 个区间）、PD loss（纯回归 on-policy，无 VSD/GAN/JVP）、N 线性头训练 + 推理层融合、一个模型多档 NFE、data-free 训练、4–8 NFE SOTA 与多样性保留
- 来源：论文 [*Parallel Decoding Distillation for Fast Image and Video Generation*](https://arxiv.org/abs/2607.26004)（arXiv:2607.26004）与 [项目主页](https://research.nvidia.com/labs/genair/pdd/)
- 配图：`pdd/assets/`；**可交互演示网页（含效果视频）：`pdd/web/index.html`**（clone 后浏览器打开，离线可用）

### 6. [Atlas](atlas/)

World Labs 新一代全模态世界模型（空间智能）调研：

- [Atlas-架构与创新点详解.md](atlas/Atlas-架构与创新点详解.md)
- 内容：多模态自回归扩散 Transformer、空间上下文（Spatial Context，图像锚定 3D 位置）、相机位姿原生输入、稀疏视角重建 + 3D Gaussian Splats 输出、1440p/1 分钟长视频、Real-to-Sim 机器人仿真、缩放叙事；附两大基准（相机控制人评 75–94%、点图重建 AbsRel 领先 5 个专门模型）
- 来源：官方博客 [*Atlas: A World Model for Spatial Intelligence*](https://www.worldlabs.ai/blog/atlas)（2026-09-01，暂无论文）
- 配图：`atlas/assets/`

### 7. [FLUX 3](flux3/)

Black Forest Labs 多模态流匹配基础模型（"真实世界模型"）调研：

- [FLUX-3-架构与创新点详解.md](flux3/FLUX-3-架构与创新点详解.md)
- 内容：图像+视频+音频联合训练的统一 DiT 流匹配（视频占 >95% 算力、音频 <0.5% token）、Self-Flow 自监督流匹配（Dual-Timestep Scheduling 双时间步信息不对称，零外部模型超越 REPA）、操作成功率 42%→71%、FLUX-mimic 冻结骨干动作解码（部署 Audi、RTX 5090 <80ms）、20s 原生音频视频、发布矩阵（Video/Image/Action/Dev 开源）
- 来源：官方博客 [*FLUX 3 — Real World Models*](https://bfl.ai/blog/flux-3)（2026-07-23）、[Self-Flow 研究页](https://bfl.ai/research/self-flow)（Chefer & Esser，BFL+MIT）与 [*FLUX 3 × mimic*](https://bfl.ai/blog/flux-3-mimic)
- 配图：`flux3/assets/`

### 8. [SolarWM](solarwm/)

全开放的长时程视频世界模型基础栈（可重配置多源数据引擎 + backbone-native 适配框架）调研：

- [SolarWM-架构与创新点详解.md](solarwm/SolarWM-架构与创新点详解.md)（另有 [HTML 版](solarwm/SolarWM-架构与创新点详解.html)）
- 内容：**1,425,694** 条 canonical clips / 10 个源数据集 → 14 个可独立寻址 owner 的统一数据契约（metric-scale 相机标注 Pi3X+MoGe-2+VIPE、LTX-2.3 Clean Plate 543k clips、Kimi-K2.6 全量 caption 的 6 字段 JSON、source-aware 三档 quality tier）；4 个 5B–33B 模型的 native 保留清单（Wan2.2-5B/14B、LTX-2.5-22B、MiniMax-H3-33B）与 fused-PRoPE 统一相机条件化；三阶段配方（Bidirectional Adaptation → Teacher-Forced AnyFlow → DMD via Self-Gradient Forcing，省掉 Causal ODE / Causal CD 初始化）；**5 秒片段训练即可做分钟到 60 分钟级 rollout**，推理 16 fps / 4 步采样 / 无 attention sink
- 来源：论文 [*SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models*](https://arxiv.org/abs/2609.02886)（arXiv:2609.02886）与仓库 [github.com/Junchao-cs/SolarWM](https://github.com/Junchao-cs/SolarWM)
- 配图：`solarwm/assets/`

### 9. [Puffin-World](puffin-world/)

S-Lab NTU 的统一多模态 3D 世界模型（三种原生世界状态：physics / geometry / appearance）调研：

- [Puffin-World-架构与创新点详解.md](puffin-world/Puffin-World-架构与创新点详解.md)（另有 [HTML 版](puffin-world/Puffin-World-架构与创新点详解.html)）
- 内容：**9 通道 Omni-Camera 表示**（gravity-aware 绝对 perspective field + ray-based 相对几何）；**physics propagation**（把参考视角感知到的重力方向沿轨迹传播，消融：PSNR 19.87→20.54、roll 误差 1.94°→1.76°）；latent 空间像素对齐的相机条件注入 + 多视角 joint-attention（view-axis 1D RoPE，无需一致性 loss）；Hilbert 可逆颜色映射让深度复用同一个 frozen VAE；chunked 自回归长轨迹；四阶段训练配方；**Puffin-16M**（15,338,221 单视角三元组 + 1,136,696 条轨迹，≈11.6 TB）与 28 个公开数据集 / ≈44.5M 图像的绝对相机标注；实验数字（相机理解 12/12 最优 median error、Puffin-Cam-Bench FID 75.93、RealEstate10K PSNR 17.22 / LPIPS 0.318、Puffin-Traj-Bench roll 0.80°）
- 来源：论文 [*Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States*](https://arxiv.org/abs/2609.04196)（arXiv:2609.04196）与 [官方项目页](https://kangliao929.github.io/projects/puffin-world/)、[GitHub](https://github.com/KangLiao929/Puffin)
- 配图：`puffin-world/assets/`
