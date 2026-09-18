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

### 10. [Sol-Engine](sol-engine/)

NVIDIA `NVlabs/Sana` 仓库 **`sol-engine`** 分支（Sol Video Inference Engine）调研 —— 面向高分辨率视频 diffusion 的 **agent-native 全栈推理加速框架**：

- [Sol-Engine-架构与创新点详解.md](sol-engine/Sol-Engine-架构与创新点详解.md)（另有 [HTML 版](sol-engine/Sol-Engine-架构与创新点详解.html)）
- 内容：把视频 diffusion 推理的三层冗余（算法层跨步 / 模型层长序列 token 与 attention / kernel 层 memory-bound glue）对应到 **五类 training-free 技术**（Cache、NVFP4 量化、kernel fusion、稀疏注意力、token pruning），并组织成 **agentic 加速栈**（master orchestrator + 每技术一个 executor 子 agent + integrator，config 契约 + LPIPS / VLM / authenticity 三重质量门）；**Sol-Attn** 在线稀疏注意力（on-the-fly block thresholding + proxy score 复用，一次 online-softmax 内完成路由/稀疏/近似校正；CuTe DSL kernel 覆盖 SM89/90/100/120，另有 Triton 与 Metal 回退）；**11 条模型加速线全部数字**（Cosmos3-Super 64B ~2.27x、LTX-2.3 22B ~2.38x、SANA-Video 2B ~2.77x、Wan2.2-5B ~2.89x、Wan2.2-A14B ~2.17x、LingBot-Video 30B-A3B ~2.60x、HunyuanVideo-13B **~5.03x**、Wan2.1-14B ~3.48x、LTX-2.5 **4.68x** / 蒸馏 1.90x、MiniMax-H3 ~3.95x、H3 Super Acceleration **22.2x / 27.7x**）；四个旗舰案例（Sol-H3 在 8xB300 上 5 秒视频 **1.653 s** 的 1/4/8 卡 x 5/10/15 s 完整延迟表；Sol-H3-Spark 单张 DGX Spark **56 s** 出 768p 5 s 及延迟分解；H3 Super Acceleration 的双 GB200 latent/TCP handoff 与 benchmark 纪律；LTX-2.5 三种分辨率/时长的逐项累积）
- 来源：分支 [NVlabs/Sana · sol-engine](https://github.com/NVlabs/Sana/tree/sol-engine)（HEAD `8e0db4fa`）与论文 [Sol Video Inference Engine](https://arxiv.org/abs/2606.23743)（arXiv:2606.23743）、[Sol-Attn](https://arxiv.org/abs/2607.24027)（arXiv:2607.24027）
- 配图：`sol-engine/assets/`

### 11. [SenseNova-U1.5](sensenova-u1/)

商汤 SenseNova 的 8B-MoT **原生统一多模态模型**（无视觉编码器、无 VAE 的 NEO-unify 架构）调研：

- [SenseNova-U1.5-架构与创新点详解.html](sensenova-u1/SenseNova-U1.5-架构与创新点详解.html)
- 内容：近无损视觉接口（每 **32×32 像素 → 1 个视觉 token**，1024× 压缩）与 Native MoT（共享注意力做跨流通信、理解/生成各自保留注意力投影与 FFN、非对称掩码阻断干净表征看噪声状态）；U1.5 两大改造 —— **空间耦合解码器**（Pixel Shuffle 2/2/8 + 3×3 卷积取代逐 patch MLP 头，消除高分辨率接缝/网格伪影）与 **specialize-then-unify** 后训练；**4096²** 标定的分辨率感知噪声条件 s_t = τ_t + NSEmb(σ̄_R)；统一目标 **L = λ_AR·L_AR + λ_Flow·L_Flow + λ_Perc·LPIPS**（x-prediction 像素空间流匹配）；五阶段配方（180K+100K+185K 生成预训练 → 80K 统一中训 → 10.5K SFT → 四专家 RL → 800 步 MOPD，含 **Beta(2+3n/N, 5−3n/N)** 噪声调度与 CFG 速度场蒸馏）；奖励建模（HPSv3++、多重集 IoU 的 **OCR 奖励**、信息图奖励路由、编辑**五维瓶颈 min 聚合**）；数据（**+5,900 万**图文对 / 78 源、88.2% 有效训练量 >1024²、**3,800 万**编辑样本、28 万美学 prompt、12 万 DPO 偏好对）；评测（Qwen-Image-Bench EN **60.22** / ZH **60.13**（w/ PE）、GenEval **0.92** 开源最佳、CVTG-2K 平均 **0.948** 全场最佳、LongText EN 0.988 / ZH 0.989、DPG 88.11、**ImgEdit 4.59 为全表（含闭源）最高**、**VBVR-Pro 交错生成 68.2** 远超专用方案 40.8）；局限与部署（GGUF Q4 + `balanced` 面向 10–12 GB 单卡、LightLLM+LightX2V 单节点 2048² 约 **9 秒** / ~0.15 s/step）
- 来源：技术报告 [*SenseNova-U1.5: Towards Native Unified Visual Intelligence*](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/pdf/SenseNOVA_U1_5.pdf)（arXiv:2609.11929）、仓库 [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)、权重 [sensenova/SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)（Apache-2.0）、架构博客 [NEO-unify](https://huggingface.co/blog/sensenova/neo-unify)
- 配图：`sensenova-u1/assets/`；**可交互网页：`sensenova-u1/SenseNova-U1.5-架构与创新点详解.html`**（clone 后浏览器打开，含官方架构图、后训练流水线、数据构成与基准图）
