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
