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
