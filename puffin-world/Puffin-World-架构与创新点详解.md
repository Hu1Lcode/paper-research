# Puffin-World：架构与创新点详解

> 本文内容整理自论文 [*Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States*](https://arxiv.org/abs/2609.04196)（arXiv:2609.04196v1，2026.09）的 HTML 全文、[项目主页](https://kangliao929.github.io/projects/puffin-world/)、[HF 模型卡](https://huggingface.co/KangLiao/Puffin-World)、[HF 博客](https://huggingface.co/blog/KangLiao/puffin-world) 以及官方仓库 [KangLiao929/Puffin](https://github.com/KangLiao929/Puffin)（`Puffin-World/` 子项目，HEAD `c6d9ae5`）。数字取自论文表格与官方数据集卡。

---

## 目录

1. [模型概览](#1-模型概览)
2. [背景与痛点：三种被拆开的能力](#2-背景与痛点三种被拆开的能力)
3. [Omni-Camera：9 通道统一相机表示](#3-omni-camera9-通道统一相机表示)
4. [统一架构：三个层次的 unification](#4-统一架构三个层次的-unification)
5. [物理传播（Physics Propagation）](#5-物理传播physics-propagation)
6. [联合外观–几何建模](#6-联合外观几何建模)
7. [长时程探索：chunked 自回归](#7-长时程探索chunked-自回归)
8. [四阶段训练配方](#8-四阶段训练配方)
9. [数据引擎 Puffin-16M](#9-数据引擎-puffin-16m)
10. [模型家族与实验结果](#10-模型家族与实验结果)
11. [消融与闭环应用](#11-消融与闭环应用)
12. [已知局限](#12-已知局限)
13. [参考资料](#13-参考资料)

---

## 1. 模型概览

| 项目 | 内容 |
|---|---|
| 论文 | *Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States*（arXiv:2609.04196v1，2026-09） |
| 机构 | S-Lab, Nanyang Technological University；University of Michigan；Beijing Jiaotong University；ACE Robotics |
| 作者 | Kang Liao, Yihang Luo, Xiao-Ming Wu, Linyi Jin, Size Wu, Chunyu Lin, Yao Zhao, Fei Wang, Wei Li, Chen Change Loy |
| 定位 | **统一多模态 3D 世界模型**：物理理解 + 自由视角空间模拟 + 3D 世界生成与重建，**不依赖外部离线模块** |
| 三种 native world state | **physics**（gravity field + latitude map）/ **geometry**（depth）/ **appearance**（RGB） |
| 模型变体 | Puffin-World-**Base** / **Pro** / **Caption**（理解专用） |
| 数据 | **Puffin-16M**：15,338,221 单视角三元组 + 1,136,696 条轨迹（≈11.6 TB）；另覆盖 **28 个公开数据集 / ≈44.5M 图像**的绝对相机标注 |
| 开源 | 代码 + 权重 + 数据集 + 训练/评测脚本全部公开 |
| 许可 | **NTU S-Lab License 1.0（仅非商业）** |

---

## 2. 背景与痛点：三种被拆开的能力

已有的两条路线各自只覆盖一部分：

1. **生成式世界模型**几乎只在 **appearance 层**预测世界——把每帧当作 RGB 内容，**不显式表达相机物理朝向，也不表达场景几何**；控制信号多用相对相机运动（Plücker 等），**缺少全局物理参照**：同一相对轨迹可对应多个绝对朝向 → orientation drift，在挑战性运动、长时程、单视角设定下退化。
2. **统一多模态模型**把理解与生成放进一个网络，但只针对 **2D 语义与外观**，仍假定相机条件被简化。

结果：缺少一个同时统一"模态 + 任务"的 3D 世界模型。Puffin-World 针对三个紧耦合挑战给出答案：

- **(i) 统一动作表示**：既能连续做单视角 / 跨视角控制，又扎根于 gravity、uprightness、orientation 等**绝对物理概念**；
- **(ii) 物理持久的坐标系**：从已观测视角推断的知识必须能一致地指导未见视角的 3D 建模；
- **(iii) 数据规模化**：需要同时提供**绝对相机 grounding** 与**多样、挑战性运动**的数据。

---

## 3. Omni-Camera：9 通道统一相机表示

相对表示（跨视角关系）利于多视角与连续生成但**无全局锚**；绝对表示（地平线、竖直方向）有物理锚但难以从单目图像获得、也不便于跨视角过渡。Omni-Camera 把二者沿通道维拼接，对每个像素 $\mathbf{x}=(u,v)$：

$$\mathbf{c}_{\mathbf{x}}=\mathrm{Concat}\left(\mathbf{a}_{\mathbf{x}},\mathbf{p}_{\mathbf{x}}\right)\in\mathbb{R}^{9},\qquad \mathbf{a}_{\mathbf{x}}=\left(\mathbf{u}_{\mathbf{x}},\varphi_{\mathbf{x}}\right)\in\mathbb{R}^{3},\quad \mathbf{p}_{\mathbf{x}}\in\mathbb{R}^{6}$$

- **绝对分量** $\mathbf{a}$：逐像素 up-vector $\mathbf{u}_{\mathbf{x}}$（Perspective Field 定义，即反重力方向在像平面的投影）与 latitude 角 $\varphi_{\mathbf{x}}=\arcsin\bigl((-{\mathbf d}_{\mathbf{x}})^{\top}\mathbf{g}/\|\mathbf{d}_{\mathbf{x}}\|_2\bigr)$；$\mathbf{g}$ 为单位重力方向；
- **相对分量** $\mathbf{p}=(\mathbf{o},\mathbf{d}_{\mathbf{x}})$：ray origin（全视角共享，$\mathbf{o}=-\mathbf{R}^{\top}\mathbf{t}$）与逐像素方向 $\mathbf{d}_{\mathbf{x}}=\mathbf{R}^{\top}\mathbf{K}^{-1}\tilde{\mathbf{x}}/\|\mathbf{R}^{\top}\mathbf{K}^{-1}\tilde{\mathbf{x}}\|_2$。相对运动由 origin 变化（平移）与 direction 变化（旋转）表达。

**拼接顺序固定为绝对在前、相对在后**，得到与图像同分辨率的 $H\times W\times 9$ 稠密相机图 $\mathbf{C}$，作为贯穿理解、单视角生成、跨视角生成的统一条件。

---

## 4. 统一架构：三个层次的 unification

框架 = **geometry-aligned vision encoder + LLM + diffusion（MMDiT）+ 轻量 connector**，三层次统一：

1. **表示统一**：RGB 与深度**共享同一 VAE latent 空间**；所有相机配置由 Omni-Camera 表达；另有 **4 通道 role mask** $\mathbf{m}\in\mathbb{R}^{4\times H\times W}$ 标记每个视角是"生成目标 / 条件参考 / 图像条件输入 / 几何视角"。
2. **模态统一**：同一 backbone 既感知又生成——vision encoder + LLM 输出自回归理解结果，**同一批 LLM hidden states** 经 learnable queries 与轻量 connector 变换为 diffusion 条件。
3. **任务统一**：任务由"可用输入"决定（text、target cameras、reference views、role mask），改输入组合即可切换相机理解 / 相机可控 T2I / 3D 世界生成 / 外观-几何联合重建。

**相机条件注入方式（相对前作 Puffin 的主要改动）**：Puffin 把 3 通道 Perspective Field 经 VAE 编码后用 cross-attention 注入，扩展到更多任务与更复杂运动受限；Puffin-World 改为**在 diffusion latent 空间内直接相加**：

$$\mathbf{h}^{0}=\mathcal{P}\!\bigl(\mathbf{z}+\mathcal{F}([\mathbf{C};\mathbf{m}])\bigr),\qquad \mathbf{h}^{l}\leftarrow\mathbf{h}^{l}+\mathbf{W}_{l}\,\mathcal{P}\!\bigl(\mathcal{F}([\mathbf{C};\mathbf{m}])\bigr),\ \ l\in\mathcal{L}$$

$\mathcal{F}$ 零初始化起手（保持预训练生成器初始行为），并在**稀疏 transformer block 集合** $\mathcal{L}$ 上重注入，让每个 latent token 在多层都被精确的相机几何"钉住"，参数开销可忽略。

**多视角建模**：$T$ 个视角共用一条 joint-attention 序列；$K$ 个参考视角以 clean latent 提供、同时进 vision encoder + LLM 供语义条件；$T-K$ 个目标视角加噪后联合去噪（flow matching）。每个视角带**自己的 Omni-Camera 条件**，并分配 **view-axis index + 1D RoPE**——跨视角一致性由单次 attention 自然涌现，**不需要显式跨视角一致性 loss**。

---

## 5. 物理传播（Physics Propagation）

绝对 grounding 的难点：相对运动可从相机位姿直接得到，**而把轨迹锚到真实世界的绝对朝向通常拿不到**。Puffin-World 先用 Physics Perception 通路从参考视角感知绝对状态 $(\phi_0,\theta_0,\alpha_0,\dots)$（roll、pitch、垂直 FoV），得到参考系下的重力方向 $\mathbf{g}_0$；再用已知相对旋转 $\mathbf{R}^{\mathrm{rel}}_{t\leftarrow0}$ 传播：

$$\mathcal{G}=\left\{\mathbf{g}_{t}\right\}_{t=0}^{T-1}=\left\{\mathbf{R}^{\mathrm{rel}}_{t\leftarrow0}\,\mathbf{g}_{0}\right\}_{t=0}^{T-1}$$

各视角的 $\mathbf{g}_t$ 再按 Perspective Field 定义渲染出该视角的绝对相机表示，与相对 ray map 拼成完整 Omni-Camera 条件。于是**整条轨迹共享一个 gravity-anchored 世界坐标系**，同时保留各自的相对运动——这是"相对运动告诉你相机怎么动，物理传播还告诉你新相机相对真实世界如何朝向"的关键补充。

**物理理解本身**沿用 Puffin 的做法：把相机理解形式化为**自回归多模态序列建模**（而非直接回归）——vision encoder 提特征 → MLP projector 映到 LLM 空间 → LLM 先做场景与空间推理、再输出数值相机参数，用 next-token 交叉熵训练。好处是既能借 LLM 的序列建模与 scaling 行为，又让估计依赖地平线、竖直结构、前景构图等**整体场景线索**。

---

## 6. 联合外观–几何建模

- **深度编码**：把 depth 用受 3D Hilbert 曲线启发（Vision Banana）的**可逆颜色映射**编码为 RGB 图，用**与外观同一个 frozen VAE** 编码——得到有界三通道表示，**不改动预训练 VAE 的输入输出接口**；映射确定性且可逆，非线性变换还把更多色阶分配给近处表面（几何精度最重要的区域）。因此**无需专门的 depth encoder/decoder**。
- **联合目标**：每个建模视角的 depth latent 作为额外 token block 追加，与其 RGB 对应项**共享相机条件与 view-axis index**，仅靠 role mask 的 geometry 通道区分。单次去噪同时产出全部目标视角的外观与几何 latent：

$$\mathcal{L}_{\mathrm{fm}}(\mathcal{V})=\mathbb{E}_{t,\boldsymbol{\epsilon}}\Bigl[\tfrac{1}{|\mathcal{V}|}\sum_{v\in\mathcal{V}}\bigl\|\mathbf{v}_{\theta}(\mathbf{z}^{v}_{t},t,\mathbf{c}^{v})-(\boldsymbol{\epsilon}^{v}-\mathbf{z}^{v}_{0})\bigr\|_2^2\Bigr]$$

$$\mathcal{L}=\mathcal{L}_{\mathrm{fm}}(\mathcal{V}_{\mathrm{rgb}})+\omega(\tau)\,\mathcal{L}_{\mathrm{fm}}(\mathcal{V}_{\mathrm{geo}}),\qquad \omega(\tau)=\omega_{\max}\min\!\bigl(1,\tfrac{\tau}{\tau_0}\bigr)$$

外观与几何**分别在各自视角集合上求平均**（避免饱和色几何 latent 干扰已收敛的外观模型），几何权重从 0 线性爬坡。

- **两处"不伤害外观"的安全设计**：
  1. **asymmetric attention**：外观与文本 query **不能** attend 到 geometry token，而 geometry token 可以 attend 所有模态；
  2. **零初始化的 learnable geometry-modality embedding**（patch embedding 之后），使模型在初始化时与原多视角生成器功能等价。

推理时外观 latent 解码为图像，几何 latent 解码后从 Hilbert 颜色空间**逆映射回标量深度**。

---

## 7. 长时程探索：chunked 自回归

joint-attention 序列以固定视角数训练，但真实探索需要任意长度轨迹。做法是**分块自回归扩展**：生成一个 chunk 的视角后，把**最后一个生成视角作为下一 chunk 的参考视角**（滑动上下文窗口）。

关键设计：**相邻 chunk 在 latent 空间衔接**——下一 chunk 直接以被携带视角的**去噪 target latent**（而非重新 VAE 编码的像素图）为条件，从而避免每个 chunk 边界累积的 **VAE encode–decode 伪影**在长时程上漂移。此外 **physics propagation 跨整条序列（而非单个 chunk）施加**，使所有 chunk 共享同一个 gravity 对齐坐标系，长轨迹保持 self-calibrated 而不是累积朝向漂移。

---

## 8. 四阶段训练配方

| | Stage I 对齐 | Stage II SFT | Stage III 3D 世界建模 | Stage IV RGB + 深度 |
|---|---|---|---|---|
| 学习率 | 1e-4 | 2e-5 | 5e-5 | 2e-5 |
| Batch size | 512 | 512 | 256 | 128 |
| 可训练模块 | MLP projector、connector、condition tokenizer（VE/LLM/DiT 冻结） | 除 VAE 外全部解冻（**VE 梯度 ×0.1**） | connector、diffusion backbone、condition fusion（VE/LLM 冻结） | 同 Stage III + depth-modality embedding |
| 数据采样比 | Und. 0.5 / Gen. 0.5 | 0.5 / 0.5 | Gen. 0.14 / Image→3D 0.86 | Gen. 0.09 / Image→3D(RGB+Recon) 0.91 |

其他设定：AdamW、cosine scheduler、weight decay 0.05、betas (0.9, 0.95)；图像与深度由 **frozen VAE 编成 16 通道 latent**；单视角生成用 Puffin-Cam-15M 的原生分辨率与长宽比，3D 世界建模统一裁到 **640×640**；每条序列 **8 个视角**（参考视角 Stage III 随机 1–3、Stage IV 固定 1）；diffusion timestep 采 logit-normal（μ=0, σ=1）+ 分辨率相关 shift，Stage III/IV 用更大 shift 偏向高噪声（利于跨视角一致性）；Stage III 数据 = DL3DV + RealEstate10K（平移）+ Puffin-Traj-1M（旋转）+ Puffin-Cam-15M 单视角 rehearsal，**多源 repeat factor 3:2:1:1**；Stage IV 再加入 Hypersim、MVS-Synth、TartanAir、ScanNet。

**推理与 guidance**：CFG 通过条件 dropout 训练（text + 全部视角 Omni-Camera 联合置空 p=0.1；**绝对 Perspective Field 单独 drop p=0.15** 而保留相对 ray map，支持组件级 guidance）；推理时理解任务 greedy decoding，生成 **50 步采样**，CFG scale **4.5（单视角相机可控生成）/ 2.0（3D 世界建模）**。

---

## 9. 数据引擎 Puffin-16M

### 9.1 规模与来源

| 子集 | 规模 | 说明 |
|---|---|---|
| **Puffin-Cam-15M** | **15,338,221** 条单视角 image–caption–camera 三元组（≈2.4 TB，1,534 个 image shard × 10,000 张） | 由 **900K 全景图**渲染（Puffin-4M 只有 200K） |
| **Puffin-Traj-1M** | **1,136,696** 条轨迹，每条 **90 帧**（≈9.2 TB，每 shard 500 scene） | 挑战性旋转运动 |
| Puffin-Cam-15M-Bench | 600 组 image–caption–camera 对（≈100 MB） | held-out，与训练全景不重叠 |
| Puffin-Traj-1M-Bench | 100 条轨迹（≈1.1 GB） | held-out |
| 公开数据集标注 | **28 个数据集 / ≈44.5M 图像**的绝对 roll/pitch/vFoV | 22 个单图集 + 6 个多视角集 |

全景来源覆盖室内外、合成与真实、多国自驾街景；因 360° 标定与采集稳定性差异（消费级与车载尤甚），用**基于线段分割与消失点估计的几何校正**把全景对齐到重力方向。**Stanford2D3D 被排除**在源全景之外，以保证该 benchmark 的公平评测。

### 9.2 相机参数分布

| 参数 | Puffin-Cam-15M | Puffin-Traj-1M |
|---|---|---|
| Roll | U[−45°, +45°] | U[−45°, +45°]（初始；非运动轴时保持不变） |
| Pitch | U[−45°, +45°] | U[−45°, +45°]（同上） |
| Yaw | — | base ∈ [−180°, +180°]，运动轴时扫掠 |
| 垂直 FoV | U[20°, 105°] | U[60°, 100°]（整条轨迹内恒定） |
| 径向畸变 k₁ | 0（针孔模型） | 0 |
| 运动模式 | 单视角渲染 | 单轴运动 pitch/roll/yaw ×（mono_pos / mono_neg / bi_pos_neg / bi_neg_pos），**每段 90 帧、每帧 1°（约 90° 行程）** |

论文特别指出：既有 3D / 序列数据集（DL3DV、RealEstate10K、ScanNet、MVS-Synth、TartanAir、Hypersim）**roll 通常只有 [−5°,5°]、pitch 只有 [−10°,10°]**——因为多用手持采集，变化主要来自平移而非旋转；这直接限制了模型对挑战性相机运动的理解与模拟。

### 9.3 标注

- **Caption**：**Qwen3-VL-8B-Instruct 与 Qwen3-VL-32B-Instruct** 生成含场景语义与**结构化 chain-of-thought**（空间关系、相机参数、物理原理）的高质量 caption；另有 **Qwen3-VL-32B** 标注 **camera height** 五档：`Underwater shot` / `Low-position shot` / `Eye-level shot` / `High-position shot` / `Aerial shot`（高度数字仅为示意，无硬阈值），由全景继承到其所有透视图。
- **绝对相机标注**：用 Puffin-World 对**每个训练帧**估计绝对相机参数，导出 gravity field 与 latitude map 作为物理传播先验；对外的 28 个数据集标注（ImageNet-1K 1.33M、CC12M 10.90M、Megalith-10M 9.38M、Open Images 8.32M、GPIC 4.9M、Objects365 1.72M、SA-1B 897.6K、ADE20K 27.6K；多视角集 DL3DV 2.16M/6,377 场景、ScanNet 2.32M/1,468、RealEstate10K 595.4K/5,134、TartanAir 306.6K/18、HyperSim 73.6K/448、MVS-Synth）为重力感知感知与大尺度空间智能研究提供监督。
- **稠密几何标注**：合成集直接用官方稠密深度；真实集（ScanNet 有传感器孔洞与无效观测，DL3DV 的 MVS 深度不完整）用 **DA3（depth-anything/da3-large）**逐帧预测，**与已有稀疏深度标注对齐**（保留原始度量的可靠尺度、填补缺失区域）。

---

## 10. 模型家族与实验结果

### 10.1 三个 checkpoint

| Checkpoint | Vision encoder | LLM | Diffusion | 体积 | 用途 |
|---|---|---|---|---|---|
| `Puffin-World-Base.pth` | C-RADIOv3-H | Qwen2.5-7B-Instruct | SD3.5-Medium-2.5B | **26.7 GB** | 统一世界建模 |
| `Puffin-World-Pro.pth` | C-RADIOv4-H | Qwen2.5-1.5B-Instruct | SD3.5-Large-8.1B | **37.5 GB** | 更高质量生成 |
| `Puffin-World-Caption.pth` | C-RADIOv3-H | Qwen3.5-0.8B-Instruct | — | **2.82 GB** | 理解专用／大规模标注 |

两个世界模型变体都用 **64 个 learnable queries + 6 层 Transformer connector** 把 LLM hidden state 投到 diffusion 条件。

### 10.2 相机理解（Table 3）

在 Stanford2D3D、MegaDepth、TartanAir、LaMAR 四个 benchmark 上达成 **12/12 最优 median error、33/36 最优 AUC**：

| Dataset | 方法 | Roll 误差 ↓ | Pitch 误差 ↓ | vFoV 误差 ↓ |
|---|---|---|---|---|
| Stanford2D3D | GeoCalib（次优） | 0.40° | 0.93° | 3.21° |
| Stanford2D3D | **Puffin-World** | **0.29°**（AUC@1/5/10 = 93.1/97.4/98.5） | **0.53°**（73.3/88.8/94.0） | **1.62°**（34.5/61.1/76.4） |
| MegaDepth | Puffin（前作） | 0.32° | 1.08° | 2.42° |
| MegaDepth | **Puffin-World** | **0.28°** | **1.01°** | **2.41°** |
| TartanAir | **Puffin-World** | **0.31°** | **0.67°** | **2.34°** |
| LaMAR | **Puffin-World** | **0.26°** | **0.71°** | 2.73°（AnyCalib† 2.25°） |

### 10.3 相机可控生成（Table 4，Puffin-Cam-Bench = 600 组 text–camera 对）

单位为角度误差（mean / median）与 FID：

| 方法 | Up vector | Latitude | Gravity | FID ↓ |
|---|---|---|---|---|
| GPT Image2 | 24.13 / 23.44 | 16.42 / 15.98 | 28.83 / 29.38 | 99.36 |
| Nano Banana2 | 24.01 / 23.19 | 15.98 / 15.52 | 28.00 / 26.92 | 90.11 |
| Qwen-Image2-Pro | 24.12 / 23.51 | 16.45 / 15.53 | 28.57 / 28.69 | 98.51 |
| FLUX.2-dev | 24.28 / 23.62 | 15.75 / 15.24 | 28.31 / 28.36 | 97.06 |
| Z-Image | 24.64 / 24.04 | 16.96 / 16.09 | 29.41 / 28.82 | 98.33 |
| PreciseCam | 13.37 / 12.61 | 12.31 / 12.01 | 17.07 / 14.01 | 90.89 |
| Puffin（前作） | 3.86 / 3.48 | 4.77 / 4.43 | 4.92 / 2.87 | 80.29 |
| **Puffin-World** | **0.96 / 0.84** | **1.34 / 1.26** | **1.32 / 0.79** | **75.93** |

### 10.4 3D 世界建模（Table 5）

RealEstate10K（**50 条 held-out clip**，与训练无重叠）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| MotionCtrl | 13.38 | 0.482 | 0.507 |
| CameraCtrl | 16.93 | 0.589 | 0.352 |
| ViewCrafter | 16.04 | 0.569 | 0.331 |
| SEVA | 14.84 | 0.529 | 0.421 |
| MVGenMaster† | 17.11 | 0.591 | 0.348 |
| **Puffin-World** | **17.22** | **0.595** | **0.318** |

Puffin-Traj-Bench（**100 条 clip**，旋转与内参跨度更大；相机指标由同一套理解分支对生成帧重新估计）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Roll 误差 ↓ | Pitch 误差 ↓ | FoV 误差 ↓ |
|---|---|---|---|---|---|---|
| MotionCtrl | 12.53 | 0.438 | 0.604 | 3.20° | 2.15° | 10.04° |
| CameraCtrl | 14.28 | 0.512 | 0.532 | 11.79° | 10.81° | 45.84° |
| ViewCrafter | 14.52 | 0.525 | 0.510 | 1.92° | 2.89° | 6.43° |
| SEVA | 17.94 | **0.614** | 0.307 | 1.35° | 1.76° | 12.59° |
| MVGenMaster | 13.64 | 0.496 | 0.589 | 9.24° | 9.43° | 26.98° |
| **Puffin-World** | **18.00** | 0.613 | **0.288** | **0.80°** | **1.10°** | **2.96°** |

（† MVGenMaster 的训练数据包含所评测的 RealEstate10K 样本；RealEstate10K 无绝对相机 GT，故不报相机控制指标。Roll/Pitch AUC@1/5/10° 上 Puffin-World 为 0.57/0.79/0.88 与 0.45/0.72/0.83。）

---

## 11. 消融与闭环应用

**Physics Propagation 消融（Table 6，Puffin-Traj-Bench 的 roll/pitch/yaw 旋转轨迹）**：

| 运动 | 指标 | Baseline | + Physics Propagation |
|---|---|---|---|
| Roll | PSNR / SSIM / LPIPS | 22.97 / 0.69 / 0.12 | **24.02 / 0.74 / 0.10** |
| Pitch | PSNR / SSIM / LPIPS | 18.53 / 0.60 / 0.26 | **19.18 / 0.65 / 0.24** |
| Yaw | PSNR / SSIM / LPIPS | 18.12 / 0.59 / 0.30 | **18.42 / 0.62 / 0.30** |
| **平均** | PSNR / SSIM / LPIPS | 19.87 / 0.62 / 0.23 | **20.54 / 0.67 / 0.21** |
| **平均** | Roll 误差 / Pitch 误差 | 1.94° / 5.14° | **1.76° / 4.75°** |

结论：roll、pitch 直接与重力耦合，收益最大；yaw 绕重力轴旋转，图像增益小但仍在**抑制虚假地平线倾斜**上有效（pitch 误差 3.10°→2.67°）。

**闭环应用**：

- **Mimic world exploration**：在同一出发视角、同一指定相机轨迹下扩展出不同的 3D 世界，耦合相机理解与一致的多视角生成；
- **Self-calibrated world exploration**：从重力失准的观测出发，模型推理当前物理状态 → 预测纠正性相机动作 → 想象纠正后的观测，形成**感知–动作–生成闭环**（论文类比 World-Action Models），全程无需外部标定器或生成器。

项目页的交互式 3D 浏览器展示 appearance / gravity / latitude / depth 四态与点云重建（示例 **670K points**）。

---

## 12. 已知局限

论文 Conclusion **没有 limitation 章节**；官方 HF 模型卡给出的限制为：

- **当前聚焦静态场景**，physics 主要通过 **gravity 与 latitude** 建模；
- 动态环境、**更长的时序**、更丰富的物体交互与更广的物理状态是重要的后续方向；
- 生成内容可能**继承训练数据与预训练组件的偏见与失败模式**。

补充观察（整理者注）：

- 与视频世界模型相比，Puffin-World 处理的是**多视角 / chunked 自回归的静态世界扩展**，而非时序因果 rollout；论文未报告长序列的时间一致性指标；
- 相机控制评测依赖**自己的理解分支**对生成帧重估相机参数（自评闭环），缺少独立第三方标定器交叉验证；
- 数据集与权重均在 **NTU S-Lab License 1.0（非商业）** 下发布，商用受限。

---

## 13. 参考资料

- 论文：[*Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States*](https://arxiv.org/abs/2609.04196)（arXiv:2609.04196v1）· [HTML 全文](https://arxiv.org/html/2609.04196v1)
- 项目页：[kangliao929.github.io/projects/puffin-world](https://kangliao929.github.io/projects/puffin-world/) · [补充结果页](https://kangliao929.github.io/projects/puffin-world/supplementary.html) · [Puffin-16M 数据集页](https://kangliao929.github.io/projects/puffin-16m/)
- 代码：[github.com/KangLiao929/Puffin](https://github.com/KangLiao929/Puffin)（`Puffin-World/`，HEAD `c6d9ae5`）
- 模型与数据：[HF KangLiao/Puffin-World](https://huggingface.co/KangLiao/Puffin-World) · [HF KangLiao/Puffin-16M](https://huggingface.co/datasets/KangLiao/Puffin-16M) · [HF Blog](https://huggingface.co/blog/KangLiao/puffin-world)
- 前作：Puffin（ICLR 2026，[arXiv:2510.08673](https://arxiv.org/abs/2510.08673)）+ Puffin-4M（4M triplets）
- 引用格式：
  ```bibtex
  @article{liao2026puffinworld,
    title   = {Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States},
    author  = {Liao, Kang and Luo, Yihang and Wu, Xiao-Ming and Jin, Linyi and Wu, Size and Lin, Chunyu and Zhao, Yao and Wang, Fei and Li, Wei and Loy, Chen Change},
    journal = {arXiv preprint arXiv:2609.04196},
    year    = {2026}
  }
  ```

---

*本文档图片（`assets/` 下各文件）取自 Puffin-World 官方项目主页的示意图，版权归原作者所有。*
