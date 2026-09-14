# Sol-Engine（Sol Video Inference Engine）：架构与创新点详解

> 本文内容整理自 NVIDIA [NVlabs/Sana](https://github.com/NVlabs/Sana) 仓库的 **`sol-engine`** 分支（HEAD `8e0db4fa`，2026-09-11）、分支内文档站 [Sol-Engine Docs](https://nvlabs.github.io/Sana/Sol-Engine/docs/)、各模型项目页（Sol-H3 / Sol-H3-Spark / LTX-2.5 / H3 Super Acceleration）与两篇论文：[*Sol Video Inference Engine*](https://arxiv.org/abs/2606.23743)（arXiv:2606.23743）与 [*Sol-Attn*](https://arxiv.org/abs/2607.24027)（arXiv:2607.24027）。数字取自 README、项目页与仓库内的 benchmark 记录。

---

## 目录

1. [项目概览](#1-项目概览)
2. [背景与痛点：加速配方是 instance-specific 的](#2-背景与痛点加速配方是-instance-specific-的)
3. [三类冗余与五类加速技术](#3-三类冗余与五类加速技术)
4. [Sol-Attn：在线稀疏化注意力（专章）](#4-sol-attn在线稀疏化注意力专章)
5. [Agent-native 工作流](#5-agent-native-工作流)
6. [模型矩阵与全部加速数字](#6-模型矩阵与全部加速数字)
7. [案例一：Sol-H3 on 8×B300](#7-案例一sol-h3-on-8b300)
8. [案例二：Sol-H3 on DGX Spark](#8-案例二sol-h3-on-dgx-spark)
9. [案例三：H3 Super Acceleration](#9-案例三h3-super-acceleration)
10. [案例四：LTX-2.5 多步 / 蒸馏双路径](#10-案例四ltx-25-多步--蒸馏双路径)
11. [仓库工程结构](#11-仓库工程结构)
12. [局限与注意事项](#12-局限与注意事项)
13. [参考资料](#13-参考资料)

---

## 1. 项目概览

| 项目 | 内容 |
|---|---|
| 名称 | **Sol Video Inference Engine（Sol-Engine）** |
| 代码位置 | [github.com/NVlabs/Sana](https://github.com/NVlabs/Sana) 的 **`sol-engine`** 分支（主仓库 Sana 的一个分支，不是独立仓库） |
| 快照 | HEAD `8e0db4fa562d727ea28b8d63015c196db7d97cae`，2026-09-11，commit message：*Announce Sol-H3 Spark and simplify setup documentation* |
| 团队 | NVIDIA SANA Team / Efficient AI Team & Singapore Lab（Yitong Li, Junsong Chen, Haopeng Li, Haozhe Liu, Jincheng Yu, Ligeng Zhu, Ping Luo, Song Han, Enze Xie） |
| 论文 | [arXiv:2606.23743](https://arxiv.org/abs/2606.23743)（框架）· [arXiv:2607.24027](https://arxiv.org/abs/2607.24027)（Sol-Attn） |
| 定位 | **面向高分辨率视频 diffusion 的效率型推理框架**：training-free、全栈加速、**agent-native 工作流** |
| 运行时底座 | 建立在 **SGLang 的 `multimodal_gen` runtime** 之上；部分 attention / token-prune backend 实现在 SGLang runtime 内，不在本仓库 |
| 覆盖模型 | Cosmos3-Super(64B)、LTX-2.3(22B)、LTX-2.5(22B)、SANA-Video(2B)、Wan2.2 TI2V-5B(5B)、Wan2.2-A14B(14B MoE)、Wan2.1-T2V-14B、LingBot-Video(30B-A3B MoE)、HunyuanVideo-13B、MiniMax-H3、MiniMax-H3 Super Acceleration |
| 加速幅度 | 单模型 **2.17×–5.03×**；MiniMax-H3 Super Acceleration 达 **22.2×（5 s）/ 27.7×（10 s）** |
| 许可 | 代码 Apache-2.0；权重沿用各自上游许可 |

![Sol-Engine](assets/sol-engine-logo.png)

## 2. 背景与痛点：加速配方是 instance-specific 的

论文（arXiv:2606.23743）把核心困难讲得很清楚：**最有效的加速策略高度依赖具体实例**——某个在特定「模型 + 硬件 + 推理配置」组合上有效的配方，换一个组合往往不成立：

- **模型侧**：架构、数值敏感度、attention 的集中程度各不相同；
- **推理配置侧**：空间/时间分辨率、视频时长差异很大；
- **硬件侧**：显存层级、支持的数值格式、kernel 吞吐都不一样。

三者组合出巨大的调参空间，而**人工性能工程极其昂贵**。Sol-Engine 的答案是把五类通用技术组织成一个 **agentic acceleration stack**：针对一个明确的部署目标（模型 + 硬件 + 服务配置），**并行 skill agent 各自优化一种技术的实现**，再由 **agent integrator** 合成为全局加速栈，**human validator** 对生成质量给出反馈。

## 3. 三类冗余与五类加速技术

框架把视频 diffusion 推理的冗余划分为三个层次，五类技术分别对应：

| 层次 | 冗余来源 | 对应技术 |
|---|---|---|
| **算法层** | 相邻去噪步在缓慢变化的 latent 上做结构相似的运算 | **Cache**（TeaCache / EasyCache / fixed-step） |
| **模型层** | 长时空序列中存在冗余 token 与冗余 attention 交互 | **Sparse attention**（PISA / Sol-Attn / SpargeAttn / SVG / SVG2）、**Token pruning**（feature-norm / ToMe-SD） |
| **Kernel 层** | DiT block 在 GEMM 周围反复启动 memory-bound 工作（布局搬移、归一化、激活、gate、精度转换） | **Quantization**（NVFP4 等）+ **Kernel fusion**（AdaLN/gate、GEMM epilogue、QK-norm+RoPE、QKV merge、torch.compile） |

| # | 方法族 | 仓库内已实现条目 | 关键工程细节 |
|---|---|---|---|
| 1 | **Cache** | TeaCache、EasyCache、fixed-step cache、TaylorSeer、Cache-DiT/DBCache | 搜索策略、跳步调度、缓存状态、补偿规则、warmup 与最大缓存步数；Cosmos3-Super 选 TeaCache，SANA-Video 选 EasyCache，LTX 用 stage-specific / First Block Cache |
| 2 | **Quantization** | NVFP4（ModelOpt 路径）、Diffusion PTQ、SVDQuant/Nunchaku、SageAttention、ModelOpt FP8 | **首尾去噪步保持 dense**（对数值误差最敏感），NVFP4 需 Blackwell + TransformerEngine，老卡自动回退 BF16 |
| 3 | **Kernel fusion** | AdaLN+残差 gate、GEMM epilogue、QK-norm+RoPE、attention 输出 gate、QKV merge、torch.compile | 各模型单独收益：**Wan-5B 1.52×**、**LingBot 1.79×**（cuDNN attention）、**Wan-14B 1.13×**；不改变 scheduler / prompt / seed / 步数 |
| 4 | **Sparse attention** | PISA、Sol-Attn、SpargeAttention、Sparse VideoGen、Sparse VideoGen2 | LTX-2.3 在 stage-2 refinement 用 PISA；Sol-Attn 用于 HunyuanVideo / Wan2.1-14B / MiniMax-H3 |
| 5 | **Token pruning** | Feature-norm pruning、ToMe-SD | LTX-2.3 在中间 refine 步裁掉低显著性 video token；需校验时序连贯性与末步保真 |

## 4. Sol-Attn：在线稀疏化注意力（专章）

**要解决的痛点**（arXiv:2607.24027）——既有 training-free 动态稀疏注意力有两个问题：

1. **路由僵硬、开销大**：按 proxy score 取固定比例 top-k block 会强加固定预算；按累积 proxy 概率质量取块则预算动态但可能失衡；两者都要计算并物化 proxy 分数，开销不可忽略。
2. **keep-or-drop 有损**：未选中的 block 被整体丢弃，激进稀疏下精度迅速下降。

**Sol-Attn 的做法**：把动态路由、稀疏计算与**近似校正**统一在**一次 online-softmax pass** 内：

- 为每个 K/V block 计算轻量 **proxy score**；
- 在 online softmax 过程中用 **on-the-fly block thresholding** 与阈值比较，决定哪些 block 精确计算；
- **未选中的 block 复用其 proxy score 近似其贡献**，而不是丢弃；
- 因此得到**动态且可控的 block 预算**，且**无需物化完整 routing map**。

**Kernel 与后端**：

| GPU | 架构 | 执行方式 |
|---|---|---|
| RTX 4090 | SM89 | CuTe DSL |
| H100 | SM90 | CuTe DSL（支持 split-KV） |
| B200 / GB200 | SM100 | CuTe DSL |
| RTX 5090 | SM120 | CuTe DSL |
| A100 | SM80 | Triton reference |
| 其他支持架构 | — | Triton reference |
| Apple M 系列 | Apple Silicon | Metal（需 `torch.mps.compile_shader`） |

**API 与限制**：`sol_attn(q, k, v, tau=1.0, thresh_type="exact")`，`tau` 越大精确计算的 KV block 越少；kernel **forward-only**，要求 contiguous BF16 Q/K/V、BTHD 布局、head dim = 128；支持 **exact KV sink**（以 64-token block 为粒度，常用于把 valid text K/V 钉成精确）；H100 支持 `kv_splits=1/2/4`，SM90 序列 ≥65,536 token 时自动选 split 4；`SOL_ATTN_STRICT=1` 让 kernel/集成错误直接报错而不是静默回落 dense。依赖 Python ≥3.10、PyTorch ≥2.10、CUDA ≥12.8、Triton ≥3.6。

**质量对照**（下图左为 dense refiner、右为 Sol-Attn refiner，来自 LTX-2.5 在 RTX 5090 的实验）：

![LTX-2.5 dense refiner](assets/ltx25-5090-dense.jpg)
![LTX-2.5 Sol-Attn refiner](assets/ltx25-5090-solattn.jpg)

## 5. Agent-native 工作流

Sol-Engine 的安装与调优**本身就是 agent 驱动的**：把一条 goal 交给编码 agent（OpenAI Codex / Claude Code），由它创建环境、拉取权重、跑 baseline 与 full-opt，并在遇到系统不兼容时**自行排障与改写脚本**。

仓库 README 给出的 goal 模板：

```text
/goal Execute the inference code for the three models using both baseline and full-opt
settings ... Refer to AGENTS.md for the environment creation, model download, and
inference guides. ... the provided guides may contain system incompatibilities, so you
are expected to troubleshoot and adapt them to your specific machine.
```

**编排模式**：

- **轻量模式**：一个 master orchestrator 监督至多 3 个 executor 子 agent（分别负责 kernel / cache / sparse-PISA），各自在自己的搜索空间里探索（带 watchdog 的 detached 执行），返回经过验证的 config，master 负责 gate、去重、集成；
- **完整框架**：每个技术一个 executor 节点，支持多轮深度搜索的更大 fan-out。

**Config 契约**：每个优化都是一份 config manifest —— 包含 id、kind（baseline / patch / control）、使用的 runtime、开启技术的环境变量、以及 eval profile。单一 launcher 渲染出可复现的 run bundle 并提交。**baseline 与同拓扑 control 是一等公民**，因此每个 speedup 都有明确写出的分母。

**质量门（不允许只看 wall-clock）**：

1. **LPIPS**：与冻结的 baseline 帧对比；
2. **多模态视觉门**：托管的 VLM 按 rubric 对 baseline vs config 并排评审（雪花/斑点、模糊、马赛克与 patch 边界、banding、ghosting、融化、时序闪烁、连贯性与运动退化）；
3. **真实性校验（authenticity check）**：确认技术确实生效（PISA 真的写了统计、cache 真的跳了步），防止"空转优化"报出假 speedup。

**约定**：report 必须写明 baseline 分母（single-GPU / same-topology / vs-naive），不得静默混用；计时**排除模型加载**、取官方验证 prompt 的**中位数**、**同 seed**；改变了浮点归约顺序或 attention 稀疏性的优化标注为 **non–bit-exact**。

**`search_space/` 六个方向**（面向 Codex goal 的"搜索空间契约"，明确不是固定超参网格）：`01_cache`、`02_token_pruning`、`03_quantization`、`04_sparse_attention`、`05_kernel_fusion`、`06_parallel_topology`（CP/SP、TP、EP、FSDP、CFG 执行、进程组、集合通信与计算重叠）。该目录由内部仓库 `cosmos_exp` ref 的一次快照导入（source commit `4049c2a0`，导入时间 2026-06-15）。

## 6. 模型矩阵与全部加速数字

| 模型 | 参数量 | 加速线 | Speedup |
|---|:---:|---|:---:|
| Cosmos3-Super（4×B200） | 64B | TeaCache + step-selective NVFP4 | **~2.27×** |
| LTX-2.3（1×B200） | 22B | kernel fusion + cache + PISA + NVFP4 + token-prune | **~2.38×**（docs 记 2.40×） |
| SANA-Video（1×B200） | 2B | EasyCache + kernel fusion + compile | **~2.77×** |
| Wan2.2 TI2V-5B（1×GB200） | 5B | EasyCache + kernel fusion + compile | **~2.89×** |
| Wan2.2-A14B（1×GB200） | 14B | kernel fusion + EasyCache + PISA | **~2.17×** |
| LingBot-Video（4×GB200） | 30B-A3B | kernel fusion + refiner PISA + EasyCache | **~2.60×** |
| HunyuanVideo-13B（1 GPU） | 13B | kernel fusion + TeaCache + **Sol-Attn** | **~5.03×** |
| Wan2.1-T2V-14B（1 GPU） | 14B | kernel fusion + EasyCache + **Sol-Attn** | **~3.48×** |
| LTX-2.5（4×B200） | 22B | parallelism + FBCache + kernel fusion | **~4.68×**（distilled 1.90×） |
| MiniMax-H3（8×GB200） | — | context parallel + kernel fusion + **Sol-Attn** + FirstBlockCache | **~3.95×** |
| **MiniMax-H3 Super Acceleration** | — | 4-step H3 draft + 3-step LTX-2.5 refine + Sol-Attn + fusion | **22.2×（5 s）/ 27.7×（10 s）** |

**硬件平台覆盖**（MiniMax-H3 系列，end-to-end）：

| 平台 | 加速 |
|---|---|
| 8×GB200 | **3.95×** |
| DGX Spark（GB10） | **3.92×** |
| GeForce RTX 5090 | **4.52×** |
| GeForce RTX 4090 | **4.44×**（含 SM89 CuTe DSL kernel） |
| 4×H100 | **3.56×** |
| 4×A100 | **3.55×** |

> 论文摘要口径：在 64B Cosmos3-Super、22B LTX-2.3、2B SANA-Video 三个不同规模与架构的模型上，全部为 **training-free** 加速；README 强调 speedup = warm end-to-end latency vs 未优化的**同拓扑** baseline，且模型/prompt/seed/分辨率/步数/batch/GPU 拓扑全部固定。

## 7. 案例一：Sol-H3 on 8×B300

![Sol-H3 on 8×B300](assets/sol-h3-b300-poster.jpg)

**结论句**：Sol-H3 在单台 8×NVIDIA B300 上**比实时播放更快**地生成带原生音频的 MiniMax-H3 视频——5 秒视频 **1.653 s** 出片。

**完整测量表**（1344×768、24 FPS、同一 prompt 与 seed `20260903`、reference-free T2VA + 立体声；Base H3 用 50 scheduler points / 49 次 DiT forward，Sol-H3 只用 **4 次 DiT forward**；数值为 1 次 warmup 后 3 次运行的**中位数**；计时含文本编码、DiT 去噪、视频/音频 VAE 解码，**不含**模型加载、编译预热与最终 MP4 编码）：

| GPU 数 | 输出时长 | Base H3（dense, 50 步） | Sol-H3（4 步） | 加速 |
|---|---|---|---|---|
| 1×B300 | 5 s | 129.898 s | **13.745 s** | 9.45× |
| 1×B300 | 10 s | 376.942 s | **37.813 s** | 9.97× |
| 1×B300 | 15 s | 746.885 s | **52.260 s** | 14.29× |
| 4×B300 | 5 s | 35.328 s | **2.918 s** | 12.11× |
| 4×B300 | 10 s | 100.440 s | **6.993 s** | 14.36× |
| 4×B300 | 15 s | 194.930 s | **12.542 s** | 15.54× |
| 8×B300 | 5 s | 18.250 s | **1.653 s** | 11.04× |
| 8×B300 | 10 s | 50.660 s | **3.732 s** | 13.57× |
| 8×B300 | 15 s | 99.513 s | **6.612 s** | 15.05× |

补充：8×B300 上 warm BF16-compute 的 T2V 延迟为 **1.653 / 3.732 / 6.612 s**（5/10/15 s 输出），可选 fused **MXFP8** 计算把 5 秒档降到 **1.472 s**；1×B300 用 Dense attention，4×/8×B300 用 Sol-Attn，并配合 **INT8 QKV / FP8 output transport**。

**一个具体的 kernel 级优化**：每个 block 直接读取自己的 cached slice，省掉两次 index 选择——跨 50 个 block × 4 步，**减少约 400 次小 kernel launch，且每步避免约 26 GB 的显存读取**，输出不变。

## 8. 案例二：Sol-H3 on DGX Spark

![Sol-H3 on DGX Spark](assets/sol-h3-spark-cover.jpg)

**结论句**：在**单张 DGX Spark** 上，从 prompt 到可播放的 **1344×768、5 秒、24 FPS**（含原生 H3 音频）**56 秒**完成（T2VA/FL2VA；Ref2VA 因 token 更多稍慢）。

**两阶段流水线**（默认 T2VA 配方，逐项固定）：

| 环节 | 配置 |
|---|---|
| Prompt 编码 | 常驻 NVFP4 AWQ Qwen；每次请求重新编码用户 prompt |
| Draft 生成 | MiniMax-H3 + FastH3 VSA DataFree LoRA + **W8A8 FP8 DiT**；4 次更新 @ 672×384×124 |
| Draft attention | **VSA，90% video sparsity，64-token block**；优化过的 cuDNN BSA backend |
| Latent 传递 | 学到的 **H3 ×2 upscaler** → **H3-to-LTX VAE latent adapter**（替代中间的 H3 VAE 解码 + LTX VAE 编码） |
| Refinement | BF16 **LTX-2.5 dev + distilled LoRA 450（strength 0.8）**；3 次联合音视频更新 |
| Refiner attention | **Triton Sol-Attn，step 阈值 1 / 1.25 / 1.5**，第一层 dense |
| 输出 | 官方 Conv VideoVAE → H.264 视频 + 原始 H3 音频（立体声 AAC） |

**延迟分解（56.17 s 的构成）**：LTX 3-step refinement **44.0%** > H3 4-step generation **35.2%** > LTX VAE decode **12.5%** > H3 upscaler + VAE adapter 3.8% > Qwen prompt encoding 2.4% > MP4 输出与音频 mux 1.3% > 请求开销 0.9%。

**同平台对比**：4-step LoRA 374 s → 4-step LoRA + 量化 153 s → **Sol-H3 56 s（6.7× faster）**。

**工程要点**：量化后的 Stage1 权重 + 仅 latent 传递 + 移除在线 Stage2 文本模型，使必需模型能在请求之间**常驻**；另外复用编译后的区域、融合的 VSA merge、NHWC Conv 上采样与分块写视频——**都不改变配方本身**。缓存的是通用 refinement prompt 与 INT8 Gemma/connector 的离线上下文，属"质量/性能折中"而非"身份不变"保证。

![Spark 生成样例](assets/spark-seaside-coffee.jpg)

## 9. 案例三：H3 Super Acceleration

**组合式两阶段**：Stage 1 用 4-step H3（896×512、124 帧、24 FPS，LightX2V 四步 LoRA 启动时合并）出 draft；Stage 2 用 LTX-2.5 三阶段 refine（x2 latent upsampler、原 VAE 高分辨率首帧条件、原 Audio VAE）。实测 **768p 5 秒 6.852 s、10 秒 14.931 s**，相对已发布的 SGLang baseline 为 **22.2× / 27.7×**。

**工程细节**：两个阶段常驻**两张 GB200**（一次请求占 2 卡；4 卡验证跑两个独立 pair 并发，**不是** 4 路 TP/CP/模型并行）；handoff 走**认证的 loopback TCP**而非中间 MP4，传输 contiguous BF16 视频张量 `[1,3,121,384,672]` 与 FP32 立体声 32 kHz PCM `[1,2,161333]`，Stage 1 直到 Stage 2 确认两个张量已驻留其 CUDA 设备才返回；Stage 2 在 sigma 点 `0.909375, 0.725, 0.421875, 0.0` 做 3 次联合更新，LTX transformer **layer 0 保持 dense，layer 1–47 用严格 Sol-Attn（τ = 1.0 / 1.25 / 1.5）**，cross-attention 与 audio attention 全 dense，只编译 48 个 block。

**benchmark 纪律**（很值得学习的一段）：formal Slurm job `6304303` 完成 20 次热请求（两个 pair 各 10 次），排除模型加载、编译预热与每阶段一次完整 warmup，跨 pair 中位数：**H3 Stage1 → 最终 Stage2 MP4 6.7605 s**（Stage1 wall 4.2919 s、Stage2 常驻服务 2.4469 s）。仓库明确声明：**该 v2 profile 没有配对的端到端 baseline，因此不宣称 speedup；没有跑感知质量门，因此"媒体/遥测校验通过"不得被描述为"质量通过"**。

**生产规模估算**（理想满利用率）：单张 GB200 连续推理 ≈ **12.6K 条 5 秒 / 12.6 h 成片每天**（30 天 378K 条 / 525 h），10 秒档 ≈ 5.79K 条/天（30 天 174K 条 / 482 h）。

![H3 Super Acceleration：参考图](assets/h3super-reference-768p.jpg)
![H3 Super Acceleration：Sol-Attn refiner 输出](assets/h3super-solattn-refiner-768p.jpg)
*上：参考图；下：Sol-Attn refiner 输出（768p / 5 s）。该路径**改变了采样轨迹**，与 SGLang baseline **非 bit-exact**，细节/纹理/运动/音频可能有差异——论文与项目页都明确把这一点摆出来。*

## 10. 案例四：LTX-2.5 多步 / 蒸馏双路径

LTX-2.5 是 22B 的**音视频联合** diffusion transformer（一条 transformer、两个 stage）。两条推理路径：

| 维度 | Multi-step Pipeline | Distilled Pipeline |
|---|---|---|
| Stage 1 步数 | 30 | 8 |
| Stage 2 步数 | 2 | 3 |
| CFG | 有 | 无 |
| 精度 | BF16 / INT8 | BF16 / INT8 / **NVFP4** |

**多步路径在 4×B200（BF16，30 步 Stage 1 + 2 步 Stage 2）的逐项累积**：

| 配置 | Baseline（SP） | + Parallel Optimization | + FBCache | + Kernel Fusion | 总加速 |
|---|---|---|---|---|---|
| 1536×1024 · 5 s | 39.23 s | 14.14 s（2.77×） | 11.09 s（1.28×） | **8.38 s**（1.32×） | **4.68×** |
| 4K（3840×2176）· 5 s | 75.57 s | 63.66 s（1.19×） | 46.29 s（1.38×） | **38.49 s**（1.20×） | **1.96×** |
| 1080p（1920×1088）· 20 s | 92.45 s | 59.65 s（1.55×） | 44.71 s（1.33×） | **36.69 s**（1.22×） | **2.52×** |

三条优化的分工：**Diffusion Cache**（First Block Cache，复用稳定残差、跳过不必要的 forward，Stage 1 性能提升近 50%）；**Parallelism**（用 4 路 CFG 分支分摊 guidance 负载，而非 sequence parallelism——第一段 token 序列较短，SP 的通信开销过大）；**Kernel**（cache-aware 编译 + 算子融合，Block 0 作缓存决策点，cache miss 才执行其后已编译的 block）。蒸馏路径在单卡（B200 / RTX 5090 / DGX Spark）上报告 **1.90×**。

## 11. 仓库工程结构

| 维度 | 数据 |
|---|---|
| 分支文件 | 1,448 个文件、约 45.7 MB（GitHub tree API 口径）；shallow clone 后约 64 MB |
| Python 代码 | **862 个 `.py` / 215,443 行** |
| Config manifest | `config/` 下 **90 个 `.toml`**，按模型分子目录（`cosmos3`、`ltx23`、`minimax_h3`、`wan22_ti2v_5b`、`wan22_t2v_a14b`、`lingbot_video`、`sana_video`、`hunyuan_video`、`wan21_t2v_14b` 等）+ 技术维度的 `sparse_attention` / `step_cache` / `token_prune` / `nvfp4_ffn` / `kwl_fusion` |
| 模型运行时 | `models/<model>/`（如 `models/minimax_h3/` 下按硬件分 `GB200` / `H100` / `A100` / `GB10` / `RTX4090` / `RTX5090` / `Sol-H3` / `Sol-H3-Spark` / `super_acceleration`（B300 结果体现在项目页与 config，未单独建目录）） |
| 技术实现 | `techniques/`：`methods/`（teacache、step_cache、payload_cache、token_prune）、`transforms/`（kwl_fusions、nvfp4_ffn、sparse_attention）、`compose.py`、`config_manifest.py`、`spec.py`、`registry.py`、`schedule.py`、`selftest.py`；`techniques/sparse_backends/sol_attn/` 含 CuTe kernel、Triton 参考实现与 `mps/` Metal 后端 |
| 统一入口 | `python3 scripts/run.py config/<model>/<arm>.toml`（`--print` 只做解析与路径检查；`--set KEY=VALUE` 做单次覆盖） |
| 运行产物 | `runs/<timestamp>-<id>/`：`launch.sh`、`manifest.resolved.toml`、`metadata.json`、`outputs/`（视频 + `benchmark.json`，后者记录实际选中的 backend） |
| 环境 | CUDA 13（torch 2.11+cu130）、Python 3.12、diffusers 0.38；Cosmos3-Super(64B) 需 4 GPU，SANA-Video / LTX-2.3 单卡可跑 |

## 12. 局限与注意事项

- **加速不是无损**：Sol-Engine 明确标注改变浮点归约顺序或 attention 稀疏性的优化为 **non–bit-exact**；H3 Super Acceleration 直接说明"这不是无损加速，采样路径已改变"；
- **速度数字有严格分母**：所有 speedup 都对应明确 baseline（single-GPU / same-topology / vs-naive），计时排除模型加载、取中位数、同 seed；不同分母不可混用；
- **质量门不等于质量结论**：H3 Super Acceleration 的 v2 profile 因缺少配对 baseline 与感知质量门，**主动声明不宣称 speedup、不宣称质量通过**——这类自我限制在工程博客里并不常见；
- **后端/硬件依赖**：NVFP4 需要 Blackwell + TransformerEngine（否则回退 BF16）；Sol-Attn 的发布 speedup 基于 CuTe DSL kernel，A100 等回退到 Triton reference（正确但非发布口径）；Metal 后端需 `torch.mps.compile_shader`；
- **Kernel 限制**：Sol-Attn kernel 目前 **forward-only**，要求 contiguous BF16、BTHD、head dim 128；
- **工作流仍需人**：README 的 To-do 里 "agent-native workflow without human-in-the-loop" **尚未完成**；论文也把 human validator 作为流程一环；
- **部分实现不在本仓库**：attention / token-prune 的具体 backend 位于 SGLang 的 `multimodal_gen` runtime，复现需同时克隆上游；`search_space/` 是从内部仓库导入的快照。

## 13. 参考资料

- 仓库分支：[NVlabs/Sana · `sol-engine`](https://github.com/NVlabs/Sana/tree/sol-engine)（HEAD `8e0db4fa`，2026-09-11）
- 论文：[*Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation*](https://arxiv.org/abs/2606.23743)（arXiv:2606.23743）· [*Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification*](https://arxiv.org/abs/2607.24027)（arXiv:2607.24027）
- 文档与项目页：[Docs](https://nvlabs.github.io/Sana/Sol-Engine/docs/) · [Homepage](https://nvlabs.github.io/Sana/Sol-Engine/) · [Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) · [Sol-H3-Spark](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark/) · [LTX-2.5](https://nvlabs.github.io/Sana/Sol-Engine/LTX25/) · [H3 Super Acceleration](https://nvlabs.github.io/Sana/Sol-Engine/H3-Super-Acceleration/)
- 依赖：[SGLang](https://github.com/sgl-project/sglang) 的 `multimodal_gen` runtime、[Diffusers](https://github.com/huggingface/diffusers)、[TransformerEngine](https://github.com/NVIDIA/TransformerEngine)、[CUTLASS/CuTe DSL](https://github.com/NVIDIA/cutlass)
- 引用格式：
  ```bibtex
  @misc{li2026solvideoinferenceengine,
    title         = {Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation},
    author        = {Yitong Li and Junsong Chen and Haopeng Li and Haozhe Liu and Jincheng Yu and Ligeng Zhu and Ping Luo and Song Han and Enze Xie},
    year          = {2026},
    eprint        = {2606.23743},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    doi           = {10.48550/arXiv.2606.23743},
    url           = {https://arxiv.org/abs/2606.23743}
  }

  @article{li2026solattn,
    title   = {Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
    author  = {Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
    journal = {arXiv preprint arXiv:2607.24027},
    year    = {2026}
  }
  ```

---

*本文档图片（`assets/` 下各文件）取自 NVlabs/Sana `sol-engine` 分支仓库与官方项目页的示意图/静帧，版权归 NVIDIA 及原作者所有。*
