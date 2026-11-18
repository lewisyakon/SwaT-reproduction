# 子问题四：SwaT 正式结果汇总

> 更新日期：2026-11-13（UTC+8）  
> 数据来源：`/mnt/hd2/lewis/SwaTReproduction/outputs/subproblem4_original_recipe/evaluations/swat_full_cache_no_am_v1` 下 24 个正式评测条件的 `summary.json` 与 `predictions.jsonl`。  
> 本文只整理第一阶段 E7 的 SwaT（不加 AM）正式结果，不混入 Dense、LCF、LCF-X、T2T 或训练中间评测。当前发送的是完整 Receiver-format KV Cache；本文不是 SwaT+AM 压缩实验报告。

## 1. 结果范围与完整性

### 1.1 模型组

| 模型组 | Sender | Receiver |
|---|---|---|
| M1 | Qwen3-0.6B | Qwen3-1.7B |
| M2 | Qwen3-0.6B | Qwen3-4B |
| M3 | Qwen3-4B | Qwen3-0.6B |

### 1.2 完成情况

| 方法 | 每个模型组的条件 | 每个模型组样本数 | 三组总样本数 | 状态 |
|---|---:|---:|---:|---|
| SwaT（AM 关闭） | 8 个：GSM8K P1/P4、HotpotQA P1/P2/P3/P4、HumanEval P1/P4 | 26,619 | 79,857 | 全部完成 |

完整性核对结果如下：

- 24 个条件均存在 `summary.json` 和 `predictions.jsonl`，摘要完成条数、期望条数与逐样本文件行数一致。
- 79,857 条结果均重新读取核对；每个条件内的 `record_id` 无重复，逐样本质量均值与摘要一致。
- GSM8K 每个模型组为 `2 × 1,319` 条；HumanEval 为 `2 × 164` 条。
- HotpotQA P1/P3/P4 各 5,918 条；P2 各 5,899 条，另有 19 条不满足互补证据协议的样本按预定规则排除。
- 队列在 2026-11-13 03:30（UTC+8）记录 `completed_all pairs=M1,M2,M3 conditions_per_pair=8 compression=none`。

### 1.3 协议

| 协议 | Sender 初始输入 | SwaT 转换后 Receiver 继续处理的内容 | 测试范围 |
|---|---|---|---|
| P1：共享完整输入 | 完整问题与上下文 | SwaT Cache 接管相同完整输入，只追加 generation prompt，避免 Receiver 重复 Prefill | 三个数据集 |
| P2：互补证据 | 问题、一个支持段落和四个干扰段落 | 在转换 Cache 后 Prefill Receiver 侧的问题、另一支持段落和四个干扰段落 | HotpotQA |
| P3：完整证据在上游 | 完整问题与全部证据 | 在转换 Cache 后再次 Prefill Receiver 本地问题 | HotpotQA |
| P4：完整任务只在上游 | 完整任务 | 转换 Cache 后只追加不含题目信息的 generation prompt | 三个数据集 |

SwaT 用转换 Cache 建立接收端前缀，不把独立 Prefill 得到的本地完整输入 KV 再拼接一次。P1 因此采用 Cache takeover；P2/P3 才存在额外的 Receiver 本地后缀 Prefill。

## 2. 训练配置与转换器规模

### 2.1 训练配置

- 三组均采用本项目锁定的 SwaT 公开两阶段配方，Sender 与 Receiver 基座模型冻结。
- Renderer 保留 `D=1024`、`H=16`、`F=4096`；M1/M2/M3 的 block 数分别为 7/9/7。
- 保留双路输入、字符区间位置对齐、因果 cross-attention、Sender de-RoPE、Receiver RoPE、RMS 统计校准等实现。
- Phase 1 使用 PG-19/WikiText-103、QASPER、QuALITY、HotpotQA、MuSiQue，执行归一化 KV 重构训练 5,000 步，峰值学习率为 `3e-4`。
- Phase 2 沿用上述训练池，并加入 ARC-C、HellaSwag、WinoGrande、MMLU 和各 Receiver 自行生成、筛选的 GSM8K 推理轨迹；以接收端 NLL 为主要目标训练 24,000 步，学习率为 `3e-6`，同时保留公开配方中的 dense text continuation 批次。
- 三个测试任务和全部适用协议对每个模型方向复用同一份 step-24000 最终权重，没有按测试集分别挑选权重。

### 2.2 参数量、额外部署与权重

参数量使用十进制量级，`1 M = 10^6` 个参数；文件大小使用十进制 MB/GB。可训练总量只统计 SwaT renderer，不计冻结的 Sender/Receiver。SwaT 主路径位于上游，因此另外列出上游需要部署但不训练的 Receiver embedding 表。

| 模型组 | 可训练 Renderer | 上游额外冻结 Receiver embedding | Renderer 运行时 FP32 权重 | 含训练状态的 checkpoint | Phase 2 完成时间 |
|---|---:|---:|---:|---:|---|
| M1 | 237.04 M | 311.16 M | 948.14 MB | 2.845 GB | 2026-11-06 14:31 |
| M2 | 287.91 M | 388.96 M | 1.152 GB | 3.455 GB | 2026-11-06 23:42 |
| M3 | 252.76 M | 155.58 M | 1.011 GB | 3.033 GB | 2026-11-07 06:15 |

冻结 embedding 不是可训练转换器参数，但属于当前“上游完成接收端分词/embedding 与 KV 转换”部署方式的额外上游资源；Receiver tokenizer 与 RoPE 配置同样需部署，但不以参数量计。

## 3. 指标、单位与耗时口径

### 3.1 质量与置信区间

- GSM8K 使用最终答案 Exact Match；HotpotQA 使用 Answer EM 与 token F1；HumanEval 使用独立执行得到的 pass@1。
- 表中方括号为逐样本 95% 置信区间。二值 EM/pass@1 使用 Wilson 区间，HotpotQA F1 使用样本均值的正态近似区间并截断到 `[0, 1]`。
- 本文重新核对了全部逐样本质量字段；未使用训练 loss、验证集结果或其他方法结果补齐。

### 3.2 通信量

- “平均通信量”采用 `logical_transmission_bytes`，等于转换后的完整 BF16 Receiver-format KV Cache 加约 190 bytes 的 JSON 元数据。
- AM 在本轮完全关闭，`am_auxiliary_bytes=0`；没有选择或压缩 KV 条目，也没有报告 AM 保留率。
- 通信量使用十进制单位：`1 MB = 10^6 bytes`、`1 GB = 10^9 bytes`。本批单样本均值均小于 1 GB，表中统一显示为 MB。
- `source_original_kv_cache_bytes` 只用于记录原始源 Cache；实际发送量以转换后 Receiver Cache 与元数据之和为准。

### 3.3 分项耗时

表中不直接采用原始 E2E 字段，而按用户指定的三段结构使用 wall time 重新相加：

1. **上游** = 提示组装/分词 + Sender prefill；
2. **转换** = Sender Cache flatten/必要搬运 + SwaT renderer 转换及 Receiver Cache 构造；
3. **下游** = Receiver 本地 suffix/generation prompt prefill + Receiver decode；
4. **合计计算耗时** = 上游 + 转换 + 下游。

质量评分时间不属于模型推理，未加入下游；真实网络传输时间没有测量，也未加入合计。原始 E2E wall time 还包含少量 Python 调度与计时边界开销，为保持固定的“上游 + 转换 + 下游”结构，表中不使用它替代三段之和。小于 1 秒的耗时显示为 ms，达到或超过 1 秒的耗时显示为 s。

### 3.4 测量环境

- 设备：NVIDIA GeForce RTX 3090；batch size 1。
- Sender/Receiver 权重按 BF16 加载；renderer 参数以 FP32 加载，在转换前向中使用 BF16 autocast。
- Greedy decoding，`enable_thinking=False`；使用冻结提示词模板。
- GSM8K/HumanEval 最大生成 512 tokens，HotpotQA 最大生成 64 tokens；转换前缀最大长度配置为 8,192 tokens。
- CUDA 阶段同时记录 CUDA Event 与 wall time；本文三段求和使用 wall time，以便包含 CPU 提示处理和同步/调度开销。
- 上游、转换和下游在同一张 GPU 上顺序执行；未包含真实跨机网络传输。

## 4. SwaT（不加 AM）正式结果

| 组 | 数据集/协议 | 样本 | 质量（均值，95% CI） | 平均通信量 | 平均计算耗时（上游 + 转换 + 下游 = 合计） |
|---|---|---:|---|---:|---:|
| M1 | GSM8K/P1 | 1,319 | EM 30.93% [28.50, 33.48] | 14.830 MB | 29.21 ms + 25.88 ms + 2.56 s = 2.62 s |
| M1 | GSM8K/P4 | 1,319 | EM 30.93% [28.50, 33.48] | 14.830 MB | 30.57 ms + 27.44 ms + 2.73 s = 2.78 s |
| M1 | HotpotQA/P1 | 5,918 | EM 13.23% [12.39, 14.12]; F1 19.50% [18.57, 20.43] | 175.685 MB | 72.54 ms + 41.09 ms + 152.97 ms = 266.60 ms |
| M1 | HotpotQA/P2 | 5,899 | EM 33.84% [32.64, 35.05]; F1 46.63% [45.49, 47.78] | 94.547 MB | 43.29 ms + 28.56 ms + 212.12 ms = 283.97 ms |
| M1 | HotpotQA/P3 | 5,918 | EM 36.85% [35.63, 38.09]; F1 50.11% [48.97, 51.26] | 175.685 MB | 71.33 ms + 40.26 ms + 163.09 ms = 274.69 ms |
| M1 | HotpotQA/P4 | 5,918 | EM 13.23% [12.39, 14.12]; F1 19.50% [18.57, 20.43] | 175.685 MB | 71.75 ms + 40.58 ms + 149.25 ms = 261.59 ms |
| M1 | HumanEval/P1 | 164 | pass@1 0.00% [0.00, 2.29] | 27.092 MB | 45.97 ms + 28.99 ms + 826.91 ms = 901.87 ms |
| M1 | HumanEval/P4 | 164 | pass@1 0.00% [0.00, 2.29] | 27.092 MB | 43.48 ms + 26.95 ms + 793.18 ms = 863.61 ms |
| M2 | GSM8K/P1 | 1,319 | EM 68.61% [66.06, 71.06] | 19.067 MB | 29.01 ms + 30.44 ms + 6.27 s = 6.33 s |
| M2 | GSM8K/P4 | 1,319 | EM 68.61% [66.06, 71.06] | 19.067 MB | 27.78 ms + 29.38 ms + 5.96 s = 6.02 s |
| M2 | HotpotQA/P1 | 5,918 | EM 25.31% [24.22, 26.44]; F1 35.01% [33.90, 36.12] | 225.881 MB | 70.35 ms + 46.53 ms + 201.44 ms = 318.33 ms |
| M2 | HotpotQA/P2 | 5,899 | EM 43.40% [42.14, 44.67]; F1 57.57% [56.45, 58.70] | 121.561 MB | 42.04 ms + 31.89 ms + 333.55 ms = 407.48 ms |
| M2 | HotpotQA/P3 | 5,918 | EM 46.21% [44.95, 47.49]; F1 60.92% [59.82, 62.01] | 225.881 MB | 70.51 ms + 46.58 ms + 203.92 ms = 321.01 ms |
| M2 | HotpotQA/P4 | 5,918 | EM 25.31% [24.22, 26.44]; F1 35.01% [33.90, 36.12] | 225.881 MB | 69.82 ms + 46.30 ms + 199.59 ms = 315.72 ms |
| M2 | HumanEval/P1 | 164 | pass@1 0.00% [0.00, 2.29] | 34.833 MB | 42.15 ms + 30.43 ms + 370.09 ms = 442.67 ms |
| M2 | HumanEval/P4 | 164 | pass@1 0.00% [0.00, 2.29] | 34.833 MB | 44.25 ms + 30.14 ms + 369.42 ms = 443.81 ms |
| M3 | GSM8K/P1 | 1,319 | EM 37.45% [34.88, 40.10] | 14.830 MB | 43.63 ms + 26.96 ms + 3.02 s = 3.09 s |
| M3 | GSM8K/P4 | 1,319 | EM 37.45% [34.88, 40.10] | 14.830 MB | 38.69 ms + 26.59 ms + 3.04 s = 3.11 s |
| M3 | HotpotQA/P1 | 5,918 | EM 23.00% [21.94, 24.09]; F1 31.77% [30.68, 32.86] | 175.685 MB | 266.00 ms + 44.70 ms + 178.15 ms = 488.85 ms |
| M3 | HotpotQA/P2 | 5,899 | EM 28.17% [27.04, 29.34]; F1 39.90% [38.77, 41.03] | 94.547 MB | 150.72 ms + 31.66 ms + 176.08 ms = 358.46 ms |
| M3 | HotpotQA/P3 | 5,918 | EM 40.25% [39.01, 41.51]; F1 54.41% [53.28, 55.54] | 175.685 MB | 265.91 ms + 44.46 ms + 161.75 ms = 472.12 ms |
| M3 | HotpotQA/P4 | 5,918 | EM 23.00% [21.94, 24.09]; F1 31.77% [30.68, 32.86] | 175.685 MB | 266.11 ms + 44.65 ms + 178.72 ms = 489.48 ms |
| M3 | HumanEval/P1 | 164 | pass@1 0.61% [0.11, 3.37] | 27.092 MB | 61.37 ms + 31.40 ms + 2.54 s = 2.63 s |
| M3 | HumanEval/P4 | 164 | pass@1 0.61% [0.11, 3.37] | 27.092 MB | 61.79 ms + 30.18 ms + 2.49 s = 2.58 s |

## 5. 主要观察

1. **M2 的质量整体最好。** M2 的 GSM8K EM 为 68.61%；HotpotQA P1/P2/P3 的 F1 分别为 35.01%、57.57% 和 60.92%，均高于对应 M1/M3。Receiver 规模增大带来更高质量，但 M2 的转换器参数、通信量和 GSM8K 下游生成耗时也最大。
2. **P1 与 P4 的质量逐样本完全相同是当前协议实现的预期结果。** 对 SwaT 而言，两者都把完整任务转换成 Receiver Cache，随后只追加 generation prompt；P1 通过 Cache takeover 避免 Receiver 对相同完整输入重复 Prefill。因此这里的 P1/P4 不是两次信息内容不同的推理路径，不能把相同质量解释成模型对“Receiver 缺少问题文本”天然鲁棒。
3. **HotpotQA P2/P3 明显高于 P1。** 三组 P3 F1 为 50.11%/60.92%/54.41%，P2 为 46.63%/57.57%/39.90%，均高于相应 P1。P2/P3 在转换 Cache 后显式 Prefill Receiver 本地证据或问题，这既改变了信息可见方式，也为 Receiver 提供了本地 token 锚点；本轮结果只能说明该协议实现更有效，不能把差值完全归因于 Cache 转换质量。
4. **P2 在 HotpotQA 上降低约 46.2% 通信量。** M1/M3 从完整前缀的 175.685 MB 降到 94.547 MB，M2 从 225.881 MB 降到 121.561 MB，原因是 Sender 只转换互补证据中的源侧部分。该下降来自输入前缀变短，不是 AM 压缩。
5. **不加 AM 时 SwaT 仍发送完整 Receiver Cache。** 平均载荷为 14.830–225.881 MB；M2 因 Receiver Cache 层数更多，通信量约为对应 M1/M3 的 `36/28`。M1 与 M3 的通信量相同源于两者 Receiver 均为 28 层且 KV 结构一致，不表示 Sender 计算量相同。
6. **HumanEval 基本失效。** M1/M2 的 P1/P4 pass@1 均为 0，M3 仅为 0.61%。当前 Phase 2 训练池没有专门的代码生成数据，这与低结果一致，但本轮没有做训练数据消融，不能据此断言是唯一原因。
7. **耗时受模型方向和输出长度共同影响。** 转换阶段均值约为 25.88–46.58 ms；GSM8K 与 HumanEval 的合计通常由下游生成主导。M3 在 HotpotQA 上使用 4B Sender，上游约 151–266 ms，明显高于 M1/M2 的约 42–73 ms。通信量可作为网络成本代理，但没有带宽和时延测量时不能换算成真实端到端网络耗时。

## 6. 结果、权重与配置位置

- 正式评测根目录：`/mnt/hd2/lewis/SwaTReproduction/outputs/subproblem4_original_recipe/evaluations/swat_full_cache_no_am_v1`
- 队列状态：`/mnt/hd2/lewis/SwaTReproduction/outputs/subproblem4_original_recipe/evaluations/swat_full_cache_no_am_v1/queue.status`
- 三组训练与权重根目录：`/mnt/hd2/lewis/SwaTReproduction/outputs/subproblem4_original_recipe`
- 每组最终权重：`m{1,2,3}_*/stage2/exit-checkpoint-24000/renderer.pt`
- 每组实际训练配置：`m{1,2,3}_*/config.json`、`stage1/args.json`、`stage2/args.json`
- 统一评测实现：`/home/lewis/Subproblem4UnifiedReproduction/scripts/evaluate_swat_unified_no_am.py`
- 固定提示词包：`/home/lewis/DenseLatentCommunication/提示词模板.zip`

逐样本生成文本、质量、HumanEval 执行结果、输入/输出 token 数、原始与转换后 Cache 字节数、元数据字节数、Cache 形状、终止原因、各阶段 CUDA/wall 时间和峰值显存均保存在各条件的 `predictions.jsonl`。均值、P50、P95、范围、置信区间、软件环境和部署参数保存在对应 `summary.json`。本文只对正式完成结果做只读统计，没有启动、停止或修改任何 GPU 训练/测评进程。
