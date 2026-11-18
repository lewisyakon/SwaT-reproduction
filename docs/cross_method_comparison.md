# 子问题四：Dense、SwaT 与 T2T 基线算术均值对比

> 更新日期：2026-11-17（Asia/Shanghai）  
> T2T 来源：`/mnt/hd2/lewis/t2t基线测试结果(更新).pdf` 的表 1（去格式质量）与表 3（通信量、分阶段计算时延算术均值）。  
> 本文只比较均在 NVIDIA GeForce RTX 3090 上完成正式计时的 Dense、SwaT 和 T2T；LCF/LCF-X 使用 RTX PRO 6000 Blackwell，因此不纳入时延横向比较。
> Dense 按统一语义保留 GSM8K/HumanEval P1/P4 和 HotpotQA/P1（P3 尚未完成）；SwaT 按实际信息路径比较：GSM8K/HumanEval 的 SwaT-P1（实际 P4）对 T2T-P4，HotpotQA 的 SwaT-P3 对 T2T-P3。

## 1. 统一口径

- 每个比较单元格均按“**方法 / T2T / ↑↓绝对差**”书写。`↑`/`↓`只表示数值增减；质量差为绝对百分点（pp），通信量与时延为相应单位的绝对差，不是相对百分比。
- 质量采用全测试集聚合值；T2T 使用更新版 PDF 表 1 的去格式后评分。Dense、SwaT 使用此前正式结果中的 EM、F1 或 pass@1。
- 通信量和计算时间全部使用**逐样本算术均值**。T2T 取更新版 PDF 表 3；Dense、SwaT 从正式逐样本记录重新计算，避免继续使用旧表的 P50。
- **传输时延口径**：固定按十进制 `10 MB/s = 10,000,000 B/s` 估算，逐样本使用 `通信字节数 / 10,000,000`，并直接加入“中间阶段”；不另列传输时延。该估算不含网络 RTT、协议栈、序列化、重传或带宽波动。
- **上游 Prefill**：Dense/SwaT 为 Sender 对源输入的 Prefill；T2T 为表 3 的上游本地 Prefill。
- **中间阶段**：Dense 为 adapter Cache 转换 + Receiver prefix Prefill + KV 传输；SwaT 为 Cache flatten/move + renderer 转换/Receiver Cache 构造 + Receiver suffix Prefill + KV 传输；T2T 为上游文本消息 Decode + 下游本地 Prefill + 新增文本 Prefill + 文本传输。
- **下游 Decode**：三种路径都只计 Receiver 最终答案的逐 token Decode。上游 Prefill、转换/Prefill 计算和下游 Decode 使用 CUDA Event 字段；中间阶段另外加入上述固定带宽传输估算。不含 CPU 提示组装、token selection、D2H、反分词及 Python 调度。
- T2T 通信量是生成文本 UTF-8 字节；Dense 是转换后完整 Receiver-compatible KV 张量载荷；SwaT 是完整 Receiver-format KV 加少量 JSON 元数据。
- **SwaT 协议映射**：SwaT 不在运行时自动识别任务，而由数据预处理显式构造 `(d,q)`。短任务把完整任务置于上游 `d`、令下游 `q` 为空；长上下文任务把上下文置于 `d`、问题置于下游 `q`。当前 SwaT-P1 对 GSM8K/HumanEval 执行 Cache takeover，Receiver 只追加 generation prompt，语义上等同 P4，因此对比 T2T-P4；HotpotQA 使用带本地问题 Prefill 的 P3，并对比 T2T-P3。HumanEval 是按短任务方式外推，不属于 SwaT 原文测试集。
- **Dense 协议筛选**：GSM8K/HumanEval 保留 P1/P4；HotpotQA 应保留 P1/P3，但 Dense/P3 尚未完成，所以当前只列 P1，不以 P4 代替。
- **Dense HotpotQA 补测替换**：Dense 的 HotpotQA/P1 使用 `dense_hotpotqa_no_prompt_limit_v1`，其 `max_prompt_tokens=null`、`prompt_truncation_enabled=false`，不再使用旧目录中截断到 512 token 的记录；三组平均 Sender prompt 均为 1,460.59 token。

> **样本边界：** `†` 表示 HotpotQA 的 Dense/SwaT 使用 5,918 条 bridge 子集，而 T2T 质量使用共同 7,345 条，因此质量差是描述性差值而非严格同题配对提升。T2T 表 3 的效率均值只使用成功且六项字段齐全的样本；个别条件有效题数略少，详见第 4 节。

### 1.1 转换器参数规模

参数量使用十进制量级，`1 M = 10^6` 个参数。表中“可训练转换模块”只统计 Dense adapter 或 SwaT renderer，不计冻结的 Sender/Receiver 基座模型。参数量与数据集和协议无关：同一模型组的所有结果复用同一份转换器权重。

| 模型组 | Dense 可训练 adapter | SwaT 可训练 renderer | SwaT 上游额外冻结 Receiver embedding（不计入转换器） |
|---|---:|---:|---:|
| M1（Qwen3-0.6B→1.7B） | 235.856 M（235,856,096） | 237.036 M（237,035,520） | 311.165 M（311,164,928） |
| M2（Qwen3-0.6B→4B） | 303.244 M（303,243,552） | 287.908 M（287,907,840） | 388.956 M（388,956,160） |
| M3（Qwen3-4B→0.6B） | 235.856 M（235,856,096） | 252.764 M（252,764,160） | 155.582 M（155,582,464） |

- Dense 的低维 latent 是 adapter 内部转换瓶颈；当前传输的是转换后的完整 Receiver-compatible KV，因此不能把 latent 维数视为通信载荷或额外参数模块。
- SwaT 的冻结 Receiver embedding 是当前上游执行 Receiver 分词/embedding 与 KV renderer 所需的额外部署资源，不参与训练，也不计入上表的 SwaT 转换器参数量；tokenizer 与 RoPE 配置不以参数量计。
- 本文第 3 节统计的是 **AM 关闭** 的 SwaT，因此这里只列 renderer。SwaT+AM 后续结果应把每个模型方向的 Query Transfer MLP 参数量单独列出，并同时给出 `renderer + Query Transfer` 的去重后可训练总量。

## 2. Dense 对 T2T（算术均值）

| 方法 | 模型组 | 数据集/协议 | n（质量：方法/T2T） | 质量：方法 / T2T / Δ | 通信量均值：方法 / T2T / Δ | 上游 Prefill 均值：方法 / T2T / Δ | 中间阶段均值（含传输@10 MB/s）：方法 / T2T / Δ | 下游 Decode 均值：方法 / T2T / Δ |
|---|---|---|---:|---|---:|---:|---:|---:|
| Dense | M1（Qwen3-0.6B→1.7B） | GSM8K/P1 | 1,319/1,319 | EM 72.71% / 37.83% / ↑34.88 pp | 10.555 MB / 377.2 B / ↑10.554 MB | 33.22 ms / 33.30 ms / ↓77.65 μs | 1.114 s / 2.436 s / ↓1.322 s | 4.612 s / 7.621 s / ↓3.009 s |
| Dense | M1（Qwen3-0.6B→1.7B） | GSM8K/P4 | 1,319/1,319 | EM 24.64% / 31.16% / ↓6.52 pp | 10.555 MB / 652.7 B / ↑10.554 MB | 28.41 ms / 34.10 ms / ↓5.69 ms | 1.114 s / 4.636 s / ↓3.522 s | 5.258 s / 16.322 s / ↓11.064 s |
| Dense | M1（Qwen3-0.6B→1.7B） | HotpotQA/P1 | 5,918/7,345† | EM 0.14% / 34.25% / ↓34.11 pp; F1 10.77% / 47.27% / ↓36.50 pp | 167.512 MB / 826.2 B / ↑167.511 MB | 55.27 ms / 51.30 ms / ↑3.97 ms | 16.937 s / 4.884 s / ↑12.053 s | 1.065 s / 191.60 ms / ↑873.78 ms |
| Dense | M1（Qwen3-0.6B→1.7B） | HumanEval/P1 | 164/164 | pass@1 50.00% / 55.49% / ↓5.49 pp | 17.688 MB / 1.129 kB / ↑17.687 MB | 36.58 ms / 37.00 ms / ↓420.96 μs | 1.850 s / 7.691 s / ↓5.841 s | 4.222 s / 1.844 s / ↑2.378 s |
| Dense | M1（Qwen3-0.6B→1.7B） | HumanEval/P4 | 164/164 | pass@1 0.00% / 40.85% / ↓40.85 pp | 17.688 MB / 862.9 B / ↑17.687 MB | 30.82 ms / 35.00 ms / ↓4.18 ms | 1.825 s / 5.127 s / ↓3.302 s | 9.377 s / 1.789 s / ↑7.588 s |
| Dense | M2（Qwen3-0.6B→4B） | GSM8K/P1 | 1,319/1,319 | EM 88.93% / 80.29% / ↑8.64 pp | 13.571 MB / 377.2 B / ↑13.570 MB | 31.92 ms / 34.50 ms / ↓2.58 ms | 1.441 s / 2.544 s / ↓1.103 s | 6.880 s / 7.400 s / ↓520.00 ms |
| Dense | M2（Qwen3-0.6B→4B） | GSM8K/P4 | 1,319/1,319 | EM 13.57% / 71.11% / ↓57.54 pp | 13.571 MB / 652.7 B / ↑13.570 MB | 27.01 ms / 32.90 ms / ↓5.89 ms | 1.425 s / 4.571 s / ↓3.146 s | 5.588 s / 5.569 s / ↑19.40 ms |
| Dense | M2（Qwen3-0.6B→4B） | HotpotQA/P1 | 5,918/7,345† | EM 0.02% / 34.88% / ↓34.86 pp; F1 11.13% / 50.11% / ↓38.98 pp | 215.372 MB / 759.2 B / ↑215.372 MB | 55.10 ms / 51.50 ms / ↑3.60 ms | 21.912 s / 4.484 s / ↑17.428 s | 1.845 s / 738.20 ms / ↑1.107 s |
| Dense | M2（Qwen3-0.6B→4B） | HumanEval/P1 | 164/164 | pass@1 61.59% / 71.34% / ↓9.75 pp | 22.741 MB / 1.129 kB / ↑22.740 MB | 29.94 ms / 34.60 ms / ↓4.66 ms | 2.369 s / 7.070 s / ↓4.701 s | 2.099 s / 3.929 s / ↓1.830 s |
| Dense | M2（Qwen3-0.6B→4B） | HumanEval/P4 | 164/164 | pass@1 0.00% / 48.78% / ↓48.78 pp | 22.741 MB / 862.9 B / ↑22.741 MB | 26.16 ms / 35.30 ms / ↓9.14 ms | 2.335 s / 4.824 s / ↓2.488 s | 13.633 s / 2.104 s / ↑11.529 s |
| Dense | M3（Qwen3-4B→0.6B） | GSM8K/P1 | 1,319/1,319 | EM 51.78% / 44.20% / ↑7.58 pp | 10.555 MB / 443.9 B / ↑10.554 MB | 34.35 ms / 44.00 ms / ↓9.65 ms | 1.103 s / 3.729 s / ↓2.626 s | 3.377 s / 3.464 s / ↓87.51 ms |
| Dense | M3（Qwen3-4B→0.6B） | GSM8K/P4 | 1,319/1,319 | EM 3.26% / 26.31% / ↓23.05 pp | 10.555 MB / 896 B / ↑10.554 MB | 33.75 ms / 43.70 ms / ↓9.95 ms | 1.102 s / 7.553 s / ↓6.451 s | 3.608 s / 2.446 s / ↑1.162 s |
| Dense | M3（Qwen3-4B→0.6B） | HotpotQA/P1 | 5,918/7,345† | EM 0.39% / 31.57% / ↓31.18 pp; F1 12.76% / 49.44% / ↓36.68 pp | 167.512 MB / 1.165 kB / ↑167.511 MB | 245.47 ms / 243.80 ms / ↑1.67 ms | 16.881 s / 8.856 s / ↑8.025 s | 679.97 ms / 321.00 ms / ↑358.97 ms |
| Dense | M3（Qwen3-4B→0.6B） | HumanEval/P1 | 164/164 | pass@1 26.22% / 53.05% / ↓26.83 pp | 17.688 MB / 2.050 kB / ↑17.686 MB | 41.81 ms / 52.30 ms / ↓10.49 ms | 1.817 s / 16.989 s / ↓15.172 s | 3.747 s / 2.495 s / ↑1.252 s |
| Dense | M3（Qwen3-4B→0.6B） | HumanEval/P4 | 164/164 | pass@1 0.00% / 47.56% / ↓47.56 pp | 17.688 MB / 2.035 kB / ↑17.686 MB | 41.84 ms / 66.80 ms / ↓24.96 ms | 1.817 s / 16.119 s / ↓14.303 s | 6.663 s / 1.914 s / ↑4.748 s |

## 3. SwaT（AM 关闭）对 T2T（算术均值）

| 方法 | 模型组 | 数据集/协议 | n（质量：方法/T2T） | 质量：方法 / T2T / Δ | 通信量均值：方法 / T2T / Δ | 上游 Prefill 均值：方法 / T2T / Δ | 中间阶段均值（含传输@10 MB/s）：方法 / T2T / Δ | 下游 Decode 均值：方法 / T2T / Δ |
|---|---|---|---:|---|---:|---:|---:|---:|
| SwaT | M1（Qwen3-0.6B→1.7B） | GSM8K/SwaT-P1（实际 P4）↔T2T-P4 | 1,319/1,319 | EM 30.93% / 31.16% / ↓0.23 pp | 14.830 MB / 652.7 B / ↑14.830 MB | 27.01 ms / 34.10 ms / ↓7.09 ms | 1.536 s / 4.636 s / ↓3.100 s | 2.374 s / 16.322 s / ↓13.948 s |
| SwaT | M1（Qwen3-0.6B→1.7B） | HotpotQA/P3 | 5,918/7,345† | EM 36.85% / 35.64% / ↑1.21 pp; F1 50.11% / 48.16% / ↑1.95 pp | 175.685 MB / 826.2 B / ↑175.685 MB | 57.02 ms / 56.20 ms / ↑815.28 μs | 17.639 s / 5.264 s / ↑12.375 s | 131.71 ms / 180.50 ms / ↓48.79 ms |
| SwaT | M1（Qwen3-0.6B→1.7B） | HumanEval/SwaT-P1（实际 P4）↔T2T-P4 | 164/164 | pass@1 0.00% / 40.85% / ↓40.85 pp | 27.092 MB / 862.9 B / ↑27.092 MB | 37.08 ms / 35.00 ms / ↑2.08 ms | 2.770 s / 5.127 s / ↓2.357 s | 787.47 ms / 1.789 s / ↓1.002 s |
| SwaT | M2（Qwen3-0.6B→4B） | GSM8K/SwaT-P1（实际 P4）↔T2T-P4 | 1,319/1,319 | EM 68.61% / 71.11% / ↓2.50 pp | 19.067 MB / 652.7 B / ↑19.067 MB | 26.82 ms / 32.90 ms / ↓6.08 ms | 1.973 s / 4.571 s / ↓2.598 s | 6.033 s / 5.569 s / ↑464.87 ms |
| SwaT | M2（Qwen3-0.6B→4B） | HotpotQA/P3 | 5,918/7,345† | EM 46.21% / 36.41% / ↑9.80 pp; F1 60.92% / 49.92% / ↑11.00 pp | 225.881 MB / 759.2 B / ↑225.880 MB | 56.77 ms / 51.70 ms / ↑5.07 ms | 22.672 s / 4.437 s / ↑18.234 s | 165.25 ms / 303.80 ms / ↓138.55 ms |
| SwaT | M2（Qwen3-0.6B→4B） | HumanEval/SwaT-P1（实际 P4）↔T2T-P4 | 164/164 | pass@1 0.00% / 48.78% / ↓48.78 pp | 34.833 MB / 862.9 B / ↑34.832 MB | 33.89 ms / 35.30 ms / ↓1.41 ms | 3.550 s / 4.824 s / ↓1.274 s | 332.00 ms / 2.104 s / ↓1.772 s |
| SwaT | M3（Qwen3-4B→0.6B） | GSM8K/SwaT-P1（实际 P4）↔T2T-P4 | 1,319/1,319 | EM 37.45% / 26.31% / ↑11.14 pp | 14.830 MB / 896 B / ↑14.829 MB | 39.59 ms / 43.70 ms / ↓4.11 ms | 1.537 s / 7.553 s / ↓6.016 s | 2.966 s / 2.446 s / ↑519.86 ms |
| SwaT | M3（Qwen3-4B→0.6B） | HotpotQA/P3 | 5,918/7,345† | EM 40.25% / 40.22% / ↑0.03 pp; F1 54.41% / 57.20% / ↓2.79 pp | 175.685 MB / 1.186 kB / ↑175.684 MB | 251.74 ms / 241.30 ms / ↑10.44 ms | 17.642 s / 9.062 s / ↑8.580 s | 131.54 ms / 218.10 ms / ↓86.56 ms |
| SwaT | M3（Qwen3-4B→0.6B） | HumanEval/SwaT-P1（实际 P4）↔T2T-P4 | 164/164 | pass@1 0.61% / 47.56% / ↓46.95 pp | 27.092 MB / 2.035 kB / ↑27.090 MB | 54.10 ms / 66.80 ms / ↓12.70 ms | 2.771 s / 16.119 s / ↓13.348 s | 2.485 s / 1.914 s / ↑570.51 ms |

## 4. T2T 表 3 的效率有效样本数例外

未列出的条件使用该数据集完整规模；以下条件因只统计成功且字段齐全样本而略少：

| 条件 | T2T 表 3 有效样本 |
|---|---:|
| M1 GSM8K/P1 | 1,318 |
| M2 GSM8K/P1 | 1,318 |
| M2 HotpotQA/P1 | 7,341 |
| M2 HotpotQA/P3 | 7,341 |
| M3 HotpotQA/P1 | 7,343 |
| M3 HotpotQA/P3 | 7,344 |

## 5. 结果解读

1. 在固定 10 MB/s 带宽下，Dense/SwaT 的中间阶段通常由完整 KV 传输主导，约为 1–23 秒；T2T 的文本传输近乎可忽略，其中间阶段仍主要由 Sender 自回归生成文本消息构成。原本几十毫秒级的 Cache 转换优势不等于包含网络后的中间阶段优势。
2. 下游 Decode 仍随输出长度、答案质量和有效 Cache 长度显著变化；本文的中间阶段已经包含固定带宽传输估算，但仍不是包含 RTT、序列化、重传和 CPU 编排的真实端到端墙钟时间。
3. Dense 在 GSM8K/P1 上提高质量，但多数 P4、HotpotQA/P1 和 HumanEval 条件低于 T2T；完整 KV 的通信量明显更高。Dense 的 HotpotQA/P3 尚未完成，当前不能评价其 `C -> q` 路径。
4. SwaT 的 GSM8K/HumanEval P1 在当前实现中是 P4 语义，故只与 T2T/P4 比较；不再报告不等价的 SwaT/P1 对 T2T/P1。HumanEval 质量基本失效。HotpotQA/P3 的 Sender 当前输入为 `q+C`，并非原文严格的仅 `C`，因此仍应表述为最接近原文信息流的统一协议结果。
5. 换用 Dense HotpotQA 无上限补测后，M1/M2/M3 的 Sender Prefill 分别为 55.27/55.10/245.47 ms，与 SwaT 的 57.57/56.78/251.79 ms 及 T2T 的 51.30/51.50/243.80 ms 基本一致；旧表中的大差距来自误用了 512-token 截断记录。
6. 本文已排除异卡计时的 LCF/LCF-X；HotpotQA 样本集合仍不一致，因此该数据集的质量差不能解释为严格同题收益。

## 6. 数据来源

- 更新版 T2T PDF：`/mnt/hd2/lewis/t2t基线测试结果(更新).pdf`
- Dense 汇总：`/home/lewis/文档/子问题四_Dense_P1_P4结果汇总.md`
- SwaT 汇总：`/home/lewis/文档/子问题四_SwaT正式结果汇总.md`
- Dense GSM8K/HumanEval 逐样本根目录：`/mnt/hd2/lewis/Subproblem4UnifiedReproduction/experiments/stage1/evaluations/dense_original_data_v1`
- Dense HotpotQA 无上限补测根目录：`/mnt/hd2/lewis/Subproblem4UnifiedReproduction/experiments/stage1/evaluations/dense_hotpotqa_no_prompt_limit_v1`
- SwaT 逐样本根目录：`/mnt/hd2/lewis/SwaTReproduction/outputs/subproblem4_original_recipe/evaluations/swat_full_cache_no_am_v1`
