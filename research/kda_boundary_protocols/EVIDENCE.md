# CP8 边界协议证据：可移植版验收与历史实测

本页分别记录 2026-10-30 普通模块可移植版的 NPU 迁移验收，以及 2026-10-29 的历史实测。**新的结果是相同 S/M/G 上的边界协议迁移验收，不是完整 KDA 或训练资格。** 历史数据与新数据保留各自来源 SHA256、样本数和采样窗口，不能合并为速度排名。

## 本次可移植版 NPU 迁移验收

新记录为 [portable_protocols_cp8_npu.json](evidence/portable_protocols_cp8_npu.json) 和 [portable_g2_cp8_npu.json](evidence/portable_g2_cp8_npu.json)。实际输入是八份真实捕获，CP8、单篇长文档、每 rank 4096 token、`[1,96,128,128]` FP32 S/M/G，未重复扩展。形状从实际张量记录核对；命令中未使用的合成输入默认 head 数不是本次真实 H96 的标注。

每个实现 warmup 3 次、正式 **7 次**，不带 profiler。统计仍为 `median_i(max_r(F[r,i] + B[r,i]))`，精简文件保留七次最大值、对应 rank 和该 rank 的 F/B 分量。各实现按固定顺序分别计时，没有交替或随机化；三协议套件与 g2 套件又是两个窗口。以下为本次验收观测，不构成稳健性能排名，也不能与历史十样本窗口直接比较速度。

| 本次套件 | 实现 | F+B 中位数 ms | 最大/全组持久 M MiB | 最大 allocator 增量 MiB |
|---|---|---:|---:|---:|
| 三协议 | flat RD | 4.514600 | 18 / 102 | 54.003418 |
| 三协议 | head g2 + RD | 5.355110 | 12 / 78 | 54.004883 |
| 三协议 | cached tree | 5.953190 | 36 / 84 | 108.007813 |
| g2 三变体 | baseline | 5.030320 | 12 / 78 | 54.005371 |
| g2 三变体 | lifetime | 5.465810 | 12 / 78 | 48.004395 |
| g2 三变体 | direct | 5.744850 | 12 / 78 | 48.004395 |

所有实现均通过顺序 FP64 参考、两次在途 forward 后逆序 backward、两份输入和缓存不变性及紧凑 cache 检查。两次 H/D、所有 rank 中最大 FP64 相对 L2 不超过 `3.587012e-8`，head/tree 对 flat 的最大互差不超过 `4.394848e-8`；g2 两次 H/D 与两份全部缓存逐字节相同，互差为零。

lifetime/direct 复现最大单次 allocator 峰值下降 `6.000977 MiB`，持久 M 不变。**direct 本轮比 baseline 更慢，不能宣称提速**；lifetime 也没有证明加速。该顺序采样不足以断言稳定的速度差或普遍排名。allocator 增量、forward 后在途分配和持久 M 分别记录，不能互相代替。

本次 NPU case 只有 `long`，没有结构 reset 资格、任意 mesh 构造、完整 KDA、本地摘要生产者或训练验证；结构 reset 仍只有随附 CPU 控制。direct 的 `out=` 适用于显式 forward/backward 合同，不代表普通 Torch autograd 图资格。本次未增加 Level0，后文 Level0 全部属于历史来源。

### 本次运行时代码身份

精简文件完整保留原运行时模块名、源码 SHA256、worker/benchmark SHA256 和原结果 JSON SHA256。归档整理时核对以下正式模块以及所有研究模块，均与记录一致：

| 运行时模块 | SHA256 前 12 位 |
|---|---|
| 当前正式 RD | `b8b1fa04fe5e` |
| RD reset planner | `da00615e4384` |
| RD pointwise epilogue | `bc77fc600f19` |
| 缓存树 | `21215c9450de` |
| head owner | `0e4979197518` |
| lifetime/direct | `afef1a244be4` |

关联发布 checkout 的 RD 提交为 `3f53b2b4`，TND 前置为 `0a3f55c8`，主干基线为 `96143af9`。模块哈希标识实际运行的内容；提交标签说明关联发布上下文，不能替代源码哈希。本次归档分支 `research/kda-tree-head-rd-20261030` 计划以 `feat/kda-cached-rd-cp` 为 base，前置合入后可再调整 base。

## 以下为历史来源证据

后续表格保持 2026-10-29 的十样本口径、来源实现和独立采样窗口。历史结果用于保留研究过程和解释，不取代上面的本次迁移验收。

## 输入与资格边界

实际输入来自当时配对的 FLA 路径：CP8、单篇长文档、每 rank 4096 token、B=1、H=96、K=V=128，摘要张量为 `[1,96,128,128]`。使用八份真实 rank 捕获，未重复扩展捕获。各协议接收相同的 FP32 S/M/G，只比较边界协议，不包含本地摘要生产或完整 KDA core。

独立参考是给定这些摘要后的 FP64 顺序仿射计算。门槛为 `norm(error) <= 2e-6 * norm(reference) + 1e-11`，用于隔离协议代数误差，不是模型训练容差，也不是原生 FLA 混精训练的独立参考。构造只覆盖自然 WORLD 顺序；head DeviceMesh 构造被预建组绕过，不能据此宣称任意 mesh 构造已通过。

这两份 NPU 结果的 case 都只有 `long`。来源驱动提到的结构零 M reset 控制不属于这里的 NPU 样本，也不代表已经实现通信剪枝。完整训练、任意拓扑和较大 CP 不在这些历史实测资格内；可移植实现的独立验收见本页前节。

## 三协议同轮比较

来源为 [protocols_cp8_npu.json](results/protocols_cp8_npu.json)。每协议 warmup 3 次、正式 10 次，无 profiler。每次先取八个 rank 的最大 F+B，再取十次中位数，即 `median_i(max_r(F[r,i] + B[r,i]))`。不能先平均 rank，也不能将不同 rank 或不同样本的 F、B 极值相加。JSON 保留每次最大值、对应 rank 及该 rank 的 F/B 分量。

| 协议 | F+B 中位数 ms | 相对 flat RD | 最大持久 M MiB | 全组持久 M MiB | 实测最大 allocator 增量 MiB |
|---|---:|---:|---:|---:|---:|
| flat RD | 4.552265 | 基准 | 18 | 102 | 54.003418 |
| head g2 + RD | 5.471870 | +20.20% | 12 | 78 | 54.004883 |
| cached tree | 5.969505 | +31.13% | 36 | 84 | 108.007813 |

三协议均通过 FP64 参考、输入/缓存不变性标志及两次在途调用的 H/D 检查。两次 H/D、所有 rank 中最大相对 L2 分别为 `3.587012e-8`、`3.582022e-8`、`3.582019e-8`；head/tree 对 flat 的 H/D 最大互差均为 `4.394848e-8` 以下。缓存均为紧凑 storage。精简 JSON 同时保留逐 rank 相对 L2、最大绝对误差、最差 head 相对 L2 和 pass/finite 标志。

每次最慢 rank 的 F+B 范围分别为 4.496410–7.143780、5.364390–5.870910、5.925150–9.380290 ms。单一形状和采样窗口不足以推出大 CP 排名。

下表按 rank 0–7 排列，单位 MiB。实际 allocator 增量是记录的 NPU 已分配内存测量，不是 reserved memory，也不是持久缓存的估算。

| 协议 | 持久 M，rank 0–7 | allocator 峰值增量，rank 0–7 |
|---|---|---|
| flat RD | 0, 6, 12, 12, 18, 18, 18, 18 | 24.0024, 36.0020, 48.0029, 48.0029, 54.0034, 54.0034, 54.0034, 54.0034 |
| head g2 + RD | 6, 6, 9, 9, 12, 12, 12, 12 | 45.0039, 45.0039, 51.0044, 51.0044, 54.0049, 54.0049, 54.0049, 54.0049 |
| cached tree | 0, 12, 0, 24, 0, 12, 0, 36 | 24.0015, 72.0049, 24.0015, 90.0063, 24.0015, 72.0049, 24.0015, 108.0078 |

head g2 已减少跨前反向的持久 M，但该历史版本单次峰值仍约 54 MiB。树的 rank 7 是缓存和计算 owner 热点；全组持久 M 较少不等于每 rank 峰值较少。JSON 另保留每 rank 的 forward 后在途分配增量，三种内存口径分别记录。

## g2 生命周期与直接发送布局

来源为 [g2_variants_cp8_npu.json](results/g2_variants_cp8_npu.json)。这是另一轮同输入、三变体的 NPU 比较，warmup/正式次数和统计口径相同。来源 JSON 明确记录 `device=npu`、`validation_performed=true` 和 `pass=true`；实现 README 中“建议执行 NPU”的段落早于这份结果。

baseline 是冻结的旧 head 实现；lifetime 保持原有 matmul/add 表达式和合成顺序，在最后消费者后释放 packed aggregate、maps 及其视图、transfers 和 boundary list；direct 进一步将结果直接生产到每次调用独立分配的发送包。它仍保留 outer 种子复制、scatter、四次组内 AllToAll 和接收后必要的 head 布局恢复，不能称为整个 restore 零拷贝。该 direct 的 `out=` 只适用于原显式 forward/backward 协议，不构成普通 torch autograd 图资格。

| 变体 | F+B 中位数 ms | 相对本轮 baseline | 最大 allocator 增量 MiB | 最大/全组持久 M MiB | 两次 H/D 与全部缓存逐字节比较 |
|---|---:|---:|---:|---:|---|
| baseline | 4.792005 | 基准 | 54.005371 | 12 / 78 | 全 rank 通过 |
| lifetime | 5.081965 | +6.05% | 48.004395 | 12 / 78 | 全 rank 通过 |
| direct | 4.871440 | +1.66% | 48.004395 | 12 / 78 | 全 rank 通过 |

三变体先执行两次 forward，再逆序执行 backward；两份输入、两份缓存均有不变性检查，两次 H/D 及全部 cache tensor 与 baseline 的逐字节比较全部通过。FP64 最大相对 L2 均为 `3.582022e-8` 以下，互差为零。JSON 保留逐 rank 的比较标志和每个被比较张量的布尔结果。

| 变体 | allocator 峰值增量，rank 0–7，MiB |
|---|---|
| baseline | 45.0044, 45.0044, 51.0049, 51.0049, 54.0054, 54.0054, 54.0054, 54.0054 |
| lifetime | 36.0039, 36.0039, 45.0039, 45.0039, 48.0044, 48.0044, 48.0044, 48.0044 |
| direct | 36.0034, 36.0034, 45.0039, 45.0039, 48.0044, 48.0044, 48.0044, 48.0044 |

证据支持最大单次 allocator 峰值下降约 6.001 MiB、持久缓存不变。direct 没有继续降低最大 rank 峰值；这十个样本也没有证明两种优化相对本轮 baseline 加速。三变体每次最慢 rank 的范围为 4.667510–7.296350、4.917300–8.765900、4.817980–7.643940 ms。前一轮与本轮的 outer RD 来源 SHA256、驱动和采样窗口不同，不能把两张表拼成一次速度比较。

## 对齐 Level0 支持的解释

[aligned_level0_summary.json](results/aligned_level0_summary.json) 摘自已解析的对齐 Level0 结果，没有重新读取大 trace。每协议在 profiler 内先做 setup F/B，再 barrier+synchronize，随后采一轮正式 F/B；解析排除 setup。通过 CPU 阶段与 `torch_to_npu` flow 关联设备任务，所有算术/布局任务均找到来源。CPU 标记只用于归属，不作为设备阶段耗时。

| 指标 | flat RD | head g2 + RD | cached tree |
|---|---:|---:|---:|
| 八 rank 前向标记起点跨度 μs | 392.96 | 126.97 | 795.31 |
| rank 7 算术/布局 kernel 数 | 11 | 31 | 45 |
| rank 7 算术/布局区间并集 μs | 129.76 | 239.68 | 395.02 |
| rank 7 GEMM 时长合计 μs | 87.56 | 85.24 | 159.22 |
| rank 7 AICPU 合批调度数 | 8 | 6 | 0 |

head g2 的外层调度数减少，同时增加四次组内 AllToAll、布局操作与提交交接。其 rank 7 GEMM 时间接近 flat，不能把差距只归因于 GEMM。树的 GEMM 数按 rank 为 `0,4,0,8,0,4,0,12`；root 的 12 个 send/receive aggregate 区间并集为 5410.78 μs，设备任务跨度为 6065.52 μs，说明通信顺序和 peer 就绪仍需分析。

head g2 的 rank 6/7 前向 restore 开始相差 1248.505 μs、结束仅相差 59.233 μs；反向分别为 390.578 和 55.803 μs。恢复阶段受最慢 lane 就绪时间约束。HCCL aggregate 包含等待，不能用这些区间推裸链路带宽；AICPU BatchSendRecv 是调度，不是完整 P2P 传输。等待、通信、计算和各阶段区间存在重叠，不能相加得到总时延。单次带 profiler 的轨迹用于解释依赖，不替代无 profiler 中位数。

原 `profile_analysis.json` 的限制文字仍保留初版启动偏斜说明；随附分析文档及记录中的 aligned trace 标识说明上述数值来自对齐重采。归档保留这一来源差异的说明，没有把初版 trace 混入数值。

## 后续路线

1. 本次可移植版已完成前述 CP8 三协议与 g2 三变体迁移验收。后续若声称提速，需以交替或随机化的配对计时复核；当前七次固定顺序观测不替代稳健性能比较。
2. g2 继续以最后消费者释放和发送布局为单一控制，优先稳定复现已测的峰值下降；保留 backward 必需的本地 M 和 outer cache。不要用删除 host 引用或持久 M 减少替代 allocator 测量。
3. 树只保留一个清晰对照，先测临时缓冲释放和 owner 上下行依赖；分别报告 36 MiB 持久 M 与约 108 MiB 历史单次峰值，不预先宣称峰值改善。
4. 大 CP 评估需包含四次组内 collective、最慢 lane、远端并发、拓扑和缓存成本，并实测验证。CP8 证据既不足以将 g2 设为默认，也不足以否定其大 CP 价值。

## 提取方式与归档内容

历史部分从两份历史 NPU JSON、已解析的 `profile_analysis.json` 及两份说明文档选择字段；新迁移验收从两份独立的新 NPU JSON 选择字段。未扫描或复制大 trace。每份精简 JSON 记录原结果文件的 SHA256；NPU 摘要另保留原运行记录的实现和驱动 SHA256，用于区分历史代码与本次可移植代码。

数值字段按原记录保留；每次最慢 rank 的时间从原始逐 rank 样本重新计算，并核对中位数与原 summary 完全一致。缓存数组与原 summary 逐项核对。Level0 只保留各 rank 的计数/区间统计和关键阶段；跨 rank restore 差值由原时间计算后仅保留差值。

归档删除了本地绝对路径、个人/机器标识、绝对时间戳、运行日志、环境和 profiler 文件定位信息。保留的设备类型、张量形状、rank 次序与数值口径用于解释实测。所有 JSON 均小于 100 KB，字节值在 JSON 中精确保留，表格的 MiB 和时间按展示精度舍入。
