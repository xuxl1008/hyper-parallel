# Ring Attention 实现、实验与后续工作

更新日期：2026-11-02。本文记录 Ring Attention 研发分支；实现尚未合入 master。
本文中的历史测量、实验分支和当前默认路径分别标明，避免把局部实验当作完整训练收益。

## 当前能力与代码入口

| 部分 | 已实现的范围 | 验证或限制 |
| --- | --- | --- |
| 纯 Ring | 固定本地 Q，轮转 KV；在线 softmax 合并；完整反向 | 2/4/8 卡 NPU |
| Attention | MHA/GQA，FP16/BF16，BNSD/BSND/TND，因果与非因果 | dropout 必须为 0；不支持任意 mask、MLA、KV cache |
| 梯度 | 本地累加 dQ；FP32 累加、轮转 dK/dV 并返回所属 rank | 不除以 CP size |
| 通算重叠 | 预取下一轮 KV，流水执行反向 dKV 通信，独立双缓冲 | `overlap=True`；API 默认仍为 False；没有完全隐藏通信的保证 |
| 布局 | none、zigzag、greedy、hybrid；文档隔离和原始因果坐标 | 层出口恢复原始 token 归属；全局 token 数需整除 CP |
| 不规则长度 | Zigzag 使用真实余数，不填充文档、不要求每篇文档整除 2×CP | Greedy/Hybrid 可以填充传输缓冲区容量，FA 只处理真实 token |
| 混合并行 | Ulysses + Ring | 已覆盖 2×2、CP8 U2R4/U4R2；Q/KV head 数需整除 Ulysses degree |
| 大 GQA 模块 | Qwen3-14B 尺寸 GQA、同尺寸 GatedGQA，含 norm/RoPE | CP8、local 8K、10 个真实文档长度包；随机参数，非完整 14B 训练 |
| 模型与训练入口 | Qwen3-MoE adapter、recipe、TextTrainer、FSDP/HSDP | 既有训练验证为小模型、256 token、完整未 packed 样本、3 step |

代码入口：

- [Ring 前后向、通信与公共 API](../../hyper_parallel/distributed/context_parallel/ring_attention.py)。
- [文档块、TND 合批与重排](../../hyper_parallel/distributed/context_parallel/ring_layout.py)。
- [与 Indexer 共享的任务规划和路由](../../hyper_parallel/distributed/context_parallel/query_balance.py)。
- [模型的 Ring adapter](../../hyper_parallel/models/qwen3_moe/adapter/distributed/context_parallel_ring.py)。
- [Trainer recipe](../../tests/torch/context_parallel/ring_trainer.yaml)。

接入使用模型拥有的 attention 接口和 recipe forward rewrite。传入本地 Q/K/V，函数内部完成均衡、Ring 和输出恢复。
无需要求异构模型的不同层持续保持同一个均衡布局。若要把通信提前到 Q/K/V 投影之间，需要模型 adapter 扩展调度边界，
仅替换接收已生成 Q/K/V 的 attention 接口无法取得这些更早的计算窗口。

## 前后向流程与负载均衡语义

前向第 s 轮，rank r 使用本地 Q 与 owner=(r-s) mod CP 的 KV 计算。
开启 overlap 时，先发起下一轮 KV 交换，再计算当前块；真正使用收到的数据前等待通信完成。
每轮更新运行中的 softmax maximum、sum 和归一化输出，最后转回输入 dtype。

反向使用最终输出和全局 softmax 统计计算各块梯度；dQ 本地累加，dK/dV 随 owner 的 KV 沿环累加。
前向和反向各有 CP-1 次 KV 轮转；反向另有 CP 次 dKV 轮转，使最终梯度回到所属 rank。
KV 与 dKV 使用独立缓冲，接收存储在消费者结束前不可复用。

对于长度 L 的文档，Zigzag 写成 L=2×CP×c+r。rank j 获得规则主体的第 j 和第 2×CP-1-j 块。
文档最前面的 r 个真实 token 进入余数池，再将余数池等量分给各 rank。
每篇文档的原始坐标和边界始终保留，因此可执行文档内因果 attention。

TND 合批会把一轮内多个合法 Q/K 区间组合为 FA 的分段输入。相同轮次中，因果模式和 KV 打包容量限制仍可能要求多个调用。
连续布局下，没有合法 QK 对的轮次可以跳过计算；Zigzag 把更多文档分布到更多 rank 后，更多轮次需要启动 FA。
因此“多个文档拼接在一起”并不意味着整个 Ring 前向只有一次 FA。

## 数据集、采样与文档长度

使用的 indexed dataset 前缀：

```text
/home/xxl2/work/datasets/deepseek_mmap/mmap_deepseekv3_datasets_text_document
```

共有 930,514 篇文档、1,161,275,332 token。文档长度 p50=579、p90=2318、p99=11995；最大长度 1,554,223。
原始索引 SHA256 为 `d535e502d62368e7a195b27a9a717543534ee1eaa9b4f9ce45f4d08d746cb5cd`。
已重新对照原始索引检查下面每个片段的 document ID、offset 和 length。

采样由[数据集 benchmark](../../tests/torch/context_parallel/_benchmark_ring_attention_dataset.py)完成，固定 seed=20261101：

1. **shuffled_000…007**：对整个数据集的文档 ID 做一次无放回随机排列，沿排列后的文档流连续切出 8 个 65,536-token 包。
2. **ordered_000**：从原始索引的第 0 篇文档开始，按原顺序取一个 65,536-token 包。没有按长度排序。
3. **long_000**：选择索引中第一篇长度至少 65,536 的文档，即 document 77（原长 110,600），取其前 65,536 token。

这三类都来自该数据集的长度。ordered 与 shuffled 不是同一组文档的不同排列；long 是特意选取的长文档对照。
8 个 shuffled 包也不是 8 次独立抽样，而是同一次随机排列中的连续窗口。

仅在包边界切分文档。上一包末尾没装完的文档，下一包从相应 offset 继续取。
每包 attention 在片段边界重新开始，不跨包访问前文；不额外增加 BOS/EOS，不做 label shift。
因此下表中的数量指包内文档片段，长度也指包内可见片段，不一定是原文全文长度。

例如 shuffled_000 的最后一段是 document 558356 的前 1192 token；shuffled_001 从同一文档 offset=1192 开始，取余下 19533 token。
ordered_000 的第 7 篇原长 9079，包内只放入前 5364 token。

| 包 | 片段数 | 最短 | 中位数 | 最长 | 有效因果 QK 对数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| shuffled_000 | 64 | 31 | 531.5 | 6100 | 82,889,943 |
| shuffled_001 | 40 | 33 | 517 | 19533 | 287,090,206 |
| shuffled_002 | 78 | 37 | 535.5 | 5835 | 67,205,523 |
| shuffled_003 | 85 | 39 | 506 | 4847 | 56,324,514 |
| shuffled_004 | 48 | 87 | 854 | 11408 | 117,393,864 |
| shuffled_005 | 44 | 36 | 528.5 | 26023 | 386,715,979 |
| shuffled_006 | 48 | 89 | 821 | 8165 | 118,684,113 |
| shuffled_007 | 51 | 83 | 667 | 9373 | 132,710,511 |
| ordered_000 | 7 | 4580 | 6133 | 16850 | 387,787,824 |
| long_000 | 1 | 65536 | 65536 | 65536 | 2,147,516,416 |

ordered_000 的完整组成是 `[5383, 16850, 15133, 4580, 12093, 6133, 5364]`。
文末保留全部 10 包的完整长度列表。文档 ID、原文 offset 和采样元数据由 benchmark 的
`--prepare` 命令写入本地 manifest；生成方式见“复现与结果文件”。

这些性能实验使用真实文档长度及边界，Q/K/V/dO 由固定 seed 随机生成。
没有把数据集的 token 内容送入模型，因此属于 attention 或投影加 attention 的实验，不能当作真实数据训练吞吐。
8 包平均值是各包耗时的算术平均，不能据此估计整个数据集或训练任务的平均性能。

## 已进入核心实现的优化

| 优化 | 目的或结果 |
| --- | --- |
| 规则 dense Zigzag 快速路径 | 规则文档每轮一次 FA，减少通用任务处理 |
| 合并文档区间并以 TND 合批 | 减少逐文档、逐碎片启动 FA |
| 合并 Q/K/V 重排 | 输入只做一次 all-to-all-v；包含 O 恢复和反向，一步共 4 次重排 collective |
| 缓存计划和设备索引 | 避免重复构造任务、索引和元数据 |
| 连续区间使用 view/narrow | 减少 gather 与复制 |
| 优化唯一行写回 | 无重复目标行使用 index_put；需归约的梯度仍累加 |
| K/V 和 dK/dV 合并传输 | CP8 一步每 rank 的 send/recv 各从 44 次降为 22 次；字节数不变 |
| 复用打包接收存储 | 后续轮转继续发送接收存储的视图，减少重复打包 |

最近一轮写回与通信打包优化的配对实验，shuffled Zigzag 平均从 60.130 ms 降到 50.270 ms，耗时降低 16.4%。
这是独立的一组前后对照，不应与下面另一轮测量的小幅差值混为一谈。

历史记录包括 CPU 规划测试、2/4/8 卡布局测试、8 卡 overlap 回归、以及 Trainer/FSDP/HSDP 测试。
Trainer 组合为 CP8 Ring+FSDP8、CP8 U2R4+FSDP8、CP8 U4R2+FSDP8、DP2×CP4 U2R2+HSDP2×4。
使用独立 torchrun 作业对比 AllGather CP、同步 Ring 与 overlap Ring，检查损失、梯度范数、首步完整梯度和参数更新。
Trainer 历史记录没有覆盖最近每一项微优化，也没有覆盖真实 packed local 8K 完整训练。

## 性能证据与均衡变慢的原因

以下是旧一轮 attention-only 配对测量：Ascend 8×910B3，纯 Ring CP8，local 8192，BF16，Hq/Hkv=16/4，D=128，因果。
所有方案开启 Ring overlap；统计 attention 前后向及所有重排。每包预热 3 次，测量 5 组，每组 8 次。
每组取最慢 rank 的时间，再取 5 组中位数；shuffled 列为 8 个包中位数的平均。

| 方案 | shuffled 平均 ms | ordered ms | 单条 64K ms |
| --- | ---: | ---: | ---: |
| none | 46.145 | 50.883 | 126.791 |
| 当前 Zigzag | 50.048 | 55.985 | 86.304 |
| 主块+真实余数实验 | 49.660 | 53.394 | 87.132 |
| 上述实验+固定输入 side stream 准备 | 49.831 | 53.577 | 87.034 |
| 上述实验+下一轮 KV side stream 准备 | 51.519 | 53.914 | 87.169 |

该表使用 `TASK_QUEUE_ENABLE=0` 的诊断配置。默认运行配置也有记录，但出现过长尾；
针对 shuffled_002 的默认配置 9 组复测，Zigzag/主块余数/Q 准备/KV 准备分别为 46.635/47.198/47.531/47.863 ms。
之前默认配置中 side stream 的额外收益没有在这次复测中稳定出现。

有效因果 QK 对数为 sum(L_i×(L_i+1)/2)，不是总 token 数的平方。
shuffled_000 约有 82.89M 对，单条 64K 有 2147.52M 对，是前者约 25.9 倍。
前者不开均衡的最大/平均 QK 对数约 1.885，Zigzag 降到约 1.000，说明算术量确实均衡了。
但短文档场景的绝对计算量更小，省下的计算等待不足以支付额外重排、碎片打包和启动成本。

代表性 shuffled_000 的 profile 中，跨 8 个 rank 平均，每 rank 每步：

| 项目 | none | Zigzag |
| --- | ---: | ---: |
| FA 前向调用 | 1.875 | 8 |
| FA 反向调用 | 1.875 | 8 |
| Gather 调用 | 4 | 127 |
| Gather 累计执行时间 | 约 0.61 ms | 约 7.4 ms |
| Ring send/recv | 各 22 次 | 各 22 次 |
| 均衡输入/输出及反向 all-to-all-v | 0 | 4 次 |

算子累计时间不是互斥的关键路径分解，不能相加或直接从总耗时中扣除。
Ring 仍轮转相同容量的 KV；none 跳过无合法 QK 对的计算轮次，并不同时跳过通信。
当前规划依据有效 QK 对数，尚未充分建模 FA 块形状、tile 利用率、启动次数、重复 KV 以及每轮最慢 rank 的等待。

## 主块、余数和提前准备实验

主块+余数实验保留原来的 token 分配，只将内部目的顺序改成主体在前、真实余数在后：

- 主体对主体走规则 Zigzag Ring。
- 主体 Q 对本篇文档余数 KV 执行全可见的小 TND attention。
- 余数 Q 对余数 KV 执行因果 attention；余数 Q 不需要访问后面的主体 KV。
- 余数 KV 前后向各一次 all-gather，余数 dKV 以 FP32 SUM reduce-scatter 返回。
- 提前准备固定 Q、O、dO 和统计布局，在 Ring 轮次间复用。

代表性样本 Gather 从 127 次降到约 63.6 次，累计约 7.42→3.78 ms；
但 FA 前后向分别从 8 次升到 10 次，并增加余数通信。
10 包中最大额外分配峰值由约 732 MiB 增到 760 MiB。
因此 shuffled 平均仅降低约 0.8%，ordered 降低约 4.6%，当前保留为实验。

KV side stream 版本的 trace 中，FA 与 Gather 有约 0.46 ms/step 的时间交集。
这证明部分操作同时执行，不代表同等大小的端到端收益；也不能用通信区间与 FA 的重叠比例认定传输全部隐藏。
这些实验只发生在 attention 内部，不包括投影。

## 投影、准备与通信的配对实验

[投影实验脚本](../../tests/torch/context_parallel/_benchmark_ring_projection_prepare.py)保持当前 Ring FA 和任务计划不变。
计时覆盖 Q/K/V 三个投影、Ring、输出投影和本地完整反向，hidden=2048，local=8192，Hq/Hkv=16/4，D=128。
参数在 rank 间使用相同初始化，但不做参数梯度同步；不含 QK norm、RoPE、FSDP、优化器或 Trainer。
这是对调度可行性和收益的受控对照，不能替代模型训练测试。

| 名称 | 调度方式 | 每步均衡路由 collective 数 |
| --- | --- | ---: |
| none | 投影后调用不均衡 Ring | 0 |
| baseline | Q/K/V 全部完成后，按当前实现合并重排 | 4 |
| prepack_serial | 各投影后立即在主 stream 准备发送/本地行，最后合并一次输入通信 | 4 |
| prepack_overlap | 同上，但在 side stream 准备 Q/K 行，与后续投影交错；消费前等待 event | 4 |
| qcomm_serial | Q 单独重排并等待，再执行 K/V 投影和 K/V 合并重排 | 6 |
| qcomm_overlap | Q 投影后立即发起 Q 重排，再执行 K/V 投影；消费 Q 前等待 | 6 |

prepack_overlap 对比 prepack_serial 用于观察提前准备的收益；qcomm_overlap 对比 qcomm_serial 用于观察更早发起通信的收益。
两者都还应与 baseline 比较，因为拆分准备、拆分 collective 也会带来额外成本。

两种运行配置均使用相同的 10 包，预热 3 次、测量 7 组、每组 8 次；下表为 8 个 shuffled 包的平均。
默认配置另测前向 7 组。单位均为 ms。

| 方案 | queue0 完整前后向 | 默认配置完整前后向 | 默认配置前向 |
| --- | ---: | ---: | ---: |
| 不均衡 Ring | 48.362 | 48.365 | 14.036 |
| 当前 Zigzag 基线 | 52.387 | 69.159 | 17.754 |
| 逐投影准备，串行 | 52.818 | 69.152 | 17.743 |
| 逐投影准备，side stream | 52.683 | 56.774 | 17.916 |
| Q/KV 分开发送，串行 | 53.123 | 53.038 | 17.795 |
| Q 提前发送，与 K/V 投影交错 | 53.144 | 53.483 | 17.703 |

queue0 下，side stream 准备比同样准备方式的串行版本降低约 0.26%，但仍比当前基线慢约 0.57%；
Q 提前通信比当前基线慢约 1.45%，也未超过其串行对照。
默认配置前向的差别较小：提前打包约增加 0.91%，Q 提前通信约降低 0.29%，不足以证明稳定收益。

默认完整前后向被 shuffled_005 的长尾显著影响：baseline 的 7 组为
`[70.61, 199.70, 233.91, 196.83, 249.59, 62.88, 65.83] ms`。
同一次测量的 prepack_serial、prepack_overlap、qcomm_serial、qcomm_overlap 和 none 也有长尾。
不能把默认配置表中的大幅差值全部归因于隐藏通信。下方另记固定包复测，不删除原始数据。

对 shuffled_005 启动独立进程复测：完整前后向 11 组、前向 7 组，每组 8 次。
中位数如下，所有重复值仍保留；基线仍有单次约 292 ms 的长尾。

| 方案 | 默认配置复测完整前后向 | 默认配置复测前向 |
| --- | ---: | ---: |
| 不均衡 Ring | 66.241 ms | 20.024 ms |
| 当前 Zigzag 基线 | 61.585 ms | 20.565 ms |
| 逐投影准备，串行 | 62.568 ms | 20.616 ms |
| 逐投影准备，side stream | 61.768 ms | 20.755 ms |
| Q/KV 分开发送，串行 | 62.376 ms | 20.790 ms |
| Q 提前发送，与 K/V 投影交错 | 62.070 ms | 20.490 ms |

此次提前打包和提前 Q 通信仍未超过当前基线；前一轮聚合中的大幅差异没有复现。
prepack_overlap 在本次固定包的重复值更集中，这可以作为继续调查长尾的线索，尚不足以认定它能稳定改善队列行为。

两个配置中，所有提前准备/通信变体在全部 10 包、8 rank 上的 O、dHidden、dWq、dWk、dWv、dWo
均与当前 Zigzag baseline 逐位一致；所有计时方案测得 live allocation growth 为 0。
不均衡 Ring 的输出/梯度与 Zigzag 存在分块计算带来的数值差异，未要求逐位相同。

异步路径显式等待 forward 接收完成，并在自定义 autograd 中执行和等待逆向 all-to-all。
发送源与接收存储保持存活；side stream 通过 event 和 record_stream 保护依赖与内存生命周期。
只改变路由顺序的方案应与 baseline 有相同输出及各投影梯度；额外记录 dHidden，避免把 BF16 分支累加顺序误差隐藏掉。

## 哪些通信可以隐藏，哪些操作适合融合

| 位置 | 可用计算窗口 | 当前判断 |
| --- | --- | --- |
| Ring KV 轮转 | 当前轮 FA | 已实现流水；短文档计算窗口不足，仍会暴露等待 |
| Ring dKV 回传 | 下一块反向 FA | 已实现延迟等待；缓冲复用前必须完成通信和消费 |
| 输入重排 | Q 先就绪后，后续 K/V 投影、norm、RoPE | 可以提前发 Q；与合并 QKV 一次通信有取舍，本次实验给出直接对照 |
| 输入本地打包 | Q/K 已就绪时，后续投影 | 可用 side stream；当前通用 gather/concat 版本有额外启动和临时缓冲 |
| 输出恢复 | 同层按 token 块执行输出投影 | 需要分块流水；当前整个输出恢复后才执行输出投影 |
| 反向输入梯度恢复 | 各分支投影梯度计算 | 需要按 Q/K/V 或 token 块提供完成通知，不能在所有梯度就绪后简单换 stream |

保持 QKV 合并为一次 collective 时，整包通信必须等 V 就绪；提前准备只能覆盖准备部分，不能提前发送尚未生成的 V。
单独提前发 Q 可以增加计算窗口，但会增加 collective 次数。GQA 的 K/V 投影较小，是否足以覆盖通信必须实测。
若模型本身采用一次 fused QKV projection，还需避免为调度重叠拆坏现有 GEMM。

输出投影是逐 token 的线性操作，可以研究在均衡布局上计算后再恢复归属，或者按块边恢复边投影。
但整个投影前移本身不等于通算重叠；当 hidden 大于 attention 输出宽度时，先投影还会增加恢复通信字节数。
这些调度都可限制在同一层内，不要求异构模型所有层共享布局。

本次 shuffled_000 的硬件 trace（8 rank、每 rank 3 step）中：

| 调度 | QKV 投影累计时间 | K/V 投影累计时间 | Gather 与 QKV 投影交集 | A2A 与 K/V 投影交集 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.436 ms | 0.142 ms | 0 | 0 |
| prepack_overlap | 0.451 ms | 0.178 ms | 0.041 ms | 0 |
| qcomm_overlap | 0.446 ms | 0.179 ms | 0 | 0.178 ms |

每个类别先合并设备 kernel 时间区间，再求交集，并对 rank 和 step 取平均。
这是实际同时执行的区间；HCCL 区间含等待，0.178 ms 不能直接解释为隐藏了等量的网络传输。
这里可以利用的 K/V 投影窗口约 0.14–0.18 ms，远小于整步 attention 时间。

同一 profile 的 baseline Gather 按输入形状粗分如下。Q/O/dO 类中也可能包含形状相同的边界重排；
统计 encode/decode 指两个物理维度的 `[T×N,8]` 布局转换。

| Gather 类别 | 次数/step/rank | 累计时间 |
| --- | ---: | ---: |
| softmax 统计 encode/decode | 32 | 3.574 ms |
| Q/O/dO 和输出状态 | 37 | 2.789 ms |
| K/V | 28 | 0.637 ms |
| softmax 统计按 token 行 gather | 28 | 0.431 ms |
| 其他形状的路由 gather | 2 | 0.045 ms |

合计 127 次、约 7.48 ms。多数发生在 Ring 轮内；只把入口 Q/K 打包移到投影旁边不会消除这些操作。
因此优先融合统计布局转换和每轮输入准备，比只增加入口 stream 更有针对性。

优先考虑下面的融合边界：

| 候选融合 | 当前多个操作 | 预期减少的成本与约束 |
| --- | --- | --- |
| QKV route pack | concat → gather → contiguous | 按路由直接从 Q/K/V 写入最终发送缓冲；支持不同 Q/KV head 数 |
| 接收与本地行合并 | zeros → local gather → 两次写回 | 按源地址映射一次写目标，确保无重复目标；通信完成后再访问接收数据 |
| 前向 softmax merge/writeback | 统计 decode → 旧状态 gather → max/exp/加权 → scatter | 融合到行索引驱动的更新，减少统计中间张量和 HBM 往返；保持 FP32 运行状态 |
| 反向输入准备 | Q/O/dO gather，max/sum gather 与 encode | 一个或少数 kernel 直接产出 FA 要求的 TND 布局，固定输入按模式复用 |
| 反向梯度写回 | cast → 多次 index_add | 融合转换和累加；重复 KV 行要使用正确归约，不能改成覆盖写 |

这属于融合本地内存访问和逐元素计算，不会自动减少网络字节，也不会自动消除 FA 内部的低 tile 利用率。
若要完全去掉 FA 前的 KV gather，需要 FA 内核支持片段/间接寻址，改动范围大于外围融合。

可参考 MindSpeed 的
[npu_ring_attention_update 接口与等价计算](https://gitee.com/ascend/MindSpeed/blob/4b3af73493fb9138373f82c8942f5aa65277b3ff/docs/ops/npu_ring_attention_update.md?skip_mobile=true)，
它将 attention 输出和 softmax 统计合并；但本实现的任意行映射、TND 统计物理布局和 FP32 状态仍需适配验证。
自定义 gather/pack/merge 可评估 Ascend C 或
[Triton Ascend](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/architecture_design_and_core_features.md)。
这里记录的是融合设计方向，尚未测量这些新融合 kernel 的收益。

## Qwen3-14B 尺寸与 Gated GQA

[官方 Qwen3-14B 配置](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json)为
hidden=5120、Q heads=40、KV heads=8、head_dim=128，Q/KV 分组比为 5。
原始 Qwen3-14B 使用普通 GQA；本节的 gated 版本在相同维度上加框架已有的逐元素 sigmoid 输出门控，
不能称为原模型本来就有 gate，也不代表已经跑过完整 Qwen3.5/Qwen3-Next 混合模型。

[大 GQA 实验](../../tests/torch/context_parallel/_benchmark_ring_gated_gqa.py)直接使用框架的
[GQAAttention / GatedGQAAttention](../../hyper_parallel/components/modules/gqa_attention.py)，
经 `attention_interface` 调用 Ring。普通基线和 gated 基线都使用实际模块的 forward，
含融合 QKV（gated 为 QKVG）投影、QK norm、RoPE、attention、输出投影和本地完整反向。
QK norm 为 Transformers `Qwen3RMSNorm`，RoPE 使用框架 NPU 算子、theta=1e6，每篇文档位置从零重新编号；
cos/sin 的生成在计时之外，应用及反向在计时内。

CP=8、local=8192，仍使用同一 manifest 的 10 包。激活、权重、上游梯度均为随机 BF16，
不加载 14B checkpoint，不含 MLP、优化器、Trainer、FSDP 或参数梯度同步。
64K 单文档超过原配置的 40960 位置范围，这里只用作算子压力对照，不验证预训练模型的长上下文效果。
本节含 norm/RoPE、融合投影，不能与前节省略这些操作、QKV 分开投影的毫秒值直接比较。

### Gate 对 Ring 语义和通信的影响

令 `A=Attention(Q,K,V)`、`G=sigmoid(X Wg)`，模块输出为 `Y=(A ⊙ G) Wo`。
当前框架把 `Wq/Wk/Wv/Wg` 放在同一融合投影内；`Wg` 这里只用于描述数学上的对应权重切片。
框架的 gate 与
[Transformers Qwen3-Next full attention](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
具有相同的输出门控依赖关系；这不是 Gated DeltaNet 的递归状态更新。

- Gate 属于本地 query token，与远端 K/V 无关。先完成 Ring，再把 A 恢复原 token 顺序并与本地 G 相乘，
  不需要让 gate 随 KV 转圈，也不需要增加 gate 的 all-to-all。
- Ring 的 KV 轮转、在线 softmax、dKV 累加和归属不变。当前 FA 不直接看到 gate。
- 设 `dZ` 是乘法输出的梯度，则 `dA=dZ⊙G`，`dGateLogit=dZ⊙A⊙G⊙(1-G)`。
  Ring backward 接收 dA；FA 保存的 `attention_in` 仍须是未门控的 A，不能改成 A⊙G。
- TND 文档隔离和 Zigzag 块计划也不变。Gate 与原始 token 位置对应，在输出恢复后相乘，满足异构层间恢复原布局的要求。

普通 GQA 的本地 KV 前向交换缓冲为 `2×8192×8×128×2 bytes = 32 MiB`，
是先前 4 KV heads 的两倍；40 个 Q heads 则是先前 16 heads 的 2.5 倍。
QK/PV 计算相对 KV 通信的比例有所增加，但这不是均衡方案必然获益的保证。
逐元素 gate 激活宽度为 5120，每卡一份 BF16 gate 为 80 MiB；这是本地计算/存储成本，不是额外 CP 通信量。
若改用每 head 一个 gate，则只有 40 个元素/token，计算窗口会小得多；本次没有测试 headwise gate。

### 门控调度实验

| 名称 | 调度 | 目的 |
| --- | --- | --- |
| plain_none / plain_zigzag | 框架普通 GQA，分别关闭/开启均衡 | 大 GQA 的均衡收益 |
| gated_none / gated_zigzag | 框架 GatedGQA，分别关闭/开启均衡 | 融合 QKVG 的实际基线 |
| gated_restore_serial | 融合 QKVG；显式发起输出恢复并等待，再 sigmoid | 控制实验中异步路由包装本身的成本 |
| gated_sigmoid_overlap | 融合 QKVG；发起输出恢复后计算 sigmoid，再等待 | 保留投影融合，只调度独立逐元素计算 |
| gated_split_serial | 融合 QKV；输出恢复并等待后单独计算 Wg GEMM、sigmoid | 拆分 gate 的串行对照 |
| gated_split_overlap | 融合 QKV；发起输出恢复后计算 Wg GEMM、sigmoid，再等待 | 用独立 gate 投影覆盖输出路由窗口 |

所有 Zigzag 变体仍为每步 4 次均衡路由 collective，gate 不参与路由。
显式异步实验沿用已验证的 `_WaitRoute` 自定义 autograd：保留发送存储、消费前等待，反向执行并等待逆向通信。
数值比较所需的融合参数梯度拆分只发生在检查阶段，不计入性能测量。

可以调度的独立工作是 gate 投影与 sigmoid；`A⊙G` 必须等 A 到达。
在当前融合模块中，gate 投影已经与 QKV 一起完成，不能把一份尚不存在的“独立 Wg GEMM”算作可隐藏时间。
拆分方案必须同时与同样拆分的串行对照、现有融合基线比较。

这里利用的是 Zigzag 的**输出恢复**通信，不是每一轮 KV P2P 的新增掩盖。
关闭均衡后没有这段 all-to-all；Ring 原有的 KV/FA、dKV/反向 FA 流水仍照常工作。
让额外 GEMM 与 Ring 轮内 FA 并行会争用计算/HBM，不能从独立性直接推出净收益。
反向的 gate 分支也有独立计算，但融合 QKVG 的权重梯度 GEMM 需要全部 dQ/dK/dV/dGate；
只有拆分后才能独立调度 gate GEMM。本次计时包含实际 autograd 的完整反向，未新增专门的反向双流调度器。

### 大 GQA 配对结果

`TASK_QUEUE_ENABLE=0` 下，10 包各预热 3 次，完整前后向测量 7 组、前向测量 5 组，每组 4 次；
轮换方案顺序，每组取最慢 rank，再取组间中位数。8 个 shuffled 包的结果为包中位数的算术平均。
全部原始重复值保留，没有删除长尾。

| 方案 | 8 包平均前后向 | 8 包平均前向 | 原顺序包前后向 | 64K 单文档前后向 |
| --- | ---: | ---: | ---: | ---: |
| 普通 GQA，不均衡 | 115.789 ms | 33.163 ms | 130.374 ms | 349.290 ms |
| 普通 GQA，Zigzag | 126.574 ms | 44.434 ms | 141.409 ms | 234.783 ms |
| GatedGQA，不均衡 | 122.913 ms | 35.391 ms | 135.811 ms | 356.293 ms |
| GatedGQA，Zigzag 融合基线 | 133.495 ms | 46.807 ms | 148.140 ms | 242.863 ms |
| 融合投影，显式恢复串行对照 | 133.285 ms | 46.647 ms | 147.636 ms | 242.191 ms |
| 融合投影，sigmoid 重叠 | 133.069 ms | 46.644 ms | 147.808 ms | 242.190 ms |
| 拆分 gate，串行 | 133.151 ms | 46.437 ms | 147.756 ms | 242.117 ms |
| 拆分 gate，与输出恢复重叠 | 132.184 ms | 45.171 ms | 146.181 ms | 241.851 ms |

对普通 GQA，Zigzag 在 8 个 shuffled 包上平均慢 9.31%，在单篇长文档上快 32.78%。
包间差异仍然很大：shuffled_005 含约 26K 的长文档，115.8 ms 的总体平均不能代表该包；
该包普通 GQA 从 162.842 ms 降至 149.746 ms，均衡已经有收益。
Gate 是逐 token 的局部计算，不改变有效 QK 对的分布，因此不会消除短文档的任务碎片和 gather 问题。

拆分 gate 的重叠版本相对同样拆分的串行对照，8 包平均前向减少 1.266 ms、完整前后向减少 0.968 ms；
相对当前融合门控基线，分别减少 1.636 ms（3.49%）和 1.312 ms（0.98%）。
不能把相对融合基线的全部差值归因于重叠，拆分 GEMM、内存布局和显式路由也改变了成本。
只提前 sigmoid 的前向平均与串行对照仅差 0.003 ms，没有实用收益的证据。

这次完整模块比较的参考是同输入、同权重的框架 Zigzag 模块，不是独立 FP32 全模型参考：

- 保留融合的两种显式路由调度在 10 包、全部 8 rank 的输出、dHidden、QKV/输出投影和 QK norm 梯度上逐位一致。
- 拆分 gate 的输出、所有参数梯度也与融合基线逐位一致；只有 dHidden 改变，最大相对 L2 为 0.002844（0.2844%），
  对应一次融合 GEMM 与分支 GEMM 后相加的 BF16 累加路径差异。
- 拆分串行与拆分重叠之间，输出及全部梯度（含 dHidden）逐位一致。门控梯度完整参与比较。
- 所有计时方案的 live allocation growth 均为 0。原顺序包的峰值额外 allocation：融合基线 2139 MiB，
  拆分重叠约 2170 MiB；这不含计时前已经存在的参数/输入，更不是完整 14B 模型显存。

### 默认队列复测与设备 trace

另启独立进程，在默认队列配置下复测 shuffled_000 和 shuffled_005，重复数与上表相同。
下面仍是每组最慢 rank 的组间中位数，单位 ms；没有把两个代表包当作全部 8 包的默认配置平均。

| 门控方案 | 000 前后向 | 000 前向 | 005 前后向 | 005 前向 |
| --- | ---: | ---: | ---: | ---: |
| 原融合 Zigzag 基线 | 126.737 | 43.845 | 156.880 | 54.705 |
| 显式恢复串行对照 | 127.389 | 43.491 | 156.186 | 54.983 |
| 只提前 sigmoid | 126.454 | 43.616 | 156.151 | 54.839 |
| 拆分 gate，串行 | 125.896 | 43.660 | 156.057 | 54.550 |
| 拆分 gate，重叠 | 124.552 | 42.703 | 154.505 | 53.126 |

拆分重叠相对融合基线的完整模块收益分别为 1.72%、1.51%；相对拆分串行对照，分别减少 1.345、1.552 ms。
两个包的前向也有同方向收益。本次未复现上一节小 GQA 默认队列中的百毫秒长尾，不能据此认定旧长尾已解决。
复测的逐位一致性和 live allocation 检查结果与 queue0 相同。

对默认配置 shuffled_000 另取硬件 trace：8 rank、每 rank 3 step，下面为 rank/step 平均。
从硬件 kernel 的起止时间合并区间再求交集；HCCL 区间含等待，交集不是直接测得的隐藏网络传输耗时。

| 方案 | QKV/QKVG 投影 | 独立 gate 投影 | sigmoid | gate 投影与输出恢复交集 | sigmoid 与输出恢复交集 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 融合，显式恢复串行 | 3.856 ms | — | 0.160 ms | — | 0 |
| 融合，提前 sigmoid | 3.826 ms | — | 0.252 ms | — | 0.252 ms |
| 拆分 gate，串行 | 2.253 ms | 1.670 ms | 0.098 ms | 0 | 0 |
| 拆分 gate，重叠 | 2.260 ms | 1.766 ms | 0.098 ms | 1.656 ms | 0.073 ms |

这次独立 gate GEMM 提供的窗口明显大于前节小 GQA 的 K/V 投影窗口。
但拆分后的投影总 kernel 时间没有更少，而且重叠时 gate GEMM 有一定变慢；仍须用完整模块结果判断收益。
只提前 sigmoid 虽然确有区间重叠，sigmoid 本身也从约 0.160 ms 增至 0.252 ms，不能把并行区间直接记作净节省。

同一组 trace 中，普通 GQA 的不均衡 / Zigzag 对比如下：

| kernel 类别 | 不均衡，次数/step/rank | Zigzag，次数/step/rank | 不均衡累计时间 | Zigzag 累计时间 |
| --- | ---: | ---: | ---: | ---: |
| GatherV3 | 4 | 127 | 1.159 ms | 15.518 ms |
| 前向 FA | 1.875 | 8 | 3.542 ms | 8.953 ms |
| 反向 FA | 1.875 | 8 | 6.286 ms | 22.758 ms |

1.875 是不同 rank 调用次数的平均，不是单个 rank 发起小数次调用。
这些累计时间不是最慢 rank 的关键路径分解，不能直接相加解释整个模块的差值。
更多 head 放大了每次 gather 的数据量；均衡也让短文档分布到更多轮次，FA 启动和块利用率的代价仍在。
Gate 的局部约 1–2% 模块收益无法抵消这类包上的全部均衡成本。
因此后续仍优先处理统计布局转换、merge/writeback 和反向输入准备的融合，而不是仅增加 stream。

本地 profile 记录保留 kernel CSV 路径与 SHA256、投影/门控/恢复所选区间、次数校验和聚合方法。

## 大 GQA 的成本分解与优化优先级

本节补充 2026-11-02 的静态任务分析、既有设备 trace 重分析和 CPU 冷缓存诊断；没有重新运行 NPU 性能实验。
对象仍是 Q40/KV8/D128、CP8、local 8192、BF16 的 attention 模块，包含投影、norm/RoPE 和反向。
它不是完整 Qwen3-14B 训练。下述设备时间来自缓存已经预热的路径，动态文档元数据的首次准备另列。

### 有效 QK 对数相同，实际执行工作不同

对固定 manifest 的每个 rank 调用当前 `ring_task_plan`，统计实际 block/batch，而非用文档长度估计调用数。
每个包都检查所有块的有效 QK 对数之和等于 `sum(L * (L + 1) / 2)`；该计数不乘 head 数。
shuffled_000 的 64 个文档片段中，76.04% 的 token 属于完整落在一个原始 source rank 内的文档。
这些文档在普通 Ring 中只需本地计算；Zigzag 将同一文档分到多个 worker 后，更多 Ring 轮次需要计算。

| shuffled_000 的执行量 | none | Zigzag |
| --- | ---: | ---: |
| 有效 QK 对数 | 82,889,943 | 82,889,943 |
| 每 rank QK 对数的 max/mean | 1.88510 | 1.000005 |
| 前向 FA 调用数，8 rank 合计 | 15 | 64 |
| FA 内 Q/K 分段数，所有 rank/轮次合计 | 78 | 4,335 |
| FA 消费的 Q 行数，包含跨轮复用 | 71,959 | 419,588 |
| FA 消费的 KV 行数，包含前缀复用 | 74,813 | 412,655 |
| Q 分段长度中位数 | 523 | 58 |
| KV 分段长度中位数 | 505 | 56 |

4,335 是 varlen FA 内的逻辑分段，不是 4,335 次 kernel 启动。
Q 行消费量增加约 5.83 倍、KV 增加约 5.52 倍，不意味着有效 QK 对重复计算：
同一 Q 需要和不同 owner 的 KV 合并，同一 KV 前缀也可能出现在不同 Q 分段的操作数中。
原来较大的 attention 被拆成更多小问题，产生更多读写、启动、统计转换和在线合并。
分段变小对 FA 利用率的影响是性能解释，实际 tile 浪费比例尚未测量，不能把某个长度阈值直接当成硬件 tile 大小。

这解释了为什么不同文档组成会有相反结果：

- shuffled_000 默认队列的未 profile 完整模块时间为 100.515 → 118.541 ms；原本较多文档只在本地计算，均衡收益有限。
- shuffled_005 含一个 26,023-token 片段，QK 对数的 max/mean 为 3.4708 → 1.000002，
  同配置时间为 162.834 → 149.599 ms。消除较大的原始偏斜足以抵消这次的附加成本。
- long_000 的有效 QK 对是 shuffled_000 的约 25.9 倍；dense 快速路径只有 36 → 64 个大分段，
  Q 分段中位数仍为 8192。queue0 时间为 349.290 → 234.783 ms，下降 32.78%。

这些配对分别来自前文注明的队列设置，不能混成同一组总体统计。
无需逐文档填充、支持真实余数是当前方法的语义优势，但不自动保证执行成本低。

### 重叠增加仍可能变慢

对 shuffled_000 的普通 GQA trace，以每步第一个 QKV GEMM 开始至第六个、即最后一个反向 GEMM 结束作为窗口，
排除外层 profiler barrier。下面先对 kernel 时间区间取并集/交集，再对 8 rank × 3 step 求均值，单位 ms。

| 区间统计 | none | Zigzag |
| --- | ---: | ---: |
| 模块设备窗口 | 100.755 | 120.383 |
| 至少有一个计算 kernel 的区间 | 35.231 | 88.688 |
| 计算与 HCCL 同时活跃的区间 | 9.509 | 54.922 |
| HCCL 活跃但没有计算 kernel 的区间 | 65.464 | 31.633 |
| 两类 kernel 都不活跃的区间 | 0.060 | 0.063 |

这组统计中，计算覆盖时间增加 53.456 ms，HCCL 活跃而无计算的时间减少 33.831 ms，窗口仍增加 19.628 ms。
因此不能把增加重叠比例作为唯一优化目标。
HCCL 区间含 peer 等待、同步和排队；这里的“无计算”不等于纯链路传输耗时，也不是已经证明的依赖阻塞时间。
各类 kernel 累计时间同样不能与 HCCL 累计时间直接相加。端到端收益仍以前文无 profiler 的最慢 rank 计时为准。

算子层面，投影 MatMul 累计时间基本不变：11.905 → 11.913 ms。
变化主要在 Gather（1.159 → 15.518 ms）、FA 前向（3.542 → 8.953 ms）、
FA 反向（6.286 → 22.758 ms），以及 merge、类型转换、梯度累加和写回。
这比“模型变大导致投影更慢”更能解释同一 GQA 形状下开启均衡的差值。

本配置每 rank 的完整 BF16 Q/O 各为 80 MiB，在线合并的 FP32 输出状态为 160 MiB，
max/sum 的 native 8-lane FP32 统计各为 10 MiB。反复 gather、转换和逐元素更新会多次读写这些状态，
所以这里既有小 kernel 启动成本，也有显著的数据搬运成本；大 GQA 的多 query heads 会放大后者。

### GQA 的重排成本与普通 Ring 的冗余传输

本配置每 rank 的 BF16 K+V 包为 32 MiB，FP32 dK+dV 包为 64 MiB。
前后向共发送 `7 × 32 + 7 × 32 + 8 × 64 = 960 MiB` 的 Ring payload。
这是逻辑发送字节，不包含接收的重复计数，也不是物理链路总流量。
none 和 Zigzag 的这部分容量相同；当前普通 Ring 即使某轮没有合法 QK 块，也保持相同轮转协议。

Zigzag 另有输入 QKV 重排、输出恢复及两者反向；shuffled_000 每 rank 平均增加 333.791 MiB 的逻辑发送。
大致对应前向 QKV 98 MiB、O 70 MiB，反向再做相应交换。
普通 Ring 只传 KV8 heads，均衡还搬运 Q/O40 heads，因此在 GQA 下附加路由尤其值得控制。
所有数据恢复原始 owner 后再进入下一层，兼容异构模型；此分析没有假设布局跨层保持。

另一个优化方向是保留文档局部性：完全本地的文档只做本地 FA，仅对跨 rank 的文档片段安排必要传输。
这需要全组共同生成通信计划、保证每条 P2P 匹配，并重新验证 dKV 归属，不能由某个 rank 自行跳过收发。
76.04% 的局部 token 比例表明存在优化空间，但不能直接推导等比例的端到端收益。

### 动态文档布局有单独的冷缓存成本

当前 `_packed_indices` 用 Python 为 Q/K 逐 token 建索引，并为 softmax 统计逐 token、逐 head 展开 encode/decode。
LRU 缓存能摊薄重复布局，但真实 packing 的新长度组合首次使用仍需准备。
本次在单独 CPU 进程、`OMP_NUM_THREADS=4`、rank 0、Q40 下分别清空计划/索引缓存，测 3 次冷准备并立即重复热查询：

| 包 / 布局 | 任务计划中位数 | CPU 索引创建中位数 | 热索引查询中位数 |
| --- | ---: | ---: | ---: |
| shuffled_000 / none | 2.475 ms | 211.932 ms | 0.047 ms |
| shuffled_000 / Zigzag | 11.965 ms | 1025.013 ms | 0.416 ms |
| shuffled_005 / Zigzag | 8.838 ms | 996.401 ms | 0.286 ms |

索引测量使用 CPU tensor；冷准备包含均衡路由索引，热查询只重复 packed 索引访问。
不含 H2D、NPU 执行、分布式等待或数据加载，因此它是 host 诊断，不是实测训练吞吐下降一秒。
同一 batch 的相同布局和 head 数可在多层复用，也不能理解为每层都增加一秒。
此前固定包 benchmark 的缓存已预热，表中的 100–150 ms 模块计时未包含该冷准备成本。
动态 packing 的发布验收必须同时记录冷/热路径和缓存内存占用。

### 优化顺序与完成条件

1. **先控制计划成本和碎片数量。** 将逐 token/head 的 Python 展开改成区间描述与向量化/设备端寻址，
   同 batch 的兼容层复用元数据。文档边界已知时，可提前在 CPU 准备区间；设备准备若异步执行，使用显式 event 建立消费依赖。
   保留短文档局部性，长文档按需要分片；选择计划时计入每轮 FA 次数、分段形状、重复 Q/KV 读取、路由字节和最大 rank 时间。
   首先保留显式选择和 `none` 默认值，不用未经验证的长度阈值宣称自动最优。
2. **融合统计转换与合并写回。** 当前前向依次进行统计 decode、旧状态 gather、多个逐元素 merge、scatter。
   可让一个 kernel 直接按文档区间读取 native 统计，更新 FP32 状态并写回对应 Q 行。
   第 j 个分段、head h、段内行 i 的 native 统计地址由 `prefix_q[j] * H + h * q_len[j] + i` 给出，
   无需先生成完整的逐 head decode/encode 索引。必须保留 native 统计的 8-lane 布局契约及反向所需信息。
   MindSpeed 已提供支持 TND 元数据的
   [Ring update 融合接口](https://github.com/Ascend/MindSpeed/blob/master/docs/zh/ops/npu_ring_attention_update.md)，
   可作为实现参考；它并不自动支持本实现的任意行映射，尚未验证直接适配或性能。
   本机 `torch_npu.npu_attention_update` 的输入是 LSE/output 列表，接口也不等同于这里的 max/sum 状态，不能直接替换名字。
3. **减少反向和路由准备。** 融合 Q/O/dO 与统计的 gather/encode；对于重复出现的 query 行模式，
   在本次调用内有界复用不变输入的打包结果。不能缓存正在变化的前向 output/max/sum，也不保留所有远端 KV。
   再评估 QKV concat+gather 的一次性 route pack、输出恢复写回融合。连续区间已有 narrow 快速路径，无需重复实现。
   通用 planned 反向还可避免空计算轮的 FP32 delta 清零与加零；P2P 协议保持一致。
   每行首次有效贡献可直接初始化状态，跳过一次通用 merge，但必须按行处理首次贡献。
4. **随后改通信调度。** 先验证文档感知的有效 KV 传输，再按真实拓扑检查链路带宽和等待。
   gate 投影与输出恢复的重叠有约 1–2% 模块收益，但不能隐藏每轮 KV 轮转；单独提前 sigmoid 没有稳定净收益。
   额外 stream 可能争用显存带宽，不能代替减少实际工作量。

每项优化用相同文档组成和同一队列配置做配对比较，记录最慢 rank 端到端时间、每轮时间、kernel/分段数、
实际 pack/route 字节、峰值显存以及动态输入的 p50/p95。融合的累计 kernel 时间下降只能作为过程指标，
最终以无 profiler 的完整模块/Trainer 收益和输出、梯度、参数更新为准，不预先承诺可相加的毫秒收益。

## 首批上库范围与依赖拆分

建议第一份 PR 合入纯 Ring 基础、可切换通算重叠、模型拥有的 adapter 和最小训练闭环。
当前研发代码需要先按依赖拆分。本节是正式上库的拆分计划；研发 PR 保留实现、测试、基准脚本和分析文档，
原始结果只在本地保存，不表示已经完成这些拆分。

| 第一份 PR 的组成 | 对外行为与边界 |
| --- | --- |
| Ring 前后向 | NPU、FP16/BF16、BNSD/BSND、MHA/GQA、因果/非因果；保留 compact KV heads |
| 梯度与缓冲生命周期 | 在线 softmax、FP32 梯度累加、dKV 返回所属 rank；梯度不做 CP-size 除法 |
| 可选 overlap | KV 预取、dKV 流水、合并 K/V 和 dK/dV 传输、双缓冲复用；同步路径保留 |
| 模型接入 | 通过现有 `attention_interface` 和 recipe forward rewrite；投影、norm、RoPE、gate 由模型拥有 |
| 用户入口 | 明确 `ring_attention` 的导入路径、参数与限制；默认行为等价于当前 `load_balance=none` |
| 验证与文档 | 独立数值对照、2/4/8 卡 ST、模型层与 Trainer/FSDP 回归、可离线复现的小型性能用例 |

API 只接受本地 Tensor、各 rank 等长且选项一致的 source shard、同维度 K/V、dropout=0。
首批明确不支持任意 mask、KV cache、GPU、MLA、不等长 source shard 及未经验证的 TP 输入组合。
GQA 的 Q/KV 比例需合法，但纯 Ring 的 KV head 数不要求整除 CP；头数整除 Ulysses degree 是后续混合接口的限制。
overlap 保留开关和同步对照，示例可显式启用；不能将开关描述为已经完全隐藏通信。

接入顺序为 `QKV 投影 → QK norm/RoPE → attention_interface(Ring) → 模型后续 gate/O 投影`，
具体 gate 位置遵循模型本身。沿用当前模型 adapter/recipe 机制，不引入 CP `ParallelStyle`，也无需改 Trainer 或 FSDP 核心。
模块退出时恢复原始序列归属，不要求异构层持续使用同一布局。

### 不能直接按现有文件提交的原因

- `ring_attention.py` 同时包含 pure Ring、TND 调度、四种均衡方案和 Ulysses 组合，并导入 `ring_layout.py`。
  普通 dense 路径目前也先构建 task plan 再走快速分支；首批需提前走基础路径，切断对通用 planner 的依赖。
  调试 trace 若保留，应放入不依赖文档路由的轻量定义。
- 模型 adapter 即使 `ulysses_degree=1` 也统一调用混合入口。首批改成直接调用 pure Ring，
  后续混合/均衡参数随各自功能一起增加，避免提前固定实验接口。
- 现有 `indexer_balance.py` 已改为复用 `query_balance.py`，这会改变已有 v4.1 Indexer 的依赖。
  第一份 Ring PR 不附带该迁移；共享调度可独立做保持行为不变的重构，并跑 Indexer/DSA 回归。
- 现有 Trainer recipe 默认是 dense Zigzag，并包含混合并行配置。
  首批应取 pure Ring、等价于当前 `none` 的配置重新验证，不能仅引用原 recipe 的历史通过记录。
- benchmark、原始 profile 与实验调度较多。首批保留精简公开用例、必要文档和复现方法，
  私有数据路径不成为 UT/ST 前置条件；原始 profile、输入清单和结果 JSON 只在本地保存。

### 合入前的明确完成条件

1. 拆分后的导入和默认 dense 路径不依赖未合入功能；公共导出、模型 recipe 与参数校验一致。
2. 2/4/8 卡覆盖前后向、owner 梯度、非连续 group rank、非连续输入、部分 requires_grad、两种 dtype/layout，
   同步与 overlap 的正确性、重复调用显存稳定性和 side stream 同步。确认 ST 确实进入 CI 作业，不能只凭 level1 标记。
3. 将 Q40/KV8/D128 的独立 oracle 数值验证补成可维护的用例；完整 local 8K 用于性能和显存检查，
   独立高精度数值对照可用较短序列。现有大 GQA benchmark 比较多个 Ring 调度，不等同于独立全局 attention 对照。
4. pure Ring + Trainer/FSDP 重新跑 loss、首步全局梯度、grad norm、参数更新及同步/overlap 多步轨迹；
   保留已有激活重计算和两个 micro-batch 的梯度累积。此项已有小模型 256-token、3-step 测试基础，需按首批配置重跑。
5. 性能说明限定设备、输入组成、缓存状态和测量范围；精度看分布及训练影响，少量元素越过历史阈值不单独阻断，
   但错序、漏算、重复累加、通信竞争和非有限值必须排除。

本轮实际重跑：Ring/任务计划/模型 adapter 三个 UT 文件共 22 项通过；
`TestSharedCompressedCPPlans` 的 3 项通过；Ring、layout、query planner、模型 adapter 四个新模块的 pylint 通过。
这些检查没有替代最终拆分代码上的分布式 ST。Indexer 的 3 项计划测试也不等于完整 DSA 数值回归。

### 后续提交顺序

| 次序 | 独立功能 | 进入稳定接口前的重点 |
| --- | --- | --- |
| 2 | Packed TND 文档语义与 none 基线 | 全局边界/位置与框架 packed 元数据接通；冷准备优化；跨 rank 文档；真实 packed local 8K Trainer/FSDP |
| 3 | 显式可选 Zigzag | dense 快速路径和不规则 TND 分别验收；统计/写回优化；短文档退化和长文档收益公开；不默认全开 |
| 4 | Ulysses + Ring 组合 | 薄组合层、子组/rank 顺序、局部 head 约束、2×2/2×4/4×2 与训练回归 |
| 后续 | Greedy/Hybrid、投影/gate 调度、文档感知通信 | 根据真实模块成本而非 QK 对数选择；先证明净收益，再固定接口 |

dense Zigzag 若已独立完成依赖拆分与回归，可以单独提前，不必等待全部 packed 融合优化。
实验性可选路径不要求在所有文档分布上都胜过 none，但必须精度可靠、退化可解释且默认行为明确。
GPU、MLA、更广的 TP/EP/PP 组合和长时间训练/checkpoint 恢复属于进一步扩展，不能从已有短训练推导支持结论。

## Rebase 后的研发 PR 验证

2026-11-02 将研发分支从 `d666fdbf` rebase 到 `origin/master@1e6f973c`，无冲突。
当前 PR 保留研发代码、测试、基准脚本和本指南，后续正式上库范围按上一节拆分。
提交整理只补充 Indexer 的两处参数文档和本指南，计算逻辑未改变。
原始 JSON 已从提交中移除并加入忽略规则，本地保留原测量结果及源码指纹。

- CPU 回归：CP UT 目录、Qwen3-MoE Ring adapter 和共享 compressed DSA 模块，共 **54 passed**。
- 分布式回归：单卡 FA 契约、2/4 卡基础 Ring、4 卡 Ulysses+Ring、8 卡 overlap、8 卡 packed/均衡、
  4 卡模型 dense/packed 接入、CP8 Ring+FSDP8 Trainer，共 **10 个 launcher 用例通过**。
  内部参数组合包含更多 worker 测试；10 不是内部参数组合总数。
- Trainer 用例分别启动 AllGather control、同步 Ring、overlap Ring，使用现有 dense Zigzag recipe；
  包含激活重计算和两个 micro-batch 梯度累积。它不替代未来 none recipe 和真实 packed local 8K 的验证。
- editable 安装和导入路径均指向当前 checkout；29 个变更 Python 文件使用仓库相同 pylint 规则、4 个并行 worker 检查通过。
  单进程全量 pylint 曾超过 AutoGit 的 120 秒工具上限，并行复查未放宽检查规则。
- 完整 AutoGit 静态检查尚未全绿：函数复杂度、公共/测试函数的参数文档仍需收口。
  CPU UT 的 `arg_mark`/ST 文档格式检查与 `.agent/rules/unit-test.md` 的 UT 约定不一致，须校正扫描范围；
  Torch-only 文档中的原生 API 名也会触发遗留的后端名检查。未为消除这些提示而给 CPU UT 增加 NPU 标记。
- 性能数字沿用此前已保存的测量，本次 rebase 没有重测完整性能矩阵。

## 精度记录与验收方式

按当前约定，少量元素略超旧阈值不单独阻止性能实验。保留以下诊断，结合偏差大小和训练影响判断：

- 有限性、误差的最大值及分布、相对 L2、超阈值元素比例。
- 输出、dQ、dK、dV 分开观察，并比较 native FA、普通 Ring 和均衡 Ring。
- 区分 BF16 累加顺序变化与错序、漏算、重复累加、跨 stream 未同步等逻辑错误。
- 最终结合完整训练的 loss、梯度范数、参数更新和收敛判断。

旧诊断为 BF16 atol=0.015625、rtol=0.06、relative L2≤0.0078125。
10 包 CP8 测量中，none 所有元素通过；Zigzag 的 dK 有 2 个、dV 有 6 个元素超过逐元素阈值，分布在 5 个包。
O 与 dQ 没有超阈值元素。dK+dV 共评估 671,088,640 个元素，超阈值比例约 1.19e-8。
该比例的分母包括 10 包、8 rank 和两个梯度张量，不能误解为单包内比例。

| 路径 | 全部结果中最大相对 L2 | O 最大绝对误差 | dQ 最大绝对误差 | dK 最大绝对误差 | dV 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.002973 | 0.01098 | 0.02195 | 0.03634 | 0.06299 |
| Zigzag | 0.003037 | 0.01521 | 0.02612 | 0.03662 | 0.06250 |
| 主块+余数 | 0.003037 | 0.01521 | 0.02612 | 0.03427 | 0.06326 |

相对 L2 是各包、各 rank、各输出/梯度张量中的最大值。
原报告没有保存所有逐元素误差分位数或每个元素超过阈值的倍数，因此不能从这些汇总量推断每处只超了多少。
主块+余数也有 8 个超阈值元素，但位置改变；计数相同不代表误差等价。
没有修改既有历史测量或把旧的“未通过”改写成“通过”。

## 后续工作

1. 消除动态 packed 索引的逐 token/head Python 展开，验证冷/热路径；按实际 FA 成本改进均衡决策，保留短文档局部性。
2. 优先融合统计 encode/decode、前向 merge/writeback、反向输入 gather，再评估 route pack；减少空轮清零与加零。
3. 大 GQA/GatedGQA 的模块实验已包含 norm/RoPE；将有稳定净收益的调度接到模型拥有的 adapter，
   验证其与 TP、重计算、动态 packed 元数据的组合。Gate 调度要位于能访问本地 gate 的模块边界。
4. 使用真实 packed 数据做 local 8K Trainer+FSDP 验证，记录吞吐、峰值显存、长尾、梯度累积和重计算。
5. 定位默认队列长尾，保留所有配对重复值；随后扩展多机、更大 CP、长时间训练、checkpoint 恢复及模型所需的 TP/EP 组合。
6. 后续推广 GPU 和 MLA Ring；dropout、更多 mask 形式、推理缓存按实际模型需求扩展。
7. 按上述依赖边界先整理 pure Ring + overlap + 模型/训练闭环，TND、均衡和混合分别提交。
   当前主块+余数、投影和 gate 调度属于实验，未改变默认核心路径。

## 复现与结果文件

环境：Ascend 910B3 × 8，CANN 9.0.0，torch 2.7.1+cpu，torch_npu 2.7.1.post4。
editable import 已确认指向当前 checkout。硬件测量需在设备空闲时执行，不并行启动两个 benchmark。

输入 manifest、原始结果 JSON 和 profile 输出均为本地实验产物，不随 PR 提交。
先将 `RING_DATASET_PREFIX` 改为本机 indexed dataset 前缀（不带 `.idx`/`.bin` 扩展名），生成 manifest：

```bash
source /usr/local/Ascend/cann/set_env.sh
RING_DATASET_PREFIX=/path/to/indexed_dataset_prefix
.venv/bin/python -m tests.torch.context_parallel._benchmark_ring_attention_dataset \
  --prepare --dataset-prefix "$RING_DATASET_PREFIX" \
  --manifest /tmp/ring_dataset_manifest.json \
  --cp 8 --local-tokens 8192 --batches 8 --seed 20261101
```

上述命令只准备文档边界，不启动分布式测量。使用相同索引与 seed 可重建本文的 10 个输入包；
换用其他数据集后，输入组成和耗时需要重新测量。接着运行投影实验：

```bash
source /usr/local/Ascend/cann/set_env.sh
OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=0 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  tests/torch/context_parallel/_benchmark_ring_projection_prepare.py \
  --manifest /tmp/ring_dataset_manifest.json \
  --output /tmp/ring_projection_queue0.json --repeats 7 --iterations 8
```

默认运行配置复测时去掉 `TASK_QUEUE_ENABLE=0`，并确认 shell 没有预先设置该变量。
增加 `--forward-repeats 7` 可同时记录前向；增加 `--profile-dir /tmp/ring_projection_profile` 可导出所选包的 trace。
固定长尾包复测使用 `--cases shuffled_005 --repeats 11 --iterations 8 --forward-repeats 7`。
大 GQA / gate 实验可运行：

```bash
source /usr/local/Ascend/cann/set_env.sh
OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=0 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  tests/torch/context_parallel/_benchmark_ring_gated_gqa.py \
  --manifest /tmp/ring_dataset_manifest.json \
  --output /tmp/ring_gqa14b_queue0.json --repeats 7 --iterations 4 --forward-repeats 5
```

默认配置复测去掉队列环境变量，增加 `--cases shuffled_000 shuffled_005`。
增加 `--profile-dir /tmp/ring_gqa14b_profile` 可采集门控四种调度；本节同时采集普通 GQA 对照时使用
`--profile-modes plain_none plain_zigzag gated_restore_serial gated_sigmoid_overlap gated_split_serial gated_split_overlap`。
报告中的源码指纹保留实际运行版本；queue0 与默认复测之间只有报告字典初始化语句的换行格式调整，汇总文件记录了此差异。

不要混合不同 queue 配置、不同输入、attention-only 与投影完整块的耗时。
本次脚本在计时前检查确定性输出和投影梯度；计时使用普通非确定性 kernel，轮换方案顺序，逐组取最慢 rank。

原始输入、历史结果和 profile 记录在本地单独保管；仓库忽略 `docs/guide/ring_attention_data/`，
避免将生成数据再次加入提交。上面的命令使用 `/tmp` 输出路径，需要长期保留时改为本地实验目录。
主块+余数代码仍保留在原临时实验目录；本文记录其历史结论，不表示该分支已成为框架公共接口。

## 附录：每包完整长度

以下顺序就是 attention 中的 packed 顺序；所有长度均以 token 为单位，每包总和 65,536。

**shuffled_000**

```text
[703, 1990, 397, 31, 152, 105, 5897, 3701, 1228, 1216, 284, 507, 522, 1624, 489, 471, 653, 2207, 2019, 413,
38, 1101, 530, 516, 1807, 1327, 6100, 2906, 683, 893, 1462, 2184, 481, 269, 147, 155, 621, 261, 1039, 1109,
115, 252, 137, 519, 1184, 1152, 222, 503, 121, 904, 1251, 533, 1080, 460, 463, 787, 4687, 184, 78, 842, 414,
43, 175, 1192]
```

**shuffled_001**

```text
[19533, 4404, 284, 9292, 1201, 242, 767, 208, 992, 1214, 1555, 507, 133, 52, 474, 823, 527, 5492, 494, 33,
1904, 453, 529, 1187, 192, 138, 215, 5753, 74, 53, 575, 760, 149, 2633, 389, 416, 462, 590, 593, 244]
```

**shuffled_002**

```text
[820, 334, 2379, 430, 330, 2952, 1590, 5835, 58, 1706, 1170, 209, 536, 56, 733, 730, 960, 1496, 407, 747,
1129, 1161, 1378, 454, 374, 543, 130, 83, 171, 197, 1015, 1090, 218, 498, 947, 581, 117, 1381, 1262, 606,
81, 673, 139, 535, 37, 103, 87, 175, 72, 474, 651, 3443, 192, 4891, 391, 232, 58, 908, 1107, 720, 388, 103,
344, 1515, 1652, 61, 141, 1278, 892, 330, 2309, 245, 48, 155, 116, 719, 1808, 1650]
```

**shuffled_003**

```text
[454, 953, 504, 1294, 956, 240, 2132, 550, 355, 913, 506, 58, 250, 499, 727, 75, 241, 96, 2196, 184, 294,
190, 55, 939, 1193, 914, 1716, 3243, 769, 396, 521, 870, 115, 529, 83, 147, 1010, 546, 910, 406, 728, 3485,
39, 513, 1005, 403, 353, 172, 223, 182, 536, 805, 849, 157, 286, 251, 468, 1588, 382, 2097, 737, 1046, 341,
149, 70, 60, 594, 195, 1663, 1485, 1022, 4847, 629, 203, 273, 172, 3185, 178, 1149, 820, 2622, 811, 298,
335, 101]
```

**shuffled_004**

```text
[165, 11408, 1166, 156, 1502, 810, 462, 939, 614, 1395, 835, 701, 2352, 739, 654, 873, 2448, 797, 1124,
1633, 228, 1021, 312, 752, 2135, 928, 1203, 129, 243, 1589, 616, 2257, 1190, 1398, 252, 297, 3744, 196, 564,
3142, 506, 281, 3056, 3184, 87, 291, 2836, 2326]
```

**shuffled_005**

```text
[26023, 196, 964, 36, 449, 2260, 974, 199, 1195, 890, 133, 448, 112, 3881, 1008, 205, 185, 5146, 350, 96,
232, 2488, 470, 733, 268, 565, 294, 133, 239, 1370, 1083, 515, 416, 319, 950, 1345, 887, 5171, 542, 562,
1059, 667, 319, 159]
```

**shuffled_006**

```text
[487, 459, 1092, 263, 252, 479, 997, 811, 258, 1917, 8165, 188, 337, 1473, 2869, 3239, 153, 5466, 500, 744,
362, 163, 1325, 907, 1517, 89, 6047, 233, 100, 895, 291, 1093, 1568, 912, 450, 1221, 779, 6811, 1805, 90,
1642, 326, 358, 1122, 2228, 823, 819, 1411]
```

**shuffled_007**

```text
[3527, 93, 530, 8019, 863, 667, 1506, 167, 1447, 466, 200, 151, 1146, 369, 1127, 945, 144, 492, 1907, 9373,
484, 137, 168, 1289, 434, 450, 144, 1181, 254, 1185, 294, 123, 1325, 956, 983, 801, 83, 1485, 578, 6412,
1334, 744, 2303, 4657, 289, 1321, 652, 333, 1405, 269, 324]
```

**ordered_000**

```text
[5383, 16850, 15133, 4580, 12093, 6133, 5364]
```

**long_000**

```text
[65536]
```
