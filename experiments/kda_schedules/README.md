# KDA 分块调度与状态通信实验参考

本目录整理 V-block、双流调度、分组 AllGather/P2P 和三角 AllToAllV 对照，供后续研究复现。
只增加四个 Python 文件和本说明，不修改生产接口、默认算法、依赖清单或 UT/ST 门禁。
它是 **Triton staged KDA core** 的参考入口，不是完整 attention 层或训练器。

## 来源与边界

- 基线：Hyper-Parallel `96143af9b4ea12ddf0bd3854fc521ca86df7cc90`。
- 历史调度来源：`kimi_k3_large_cp_new_arch@9b5c253be489cb78afe0b91ff53675afe0142fd5`
  中 `experiments/kda_cp/runtime/chunk_kda.py`。这里只迁移独立调度，不搬旧 platform 层、
  综合入口、结果目录、分析脚本或整套 kernel。
- Local KDA 通过当前 `get_fla_kda_staged_ops()` 调用外部 FLA；S/M/G 复用当前 Hyper kernel。
  本目录没有复制、修改 FLA 或 Triton kernel。
- `stages.py` 和 AllToAllV 使用固定基线的 staged/internal scan 契约；后续升级依赖须重跑验证，
  不是向框架承诺长期稳定的新 public API。
- 固定 dense `B,T,H,128`，`Dk=Dv=128`、`chunk=64`、`T % 64 == 0`；token BF16，
  参数 `A_log/dt_bias` 和状态 FP32，`scale=128**-0.5`、`lower_bound=-5`。
- 当前仅 WORLD 按时间顺序切 sequence。等 Q/K/V head 数，不含 GQA、TND/varlen、推理 cache、
  外部初态/末态 API、FSDP/TP、投影/卷积/output norm 或 optimizer。内部跨 rank h0/dHT 非零。
- 支持一阶反向和 non-reentrant checkpoint。同一通信组各 rank 必须使用一致的调用/反向顺序；
  不承诺任意并发线程、不同 rank 各自乱序反向或高阶梯度。

## 模式对照

| 模式 | 准备与状态交换 | 保留目的 |
|---|---|---|
| `p2p` | 当前生产 P2P，不分块 | 基线 |
| `v-packet` | 完整 S/M/G 各计算一次；H/dH 沿 V 列分包，共用 M | 隔离 transport/apply 分块收益 |
| `m-once` | M 一次；各 V block 单独生成 S/G；全 head prepare/finish | 显示重复 producer 扫描的成本 |
| `heads-whole` | 每个 head block 独立执行完整生产 P2P autograd Function | whole-core 对照 |
| `heads-single` | 单流按 head prepare/summary/state，全部完成后 finish | state-first 对照 |
| `dual` | producer/state 两条工作流；先提交所有 producer，再提交 consumers；逐 block 汇合 finish | 双流基线 |
| `dual-drain` | 同 dual，但所有本地 state send 已建立完成依赖后才开始任何 finish | 检验 finish 争用是否拖慢状态链 |
| `dual-immediate` | 每个 producer 提交后立即下发对应 consumer；后续 producer 不等远端 state | 检验 host 下发时机 |
| `allgather` | 当前生产全量 AG 与 fused merge | 集合通信基线 |
| `grouped` | 当前生产组内 AG、组间 owner P2P、组入口广播和本地 fused scan | 层次协议，无重复旧实现 |
| `alltoallv` | 前向只发给后继，反向只发给前驱；保持 source 顺序，用当前 fused scan | 不均匀通信的负结果/对照 |

`--blocks` 可选 1/2/4/8，只作用于分块模式；分 head 时必须整除 H。
`--group-width` 只作用于 grouped，要求至少 2 且整除 CP。进程组在计时之前统一创建。

`m-once` 是迁移后的对照实现：窄 U/do/dv 显式 contiguous，再用当前 kernel 的 V 参数运行。
这比历史带 VALUE_OFFSET 的实验 kernel 多了显式 packing，也采用当前 FP32 summary 算术，
不能拿旧 kernel 的毫秒数当作本代码性能。它确实只生成一次 M，仍为每个 V block 扫描 K/W/g。
`v-packet` 不重复这个扫描，但也不会提前第一个 summary 的 ready 时刻。

双流使用 event、Work.wait 和 record_stream 维护生产者/接收/消费者依赖，保存 send buffer 到
通信完成；每次调用独占 tensor workspace，executor 只保存通信元数据。双流不代表一定存在
设备重叠，长 kernel 的资源争用仍需 Level0 验证。本版 drain 包含实际 send work 的等待依赖，
比旧版仅等待 state-forwarded event 的条件更严格。

AllToAllV 可以减少平均逻辑复制量，却不能改变末端 rank 接收 O(CP) 份 summary 的最坏情况；
还需 repeat send buffer 和不均匀 split。它不是 logarithmic scan，更不是单边通信。

## 文件

| 文件 | 用途 |
|---|---|
| `protocols.py` | V packet 链、三角 AllToAllV、已有 AG/grouped 构造入口 |
| `stages.py` | 与当前正式 KDA 相同的 FLA prepare/finish 调用及 V producer launcher |
| `schedules.py` | head 单流/双流和 M-once 的完整一阶 autograd |
| `run.py` | 唯一显式验证/benchmark 入口，不被 pytest 自动收集 |

## 运行

先在匹配的 Ascend 环境中加载 CANN `set_env.sh`，安装 PyTorch、torch-npu 和 Triton-Ascend。
本轮验证环境为 910B3、CANN 9.1.0-beta.3、PyTorch/torch-npu 2.10.0、
triton-ascend `3.2.2+gitc22984ba`。不设置 UB capacity override，不静默回退。
FLA 是独立安装的干净快照 `6dff3e1995195d6f192b1b01f5ddf66be7cc0a35`，不是任意 main 的兼容保证。
FLA 的普通依赖仍须按其安装说明准备；`--no-deps` 用于避免替换已有的 NPU PyTorch 工具链。

```bash
python -m pip install -e .
git clone https://github.com/fla-org/flash-linear-attention.git ../flash-linear-attention
git -C ../flash-linear-attention checkout 6dff3e1995195d6f192b1b01f5ddf66be7cc0a35
python -m pip install --no-deps -e ../flash-linear-attention

# 从 Hyper-Parallel 仓库根目录运行。CPU 检查不导入 Triton/FLA。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m experiments.kda_schedules.run --cpu-protocols

# 默认验证全部 11 个模式，global T = CP * local T。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m experiments.kda_schedules.run \
  --local-tokens 128 --heads 8 --blocks 2

# checkpoint 重放及两次 forward 后逆序 backward。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m experiments.kda_schedules.run \
  --checkpoint --inflight 2 --blocks 4

# 先审计设备独占；此开关只是确认，不会自动停止其他进程。
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m experiments.kda_schedules.run \
  --benchmark --exclusive-devices-confirmed --local-tokens 8192 --heads 96 \
  --modes p2p v-packet m-once dual dual-drain dual-immediate allgather grouped alltoallv
```

精度 reference 是同输入的完整 global sequence FLA local KDA，而不是另一种 CP 的结果。
逐项比较 output、dq/dk/dv/dgate/dbeta/dA_log/ddt_bias，参数梯度先做跨 rank sum。
使用慢衰减 gate 使跨 rank 状态和状态梯度有实际贡献。阈值固定 relative-L2 <= 0.02，
并检查所有结果 finite。CPU 另外验证非交换 affine map 的正向 prefix/反向转置 suffix。

benchmark 输出 forward 和完整 forward+backward 平均时间的跨 rank 最大值，以及 peak allocated。
不包含参数梯度通信、optimizer 或完整 attention 外层。验证不等于性能收益。

## 验证与历史结论

2026-10-30 对当前迁移版验证如下，所有数值均为跨 rank、跨输出/七类梯度的最大 relative-L2：

| 配置 | 范围 | 结果 |
|---|---|---|
| CPU/Gloo CP4/8 | V1/V2/V4/V8 和三角 AllToAllV，正向/反向 | 全部通过 |
| NPU CP4，local128/global512，H8，blocks2 | 全部 11 模式 | 最大 `0.003077` |
| NPU CP4，local128/global512，H8，blocks4，g2 | 全部模式，checkpoint，两次 forward 后逆序 backward | 最大 `0.003557` |
| NPU CP8，local128/global1024，H8，blocks4，g4 | 全部模式，checkpoint，两次 forward 后逆序 backward | 最大 `0.003482` |
| NPU CP4，local128/global512，H96，blocks8，g2 | 全部模式，checkpoint，另一随机种子 | 最大 `0.002962` |

尚未在本迁移版重跑 local8K/H96 的稳定性能和内存快照，也没有 CP16+ 或跨机结果。
因此这里只验收归档接口、数学接线和上述数值/重放范围，不宣称双流或 V 分包获得加速。

历史 910B3 `B1,H96,D128,local8K` 的 Full S/G once V2 在 CP4/8 没有稳定加速；
旧 CP8/64K 的 P2P / AllToAllV / AG 完整 core 步分别约 235.121 / 233.172 / 232.818 ms。
这些只说明为何保留对照，不是本分支当前测量，也不能跨 FLA/summary 版本直接比较。

## 相关参考 PR

- [AscendC stateful CP、分 head 和内存变体 #6](https://github.com/xuxl1008/hyper-parallel/pull/6)。
- [AscendC TND Ulysses #7](https://github.com/xuxl1008/hyper-parallel/pull/7)。
- [cached Tree、head RD 与 g2 边界原型 #9](https://github.com/xuxl1008/hyper-parallel/pull/9)。

本目录不重复上述实现。设备端 All-Scan/ACLSHMEM、producer-summary 片上融合、自动 planner、
reset-aware TND state CP 仍是后续课题，不作为本次已完成代码交付。
