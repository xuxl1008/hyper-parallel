# KDA AscendC 状态并行参考实现

这是 [fork 参考 PR](https://github.com/xuxl1008/hyper-parallel/pull/6) 的最小运行代码，
包含 P2P/AllGather、分 head 和内存变体。实验表和历史分析放在 PR 描述中；
不随源码分发历史 JSON、快照、日志、wheel 或独立 profile 工具。

## 支持范围

- 固定 Hyper 基线 `53788bdd5e3c9d05a6a3401ad3a9e47364a624fe`。
- FLA-NPU 基线 `47d7c97ce439228e0ead6fab4dad8b0440b663a8`，需要应用
  [非零 dHT/FP32 状态补丁](patches/README.md)，不是未修改上游即可运行。
- 已验证：910B3/A2、CANN 9.1.0-beta.3、Torch/torch-npu 2.10.0、Triton-Ascend 3.2.2；
  KDA 为 BF16 token、FP32 state、Dk=Dv=128、chunk64、固定长度、按序 WORLD。
- 完整 attention 包含 projection、ShortConv、KDA core、output gate/norm/projection。
  使用 AscendC core、Hyper Triton summary/merge 和 HCCL，不是纯 AscendC CP。
- 公开 `backend="ascendc"` 只接 local/Ulysses；P2P/AllGather 使用本目录实验执行器。
  不放开正式 State-CP guard，不改变 eager/triton 默认行为。
- 尚未验收：TND/varlen 整层、RD/Tree/subgroup 接线、多层/FSDP/100步训练、A3/A5、V-first。
  Packed Dhu 的通过不代表完整 TND 支持；FP32 dHT 暂要求显式 h0 descriptor。

普通拉取、安装和 Hyper 导入不执行这些实验，也不自动编译或安装外部算子。
只有显式运行 AscendC 入口才需要下面的独立依赖。Ulysses 实验含作用域内函数替换，
仅供单线程基准，不作为并发服务入口。

## 文件职责

| 文件 | 用途 |
|---|---|
| `scripts/stateful_kda.py` | staged 前后向、非零 h0/dHT、本地与 P2P/AG 状态传播 |
| `scripts/stateful_layer.py` | 完整 attention 接线与三种 CP 模式 |
| `scripts/stateful_heads.py` | whole-core、单/双流 state-first 分 head |
| `scripts/stateful_summary_heads.py` | 保持完整 core，仅切 summary/state 链 |
| `scripts/memory_options.py`、`fused_output_norm.py` | 保存生命周期、sigmoid-gated RMSNorm、仅 backward-finalize 分块 |
| `scripts/run_stateful_benchmarks.py` | 统一整层精度/性能/显存入口，保留外部进程占用审计 |
| `scripts/benchmark_stateful_layer.py` | 分布式 worker，含 CPU 参考、checkpoint、跨模式与内存变体比较 |
| `scripts/check_stateful_kda.py` | 随机非零 h0/dHT 的完整 local KDA 梯度检查 |
| `scripts/check_dht_matrix.py` | 外部 Dhu 补丁的 dtype、packed、空序列与尾块边界检查 |
| `scripts/run_npu.sh` | 显式选择 CANN、Python、patched wheel 并核对 Hyper 导入路径 |

## 安装与验证

在独立环境中，按 [补丁构建说明](patches/README.md) 构建并安装 FLA-NPU。
不要复用未修复的旧 wheel；wheel 版本号相同不能替代源码/产物核验。
下列命令从本参考仓库根目录运行；路径均需替换成使用者自己的安装位置。

```bash
export PYTHON=/path/to/venv/bin/python
export CANN_HOME=/path/to/cann-9.1.0-beta.3
export FLA_NPU_INSTALL=/path/to/patched-fla-npu-install
export PATH="$(dirname "$PYTHON"):$PATH"
"$PYTHON" -m pip install -e . --no-deps --no-build-isolation
export EXP="$PWD/experiments/kda_ascendc"
mkdir -p "$EXP/runs"

# 非零状态和终态梯度：output-only、state-only、combined loss。
ASCEND_RT_VISIBLE_DEVICES=0 bash "$EXP/scripts/run_npu.sh" \
  "$EXP/scripts/check_stateful_kda.py" --install "$FLA_NPU_INSTALL" \
  --heads 2 --tokens 128 --seeds 7 23 --output "$EXP/runs/local.json"

# CP4 整层 CPU 参考与 checkpoint；也可使用八卡、相同 local64/H8。
"$PYTHON" "$EXP/scripts/run_stateful_benchmarks.py" \
  --install "$FLA_NPU_INSTALL" --devices 0 1 2 3 \
  --tokens 64 --heads 8 --hidden-size 64 --seed 101 \
  --modes p2p allgather --validate-only --checkpoint \
  --suffix cp4_precision --output-dir "$EXP/runs"

# CP8 整层性能和 peak allocated，必须协调连续空卡窗口。
"$PYTHON" "$EXP/scripts/run_stateful_benchmarks.py" \
  --install "$FLA_NPU_INSTALL" --devices 0 1 2 3 4 5 6 7 \
  --tokens 8192 --heads 96 --hidden-size 4096 \
  --modes p2p p2p_summary2 p2p_sf2 allgather \
  --warmup 5 --repeat 20 --suffix cp8_perf_r1 --output-dir "$EXP/runs"
```

入口现在固定测完整层，**不再需要 `--full-layer`**；已删除 core-only、profile 和快照入口。
运行器拒绝覆盖同名审计结果，污染样本不用于性能。共享卡只能通过
`--allow-shared-validation` 做精度检查。时间不包括 optimizer、参数梯度归约或 FSDP。
若编译环境要求自己的 libstdc++，使用该环境的路径配置 `LD_PRELOAD`，不要复制旧机器路径。
不要用 UB 容量覆盖来绕过编译错误；torch-npu 推荐版本提示保留，不静默屏蔽。

## 分 head P2P 对照

以下四类均已包含在本参考 PR，不需要复制另一套目录。模式名末尾的 `2` 或 `4` 是
head 块数，不是每块的 head 数；H96 对应每块 H48/H24。统一基准要求 H 能被 CP 整除，
分头模式还需能被块数整除。现有入口只公开 h2/h4，不把底层可解析任意整数视为已经验收。

| 模式 | 切分起点 | 本 rank 的调度 | 实现入口 |
|---|---|---|---|
| `p2p` | 不切分 | 完整 prepare、summary、state、finish | [StateCP](scripts/stateful_kda.py) |
| `p2p_whole2` / `p2p_whole4` | normalized core 输入 | 每块完成 prepare、summary、state、finish 后才进入下一块；每块一个 autograd Function | [head_core](scripts/stateful_heads.py) |
| `p2p_sf2` / `p2p_sf4` | normalized core 输入 | 单流先逐块 prepare、summary、state，再逐块 finish；统一一个 autograd Function | [HeadStateCP](scripts/stateful_heads.py) |
| `p2p_dual2` / `p2p_dual4` | normalized core 输入 | producer/state 两条流，逐块用 event 连接 summary-ready 与 recv/apply/send，最后逐块 finish | [HeadStateCP / StatePackets](scripts/stateful_heads.py) |
| `p2p_summary2` / `p2p_summary4` | 完整 AscendC prepare 之后 | 只按 head range 生成 S/M/G 并传 state，拼接边界后调用完整 H 的 finish | [SummaryHeadCP](scripts/stateful_summary_heads.py) |

这些方法都不是把 summary 发给下一个 rank，而是发送 `H_out=M@H_in+S` 的 FP32 state；
backward 从高 rank 向低 rank 发送 state gradient。Summary-only 也不是在最后一个
`M@H+S` 才切：其 S/M/G producer 本身按 head range 运行，只有 AscendC prepare/finish 保持完整 H。
所有分头都在 projection、ShortConv、Q/K L2Norm 之后，output norm/projection 仍在 core 输出拼接后执行。

```text
whole:       (prepare0 -> S0/M0 -> state0 -> finish0)
             (prepare1 -> S1/M1 -> state1 -> finish1) -> concat
state-first: (prepare0 -> S0/M0 -> submit state0)
             (prepare1 -> S1/M1 -> submit state1) -> wait dependencies -> finish0/1 -> concat
summary:     prepare full H -> S0/M0,state0 -> S1/M1,state1
             -> concat incoming states -> finish full H
```

State-first 的等待是**本 rank** 的流依赖，不是所有 rank 的全局 barrier；finish 前已经提交
各包 consumer 并等待所需状态事件，但 `isend` 的最终 drain 在 finish 后完成。双流可让下一块
producer 与当前 state 路径重叠，不保证 AICore/HBM 无竞争，也不是 device-side All-Scan。
Backward 同样区分 producer 与 Dhu/finalize；单/双流 state-first 按正 head 顺序发包，
summary-only 按逆 head 顺序发包并反转接收槽拼接，不能单边更改顺序。

这些方式仍是 O(CP) 的 rank 链，不是 RD/Tree。分头可能降低临时 workspace，也可能因更多
调用、layout/cat 和 producer 利用率下降而变慢，不作为默认性能优化。

另一个 [Tree / summary-head RD 参考 PR](https://github.com/xuxl1008/hyper-parallel/pull/9)
是把摘要的 head 分工到不同 rank，再执行外层 RD；不是这里同一 rank 内的 head-block P2P。
旧 Triton 的 dual-drain、dual-immediate、V-packet 也不包含在本 AscendC 模式列表里。

### 整层精度复现

完成上文环境设置后运行以下命令。它用 CPU FP32 recurrence 验证未分头基线，并在相同
权重、输入和上游梯度下比较八个分头组合及 AllGather/Ulysses；不是只测 core。

```bash
"$PYTHON" "$EXP/scripts/run_stateful_benchmarks.py" \
  --install "$FLA_NPU_INSTALL" --devices 0 1 2 3 \
  --tokens 64 --heads 8 --hidden-size 64 --seed 23 \
  --modes p2p --validate-only --checkpoint \
  --compare-modes p2p_whole2 p2p_whole4 p2p_sf2 p2p_sf4 \
    p2p_dual2 p2p_dual4 p2p_summary2 p2p_summary4 allgather ulysses \
  --suffix head_cp4_check --output-dir "$EXP/runs"
```

CP8 使用八张卡并更换 suffix，其余参数不变。性能用 `--modes` 分别启动各模式的新进程，
不要把该交叉精度模式耗时作为性能；H96/local8K 的命令见上文。

2026-10-30 在上述固定环境复验，B1/H8/D128/hidden64/local64、seed23、开启 non-reentrant
checkpoint，所有 output、hidden gradient 和14项参数梯度 finite。以下为最大 relative-L2：

| 对照 | CP4/global256 | CP8/global512 |
|---|---:|---:|
| 未分头 P2P vs CPU FP32 | 9.62018e-3 | 1.83892e-2 |
| 八个分 head 组合 vs 同轮 P2P | 4.08693e-8 | 1.27433e-7 |
| AllGather vs 同轮 P2P | 4.03252e-8 | 1.27433e-7 |
| Ulysses vs 同轮 P2P | 2.42240e-3 | 2.43656e-3 |

全部低于入口既定 0.03 门槛。后面三行是跨模式误差，不是各模式独立对 CPU 的误差。
本次没有重测长序列性能，也不将单层、小 shape 的验证扩大为任意消息并发或多层训练保证。

## 内存变体

| `KDA_MEMORY_VARIANT` | 改动 |
|---|---|
| `baseline` | 原始保存与完整 backward-finalize |
| `trim` | 不保存 backward 无读者的 prepared.u/Qg |
| `trim_norm` | trim 加 sigmoid-gated RMSNorm 融合前后向，不是 SiLU |
| `trim_norm_bwd2`、`trim_norm_bwd4` | 再将 backward-finalize 分2/4份，保留完整 producer/Dhu |

在小 shape 精度命令添加
`--memory-variants baseline trim trim_norm trim_norm_bwd2 trim_norm_bwd4`
可检查全部变体；性能测试用环境变量选一个变体，以新进程、新 suffix 运行并同轮补 baseline。
不要将跨变体精度诊断的确定性设置用于性能测量。

2026-10-24 已通过 local 非零 h0/dHT、CP4/CP8 P2P/AG 完整层 checkpoint、五种内存变体检查。
完整层 CPU 参考最大 relative-L2 约0.01839，低于预先固定0.03门槛；不是所有梯度都小于1%。
历史性能、显存和未获收益的实验结论见 [PR 描述](https://github.com/xuxl1008/hyper-parallel/pull/6)。
