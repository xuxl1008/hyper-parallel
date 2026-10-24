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

## 分头与内存变体

模式包括 `p2p/allgather/ulysses`、`p2p_whole2/4`、`p2p_sf2/4`、`p2p_dual2/4`、
`p2p_summary2/4`。接口提供这些实验选项，不意味着每个组合都已广泛验收或能加速。
H 必须满足当前基准的 CP 整除检查，分头模式还需能被对应块数整除。

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
