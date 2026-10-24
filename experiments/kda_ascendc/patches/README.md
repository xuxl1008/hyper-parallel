# FLA-NPU 非零 dHT 与 FP32 状态补丁

这是基于 FlashServe FLA-NPU `47d7c97ce439228e0ead6fab4dad8b0440b663a8` 的研究补丁，
不是已合入上游的版本。Hyper 仓库不编译这些外部文件，也不将其纳入本仓 Python UT。

## 修改内容

`chunk_gated_delta_rule_bwd_dhu` 原路径未消费 dHT，本补丁从 GM 载入终态梯度，
初始化已有 FP32 carry；同时贯通 host dtype 组合、tiling、vector 读写、dh0 分配和
Python/C++ binding 的 FP32 边界。FP32 空序列 scratch 加入写出完成后的复用同步。
没有改 Cube GEMM 或 UB 容量。完整 KDA 用 staged Prepare/Dhu/Finalize 接线，
不是直接取消 monolithic backward 的状态检查。

支持边界：910B3/A2 实测，K-first、D128；BF16/FP16 token 的 Dhu，FP32 h0/dHT/dh0。
FP32 dHT 要求显式 h0 descriptor。A3/A5、V-first 和任意 head dim 未验收。
完整 KDA CP 的更窄约束见 [上级说明](../README.md)。

## 应用与构建

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu.git fla-npu-reference
cd fla-npu-reference
git switch --detach 47d7c97ce439228e0ead6fab4dad8b0440b663a8
git apply --check /path/to/hyper-parallel/experiments/kda_ascendc/patches/fla-npu-fp32-dht-a2.patch
git apply /path/to/hyper-parallel/experiments/kda_ascendc/patches/fla-npu-fp32-dht-a2.patch
source "$CANN_HOME/set_env.sh"
export PATH="$(dirname "$PYTHON"):$PATH"
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_kda_fwd,chunk_kda_bwd \
  "$PYTHON" scripts/build_wheel.py --no-progress
# 在构建输出中选择本次生成的完整 wheel，不安装旧的 single-op 或 FP32 v1 wheel。
"$PYTHON" -m pip install --no-deps --target "$FLA_NPU_INSTALL" /path/to/newly-built.whl
```

需独立 Python 环境、CANN 编译工具和该仓库构建依赖；不能只安装 Triton 代替 native 库。
完整 CP 需要 KDA forward/backward 的依赖集合，不是只编译 Dhu 单算子。
历史最终 wheel SHA256 为 `29e9dbebda55ed5267d0cd0efa2bc782034389bef1840e700fa089169492b7b5`。
重新构建不保证二进制哈希一致，应记录自己的 SHA、源码状态并重跑数值检查。

补丁精度入口为 `scripts/check_dht_matrix.py --fp32-state`，完整边界梯度入口为
`scripts/check_stateful_kda.py`。实验结果与失败过程汇总在
[参考 PR](https://github.com/xuxl1008/hyper-parallel/pull/6)，不随源码分发历史 JSON 和日志。

## 许可证

外部源码和补丁保留上游文件头；其中含 BSD-3-Clause 与 CANN Open Software License 2.0
适用内容，**不是统一按 MIT 或本仓 Apache-2.0 重新授权**。
保留 [LICENSE](licenses/LICENSE)、[NOTICE](licenses/NOTICE) 和
[CANN License](licenses/CANN-Open-Software-License-Agreement-Version-2.0.txt)。
分发或上游贡献应按各文件原始许可证、NOTICE 和权利人要求处理。
