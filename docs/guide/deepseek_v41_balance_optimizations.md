# DeepSeek V4.1：Indexer 均衡与 attention 布局优化

本文说明 `feat/v41-balance-layout-optim` 分支相对于已有 CP、Indexer 均衡和 Index K 因果前缀通信实现的增量。整体算法、张量来源、qr 分发与 TopK 恢复见 [Issue #1286](https://github.com/mindspore-ai/hyper-parallel/issues/1286)。

## 改动与原因

1. **简化 hybrid 配置**：只用 `indexer_zigzag_min_length` 判断长文档，删除最小 zigzag 段长参数及相关缓存键。长文档主体按头尾配对分给 rank，开头余数与其他文档共享 greedy 池。文档短于 `2×CP` 时没有非空完整段，全部进入 greedy 池；不丢 token、不添加假 token。
2. **复用 split/unbind RoPE 布局**：相邻复数对以 `unflatten/unbind` 访问，旋转部分与直通部分用 `split` 分开，减少通用切片反向的写回。compressor 使用同一数学公式的 V4.1 helper。no-grad worker Index Q 直接在 BSND 上执行 RoPE，owner 的可微 Index Q 路径保留。
3. **主 Q/输出直接使用 BSND**：`_expand_query` 在 BSND 上做 RoPE，再以 view 保持原 BNSD 接口；`_project_output` 直接处理 BSND attention 输出。减少融合入口前的全张量连续化，以及输出逆 RoPE 的布局搬运，不改变主 attention、teacher 或输出投影公式。
4. **唯一压缩行的反向写回**：文档内非重叠压缩块的 pool 行号不重复。BF16 路径采用 `_UniqueCompressionRows`，反向新建零梯度并 `index_copy_` 写回，替代通用确定性重复索引归约。其他消费者仍由 autograd 累加。FP32 保留原 `index_select` 路径，避免实测临时峰值增加。

唯一行优化仅用于 `DocumentCompressionPlan.pool`，不能用于可能重复选中同一 K 的 teacher、KL 或一般稀疏 gather。主 attention 输出的 grouped BMM 不额外强制 contiguous。

## 配置

同一 attention replacement 的参数示例：

```yaml
replace_module:
  _target_: >-
    hyper_parallel.components.modules.shared_compressed_dsa_attention.SharedCompressedDSAAttention
  use_fused_ops: true
  main_kv_comm: all_to_all_v
  indexer_load_balance: hybrid
  indexer_balance_block_size: 2048
  indexer_zigzag_min_length: 131072
```

- `greedy` 删除文档长度阈值；`none`、`zigzag` 同时省略 block size 与文档长度阈值。
- `greedy/hybrid` 的 block size 默认2048，必须为正且不大于原 owner local Q；该参数不二次切分完整 zigzag 段。
- 旧配置中的最小段长参数必须移除；本分支不保留静默忽略的兼容参数。
- 布局及 BF16 唯一行优化直接生效，不新增算子包依赖或数值精度开关。

## 已有性能证据及测量范围

Ascend 910B3，torch/torch-npu 2.10，开启确定性；CP8，六层实际 attention 宽度，参数783,921,280。hidden5120、qr1280、main 64×512、Index 32×128、TopK512；包含 SWA、r2 Full/Reuse、r1 Full/Reindex/Reuse。真实 DeepSeek 数据集提供文档边界，hidden 与参数使用固定随机初始化。

两次预热后七轮正反顺序交替配对，每轮取最慢 rank，再统计配对收益。计时包含 attention 完整前后向及 CP 激活通信；不含 MoE/mHC、LM head、optimizer、DP 参数梯度归约。共同基线已经应用 RoPE split/unbind 和 worker BSND，均衡模式固定 greedy。

|独立消融|global64K / local8K，61篇文档|global128K / local16K，3篇文档|
|---|---:|---:|
|共同基线，整条 attention 链|947.344 ms|2315.542 ms|
|仅主 Q/输出 BSND，整链|919.257 ms|2257.856 ms|
|同轮下降中位数|28.968 ms / 3.05%|58.283 ms / 2.51%|
|仅 BF16 唯一行 pool 反向，整链|941.428 ms|2303.923 ms|
|同轮下降中位数|5.506 ms / 0.58%|11.719 ms / 0.51%|
|主布局单项额外峰值 allocated|15517.99 → 15004.63 MiB|29679.78 → 28657.01 MiB|

两个消融在各自两组七轮配对中都为正收益。峰值为调用开始前 allocated 之上的额外峰值，不是完整模型 HBM；各列中位数之差不要求等于配对差值中位数。不能相加独立收益推算本分支组合性能，也不能将这些数字作为 greedy 相对 none 的收益。

另有 worker partial-RoPE 与接收融合的四项组合实验，整链配对下降4.27%/3.40%；该组合包含本分支未启用的额外实现，不用它作为本分支性能声明。partial-RoPE 需要额外官方算子，接收融合需要 Triton Ascend；二者继续保留独立实验，待进一步验证整链收益后再迁入。

## 正确性与边界

局部 CPU 检查覆盖 BF16/FP32、partial/full RoPE、B=1/2，以及自然奇数文档、跨 rank 压缩块、空 owner、额外梯度消费者。主 Q/输出及唯一行 pool 共74个输出、输入梯度和参数梯度张量逐位一致。

迁移后的正式代码已完成 NPU/HCCL CP2、local8K 对照：三组真实 packed 文档边界分别运行 greedy、zigzag、hybrid，六层覆盖 r1/r2、Full/Reindex/Reuse 与多个共享 source，每 rank 的68项输出、TopK、输入梯度和全参数梯度逐位一致。该精度模型保留算子所需 head dimension，缩小 hidden、qr 和 head 数；不将它称为完整模型训练验证。

另完成两步 AdamW：每步68项结果与63个更新参数逐位一致；逐层 checkpoint 的六层实际执行12次，68项结果也逐位一致。hybrid 的长文档阈值设为1，实际覆盖完整 zigzag 主体与余数，而不是只覆盖退化后的 greedy 池。规划检查还覆盖 CP2/4/8/128、r1/r2、短于2P的文档、空压缩 owner、收发路由和 Index K 需求。现有 shared attention 模块13项 UT 通过，未新增实验脚本到仓库 tests。

性能表引用此前独立消融；这次迁移验证使用两张空闲卡，没有冒充重新完成 CP8 性能测量。

本分支没有迁入分页 Index K、接收/TopK 恢复融合、owner reduce 或后置 block 交换，也没有修改 teacher/KL 算法。均衡只迁移 no-grad 选择工作，不承诺消除 owner 学习路径、确定性 SMLA 反向或 KV 梯度通信的不均衡。
