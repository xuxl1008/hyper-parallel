# DeepSeek-V3.2 模型族适配

[English](README.md) | **简体中文**

DeepSeek-V3.2 在 DeepSeek-V3 的 MLA + MoE 结构上加入 DeepSeek Sparse Attention（DSA）：每层由
lightning indexer 为每个 query 选出 `index_topk` 个 key（官方 config 为 2048），MLA 只在这些 key 上
计算 attention。HyperParallel 训练的是 Transformers 原生的 `DeepseekV32ForCausalLM`，本目录提供它的
模块替换、并行规则和训练 recipe。

## 注册身份

| 项目 | 值 |
|---|---|
| HyperParallel 族名（`models/<family>/` 目录名、注册身份） | `deepseek_v32` |
| HF `config.model_type` | `deepseek_v32` |
| HF `architectures` | `DeepseekV32ForCausalLM` |
| 实现来源 | Transformers 原生实现（`transformers.models.deepseek_v32`），`trust_remote_code: false` |

## 目录结构

```text
models/deepseek_v32/
├── __init__.py
├── adapter/
│   ├── conversion/
│   │   └── module_replacement.py    # 替换工厂：RMSNorm、DSA attention、GroupedExperts
│   ├── distributed/
│   │   └── context_parallel.py      # DSA 的 CP wrapper
│   ├── policies/
│   │   └── sharding.py              # MLA 与 indexer 的 TP 分片规则
│   ├── materialization.py           # 加载权重后恢复 RoPE 的 buffer
│   └── registration.py              # ModelAdapterSpec
├── generated/                       # codegen 生成的模型代码，仅 codegen: true 时使用
└── recipes/
    ├── run_parallel_offline.sh      # 在 8 卡上启动 train_parallel_offline.yaml
    ├── train.yaml                   # 加载官方 checkpoint 的单进程配方
    └── train_parallel_offline.yaml  # 8 层裁剪模型的 8 卡配方
```

## 功能

- **NPU 模块替换**，由 recipe 的 `plan_overrides` 选用：`*.input_layernorm`、`*.post_attention_layernorm`、
  `model.norm` 换成融合 `RMSNorm`；`*.self_attn` 换成 `DeepseekV32DSAAttention`，indexer 与稀疏 attention 走
  CANN 融合算子；`*.mlp.experts` 换成 `GroupedExperts`。
- **并行**：FSDP、TP（含 loss parallel）、CP、EP。不支持 PP 和 sequence parallel。
- **训练行为**：
  - router 用 `e_score_correction_bias` 选专家（sigmoid 分数、分组 top-k），bias 保持初始值；
  - 没有负载均衡 loss；
  - 官方 checkpoint 的 MTP 层（第 61 层）加载时被忽略，不训练；
  - TopK 选择本身不可导；启用 `dsa_loss_coeff` 且未冻结 Indexer 时，通过辅助 KL 训练 Indexer。
- **数据**：支持连续因果序列，以及通过累计文档边界隔离的 packed 序列。DSA 不读取 `attention_mask` 的内容，不支持任意 padding mask、dropout 和 KV cache。

## 环境要求

- Transformers 需要包含 `deepseek_v32`：`requirements.txt` 要求 `transformers>=5.11.0`，验证版本 5.13.1。
- 昇腾 NPU，DSA 算子依赖 CANN 与 `cann_ops_transformer`，见
  [functional README](../../components/functional/README.md#deepseek-v32-dsa)。
- 权重：`from_pretrained` 可以直接加载官方 FP8 checkpoint（读取时反量化），也可以加载 BF16 checkpoint。第 61 层
  MTP 的权重会被忽略并打印 warning。

## 使用

### 用 recipe 训练

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/train_lm.py hyper_parallel/models/deepseek_v32/recipes/train.yaml \
  --model.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2 \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2
```

末尾追加的参数会覆盖 YAML 字段，例如 `--training.train_iters=200`。

### 多卡

用命令行覆盖设置并行拓扑。recipe 里 `when: cp` 的条目在 `cp_size > 1` 时启用 DSA 的 CP wrapper。EP 需要加一个
`when: ep` 条目，只匹配 MoE 层，即 `i >= first_k_dense_replace` 的 `model.layers.<i>.mlp.forward`（官方 config 为
第 3–60 层）：

```yaml
plan_overrides:
  - match: "model.layers.[3-9].mlp.forward"
    when: ep
    replace:
      kind: method
      implementation:
        _target_: hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn
        use_grouped_gemm: true
        overlap_shared_expert: true
  # 再用 match: "model.layers.[1-9][0-9].mlp.forward" 写一条同样的条目，覆盖第 10–60 层。
```

例如 8 卡、TP2 × CP2 × DP2、EP4：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py <你的 recipe> \
  --accelerator.tp_size=2 --accelerator.cp_size=2 --accelerator.ep_size=4 \
  --accelerator.loss_parallel=true \
  --fsdp_config.dp_shard_size=2 --fsdp_config.edp_shard_size=2
```

需要满足：`tp_size > 1` 时 `ep_size` 也要大于 1（MoE 层的专家只按 EP 切分）；`world_size = tp × cp × dp`；
`n_routed_experts` 能被 `ep_size` 整除；`global_batch_size` 能被 `micro_batch_size × dp` 整除。

### 8 层裁剪模型的 8 卡 recipe

`recipes/train_parallel_offline.yaml` 在 8 卡上训练 DeepSeek-V3.2 的 8 层裁剪模型（3 个 dense 层 + 5 个 MoE 层，16 个路由
专家，8.40B 参数）：TP2 × CP2 × DP2，EP4 × EDP2，loss parallel 加 FSDP，启用全部模块替换。裁剪模型只需要官方目录中的
`config.json` 和 tokenizer 文件，权重为随机初始化的 BF16：

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

source, target = "/path/to/DeepSeek-V3.2", "/path/to/DeepSeek-V3.2-8layer-crop"
released = AutoConfig.from_pretrained(source)
config = AutoConfig.from_pretrained(
    source, num_hidden_layers=8, n_routed_experts=16,
    # 逐层类型从官方列表里截取：类型名在不同 Transformers 版本里不同。
    layer_types=released.layer_types[:8], mlp_layer_types=released.mlp_layer_types[:8],
)
del config.quantization_config
torch.manual_seed(1234)
AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(target)
AutoTokenizer.from_pretrained(source).save_pretrained(target)
```

数据为用 DeepSeek-V3.2 tokenizer 处理好的 Indexed Dataset（`.bin`/`.idx`），见
[离线数据准备指南](../../../docs/guide/data/offline_preparation_guide.zh.md)。传入 checkpoint 目录和不带扩展名的数据
前缀，末尾追加的参数会覆盖 YAML 字段：

```bash
bash hyper_parallel/models/deepseek_v32/recipes/run_parallel_offline.sh \
  /path/to/DeepSeek-V3.2-8layer-crop /path/to/dataset_text_document
```

日志、数据缓存和 checkpoint 写到 `output/deepseek_v32`，可用 `OUTPUT_DIR` 修改。

## DSA 的文档级 CP 与 Indexer 选择均衡

`train.yaml` 通过 `DeepseekV32Runtime` 将全局 `cu_seq_lens` 转为 `actual_seq_len`，
CP 只切 token，不切全局文档边界。每层共用缓存的文档计划，仅打包拥有本地 Q 的文档及其因果 K 前缀。
原生 LI、SFA 和 KL 使用一致的 TND 文档顺序与文档内 TopK 坐标。
数据侧 `attention_mode: compressed` 表示传递紧凑边界而不构造 dense mask，**不表示 DSA 压缩 token**。
`reset_position_ids: true` 让 RoPE 使用文档内位置；位置与文档隔离由不同字段表达。

在现有 `*.self_attn.forward` 的 CP replacement 中配置：

```yaml
implementation:
  _target_: hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel.deepseek_v32_dsa_attention_cp_wrapper
  dense_warmup: null
  indexer_load_balance: greedy
  indexer_balance_block_size: 2048
  main_kv_comm: all_to_all_v
  index_k_comm: all_to_all_v
```

- `none`：维持原 Q owner；`zigzag`：文档首尾片段配对、余数均分；`greedy`：按可见 Q×K 工作量分配块；
  `hybrid`：长文档主体 zigzag，余数和短文档统一 greedy。
- `indexer_balance_block_size` 仅用于 greedy/hybrid，省略时为 2048；none/zigzag 不接受此参数。
  hybrid 还需显式设置 `indexer_zigzag_min_length`，单位为原始文档 token 数。
- 两种 KV 通信分别可选 `all_gather`（默认）或 `all_to_all_v`。均衡和前缀 A2AV 要求 packed batch size 为 1；
  原始序列仍按 CP 等长切分，单篇文档无需被 CP 大小整除。
- 均衡只迁移不求导的选择任务：发送 qr、weights、cos/sin，worker 重投影 Index Q 并返回 TopK。
  主 attention、teacher/KL 及可微 Index Q/K/weights 留在原 Q owner；不会自动均衡这些模块的计算。
- Index K 的前向需求是 worker 选择与 owner KL 前缀的并集，反向仅回传 owner KL 贡献。
  半精度 Index K 梯度在源 owner 用 FP32 累加再转换回输入 dtype；不承诺与 BF16 AG/RS 逐位相同。
- worker 收到 qr 后可在 Index K 传输期间做投影；每个原生消费者读取通信结果前仍显式 wait。
  通信事件包含等待其他 rank 的时间，异步提交并不保证整层加速。

Dense warmup 保留基础 CP 和两种 KV 通信，要求 `indexer_load_balance: none`。
选择均衡是可选实验项，默认不启用；收益取决于文档分布、LI 占比及额外分发/返回成本。
BF16 打分接近 TopK 分界时，重排后的 native 调用可能产生不同选择，应分别检查同输入精度和训练轨迹。
CE 的 ignore label 不会自动排除 Indexer KL 的 padding Q；不要将任意 padding mask 支持视为本改动的能力。

## 算子与调度实验草稿

可选的算子侧补丁、worker partial-RoPE及eager反向调度原型见
[实验说明](../../../examples/torch/dsa/README.md)。它们需要显式运行，不默认启用，也不是Trainer配置项。
