# 缓存树与 summary-head RD 研究归档

本目录保存可运行的边界协议研究：真正的缓存树、摘要按 head 分工后执行外层 RD，以及 g2 的生命周期与直接发送布局变体。所有模块采用普通 Python 导入和原生 `torch.distributed`，没有旧源码动态提取、本机路径、旧框架依赖或生产默认配置修改。

这些实现比较的是相同局部摘要 `S/M/G` 上的协议数学。原始 token、Q/K/V、本地 KDA 算术仍留在 token owner；本目录没有把完整 attention 改成 head parallel，也不提供训练默认入口。

## 文件与依赖

| 文件 | 内容 |
|---|---|
| [cached_tree.py](cached_tree.py) | 有序树的后序上行、边界下行及反向；forward 缓存左右子树 M |
| [head_owner_rd.py](head_owner_rd.py) | summary-head 分工、组内合成、外层 RD、状态恢复基线 |
| [head_rd_lifetime.py](head_rd_lifetime.py) | 按最后消费者清理临时量；直接生产 restore 发送包 |
| [protocol_sources.py](protocol_sources.py) | 显式预建通信组，绑定当前 RD，记录模块及 SHA256 |
| [benchmark.py](benchmark.py)、[runner.py](runner.py) | 顺序 FP64 参考、双在途资格、计时、实际 allocator 与对齐 Level0 |
| [compare_protocols.py](compare_protocols.py)、[compare_g2.py](compare_g2.py) | 两个可用 `torchrun --module` 启动的研究 worker |
| [EVIDENCE.md](EVIDENCE.md)、[results/](results/)、[evidence/](evidence/) | 历史证据、本次 CPU 验证与可移植版 NPU 迁移验收 |

唯一项目实现依赖是当前正式 RD 模块
`hyper_parallel.distributed.context_parallel.kimi_delta_attention_rd.KDAStateRecursiveDoubling`。
归档使用其 `forward`、`backward`、通信元数据与不可变 cache 合同；默认不绑定 reset plan。
这是仓库内研究代码，与对应 RD 分支共同版本化，不承诺脱离该 API 的跨版本兼容性。
CPU 使用 PyTorch/Gloo；NPU 使用当前 RD 所需的 torch-npu、HCCL 与可选 Triton epilogue。

当前归档 PR 的 base 为 `feat/kda-cached-rd-cp`，研究分支为 `research/kda-tree-head-rd-20261030`；合入前置 TND/RD 后可改为主干 base。不能在缺少该模块的旧主干上直接运行，也不应通过手工载入另一 checkout 的文件掩盖依赖缺失。

## 数学与严格顺序

令 rank `r` 的局部仿射映射为 `F_r(x) = M_r @ x + S_r`，入口边界为 `H_r`，其局部目标对入口状态的梯度摘要为 `G_r`。测试全局初始状态和最终梯度均为零，参考为：

```text
H_0 = 0
H_(r+1) = M_r @ H_r + S_r
D_(P-1) = 0
D_(r-1) = G_r + M_r.T @ D_r
```

`H_r` 是 exclusive prefix，`D_r` 是 rank `r` 出口的 exclusive suffix adjoint。M 通常不可交换，所有组合必须保留时间顺序。左区间在先、右区间在后时：

```text
M_parent = M_right @ M_left
S_parent = M_right @ S_left + S_right
G_parent = G_left + M_left.T @ G_right
```

树采用平衡区间后序，右端 rank 拥有父节点。forward 上行发送 packed `[S,M]`，owner 保存左右子树的紧凑 M；下行发送 prefix 边界。backward 上行合成 G，下行用已存左右 M 恢复 suffix，**反向不重传 M**。非二次幂 CP 使用同一递归区间切分。树的 owner 热点是实现本身的一部分，没有用重复 RD 或无缓存树冒充此对照。

summary-head 每 `g` 个连续 rank 为一组。每个 lane 负责 `H/g` 个摘要 head，流程为：

1. 组内 AllToAll 收集这些 head 的 S/M，按 rank 顺序合成组摘要。
2. 相同 lane 在 `P/g` 个组之间执行当前 RD，得到组入口；组内按顺序恢复每个 token owner 的 prefix。
3. 第二次 AllToAll 将完整 H 返回 token owner。
4. backward 同样收集 G/本地 M，反序合成 G，调用外层 RD 的 backward，恢复 suffix 并返回 D。

完整 F+B 固定四次组内 AllToAll；外层 RD 并未消失。lifetime 不改 matmul/add 括号，只在最后消费者后释放 aggregate、maps、别名视图及列表。direct 使用每次调用独立的 `[g,B,H/g,128,128]` 发送包，将后续 matmul 和单独 add 写入其切片；反向仍保持 `gradient + product` 的操作数顺序。

三个 head 变体保留相同本地 M 和 outer cache。没有共享可变 workspace，没有删除 backward 需要的 M，没有融合 addmm 或降低精度。direct 的 `out=` 只适合该显式 forward/backward 合同，不能当作普通可微分 Torch 运算；接入完整模型仍需原 custom autograd 边界。

## 成本与显存

令 `P` 为 CP 数，`g` 为组宽，`Q=P/g`，`W=4*B*H*128*128` 字节为一个 FP32 状态或 M 的大小。以下是无 reset 剪枝、紧凑 M 的协议账本，不包含本地 token core 或 allocator 临时峰值。

| 协议 | 依赖和通信 | 最大 rank 持久 M | 全组持久 M |
|---|---|---|---|
| flat RD | 每方向 `O(log P)` 轮及一次 exclusive shift | `ceil(log2 P)*W` | `sum_k(P-2^k)*W`，`0 <= k < ceil(log2 P)` |
| summary-head + RD | 四次组内 AllToAll；每 lane 外层 `O(log Q)` 轮；组内 `O(g)` 顺序合成/恢复 | `W + ceil(log2 Q)*W/g` | `P*W + sum_k(Q-2^k)*W` |
| cached tree | 四次有序上/下遍历；整组 `4*(P-1)` 条消息；平衡树深度 `O(log P)` | 最重 owner 为 `2*ceil(log2 P)*W` | `2*(P-1)*W` |

树全组有效载荷为 `5*(P-1)*W`（forward 上行 2W，其余三遍各 W；不计传输元数据），总合成工作为 `O(P*B*H*128^3)`，owner 的工作与缓存不均衡。flat RD 总工作约为 `O(P*log(P)*B*H*128^3)`；head 分工将外层 head 维度缩小为 `H/g`，仍新增组内算术、布局与 collective。该渐近式不能代替小 CP 的实际时间。

CP8、g2、B1/H96 时，`W=6 MiB`。flat/head/tree 的最大持久 M 分别为 18/12/36 MiB，全组为 102/78/84 MiB。这些数值不是 allocator 峰值。restore 会等待最慢 head lane，模型必须包含 `max(lane_ready_time)`，不能对各 lane 平均。

B1 时旧 restore 接收端的 reshape 已可能共享 storage；B>1 的 source-major `[g,B,h,K,V]` 到 `[B,g*h,K,V]` 仍需重排。direct 省去发送侧 list/stack 和部分临时分配，仍有 outer 种子 copy、scatter 和必要接收布局复制，不能称整个 restore 零拷贝。

## 运行

从包含本目录及正式 RD 的仓库根目录执行。先将当前 checkout 安装为 editable，并核对实际导入位置；下面命令不依赖固定虚拟环境或机器目录。

```bash
python -m pip install -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'

python -m torch.distributed.run --standalone --nproc-per-node=4 --module \
  research.kda_boundary_protocols.compare_protocols \
  --device cpu --batch 2 --heads 4 --warmup 0 --iterations 1 \
  --output /tmp/protocols_cpu.json

python -m torch.distributed.run --standalone --nproc-per-node=8 --module \
  research.kda_boundary_protocols.compare_g2 \
  --device cpu --batch 1 --heads 4 --warmup 0 --iterations 1 \
  --output /tmp/g2_cpu.json
```

两个 worker 默认同时运行 `long` 与 `reset`。后者将隔 rank 的 M 设为结构零，只验证代数，不启用 RD reset planner 或减少消息。`--protocols flat tree` 可在奇数 CP 下比较，`--group-width` 必须整除 CP 和 head 数；g2 worker 可用 `--variants baseline lifetime` 或 `baseline direct` 缩小对照。

NPU 合成输入可直接将 `--device` 改为 `npu`，按实际 head 数设置 `--heads`。复用真实捕获时，每个外部文件 `rankN_boundaries.pt` 须包含同形状连续 FP32 `S`、`M`、`G`，数量与 WORLD 相同。仓库不携带这些张量；下例由调用者提供 `BOUNDARY_CAPTURE_DIR`：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 --module \
  research.kda_boundary_protocols.compare_protocols \
  --device npu --source-dir "$BOUNDARY_CAPTURE_DIR" --source-ranks 8 \
  --cases long --warmup 3 --iterations 7 --output /tmp/protocols_npu.json

python -m torch.distributed.run --standalone --nproc-per-node=8 --module \
  research.kda_boundary_protocols.compare_g2 \
  --device npu --source-dir "$BOUNDARY_CAPTURE_DIR" --source-ranks 8 \
  --cases long --warmup 3 --iterations 7 --output /tmp/g2_npu.json
```

`--repeat-captures` 必须显式启用，并且重复捕获只能算代数或扩容控制，不能作为真实大 CP 输入。`--force-public` 选择当前 RD 的 public batch P2P，对照需要保持相同开关。

profile 应单独运行：追加 `--profile-only --profile-dir /tmp/new_profile_dir` 并使用新 `--output`。每协议在 profiler 内先做一轮 setup F+B，然后 barrier+synchronize，最后只标记一轮正式 F+B。`profile-only` 输出的 `pass=null`，不会冒充数值或正常计时验收。大 trace 不属于归档内容。

## 验证与证据

本次可移植版 CPU/Gloo 覆盖三协议 CP4/B2、CP6/B1，以及 g2 三变体 CP4/B2、CP8/B1；每项均有 long/reset、两次 forward 后逆序 backward、两份输入/缓存不变性、紧凑 cache 与顺序 FP64 参考。g2 另比较两次 H/D 和所有 cache 的逐字节结果。详见 [portable_cpu_validation.json](results/portable_cpu_validation.json)。CPU 单样本时间只诊断流程，不能推断 NPU 性能。

本次可移植版已完成真实 CP8、B1/H96/K=V128、每 rank 4096 token 的 NPU **边界协议迁移验收**。三协议与 g2 三变体全部通过相同 S/M/G、两次在途调用逆序反向、两份输入/缓存不变性与 FP64 参考；g2 两次 H/D 和全部缓存逐字节一致。NPU 此次仅测 `long`，不含 reset、完整 KDA 或训练。

| 本次可移植版同套件顺序观测 | 每次最慢 rank F+B 的中位数 | 最大 allocator 增量 |
|---|---:|---:|
| flat RD / head g2 / cached tree | 4.5146 / 5.3551 / 5.9532 ms | 54.0034 / 54.0049 / 108.0078 MiB |
| g2 baseline / lifetime / direct，另一套件窗口 | 5.0303 / 5.4658 / 5.7448 ms | 54.0054 / 48.0044 / 48.0044 MiB |

每个实现 warmup 3 次、正式 **7 次**，按固定顺序分别计时，没有交替或随机化；这是迁移验收中的性能观测，不能作为稳健速度排名。direct 本轮中位数比 baseline 更慢，未证明提速。lifetime/direct 复现约 6 MiB 的最大单次峰值下降，持久 M 仍为 12 MiB/rank、78 MiB/全组。

新摘要分别见 [可移植三协议](evidence/portable_protocols_cp8_npu.json) 与 [可移植 g2](evidence/portable_g2_cp8_npu.json)，保留运行时源码 SHA、每次最大 rank 的 F/B 样本和实际 allocator 数据。实际输入形状取捕获张量；原命令 `--heads` 的合成默认值不代表真实捕获的 H96。运行时 RD、planner、epilogue 哈希已与发布 checkout 核对，关联 RD 提交为 `3f53b2b4`，TND 前置 `0a3f55c8`，主干基线 `96143af9`。模块哈希是运行代码身份的直接依据；提交标签说明关联发布上下文。

历史真实 NPU 的主要结果如下，完整口径及 source SHA 在 [EVIDENCE.md](EVIDENCE.md)：

| 历史同轮比较 | 每次最慢 rank F+B 的中位数 | 最大 allocator 增量 |
|---|---:|---:|
| flat RD / head g2 / cached tree | 4.5523 / 5.4719 / 5.9695 ms | 54.0034 / 54.0049 / 108.0078 MiB |
| g2 baseline / lifetime / direct，另一采样窗口 | 4.7920 / 5.0820 / 4.8714 ms | 54.0054 / 48.0044 / 48.0044 MiB |

历史 g2 三变体逐字节一致；lifetime/direct 的实测最大峰值下降约 6 MiB，持久缓存仍为 12 MiB/rank、78 MiB/全组。现有样本没有证明加速，两轮不同来源 RD 和采样窗口不能拼成速度排名。历史三协议 FP64 最大相对 L2 不超过 `3.59e-8`。这些数据属于历史来源实现，与上面的本次可移植版验收分别记录，不能互换来源或合并采样窗口。

所有数值 gate 都是给定 FP32 摘要相对 FP64 仿射参考的协议资格，不是独立 FLA 混精训练参考。没有完整 KDA、本地摘要生产者、任意 mesh、任意拓扑、训练收敛或更大 CP 的实测承诺。输出不含本地路径；source manifest 只记录模块名和哈希，发布新结果前仍应只挑选必要的精简字段。

后续研究树的临时量生命周期和 owner 上下行依赖；若提出性能改进结论，应使用交替或随机化配对计时验证。大 CP 方案必须同时计入四次组内通信、最慢 lane、远端竞争、拓扑和缓存，不从 CP8 的单个窗口外推默认策略。

## 正式交付前仍需整理

本 PR 用于保存可运行研究实现，保持 Draft。源码语法与数值迁移检查已通过，但还未通过全部生产静态门禁：5个函数的圈复杂度超过15（最高30）；现有旧 platform lint 规则对 research 目录中的原生 Torch 导入报 C9002。仓库已要求直接使用 Torch，因此未来应按正式模块边界处理该规则，不恢复已移除的 platform 层。

这次保留实际通过设备验证的代码及哈希，没有为了归档而重写递推或扩大忽略名单。迁入生产前还需拆分复杂函数、整理正式入口与 mesh 资格、补完整 KDA/训练验证。研究目录不会由生产模块自动导入，也不改变默认配置。
