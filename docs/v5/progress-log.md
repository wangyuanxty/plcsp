# v5 进展记录（E2E-MAPPO 基线 + Stage-0 AGV）

> v5 来源：`third_party/E2E-MAPPO-for-MT-FJSP`，上游 commit `bdd8719144a62f66c6c16c4243056047c050dc6d`，MIT。
> 本文件只记 v5 线。不得引用 v1 / v2 / v3 的结论。

## §1 2026-10-22 AGV 运输记账修复

### 问题

旧实现把 `transport_override = wait + travel` 用于整个 `env.step`。
`find_transportT` 同时服务调度、边权、TEC 三处。
结果：

- AGV 等待同时进入 TEC 和 SEC（双算）。
- 等待进入同机边。同机没有运输。
- 等待进入所有 job 边。override 是标量。
- 前序工序的开始时间反而没有等待。`find_max_arrivaTime` 用原矩阵。

### 修复

1. `trainer/DGenv_func.py`
   - `find_transportT` 改为按 `(prev_node, curr_node)` 配对的 `transport_override`，只返回 **travel**。
   - `find_max_arrivaTime_for_currentNode` 加 `transport_wait_override`，只加到开始时间。
   - AGV 模式下已调度节点直接返回 `start_time`，保留已付等待。
2. `trainer/parallel_env.py`
   - AGV 分配用前序工序 `finish_time` 作为 ready。
   - 同机或前序未调度：不派 AGV。
   - `agv_states` 每 episode 重置（`init_DGFJSPEnv_state0`）。
   - `available_mask` 按 `free_time <= now`。
   - 机器等待特征用本工序 ready；同机候选等待设 0。
3. `agv/state.py`
   - mask 按 `free_time`；busy 特征按 `free_time`。
4. `Run.py`
   - `Loss_Cost` 文件名加 `run_tag`，避免两个 run 混写。
5. `tests/test_agv_transport.py`
   - 4 个测试：TEC = travel、start = travel + wait、已调度节点保留等待、无 override 保持上游行为。

### 验证

- `pytest plcsp_v5/tests/test_agv_transport.py plcsp_v5/tests/test_agv.py`：10 passed。
- 2 episode smoke（`--use_agv 1`）：完成，验证跑通。
- AGV-aware 验证 smoke 读数：obj 1090.03、mk 751.34、pt 508.44、transT 271.30、idleT 1329.65。
  这是 2 episode 随机初始化模型，只用于确认验证路径可跑，**不是性能结果**。

### 补充修复

- **验证 / 测试环境现在模拟 AGV。** `trainer/validate.py` 在单环境上使用与训练相同的
  transport override，并用 `va_select_agv` 调用**学习到的 AGV actor**，不再用固定规则。
  机器候选特征为真实 10 维，删除补零。
- `esa_validate_cost_gcn_jointActor_GAT`（测试路径）也已 AGV-aware。
- `eval_st_seed3.py` 保留命令行 `--use_agv` 和 `--run_tag`。AGV 评测用
  `--use_agv 1 --run_tag agv`，并加载 `PPO_agv_actor_J6M6E2_top1.pth`。
- **恢复并训练 AGV local critic。** 目标 `(mk, idle)`，与 job local critic 同构：
  `-2*global - local + 0.5*local_critic_MSE - entropy`。
  Replay buffer 新增 `agv_v`；checkpoint 保存 AGV actor 和 optimizer。
- 修复 `Run.py` 的 `agv_v_store` 未初始化：no-AGV 新 run 之前会 NameError。
- 修复训练日志文件名：`weight_mk` 写了三次，`Weight444` 应为 `Weight442`。
- Resume 同时检查 `agv_actor` 和 `agv_optimizer`。

### 运行中

- no-AGV：PID 14800（`--run_tag noagv`，2026-10-22 01:11 启动，修复后代码）。
- AGV：PID 12272（`--run_tag agv`，2026-10-22 01:11 启动，修复后代码）。
- 旧 no-AGV run（PID 13580，ep200）已停止。它的输出是**修复前**读数，不与新 run 混用。
- 两者每 100 episode 存 `full_ckpt`；`eval_freq=10`。

### 开放问题（未修）

1. **空驶未建模。** travel 仍用原 `transT`（前序机器 → 当前机器），不是 AGV 当前位置 → 取货点。
2. **AGV 待机能耗未入目标。** TEC 只算行驶；SEC 只算机器待机。
3. **WAIT 语义。** `agv_action >= n_agv` 映射到最早空闲 AGV，不是真正“等待”。
4. **`--use_load_model` 不加载 AGV actor。** 整 checkpoint 恢复用 `--resume_ckpt`。
5. **机器等待特征用最早空闲 AGV 的等待。** 学习到的 AGV 可能选别的车；这是启发式。

## §2 2026-10-22 独立复审与修复

独立只读复审发现 7 条。全部处理。

1. **ESA replay buffer 的 AGV 参数被静默丢弃。** 第二个缓存类 `esa_ReplayBuffer_JointActor_GAT` 没有 AGV 字段。
   已修：`machine_fea1` 改为 `10 if use_agv else 6`；`args.n_machine` 改为 `args['n_machine']`；
   `store_operation` 在 `agv_feat` 非空时显式 `NotImplementedError`。`test_all.py` 也加了同样的守卫。
2. **AGV 的 `edge` 从不更新。** `location` 更新了，`edge` 没更新，导致 `_idle_ratio` 和 AGV token 的车间特征长期错误。
   已修：训练和验证在分配 AGV 时更新 `agv.edge`；`AgvState.assign` 增加 `dropoff_edge` 参数。
3. **评估次数少于 3 时没有 `_top1.pth`。** 旧代码按下标映射 `top_prex`，1 次评估只生成 `_top3.pth`。
   已修：按排名重命名，`rank=1` 永远写 `_top1.pth`。AGV actor 同步。
4. **`full_ckpt` 绑定在评估 episode 上。** `eval_freq` 不整除 100 时不会存 ep100/ep200。
   已修：`full_ckpt` 移到评估块外，每 100 episode 保存。
5. **Resume 时 episode 不是 `resample_freq` 的倍数会崩溃。** 首轮不调用 `get_batch`，`ability_instance` 为空。
   已修：首轮若 `ability_instance` 为空则强制取样。⚠️ `train_loader` 的 shuffle RNG 未恢复，恢复后的数据顺序不保证与中断前一致。
6. **`test_all.py` 不加载 AGV 权重。** `--use_agv 1` 时用随机 AGV actor 产生无效数字。
   已修：`test_all.py` 遇到 `use_agv` 直接报错，指向 `eval_st_seed3.py`。
7. **梯度裁剪在 `backward()` 之前执行。** `use_grad_clip` 实际失效。
   已修：`clip_grad_norm_` 移到 `backward()` 之后、`step()` 之前；AGV actor 也加裁剪。
   这改变了训练动态，所以两个 run 都从 0 重启。

   ⚠️ **这是上游 E2E-MAPPO 自带的 bug**，不是 v5 引入的。证据：
   `third_party/E2E-MAPPO-for-MT-FJSP/algorithm/ppo_algorithm.py:965-983`。
   上游同样在 `zero_grad()` 后、`backward()` 前调 `clip_grad_norm_`。
   全局 critic 的裁剪在上游是正确的（`backward()` 之后）。

   **用户裁定（2026-10-22）：保留修复。**
   两个 run 都用修复后的裁剪，彼此可比。
   但 no-AGV 基线必须标记为 **「E2E + 裁剪修复」**，
   **不得与论文已发表数字直接并列**。

### 复审后的 smoke

- 10 个 AGV 单元测试通过。
- AGV 2 episode smoke：通过；`PPO_agv_actor_J6M6E2_top1.pth` 已生成（第 3 条修复验证）。
- no-AGV 2 episode smoke：通过（`agv_v_store` 初始化修复验证）。
- AGV seed3 100 实例评测：通过（学习到的 AGV actor 可加载）。

## §3 2026-10-22 跨车间模型修正（AGV 不跨车间）

### 论文假设

E2E-MAPPO 原文（p. 4424）假设 “TMs at each edge are adequate”。
即每个车间自己配运输机器。**job 可以跨车间，AGV 不跨车间。**

旧实现是**单车队**：AGV 可以接跨 edge 的整段运输，跨过去后 `agv.edge` 改成目标车间。
这与论文假设不一致。旧输出已归档到
`trained_model/agv_singlefleet_pre_crossdelay/`，不参与新比较。

### 新实现

- **每车间独立车队。** `n_agv` 总量按 `k % nedges` 分配；每辆 AGV 的 `edge` 是**固定 home edge**，不再改变。
- **同车间运输：** 一辆本车间 AGV，`travel = transT[prev_m][target_m]`。
- **跨车间运输：** 拆成三段：
  ```text
  源车间 AGV 段 t_src
  + 固定延迟 D（不占 AGV）
  + 目标车间 AGV 段 t_dst
  ```
  `t_src + t_dst = transT[prev_m][target_m] - D`；默认 `t_src = t_dst`。
- **AGV actor 只选源车间 AGV。** 目标车间 AGV 用最早空闲规则。
- 边界位置没有建模。源 AGV 送到边界后 `location = -1`；下一次取货的空驶时间近似为 0。
- 跨车间固定延迟：`--cross_delay 5`；两段比例：`--cross_leg_ratio 0.5`。

### 记账口径

| 量 | 是否含固定延迟 D |
|---|---|
| MKT | 是 |
| SEC（机器空闲） | 是 |
| TEC（运输能耗） | **否**。TEC 只算 `t_src + t_dst` |
| 机器候选特征 `transT` | 是（仍用完整矩阵值） |
| 机器候选特征 “AGV 等待” | 否（只放 AGV 等待） |

### 验证

- 13 个 AGV 单元测试通过（新增 `split_cross_edge` 和 edge mask 测试）。
- 2 episode AGV smoke（`--cross_delay 5`）：通过。
- smoke 验证读数：obj 1356.62、mk 860.34、pt 515.45、transT 191.26、idleT 1920.13。
  这是 2 episode 随机初始化模型，**不是性能结果**。
- `PPO_agv_actor_J6M6E2_top1.pth` 已生成。

### 运行

- no-AGV：PID 14800（`--run_tag noagv`），继续跑。no-AGV 路径未受跨车间改动影响。
- AGV：PID 32352（`--run_tag agv`，2026-10-22 01:49 启动），
  参数 `--use_agv 1 --n_agv 6 --cross_delay 5 --cross_leg_ratio 0.5`。

### 开放问题

- 目标车间 AGV 用规则，不参与学习。
- 边界位置与源 AGV 回程空驶未建模。
- 固定延迟 D 是新增参数，论文没有；和论文数字并列时必须说明口径。
- AGV 待机能耗仍未入目标。

## §4 2026-10-22 两条线的完成情况

### no-AGV：完成

- 1000/1000 episode，exit 0，结束于 2026-10-22 06:44:56。
- 100 次验证。
- 最好验证 obj = **768.47**，在 episode 90（`top1` = EP90）。
- 最后验证 obj = **898.05**，在 episode 999。
- `full_ckpt_settrans_ep100.pt` … `ep1000.pt` 齐全；`top1/top2/top3/final` 齐全。
- 口径：**「E2E + 裁剪修复」**，不是上游原样。不得与论文数字直接并列。
- seed3 测试（top1 EP90，100 实例）：
  **obj 777.20 ± 116.62**；mk 508.92、pt 455.76、transT 260.78、idleT 847.93。
  该测试用 seed1 验证选出的 top1，在 seed3 测试集上跑一次。
  与论文 732.5 **不可直接并列**（裁剪修复 + 选择协议不同）；
  与旧 ST 的 804.39 也不是同一训练配置。
- 额外对照：
  - top1 EP90，sampling：obj 829.31。
  - final ep1000，greedy：obj 914.61；sampling：919.94。
  - **final 在 greedy 和 sampling 下都更差**，不是 greedy 评测的假象。
- 验证曲线：ep90 **768.47** → ep350 约 900 → final **898.05**。
  不是「停在 ep90」，是**触底后反弹**。
- 训练侧：200 次 update 内，job/machine/critic loss 都在固定区间振荡；
  critic loss 约 0.55，没有下降趋势。策略没有收敛。
- 随机初始化对照（seed0，未训练，greedy，同一 100 个验证实例）：
  **obj 1700.44 ± 386.43**。这是随机网络，不是论文的 RA（随机动作）基线。
- 学习幅度：
  - 随机 1700.44 → ep10 841.45：**−50.5%**。
  - ep10 → ep90 768.47：**−8.7%**。
  - ep90 → final 898.05：**+16.9%**。
  - 随机 → ep90：**−54.8%**。
  结论：**训练主要集中在前 2 次 update（ep10）**。
  之后 980 episode 只再降 8.7%，然后反弹。后期的边际训练几乎无效。
- 最可能原因：**训练和评测目标不一致**。
  训练用随机奖励权重 + reward scaling；验证/测试用固定权重 + 原始 cost。
  策略向随机权重平均漂移，固定权重表现退化。
- 次要原因：PPO 小 buffer（5 episode × 16 env，mini-batch 36）、
  LR 衰减慢（每 50 episode ×0.9）、没有 early stopping。
- 待做诊断：固定权重（0.4/0.4/0.2）训练 vs 随机权重训练；
  记录训练原始 4 cost 和策略 entropy。

## §5 2026-10-22 E2E 原始模型 1000 episode 对照

- 目的：判断 ep90 后退化是否与 Set Transformer 编码器有关。
- 模型：E2E 原始架构（GIN + GAT），`--encoder_type e2e`。
- 其余完全同 ST run：
  `--episode_num 1000 --eval_seed 1 --LR 3e-4 --use_lr_decay 1 --decay_step_size 10 --decay_ratio 0.9`。
- 代码：v5 当前代码（含裁剪修复、top1/full_ckpt 修复）。**只有编码器不同。**
- run_tag：`e2e_control`；PID 12268；2026-10-22 09:34 启动。
- 与 AGV 恢复 run 并行。
- `eval_st_seed3.py` 新增 `EVAL_ENCODER` 环境变量。
  E2E 测试用：
  `EVAL_ENCODER=e2e eval_st_seed3.py --run_tag e2e_control`。
- ⚠️ 训练中每 10 episode 的读数是 **seed1 验证**（100 个验证实例），
  **不是 seed3 测试**。论文的 732.5 是 seed3 测试。
  两者只能在训练结束后用上面的测试命令比较。
- 当前进度（10:58）：ep220/1000。
  验证 ep10–220：735.37、728.34、935.93、737.71、732.72、723.44、722.70、730.27、
  750.33、800.85、888.97、979.84、782.27、751.02、721.12、763.95、771.24、875.11、
  751.60、718.99、741.67、**708.69**。
  当前最好 **ep220 = 708.69**。
  ep90–120 和 ep170–180 有尖峰；之后都能恢复并继续下降。
  `full_ckpt_e2e_ep100/ep200` 已存；当前 top3 是 EP150/EP200/EP220。

## §6 2026-10-22 历史 run 对照：退化不是第一次出现

| Run | 配置 | 最好验证 | 最好点 | 最终验证 | seed3 测试 |
|---|---|---|---|---|---|
| `seed3_run` | ST，LR 1e-3，裁剪 no-op | 770.63 | EP20 | 897.65（ep199） | — |
| `seed1_lr1e3` | ST，LR 1e-3，裁剪 no-op | 793.52 | EP120 | 904.21（ep199） | 804.39（top1 EP120） |
| 旧 no-AGV（0:02，停在 ep210） | ST，LR 3e-4+decay，裁剪 no-op | 851.5 | EP10 | 877.3（ep210） | — |
| 当前 no-AGV | ST，LR 3e-4+decay，裁剪修复 | **768.47** | EP90 | **898.05** | **777.20** |

结论：

- **早期最好、后期变差，在之前的 run 里也出现。** 不是当前 run 独有。
- 换 LR（1e-3 → 3e-4 + decay）和修裁剪，把最好点从 793/851 改善到 768，
  但没有消除退化。
- 因此主因不太可能是单个超参数。更可能是：
  训练目标（随机权重 + reward scaling）与评测目标（固定权重 + 原始 cost）不一致，
  加上 PPO 小 buffer 下的不稳定性。
- 原论文 4000 episode 的日志里，3/4 次最好点在 EP2610–3820，1/4 在 EP290；
  我们的 run 都短（200–1000 episode），最好点都早。

## §7 2026-10-22 ST 图注意力偏置（200 episode 对照）

### 之前的事实

- `model/actor_critic_st.py` 的 `Operation_Actor` / `Global_Critic` 接收 `adj`，
  但 forward 完全没用。
- `model/set_transformer.py` 的 `MAB` 只有 `key_padding_mask`，没有 `attn_mask`。
- 所以「ST 丢图」成立；之前没有任何图偏置。

### 现在补的偏置

- `MAB` / `SAB` / `SetTransformerEncoder` 新增可选 `attn_bias`，
  作为 `nn.MultiheadAttention(attn_mask=...)` 的加性偏置。
- **Job actor**：任务图二值偏置
  `bias_ij = edge_bias[h] if adj_ij > 0 else none_bias[h]`。
  `edge_bias` / `none_bias` 是每头可学习标量，初始化为 0。
- **Global critic** 的任务编码器：同样的任务图偏置。
- **Machine actor**：没有加偏置。它没有机器图输入，保持集合模型。
- 没有边类型（job edge vs machine arc）、没有边权、没有邻居限制、没有消息传递。

### 和原 GNN 对得上吗

**对不上。**

| 项 | 原 GIN/GAT | 当前 ST + bias |
|---|---|---|
| Job 侧 | GIN 逐层沿边聚合邻居 | 只在 attention logits 加 edge/no-edge 偏置 |
| Machine 侧 | 机器 GAT（2 节点） | 无图，纯集合 |
| 边类型/权重 | 使用图结构与边信息 | 只有二值邻接 |
| 消息传递 | 有 | 无 |

所以这是**弱结构先验**，不是 GNN 等价实现。

### 运行

- 2 episode smoke：通过。
- 200 episode 对照：`--st_graph_bias 1 --episode_num 200 --run_tag st_bias`，
  PID 47876，2026-10-22 11:13 启动。
- 结果（200 episode 完成，12:45）：
  20 个验证点；最好 **ep20 = 754.15**；final **853.14**。
  曲线：ep10 1092、ep20 754、ep30 918、ep40 803、ep50 784、ep60 801、
  ep70 809、ep80 836、ep90 825、ep100 808、ep110 825、ep120 790、
  ep130 819、ep140 815、ep150 829、ep160 848、ep170 865、ep180 855、
  ep190 844、ep200 853。
  ep20 之后没有持续改进。
- 基线：no-AGV ST 前 200 episode：最好 ep90 = 768.47，ep200 = 856.5。
- 单类偏置把最好点改善 14.3（1.8%），**没有消除退化**；最好点更早（ep20）。
- E2E 对照继续跑：最好 **703.26 @ep240**；仍领先约 50。
- 结论：单类偏置不足。ST 的结构性差距在**消息传递**，
  一个二值 edge/none 标量补不回来。两类偏置优先级降低。

## §8 2026-10-22 消融：工序 GIN + 机器 ST

### 设置

`--encoder_type hybrid`：

| 组件 | 编码器 |
|---|---|
| Job actor | **原始 E2E GIN** |
| Machine actor | **ST** |
| Global critic | **原始 E2E**（GIN + 2 节点 GAT），未换 |

只换机器 actor，保持 job actor 和 critic 为 E2E。这样隔离机器侧。

### 运行

- 用户裁定：**不要 actor-only 控制臂。** `hybrid_gin_st` 已停并清理。
- 完整混合 `--encoder_type hybrid_full`：
  - Job actor：原始 E2E GIN。
  - Machine actor：ST。
  - Global critic：**GIN 任务编码 + ST 机器编码 + 4 维头**（`Global_Critic_Hybrid`）。
- 2 episode smoke：通过，但**没有触发 PPO update**（buffer_size=5）。
- 首次 200 episode 在 ep5 的 critic update 处失败：
  `Global_Critic_Hybrid` 把 `x` reshape 成 3D，GIN 需要 2D。
  已修：hybrid critic 不再 reshape `x`。
- 6 episode smoke（触发 PPO update）：通过。
- 200 episode 重启：`--encoder_type hybrid_full --episode_num 200 --run_tag hybrid_full_gin_st`，
  PID 39928，2026-10-22 13:15 启动。
- 当前（14:13）：ep154/200。
  验证 ep10–150：1151.06、1209.68、1045.90、944.93、871.66、836.98、
  837.17、851.48、830.12、781.38、**764.89**、773.91、778.54、779.33、797.58。
  当前最好 **ep110 = 764.89**。
  已接近全 ST 的 768.47，但仍比全 E2E 的 718.99 差约 46。
- E2E 对照：ep730；最好 **703.26 @ep240**；最近 ep700–730：734.77、745.81、741.11、741.29。

### 对照

| 模型 | 前 200 最好验证 | final ep200 |
|---|---|---|
| 全 E2E（GIN + GAT） | 718.99 | 718.99 |
| 全 ST（ST + ST） | 768.47 @ep90 | 856.5 |
| **完整混合（GIN + ST + ST 机器 critic）** | 待跑 | 待跑 |

### 判读

- 完整混合 ≈ E2E → 机器侧 ST 可以；主要差距在工序 ST。
- 完整混合 ≈ ST → 机器侧 ST（actor + critic）是主要问题。
- 中间 → 两侧都有贡献。

### 注意

- 这里已经**没有留 E2E critic**。critic 的机器编码也是 ST。
- 机器侧差异是「每台机器 2 节点 GAT」vs「机器 token + 跨机器 self-attention」，
  不是机器-机器图。

⚠️ **上表除“原模型”外都是 ST run，不是原模型。** 原模型证据：

- 作者日志 4 次 4000 episode run：
  `Best_model` 分别在 EP2610、EP3500、EP3820、**EP290**。
  **3/4 次晚期最好；1/4 次早期最好**（EP290，best 957.8）。
- 论文说 “gradually decreases and stabilizes”，**没有提退化**。
- 我们此前的 E2E retrain 日志被后续 ST run 混合/覆盖，不能给干净曲线。
- `e2e_control` 1000 episode 正在跑；这是干净的原模型对照。

### AGV：跑到 98 次验证后失败，已从 ep900 恢复

- 原 run 从 2026-10-22 01:49 跑到 07:53，exit code 4。
- 输出里没有 traceback；最后保存了 `final.pth` 和 98 个验证点。疑似后台任务运行约 6 小时后被环境终止。
- 98 次验证中最好 obj = **884.60**，在 episode 380（`EP380` 保留在 `trained_model/agv/`）。
- `full_ckpt_settrans_ep900.pt` 存在，包含 `agv_actor` 和 `agv_optimizer`。
- 已用 `--resume_ckpt ...ep900.pt --run_tag agv_resume` 从 ep900 恢复，跑最后 100 episode。
- 恢复后新的 `top1` 只基于 ep910–1000 的验证；**总体最好仍是 EP380（884.60）**。
- 恢复 run 使用新 tag，旧 `agv` 输出保留。
- 恢复完成（exit 0）：ep910–1000 验证在 **910.95–917.85**；
  最好 **910.95 @EP970**；最终 914.61。**总体最好仍是 EP380 884.60**。恢复没有改善。
- AGV 最好 EP380 的 seed3 测试：
  **obj 902.92 ± 136.16**；mk 590.28、pt 447.19、transT 199.94、idleT 1119.85。
  no-AGV 同一测试是 777.20。AGV 环境多一个资源约束，**不能直接当方法优劣**。

### 注意

- no-AGV 与 AGV 的验证口径不同：AGV 多一个资源约束（每车间车队 + 跨车间固定延迟）。
  884.60 vs 768.47 **不能直接当作方法优劣**。
- AGV 的 seed3 测试要等恢复完成后再跑。

## §9 2026-10-22 ST 机器 actor 补 ×10 打分缩放（200 episode 对照）

### 改动

- `Machine_Actor_SetTransformer` 增加 `SCORE_SCALE = 10.0`。
- 打分改为：`scores = self.m_policy(feat2).squeeze(-1) * self.SCORE_SCALE`。
- 原因：E2E 机器 actor 在 `actor_critic.py:477` 乘 10；
  ST 之前没有乘。`MLPActor` 末层前用 tanh，输出约在 (−1, 1)。
  少乘 10 后 softmax 更平，机器选择接近随机。
- 只改 actor 的打分尺度。Critic 输出 value，不需要缩放。
- `settrans` 和 `hybrid_full` 共用 `Machine_Actor_SetTransformer`，
  因此两者都会带上 ×10。旧 checkpoint 不受影响。
- 旧 ST 基线：最好 768.47 @ep90；final 856.5。
- E2E 对照：最好 703.26 @ep240（仍在跑）。

### 运行

- 6 episode smoke（`--encoder_type settrans --run_tag smoke_st_x10`）：
  通过，ep5 触发 PPO update。
- 200 episode：`--encoder_type settrans --episode_num 200 --run_tag st_x10`，
  2026-10-22 14:55 启动，后台任务 `be3drnz3h`。
- 当前（15:10）ep40/200。验证 ep10–40：
  **884.96、859.53、869.28、910.49**。
- 旧 ST 同期：841.45、863.32、795.08、832.12。
  ep20 略好 3.8；ep10、ep30、ep40 更差（+43.5、+74.2、+78.4）。
- 早期没有改善。需要跑完 200 再判断。
- `e2e_control`：ep900/1000；最好 **703.26 @ep240**；最近 ep900 = 849.21。

### 后续更正（18:30）

- ×10 已从 `Machine_Actor_SetTransformer` **移除**。
- 全 ST ×10（`st_x10`）跑到 ep150 已停：best 859.53 @ep20，ep150 889.72；
  比旧全 ST 768.47 更差。
- `hybrid_full` ×10 跑到 ep110 已停：best 865.30 @ep80，ep110 920.73；
  比旧 `hybrid_full` 764.89 更差。
- 结论：少 ×10 不是 ST 落后的主因；×10 反而让机器策略过早变尖。

## §10 2026-10-22 全量特征与跨 agent token

### 特征

- Job token：
  - 基础 12 维。
  - 原 8 维上下文（前驱延迟、计数、最晚前驱 ft、slack、剩余工序/加工时间）。
  - 4 维 edge 拆分（transport / idle 的 max/sum）。
  - 13 维新上下文（工序序号、候选机器数、t/p 的 min/mean/max、
    运输感知最早开始、makespan、已调度比例、平均利用率、job 计划完成）。
  - 合计 37 维。
- Machine token：
  - `m1`（6/10）+ `m2`（8）+ `m3`（8）。
  - `m3` 新增：空闲功率 p2、队列剩余加工时间、当前等待时间、
    候选任务数、机器关键度代理。
- 跨 agent：
  - `h_g_m_pooled` 作为全局 machine token 进 job ST encoder。
  - `h_pooled_o` 作为全局 job token 进 machine ST encoder。
  - 两者从打分头移除；`o_policy` / `m_policy` 输入从 `3H` 改为 `2H`。
- 只对 ST。E2E GAT 不变。

### 代码

- `DGenv_func.py`：
  - `split_edge_matrices`
  - `task_context_features`（13）
  - `machine_context_features_full`（8）
- `parallel_env.py`：
  - `cal_cur_task_context`
  - `cal_cur_task_machine_context`（3 或 8）
  - `edge_split_batch`
- `ReplayBuffer`：`task_ctx`、`task_ctx_`、`trans`、`idle`、`trans_`、`idle_`、
  `machine_fea3` 3/8 维。
- `Run.py` / `PPOAlgorithm` / `validate.py`：全部接线。
- 开关：`--st_full_context 1`。

### Smoke

- Job 全量（`job_st --st_token_context 1 --st_full_context 1`）：
  通过，ep5 PPO update，ep6 完成。
- Machine 全量（`hybrid_full --st_full_context 1`）：
  通过，ep5 PPO update，ep6 完成。
- 修复过：
  - `Parallel_env.total_task` → `ntasks`。
  - 最后一步 job_actor 未传 `task_ctx/trans/idle`。
  - `Global_Critic_JobST_MachineGAT` 的 `task_enc` 仍是 20 维。
  - `multi_v_s` 未传 `task_ctx/trans/idle`。

### 旧消融（已停）

| run | 最好 | 最后 |
|---|---|---|
| `ablation_job_st`（旧 8 维 + GAT machine） | **735.74 @ep110** | 750.69 @ep140 |
| `ablation_machine_st`（旧 m3=3） | 803.24 @ep50 | 828.81 @ep110 |

### 全量消融（18:28 启动）

- `ablation_machine_st_full`：`hybrid_full --st_full_context 1`。
- `ablation_job_st_full`：`job_st --st_token_context 1 --st_full_context 1`。
- 各 200 episode，无 ×10，同 seed / LR / schedule。
- 对照：E2E 703.26 @ep240 / 830.44；旧 hybrid_full 764.89 / 791.54。

## §11 2026-10-23 训练/验证/预测统一指标权重（E2E）

### 事实

- 环境奖励函数用 `configs.weight_*`，不是随机权重。
- 随机权重 `reward_random_weight` 有两个作用：
  1. 进入 task / machine 的输入特征；
  2. 进入 PPO 的 actor / local critic / global critic 四目标损失加权
     （`rw` 从 env 读取）。
- 所以“统一权重”必须同时固定这两处。

### 代码

- `trainer/parallel_env.py`：新增
  `self.weight_type = args.get('random_weight_type', '01')`；
  `init_DGFJSPEnv_state0` 的 `reset` 传
  `Random_weight_type=self.weight_type`。
- `Run.py`：done 后的 `reset` 也传同一类型。

### 冒烟

- 2 episode 通过；100 验证实例完成。
- 直接检查：env weights、task 特征尾部、machine 特征尾部
  均为 `[0.4, 0.4, 0.2]`。

### 运行（10:17 启动，各 200 episode）

- 统一权重：
  `--encoder_type e2e --LR 3e-4 --use_lr_decay 0 --random_weight_type eval --run_tag e2e_fixedw_nox`。
- 随机权重对照：同参数，`--random_weight_type 01 --run_tag e2e_randw_nox`。
- 旧 `e2e_control` 用 LR 衰减，不能作为干净对照。

### 判据

- 每 10 episode 的 seed1 验证（100 实例）。
- 看 best 之后是否持续上升；两条曲线同 schedule，只差权重。
- 单 seed、单次 run，结论需留噪声余量。

### 首批读数（11:10 已停）

| run | ep10 | ep20 | ep30 | ep40 | ep50 | ep60 | ep70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `e2e_fixedw_nox` | 744.02 | 738.61 | 745.59 | 725.65 | **720.30** | 726.99 | 720.78 |
| `e2e_randw_nox` | 1397.89 | 1369.00 | 1357.48 | **1347.68** | 1390.73 | 1355.48 | 1375.67 |

- 固定组 best 720.30 @ep50；随机组 best 1347.68 @ep40。
- 随机组到 ep70 仍约 1376。

### 随机组为什么比旧 e2e_control 高

- 随机权重路径没改：`random_weight_type=01` 仍走
  `generate_random_weights("01")`。
- 两个差异：
  1. 旧 run 用 LR 衰减；新 run 不衰减。StepLR 每次 PPO update 步进一次，
     update 每 5 episode 一次。
     ep10：旧约 2.43e-4 vs 新 3e-4；
     ep70：旧约 6.9e-5 vs 新 3e-4。
  2. 训练代码没有 `random.seed`。随机权重序列每次 run 不同。
     旧 run 的 ep10=735 是一次抽样；新 run 的 ep10=1398 是另一次。
- 旧 run 的 735 不能当作随机权重的稳定均值。
- 结论：这次 1300+ 不是固定权重改动造成的；
  是随机权重序列 + 无衰减的联合结果。

### 1000 episode 新 run

- 当前 `e2e_fixedw_nox` 是 200 episode 配置，已停在 ep70。
- 11:10 重启：
  `--random_weight_type eval --episode_num 1000 --LR 3e-4 --use_lr_decay 0 --run_tag e2e_fixedw_1000`。
- 单独跑，预计约 6–7 小时。

### 11:49 读数

- E2E `e2e_fixedw_1000`：
  ep10 811、ep20 1192、ep30 1410、ep40 1371、ep50 1325、
  ep60 1043、ep70 1011、ep80 739、ep90 **733**、ep100 774。
  ep30 是尖峰，已回落；当前 best 732.98 @ep90。
- job ST `job_st_taskf_fixedw`：
  ep10 985.8、ep20 1020.2、ep30 1038.1、ep40 1046.6。
  慢速变差；上次随机同段 753 / 1003 / 768 / 924。
- v5 源码已入库：
  `cb60009 chore(v5): track source and ignore training artifacts`（71 文件）。

### job ST 固定权重完成 + seed3 测试（12:47）

- `job_st_taskf_fixedw`：200/200，exit 0。
- seed1 验证 best：**676.17 @ep200**（top1 = EP200）。
- 曲线高度非单调：
  ep10 986、ep20 1020、ep50 1051、ep60 879、ep70 761、ep90 751、
  ep100 1069、ep110 870、ep150 1083、ep170 1121、ep190 784、ep200 **676**。
- seed3 测试（top1 EP200，greedy，100 实例）：
  **obj 659.85 ± 93.51**；
  mk 522.63、pt 513.85、transT 258.26、idleT 484.01。
- 对照论文 Table V P-G：
  732.5 ± 96.8（seed3 测试，100 实例，均值）。
  本次低 **72.65**（约 9.9%）。
- 注意：
  - 单 seed、单 run；top1 由 seed1 验证选出。
  - 模型不同：job ST + task 13 特征 + 固定权重；不是论文 P-G。
  - 结果文件：`trained_model/job_st_taskf_fixedw/eval_st_seed3.npz`。

### job ST 随机权重对照（13:37 启动）

- 配置同 `job_st_taskf_fixedw`，只改 `--random_weight_type 01`。
- `--encoder_type job_st --st_token_context 1 --st_task_features 1
  --episode_num 200 --LR 3e-4 --use_lr_decay 0 --run_tag job_st_taskf_randw`。
- 目的：判断 676.17 / 659.85 是否来自固定权重，还是单 seed 运气。
- 完成后同法跑 seed3 测试。

### job ST 随机权重完成 + seed3（15:11）

- seed1 验证 best：**699.50 @ep200**（top1 = EP200）。
- seed3 测试（greedy，100 实例）：
  **obj 673.82 ± 109.12**；
  mk 530.91、pt 513.91、transT 260.09、idleT 509.68。
- 对照固定组：
  - seed1：676.17
  - seed3：659.85 ± 93.51
- 固定组更低：
  - seed1 低 23.33
  - seed3 低 13.97
  - std 也更小：93.51 vs 109.12
- 两者都低于论文 P-G 732.5。
- 仍是单 seed；固定/随机各一次，未做同 seed 配对。
- 结果文件：`trained_model/job_st_taskf_randw/eval_st_seed3.npz`。

### E2E 固定权重 top1 EP200 seed3 测试（14:30）

- 训练验证 best **697.23 @ep200**。
- 之后退化：ep520 约 796。
- top1 EP200 的 seed3 测试（greedy，100 实例）：
  **obj 681.98 ± 91.25**；
  mk 486.21、pt 513.53、transT 265.54、idleT 572.43。
- 对照论文 P-G 732.5 ± 96.8：
  低 **50.52**（约 6.9%）。
- 对照 job ST fixed 659.85：
  E2E 高 **22.13**。
- 注意：
  - 单 seed；top1 由 seed1 验证选出。
  - 验证曲线在 ep200 后退化，但 top1 仍可用。
  - 结果文件：`trained_model/e2e_eval_ep200/eval_st_seed3.npz`。

### E2E 1000 停止（14:33）

- 停在 ep530/1000。
- top1 EP200 已保存；seed3 681.98 ± 91.25。
- 原因：ep200 后验证退化到约 796。

### 双 ST 随机权重（14:34 启动；14:47 改 1000）

- 14:34 先按 200 episode 启动；14:47 按用户要求停掉，改 1000。
- `--encoder_type settrans --episode_num 1000 --LR 1e-3
  --use_lr_decay 0 --random_weight_type 01 --run_tag settrans_randw_nox_1000`。
- job 和 machine 都用 ST；task/edge/full context 默认关闭。
- Adam（代码默认），无 LR 衰减。
- 1000 episode；第一个验证点 ep10。

### AGV WAIT 去掉（按 A，14:53）

- `available_mask` 改为只按车间过滤；busy AGV 也可选。
- `AgvActor_SetTransformer`：
  - 删除 `wait_token`。
  - action = `n_agv`，不再有 WAIT。
  - 全 mask 行加安全兜底，避免全 `-inf`。
- `parallel_env` / `validate`：
  - 去掉 WAIT 兜底。
  - 无 transport 的行 mask 全开；AGV loss 仍由 `agv_valid` 屏蔽。
- 单测：`test_agv.py` + `test_agv_transport.py` **13 passed**。
- 兼容性：旧 AGV checkpoint 含 `wait_token`，新 actor 不能直接 `load_state_dict`。
- 冒烟：2026-10-23 15:55 启动 `settrans_agv_randw_nox_200`。
  - `--encoder_type settrans --use_agv 1 --n_agv 6
    --cross_delay 5 --cross_leg_ratio 0.5
    --episode_num 200 --LR 1e-3 --use_lr_decay 0
    --random_weight_type 01`。
  - job / machine 都用 ST；AGV 用 A 机制（无 WAIT，busy AGV 可选）。
  - 1000 settrans 已停在 ep154。

### AGV + 两个特征开关（16:00 重启）

- 停掉 `settrans_agv_randw_nox_200`（ep10）。
- 新 run：`settrans_agv_task_200`。
- `--encoder_type settrans --use_agv 1 --n_agv 6
  --cross_delay 5 --cross_leg_ratio 0.5
  --st_token_context 1 --st_task_features 1
  --episode_num 200 --LR 1e-3 --use_lr_decay 0
  --random_weight_type 01`。
- job token 维度：12 + 8 + 13 = 33。
- task 13 只进 job token；machine 侧仍是 ST，m3 用默认 3 维。

### 4090 远端部署（16:26）

- 远端：RTX 4090 24 GB，30 GB 根盘 + 50 GB 数据盘。
- 环境：`/root/venv`，基于远端 base Python 3.12 + torch 2.12.1+cu130。
- 补装：`gym==0.26.2`、`numpy==1.26.4`、`matplotlib==3.8.4`、
  `opencv-python-headless`、`plotly`、`rich`、`pynvml`、`wandb`。
- 代码/实例数据：`/root/plcsp_v5`；绝对路径已改为远端路径。
- 远端兼容修复：
  - `matplotlib.cm.get_cmap`：降 matplotlib 到 3.8.4。
  - visualizer：`jobs = sorted(df['Task'].unique())`，删掉 `jobs.sort()`。
  - 建 `results/`、`trained_model/`、`trajectory/`、`wandb/`。
- 运行：`settrans_randw_nox_1000_remote`
  - `--encoder_type settrans --episode_num 1000
    --LR 1e-3 --use_lr_decay 0 --random_weight_type 01`
  - 无 AGV，无 task features。
  - PID 3233，log `/root/plcsp_v5/remote_1000.log`。
- 16:26 状态：ep5，PPO update 1/200 正常；GPU 1.2 / 24 GB。
- 远端速度：约 2 s/episode；update 约 5–8 s。

### 10×10×2 远端并行（16:44）

- 本地/原仓库都没有 J10M10E2 的 pkl，远端现生成：
  - `Instance_J10M10E2.pkl`：12800，seed 0
  - `eval_Instance_J10M10E2.pkl`：100，seed 1
  - `test_Instance_J10M10E2.pkl`：100，seed 3
- 生成器要求 `csv_pth` 非空；传一个父目录不存在的路径，跳过 CSV，只存 pkl。
- 生成后自动启动：
  - `--n_job 10 --n_machine 10 --n_edge 2`
  - `--encoder_type settrans --episode_num 1000`
  - `--LR 1e-3 --use_lr_decay 0 --random_weight_type 01`
  - `--run_tag settrans_J10M10E2_randw_nox_1000_remote`
  - 无 AGV，无 task features。
- 远端 128 核 / 1 TB 内存；与 6×6×2 的 1000ep 并行。
- 16:44 状态：J10M10E2 生成中；6×6×2 run ep110。

## §12 2026-10-23 GRPO（critic-free）接入

- `--algo {ppo,grpo}`，默认 `ppo`；PPO 路径保留，未删。
- GRPO 组：`buffer_size = resample_freq = G`，同一批实例的 G 个 episode。
- λ：GRPO 强制 `random_weight_type eval`（固定 0.4/0.4/0.2），
  保证组内回报可比。
- 组回报：每个 env / episode 的 4 个最终 cost；λ 加权成标量；组内 z-score。
- 优势：所有 agent 共享 team advantage；无 critic / GAE。
- 损失：PPO clip + entropy；无 critic loss；可选 LR decay。
- `Run.py`：
  - GRPO 更新移到 episode 结束后（先收齐 G 个最终 cost）。
  - PPO 的 step 触发保留；GRPO 跳过。
  - 新增 `group_costs`。
- `algorithm/ppo_algorithm.py`：新增
  `global_update_JointActions_GAT_grpo`。
- Smoke：E2E，10 episode，G=5，两次 update，通过；
  验证 1 实例 777.39（不作数）。
- PPO smoke：E2E，6 episode，ep5 update 1/1，通过；PPO 路径未破坏。
- 未做：
  - local group advantage
  - KL / reference policy
  - 每组随机 λ
  - AGV GRPO smoke
  - multi-seed

### 远端 ST + GRPO（17:26 启动）

- `--encoder_type settrans --algo grpo --episode_num 200
  --buffer_size 8 --resample_freq 8
  --LR 1e-3 --use_lr_decay 0 --eval_sample 100`
- 无 AGV；17:36 改为**每组随机 λ**，验证固定 `0.4/0.4/0.2`。
- 旧固定 λ run 已停；新 run：`settrans_grpo_randlam_200_remote`。
- 新 run 首个 group λ = `[0.598, 0.245, 0.157]`。
- 首个 GRPO update 在 ep8。
- 远端 6×6×2 / 10×10×2 的 PPO run 继续用旧代码；
  新 GRPO run 用新代码。

### λ 网格（17:40）

- `e2e_control`（已完成）：
  - 四个 cost 几乎不随 λ 变。
  - 目标点 0.4/0.4/0.2：Obj 686.91。
- 论文 P-G（21 点，完成）：
  - 目标点 0.4/0.4/0.2：Obj **731.99**，与论文 732.5 对齐。
  - mk 530.8–543.0、pt 512.9–515.7、transT 263.4–274.1、
    idleT 594.2–695.4。
  - idle 有一定变化，但整体 trade-off 仍弱。
  - CSV：`/root/plcsp_v5/lambda_grid_paper_pg.csv`。

### AGV + task 200（完成）

- `settrans_agv_task_200`，随机权重，200/200，exit 0。
- best **774.96 @ep30**；final 818.36。
- 不如 job ST fixed 659.85 / E2E fixed 681.98。
- 但 encoder / 权重不同，不是干净的 AGV 对照。
- GRPO per-group λ run：ep96/200，update 正常。

### GRPO 固定 λ（已停）

- `remote_grpo_200.log`：6 次验证。
- best **1547.30 @EP20**；最后 EP60 = 1803.07。
- 为改每组随机 λ，未跑完；不是完整结果。

### GRPO per-group λ 完成（18:07）

- `settrans_grpo_randlam_200_remote`：200/200，exit 0。
- seed1 验证 best：**1434.33 @ep20**（top1 = EP20）。
- ep199：obj 1943.55；mk 1030.27、pt 541.56、
  transT 244.59、idleT 3164.76。
- 远差于 PPO 的 660–700。当前 GRPO 实现不稳，
  不能用于主结果；先保留作消融。
- 可能原因：组内共享标量优势 + 每组随机 λ +
  无 critic；尚未做单因素拆解。

## §13 2026-10-23 分层设计 A：顶层规则选择（PPO）

### 目标

- 底层不动：job / machine / AGV actor 的动作空间、
  critic、PPO loss 都不改。
- 顶层每 `ΔT=8` 个底层 step 选一次规则对。
- 先做 PPO。GRPO 代码路径保留，未删。

### 规则集（4 × 3 = 12）

Job 规则（对候选 task 打分）：

- `est`：最早预估开始。
- `eft`：最早预估完成。
- `spt`：最短预估加工时间。
- `mwko`：剩余工序最多（任务索引结构给出）。

Machine 规则（对候选 machine 打分）：

- `min_t`：最短加工时间。
- `min_pt`：最低加工能耗。
- `min_trans`：最短运输时间。

每个规则对 = `job_rule × 3 + machine_rule`。

### 接线

- 规则偏好按 batch 内 min-max 归一化，再减均值；
  只对未 mask 的候选生效。
- `bias = α × preference`，加到 job / machine logits，
  再过 mask 和 softmax。α 默认 1.0。
- 底层 PPO 更新重新跑 actor 时，从 replay buffer
  取回同一 step 的 bias；采样与更新口径一致。
- 顶层状态（10 维）：进度、可选 job 比例、
  候选 EST 最小/均值、候选 PT 均值、剩余工序均值、
  已调度 machine 的 FT / PT / 运输 / 空闲均值。
- 顶层 PPO：64 隐层 actor + critic；GAE；
  段回报 = 加权 cost 增量（负），按 episode 最终 cost 归一化。
- 验证：顶层按 argmax 选规则，不采样；
  与训练共用同一 bias 路径。

### 参数

```text
--hier 1
--hier_interval 8
--hier_alpha 1.0
--hier_lr 1e-3
--hier_epochs 4
--hier_entropy 0.01
```

### 保存

- `full_ckpt_*.pt` 新增 `hier_top_model` / `hier_top_optimizer`。
- 每次 eval 另存 `PPO_hier_top_*_EP{k}_.pth`；
  最终与 job/machine/critic 一起生成 `_final` / `_top1`。

### 验证

- 新增 `plcsp_v5/tests/test_hierarchy.py`：7 passed。
- `plcsp_v5/tests`：20 passed。
- Smoke 1：`settrans`，6 episode，buffer=1，hier=1；
  ep1–ep6 顶层 update 正常，验证走顶层 greedy。
- Smoke 2：`e2e`，2 episode；顶层 update 正常。
- 两个 smoke 都保存了 top 模型与 top1。

### 未解决 / 下一轮

- 顶层的加权 cost 用固定 `0.4/0.4/0.2`；
  底层 loss 仍用随机权重。这是有意保留的训练/评估口径，
  但需要单独说明，不能声称顶层也 preference-conditioned。
- 第一个 segment 的 cost 基线来自 env reset 的理想估计，
  含一次性偏移；未修。
- α、ΔT 都未调；12 条规则未做单规则消融。

### 远端启动（18:30）

- 代码已上传 `/root/plcsp_v5`，路径补丁已恢复；
  remote `py_compile` 与 `hierarchy` import 通过。
- 无 AGV。
- 配置与 `job_st_taskf_fixedw` 对齐，只加 `--hier 1`：

```text
--encoder_type job_st --st_token_context 1 --st_task_features 1
--random_weight_type eval --algo ppo
--hier 1 --hier_interval 8 --hier_alpha 1.0
--episode_num 200 --LR 3e-4 --use_lr_decay 0 --eval_sample 100
--run_tag job_st_taskf_fixedw_hier_200_remote
```

- PID 11586；log `/root/plcsp_v5/remote_hier_200.log`。
- 18:30 状态：`Training/hierarchy/init` 已打印，开始训练。
- 对照：job ST fixed seed3 **659.85 ± 93.51**。

## §14 2026-10-23 远端完成读数（21:40）

Episode 号按保存模型计（log 的 `episode=k` 对应 `_EP{k+1}`）。

| run | 状态 | 最好验证 | 最后验证 |
|---|---|---:|---:|
| 6×6 settrans 随机权重 1000 | 完成 | **826.86 @EP70** | 907.77 @EP1000 |
| 10×10 settrans 随机权重 1000 | 运行中 | 1802.64 @EP70 | 2208.75 @EP360 |
| job ST + task + hier 200 | 完成 | **762.48 @EP70** | 817.33 @EP200 |
| GRPO 每组随机 λ 200 | 完成 | 1434.33 @EP20 | 1943.55 @EP200 |

### 读数

- 6×6 settrans 1000 的最好点在 EP70；之后没有继续下降。
- 10×10 仍慢，约 5 小时跑到 EP360；当前读数很差。
- 分层 run 的 EP70 = 762.48，与非分层 job ST fixed 的
  EP70 ≈ 761 很接近；但到 EP200 没有出现 676 的下降。
- 分层 run 最好 762.48，差于非分层 job ST fixed
  seed1 676.17、seed3 659.85。
- 但训练未固定 seed；单次 run 不能证明分层有害。
- GRPO 仍不稳，不作为主结果。

### 下一步

- 对分层 top1（EP70）跑 seed3 测试，确认 762.48 的测试读数。
- 若要做因果对照：加 seed，跑同 seed 的
  `hier=0` / `hier=1` 两组。
- 10×10 是否继续跑，待定。

## §15 2026-10-23 主模型口径：双 ST

- 用户决定：主模型按 ST 走，即 job 和 machine 都用 Set Transformer。
- 主模型配置：

```text
--encoder_type settrans
--st_token_context 1 --st_task_features 1 --st_edge_features 0
--random_weight_type eval
--episode_num 200 --LR 3e-4 --use_lr_decay 0
```

- job token = 12 + 8 context + 13 task = **33 维**。
- `--st_edge_features 1` 暂不开：
  - 4 维 edge 与 13 维 task 的 transport-aware EST 部分重叠；
  - 未证明有增益；
  - 只单开 edge、不开 context 时，edge 不生效。
- `job_st`（ST job + GAT machine）降为**消融项**，不再是主模型。
- 之前的 `settrans_randw_nox_1000_remote` / 10×10 裸 ST：
  只作“无 context/task 的下界”，不能当主结果。

### 22:04 启动：主 ST 基线

- PID **20164**
- `--run_tag settrans_ctx_taskf_fixedw_200_remote`
- log `/root/plcsp_v5/remote_settrans_33_200.log`
- 确认 `Linear(in_features=33, ...)`。

### 22:05 启动：主 ST + 分层

- PID **20535**
- `--hier 1 --hier_interval 8 --hier_alpha 1.0`
- `--run_tag settrans_ctx_taskf_fixedw_hier_200_remote`
- log `/root/plcsp_v5/remote_settrans_hier_200.log`
- 确认 `Linear(in_features=33, ...)` 和 `Training/hierarchy/init`。

### 对照口径

- 主 ST 基线 vs 主 ST + 分层：只差 `--hier`。
- 旧的 `job_st + hier`（762.48 @EP70）保留为
  “GAT machine + 分层”消融，不再作主结果。

### 22:07 10×10 改为主 ST 口径

- 停掉裸 ST 10×10：`settrans_J10M10E2_randw_nox_1000_remote`。
  - 停在 EP360 / 1000。
  - 16:44–22:07，约 **5.4 小时** 单卡 4090。
  - 该读数只作“无 context/task 下界”，不进主结果。
  - 这是我的启动错误：主模型未定就开了 1000 ep。
- 重跑：

```text
--n_job 10 --n_machine 10 --n_edge 2
--encoder_type settrans
--st_token_context 1 --st_task_features 1 --st_edge_features 0
--random_weight_type eval
--episode_num 200 --LR 3e-4 --use_lr_decay 0
```

- PID **20936**
- `--run_tag settrans_ctx_taskf_J10M10E2_fixedw_200_remote`
- log `/root/plcsp_v5/remote_J10M10E2_st33_200.log`
- 与主 6×6 同特征、同权重、同 LR；只差实例尺寸。

### 22:10 清理死代码

- `encoder_type=settrans` 时不再先构造 E2E GIN/GAT 再替换。
- 只改构造顺序；主网络、损失、更新和结果不变。
- 本地实例化检查：job / machine / critic 均为 ST。
- 远端 `py_compile` 通过；运行中的进程不受影响。

## §16 2026-10-23 偏好嵌入（原始 E2E）

### 目的

- 论文 P-G 与 `e2e_control` 的 λ 网格都太弱：
  - mk / pt / transT 几乎不随 λ 变。
  - idle 有一点变化。
- 试一个学习到的偏好嵌入，替换原始的 3 个 λ 标量特征。
- 基线用**原始 E2E**，不用 ST。

### 实现

- 新增 `model/preference.py`：

```text
e(λ) = MLP(3 → 32 → 16), GELU
```

- 在 `model/actor_critic.py` 的 E2E 三类网络中加入：
  - job actor 打分头：3H + 16
  - machine actor 打分头：3H + 16
  - job / machine 局部 critic：H + 16
  - global critic：2H + 16
- `pref_embed=1` 时：
  - task 特征第 9–11 列（3 个 λ）置 0。
  - machine m2 特征第 5–7 列（3 个 λ）置 0。
  - 偏好只从 `e(λ)` 进入。
- PPO 四目标 loss 仍按 λ 加权，不改。
- `PPOAlgorithm`：
  - 共享 `pref_embed` 与独立 `pref_optimizer`。
  - PPO 更新中，actor loss 用 `retain_graph=True`；
    全局 critic loss 反传后，再 step `pref_optimizer`。
  - GRPO 路径也接入了 `pref_embed`。
- `Run.py`：
  - 每个 episode 从 env 的 `reward_random_weight` 算 `e(λ)`。
  - 验证用 `eval_variant` 的 `0.4/0.4/0.2`。
  - full ckpt 保存 `pref_embed` / `pref_optimizer`。
  - 另存 `PPO_pref_embed_*_final/_EP/_top1`。
- `parameters.py`：

```text
--pref_embed 1
--pref_dim 16
```

- 仅支持 `encoder_type=e2e`；其他 encoder 会报错。

### Smoke

- `e2e + pref_embed=1`，1 episode，buffer=1，K=1：通过。
  - job / machine 打分头 `in_features=400`。
  - global critic `in_features=272`。
  - 保存 `PPO_pref_embed_*`。
- `e2e + pref_embed=0`，1 episode：通过。
- 新增 `tests/test_preference.py`：2 passed。
- `test_hierarchy + test_preference`：9 passed。

### 未做

- 未跑长训练。
- 未改 `eval_lambda_grid.py`：
  - 需要加载 `PPO_pref_embed_*`。
  - 每个 λ 点要算 `e(λ)` 并传入 `validate_cost...`。
- 未做同协议重训的 E2E 基线；先用现成 `e2e_control` 网格。

### 下一步

- 训练 B 组：E2E + pref，200 ep，随机 λ。
- 跑 21 点 λ 网格。
- 对照：
  - `e2e_control` 目标点 686.91。
  - 论文 P-G 731.99。
- 判定：
  - 目标点不能明显变差。
  - mk / pt / transT 至少一个范围变宽。
  - 只看 idle 变宽不算成功。

### 22:59 本地 B 组启动

- 本地 4060 Laptop；`D:/anaconda/envs/e2e/python.exe`。
- PID **43544**。
- log `C:/Users/evari/AppData/Local/Temp/e2e_pref_200_local.log`。
- 配置：

```text
--encoder_type e2e --pref_embed 1 --pref_dim 16
--random_weight_type 01 --algo ppo
--episode_num 200 --LR 3e-4 --use_lr_decay 0
--eval_sample 100 --run_tag e2e_pref_200_local
```

- 22:59:42 开始第一个 batch；23:00:12 完成 ep1，23:00:14 ep2。
- 约 2 s/episode；200 ep + 20 次 100 实例验证。
- 训练完后需要用本地版 `eval_lambda_grid.py` 跑 21 点网格；
  该脚本已按 `pref_embed` 改好，但当前部署在远端。
- 备注：后台启动时用 `runpy` 包装 Run.py，
  DataLoader worker 会重新导入包装脚本；
  必须加 `if __name__ == "__main__"` 守卫，否则 worker 递归启动。

## §17 2026-10-23 23:00 主 ST 两组完成

| run | 状态 | 最好验证 | 最后验证 |
|---|---|---:|---:|
| 6×6 双 ST 基线 | 完成 | **811.45 @EP20** | 887.53 @EP200 |
| 6×6 双 ST + 分层 | 完成 | **743.06 @EP50** | 871.25 @EP200 |
| 10×10 双 ST | 运行中 | 2354.40 @EP40 | 2354.40 @EP40 |

### 读数

- 双 ST 基线最好 811.45；分层把最好点降到 743.06。
- 但两者都差于 `job_st`（ST job + GAT machine）：
  - seed1 **676.17**
  - seed3 **659.85**
- `job_st + hier` 最好 762.48 @EP70。
- 因此：
  - 分层对双 ST 有改善（811 → 743）。
  - 但双 ST 仍不如 `job_st`。
  - machine ST 至少在当前设置下没有带来收益。
- 10×10 双 ST：EP40 = 2354.40，约 57 分钟跑 40 ep；
  200 ep 预计还要约 4–5 小时。

### 待定

- 是否停 10×10 双 ST。
- 主模型是否回到 `job_st`，或先做 machine encoder 的单变量消融：
  `job_st`（GAT machine） vs `settrans`（ST machine），
  同 job 特征、同 LR、同 seed。
- 当前主 ST 读数不支持“双 ST 一定优于 job_st”。

## §18 2026-10-23 默认口径与论文学习率

### 用户规定的默认口径

以后跑，除非特殊说明：

```text
--encoder_type settrans
--st_token_context 1 --st_task_features 1
--random_weight_type 01
--LR 1e-3
```

- job 和 machine 都用 ST。
- 全特征 = 33 维（12 + 8 context + 13 task）。
- edge 4 维暂不开。
- 默认无 AGV、无分层。
- 特殊实验必须显式写全参数。

已写入根 `CLAUDE.md` §4。

### 论文学习率

E2E-MAPPO 论文超参（记录于 `docs/citation-cards.md`）：

```text
Adam lr = 1e-3
Adam eps = 1e-10
γ = 0.99
GAE λ = 0.98
PPO clip ε = 0.2
Training iteration = 1000
```

代码默认：

```text
--LR 0.001      ← 与论文一致
--lr_eps 1e-5   ← 与论文 1e-10 不一致
```

所以复现论文口径必须显式：

```text
--LR 1e-3 --lr_eps 1e-10
```

### 与当前 run 的关系

- 6×6 双 ST 固定 λ、LR 3e-4：是 machine encoder 对照，不是默认口径。
- 10×10 双 ST 固定 λ、LR 3e-4：不符合新默认，建议停。
- 本地 E2E + pref：特殊实验（原始 E2E）；
  用随机 λ，但 LR 3e-4。若要按论文 LR，需要停掉重起 1e-3。
- 以后默认 run 按 §18 口径。

### 23:25 本地 pref 按论文 LR 重启

- 停掉旧的 LR 3e-4 run；删除其 `trained_model/results/trajectory`。
- 新 run：

```text
--encoder_type e2e --pref_embed 1 --pref_dim 16
--random_weight_type 01 --algo ppo
--episode_num 200 --LR 1e-3 --use_lr_decay 0
--eval_sample 100 --run_tag e2e_pref_200_lr1e3_local
```

- PID **18084**
- log `C:/Users/evari/AppData/Local/Temp/e2e_pref_200_lr1e3_local.log`
- 23:25:31 开始第一个 batch。
- ⚠️ `lr_eps` 仍是代码默认 **1e-5**；论文是 **1e-10**。
  这一组只改了 LR，没有改 Adam eps。

### 23:25 远端状态

- 只剩 10×10 固定 λ run（PID 20936）。
- EP60：obj 2230.31；EP50 2318.64；EP40 2354.40。
- 不符合新默认（固定 λ、LR 3e-4）；建议停。

### 23:34 10×10 按默认口径重启

- 停掉固定 λ run（EP60 = 2230.31）。
- 新 run：

```text
--n_job 10 --n_machine 10 --n_edge 2
--encoder_type settrans
--st_token_context 1 --st_task_features 1 --st_edge_features 0
--random_weight_type 01 --algo ppo
--episode_num 200 --LR 1e-3 --use_lr_decay 0 --eval_sample 100
--run_tag settrans_ctx_taskf_J10M10E2_randw_lr1e3_200_remote
```

- PID **25238**
- log `/root/plcsp_v5/remote_J10M10E2_randw_lr1e3_200.log`
- 23:34:19 开始第一个 batch。
- 首次启动崩溃：ST actor / critic 的 `forward` 没有
  `pref_emb` 参数，而新 Run.py 无条件传了这个关键字。
  已在 ST 三个类上加入 `pref_emb=None` 并忽略；
  本地 1 ep smoke 通过。

### 6×6 主 ST / 分层是否重跑

- 是。
- 旧 6×6 双 ST 基线 811.45、双 ST + 分层 743.06：
  都是固定 λ + LR 3e-4，不是默认口径。
- 它们只能作 machine encoder / 分层的**对照**，
  不能当主模型读数。
- 要得到默认口径的主结果，需要重跑：

```text
--encoder_type settrans
--st_token_context 1 --st_task_features 1 --st_edge_features 0
--random_weight_type 01 --algo ppo
--episode_num 200 --LR 1e-3 --use_lr_decay 0 --eval_sample 100
```

- 基线：不加 `--hier`。
- 分层：`--hier 1 --hier_interval 8 --hier_alpha 1.0`。
- 两组都应与旧固定 λ 组分开命名，旧组保留为对照。

### 23:49 默认 6×6 两组启动

- 基线：PID **25949**
  - `--run_tag settrans_ctx_taskf_randw_lr1e3_200_remote`
  - log `/root/plcsp_v5/remote_settrans_randw_lr1e3_200.log`
- 分层：PID **25950**
  - `--hier 1 --hier_interval 8 --hier_alpha 1.0`
  - `--run_tag settrans_ctx_taskf_randw_lr1e3_hier_200_remote`
  - log `/root/plcsp_v5/remote_settrans_randw_lr1e3_hier_200.log`
- 两组都是：双 ST、33 维、随机 λ、LR 1e-3、200 ep。
- 23:49 都已开始第一个 batch；分层已打印 `Training/hierarchy/init`。
- 10×10 默认 run（PID 25238）继续在跑，已到 EP60 附近。

## §19 2026-10-23 AGV 结果汇总

### Stage-0 AGV 口径

- 每车间独立车队；AGV 不跨车间。
- 跨车间 = 源车间 AGV 段 + 固定延迟 + 目标车间 AGV 段。
- 目标车间 AGV 用最早空闲规则；源车间 AGV 由 actor 选。
- 无 WAIT；busy AGV 可选。
- 无充电、无 L/U、无 AGV 故障。
- TEC 只计载货行驶；空驶和 AGV 等待不计；无 AGV 功率。

### run 1：`agv`（settrans + AGV，1000 ep）

- 98 次验证；最好 **884.60 @EP380**。
- 环境中断后从 ep900 恢复；恢复最好 910.95 @EP970，最终 914.61。
- seed3 测试（EP380）：
  **902.92 ± 136.16**；mk 590.28、pt 447.19、trans 199.94、idle 1119.85。
- 同测试 no-AGV = 777.20。
- ⚠️ AGV 环境多一个资源约束，不能直接当方法优劣。

### run 2：`settrans_agv_task_200`（默认特征 + AGV）

```text
--encoder_type settrans
--st_token_context 1 --st_task_features 1
--use_agv 1 --n_agv 6 --cross_delay 5 --cross_leg_ratio 0.5
--random_weight_type 01
--episode_num 200 --LR 1e-3 --use_lr_decay 0
```

- 200/200，exit 0。
- best **774.96 @EP30**；final 818.36。
- 这组是“默认口径 + AGV”。
- 对应的 no-AGV 默认口径刚在 23:49 启动（PID 25949）。
- 因此现在还不能给出干净的 AGV / no-AGV 对照。
- 不能拿它直接对 job ST fixed 659.85 / E2E fixed 681.98：
  encoder、权重、长度都不同。

## §20 2026-10-24 00:03 AGV 规则派车 B 组

### 实现

- 新增 `--agv_rule 1`。
- 环境仍是 `--use_agv 1`。
- 源车间 AGV：最早空闲规则。
- 目标车间 AGV：保持最早空闲规则。
- AGV actor 不参与训练：
  `train_agv_actor = use_agv and not agv_rule`。
- 其他网络、特征、LR、λ 采样不变。
- 新增 `agv.state.earliest_free_agv`。
- `pytest test_agv + test_hierarchy + test_preference`：19 passed。
- 本地 1 episode rule smoke：通过；验证 770.25（单实例，不作数）。

### B 组启动

```text
--encoder_type settrans
--st_token_context 1 --st_task_features 1 --st_edge_features 0
--use_agv 1 --n_agv 6 --cross_delay 5 --cross_leg_ratio 0.5
--agv_rule 1
--random_weight_type 01 --algo ppo
--episode_num 200 --LR 1e-3 --use_lr_decay 0 --eval_sample 100
```

- PID **27265**
- log `/root/plcsp_v5/remote_settrans_agv_rule_task_200.log`
- 00:03 开始第一个 batch。

### A / B / C

| 组 | 环境 | 源 AGV | 状态 |
|---|---|---|---|
| A | no-AGV | 无 | 默认 6×6 在跑 |
| B | AGV | 最早空闲规则 | 00:03 启动 |
| C | AGV | 学习 actor | 已完成：774.96 @EP30 / 818.36 |

### 同时段其他读数

- 6×6 默认基线：EP50，best 817.31 @EP30，last 913.85。
- 6×6 默认分层：EP50，best 801.55 @EP40，last 838.28。
- 10×10 默认：EP20，2013.08；仍慢。
- 全部是早期读数，不作结论。

## §21 2026-10-24 v5 实验计划

- 已写 `docs/v5/experiment-plan.md`。
- 主模型工作名：**H-ST-MAPPO**。
- Agent 命名：
  - Rule Selector（RS）
  - Job Agent（JA）
  - Machine Agent（MA）
  - AGV Agent（AA）
  - Global Critic（GC）
- 主模型 = RS on + JA / MA / AA 全 ST + GC ST + AGV actor。
- 计划含 E0–E9、预算、停止规则、口径风险。

## §22 2026-10-24 00:28 本地 E2E + pref（LR 1e-3）完成

- run：`e2e_pref_200_lr1e3_local`
- 200/200，exit 0。
- seed1 验证 best：**1241.28 @EP110**。
  - mk 956.40、pt 523.67、trans 213.37、idle 1516.45。
- final EP200：**1407.37**。
- 对照：
  - `e2e_control` 目标点 686.91（LR 3e-4 + decay，不同口径）。
  - 论文 P-G 731.99（原 E2E 口径）。
- 结论：
  - 按论文 LR 1e-3，这条 pref embedding 线不成立。
  - 目标点差约 554，不能进入主表。
  - 但缺少同 LR 的 E2E no-pref 对照，不能把全部差距归因于 embedding。
  - 也可能是 LR 1e-3 对新模块过高；旧的 LR 3e-4 pref run 在 EP10 被停，没有结果。
- 21 点 λ 网格尚未跑。
  - 目标点已失败；先不烧网格。
  - 若要诊断 trade-off，再跑。

## §23 2026-10-24 E8 取消

- 用户决定：偏好嵌入线不做了。
- 已有结果保留在 §22：
  - LR 1e-3 best 1241.28 @EP110；final 1407.37。
- 不再跑 21 点 λ 网格。
- 不再重跑 LR 3e-4。
- `pref_embed` 代码保留，但不进 v5 实验计划。
- v5 计划更新为：主模型 H-ST-MAPPO，baseline 表见
  `docs/v5/experiment-plan.md` §6。

## §24 2026-10-24 NSGA-II 加入 baseline

- 用户指出 NSGA-II 缺失。
- 已加入 `docs/v5/experiment-plan.md`：
  - B13：NSGA-II baseline。
  - E10：主模型 21 点 λ 扫描 + NSGA-II 前沿；HV / IGD。
- 现状：
  - `plcsp_v3/baselines/nsga2.py` 存在。
  - 但它面向 v3 的三目标 / v3 DES，不能直接用于 v5。
  - 需要重写 v5 解码器（工序优先级 + 机器分配），评价走 v5 no-AGV 环境。
- 指标口径：
  - 3 目标：MKT / EC（PEC+SEC）/ TEC。
  - 同时报 4 个原始 cost。
  - 同一批 seed3 100 实例。
- 未跑；E10 未开始。
- 注意：E10 只做评测，不改网络；E8 仍取消。

## §25 2026-10-24 00:45 远端默认口径读数

### 6×6 默认两组完成

| 组 | 最好验证 | 最后验证 |
|---|---|---:|
| E0 双 ST 基线 | **774.72 @EP130** | 833.49 @EP200 |
| E1 双 ST + 分层 | **793.04 @EP180** | 816.41 @EP200 |

- E0 最好 cost：mk 520.17、pt 460.62、trans 235.15、idle 838.44。
- E1 最好 cost：mk 534.65、pt 461.89、trans 208.18、idle 881.98。
- 默认口径下，E0 优于旧的固定 λ 对照（811.45）。
- E1 没有优于 E0；与固定 λ 旧结果（743.06 vs 811.45）相反。
- 单 seed；训练未固定 seed。不能据此判分层无效。

### AGV 规则 B（EP160）

- best **761.60 @EP90**；last 781.19 @EP160。
- best cost：mk 518.11、pt 478.81、trans 140.70、idle 836.73。
- 对照 C（AGV actor）pilot：774.96 @EP30 / 818.36。
- B 目前优于 C；但 B 未跑完，C 是不同 seed。
- 还不能下“规则优于 actor”的结论。

### 10×10 默认（EP50）

- best **1579.28 @EP40**；last 1685.62 @EP50。
- 仍在跑；约 1 小时 12 分钟到 EP50。

### 当前

- 25238：10×10 默认
- 27265：AGV 规则 B
- E0/E1 已完成；E3/E8 已有结果；E10 未开始。

## §26 2026-10-24 00:50 E0 seed3 测试

- E0 top1（EP130）在 seed3 100 实例上：
  - **obj 775.27 ± 109.13**
  - mk 521.96
  - pt 452.41
  - transT 234.43
  - idleT 846.59
- 对照：
  - 论文 P-G 732.5：E0 差 **+42.77（约 5.8%）**。
  - `job_st`（ST job + GAT machine）seed3：**659.85**。
  - E2E 固定权重 seed3：**681.98**。
- 旧的全 ST seed3：约 777.20。E0 的 775.27 与它一致。
- 结论：
  - “全 ST 比论文好”的记忆不成立于现有记录。
  - 比论文好的是 `job_st`（machine 用 GAT）和 E2E（GIN/GAT）。
  - 全 ST 的 job 侧没输；machine ST 是主要嫌疑。
- 仍缺：同 LR 1e-3、同随机 λ 的 `job_st` 对照。
- 学习率默认：以后未提 LR 时用 **1e-3**；已写入 `CLAUDE.md` §4。

## §27 2026-10-24 现实约束加入实验计划（C 线）

- 用户提出：机器故障、AGV 故障、机器缓冲区。
- 已写入 `docs/v5/experiment-plan.md` §12：
  - C1 机器故障：先 eval-only，再训练。优先级高。
  - C2 AGV 故障：先 eval-only，再训练。优先级高。
  - C3 有限缓冲：base 稳定后再做。优先级中高。
  - C4 交期 / 维护：可选，单独口径。
- 原则：
  - 主表 E0–E10 先不加约束。
  - 每次只加一个。
  - base 与约束结果分表。
- 当前不实现；先等主模型和 AGV B/C 读数。

## §28 2026-10-24 约束代码：C1 / C2 第一版

### 已实现

- 新增 `plcsp_v5/constraints.py`：
  - `failure_probability(active, mtbf)`
  - `sample_repair(rng, active, mtbf, mttr)`
- 参数：
  - `--machine_failure --machine_mtbf --machine_mttr`
  - `--agv_failure --agv_mtbf --agv_mttr`
  - `--failure_seed`
- 机器故障：
  - 在 `_schedule_task` 里，每个被调度工序按
    `1 - exp(-dur / MTBF)` 触发。
  - 修复时间 ~ Exponential(MTTR)。
  - 修复时间加到节点 `duration`，影响工序完工、机器可用性和后续排程。
  - 能耗仍用原始 `p*t`，不把修复算成加工能耗。
  - 每个并行 env 独立 RNG。
- AGV 故障：
  - 只对源车间被选 AGV 生效。
  - 按运输活跃时间触发；修复加到 `free_time` 和 transport override。
  - 训练在 `Parallel_env`；验证在 `validate.py`（GAT / ESA 两条路径）。
- 状态特征：
  - 不加新特征；策略通过延迟后的时间/空闲和 mask 间接看到影响。

### 测试

- `test_constraints.py`：3 passed。
- `test_constraints + test_agv`：13 passed。
- `plcsp_v5/tests` 全量：**26 passed**。
- 1 episode smoke：
  - `--machine_failure 1 --agv_failure 1 --use_agv 1 --agv_rule 1`
  - MTBF=5 / MTTR=2
  - PPO update 和验证完成，exit 0。
  - 单实例 1179.87；只证明代码通，不作读数。

### 未实现

- C3 有限缓冲：只有 `--buffer_capacity` 参数，**逻辑未写**。
- 未部署到远端；远端正在跑的三组不受影响。
- 未跑正式故障训练/评测。

## §29 2026-10-24 C3 缓冲 + 批量队列

### C3 有限缓冲（Stage-0）

- `constraints.py` 新增：
  - `buffer_mask`
  - `buffer_mask_for_env`
- `--buffer_capacity N`：
  - 统计 machine route 中 `start_time > ready` 的任务。
  - 等待数 >= N 时 mask 该机器。
  - 若某行全部被 mask，回退到原 mask，避免 softmax 全 -inf。
- 训练：`Parallel_env.machine_buffer_mask`，在 `Run.py` 合并。
- 验证：`validate.py` 的 GAT 路径合并。
- 这是启发式 Stage-0，不是精确阻塞仿真。

### 批量队列

- `plcsp_v5/run_queue.py`
- `plcsp_v5/experiment-queue.json`
- 功能：
  - 最多并行 2 个
  - GPU 空闲 < 3 GB 暂停启动
  - `after` 依赖
  - `.done` 标记
  - `status.json`
  - `--dry-run` / `--only` / `--force` / `--max-parallel`
- 队列内容：
  - E4b 主模型
  - E4a RS + AGV 规则
  - E5 job_st
  - E3 AGV actor
  - E4b 三个 eval-only 约束测试
  - E5 / E4a / E3 的 seed3 评测
  - E7 10×10 默认关闭
- 本地验证：
  - `plcsp_v5/tests`：**27 passed**
  - `run_queue.py --dry-run`：通过
  - 1 episode 队列 smoke：exit 0，`.done` 与 `status.json` 正常
- 未部署、未启动正式队列。

## §30 2026-10-24 01:48 远程队列启动

### 队列

- PID **32205**
- `experiment-queue.json`，`max_parallel=3`
- log `/root/plcsp_v5/queue_logs/queue.log`
- 状态 `/root/plcsp_v5/queue_logs/status.json`
- 启动即跑：
  - `gen_J10M6E2`
  - `E4b_main`（RS + AGV actor）
  - `E4a_rule`（RS + AGV 规则）
- `gen_J10M6E2` 已完成：
  - 生成 `Instance_J10M6E2.pkl`
  - `eval_Instance_J10M6E2.pkl`
  - `test_Instance_J10M6E2.pkl`
- 后续由队列自动跑：
  - E5 job_st
  - E3 AGV actor
  - J10M6E2 主模型 / no-AGV（seed 1）
  - 各 seed3 与约束 eval

### AGV 规则 B 完成

- `settrans_agv_rule_task_200_remote`：200/200。
- best **756.94 @EP170**；final **762.85**。
- 对照 C（AGV actor）pilot：
  - best 774.96 @EP30
  - final 818.36
- B 优于 C，但单 seed；还不能定论。

### 10×10 默认（旧 run）

- 跑到 EP100。
- best 1579.28 @EP40；last 2092.71。
- 仍在跑；不是队列里的任务。

## §31 2026-10-24 01:55 停旧 10×10

- 用户确认停止旧 10×10（PID 25238）。
- 最后验证 EP100：obj 2092.71。
- best 仍是 1579.28 @EP40。
- 进程和数据加载 worker 已清理。
- GPU 从 6.8 GB 降到 3.5 GB。
- 队列继续：
  - E4b_main（32274）
  - E4a_rule（32277）
  - E5_job_st（32907）
- 队列状态：`queue_logs/status.json`。
- 后续：E3、J10M6E2 主模型 / no-AGV、seed3 与约束 eval。

## §32 2026-10-24 01:54 队列并行度 3 → 6

- 用户要求提高 GPU / CPU 利用率。
- 旧队列与 3 个 run 已停；partial 产物已清。
- 旧日志归档：
  - `queue_logs/queue_3par.log`
  - `queue_logs/status_3par.json`
- `max_parallel` 改为 **6**。
- 新队列 PID **33524**。
- 6 个训练 job 同时启动：
  - E4b_main 33528
  - E4a_rule 33531
  - E5_job_st 33534
  - E3_agv_actor 33537
  - J10M6E2_main 33540（seed 1）
  - J10M6E2_noagv 33543（seed 1）
- 启动时 GPU 3.0 / 24 GB，17%。6×6 模型小，env 是 CPU 单线程；
  显存与 GPU 利用率不会接近饱和。PPO update 时会短时上升。
- 若还要提高吞吐，只能加更多 job：
  - seed 2 / 3 重复主模型
  - 10×10 主模型
  - 或增加每 run 的 env_batch（会改协议，不建议）。

## §33 2026-10-24 05:11 队列完成

15 个启用的 job 全部完成，失败 0。GPU 已空闲。

### 训练（seed1 验证）

| job | 最好 | 最后 |
|---|---:|---:|
| E4b 主模型（RS + AGV actor） | 749.22 @EP60 | 821.55 |
| E4a（RS + AGV 规则） | 820.52 @EP80 | 848.61 |
| E5 job_st（ST job + GAT machine） | **715.39 @EP20** | 777.37 |
| E3 AGV actor（无分层） | 848.87 @EP110 | 906.23 |
| J10M6E2 主模型（seed1） | **670.33 @EP70** | 774.87 |
| J10M6E2 no-AGV（seed1） | 800.26 @EP160 | 830.67 |

### seed3 测试

| job | obj |
|---|---:|
| E5 job_st | **696.28 ± 86.24** |
| E4b 主模型 base | **743.95 ± 105.92** |
| E4b + 机器故障 | 793.67 ± 109.79 |
| E4b + AGV 故障 | 749.32 ± 106.84 |
| E4b + 缓冲 cap1 | 697.12 ± 84.16 |
| E4a（规则 + 分层） | 842.43 ± 120.35 |
| E3（AGV actor，无分层） | 846.11 ± 120.88 |
| J10M6E2 主模型 | **672.66 ± 92.66** |
| J10M6E2 no-AGV | 805.94 ± 108.25 |

E4b base 是队列完成后手动补的。

### 读数

- 10×6×2：主模型 **672.66**，no-AGV **805.94**。
  - 同 seed1 训练，只差 AGV。
  - 主模型低 **133.28**；单 seed。
- E5 job_st seed3 **696.28**，比论文 P-G 732.5 低 **36.22**。
  - 旧 job_st（LR 3e-4）659.85 仍更好。
- E4b 主模型 seed3 **743.95**：
  - 比 E4a 低 98.48。
  - 比 E3 低 102.16。
  - 但仍差于 E5 696.28。
- 约束：
  - 机器故障 +49.72。
  - AGV 故障 +5.37。
  - 缓冲 cap1 −46.83。
  - ⚠️ 缓冲是 Stage-0 mask，有“全 mask 回退”；不能声称缓冲改善方法。

### 口径

- 6×6 组未设 `--seed`；10×6×2 是 seed 1。
- 全部单 seed。
- 论文 732.5 是 6×6 原 E2E；E5 696.28 是 ST job + GAT machine，
  口径不同，只作量级参考。
- 10×6 的 672.66 不能和 6×6 的 732.5 直接比。

## §34 2026-10-24 论文 10×6×2 对照与 machine ST 投影

### 论文数字（更正）

从论文 PDF Table V / Table VI 直接读：

| 规模 | 论文 P-G Obj | Std |
|---|---:|---:|
| 6×6×2 | 732.5 | 96.8 |
| **10×6×2** | **674.5** | **88.9** |
| 20×6×3 | 949.1 | 67.9 |
| 10×10×2 | 1654.8 | 126.1 |

来源：Wang et al., IEEE IoTJ 12(4):4420–4434, 2025,
Table V “Results of Proposed Approaches and Top-Ranked PDRs”；
Table VI 同值 674.5。

我们的 J10M6E2 主模型 seed3：

```text
672.66 ± 92.66
```

对照论文 10×6×2 P-G：

```text
674.5 ± 88.9
```

差 **−1.84**，std 92.66 vs 88.9。  
基本一致。

注意：

- 论文 P-G 假设运输资源充足（无 AGV 车队）。
- 我们的主模型有 AGV actor + 有限车队。
- 同规模下基本打平，但单 seed；不能当最终结论。
- 之前用 732.5 对比 10×6 是错的；732.5 是 6×6。

### machine ST 减少一层投影

当前代码：

- `Machine_Actor_SetTransformer.input_proj`：
  `Linear(in→128) → GELU → Linear(128→128)`。
- 即 **两层投影**。
- 之后 3 层 SAB + PMA。

记录里能查到：

- `hybrid_full`（GIN job + ST machine）：best 764.89 @EP110，
  final 791.54；比 E2E 703.26 差。
- `ablation_machine_st`（旧 m3=3）：803.24 @EP50 / 828.81 @EP110。
- ×10 打分：`st_x10` 859.53 @EP20 / 889.72；
  `hybrid_full ×10` 865.30 @EP80 / 920.73；更差，已移除。
- `ablation_machine_st_full` 只记录了启动，没有结果。
- 当前队列的 machine ST：
  - E4b 6×6 seed3 743.95。
  - E5 job_st（GAT machine）696.28。
  - machine ST 在 6×6 差约 48。
  - 10×6 主模型 672.66 与论文持平，但没有 10×6 的 GAT machine 对照。

结论：

- **“减少一层投影”的单独效果没有干净记录。**
- 不能把 6×6 的差距归因于这一层。
- 要做因果判断，必须同 seed、同特征、同 LR 跑：
  - machine ST：1 层投影
  - machine ST：2 层投影
  - machine GAT
- 当前代码是 2 层；没有 1 层的开关。

## §35 2026-10-24 machine ST vs machine GAT（统一换）

### 最干净的一对

同为：

- 6×6×2
- job 侧 ST
- 无 AGV
- 无分层
- 随机 λ
- LR 1e-3
- 200 ep
- 单 seed（未固定）

| run | machine actor | machine critic | seed3 |
|---|---|---|---:|
| E5 job_st | **GAT** | **GAT** | **696.28 ± 86.24** |
| E0 settrans | **ST** | **ST** | **775.27 ± 109.13** |

差：

```text
775.27 - 696.28 = +78.99
```

即 machine 侧全换 ST 后，seed3 差约 **79 分（11.3%）**，
std 也从 86.24 升到 109.13。

### 第二对（GIN job）

| run | job | machine actor | machine critic | 最好验证 |
|---|---|---|---|---:|
| E2E | GIN | GAT | GAT | 703.26 |
| hybrid_full | GIN | ST | ST | 764.89 |

差约 **+61.63**，方向一致。

### 当前是 actor + critic 一起换

- `settrans`：
  - machine actor = ST
  - global critic 的 machine encoder = ST
  - 没有“ST actor + GAT critic”的混合臂。
- `job_st`：
  - machine actor = GAT
  - global critic 的 machine encoder = GAT。
- 旧的 actor-only `hybrid` 已停，没有结果。

### 还没回答的部分

- 6×6 的 E4b（ST machine + AGV + RS）743.95
  —— 没有 GAT machine 对照。
- 10×6 的 main（ST machine + AGV + RS）672.66
  —— 没有 10×6 的 GAT machine 对照。
- 所以“machine ST 一定差”只在 6×6、无 AGV、无分层时成立；
  10×6 还不能下结论。

### 要补的对照

1. 6×6：E5 / E0 都加 `--seed 1`，确认 79 分。
2. 10×6：`job_st + hier + AGV`，seed 1，
   对 672.66 的 ST machine 主模型。

## §36 2026-10-24 10:30 四尺寸 × 3 seed 编码器对比

### 矩阵

| 项 | 值 |
|---|---|
| 尺寸 | 6×6×2、10×6×2、20×6×2、10×10×2 |
| 变体 | E5 `job_st`（GAT machine） / E0 `settrans`（ST machine） |
| seed | 1 / 2 / 3 |
| episode | 200 |
| λ | 随机 |
| LR | 1e-3，无衰减 |
| AGV / 分层 | 都关 |

- 训练：4 × 2 × 3 = **24**
- seed3 评测：**24**
- 数据生成：4
- 合计：**52 个 job**

队列：

- `experiment-queue-seeds.json`
- PID **53790**
- `max_parallel=6`
- log `queue_logs/queue_seeds.log`
- 状态 `queue_logs/status.json`
- 旧队列状态归档：
  - `queue_logs/status_4size_prev.json`
  - `queue_logs/queue_4size_prev.log`

### 启动状态

- 已启动：
  - `gen_J6M6E2`
  - `gen_J20M6E2`
  - `gen_J10M10E2`
  - `E5_J10M6E2_s1/s2/s3`
- `gen_J10M6E2` 因上一轮 `.done` 被跳过；pkl 已存在。
- 20×6×2 是**新生成**的尺寸。
- ⚠️ 论文的 20×6 是 **20×6×3**（949.1）。
  20×6×2 没有论文直接数字，不能并列。
- 论文同尺寸参考：
  - 6×6×2 P-G 732.5
  - 10×6×2 P-G 674.5
  - 10×10×2 P-G 1654.8

## §37 2026-10-24 鲁棒性实验状态

### 已做（只有一组）

对象：E4b 主模型（RS + AGV actor），单 seed，seed3 100 实例。
全部是 **eval-only**；模型没有在故障下训练。

| 设置 | Obj | 相对 base |
|---|---:|---:|
| base | 743.95 | — |
| 机器故障 MTBF=100 / MTTR=10 | 793.67 | **+49.72** |
| AGV 故障 MTBF=100 / MTTR=10 | 749.32 | **+5.37** |
| 缓冲 cap=1 | 697.12 | **−46.83** |

说明：

- 机器故障：每工序 Bernoulli + 指数修复；修复加到 duration。
- AGV 故障：只对源车间被选 AGV 生效；目标 AGV 仍是规则。
- 缓冲：Stage-0 mask；全 mask 行会回退原 mask。
  −46.83 不能当“缓冲约束改善方法”的证据。
- 单 seed，单 MTBF/MTTR，单 model。

### 未做

- 3 seed 重复。
- 故障率 / 修复时间网格（MTBF、MTTR）。
- 训练时开故障；现在只是测试扰动。
- AGV 目标侧故障、运输中故障、故障状态特征、动作 mask 的完整版。
- 缓冲容量 2/3；精确阻塞仿真。
- 10×6 / 10×10 的鲁棒性。
- 规则派车 vs AGV actor 在故障下的对比。
- PDR / 外部基线在故障下的对比。

### 结论

鲁棒性实验只做了 **一个 model、一个设置、单 seed**。
不能说“已做鲁棒性实验”。
当前 52-job 队列是编码器对比，不是鲁棒性。

### 决定（2026-10-24）

用户决定：**鲁棒性实验先不做。**
不建第二个鲁棒性队列。
§37 的“未做”清单保留，作为以后恢复的入口。
当前只跑 52-job 编码器对比队列。

## §38 2026-10-24 12:30 52-job 队列：6×6 训练内读数

口径：训练内每 10 ep 在固定 100 实例上 greedy 评测。
权重固定 0.4 / 0.4 / 0.2。与最终报告口径相同。
**不是**独立 `eval_st_seed3.py` 复算；复算 job 还没跑。
所有臂 **无 AGV、无分层**。

### 6×6×2 · `job_st`（job ST + machine GAT）

| seed | EP199 final | top1 值（所在 EP） |
|---:|---:|---:|
| 1 | 734.00 | 664.83（EP90） |
| 2 | 1031.84 | 706.32（EP10） |
| 3 | 939.27 | 935.50（EP190） |

- final 均值 **901.71 ± 152.44**（样本标准差，n=3）。
- top1 均值 **768.88 ± 145.78**。
- seed 2 的 top1 在 EP10；之后退化。

### 6×6×2 · `settrans`

| seed | EP199 final | top1 值 |
|---:|---:|---:|
| 1 | 839.80 | 789.05 |
| 2 | 854.91 | 806.62 |
| 3 | 843.17 | 790.31 |

- final 均值 **845.96 ± 7.93**。
- top1 均值 **795.33 ± 9.80**。
- 三个 top1 都在约 EP20。

### 6×6 对照

- top1：`job_st` 768.88 vs `settrans` 795.33，差 **26.45**。
- 但 `job_st` 的 seed 方差是 `settrans` 的约 15 倍。现在不能下结论。
- 论文 6×6×2 P-G **732.5 ± 96.8**（IEEE IoTJ 12(4), 2025, Table V/VI）。
  **论文不建 AGV 车队**；运输资源假设充足（见 §34）。
  所以本轮“无 AGV”与论文设定一致，**可以并列**。
  但 job 编码器不同：论文用 GIN，本轮两臂都用 ST。
  `job_st` 的 machine 是 GAT，与论文相同；machine 两边没有差异。
  所以对论文的差距**不在 machine**，在 job 编码器。
  只有 `job_st` vs `settrans` 才隔离 machine。
- 早先主模型（AGV + 分层）6×6 读数 743.95，单 seed。

### 10×6×2 · `job_st`（已完成）

| seed | EP9 | EP199 final | top1（所在 EP） |
|---:|---:|---:|---:|
| 1 | 820.51 | 826.05 | 721.93（EP30） |
| 2 | 674.70 | 865.53 | 674.70（EP10） |
| 3 | 1689.03 | 927.78 | 927.78（EP200） |

- EP199 均值 **873.12 ± 51.29**（跨 seed 标准差，n=3）。
- top1 均值 **774.80 ± 134.57**（跨 seed 标准差，n=3）。
- 三个 run 的 EP 编号相同，因为预算相同、同时启动。
  EP 是循环计数，不是 seed 变量。
- EP9 已经不同（820.51 / 674.70 / 1689.03），说明 seed 生效。

### 10×6×2 对论文

论文 10×6×2 P-G：**674.5 ± 88.9**。
来源：Wang et al., IEEE IoTJ 12(4):4420–4434, 2025, Table V；Table VI 同值。
性质：seed3 测试，100 实例；论文不建 AGV 车队。

| 臂 | 值 | 对论文 |
|---|---:|---:|
| `job_st` top1，seed1 | 721.93 | +47.43 |
| `job_st` top1，seed2 | 674.70 | +0.20 |
| `job_st` top1，seed3 | 927.78 | +253.28 |
| `job_st` top1 均值 | 774.80 ± 134.57 | +100.30 |

注意：

- 标准差不等于同一统计量：论文 ±88.9 是 100 实例内标准差；
  这里 ±134.57 是跨训练 seed 标准差。
- `job_st` 的 machine 是 GAT，与论文相同。
  对论文的差距**不在 machine**，在 job 编码器（GIN vs ST）。
- seed2 与论文持平；seed1 / seed3 明显更差。方差很大。
- `settrans` 10×6 三臂还在跑（查询时 EP39）。

### 观察

- 三个 `job_st` seed 的 top1 出现在 EP10 / EP90 / EP190。
- LR 1e-3 下存在明显不稳定迹象。本轮数据不足以判断原因。
- 需等 24 个独立 eval job 完成后再与论文并列。

## §39 2026-10-24 论文 Table IV 超参（直接读 PDF）

来源：Wang et al., IEEE IoTJ 12(4):4420–4434, 2025,
Table IV “Hyperparameters List Used in Training”，p. 4428。

| 超参 | 值 |
|---|---|
| Optimizer | Adam |
| Adam ε | 1×10⁻⁵ |
| Learning rate | 0.001 |
| Discount factor γ | 0.99 |
| GAE λ | 0.98 |
| PPO clip ε | 0.2 |
| Gradient clipping | 0.5 |
| LR decay step / ratio | 20 / 0.96 |
| Update epoch | 5 |
| Training iteration | 1000 |

### buffer 大小

**Table IV 没有 buffer 大小这一行。**
Algorithm 1 用符号 `TB`（ReplayBuffer batch size），也没有给数值。
唯一相关的数值是 Table IV 的 **Update epoch = 5**（即 `KE`）。

作者公开代码默认：

```text
third_party/E2E-MAPPO-for-MT-FJSP/parameters.py:77
buffer_size = 5
K_epochs = 5
```

所以 `buffer_size=5` 是**代码默认**，不是论文报告的数。

### 与当前 v5 队列的差异

当前：buffer 5、K_epochs 5、200 ep、`--use_lr_decay 0`。

| 项 | 论文 | v5 当前 |
|---|---|---|
| LR decay | step 20 / ratio 0.96 | 不衰减 |
| 训练预算 | 1000 iterations | 200 ep |

这两项都要在论文里交代。

## §40 2026-10-24 14:01 停旧队列，重跑 LR 3e-4 / buffer 10 / 400 ep

### 用户决定

```text
--LR 3e-4
--buffer_size 10
--episode_num 400
--resample_freq 5
--use_lr_decay 0
```

其余不变：3 seeds、4 尺寸、`job_st` / `settrans`、33 维特征、随机 λ、无 AGV、无分层。

### 旧队列

- 旧 PID 53790，配置 `experiment-queue-seeds.json`。
- 已停：队列父进程 + 5 个运行中的 `Run.py`。
- 旧 `.done` 和 `status.json` 归档到
  `/root/plcsp_v5/queue_logs/archive_200ep_lr1e3_bs5/`。
- 旧 200 ep 结果保留在旧 tag（`*_lr1e3_sX_remote`），不覆盖。

### 新队列

- 配置：`plcsp_v5/experiment-queue-seeds-lr3e4-bs10.json`
- 远端：`/root/plcsp_v5/experiment-queue-seeds-lr3e4-bs10.json`
- 父进程 PID **71461**
- log：`/root/plcsp_v5/queue_logs/queue_seeds_lr3e4_bs10.log`
- 52 job：24 训练 + 24 eval + 4 数据生成
- `max_parallel=6`
- 14:01 启动，先跑 6×6 两臂 × 3 seed。

### 口径说明

- 这不是论文 Table IV 口径。论文是 LR 1e-3 + decay 20/0.96；
  buffer 大小论文没报，作者代码默认 5。
- 本批是为了压不稳定：更低 LR + 每批 2 倍数据 + 2 倍预算。
- update 次数：400 / 10 = 40，和旧 200 / 5 相同。
- 每次 update：10 个 mini-batch × 5 epoch = 50 次梯度更新。

## §41 2026-10-24 16:25 用户要求停止远端全部任务

### 停止结果

- 队列父进程 71461 已停。
- 6 个 `Run.py` 已停。
- GPU：**0%**，0 MiB。

### 停止前的状态

6×6 两臂 × 3 seed **已跑完 400 ep**。
`top1` / `final` / `.done` 都已保存。

10×6 两臂 × 3 seed 在 15:55–16:01 启动，16:25 被停。
只跑了约 30 分钟，没有可用结果。

### 6×6 新配置 top1（验证集）

配置：LR 3e-4、buffer 10、400 ep、resample 5、不衰减。

| 臂 | s1 | s2 | s3 | 均值 ± 样本 std |
|---|---:|---:|---:|---:|
| `job_st` | 683.91 | 643.38 | 909.15 | **745.48 ± 143.18** |
| `settrans` | 803.87 | 793.84 | 762.73 | **786.82 ± 21.45** |

对照旧配置（LR 1e-3、buffer 5、200 ep）top1：

| 臂 | 旧均值 ± std | 新均值 ± std | 差 |
|---|---:|---:|---:|
| `job_st` | 768.88 ± 145.78 | 745.48 ± 143.18 | −23.40 |
| `settrans` | 795.33 ± 9.80 | 786.82 ± 21.45 | −8.51 |

注意：

- 这是**验证集** top1，不是 seed3 test。
- n=3，`job_st` 的 seed std 约 143。均值差不能当结论。
- `job_st` s3 = 909.15，和 `settrans` 重叠。
- 6×6 的 seed3 test eval 还没跑。

### 待定

远端已空。等用户决定下一步：

- 是否先改 machine ST 的 mask / attention；
- 是否只跑 6×6 的诊断对照；
- 是否恢复 10×6 / 20×6 / 10×10。

## §42 2026-10-24 16:39 machine mask 诊断对照

### 代码改动

新增参数：

```text
--machine_mask_attn   default 0
```

- 0：旧行为，encoder 不接收 mask。
- 1：machine actor 把 `mask=~mask` 传给 `SetTransformerEncoder`。
  encoder 的约定是 True = valid；actor 的 mask 是 True = blocked。
- `no_attn` 分支也做 masked mean。
- 只改 machine actor；global critic 和 job actor 不动。

文件：

- `plcsp_v5/model/actor_critic_st.py`
- `plcsp_v5/parameters.py`

远端：

- `actor_critic_st.py` 整文件上传。
- `parameters.py` 只加新行，保留远端的 `/root/...` 路径。
- 已确认远端 `--machine_mask_attn` 可解析。

### 验证

- 冒烟测试：mask0 / mask1 都能跑；概率和为 1；被屏蔽机器概率为 0；无 NaN。
- `pytest plcsp_v5/tests -q`：**27 passed**。
- `ruff`：`parameters.py` 有 10 个既有未用 import；本次没碰，v5 不在门禁范围。

### 诊断队列

配置：`plcsp_v5/experiment-queue-mask-diag.json`

| 项 | 值 |
|---|---|
| 尺寸 | 6×6×2 |
| 编码器 | `settrans` |
| 臂 | `mask0`（对照） / `mask1`（修复） |
| seed | 1 / 2 / 3 |
| ep | 400 |
| LR | 3e-4 |
| buffer | **5** |
| resample | 5 |
| decay | 0 |
| 训练 | 6 |
| seed3 eval | 6 |

- 父进程 **87544**
- log `queue_logs/queue_mask_diag.log`
- `max_parallel=6`
- 16:39 启动，6 个训练全部开始。

### 口径

- 唯一变量是 machine encoder 是否接收 mask。
- buffer 按用户要求恢复 5。
- 没有 GAT 臂；这个对照只回答 mask 对 ST 的影响。

## §43 2026-10-24 16:45 mask 当 bug 修：job + machine 都改，不设参数

### 用户决定

- 缺 mask 是 bug，不是可选口径。
- 不加 `--machine_mask_attn`；直接修。
- job 侧也一起修。
- buffer 恢复 5。

§42 的 `--machine_mask_attn` 参数和对应队列已撤销。

### 代码（最终）

`plcsp_v5/model/actor_critic_st.py`：

- **job actor**：
  - `valid_job = ~mask_operation`
  - 任务 token 属于 job `t // n_machine`。
  - `token_valid = valid_job[:, token_job]`
  - 传给 `encoder(x, mask=token_valid, ...)`。
  - 效果：已完成的 job 不再进入 job encoder 的 attention / PMA。
- **machine actor**：
  - `valid = ~machine_mask`
  - 传给 `encoder(feat, mask=valid, ...)`。
  - `no_attn` 分支也改成 masked mean。
- 两边都有 all-blocked 保护：encoder 不会 NaN。
  但全屏蔽不是合法动作状态，scorer 仍要求至少一个可选动作。
- **没有改** global critic 的 job/machine encoder。
  它们不接收动作 mask。局部 critic 用 actor 的 masked pool，自动继承。

`plcsp_v5/parameters.py`：

- 已删除 `--machine_mask_attn`。
- 远端 `parameters.py` 只删这一行，`/root/...` 路径保留。

### 验证

- 冒烟测试：
  - job 侧被屏蔽 job 概率为 0；概率和为 1；无 NaN。
  - mask 改变时 job pool 最大差 **0.317**。
  - machine 侧同样通过。
- `pytest plcsp_v5/tests -q`：**27 passed**。

### 新诊断队列

配置：`plcsp_v5/experiment-queue-maskfix.json`

| 项 | 值 |
|---|---|
| 尺寸 | 6×6×2 |
| 臂 | `job_st`（GAT machine） / `settrans`（ST machine） |
| seed | 1 / 2 / 3 |
| ep | 400 |
| LR | 3e-4 |
| buffer | **5** |
| resample | 5 |
| decay | 0 |
| 训练 | 6 |
| seed3 eval | 6 |

- 父进程 **89545**
- log `queue_logs/queue_maskfix.log`
- `max_parallel=6`
- 16:45:54 启动，6 个训练全部开始。

### 17:24 读数（EP99–109 / 400）

| 臂 | s1 | s2 | s3 |
|---|---:|---:|---:|
| `job_st` | 814.94 | 722.22 | 1082.71 |
| `settrans` | 827.76 | 835.16 | 802.55 |

mask 修复后，settrans 三 seed 在 EP99 都是 802–835。  
还没跑完，不能与旧结果配对比。

### 18:30 判断

EP239–259 中段均值：

- `job_st`：770.86
- `settrans`：839.72

对旧 400 ep / buffer 10 的 final EP399：

- `job_st`：+11.27
- `settrans`：−8.30

两个差都在 seed 噪声内。  
**mask 不是主因。**  
主因更可能在训练不稳定、预算、种子方差。

### 对照口径

- 两臂都带 **job 侧 mask 修复**。
- machine 侧：`job_st` 用 GAT（不接收 m3 / mask）；`settrans` 用 ST + mask。
- 旧的 `job_st` / `settrans` 读数都是**未修复 job encoder**的口径。
  不能把旧 `job_st` 和新 `settrans` 直接配对比。

## §44 2026-10-24 16:58 GRPO 符号修复 + E2E 重跑

### 发现

GRPO 的 group advantage 用了**正 cost**，直接送进 PPO clip loss。

- `Run.py:912-922`：`episode_costs` 是正数。
- `ppo_algorithm.py:1393-1398`：`adv = (scalar - mean) / std`。
- `ppo_algorithm.py:1482-1487`：`loss = -min(ratio*adv, clip*adv)`。

结果：cost 越高的 episode，advantage 越正，被越鼓励。  
PPO 的 reward 是负 cost，所以 PPO 没这个问题。

### 修复

```python
adv = -((scalar - mean) / std)
```

- 本地：`plcsp_v5/algorithm/ppo_algorithm.py:1398`。
- 远端：同一行已打补丁。
- 语义：正 advantage = cost 更低。

### 旧结果定性

`settrans_grpo_randlam_200_remote`
（best 1434.33 @EP20，final 1943.55）是**符号错误下的结果**。  
不能用来判断 GRPO 本身。

### 新队列

配置：`plcsp_v5/experiment-queue-grpo-e2e.json`

| 项 | 值 |
|---|---|
| 尺寸 | 6×6×2 |
| 编码器 | `e2e` = GIN job + GAT machine |
| 算法 | `grpo` |
| AGV | 无 |
| seed | 1 / 2 / 3 |
| ep | 200 |
| 组大小 G | 8 |
| buffer / resample | 8 / 8 |
| LR | 1e-3 |
| decay | 0 |
| λ | 每组随机，组内固定；评测固定 0.4/0.4/0.2 |

- 父进程 **92316**
- log `queue_logs/queue_grpo_e2e.log`
- `max_parallel=3`
- 16:57:57 启动，3 个训练全部开始。
- 与 mask-fix PPO 队列（89545）并行；GPU 8.3 GB。
- 首个 group λ：
  - s1 `[0.367, 0.633, 0.0001]`
  - s2 `[0.431, 0.026, 0.543]`
  - s3 `[0.355, 0.457, 0.188]`
- 首个 GRPO update 在 ep8。

### 17:24 读数（EP59 / 200）

| seed | EP29 | EP59 |
|---|---:|---:|
| 1 | 1325.71 | **728.02** |
| 2 | 718.20 | **717.24** |
| 3 | 1060.36 | **834.14** |

均值：1034.76 → **759.80**。  
无 NaN、无 Traceback。  
符号修复后不再朝高 cost 走。还没跑完。

### 18:18 读数（EP189 / 200）

| seed | EP59 | EP189 |
|---|---:|---:|
| 1 | 728.02 | **714.87** |
| 2 | 717.24 | **687.82** |
| 3 | 834.14 | **712.24** |

均值 **704.98**。三个 seed 收敛到 688–715。  
接近 PPO，也接近论文 6×6 P-G 732.5。  
还差 10 个 episode；top1 / final 未出。

### 18:20 seed3 test（完成）

| seed | test obj | std |
|---|---:|---:|
| 1 | **671.86** | 100.47 |
| 2 | **665.77** | 79.38 |
| 3 | **693.35** | 123.45 |
| 均值 | **676.99** | 跨 seed std 约 14.3 |

训练 best（验证集）：693.58 / 673.88 / 704.28。

对论文 6×6×2 P-G **732.5 ± 96.8**：

```text
676.99 - 732.5 = -55.51
```

口径：

- `e2e` = GIN job + GAT machine，无 AGV，无分层。
- GRPO，200 ep，G=8，LR 1e-3，每组随机 λ。
- seed3 test，100 实例。
- 论文是 PPO，1000 iterations；我们只跑 200 ep。
- 实例由本项目 `gen_dataset.py` 生成，不是论文实例的逐字复现。
- 单次训练预算下的 3 个训练 seed；不是严格复现。

这是目前最强的 GRPO 结果。符号修复是关键。

### 仍要看的

- 修符号后，GRPO 是否还会崩塌。
- 终端优势广播到所有 step、team advantage 共享、G=8 的方差。
  这三项没有改。

## §45 2026-10-24 17:21 本地 hybrid_full（GIN job + ST machine）1 seed

### 目的

验证 machine ST 的 mask 修复，在 **GIN job** 下是否有效。

- `hybrid_full` = GIN job actor + ST machine actor + ST machine critic。
- job 侧 mask 修复不适用：GIN job actor 不是 ST。
- machine 侧 mask 修复适用。

### 配置

```text
--n_job 6 --n_machine 6 --n_edge 2
--encoder_type hybrid_full
--algo ppo
--episode_num 400
--LR 3e-4
--use_lr_decay 0
--buffer_size 5
--resample_freq 5
--seed 1 --train_seed 1
--random_weight_type 01
--run_tag hybrid_full_J6M6E2_maskfix_s1_local
```

无 AGV、无分层。

### 环境

- 解释器：`D:/anaconda/envs/e2e/python.exe`
- torch 2.4.1+cu121，CUDA 可用。
- GPU：RTX 4060 Laptop 8 GB。
- 启动：17:21，device=cuda:0，global seed=1。
- log：系统临时目录 `hybrid_full_maskfix_s1_local.log`。
- checkpoint：`plcsp_v5/trained_model/hybrid_full_J6M6E2_maskfix_s1_local/`。

### 对照

跑完后与 **E2E（GIN + GAT）** 同协议对照。
旧 §35 第二对：E2E 703.26 vs hybrid_full 764.89（旧代码、单 seed）。
本轮是修复后的 machine ST，单 seed。

### 18:18 读数（EP29 / 400）

| EP | obj |
|---:|---:|
| 9 | 772.78 |
| 19 | 844.74 |
| 29 | 791.24 |

- 本地 4060 Laptop 约 **1.5 分钟 / episode**。
  400 ep 约 10 小时。
- GPU 22–38%，显存 7.2 / 8.2 GB。
- 还早，不能和旧 764.89 比。

### 18:24 本地停止，改远端

- 本地在 EP29 停止。
- 远端队列：`experiment-queue-hybrid-full.json`
  - 父进程 **102289**
  - 训练 PID 102292
  - log `queue_logs/queue_hybrid_full.log`
  - 18:24 启动。
- 配方不变：`hybrid_full`、6×6、seed1、400 ep、LR 3e-4、buffer 5。

## §46 2026-10-24 17:42 GRPO E2E 10×10×2 试点

### 目的

6×6 上 GRPO 修符号后接近 PPO。
现在看 10×10 是否也能站住。

### 配置

```text
--n_job 10 --n_machine 10 --n_edge 2
--encoder_type e2e
--algo grpo
--episode_num 200
--buffer_size 8 --resample_freq 8
--LR 1e-3
--use_lr_decay 0
--seed 1 --train_seed 1
--random_weight_type eval
--run_tag e2e_grpo_J10M10E2_randlam_s1_remote
```

无 AGV、无分层。  
1 seed 试点。

### 队列

- 配置：`plcsp_v5/experiment-queue-grpo-10x10.json`
- 父进程 **97570**
- log `queue_logs/queue_grpo_10x10.log`
- `max_parallel=1`
- 17:42 启动，训练 PID 97573。
- 首组 λ `[0.367, 0.633, 0.0001]`。
- 无报错。

### 并发情况

同时运行：

- mask-fix PPO 6×6：89545
- GRPO E2E 6×6：92316
- GRPO E2E 10×10：97570

GPU 99%，显存约 10 GB / 24.6 GB。

### 注意

- 10×10 单 episode 100 step，比 6×6 慢很多。
- 1 seed 只能看趋势，不能当最终结论。
- 6×6 的 GRPO 读数（EP59 均值 759.80）也还没跑完。

### 18:18 读数（EP19 / 200）

- obj **4886.62**。
- 首个 GRPO update 已完成，loss 有限，无 NaN。
- 10×10 单 episode 100 step，约 1.9 分钟 / episode。
  200 ep 约 6 小时以上，还要和另外 9 个进程共享 GPU。
- EP19 太早。6×6 GRPO 在 EP19 也很高（s1 1411），到 EP59 才降。
  等 EP50 后再看。

### 18:29 10×10 停止，改 10×6

- 10×10 停在 EP19（obj 4886.62）。
- 原因：约 1.9 分钟 / episode，200 ep 要 6 小时以上。
- 改为 10×6 GRPO，见 §47。

## §47 2026-10-24 18:29 buffer 转换修复 + 10×6 GRPO

### buffer 修复

`plcsp_v5/trainer/replaybuffer.py`：

- 新增 `use_task_features` / `use_edge_features` 标记。
- `numpy_to_tensor_operation` 只在对应开关打开时转换：
  - `task_ctx` / `task_ctx_`
  - `trans` / `idle` / `trans_` / `idle_`
- 关闭时直接返回 `None`，不再搬到 GPU。
- 数组分配没有改；`store_operation` 仍写这些数组。

效果：

- 10×10 的 4 个 100×100 矩阵约 2 GB，不再每次 update 搬 GPU。
- 当前 10×6 也受益。

验证：

- `pytest plcsp_v5/tests -q`：**27 passed**。
- 远端 `replaybuffer.py` 已同步。

### 10×6 GRPO

配置：`plcsp_v5/experiment-queue-grpo-10x6.json`

| 项 | 值 |
|---|---|
| 尺寸 | 10×6×2 |
| 编码器 | `e2e` = GIN + GAT |
| 算法 | `grpo` |
| seed | 1 |
| ep | 200 |
| 组大小 G | 8 |
| buffer / resample | 8 / 8 |
| LR | 1e-3 |
| decay | 0 |
| AGV | 无 |

- 父进程 **102961**
- 训练 PID 102964
- log `queue_logs/queue_grpo_10x6.log`
- 18:29 启动，GPU 99%，显存 8.2 GB。
- 首组 λ 待日志确认。

### 19:46 读数（EP99 / 200）

- obj **718.92**。
- 论文 10×6×2 P-G **674.5 ± 88.9**。
- 还差 100 个 episode。

### 20:12 读数（EP129 / 200）

- obj **637.09**。
- 已低于论文 674.5。
- 还差 70 个 episode。

### 20:30 读数（EP149 / 200）

- obj **637.32**。
- 稳定在 637 附近。
- 还差 50 个 episode。

## §48 2026-10-24 18:56 去掉 ST mask + machine logit 尺度测试

### 去掉 mask

`plcsp_v5/model/actor_critic_st.py`：

- job actor：删除 `valid_job` / `token_valid`，encoder 不再接收 mask。
- machine actor：删除 `valid` / masked mean，encoder 不再接收 mask。
- 回到 mask 修复前的行为。

保留：

- `replaybuffer.py` 的条件 tensor 转换（§47）。
- GRPO 符号修复（§44）。

验证：`pytest plcsp_v5/tests -q`：**27 passed**。

### 停止 mask-on 队列

用户判断 mask 不是主因，停止：

- mask-fix PPO 队列（6 个 6×6，约 EP260 / 400）。
- hybrid_full 远端队列（约 EP30）。
- 两者都是 mask-on；结果作废。

保留：

- GRPO 10×6：102961 / 102964。
- GRPO 6×6 已完成，结果见 §44。

### 新参数

```text
--machine_logit_scale   default 1.0
```

- `Machine_Actor_SetTransformer.forward`：
  `scores = scores * self.logit_scale`。
- GAT machine 原本硬编码 `* 10`。
- 这个参数只影响 ST machine。

### logit 测试

配置：`plcsp_v5/experiment-queue-machine-logit.json`

| 项 | 值 |
|---|---|
| 尺寸 | 6×6×2 |
| 编码器 | `settrans` |
| 臂 | `logit1`（scale 1） / `logit10`（scale 10） |
| seed | 1 / 2 / 3 |
| ep | 400 |
| LR | 3e-4 |
| buffer | 5 |
| AGV | 无 |
| mask | 无 |

- 父进程 **107632**
- log `queue_logs/queue_machine_logit.log`
- 18:59:48 启动，6 个训练全部开始。

### 第一次启动崩溃

- 18:56 第一次启动，6 个 job 在第一次 update 全部 rc=1。
- 原因：buffer 修复返回 `task_ctx=None` / `trans=None`；
  `Run.py:823` 的 trajectory 转换调用了 `None.cpu()`。
- 修复：
  `tensor.cpu().numpy() if tensor is not None else None`。
- 本地 py_compile 通过；远端已打补丁。
- 18:59 重启，6 个 job 正常，无 Traceback。

### 19:14 读数（EP49 / 400）

| 臂 | seed1 obj |
|---|---:|
| `logit1` | **726.22** |
| `logit10` | **797.06** |

×10 目前更差。还很早，不能下结论。

### 19:46 读数（EP139–149 / 400）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| `logit1` | 821.91 | 869.35 | 808.59 | **833.28** |
| `logit10` | 843.76 | 874.02 | 843.28 | **853.69** |

×10 三个 seed 都更差，均值 +20.41。  
还没跑完，但方向一致。

### 19:50 停掉 ×10

用户决定：停止 `logit10`，只保留 `logit1`。

- 停掉 3 个 `logit10_s1/s2/s3`。
- `logit1` 的 3 个继续跑到 400 ep。
- 最后读数：`logit1` 833.28 vs `logit10` 853.69。
- ×10 作废；×1 作为无 ×10 的 ST 基线。

### 20:12 读数（EP229 / 400）

| seed | obj |
|---|---:|
| 1 | 860.26 |
| 2 | 855.10 |
| 3 | 835.04 |
| 均值 | 850.13 |

比 EP139–149 的 833.28 更高（更差）。  
ST machine 在 835–860 附近波动。

### 20:30 读数（EP289 / 400）

| seed | obj |
|---|---:|
| 1 | 893.50 |
| 2 | 892.18 |
| 3 | 850.14 |
| 均值 | 878.61 |

继续变差。  
ST machine 的 best 应该在更早的 episode。

## §49 2026-10-24 19:13 E2E PPO 同协议对照

### 目的

GRPO 6×6 seed3 test 均值 **676.99**，低于论文 732.5。  
需要同架构、同实例、同预算的 PPO 对照，判断优势来自算法还是实例/预算。

### 配置

| 项 | 值 |
|---|---|
| 尺寸 | 6×6×2 |
| 编码器 | `e2e` = GIN + GAT |
| 算法 | `ppo` |
| seed | 1 / 2 / 3 |
| ep | 200 |
| buffer / resample | **8 / 8** |
| LR | 1e-3 |
| decay | 0 |
| λ | 每 episode 随机（`01`） |
| AGV | 无 |
| 分层 | 无 |

和 GRPO 的差别只有：

- PPO：critic + GAE。
- GRPO：group z-score，无 critic。
- λ 粒度：PPO 每 episode；GRPO 每组（算法内在差异）。

### 队列

- 配置：`plcsp_v5/experiment-queue-ppo-control.json`
- 父进程 **110707**
- 3 个训练已启动（19:13:38）。
- log `queue_logs/queue_ppo_control.log`
- eval：`EVAL_ENCODER=e2e`，seed3 test。

### 判读

- 如果 PPO 也约 730：GRPO 的 677 是算法优势。
- 如果 PPO 也约 680：优势来自实例或预算，不是 GRPO。

### 19:46 读数（EP49 / 200）

| seed | obj |
|---|---:|
| 1 | 804.78 |
| 2 | 720.64 |
| 3 | 978.43 |
| 均值 | 834.62 |

GRPO 在 EP59 的均值是 759.80。  
PPO 在 EP49 是 834.62。  
都还在早期，不能下结论。

### 20:12 读数（EP99 / 200）

| seed | obj |
|---|---:|
| 1 | 738.97 |
| 2 | 1020.09 |
| 3 | 1096.87 |
| 均值 | 951.98 |

seed 方差很大。  
PPO 还没有收敛到稳定区间。

### 21:07 seed3 test（完成）

| seed | test obj | std |
|---|---:|---:|
| 1 | 721.16 | 112.41 |
| 2 | 690.30 | 94.65 |
| 3 | 959.14 | 162.25 |
| 均值 | **790.20** | 跨 seed std **147.12** |

对 GRPO 同协议结果：

| 项 | GRPO | PPO |
|---|---:|---:|
| 均值 | **676.99** | 790.20 |
| 跨 seed std | **14.3** | 147.12 |
| 差 | | **+113.21**（GRPO 更好） |

逐 seed 差（GRPO − PPO）：

```text
s1: -49.30
s2: -24.53
s3: -265.79
```

三个 seed 全部 favor GRPO。

注意：

- n=3；PPO 的 s3 是离群点（959.14）。
- 配对 t 检验约 t=-1.48, p≈0.28；n=3 不能当显著。
- 但方向和稳定性都一致：GRPO 均值低 113.21，std 小一个数量级。
- 同协议：e2e GIN+GAT、6×6、200 ep、buffer 8、LR 1e-3、
  随机 λ、无 AGV、seed3 test。
- 只差算法：PPO critic/GAE vs GRPO group z-score；
  λ 粒度：PPO 每 episode，GRPO 每组。

### 20:30 读数（EP129 / 200）

| seed | obj |
|---|---:|
| 1 | 731.46 |
| 2 | 766.79 |
| 3 | 993.05 |
| 均值 | 830.43 |

比 EP99 的 951.98 好。  
seed3 仍高。还差 70 ep。

## §50 2026-10-24 20:04 machine ST 去掉跨机器 attention

### 目的

验证跨机器 attention 是不是 machine ST 的负担。

### 实现

已有开关：`--machine_no_attn 1`。

- 去掉 SAB 和 PMA。
- 保留 input projection。
- 逐机器输出 `h_nodes = feat`。
- `h_pool = feat.mean(dim=1)`。

注意：这是**同时去掉 SAB 和 PMA**。  
如果只想去掉 SAB、保留 PMA，用 `--machine_no_sab 1`。

### 配置

| 项 | 值 |
|---|---|
| 尺寸 | 6×6×2 |
| 编码器 | `settrans` |
| 臂 | `noattn` |
| seed | 1 / 2 / 3 |
| ep | 400 |
| LR | 3e-4 |
| buffer / resample | 5 / 5 |
| decay | 0 |
| AGV | 无 |

- 对照：`logit1`（同协议、machine 有 attention、scale 1），正在跑。
- 队列：`plcsp_v5/experiment-queue-machine-noattn.json`
- 父进程 **116225**
- log `queue_logs/queue_machine_noattn.log`
- 3 训练 + 3 eval；eval 也传 `--machine_no_attn 1`。
- 20:04 启动。

### 判读

- 如果 `noattn` 低于 `logit1`：跨机器 attention 是负担。
- 如果持平或更差：attention 不是主要问题，回到训练/预算。

### 20:12 读数（EP19 / 400）

| seed | obj |
|---|---:|
| 1 | 1020.70 |
| 2 | 973.93 |
| 3 | 866.01 |
| 均值 | 953.55 |

太早，不能比较。

### 20:30 读数（EP89 / 400）

| seed | obj |
|---|---:|
| 1 | 965.33 |
| 2 | 1061.87 |
| 3 | 825.51 |
| 均值 | 950.90 |

仍高。  
和 `logit1` 的 EP 不同，不能直接配对比。

### 20:43 读数（EP139 / 400）

`noattn`：

| seed | EP139 |
|---|---:|
| 1 | 906.83 |
| 2 | 906.69 |
| 3 | 811.39 |
| 均值 | **874.97** |

`logit1` 在 EP139 附近：

- s1 约 822（EP149）
- s2 869.35
- s3 808.59
- 均值约 **833**

同 EP 下 `noattn` 更差约 **42**。  
`logit1` 到 EP329 才约 874；`noattn` 在 EP139 已经到这一水平。

注意：`no_attn` 同时去掉 SAB 和 PMA。  
如果只想去 SAB、保留 PMA，要用 `--machine_no_sab 1`。

### 21:00 恢复 attention

用户决定：停 `noattn`，恢复 attention 默认。

- `noattn` 停在 EP139。
- 最后读数：`noattn` 874.97 vs `logit1` 约 833.28。
- 代码默认 `--machine_no_attn 0`，不需要改代码。
- `logit1`、PPO 对照、GRPO 10×6 继续。

## §51 2026-10-24 21:03 ST machine + BatchNorm

### 假设

ST machine 没有 BatchNorm；GAT machine 有 `self.bn`。  
差距可能来自归一化。

### 实现

新增：

```text
--machine_bn   default 0
```

`Machine_Actor_SetTransformer`：

- `machine_bn`：作用于 `h_nodes`（逐机器嵌入）。
- `machine_pool_bn`：作用于 `h_pool`。
- 只在 `machine_bn=1` 时构造。
- 只影响 ST machine；GAT 不变。

GAT 对照：

```python
h_m_nodes = self.bn(h_machine_feas.reshape(-1, hidden)).reshape(...)
```

### 验证

- `pytest plcsp_v5/tests -q`：**27 passed**。
- 冒烟：训练模式 / eval 模式都通过；概率和为 1；被屏蔽机器概率 0。
- 远端代码和 `parameters.py` 已更新。

### 队列

- 配置：`plcsp_v5/experiment-queue-machine-bn.json`
- `settrans` + `--machine_bn 1`
- 6×6×2，3 seeds，400 ep，LR 3e-4，buffer 5
- 无 mask、无 AGV、attention 开
- 对照：`logit1`（同协议，无 BN）
- 父进程 **123251**
- log `queue_logs/queue_machine_bn.log`
- 21:03:59 启动，3 个训练全部开始。

### 判读

- BN 低于 `logit1` → BN 是差距的一部分。
- 持平或更差 → BN 不是主因。

### 21:05 第一次启动崩溃

- 3 个 job 在第一次验证（EP9）全部 rc=1。
- 原因：验证时 batch=1，`machine_pool_bn` 在 train 模式下报
  `Expected more than 1 value per channel`。
- 修复：去掉 `machine_pool_bn`；只对 `h_nodes` 做 BN。
  - `h_nodes` 的 batch = B × m ≥ 6，不会触发。
- 验证：27 tests passed；B=1 的 train / eval 冒烟都通过。
- 21:26 重启，3 个训练开始（父进程 126298）。

### 21:50 读数（EP119 / 400）

| seed | EP119 |
|---|---:|
| 1 | 845.09 |
| 2 | 871.56 |
| 3 | 792.90 |
| 均值 | **836.52** |

同 EP 对照 `logit1`：

| EP | BN 均值 | logit1 均值 | 差 |
|---:|---:|---:|---:|
| 39 | 818.85 | 831.71 | −12.86 |
| 49 | 799.63 | 804.47 | −4.84 |
| 119 | 836.52 | — | — |

`logit1` 在 EP139 均值 833.28；BN 在 EP119 是 836.52。  
暂定：BN 早期略好，后期持平。  
还没到 top1 / final。

### 22:13 读数（EP209 / 400）

| seed | EP209 |
|---|---:|
| 1 | 870.74 |
| 2 | 846.85 |
| 3 | 806.39 |
| 均值 | **841.33** |

EP119 836.52 → EP209 841.33。  
没有明确改善。

## §52 2026-10-24 21:10 主模型改为 GRPO

用户决定：实验计划的主模型从 **H-ST-MAPPO** 改为 **H-ST-GRPO**。

### 依据

同协议 6×6（e2e GIN+GAT，200 ep，buffer 8，LR 1e-3，随机 λ，3 seeds）：

| 算法 | seed3 test 均值 | 跨 seed std |
|---|---:|---:|
| GRPO | **676.99** | **14.3** |
| PPO | 790.20 | 147.12 |

- GRPO 低 **113.21**。
- 3/3 seed 方向一致。
- n=3，配对 p≈0.28，不显著。

GRPO 10×6 flat：EP149 **637.32**，论文 674.5。

### 已更新 `docs/v5/experiment-plan.md`

- §0：分层多智能体 GRPO。
- §2：H-ST-GRPO；JA / MA / AA 无 critic；GC 只在 PPO 对照。
- §3.1：GRPO 组；无局部 / 全局 critic。
- §3.3 / §4：每组随机 λ；默认 `--algo grpo`、G=8。
- §6.2：B14 GRPO、B15 PPO 同协议。
- §7：E0 / E4 / E5 / E9 改 GRPO；E4b 标注未跑。
- §10：主模型结果 + PPO 历史线。
- §11：GRPO 符号、显著性、结构、集成风险。

### 未跑

E4b（RS + AGV + GRPO）。  
当前 GRPO 结果是 **flat no-AGV**，不能当完整主模型结论。

## §53 2026-10-24 21:56 GIN job + ST machine 的 BN 消融

### 背景

用户问：§51 的 BN run，job 侧是不是 GIN？

答：不是。  
`settrans` = **ST job** + ST machine。  
§51 测的是 job ST 固定时的 machine BN。

### 新队列

为补 job GIN 的情形，跑 `hybrid_full`：

```text
GIN job + ST machine actor/critic
```

BN 0 / 1 两臂，各 3 seed。

配置：

| 项 | 值 |
|---|---|
| 尺寸 | 6×6×2 |
| 编码器 | `hybrid_full` |
| 臂 | `nobn` / `bn` |
| seed | 1 / 2 / 3 |
| ep | 400 |
| LR | 3e-4 |
| buffer | 5 |
| resample | 5 |
| mask | 无 |
| AGV | 无 |

- 队列：`plcsp_v5/experiment-queue-hybrid-full-bn.json`
- 父进程 **129473**
- log `queue_logs/queue_hybrid_full_bn.log`
- 6 个训练已启动（21:56:37）。
- 每臂 eval 走 `EVAL_ENCODER=hybrid_full` + 对应 `--machine_bn`。

### 当前并发

- `settrans` BN 队列：126298，3 个，EP119。
- `hybrid_full` BN 对照组：129473，6 个。
- 共 9 个 6×6；GPU 99%，显存 6.35 GB。

### 判读

- 同 job GIN、同协议：BN 1 低于 BN 0 → BN 有用。
- 对 §51（job ST）可交叉看：BN 的作用是否依赖 job 编码器。

### 22:13 读数（EP19 / 400）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| `bn` | 762.60 | 1034.03 | 868.43 | **888.35** |
| `nobn` | 805.70 | 797.52 | 820.24 | **807.82** |

BN 更差，但 s2 是离群点。  
s1 的 BN 更好（762.60 vs 805.70）。  
EP19 太早，不能下结论。

## §54 2026-10-24 22:19 AGV 空驶修复 + AGV PPO/GRPO 重跑

### 问题

训练 / 验证的 AGV 计时只算：

```text
wait + A→B
```

不算 AGV 当前位置 → A 的空驶。

但：

- `AgvState.assign` 会算 `travel(location, pickup)`。
- `agv_features` 也把 `travel(location, pickup)` 喂给策略。
- 训练环境不调用 `assign`。

判定：**自相矛盾，是错。**

### 修复（按选项 A）

新增 `plcsp_v5/agv/state.py: transport_timing(...)`：

- 同车间：

```text
deadhead(location → pickup) + loaded(A → B)
```

- 跨车间：

```text
源空驶 + 源段
+ 固定延迟
+ 目标空驶(到边界) + 目标段
```

- 边界 = **目标车间第一台机器**。
- `wait` 含空驶；`loaded` 只含载货；TEC 口径不变。

接线：

- `trainer/parallel_env.py`：训练 step。
- `trainer/validate.py`：两个验证路径（主 + ESA）。
- 远端三个文件已同步。

测试：

- 新增 4 条 AGV timing 测试。
- `pytest plcsp_v5/tests -q`：**31 passed**。

### AGV 重跑队列

配置：`plcsp_v5/experiment-queue-agv-fix.json`

| 项 | 值 |
|---|---|
| 尺寸 | 6×6×2 |
| 编码器 | `settrans` |
| 算法 | PPO / GRPO |
| 分层 | RS on |
| AGV | actor |
| seed | 1 / 2 / 3 |
| ep | 200 |
| buffer / resample | 8 / 8 |
| LR | 1e-3 |

- 13 job：1 个 10-ep smoke + 6 训练 + 6 eval。
- 父进程 **133184**
- `max_parallel=3`
- smoke 22:19 启动（GRPO + hier + AGV）。
- 全量 run 用 `after` 等 smoke；smoke 失败则全量 blocked。

### 注意

- 旧 AGV 读数（E2 / E3 / E4a / E4b）**全部作废**。
- no-AGV 读数不受影响。
- smoke 已通过；AGV PPO seed1/s2/s3 已启动。

### 22:57 读数（EP39 / 200）

| seed | obj |
|---|---:|
| 1 | 905.41 |
| 2 | 918.22 |
| 3 | 900.80 |
| 均值 | **908.14** |

比旧 E4b 的 743.95 高。  
但旧 AGV 计时不含空驶，**不可比**。  
AGV GRPO 全量还没启动（PPO 占满 3 槽）。

## §55 2026-10-24 22:46 GIN job + ST machine：机内 attention

### 背景

用户提案：

```text
分开投影
→ 机内 attention 融合
→ 机间 attention
```

目的：补上 GAT 的逐机器交互，同时保留 ST 的跨机器部分。

### 实现

新增：

```text
--machine_intra_attn   default 0
```

`Machine_Actor_SetTransformer`：

- m1 / m2 / m3 三个独立 Linear（`bias=False`）。
- 堆成 3 个 token：`(B*m, 3, H)`。
- 1 层 SAB（2 头）+ PMA。
- 融合后 reshape 回 `(B, m, H)`。
- 再接现有机器间 SAB + PMA。

限制：

- 只改 machine actor。
- global critic 的 machine encoder 未改。

验证：

- `pytest plcsp_v5/tests -q`：**31 passed**。
- intra smoke：前向、mask、概率和都通过。

### 队列

配置：`plcsp_v5/experiment-queue-hybrid-intra.json`

| 项 | 值 |
|---|---|
| 编码器 | `hybrid_full` = GIN job + ST machine |
| 臂 | `intra0` / `intra1` |
| seed | 1 / 2 / 3 |
| ep | 400 |
| LR | 3e-4 |
| buffer | 5 |
| AGV | 无 |

- 12 job：6 训练 + 6 eval。
- 父进程 **136069**
- `max_parallel=3`
- 22:46 启动；`intra0` 三个先跑。
- `intra1` 等空位。

### 当前并发

- AGV fix 队列：133184，PPO 3 个在跑。
- hybrid_full intra：136069，intra0 3 个在跑。
- GPU 30–99%，显存约 5 GB。

### 22:57 读数（EP9 / 400）

| seed | obj |
|---|---:|
| 1 | 815.10 |
| 2 | 921.97 |
| 3 | 827.43 |
| 均值 | **854.83** |

`intra1` 还没启动，等 `intra0` 完成。  
现在不能比较机内 attention。

## §56 2026-10-24 23:04 6 并行重启 + AGV 规则臂

### 用户要求

- AGV actor 必须和**规则 AGV** 对照。
- 两个队列都要提速。

### AGV 队列

改为 4 臂：

| 算法 | AGV | 臂 |
|---|---|---|
| PPO | actor | `agvactor_ppo` |
| PPO | 规则 | `agvrule_ppo` |
| GRPO | actor | `agvactor_grpo` |
| GRPO | 规则 | `agvrule_grpo` |

- 每臂 3 seeds。
- 12 训练 + 12 eval + 1 smoke = 25 job。
- `max_parallel=6`。
- 23:04 重启。
- 第一波：PPO actor 3 + PPO rule 3。
- GRPO 两臂等空位。

### 23:17 读数（EP9 / 200）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| PPO actor | 763.88 | 942.09 | 946.71 | **884.23** |
| PPO rule | 966.98 | 845.94 | 949.20 | **920.71** |

actor 均值低 36.48。  
但 EP9 是单点，不能下结论。

`hybrid_full` intra0 / intra1 还没出 EP9 读数。  
两个都在跑。

### 23:24 读数（EP9 / 400）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| intra0 | 815.10 | 921.97 | 827.43 | **854.83** |
| intra1 | 785.29 | 688.25 | 721.76 | **731.77** |

intra1 低 **123.06**。  
三个 seed 都 lower。  
但这是 EP9 单点，不能下结论。  
等 EP50–100 和 top1。

### 23:39 critic 一起改

用户指出：换结构必须 actor + critic 一起换。

已改：

- `Global_Critic_SetTransformer`
- `Global_Critic_Hybrid`

两处都加：

```text
m1/m2/m3 分开投影
→ 机内 attention
→ 机间 SAB + PMA
```

- 只对 ST machine 路径。
- GAT critic 不变。
- 31 tests passed；critic intra smoke passed。

旧 `intra1` 是 actor-only；EP9 的 731.77 **作废**。  
23:40 重启，`intra0` / `intra1` 各 3 seed 重新跑。

清理：

- 之前 `pkill` 用了队列 ID（`HHintra0_s1`），没匹配 run_tag，
  导致多代 intra 进程残留。
- 已按 run_tag 全部清掉。
- 当前父进程 **144710**，6 个训练。

### 23:48 AGV PPO 读数（EP39 / 200）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| actor | 905.41 | 918.22 | 900.80 | **908.14** |
| 规则 | 911.24 | 771.77 | 898.22 | **860.41** |

规则低 **47.73**。  
但 EP9 时 actor 是 884.23 vs 规则 920.71，actor 领先。  
两条曲线交叉了。

- actor 的 s1：763.88 → 982.52 → 1015.03 → 905.41。
- 规则的 s1：966.98 → 935.43 → 919.44 → 911.24。

规则更稳，actor 更跳。  
还不够下结论。

GRPO 两臂还没启动：PPO 6 个占满槽。

### 23:55 intra 读数（EP9 / 400，critic 一起改后）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| intra0 | 815.10 | 921.97 | 待出 | — |
| intra1 | 1025.76 | 987.91 | 857.30 | **957.0** |

- `intra1` 现在比 actor-only 的 731.77 差很多。
  两者不可比：critic 已改。
- 现在 `intra1` 也差于 `intra0` 的 s1/s2。
- EP9 单点；critic 新参数可能拖慢早期学习。
- 等 EP50+ 和 top1。

### 00:12 读数（EP19 / 400）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| intra0 | 805.70 | 797.52 | 待出 | — |
| intra1 | 972.37 | 804.72 | 889.26 | **888.78** |

- `intra1` 仍差于 `intra0` 的 s1/s2（约 801.6）。
- 差约 **87**。
- `intra1` s1 曲线：1025.76 → 972.37。
- `intra0` s1 曲线：815.10 → 805.70。

还早，但方向不利。

## §57 2026-10-25 00:19 AGV PPO 读数（EP69 / EP79）

### 读数

| 臂 | 最新 EP | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|---:|
| actor | 69 | 825.71 | 870.59 | 903.03 | **866.45** |
| 规则 | 79 | 835.76 | 830.70 | 860.80 | **842.42** |

两个臂速度不同。actor 带 AGV actor，约 65 s/EP；规则约 57 s/EP。  
故按同 EP 比较。

### 同 EP69 比较

| EP | actor 均值 | 规则均值 | 差（actor − 规则） |
|---:|---:|---:|---:|
| 9 | 884.23 | 920.71 | −36.48 |
| 19 | 957.61 | 868.05 | +89.56 |
| 29 | 935.17 | 855.93 | +79.24 |
| 39 | 908.14 | 860.41 | +47.73 |
| 49 | 866.03 | 851.28 | +14.75 |
| 59 | 857.13 | 822.85 | +34.28 |
| 69 | 866.45 | 825.41 | +41.04 |

规则在 EP19 之后一直领先。差距先收窄，再扩大。  
没有收窄到 0 的趋势。

### 分 seed（EP69）

| seed | actor | 规则 | 差 |
|---|---:|---:|---:|
| 1 | 825.71 | 822.17 | +3.54 |
| 2 | 870.59 | 781.12 | +89.47 |
| 3 | 903.03 | 872.93 | +30.10 |

规则赢 2/3。seed1 基本持平。  
seed2 是主要差距来源。

### top1（验收集最小值，近似）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| actor | 763.88 (EP9) | 858.66 (EP59) | 860.68 (EP29) | **827.74** |
| 规则 | 809.98 (EP59) | 765.12 (EP49) | 829.55 (EP29) | **801.55** |

规则低 **26.19**。  
actor 的 seed1 最优仍在 EP9。

说明：top1 取日志 `obj_100ins_mean` 的最小值，是近似值。  
真 top1 由验证逻辑的选模规则决定。

### ckpt

已确认保存。每 10 EP 一次，另有 `final`：

- `trained_model/settrans_hier_agvactor_ppo_s1_fixagv_remote/PPO_agv_actor_J6M6E2_EP70_.pth`
- `trained_model/settrans_hier_agvrule_ppo_s2_fixagv_remote/PPO_agv_actor_J6M6E2_EP80_.pth`

### 当前状态

- 两个队列共用 GPU：AGV 6 个 + intra 6 个。
- GPU 85%，显存 12.5 / 24.6 GB。
- PPO 两臂约 EP70 / 200。
- GRPO 两臂仍未启动。
- 暂不下结论。

## §58 2026-10-25 00:46 machine ST 机内 attention 读数（EP49 / 400）

### 读数

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| intra0 | 825.03 | 774.21 | 862.96 | **820.73** |
| intra1 | 692.98 | 660.73 | 815.22 | **722.98** |

`intra1` 低 **97.75**。  
3/3 seed 都是 `intra1` 更好。

### 反转

EP19 时 `intra1` 差约 87。EP49 时它反超。  
交叉发生在 EP29 到 EP39 之间。

| EP | intra0 | intra1 | 差（intra1 − intra0） |
|---:|---:|---:|---:|
| 9 | 854.83 | 956.99 | +102.16 |
| 19 | 807.82 | 888.78 | +80.96 |
| 29 | 790.55 | 831.88 | +41.34 |
| 39 | 847.62 | 742.76 | −104.86 |
| 49 | 820.73 | 722.98 | −97.75 |

### 分 seed（EP49）

| seed | intra0 | intra1 | 差 |
|---|---:|---:|---:|
| 1 | 825.03 | 692.98 | −132.04 |
| 2 | 774.21 | 660.73 | −113.47 |
| 3 | 862.96 | 815.22 | −47.74 |

### 曲线

`intra1` 的 s1、s2 单调下降：

- s1：1025.76 → 972.37 → 839.73 → 722.00 → 692.98
- s2：987.91 → 804.72 → 754.64 → 677.51 → 660.73

`intra0` 的 s1 在 EP29 触底后回升：

- s1：815.10 → 805.70 → 761.73 → 809.49 → 825.03

`intra1` 的 s3 仍弱：857.30 → 889.26 → 901.28 → 828.77 → 815.22。

### top1（验收集最小值，近似）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| intra0 | 761.73 (EP29) | 774.21 (EP49) | 820.24 (EP19) | **785.39** |
| intra1 | 692.98 (EP49) | 660.73 (EP49) | 815.22 (EP49) | **722.98** |

`intra1` 低 **62.41**。

### 说明

- 配置：PPO + hier + `hybrid_full`，LR 3e-4，buffer 5，6×6×2，无 AGV。
- actor 与 critic 都用了机内 attention。
- 只到 EP49 / 400，还不能定论。
- 不可与 E2E GRPO 的 676.99 直接比：算法、学习率、buffer、编码器都不同。

## §59 2026-10-25 01:05 machine ST 默认改为 intra1

### 用户决定

machine ST 直接采用 `intra1`：

- m1 / m2 / m3 分开投影
- 机器内 attention（3 token）
- 机器间 attention

### 代码

- `parameters.py`：`--machine_intra_attn` 默认 `0 → 1`。
- `model/actor_critic_st.py`：三处 fallback `0 → 1`。
  - `Machine_Actor_SetTransformer`
  - `Global_Critic_SetTransformer`
  - `Global_Critic_Hybrid`
- `--machine_intra_attn 0` 保留，作为消融开关。

### 验证

- `pytest plcsp_v5/tests`：**31 passed**。
- 默认构造检查：
  - parser default = 1
  - actor / critic ST / critic hybrid 都有 `m1_proj`，无 `input_proj`
  - 显式 0 时相反
- `ruff`：`actor_critic_st.py` 干净；
  `parameters.py` 有 10 个既存 F401，非本次引入。

### 在飞实验保护

改动前，AGV 队列的 18 个待跑 job 没有钉 `machine_intra_attn`。  
新默认会让它们与已跑的 PPO 臂架构不同，且 eval 会加载失败。

处理：

1. 只杀 AGV 队列父进程 137815。  
   6 个 PPO 子进程在独立 session，继续训练。
2. 新建 `experiment-queue-agv-fix-rest.json`。  
   18 个剩余 job 全部显式 `--machine_intra_attn 0`。
3. 启动一个 watcher。  
   等 6 个 PPO 训练退出后，再启动 rest 队列。
4. local + remote 的 `experiment-queue-agv-fix.json` 也补上显式 0。
5. intra 队列的 12 个 job 本来就逐条显式传 0/1，不受影响。

### 消融登记

已写入 `docs/v5/experiment-plan.md` §E5b：

- `intra0`：拼接 + MLP 投影 + 机器间 attention
- `intra1`：分开投影 + 机器内 + 机器间 attention

⚠️ 当前消融口径是 PPO + hier + `hybrid_full`，LR 3e-4，buffer 5。  
不是主模型 GRPO 口径。  
进论文前要重跑，或标注口径差异。

## §60 2026-10-25 01:09 AGV PPO 读数（EP119 / EP149）

### 读数

| 臂 | 最新 EP | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|---:|
| actor | 119 | 823.01 | 859.71 | 817.47 | **833.40** |
| 规则 | 149 | 808.42 | 853.50 | 900.52 | **854.15** |

规则多跑了 30 EP。按同 EP 比较。

### 同 EP 比较

| EP | actor | 规则 | 差（actor − 规则） |
|---:|---:|---:|---:|
| 69 | 866.45 | 825.41 | +41.04 |
| 79 | 841.41 | 842.42 | +1.01 |
| 99 | 831.27 | 834.61 | −3.34 |
| 109 | 830.23 | 846.23 | −16.00 |
| 119 | 833.40 | 866.17 | −32.77 |

actor 在 EP79 到 EP99 之间再次反超。  
之后差距扩大。

### 分 seed（EP119）

| seed | actor | 规则 | 差 |
|---|---:|---:|---:|
| 1 | 823.01 | 827.57 | −4.55 |
| 2 | 859.71 | 857.01 | +2.70 |
| 3 | 817.47 | 913.94 | −96.47 |

actor 赢 2/3。  
主要差距来自 seed3：actor 在降，规则在升。

### top1（验收集最小值，近似）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| actor | 763.88 (EP9) | 832.09 (EP89) | 817.47 (EP119) | **804.48** |
| 规则 | 801.53 (EP89) | 765.12 (EP49) | 829.55 (EP29) | **798.73** |

规则仍低 **5.75**。  
`actor` 的 s1 最优仍在 EP9。  
`规则` 的 s2 最优在 EP49。  
两边都出现过早触底。

### 状态

- 6 个 AGV PPO 训练在跑。
- 规则约 EP149 / 200；actor 约 EP119 / 200。
- watcher 存活，等 PPO 退出后启动 rest 队列。
- 6 个 intra 训练同时在跑。
- GPU 99%，显存 12.5 / 24.6 GB。
- 两条曲线仍在交叉，不下结论。

## §61 2026-10-25 01:15 GRPO 10×6 完成 + 计划同步

### GRPO 10×6 flat（e2e）

| 项 | 读数 |
|---|---:|
| 验证 EP199 | **609.23** |
| seed3 测试均值 | **613.98** |
| 测试实例 std | 81.79 |
| 论文 Table V 10×6×2 P-G | 674.5 ± 88.9 |

比论文低 **60.52**。  
只有 1 个训练 seed，不能当最终结论。

口径：

- `e2e`（GIN job + GAT machine）
- flat：无 RS、无 AGV
- 200 ep，随机 λ
- seed3 100 实例，贪心

### 计划同步

已更新 `docs/v5/experiment-plan.md`：

- 头部：machine ST 默认 intra1；AGV 批次钉 intra0。
- §3.1：machine token 组成与 intra1 结构。
- §3.4：AGV 规则的精确定义；四臂钉 intra0。
- §4：新默认 `--machine_intra_attn 1`；旧 ckpt 要显式 0。
- §6.2：B8–B12 状态修正；新增 B16 / B17 / B18。
- E5：机内 attention 记为 encoder 家族的一部分。
- E5b：补 EP49 读数。
- E7：10×10 改为 10×6。
- §8：10×6 预算；10×10 停用。
- §10：状态改到 2026-10-25 01:15；旧 AGV 读数标作废。
- §11：新增 5 条口径风险。
- §13：当前队列与 `max_parallel=6`。

### 未决

- E5b 需在 GRPO 口径重跑。
- 10×6 需补 seed。
- AGV 四臂未到 200 ep。

## §62 2026-10-25 01:25 E10 暂缓

用户决定：λ 扫描 + Pareto + NSGA-II **先不做**。

已改 `docs/v5/experiment-plan.md`：

- RQ6 标为暂不做。
- B13 NSGA-II 标为暂不做；保留理由与来源。
- E10 标题加「暂不做」；内容保留，供以后恢复。
- §5 的 λ 网格行标为暂不做。
- §8 的 λ 网格预算行标为暂不做。
- §10 状态行改为暂不做。
- §12 主表范围改为 E0–E9。

`plcsp_v3/baselines/nsga2.py` 保留。  
它仍面向 v3 三目标，未适配 v5。

### B3 / B4 是同一篇

查 E2E-MAPPO 论文的参考文献：

- [23] = K. Lei, P. Guo, W. Zhao, Y. Wang, L. Qian, X. Meng, L. Tang,
  “A multi-action deep reinforcement learning framework for flexible
  job-shop scheduling problem,”
  *Expert Systems with Applications*, vol. 205, Art. no. 117796, 2022.
  DOI: 10.1016/j.eswa.2022.117796.
- 原计划 B3 写“ESA（ESWA_MPPO）”，B4 写“DRL [23]”。
  两者是同一篇。
- `tester/ESWA_MPPO/` 有六个尺寸的预训练权重。
  这些权重由 E2E-MAPPO 作者改写为四目标（critic 输出 4 个值）。

计划已改：

- B3 换成完整题录。
- B4 标为与 B3 同一篇。
- 只允许写 “adapted”，不得写 “复现”。

## §63 2026-10-25 01:40 新增 B19 RGA

用户决定：加 B19（RGA），2SGA 先不加。

### 为什么补

原计划缺元启发式基线：

- B1 PDR 是派工规则，不是元启发式。
- B2 MIP 是精确解。
- B3 是 DRL。
- B13 NSGA-II 已暂缓，且是多目标。

### 题录

G. Zhang, L. Gao, Y. Shi, “An effective genetic algorithm for the
flexible job-shop scheduling problem,”
*Expert Systems with Applications*, vol. 38, no. 4, pp. 3563–3573, 2011.

来源：E2E-MAPPO 论文参考文献 [43]。

### 计划改动

- §6.1 新增 B19；标明单目标 RGA。
- §6.3 新增 RGA 口径：no-AGV、seed3 100 实例、固定权重；
  必须报种群、代数、评价次数、墙钟、种子。
- §6.4 新增「RGA 未实现」。
- §10 新增外部基线状态表。
- §11 新增 RGA 口径风险。

### 状态

- 仓库里没有 RGA 代码；未实现。
- 染色体、解码器、种群、代数都未定。
- 2SGA 暂不加。

### 预期风险

论文里 2SGA 质量可以高于其 DRL 方法。  
RGA 与论文方法接近。  
我们的模型可能质量不占优。  
主表必须同时报质量和在线时间。

## §64 2026-10-25 01:46 AGV PPO 读数（EP159 / EP199）

### 读数

| 臂 | 最新 EP | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|---:|
| actor | 169 / 159 / 169 | 833.93 | 867.70 | 833.69 | **845.11** |
| 规则 | 199 / 199 / 189 | 820.87 | 844.98 | 884.85 | **850.23** |

两臂进度不同。按同 EP159 比较。

### 同 EP159

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| actor | 830.58 | 867.70 | 822.05 | **840.11** |
| 规则 | 815.01 | 835.06 | 886.51 | **845.52** |

actor 低 **5.41**。

### 差距在缩小

| EP | 差（actor − 规则） |
|---:|---:|
| 119 | −32.77 |
| 129 | −21.60 |
| 139 | −11.49 |
| 149 | −11.30 |
| 159 | −5.41 |

actor 的领先在持续收窄。  
可能有再次交叉的风险。

### 分 seed（EP159）

| seed | actor | 规则 | 差 |
|---|---:|---:|---:|
| 1 | 830.58 | 815.01 | +15.58 |
| 2 | 867.70 | 835.06 | +32.64 |
| 3 | 822.05 | 886.51 | −64.46 |

只有 seed3 支持 actor。  
seed1 和 seed2 都是规则更好。

### top1（验收集最小值，近似）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| actor | 763.88 (EP9) | 832.09 (EP89) | 817.47 (EP119) | **804.48** |
| 规则 | 801.53 (EP89) | 765.12 (EP49) | 829.55 (EP29) | **798.73** |

规则仍低 **5.75**。  
两边 top1 都没有再刷新。

### 状态

- 规则 s1 / s2 已到 EP199；s3 到 EP189。
- actor 三个 seed 到 EP159–169。
- 4 个 AGV 训练仍存活。
- watcher 存活；rest 队列未启动。
- GPU 43%，显存 10.2 / 24.6 GB。

规则正在追平。  
actor 的优势不稳。  
等 200 ep 与 seed3 测试。

## §65 2026-10-25 01:50 为什么 AGV actor 赢不多

### EP159 的三 seed 均值分解

| 分量 | actor | 规则 | 差 | 权重 | 对 Obj 的贡献差 |
|---|---:|---:|---:|---:|---:|
| mk | 563.87 | 561.63 | +2.25 | 0.4 | +0.90 |
| pt | 461.27 | 462.14 | −0.87 | 0.4 | −0.35 |
| idleT | 1001.21 | 1013.81 | −12.61 | 0.4 | −5.04 |
| transT | 147.86 | 152.47 | −4.61 | 0.2 | −0.92 |
| **合计** | 840.11 | 845.52 | −5.41 | — | **−5.41** |

Obj = `0.4·mk + 0.4·(pt + idleT) + 0.2·transT`。

### 结论

actor 只在 idleT 和 transT 上赢。  
mk 反而略差。  
净收益约 5 点，占 Obj 约 **0.6%**。

直接原因：

- transT 只占 Obj 的 **3.5%**。
- transT 改善 3%，只换来 0.92 点。
- idleT 改善 1.2%，换来 5.04 点。
- mk 变差 2.24，抵消 0.90 点。

### 结构原因

1. 规则已接近最优。  
   最早空闲直接最小化取货等待。
2. actor 的额外信息只有空驶距离。  
   用「更近但更晚空闲」换「等待」，会伤 mk。
3. AGV 决策次数少。  
   只有换机器时才产生运输决策。
4. 信用分配弱。  
   终端 4 目标回报，team advantage 共享。
5. 统计功效低。  
   3 seeds，差异 0.6%，种子间波动更大。
6. 规则也在受益于 job/machine 策略的进步。  
   EP119 → EP159，actor 领先从 33 收到 5。

### 可选诊断

- 加「空驶感知规则」：在 `free_time` 差 δ 内选最近的 AGV。  
  如果它也赢不了，说明上限低。
- 加 myopic oracle：每步选实际最早到达取货点的 AGV。  
  用它的收益当上限。
- 统计 actor 与规则选择不同的比例。
- 降低 AGV 数量，制造稀缺，看 actor 是否更有价值。
- 目标车间 AGV 目前固定用规则；actor 只控源车间。
- 旧 PPO EP39 的读数作废（被重启覆盖）。

### hybrid_full 队列

- `max_parallel=6`。
- `intra0` 3 + `intra1` 3 同时在跑。
- 旧 `intra0` EP9 读数作废。

### 当前

- AGV 队列父进程 **137815**。
- hybrid 队列父进程 **137821**。
- 共 12 个 6×6 训练。
- GPU 99%，显存 8.5 GB。

### 注意

- AGV actor vs 规则现在同协议、同 seed。
- 旧 AGV 数字仍不可用（空驶修复前）。

## §66 2026-10-25 02:18 两端 AGV actor + 新批次

### 用户决定

- 所有车间的 AGV 都由 actor 选，不只源车间。
- 停掉远程所有未完成实验。
- 改完后按 3 个实例 × 3 个 seed 批量跑。
- 明早看结果。

### 代码改动

| 文件 | 改动 |
|---|---|
| `parameters.py` | 新增 `--agv_dst_actor`，默认 1 |
| `trainer/parallel_env.py` | 新增 `get_agv_dst_decision_batch`；`DGFJSPEnv_paral_step` 接收 `agv_dst_actions` |
| `Run.py` | rollout 里源端、目标端各一次 actor 调用；缓存两组动作 |
| `trainer/replaybuffer.py` | 新增目标端特征 / 动作 / logprob / value 字段 |
| `algorithm/ppo_algorithm.py` | PPO 与 GRPO 都加目标端损失；两端共享同一个 actor |
| `trainer/validate.py` | 目标端也用 `va_select_agv`；时刻用预计边界到达 |
| `tests/conftest.py` | 把 `plcsp_v5/` 加进测试 import 路径 |
| `tests/test_agv.py` | 新增 2 个目标端决策测试 |

设计：

- 源端在派发时刻决策。
- 目标端在派发时刻决策，用预计边界到达时间。
- 同车间运输只决定源端；目标槽无效。
- 规则臂两端都用最早空闲。
- 目标端与源端共享同一个 AGV actor。

### 验证

- 本地 `pytest plcsp_v5/tests`：**33 passed**。
- 远端 PPO 两端 smoke：rc=0。
- 远端 GRPO 两端 smoke：rc=0。
- smoke 参数：6×6×2，`--agv_dst_actor 1`，`--machine_intra_attn 1`。

### 远端路径修复

上传本地 `Run.py` / `parameters.py` 时覆盖了远端路径。  
已把 `D:/research/DeepReinforcementLearningScheduling/plcsp_v5`
替换为 `/root/plcsp_v5`。

以后上传这两个文件必须重做替换。  
`validate.py` 没有本地路径，已检查。

### 停止旧实验（02:12）

旧 AGV PPO（source-only）已到 EP199：

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| actor | 861.09 | 862.48 | 852.18 | **858.58** |
| 规则 | 820.87 | 844.98 | 873.15 | **846.33** |

规则在最终验证低 **12.25**。  
近似 top1：actor 804.48，规则 798.73。规则仍低 **5.75**。  
这批是 source-only pilot，不再使用。

intra 消融停止时：

- intra0（EP129）：821.02 / 807.78 / 821.91，均值 **816.90**
- intra1（s1 EP119、s2/s3 EP129）：654.99 / 672.28 / 796.95，均值 **708.07**

EP 不完全对齐，但 intra1 明显更好。  
已停，读数只作 pilot。

### 新批次

配置：`plcsp_v5/experiment-queue-agv-both.json`

- 3 尺寸：6×6×2 / 10×6×2 / 10×10×2
- 2 臂：actor（两端）/ rule（两端）
- 3 seeds
- GRPO + hier + settrans + intra1 + n_agv 6
- 200 ep，buffer 8，resample 8，LR 1e-3
- 每臂训练后跑 seed3 100 实例评测
- 共 36 job（18 训练 + 18 评测）
- `max_parallel=6`

优先级：

1. 6×6 训练
2. 6×6 评测
3. 10×6 训练
4. 10×6 评测
5. 10×10 训练
6. 10×10 评测

02:18 启动。  
前 6 个 6×6 训练已开始。  
GPU 69%，显存 3.1 / 24.6 GB。

02:21 核对：

- 6 个训练存活。
- 队列父进程存活。
- 各 run 到 episode 7 附近。
- 无 Traceback。
- GPU 21%，显存 8.0 / 24.6 GB。
- 显存上升符合 6 个并行 run。

02:27 首次评测通过：

- 6×6 的 EP9 评测已出读数。
- 无 Traceback。
- 6 个训练仍存活。
- GPU 98%，显存 8.0 / 24.6 GB。
- 目标端 actor 的验证路径已跑通。

样例读数（EP9）：

- 753.95（mk 608.82 / pt 533.44 / transT 82.60 / idleT 701.32）
- 842.77（mk 662.72 / pt 519.09 / transT 76.69 / idleT 886.77）
- 855.47（mk 563.46 / pt 471.01 / transT 204.06 / idleT 1002.18）

单点，不作结论。

### 明早看

- `queue_logs/queue_agv_both.log`
- `queue_logs/status.json`
- `queue_logs/20261025_*.log`
- 各 run 的 `trained_model/agvboth_*`

预计：

- 6×6 训练约 3 h。
- 10×6 训练约 5–6 h。
- 10×10 可能跑不完。

### 风险

- 10×10 训练与评测可能未完成。
- 目标端用预计边界时间，不是实际到达时间。
- 两端结果不能与 source-only pilot 混。  
- 这批全部是 GRPO；没有 PPO 对照。

## §67 2026-10-25 07:13 6×6 两端结果 + 10×6 进度

### 6×6×2 完成

6 个训练全部完成，6 个 seed3 评测全部完成。  
无 Traceback。

验证（EP199）：

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| actor | 585.36 | 591.40 | 590.64 | **589.13** |
| rule | 619.78 | 629.38 | 594.08 | **614.42** |

actor 低 **25.28**。

seed3 测试（top1，100 实例）：

| 臂 | s1 | s2 | s3 | 均值 | 3-seed std |
|---|---:|---:|---:|---:|---:|
| actor | 580.18 | 592.43 | 575.17 | **582.59** | ±8.9 |
| rule | 596.81 | 612.00 | 586.24 | **598.35** | ±13.0 |

actor 低 **15.76**。  
**3/3 seed actor 赢。**

分 seed 差（rule − actor）：

- s1：+16.63
- s2：+19.57
- s3：+11.07

### 口径

- GRPO + hier + settrans + intra1。
- 两端 AGV 都由 actor 选；rule 臂两端规则。
- `n_agv 6`，`cross_delay 5`，`cross_leg_ratio 0.5`。
- 200 ep。
- 测试是 seed3 100 实例，加载 seed1 验证选出的 top1。

⚠️ 这个 582.59 不能与旧的 676.99 直接比。  
旧的是 e2e、flat、no-AGV、intra0；新的是 settrans、hier、AGV、intra1。  
也不能与论文 732.5 直接并列（论文无 AGV，且编码器与训练预算不同）。

### 10×6×2 进度（07:13）

| run | 最新 EP |
|---|---:|
| actor s1 | 104 |
| actor s2 | 96 |
| actor s3 | 70 |
| rule s1 | 128 |
| rule s2 | 90 |
| rule s3 | 88 |

6 个都在跑，无 Traceback。  
GPU 99%，显存 15.6 / 24.6 GB。

预计：

- 10×6 训练约 08:40–10:00 完成。
- 之后跑 10×6 评测。
- 10×10 预计 11:00 后才开始，可能跑不完。

### 结论

6×6 上，两端 actor 稳定赢规则，3/3 seed。  
这是当前最强的一条正结果。

但还只是 1 个尺寸、1 个批次。  
10×6 / 10×10 未完成。

## §68 2026-10-25 07:17 全计划矩阵启动

### 用户澄清

用户要的是**整个实验计划**，训练 3 个 seed。  
不是只跑 AGV actor vs rule。  
10×10 暂不跑。

### 发现口径问题

之前的 AGV 批次没有传：

```
--st_token_context 1 --st_task_features 1
```

parser 默认是 0。  
所以 6×6 的 582.59 和 10×6 的在跑批次都是 **12 维**任务特征。  
计划的默认是 **33 维**。

处理：

- 6×6 的 582.59 保留为 **12 维 pilot**，不并入 33 维矩阵。
- 10×6 的 12 维批次停止。
- `parameters.py` 两个默认值改为 1，与 CLAUDE.md §4 一致。
- 远端路径重新替换为 `/root/plcsp_v5`。

### 新矩阵

配置：`experiment-queue-plan-matrix.json`

6×6×2（9 个配置 × 3 seeds = 27）：

| 配置 | encoder | RS | AGV |
|---|---|---|---|
| E0 | settrans | off | 无 |
| E1 | settrans | on | 无 |
| E2 | settrans | off | 规则 |
| E3 | settrans | off | actor |
| E4a | settrans | on | 规则 |
| E4b | settrans | on | actor |
| E5_e2e | e2e | off | 无 |
| E5_jobst | job_st | off | 无 |
| E5b_intra0 | settrans，intra0 | off | 无 |

10×6×2（4 个配置 × 3 seeds = 12）：

- E0 / E1 / E4a / E4b。

10×10：不跑。

每个训练后接 seed3 评测。  
共 **78 job**：39 训练 + 39 评测。  
`max_parallel=6`。

公共口径：

- GRPO
- 200 ep，buffer 8，resample 8
- LR 1e-3
- 33 维任务特征
- `--machine_intra_attn 1`，E5b 为 0
- seed 1/2/3

### 启动

07:17 启动。  
前 6 个：E0 / E1 的 3 seeds。  
GPU 39%，显存 3.0 / 24.6 GB。

### 预计

- 6×6：27 个训练，约 4–5 波，13–15 h。
- 10×6：12 个训练，约 2 波，12 h。
- 总计约 25–27 h，加评测。
- 10×10 不在本批。

### 口径

- 新矩阵是 33 维。
- 旧的 12 维 AGV pilot（582.59）不能混入。
- 旧 flat e2e 676.99 是 12 维，也不能与新矩阵直接比。
- E5_e2e 的 ST 特征开关可能被忽略；这是 encoder 家族差异的一部分。

## §69 2026-10-25 07:20 统一 200 ep

用户决定：所有实验统一 **200 ep**。  
不跑 1000。

当前矩阵已全部是 200 ep，无需重启。

计划已改：

- §4：训练预算改为「所有实验统一 200 ep」。
- E7：10×6 也是 200 ep；不再上 1000。
- §8：删掉 1000 ep 行；规则改为统一 200。
- §9：停止规则不再提 1000。
- E5b：旧 PPO 批次 400 ep 标为「已停」；
  新矩阵 `E5b_intra0` 用 200 ep，对 `E0`。

旧的 400 ep / 1000 ep 读数只作历史。  
不得与新矩阵并列。

## §70 2026-10-25 07:26 先跑完 200 ep 全矩阵

用户决定：先跑 200 ep 的全矩阵。  
全部跑完再看。  
现在不补 400 ep，也不重启。

当前队列继续：

- `experiment-queue-plan-matrix.json`
- 78 job
- 39 训练 + 39 评测
- 全部 200 ep
- 33 维
- 3 seeds
- 6×6 九配置 + 10×6 四配置

跑完后评估：

- 各配置的 top1 / seed3 测试。
- 哪些格明显未收敛。
- 再决定是否补 400 ep。

计划 §4 已记录此决定。

## §71 2026-10-25 09:31 E0 / E1 完成：RS 反而更差

### 6×6×2 E0 / E1（33 维）

验证（EP199）：

| 配置 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| E0（RS off） | 582.21 | 579.44 | 586.56 | **582.74** |
| E1（RS on） | 576.84 | 595.02 | 594.10 | **588.65** |

E1 高 **5.92**。

seed3 测试（top1，100 实例）：

| 配置 | s1 | s2 | s3 | 均值 | 3-seed std |
|---|---:|---:|---:|---:|---:|
| E0（RS off） | 550.71 | 553.90 | 573.33 | **559.31** | ±12.2 |
| E1（RS on） | 568.59 | 561.36 | 582.22 | **570.72** | ±10.6 |

E1 高 **11.41**。  
**3/3 seed 都是 E0 更好。**

分 seed 差（E1 − E0）：

- s1：+17.88
- s2：+7.46
- s3：+8.89

### 结论

在当前口径下，RS on **没有改善**，反而稳定变差。  
这是 RQ2 的负结果。

口径：

- settrans + intra1
- 33 维
- GRPO
- 200 ep
- 无 AGV
- 3 seeds

⚠️ 不能与旧的 12 维 582.59 或旧 flat e2e 676.99 直接比。

### 当前进度

| 状态 | 配置 |
|---|---|
| 完成 | E0 / E1 的 3 seeds + 评测 |
| 在跑 | E2 / E3 的 3 seeds |
| 待跑 | E4a / E4b / E5_e2e / E5_jobst / E5b_intra0 |
| 10×6 | 未开始 |

E2 最新 EP149–159。  
E3 最新 EP79–89。  
无失败，无 Traceback。

6×6 预计 12:00–13:00 跑完。  
之后跑 10×6。

## §72 2026-10-25 10:30 E2 完成：AGV 约束的代价

### 6×6×2 E2（AGV 规则，RS off，33 维）

验证（EP199）：

| seed | obj |
|---|---:|
| s1 | 592.40 |
| s2 | 618.17 |
| s3 | 605.77 |
| 均值 | **605.45** |

seed3 测试（top1，100 实例）：

| seed | obj |
|---|---:|
| s1 | 585.95 |
| s2 | 591.97 |
| s3 | 583.27 |
| 均值 | **587.06 ± 4.5** |

### 与 E0 比：AGV 资源约束的代价

| 指标 | E0（无 AGV） | E2（AGV 规则） | 差 |
|---|---:|---:|---:|
| 验证 EP199 | 582.74 | 605.45 | +22.71 |
| seed3 测试 | 559.31 | 587.06 | **+27.75** |

有限车队 + 规则派车，测试高 **27.75**（约 5.0%）。  
这是 RQ4 的 A vs B 读数。

### 当前进度

| 状态 | 配置 |
|---|---|
| 完成 | E0 / E1 / E2（各 3 seeds + 评测） |
| 在跑 | E3（EP149–159） |
| 在跑 | E4a（EP30–39） |
| 待跑 | E4b / E5_e2e / E5_jobst / E5b_intra0 |
| 10×6 | 未开始 |

无失败，无 Traceback。  
GPU 48%，显存 7.9 / 24.6 GB。

### 下一步

E3 完成后看：

- E3（AGV actor）能否收回 E2 的 27.75。
- RQ3：B vs C。

预计：

- E3 约 11:00 完成。
- E4a 约 11:30。
- E4b 约 12:30。
- E5 / E5b 约 14:00。
- 然后 10×6。

## §73 2026-10-25 11:13 E3 完成：AGV actor 收回 58% 的约束代价

### 6×6×2 E3（AGV actor，RS off，33 维）

验证（EP199）：

| seed | obj |
|---|---:|
| s1 | 586.43 |
| s2 | 594.11 |
| s3 | 592.23 |
| 均值 | **590.93** |

seed3 测试（top1，100 实例）：

| seed | obj |
|---|---:|
| s1 | 568.81 |
| s2 | 566.83 |
| s3 | 577.22 |
| 均值 | **570.95 ± 5.5** |

### RQ3：actor vs 规则

| 配置 | seed3 测试 |
|---|---:|
| E2（AGV 规则） | 587.06 |
| E3（AGV actor） | **570.95** |
| 差 | **−16.11** |

actor 低 **16.11**。  
**3/3 seed actor 更好。**

分 seed 差（E2 − E3）：

- s1：+17.14
- s2：+25.14
- s3：+6.05

验证也一致：

- E2：605.45
- E3：590.93
- 差：−14.52

### 与无 AGV 比

| 配置 | seed3 测试 |
|---|---:|
| E0（无 AGV） | 559.31 |
| E3（AGV actor） | 570.95 |
| 差 | +11.64 |

AGV 约束的代价：

- 规则派车：+27.75
- actor 派车：+11.64
- 收回 **16.11 / 27.75 = 58%**

### 当前进度

| 状态 | 配置 | 最新 |
|---|---|---|
| 完成 | E0 / E1 / E2 / E3 | 各 3 seeds + 评测 |
| 在跑 | E4a（RS + AGV 规则） | EP96–104 |
| 在跑 | E4b（RS + AGV actor） | EP7–10 |
| 待跑 | E5_e2e / E5_jobst / E5b_intra0 | — |
| 10×6 | 未开始 | — |

无失败，无 Traceback。  
GPU 19%，显存 7.9 / 24.6 GB。

### 预计

- E4a 约 11:30。
- E4b 约 12:30。
- E5 / E5b 约 14:00。
- 然后 10×6。

## §74 2026-10-25 11:20 E0 vs E1 配对检验

同一批 seed3 100 实例，逐实例配对。

### 分 seed

| seed | E0 | E1 | 差（E1 − E0） | t（n=100） | E1 更差比例 |
|---|---:|---:|---:|---:|---:|
| 1 | 550.71 | 568.59 | +17.88 | 2.25 | 63% |
| 2 | 553.90 | 561.36 | +7.45 | 1.02 | 55% |
| 3 | 573.33 | 582.22 | +8.89 | 1.09 | 58% |

只有 seed1 单 seed 显著。  
seed2 / seed3 单 seed 不显著。

### 跨 seed 平均后

每个实例先对 3 个 seed 求平均，再做配对检验：

- 差：**+11.41**
- sd：47.21
- t：**2.42**
- n：100
- 近似 p：**0.017**
- E1 更差比例：**61%**

### seed 级检验

3 个 seed 的均值差：

- 均值：+11.41
- sd：5.65
- t：**3.50**
- n：3
- 近似 p：**0.07**

n=3 不显著。

### 解释

方向一致：

- 3/3 seed 均值都是 E1 更差。
- 61% 实例是 E1 更差。
- 实例级配对显著。

但 seed 级 n=3 不显著。  
不能说「RS 一定有害」。

### 当前判断

- 在 6×6、33 维、GRPO、200 ep 下，RS 没有改善。
- 效应量约 **2%**（11.41 / 570）。
- 历史正结果（743.06 vs 811.45）没有复现。
- E4a / E4b 未完成；RS + AGV 的交互还没读数。

### 待做

- 看顶层动作分布与熵。
- `hier_alpha=0.5`。
- 固定 λ 下重跑 E0 / E1。
- 200 ep 跑完后决定是否补 400 ep。

## §75 2026-10-25 12:04 顶层塌缩到固定规则

### 训练熵轨迹（E1，3 seeds）

| seed | 首 3 个 | 末 3 个 | 最小 | 平均 |
|---|---|---|---|---:|
| 1 | 2.483 / 2.480 / 2.476 | 0.690 / 0.738 / 0.729 | 0.683 | 1.612 |
| 2 | 2.483 / 2.476 / 2.465 | 0.868 / 0.834 / 0.823 | 0.139 | 0.933 |
| 3 | 2.483 / 2.479 / 2.475 | 1.619 / 1.548 / 1.493 | 0.705 | 1.340 |

uniform 熵 = ln(12) = **2.485**。  
三个 seed 都从均匀开始，向低熵塌缩。  
seed2 最低到 **0.139**。

### 测试时动作分布（E1 s1，top1，500 次决策）

| 项 | 值 |
|---|---:|
| 决策数 | 500 |
| greedy argmax = 11 | **500 / 500** |
| mean entropy | 1.069 |
| min / max | 0.615 / 1.835 |

动作概率：

| action | 概率 | 规则对 |
|---:|---:|---|
| 11 | **0.690** | mwko + min_trans |
| 5 | 0.135 | eft + min_trans |
| 2 | 0.070 | est + min_trans |
| 8 | 0.043 | spt + min_trans |
| 其他 | 0.062 | — |

机器规则 `min_trans` 的概率总和：**0.937**。  
job 规则以 `mwko` 为主（0.690）。

索引拆法：

```
action = job_rule_index * 3 + machine_rule_index
job: 0=est, 1=eft, 2=spt, 3=mwko
machine: 0=min_t, 1=min_pt, 2=min_trans
```

### 解释

顶层没有学成状态相关的规则策略。  
它在 greedy 下固定选 **mwko + min_trans**。

这解释了为什么 RS 反而更差：

- 每 8 步强制同一规则对。
- 低层 actor 的自由度被压缩。
- `min_trans` 只服务 0.2 权重的 transT，忽视 mk / pt。
- 顶层只学到「固定最省运输」，不是多目标调度。

注意：

- 这是 E1 s1 的 top1。
- s2 / s3 的测试动作分布还没测。
- 训练时是采样，不是 greedy；所以仍有熵。
- E4a / E4b 的顶层分布待测。

### 下一步

1. `hier_alpha=0.5`。  
   降低规则先验，看低层是否恢复。
2. 固定 λ 下重跑 E0 / E1。  
   隔离 λ 口径。
3. 顶层加熵正则或状态相关奖励。  
   避免塌缩到单一规则对。

## §76 2026-10-25 13:15 E4a 完成：RS + AGV 规则不稳定

### 6×6×2 E4a（RS on + AGV 规则）

seed3 测试（top1，100 实例）：

| seed | obj |
|---|---:|
| s1 | **745.25** |
| s2 | 600.54 |
| s3 | 589.86 |
| 均值 | **645.22 ± 86.8** |

验证（EP199）：

| seed | obj |
|---|---:|
| s1 | 769.95 |
| s2 | 611.19 |
| s3 | 602.39 |
| 均值 | 661.18 |

s1 是明显异常点。  
s2 / s3 与 E2（587.06）接近或略差。

### 对比

| 配置 | seed3 测试 |
|---|---:|
| E2（AGV 规则，RS off） | **587.06** |
| E4a（AGV 规则，RS on） | **645.22** |
| 去掉 E4a s1 | 595.20 |

RS on 让 s1 崩了。  
即使去掉 s1，E4a 也没有优于 E2。

### 当前进度

| 状态 | 配置 | 最新 |
|---|---|---|
| 完成 | E0 / E1 / E2 / E3 / E4a | 各 3 seeds + 评测 |
| 在跑 | E4b（RS + AGV actor） | EP159–167 |
| 在跑 | E5_e2e | EP69–79 |
| 待跑 | E5_jobst / E5b_intra0 | — |
| 10×6 | 未开始 | — |

E5_e2e 最新验证：

- s1 721.97
- s2 704.17
- s3 825.09

还没到 200 ep，不作结论。  
无失败，无 Traceback。

### 结论

RS on 在 6×6 目前有三个负结果：

- E1：570.72 vs E0 559.31。
- E4a：645.22 vs E2 587.06，且 s1 崩。
- 顶层诊断：塌缩到固定 mwko + min_trans。

RS 不只是没用，还会带来不稳定。

## §77 2026-10-25 13:30 顶层奖励口径 bug + 修复

### bug

计划写：顶层段回报固定 `0.4/0.4/0.2`。  
代码不是。

路径：

| 位置 | 行为 |
|---|---|
| `Run.py:291-294` | GRPO 每组随机 λ 写进 `variant['weight_*']` |
| `Run.py:38-49` | `_weighted_cost` 读 `variant['weight_*']` |
| `Run.py:793` | 顶层段回报用 `_weighted_cost` |
| `Run.py:1057-1059` | 验证用 `eval_variant`，固定 0.4/0.4/0.2 |

所以：

- 低层：随机 λ。
- 顶层：也是随机 λ。
- 验证：固定 0.4/0.4/0.2。

### 更严重

顶层状态 10 维，**不含 λ**。  
它在优化一个自己看不见的移动目标。

这可以解释：

- 顶层塌缩到固定 `mwko + min_trans`。
- E1 / E4a 的负结果。

### 影响

| 配置 | 是否受 bug 影响 |
|---|---|
| E0 / E2 / E3（无 RS） | 不受影响 |
| 6×6 E1 / E4a / E4b | **受影响；待重跑** |
| 10×6 E1 / E4a / E4b | 待跑；会用修复后代码 |
| E5 / E5b | 不受影响 |

### 修复

`_weighted_cost` 改用固定：

```python
TOP_WEIGHT_MK = 0.4
TOP_WEIGHT_EC = 0.4
TOP_WEIGHT_TT = 0.2
```

低层组 λ 不变。  
验证不变。

### 验证

- 本地 33 tests passed。
- 远端 10 ep GRPO + RS smoke：rc=0。
- 顶层熵 2.42–2.48，正常。
- `group lambda` 正常打印。
- 无 Traceback。

### 当前

远端 `Run.py` 已更新。  
正在跑的 E4b / E5 用旧代码：

- E4b 是 RS 臂，结果标为旧口径。
- E5 无 RS，不受影响。

### 下一步

1. 当前矩阵跑完。
2. 用修复后代码重跑 6×6 E1 / E4a / E4b。
3. 看 RS 是否还差。
4. 若仍差，再加 free 或降 alpha。

## §78 2026-10-25 13:38 并行重跑 RS（修复后代码）

用户要求：先并行重跑 RS，看效果。

### 处理

- 暂停 matrix 队列（父进程 185208）。
- 正在跑的 6 个继续：
  - 3 个 E4b（旧代码，结果作废）。
  - 3 个 E5_e2e（无 RS，有效）。
- 新队列：`experiment-queue-rs-fixed.json`
  - E1 / E4a / E4b × 3 seeds
  - 9 训练 + 9 评测
  - 6×6×2
  - GRPO + hier，`hier_alpha=1.0`
  - 33 维
  - 200 ep
  - 顶层段回报固定 0.4/0.4/0.2（修复后）
  - `max_parallel=9`

### 启动

13:38 启动。  
9 个 RS 训练全部开始。  
加上原有 6 个，共 **15 个 run**。  
GPU 11%，显存 10.1 / 24.6 GB。

### 待恢复

matrix 队列暂停后，以下未跑：

- E5_jobst
- E5b_intra0
- 10×6 四配置
- E5_e2e 的 3 个评测

等 RS 结果出来后再恢复。

### 预期

15 个 run 并发，6×6 会变慢。  
约 1.5–2 h 出 200 ep 结果。

### 口径

- 这批复用修复后的顶层奖励。
- `hier_alpha=1.0`，不降 alpha。
- 不加 free。
- 只隔离「顶层奖励修复」这一项。

## §79 2026-10-25 13:52 更正：保留随机 λ，顶层状态加 λ

### 用户纠正

§78 的修法作废。

用户指出：

- 实际训练就是随机 λ。
- 顶层看不见 λ 才是问题。
- 不该把顶层奖励冻结成 0.4/0.4/0.2。

这个判断是对的。  
我之前把计划当成了对的，没有检查计划是否自洽。

### 现在的实现

| 项 | 改法 |
|---|---|
| 低层 GRPO | 保持每组随机 λ，组内固定 |
| 顶层段回报 | 用**当前 λ**，不冻结 |
| 顶层状态 | 10 → **13 维**：10 摘要 + 3 个 λ |
| 验证 / 测试 | λ 固定 0.4/0.4/0.2，顶层状态收到这 3 个值 |
| 顶层性质 | **preference-conditioned** |

代码：

- `hierarchy/rules.py`：`top_state_features` 加 `weights`，返回 13 维。
- `Run.py`：两处传当前 λ；`TopLevelController(input_dim=13)`。
- `trainer/validate.py`：两处传验证 λ。
- `eval_st_seed3.py`：`input_dim=13`。
- `tests/test_hierarchy.py`：状态 13 维 + λ 默认值检查。

### 验证

- 本地 33 tests passed。
- 远端 10 ep GRPO + RS smoke：rc=0。
- group λ 正常。
- 顶层熵 2.40–2.48。
- 无 Traceback。

### 队列

- §78 的 rsfix 队列和 9 个 run 已停。
- 新队列：`experiment-queue-rs-lambda.json`
  - E1 / E4a / E4b × 3 seeds
  - 标签 `rslam_*`
  - 9 训练 + 9 评测
  - 6×6×2
  - GRPO + hier，随机组 λ
  - 顶层 13 维状态
  - 200 ep
  - `max_parallel=9`

13:52 启动。  
9 个 RS 训练 + 3 个 E5_e2e = 12 个 run。  
GPU 99%，显存 7.0 / 24.6 GB。

### 作废

旧顶层 checkpoint 是 10 维。  
全部作废。  
RS 必须全部重跑。

### 一个遗留

PPO 路径里：

- `variant['weight_*']` 是命令行固定值。
- 环境 reset 采样随机 λ，但没有写回 `variant`。
- 所以 PPO 的顶层状态看到的是固定权重，不是环境的随机 λ。

主模型是 GRPO，不受影响。  
PPO 只作对照。  
这个问题先记录，不在本轮修。

## §80 2026-10-25 14:08 PPO per-env λ 已修 + RS 进度

### PPO per-env λ 修复

用户要求：是 bug 就顺手改。

实现：

- 新增 `_current_weights`：
  - GRPO：组 λ 复制到所有 env。
  - PPO：读每个 env 的 `reward_random_weight`。
- 顶层状态用 per-env λ。
- `_weighted_cost` 用 per-env λ。
- 最终段 cost 用 per-env λ。
- `Objective` 日志用 per-env 加权均值。

对 GRPO：

- 组 λ 是标量；tile 后数值不变。
- 正在跑的 rslam 不受影响。

### PPO + RS 冒烟

3 ep，`--random_weight_type 01`：

- rc=0。
- 顶层更新 EP1–3。
- 每 env 用自己的随机 λ。
- 无 Traceback。

### RS 重跑进度（14:08）

9 个 rslam 训练存活：

| 配置 | 最新 EP |
|---|---:|
| E1 s1/s2/s3 | 16 |
| E4a s1/s2/s3 | 16 |
| E4b s1/s2/s3 | 10 |

无 Traceback。  
3 个 E5_e2e 也在跑。  
GPU 53%，显存 15.2 / 24.6 GB。

### 预计

12+ 个 run 并发，6×6 约 45 s/ep。  
200 ep 约 **2.5 h**。  
预计 **16:30** 左右出 RS 结果。

## §81 2026-10-25 15:50 RS 重跑中途读数

### 还没跑完

| 配置 | 最新 EP |
|---|---:|
| E1 s1/s2/s3 | 128 |
| E4a s1/s2/s3 | 128 |
| E4b s1/s2/s3 | 96 |

评测还没开始；seed3 测试要等 200 ep。

### 最新验证读数

| 配置 | EP | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|---:|
| E1（RS，无 AGV） | 119 | 589.76 | 583.24 | 592.69 | **588.56** |
| E4a（RS + AGV 规则） | 119 | 603.20 | 617.87 | 607.81 | **609.63** |
| E4b（RS + AGV actor） | 89 | 612.35 | 625.56 | 613.37 | **617.09** |

E4b 只到 EP89，不能和 EP119 对齐。

### 与旧结果对照

| 配置 | 旧（10 维） | 新（13 维） |
|---|---:|---:|
| E1 验证 | 588.65 @EP199 | 588.56 @EP119 |
| E4a 验证 | 661.18 @EP199 | 609.63 @EP119 |
| E0 验证 | 582.74 @EP199 | — |

初步看：

- E1 到 EP119 已接近旧 E1 的最终值，后面可能还会降。
- E4a 不再出现 s1 崩溃（旧 s1 = 769.95）。
- 还不能下结论。

### RS vs 无 RS（同 EP 验证）

| 对比 | RS | 无 RS | 差（RS − 无 RS） | EP |
|---|---:|---:|---:|---:|
| E1 vs E0（无 AGV） | 588.56 | 590.58 | **−2.02** | 119 |
| E4a vs E2（AGV 规则） | 609.63 | 603.66 | **+5.97** | 119 |
| E4b vs E3（AGV actor） | 617.09 | 603.75 | **+13.34** | 89 |

读法：

- E1：RS 反而低 2.02，但落在噪声内。
- E4a：RS 高 5.97，明显比旧 10 维的 +55.7 小。
- E4b：只到 EP89，RS 高 13.34；不终值。

旧 10 维对照（EP199 验证）：

| 对比 | 差（RS − 无 RS） |
|---|---:|
| E1 vs E0 | +5.91 |
| E4a vs E2 | +55.73（s1 崩） |

所以 λ 进状态后：

- E4a 的崩溃消失了。
- E1 从 +5.91 收到 −2.02。
- 但 E4b 还差 13.34。

还不能说 RS 已经赢。  
要等 200 ep 和 seed3 测试。

### 顶层熵

部分 seed 仍在塌缩：

| run | 熵 @ 最新 EP |
|---|---:|
| E1 s1 | 0.08 |
| E1 s3 | 0.42 |
| E4a s3 | 0.53 |
| E4b s1 | 0.36 |
| E1 s2 | 1.41 |
| E4a s2 | 1.45 |
| E4b s3 | 1.77 |

λ 进状态没有完全阻止塌缩。  
但可能塌缩成「按 λ 固定规则」；要看最终动作分布。

### 状态

- 9 个 run 存活。
- 无 Traceback。
- GPU 99%，显存 11.7 / 24.6 GB。

### 预计

- E1 / E4a：约 16:50。
- E4b：约 18:00。
- 之后跑 9 个 seed3 评测。

## §82 2026-10-25 16:20 顶层加机器负载不均衡 + alpha 0.5

### 用户要求

- 顶层状态加「机器负载不均衡」。
- 降 alpha。

### 状态

`top_state_features` 从 13 维改为 **14 维**：

- 原 10 维摘要
- 机器负载不均衡：`m2[:, :, 1]` 的变异系数
- 3 个 λ

### alpha

`--hier_alpha` 默认从 **1.0 改为 0.5**。

### 验证

- 本地 33 tests passed。
- 远端 10 ep GRPO + RS smoke：rc=0。
- `hier_alpha=0.5`。
- 顶层熵 2.34–2.48。
- 无 Traceback。

### 批次安排

当前 rslam（13 维，alpha=1.0）还在跑。

新批次 `experiment-queue-rs-lambda2.json`：

- E1 / E4a / E4b × 3 seeds
- 标签 `rslam2_*`
- 顶层 14 维
- `hier_alpha=0.5`
- 200 ep
- 9 训练 + 9 评测

watcher 已启动：

- 等所有 `rslam_` 训练和评测退出。
- 然后启动 rslam2 队列。

### 说明

两批不可混：

| 批次 | 状态 | alpha |
|---|---|---:|
| rslam | 13 维 | 1.0 |
| rslam2 | 18 维（10 + 5 状态量 + 3 λ） | 0.5 |

13 维与 18 维不兼容；旧 checkpoint 作废。

## §83 2026-10-25 17:00 名称口径：H-ST-GRPO

用户问：能不能不叫 GRPO。

结论：可以保留 H-ST-GRPO，但必须限定。

全称：

> Hierarchical Set-Transformer Multi-Agent Policy Optimization
> with Group-Relative Advantages

借的（Shao et al., arXiv:2402.03300, 2024）：

- 组内相对优势
- PPO clip
- 低层无 critic

不同的：

- 无 reference model，无 KL
- 组 = 同一批实例的 G 个 episode
- 三个 actor 共享 team advantage
- 每组随机 λ，4 目标
- 顶层 RS 是独立分层 PPO
- AA 同时选两端 AGV

我们的：

- 分层 RS
- 多 actor 共享优势
- 每组随机 λ
- 两端 AGV actor
- ST 编码器

写法：

- 首次出现写全称
- 引用 Shao et al. 2024
- 不说“我们提出了 GRPO”
- 说“采用 group-relative 优势并做多智能体/分层适配”

已写入计划 §2。

## §84 2026-10-25 17:01 先跑 ST 和基线，RS 最后

### 用户调整

先跑完 Set Transformer 消融和基线。  
RS 最后再跑。

### 已停

- rslam（13 维，alpha=1.0）队列和 9 个 run 已停。
- rslam2 watcher 已停。
- 停止时 E1 / E4a 已到 EP199；E4b 到 EP159–169。

最后验证：

| 配置 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| E1（13 维 RS） | 577.69 | 567.46 | 580.59 | **575.25** |
| E4a（13 维 RS） | 611.30 | 606.22 | 621.26 | **612.93** |
| E4b（13 维 RS，EP159–169） | 603.40 | 597.61 | 583.04 | 594.69 |

对照：

- E1 比 E0 最终 582.74 低 **7.49**。
- E4a 比 E2 最终 605.45 高 **7.48**。
- E4b EP 不齐，不能比。
- 这是验证集，没有 seed3 测试。

### E5_e2e 已完成

验证 EP199：

| seed | obj |
|---|---:|
| s1 | 719.22 |
| s2 | 676.79 |
| s3 | 735.99 |
| 均值 | **710.67** |

对照 E0（settrans）582.74：e2e 高 **127.93**。  
seed3 评测已排入新队列。

### 新队列

`experiment-queue-st-ablation.json`，17:01 启动：

- E5_jobst × 3 seeds
- E5b_intra0 × 3 seeds
- E5_e2e 的 3 个 seed3 评测
- 6×6，200 ep，GRPO，33 维
- `max_parallel=6`

当前 6 个训练在跑。  
GPU 16%，显存 3.0 / 24.6 GB。

### 下一步

1. ST 消融跑完。
2. 基线：RGA 实现；PDR / MIP / B3 适配。
3. RS 最后：18 维 + alpha0.5 的 rslam2。

## §85 2026-10-25 17:10 加 E0 PPO 对照

用户批准：在 ST 之后加 E0 PPO 对照。

### 队列

`experiment-queue-e0-ppo.json`

| 项 | 值 |
|---|---|
| encoder | settrans |
| 特征 | 33 维 |
| AGV | 无 |
| RS | off |
| 算法 | **PPO** |
| 预算 | 200 ep |
| seeds | 1/2/3 |
| 评测 | seed3 100 实例 |

3 训练 + 3 评测，`max_parallel=3`。

### 对照

| 算法 | 已有/待跑 | 读数 |
|---|---|---:|
| GRPO | 已有 E0 | **559.31 ± 12.2** |
| PPO | 待跑 | — |

用途：隔离 **PPO vs GRPO**。  
不是主模型对照。

### 启动方式

watcher：

- 等 `experiment-queue-st-ablation.json` 父进程退出。
- 然后启动 E0 PPO 队列。

watcher PID 250610，已确认在等。

### 主模型对照

E4b PPO 要等 E4b GRPO 18 维。  
两者放在 RS 最后一并跑。

### 当前

- ST 消融：6 个训练在跑。
- E0 PPO：watcher 等 ST 结束。
- RGA：实现中。

## §86 2026-10-25 17:30 论文基线必须带 AGV

### 用户澄清

和原论文对照的目的：**检验代码写得对不对**。  
不是论文的基线表。

论文的基线表**必须带 AGV**。  
主模型带 AGV，基线也必须带 AGV。

### 修正后的两张表

| 表 | 环境 | 用途 |
|---|---|---|
| A | no-AGV | 对论文 732.5；代码校验 |
| B | **AGV** | **论文主表** |

表 A：

- E0 vs 论文数字。
- 只判断实现是否同一量级。
- 不用于方法优劣。

表 B：

- 主模型 E4b。
- 基线全部带 AGV。

### AGV 基线计划

| 基线 | 做法 | 状态 |
|---|---|---|
| PDR-AGV | 规则选工序/机器；运输用最早空闲 | 要写 |
| RGA-AGV | OS/MS 解码；运输用最早空闲 | 要写 |
| B3-AGV | Lei 2022 的 job/machine + AGV 规则 | 要适配；标 adapted |
| MIP-AGV | 新 MILP，含 AGV 变量 | 6×6 限时试 |
| NSGA-II | 暂不做 | — |

### 实现顺序

1. PDR-AGV。
2. RGA-AGV（RGA 本体同时支持 `--agv 0/1`）。
3. B3-AGV 适配。
4. MIP-AGV 原型。

### 说明

- MIP / B3 的原模型没有 AGV。
- 原样数字只进表 A。
- 表 B 里必须写 adapted。
- 不能拿 no-AGV 基线评价带 AGV 的主模型。

计划 §6.3 / §6.4 已改。

## §87 2026-10-25 17:35 AGV 基线实现 + 本地读数

### 实现

`plcsp_v5/baselines/agv_baselines.py`

- 直接解码器：OS + MS，左移插入，支持 AGV / no-AGV。
- PDR：6 job 规则 × 2 machine 规则 + RA。
- RGA：OS/MS，POX + 均匀交叉 + 交换/改机器变异。
- 目标：`0.4·mk + 0.4·(pt/36 + idleT) + 0.2·transT`。

### 解码器校验

no-AGV，同一调度，本地 vs v5 环境：

| 量 | 本地 | 环境 |
|---|---:|---:|
| mk | 617.007282036018 | 617.007282036018 |
| pt | 16353.220716318267 | 16353.220716318267 |
| transT | 251.92555068685846 | 251.92555068685846 |
| idleT | 1003.9095566805192 | 1003.9095566805192 |
| obj | 880.4542980275229 | 880.4542980275229 |

**完全一致。**  
AGV 路径用同一调度核心 + `transport_timing` 等价公式；AGV 环境的整体校验待补。

### AGV 基线读数（100 实例，0.4/0.4/0.2）

| 基线 | 均值 | std |
|---|---:|---:|
| RA（随机） | 1258.30 | 207.41 |
| PDR 全部 12 条 | 1471.95 | 499.58 |
| 最好 PDR：MOR + SPT | **951.20** | 139.49 |
| **RGA** | **478.04** | 60.38（300 run 总体） / 23.85（实例内 3 seed） / 1.67（seed 均值） |

RGA：

- pop 50，gen 100，4850 evals/实例。
- seed1 475.72，seed2 478.81，seed3 479.59。
- 300 run，8 workers：267.9 s（与 no-AGV 批次并发，读数偏慢）。
- 单 worker，远程 CPU：1.63 s/实例（5 实例）。
- 单 worker，本地 CPU：4.81 s/实例（5 实例）。

对照我们的模型（旧检查点，固定环境重评）：

| 方法 | seed3 测试 |
|---|---:|
| E2 AGV 规则 | 576.04 |
| E3 AGV actor | 557.72 |
| RGA-AGV | **478.04** |

### 结论

- **RGA 质量明显更好。**  
  比 E3 低 **79.68**（旧环境读数 570.95 已作废）。
- 这和论文一致：RGA / 2SGA 可以优于 DRL。
- 我们的模型不能在质量上声称赢 RGA。
- DRL 的卖点只能是：
  - 在线时间
  - 多目标 / 随机 λ
  - 在线重调度
- RGA 单 worker 1.63 s/实例（远程 CPU）；
  DRL 评测约 1.0 s/实例（远程 GPU，6 个评测并发）。
  两者的时间优势不大，必须用统一口径重测。

### 文件

- `plcsp_v5/results/agv_pdr_100.json`
- `plcsp_v5/results/agv_ra_100.json`
- `plcsp_v5/results/agv_rga_100x3.json`

### 待补

- AGV 环境对解码器的整体校验。  
  → 已完成，见 §88；同时发现并修复一个环境 bug。
- PDR 分规则表（12 条）。  
  → 已完成，见 §90。
- 统一墙钟协议：DRL 推理 vs RGA evals。  
  → 已定义，见 §91。
- RGA 在 no-AGV 模式下的表 A 读数。  
  → 已完成：474.87 ± 55.99。

## §88 2026-10-25 18:10 AGV 环境校验：跨车间固定延时被重复计费

### 现象

固定调度（PDR MOR + SPT，实例 0），解码器 vs v5 环境：

| 量 | 解码器 | 环境 | 差 |
|---|---:|---:|---:|
| mk | 521.016 | 526.016 | +5.000 |
| pt | 16353.2207 | 16353.2209 | ≈0 |
| transT | 191.9256 | 191.9256 | 0 |
| idleT | 807.1011 | 822.1011 | +15.000 |
| obj | 751.3344 | 759.3344 | +8.000 |

逐任务开始时间：

- 跨车间运输的任务：环境比解码器晚 **5.0**。
- 连续两次跨车间的任务：晚 **10.0**。
- 同车间运输：完全一致。

### 根因

`find_max_arrivaTime_for_currentNode` 有两处时间来源：

1. `edge_init_weight`：整段机器到机器时间。  
   跨车间时，它**已含**固定延时 `d_eff = 5`。
2. `transport_wait_override`：AGV 等待。  
   跨车间时，它也**已含** `d_eff = 5`。

两处相加，`d_eff` 被计两次。  
`find_transportT` 已用 `transport_override`（只含装载段）；  
只有开始时间路径用了旧表值。

### 修复

`plcsp_v5/trainer/DGenv_func.py`：

- `find_max_arrivaTime_for_currentNode` 先查 `transport_override`。
- 命中时用装载段时间，不再用整段表值。
- 然后只加一次 `wait`。

新增回归测试：

- `test_cross_edge_start_uses_loaded_override_not_table`
- 旧代码给 17.0；新代码给 15.0。

### 校验

`plcsp_v5/baselines/check_agv_env.py`：

- 3 个实例，PDR MOR + SPT，AGV 与 no-AGV 都跑。
- 最大绝对差：
  - mk 4.5e-6
  - pt 1.5e-4
  - transT 2.7e-6
  - idleT 6.2e-6
  - obj 5.8e-6

float32 精度内一致。  
no-AGV 路径不受影响；修复只在 `transport_override` 存在时生效。

### 影响

旧 E2 / E3 / E4a / E4b 读数都带重复计费，全部作废。  
旧检查点用固定环境重评：

| 配置 | 旧读数 | 固定环境 | 变化 |
|---|---:|---:|---:|
| E2 AGV 规则 | 587.06 | 576.04 | −11.02 |
| E3 AGV actor | 570.95 | 557.72 | −13.23 |

E3 仍优于 E2，3/3 seed；差距 18.32。  
E2 比 E0（559.31）高 16.73。  
E3 比 E0 低 1.59。  
注意：AGV 与 no-AGV 的 `transT` 口径不同。  
AGV 只计装载段；no-AGV 计整段。  
两者不能直接比较绝对值。

E4a / E4b 的顶层检查点是 **10 维**输入。  
当前评测是 **18 维**，不能直接重评。  
两者必须重训。

### 重训队列

`plcsp_v5/experiment-queue-agv-fix-6x6.json`：

- E2 / E3 / E4a / E4b × 3 seeds。
- 6×6×2，200 ep，GRPO，33 维特征。
- E4a / E4b：18 维顶层状态，`hier_alpha=0.5`。
- 24 个 job（12 训练 + 12 评测）。
- 运行标签带 `_fix_remote`，不覆盖旧检查点。

10×6 的 AGV 分支也必须用修复后的环境。  
10×6 尚未开始。

## §89 2026-10-25 17:50 Set Transformer 消融完成

条件：6×6×2，GRPO，no-AGV，no-RS，200 ep，3 seeds。  
读数：seed3 100 实例，greedy。

| 编码器 | seed1 | seed2 | seed3 | 均值 |
|---|---:|---:|---:|---:|
| settrans，machine intra-attn 开 | 550.71 | 553.90 | 573.33 | **559.31** |
| settrans，machine intra-attn 关 | 620.75 | 622.29 | 630.86 | 624.63 |
| job ST + machine GAT | 636.35 | 646.42 | 663.67 | 648.81 |
| e2e（GIN/GAT） | 671.86 | 665.77 | 693.35 | 676.99 |

结论：

- 双 ST + machine intra-attn 最好，3/3 seed。
- 关掉 intra-attn：+65.32。
- job ST + GAT：+89.50。
- e2e：+117.68。

这是 no-AGV、no-RS 的编码器消融。  
AGV 主表要在修复环境重训后复核同一排序。

## §90 2026-10-25 18:05 基线：PDR 12 条 + RA + RGA 两模式

### PDR（AGV，100 实例）

| job 规则 | machine 规则 | 均值 | std |
|---|---|---:|---:|
| mor | spt | **951.20** | 139.49 |
| mor | seyc | 975.70 | 150.92 |
| mwkr_t | spt | 1011.31 | 136.54 |
| mwkr_t | seyc | 1063.07 | 182.03 |
| mwkr_pt | spt | 1150.76 | 156.53 |
| mwkr_pt | seyc | 1179.19 | 185.64 |
| lwkr_t | spt | 1837.16 | 328.63 |
| lwkr_t | seyc | 1880.06 | 350.01 |
| fifo | spt | 1880.91 | 309.56 |
| lwkr_pt | spt | 1887.11 | 370.06 |
| lwkr_pt | seyc | 1902.03 | 367.64 |
| fifo | seyc | 1944.94 | 331.02 |

全部 1200 run：1471.95 ± 499.58。

### PDR（no-AGV，表 A，100 实例）

最好四条：

| job 规则 | machine 规则 | 均值 | std |
|---|---|---:|---:|
| mor | spt | **913.87** | 135.99 |
| mwkr_t | spt | 933.34 | 136.62 |
| mor | seyc | 935.36 | 146.46 |
| mwkr_t | seyc | 975.04 | 169.94 |

全部 1200 run：1078.68 ± 230.18。  
完整 12 条见 `plcsp_v5/results/agv_pdr_100_nagv.json`。

### RA 与 RGA

| 模式 | 方法 | 均值 | 300/100 run 总体 std | 实例内 3 seed std |
|---|---|---:|---:|---:|
| AGV | RA | 1258.30 | 207.41 | — |
| AGV | RGA | **478.04** | 60.38 | 23.85 |
| no-AGV | RA | 1085.46 | 178.75 | — |
| no-AGV | RGA | **474.87** | 55.99 | 20.94 |

RGA 参数：pop 50，gen 100，4850 evals/实例，3 seeds。  
no-AGV RGA 只比 AGV 低 3.17。  
原因：AGV 的 `transT` 只计装载段；no-AGV 计整段。  
两模式的目标值不能直接跨表比较。

### 结论

RGA 质量仍显著优于 DRL。  
修复环境后 E3 为 557.72；RGA 低 **79.68**。  
论文不能声称质量赢 RGA。  
表 A 只用 no-AGV 检查代码；表 B 用 AGV 做主比较。

## §91 2026-10-25 18:08 墙钟口径（草案）

### 协议

1. 同一台机器：远程 4090 主机。
2. 一次一个实例，单进程；计时不用多 worker。
3. 不计模型加载和实例文件读取。
4. 计入调度构造和成本计算。
5. RGA：pop 50，gen 100，1 seed，4850 次解码。
6. DRL：1 个 greedy episode，36 次决策。
7. 报 100 个测试实例的均值和中位数。
8. 报硬件、CPU/GPU、完整调度评价次数。

### 当前读数

| 方法 | 硬件 | 每实例 | 说明 |
|---|---|---:|---|
| RGA-AGV | 远程 CPU，1 worker | 1.63 s | 5 实例 |
| RGA no-AGV | 远程 CPU，1 worker | 0.70 s | 5 实例 |
| DRL-AGV | 远程 GPU | **0.56 s** | 100 实例，单评测，无并发 |
| DRL no-AGV | 远程 GPU | ≈0.44 s | 早期 3 并发日志 |

DRL 的优势约 **2.9×**（0.56 vs 1.63），不是数量级。  
DRL 用 GPU；RGA 用 CPU。  
两者的硬件不同，论文必须写清。

### 待补

- DRL 单实例、无并发的干净计时。
- DRL CPU-only 计时。
- RGA 100 实例、单 worker 计时。
- 中位数和四分位，不只均值。

## §92 2026-10-25 18:25 B3-AGV 与 MIP 读数

### B3（Lei et al. 2022）

预训练权重：

- `tester/ESWA_MPPO/esa_PPO_operation_actor_J6M6E2_800.pth`
- `tester/ESWA_MPPO/esa_PPO_machine_actor_J6M6E2_800.pth`

实现：

- `plcsp_v5/baselines/eval_b3.py`。
- job/machine 用 B3 的 actor，greedy。
- 运输用最早空闲 AGV 规则。
- 机器头首层按 6→10 零填充，保留原权重，忽略 4 个 AGV 特征。
- 环境用修复后的 `DGenv_func.py`。

读数（seed3 100 实例，0.4/0.4/0.2）：

| 模式 | 均值 | std |
|---|---:|---:|
| B3 no-AGV（表 A） | 700.60 | 131.48 |
| B3-AGV（表 B） | 707.21 | 127.81 |

结论：

- B3-AGV 比 E2（576.04）和 E3（557.72）都差。
- B3-AGV 比最好 PDR（951.20）好。
- 只能写 **adapted**：原权重没有 AGV 输入。

### MIP

`tester/Solver_seed3/MO_FJSP_MIP_result_(J6_M6_seed3_sample100_w442).csv`：

- 重算目标：**585.83 ± 82.66**。
- 求解时间：均值 18.34 s，中位 13.99 s。
- CSV 没有 gap 列，也没有调度解。
- 这是限时 incumbent，**不是已证最优界**。

⚠️ RGA no-AGV（474.87）低于 MIP incumbent。  
说明限时 MIP 解可以被超越。  
论文不得把该 CSV 写成最优界。

### 当前表 A（no-AGV，6×6，seed3，100 实例）

| 方法 | 均值 |
|---|---:|
| RGA | **474.87** |
| E0 双 ST + GRPO | 559.31 |
| MIP incumbent | 585.83 |
| B3 adapted | 700.60 |
| 最好 PDR | 913.87 |

### 当前表 B（AGV，6×6，seed3，100 实例）

| 方法 | 均值 |
|---|---:|
| RGA | **478.04** |
| E3（旧检查点，固定环境重评） | 557.72 |
| E2（旧检查点，固定环境重评） | 576.04 |
| B3-AGV adapted | 707.21 |
| 最好 PDR | 951.20 |

E2 / E3 的正式数字等 `_fix_remote` 重训完成后替换。

### 缺口

- MIP-AGV 未建模；原 CSV 没有调度解，不能直接加 AGV 规则。
- B3-AGV 的零填充是适配手段，不是原方法。
- B3 的在线时间还没测。

### 文件

- `plcsp_v5/baselines/eval_b3.py`
- 远程 `results/b3_nagv_100.json`
- 远程 `results/b3_agv_100.json`

## §93 2026-10-25 19:50 基线跑完：6×6 + 10×6

### 汇总

`plcsp_v5/results/baselines_summary.json`  
所有结果都是 seed3 100 实例，目标 `0.4/0.4/0.2`。

### 6×6×2

| 方法 | AGV | no-AGV |
|---|---:|---:|
| PDR 最好（12 条） | 951.20（MOR+SPT） | 913.87（MOR+SPT） |
| PDR 全部 12 条 | 1471.95 | 1078.68 |
| RA | 1258.30 | 1085.46 |
| B3 adapted | 707.21 | 700.60 |
| RGA | **478.04** | **474.87** |
| MIP | 未建模 | 585.83（限时 incumbent） |

### 10×6×2

| 方法 | AGV | no-AGV |
|---|---:|---:|
| PDR 最好（12 条） | 1101.30（MOR+SEYC） | 1032.77（MOR+SEYC） |
| PDR 全部 12 条 | 2096.45 | 1322.14 |
| RA | 1603.37 | 1302.95 |
| B3 adapted | 651.30 | 658.32 |
| RGA | **565.63** | **564.62** |
| MIP | 未建模 | 508.39（60 s 上限） |

RGA 10×6：3 seed 均值 567.61 / 564.75 / 564.52（AGV）。  
实例内 std 26.66（AGV）/ 22.97（no-AGV）。

### 结论

- RGA 是两种规模最好的 AGV 基线。
- B3 adapted 明显差于 RGA 和 E3。
- MIP 只是限时 incumbent，不是最优界：
  - 6×6：RGA no-AGV（474.87）低于 MIP（585.83）。
  - 10×6：MIP（508.39）低于 RGA no-AGV（564.62）。
  - 规模变大后，限时 MIP 更强。
- PDR / RA 与 DRL 差距大，只能作规则参照。
- MIP-AGV 仍缺；原 CSV 没有调度解。

### 文件

- `plcsp_v5/results/agv_pdr_100*.json`
- `plcsp_v5/results/agv_ra_100*.json`
- `plcsp_v5/results/agv_rga_100x3*.json`
- `plcsp_v5/results/b3_*_100*.json`
- `plcsp_v5/results/baselines_summary.json`

## §94 2026-10-25 19:05 E0 PPO 对照完成

条件：6×6×2，settrans 33 维，PPO，G=8，200 ep，3 seeds。  
读数：seed3 100 实例，greedy。

| seed | PPO | GRPO E0 |
|---|---:|---:|
| 1 | 676.03 | 550.71 |
| 2 | 621.10 | 553.90 |
| 3 | 615.11 | 573.33 |
| 均值 | **637.41** | **559.31** |

GRPO 3/3 更低。  
差距 **78.10**。  
这是主编码器（settrans）下的干净 PPO 对照。

更早的 e2e 对照：  
GRPO 676.99 vs PPO 790.20，也是 3/3 GRPO 更低。  
两个编码器下方向一致。

文件：

- `plcsp_v5/results/e0_ppo_settrans_3seeds.json`
- 远程 `queue_logs/20261025_1903*_E_e0ppo_s*.log`

## §95 2026-10-25 20:10 AGV actor 同策略对照

目的：分离「AGV 策略」和「job/machine 策略差异」。  
做法：同一个 E3 checkpoint，评测时切换 AGV 策略。

- E3 actor：同一 job/machine 策略 + 学习派车。
- E3 rule：同一 job/machine 策略 + 最早空闲规则。
- 环境：修复后的 AGV 环境。

| seed | E3 actor | E3 + 规则 | 差 |
|---|---:|---:|---:|
| 1 | 557.63 | 562.59 | +4.96 |
| 2 | 555.61 | 565.74 | +10.13 |
| 3 | 559.93 | 576.21 | +16.28 |
| 均值 | **557.72** | **568.18** | **+10.46** |

actor 3/3 更好。

分解 E3 vs E2：

| 项 | 值 |
|---|---:|
| E2（自己的 jm 策略 + 规则） | 576.04 |
| E3 rule（E3 jm 策略 + 规则） | 568.18 |
| E3 actor（E3 jm 策略 + actor） | 557.72 |
| actor 效应 | **10.46** |
| jm 策略差异（同规则） | **7.86** |
| 合计 E3 vs E2 | **18.32** |

结论：

- AGV actor 本身有正贡献，约 10.46。
- E3 的 job/machine 策略在规则派车下也比 E2 好 7.86。
- 总差距 18.32 = 10.46 + 7.86。
- 这不是纯 actor 效应；旧 E2 vs E3 的对比混了两部分。

文件：

- `plcsp_v5/results/e3_rule_control_3seeds.json`
- 远程 `queue_logs/agvfix_eval/E3_s*_rule.log`

### 顶层 AGV 先验

顶层 RS 目前只选 job/machine 规则，不给 AGV 先验。  
是否加 AGV 规则头，待用户决定（A/B/C）。

## §96 2026-10-25 20:45 顶层改成 3 个头（A1，K=3）

### 设计

顶层三个头，独立采样：

| 头 | 动作数 | 规则 |
|---|---:|---|
| job | 4 | est / eft / spt / mwko |
| machine | 3 | min_t / min_pt / min_trans |
| AGV | 3 | earliest_free / nearest / random |

- log-prob 相加。
- 熵相加。
- critic 仍是一个标量。
- E1（无 AGV）用 2 个头。
- E4a / E4b（有 AGV）用 3 个头。

规则先验：

- job / machine：固定 logit 向量。
- AGV：状态相关，按当前 AGV 特征归一化到 [-1, 1]。
- source 和 destination 两个决策用同一个选中的 AGV 规则。

AGV 臂：

- E4a：顶层选 AGV 规则，直接派车（`--agv_top_rule 1`）。
- E4b：AGV actor + 顶层规则 logit 先验（`--agv_top_rule 0`）。
- E2 / E3：无顶层，不变。

### 代码

- `hierarchy/rules.py`：`AGV_RULES`、`agv_rule_bias`、`select_agv_by_rule`。
- `hierarchy/top_ppo.py`：三头 actor；动作是 (B, H)。
- `agv/actor.py`：`logit_bias`。
- `trainer/replaybuffer.py`：`agv_bias` / `agv_dst_bias`；`rule_id` 改成 (B, 3)。
- `algorithm/ppo_algorithm.py`：低层更新时重放 AGV bias。
- `Run.py` / `trainer/validate.py` / `eval_st_seed3.py`：接线。
- `parameters.py`：`--agv_top_rule`。
- `Run.py` 路径改成 `V5_ROOT`，本地和远程都能跑。

### 验证

- `pytest plcsp_v5/tests`：39 passed（路径改动前；待重跑）。
- 远程 smoke，actor 先验臂：rc=0；3 头熵 3.56。
- 远程 smoke，顶层 AGV 规则臂：rc=0。
- 两者都完成了 1 个 episode + 低层 PPO 更新 + 顶层更新。

### 影响

- 所有旧 RS 读数作废：
  - E1 570.72
  - E4a 645.22
  - E4b
  - rslam 13 维 / 18 维
- E0 / E2 / E3 不是 RS，不受影响。
- E2 / E3 环境修复队列继续跑。
- RS 重跑队列：
  - 6×6 E1 / E4a / E4b × 3。
  - 10×6 同样。
- RS λ / 状态 / 超参扫在核心臂之后。

### 熵

三头熵相加，约 3.5。  
`hier_entropy=0.01` 的等效强度约为单头的 3 倍。  
先保留，观察是否仍坍缩。

## §97 2026-10-25 21:00 RS 3 头启动 + E2/E3 修复 s1

### RS 3 头重跑

2026-10-25 20:55:31 启动 `experiment-queue-rs-heads3-6x6.json`。

- 18 个 job（9 训练 + 9 评测）。
- max_parallel 3。
- 首轮：E1 s1 / E4a s1 / E4b s1。

前 7 个 episode 的顶层熵：

| 臂 | 头数 | 熵 |
|---|---:|---:|
| E1 | 2 | 2.43 |
| E4a | 3 | 3.52 |
| E4b | 3 | 3.42 |

都接近均匀分布（理论值：2 头 2.485，3 头 3.584）。  
暂未坍缩。

### E2 / E3 环境修复重训

s1 评测：

| 臂 | 旧检查点固定环境重评 | 修复后重训 |
|---|---:|---:|
| E2 s1 | 577.43 | **568.70** |
| E2 s2 | 581.39 | **561.15** |
| E2 s3 | 583.27 | **571.46** |
| E3 s1 | 557.63 | **558.11** |

E2 修复后 3 seed 均值：**567.10**（旧固定环境重评 576.04）。  
E3 s1 仍低于 E2；s2 / s3 评测待出。

### 首轮 buffer 更新

E1 / E4a / E4b 都完成了第 1 次 buffer 更新（episode 8）。

- E1 / E4a：约 20:59 重采样。
- E4b：第一次带 AGV actor 的 GRPO 更新约 5 分钟；21:01 完成并重采样。
- 无报错。

3 个 RS 训练 + E2/E3 评测并发时，pkl 重采样明显变慢。  
等 E2/E3 队列结束后会加快。

## §98 2026-10-25 21:12 10×6 主矩阵启动

`experiment-queue-10x6-main.json`，24 个 job：

- E0 × 3 seeds（无 RS、无 AGV）。
- E1 × 3（2 头 RS）。
- E4a × 3（3 头，顶层 AGV 规则）。
- E4b × 3（3 头，AGV actor + 顶层先验）。
- 每个训练都带一个 seed3 评测。

max_parallel = 3。  
21:12 启动首轮：E0 s1 / E1 s1 / E4a s1。  
E4b s1 等下一个空位。

当前远程 8 个训练：

- E3 s2 / s3（6×6 环境修复）。
- E1 / E4a / E4b s1（6×6 RS 3 头）。
- E0 / E1 / E4a s1（10×6 主矩阵）。

GPU：8.2 / 24.6 GB，利用率 79%。

### 备注

- 10×6 E0 是新跑，旧 e2e 10×6 613.98 只有 1 seed，不作主表。
- 10×6 E1 / E4a / E4b 旧读数作废；现在用 3 头重跑。
- 10×10 不跑。

## §99 2026-10-25 21:50 RGA 等预算对照

6×6，seed3 100 实例，3 seeds。  
RGA：pop 50；改代数控制预算。

| 评价次数 | AGV | no-AGV |
|---:|---:|---:|
| 530（gen10） | 695.61 | 650.35 |
| 1010（gen20） | 606.04 | 580.41 |
| 1586（gen32） | **550.85** | **536.64** |
| 2450（gen50） | 511.95 | 503.94 |
| 4850（gen100） | 478.04 | 474.87 |

DRL 对照：

| 方法 | 均值 |
|---|---:|
| E3 AGV s1 / s2 | 558.11 / 558.35 |
| E2 AGV 3 seed | 567.10 |
| E0 no-AGV 3 seed | 559.31 |

等时口径：

- DRL-AGV：0.56 s/实例（GPU，单实例）。
- RGA-AGV：4850 evals 1.63 s（CPU，单 worker）。
- 1586 evals ≈ 0.53 s。

关键读数：

- 1586 evals：RGA-AGV **550.85** 优于 E3 约 **7.3**。
- 1010 evals（≈0.34 s）：RGA 606.04，DRL 更好。
- 交叉点约 **1400 evals / 0.47 s**。

结论：

- 等时 0.56 s，DRL **不胜** RGA。
- 质量优势不存在。
- 在线时间优势只在小预算（<0.5 s）成立。
- RGA 8 worker 吞吐 0.145 s/实例，比 DRL 单实例更快。
- 论文不能写「质量更好」或「在线时间更快」。
- 只剩多目标 / 随机 λ、在线重调度可写；这两项还没实验。
- 等预算曲线必须进论文。

### 文件

远程 `plcsp_v5/results/rga_budget_g*_agv*.json`。

## §100 2026-10-25 22:00 论文基线范围：暂不放 RGA

用户决定：主表按 E2E 论文的基线集合，**暂不放 RGA**。

### 表 A（no-AGV）

- 12 条 PDR
- RA
- MIP（限时 incumbent）
- B3 / ESA adapted
- E2E P-G（论文数）
- 我们的 E0

### 表 B（AGV）

- PDR-AGV
- RA-AGV
- B3-AGV adapted
- 我们的 E2 / E3
- MIP-AGV 仍缺

### RGA 的处理

- 读数保留在 `plcsp_v5/results/` 和本记录。
- 不删除。
- 不放进主表。
- 等预算曲线也留在内部。

### 风险

- 审稿人可能要求 RGA。
- 不得写「优于所有基线」；只能写「优于所比较的基线」。
- 如需，可在附录或限制说明中交代 RGA 未纳入主表。

### 待办

- RS 3 头结果。
- 10×6 主矩阵。
- MIP-AGV（或明确写缺口）。
- 论文骨架等这些稳定后再写。

## §101 2026-10-25 22:25 RGA 参数核查与基线决定

### E2E 原文

- 基线含 **RGA** 和 **2SGA**。
- 原文只说 RGA 在所有规模用同一套参数，未按规模调参。
- **没有给出 population size / iterations / crossover / mutation 的数值。**
- Table IV 是 DRL 超参数，不是 RGA。

### 官方仓库

- `RKWin93/E2E-MAPPO-for-MT-FJSP` 没有 RGA / 2SGA 代码。
- `test_all.py` 只跑 PDR / RA / MIP / ESA / P-G / P-S。

### Zhang et al. 2011（RGA 原始文献）

- 可访问片段没有 pop / gen / crossover / mutation 数值。
- 全文需要机构权限或 PDF。
- 可确认的算子：
  - 初始种群 60% GS + 30% LS + 10% RS
  - 三元锦标赛
  - MS：两点交叉 + 均匀交叉
  - OS：POX

### Table VI 的 RGA / 2SGA 读数

E2E 论文 Table VI（no-AGV，100 测试实例，ω={0.4,0.4,0.2}）：

| 规模 | P-G | RGA [43] | 2SGA [44] | RGA 时间 |
|---|---:|---:|---:|---:|
| 6×6×2 | 732.5 | **600.2** | **492.0** | 638.8 s |
| 10×6×2 | 674.5 | **653.3** | **532.8** | 1994.0 s |
| 20×6×3 | 949.1 | 990.4 | 846.9 | 7787.1 s |
| 10×10×2 | 1654.8 | 1506.0 | 1012.7 | 5590.3 s |
| 15×10×2 | 1614.0 | 1615.2 | 964.6 | 12386.0 s |
| 20×10×5 | 2089.9 | 2451.3 | 1420.3 | 24608.8 s |

注意：

- RGA / 2SGA 的实例数在大部分规模不是 100：25 / 30 / 10 / 5。
- 表中时间是整批时间还是单实例时间，原文没写清；按数值看更像整批。
- 原文正文说「P-G 与 RGA 接近」，但 6×6 是 732.5 vs 600.2，并不接近。
- 原文明确承认 RGA / 2SGA 可以得到更高质量的解，卖点是在线时间。

### 关键对比

| 方法 | 6×6 |
|---|---:|
| E2E P-G | 732.5 |
| E2E RGA [43] | 600.2 |
| E2E 2SGA [44] | **492.0** |
| 我们的 E0（no-AGV） | **559.31** |
| 我们的 RGA（pop50/gen100） | **478.04** |

结论：

- 我们的 E0 **优于 E2E 的 RGA**（559.31 < 600.2）。
- 我们的自建 RGA 比 E2E 的 RGA 更强（478.04）。
- 我们的 E0 **不如 E2E 的 2SGA**（559.31 > 492.0）。
- 所以「RGA 看起来打不过」主要是因为我们用了**自己更强的 RGA**，不是 E2E 的 RGA。

### 决定

- 论文主表**暂不放 RGA**。
- 不写「与 E2E 相同基线」；E2E 还有 RGA / 2SGA。
- 我们的 RGA 结果和等预算曲线保留在 `results/` 与本记录。
- 若审稿人要求，再补：  
  a) E2E Table V/VI 的 RGA 数值；  
  b) Zhang et al. 2011 的原始参数；  
  c) 或明确写缺口。

## §102 2026-10-25 22:15 训练进展

### E2 / E3 环境修复：完成

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| E2 AGV 规则 | 568.70 | 561.15 | 571.46 | **567.10** |
| E3 AGV actor | 558.11 | 558.35 | 575.69 | **564.05** |

逐 seed 差（E3 − E2）：

- s1：−10.59
- s2：−2.80
- s3：+4.23

E3 只在 2/3 seed 更好；均值差 **3.05**。  
s3 是明显离群点。  
AGV actor 的增益不稳，不能写「稳定优于规则」。

同策略对照（旧 E3 checkpoint）仍显示 actor 本身低 10.46。  
但新重训的 3 seed 结果表明：训练方差会吃掉大部分增益。

### 6×6 RS 3 头 s1

| 臂 | episode | 熵 |
|---|---:|---:|
| E1（2 头） | 144 | 1.186 |
| E4a（3 头，AGV 规则） | 140 | 2.561 |
| E4b（3 头，AGV actor） | 110 | 3.105 |

E1 仍接近坍缩。  
E4a 从 3.58 降到 2.56。  
E4b 从 2.24 回升到 3.10。

### 10×6 主矩阵 s1

| 任务 | episode | 读数 |
|---|---:|---|
| E0 | 49 | 验证 619.37 |
| E1（2 头） | 56 | 熵 2.340 |
| E4a（3 头） | 56 | 熵 2.480 |
| E4b | 未启动 | — |

E4a 在 10×6 的熵下降较快。  
E4b 等 3 个槽位之一空出。

### 队列

- E2/E3 修复：**12/12 完成**。
- 6×6 RS：s1 三个臂训练中。
- 10×6 主矩阵：E0 / E1 / E4a s1 训练中。
- 当前 6 个训练，GPU 11.0 / 24.6 GB，利用率 90%。
- 无报错。

## §103 2026-10-25 22:35 RS 进展

### 6×6 s1

| 臂 | episode | 熵 |
|---|---:|---:|
| E1（2 头） | **200（完成）** | **1.302** |
| E4a（3 头，AGV 规则） | 190 | 1.980 |
| E4b（3 头，AGV actor） | 150 | 2.771 |

E1 已跑完 200 ep，评测待启动。  
E1 / E4a 都明显坍缩。  
E4b 仍较高。

### 10×6 s1

| 任务 | episode | 读数 |
|---|---:|---|
| E0 | 79 | 验证 **596.95** |
| E1（2 头） | 79 | 熵 **1.927** |
| E4a（3 头） | 79 | 熵 **2.356** |

10×6 的 E1 / E4a 也在快速降熵。

### 与 RS off 的对比（seed1 验证集）

同一验证集、同一 seed1：

| 组 | 方法 | 最后验证 | 最好验证 |
|---|---|---:|---:|
| RS off | E0（无 AGV） | 582.21（ep199） | 566.39（ep139） |
| RS on | E1（2 头） | **599.89（ep199）** | **589.54（ep79）** |
| Δ | E1 − E0 | **+17.68（更差）** | +23.15 |
| RS off | E2（AGV 规则） | 585.11（ep199） | 572.63（ep89） |
| RS on | E4a（3 头，AGV 规则） | **573.35（ep189）** | **566.22（ep139）** |
| Δ | E4a − E2 | **−11.76（更好）** | −6.41 |
| RS off | E3（AGV actor） | 583.71（ep199） | 571.98（ep149） |
| RS on | E4b（3 头，AGV actor） | **569.83（ep149）** | **564.02（ep89）** |
| Δ | E4b − E3 | **−13.88（更好）** | −7.96 |

注意：

- 这是 **seed1 验证集**，不是 seed3 测试集。
- E1 已跑完 200 ep；E4a 在 189，E4b 在 149。
- E4a / E4b 还会变。
- RS 在无 AGV 时更差；在 AGV 两个臂上暂时更好。
- 最终结论等 seed3 测试评测。

### 判断

3 个头没有阻止 E1 / E4a 坍缩。  
E1：2.485 → 1.30。  
E4a：3.584 → 1.98。  
E4b 目前 2.77，但还没到 200 ep。

下一步：

1. 等 E1 s1 评测。
2. 看最终 action 分布 / argmax（新 checkpoint）。
3. 若仍是固定规则，RS 的结论按负结果写。
4. 10×6 E0 继续观察。

## §104 2026-10-25 22:55 RGA 低值核查：不是 bug

### 核查

用 RGA 的**最优解**本身，做三项检查：

1. 解码器 vs v5 环境：mk / pt / transT / idleT / obj 全部一致（误差 <1e-3）。
2. 独立可行性检查：
   - 每任务一次
   - 机器可用
   - job 先后 + 运输
   - 机器不重叠
   - 独立重算 idleT
   - 实例 0 / 1 / 2 全部 **0 违规**。
3. 解码器的 idleT 与独立重算完全一致。

结论：**RGA 的解码器/实现没有 bug。**  
RGA 找到的调度在 v5 环境下可行。

### 实例 0：RGA vs MIP

| 量 | RGA（no-AGV） | MIP CSV |
|---|---:|---:|
| mk | 448.46 | **377.19** |
| pt（平均） | **471.29** | 486.66 |
| transT | 225.26 | **78.20** |
| idleT | **157.13** | 609.50 |
| obj | **475.81** | 604.98 |

RGA 在 mk / transT 上更差，在 pt / idleT 上更好。  
**idleT 的差距（157 vs 609）决定目标值。**

实例 1 / 2 同样的模式：

- 实例 1：RGA 493.17 vs MIP 629.72；idleT 232 vs 592。
- 实例 2：RGA 588.65 vs MIP 672.00；idleT 423 vs 675。

### MIP 的问题

- CSV 是限时 incumbent，不一定是 v5 目标下的最优。
- CSV 只有指标，没有调度解；不能直接重评。
- MIP 的 `MachineIdleT` 与 v5 的 `idleT` 口径可能不同。
- 论文 Table VI 的 `*` 写「Gurobi optimal」，但那是对 **MIP 自己的模型**。
- 不能把 585.83 当成 v5 目标的下界。

### 结论

- RGA 478.04 不是 bug；它在 v5 口径下有效。
- RGA 赢 MIP，是因为 MIP 的 idleT 高、且不是同模型下界。
- 若要真下界：需要重新解一个 **v5 目标**下的 MIP/CP-SAT，或拿到 MIP 的调度解。
- 当前不能写「RGA 低于 MIP 所以 MIP 错」；只能写「MIP CSV 是限时 incumbent，口径不同」。

## §105 2026-10-25 23:05 新 E3 同策略对照

用 **新重训的 E3 checkpoint**，评测时把 AGV 切回规则：

| seed | E3 actor | E3 + 规则 | 差（actor − rule） |
|---|---:|---:|---:|
| 1 | 558.11 | 565.66 | **−7.55** |
| 2 | 558.35 | 572.02 | **−13.67** |
| 3 | 575.69 | 583.02 | **−7.33** |
| 均值 | **564.05** | **573.57** | **−9.52** |

actor **3/3** 更好。  
新权重的 actor 效应 **9.52**；旧权重是 10.46，方向一致。

### 和 E3 vs E2 的关系

E3 vs E2 是 **2/3**，均值差 3.05。  
原因不是 actor 没效，而是 job/machine 策略方差：

- s3：E3 的 jm 策略 + 规则 = 583.02；E2 s3 = 571.46。  
  E3 的 jm 策略差 11.56；actor 收回 7.33，仍差 4.23。
- s1：E3 的 jm 策略 + 规则 = 565.66；E2 s1 = 568.70。  
  E3 的 jm 策略更好；actor 再收回 7.55，总共低 10.59。
- s2：E3 的 jm 策略 + 规则 = 572.02；E2 s2 = 561.15。  
  E3 的 jm 策略差 10.87；actor 收回 13.67，净低 2.80。

结论：

- **AGV actor 本身是正贡献**：同策略下 3/3，均值 9.52。
- **E3 vs E2 的端到端优势不稳**：2/3，均值 3.05。
- 论文要分开写：
  - actor 消融：同策略 3/3，−9.52。
  - 端到端 E3 vs E2：2/3，−3.05，seed 方差大。
- 不能把端到端写成「3/3 稳定占优」。

## §106 2026-10-25 23:20 RS s1 完成 + 10×6 熵坍缩

### 6×6 RS s1（200 ep，seed1 验证集）

| 臂 | 最后验证 | 最好验证 | 最后熵 |
|---|---:|---:|---:|
| E1（2 头） | 599.89 | 589.54 | 1.302 |
| E4a（3 头，AGV 规则） | 571.23 | 566.22 | 1.533 |
| E4b（3 头，AGV actor） | 564.48 | 564.02 | 1.955 |

和 RS off 比（最后验证）：

| 对比 | RS off | RS on | 差 |
|---|---:|---:|---:|
| 无 AGV（E1 vs E0） | 582.21 | 599.89 | **+17.68（更差）** |
| AGV 规则（E4a vs E2） | 585.11 | 571.23 | **−13.88（更好）** |
| AGV actor（E4b vs E3） | 583.71 | 564.48 | **−19.23（更好）** |

结论不变：RS 在 AGV 两臂为正，在无 AGV 为负。  
熵都坍缩到 1.3–2.0（均匀值是 2.485 / 3.584）。

### 10×6 s1

| 任务 | 最后验证 | episode | 最后熵 |
|---|---:|---:|---:|
| E0 | 593.43 | 139 | — |
| E1（2 头） | 588.53 | 119 | **0.506** |
| E4a（3 头，AGV 规则） | 573.94 | 119 | **0.565** |
| E4b | 未启动 | — | — |

10×6 的 RS 熵坍缩到 **0.5**，比 6×6 更严重。  
顶层基本退化成固定规则。  
验证值暂时不差，但「状态相关规则选择」已经不成立。

### 队列

- 6×6 RS：s1 完成；s2 三个臂在跑；s3 未开始。
- 10×6：E0 / E1 / E4a s1 在跑；E4b s1 等槽位。
- E2/E3 修复：12/12 完成。
- 当前 6 个训练，GPU 11 GB / 99%。
- s1 的 seed3 测试评测还没跑：队列先跑完所有 T 任务，再跑 E 任务。

### 缺口

10×6 队列只有 E0 / E1 / E4a / E4b，**没有 E2 / E3**。  
所以 10×6 上无法单独拆出「RS + AGV 规则 vs 无 RS + AGV 规则」。  
如果要和 6×6 一样做三组 RS on/off，需要补 E2 / E3 10×6。

## §107 2026-10-25 23:25 10×6 补 E2 / E3

`experiment-queue-10x6-e2e3.json`，12 个 job：

- E2 × 3 seeds（AGV 规则，无 RS）。
- E3 × 3 seeds（AGV actor，无 RS）。
- 每个训练带一个 seed3 评测。

23:23 启动首轮：

- E2 s1
- E3 s1
- E2 s2

E2 s3 / E3 s2 / E3 s3 随后。

当前 9 个训练：

- 10×6：E0 / E1 / E4a s1；E2 s1/s2；E3 s1。
- 6×6 RS：E1 / E4a / E4b s2。

GPU：12.5 / 24.6 GB，31%。

目的：补齐 10×6 的 RS on/off：

| 对比 | RS off | RS on |
|---|---|---|
| 无 AGV | E0 | E1 |
| AGV 规则 | E2 | E4a |
| AGV actor | E3 | E4b |

预计 12 个 job / 3 并行 ≈ 6 h，不含评测。

## §108 2026-10-25 23:45 停 RS，先跑非 RS

用户决定：先跑完非 RS，再写论文；RS 以后补。

### 已停

- 6×6 RS 队列 `experiment-queue-rs-heads3-6x6.json`：
  - queue PID 279334
  - s2 子进程 E1 / E4a / E4b
- 10×6 主队列 `experiment-queue-10x6-main.json`：
  - queue PID 281912
  - RS 子进程 E1 s1 / E4a s1
  - **保留 E0 s1**（PID 281915，未杀）

RS s1 的 checkpoint 已保存；s2/s3 停止。  
10×6 RS s1 被停，没有结果。

### 保留 / 新启

- 10×6 E2/E3 队列继续（全非 RS）。
- 新队列 `experiment-queue-10x6-e0-rest.json`：
  - E0 s2 / s3
  - max_parallel 2

当前非 RS 训练 6 个：

- E0 s1 / s2 / s3
- E2 s1 / s2
- E3 s1

待启动 3 个：

- E2 s3
- E3 s2 / s3

GPU：10.7 / 24.6 GB，27%。

### 待办

- E0 s1 评测：原主队列已停，需在 s1 训练结束后手动跑。
- 非 RS 9 个训练全部完成后，再跑 9 个 seed3 评测。
- RS s1 的 seed3 评测也待补（等 RS 恢复）。

## §109 2026-10-26 00:20 10×6 E0 s1 测试

E0 10×6 s1，seed3 100 实例，greedy：

- Obj **579.92 ± 52.63**
- 这是 no-AGV、无 RS、settrans、GRPO、200 ep。

对照 E2E 论文 Table VI（10×6×2，no-AGV）：

| 方法 | Obj |
|---|---:|
| E2E P-G | 674.5 |
| E2E RGA [43] | 653.3 |
| E2E 2SGA [44] | **532.8** |
| 我们的 E0 s1 | **579.92** |

结论：

- 我们的 E0 比 E2E P-G 低 **94.6**。
- 比 E2E RGA 低 **73.4**。
- 比 2SGA 高 **47.1**。
- 和 6×6 一样：赢 P-G / RGA，输 2SGA。

## §110 2026-10-26 01:05 每 phase 目标权重 pilot

不重训。  
用 E0 6×6 s1（no-AGV，settrans，GRPO）的 top1。  
100 个 seed3 测试实例，greedy。  
低层 actor 按 phase 权重条件化；目标仍用 0.4/0.4/0.2 报。

| schedule | 均值 | std |
|---|---:|---:|
| identity（0.4/0.4/0.2 全段） | 550.71 | 75.33 |
| uniform | 551.54 | 76.51 |
| mk_early | 550.97 | 74.27 |
| ec_early | 548.78 | 75.09 |
| mk_sharp | 552.08 | 75.93 |
| **ec_sharp** | **547.16** | 73.88 |
| trans_early | 551.54 | 76.51 |

最好的是 ec_sharp：比 identity 低 **3.55**。  
但配对检验：

- n = 100
- 均值差 **−3.55**
- 差值 std **23.85**
- t = **−1.49**
- p ≈ **0.14**
- 6 win / 3 loss / 91 tie

**不显著。**  
而且只在 9/100 个实例上改变了调度。

结论：

- 低层策略对 phase 权重几乎不敏感。
- ec_sharp 的均值优势来自少数实例。
- 「动态目标权重」没有稳健信号。
- 不要写成论文贡献。
- 不建议再实现学出来的 per-phase λ′ 控制器。

顶层路线到此为止。  
论文回到 C1–C3：双 ST、GRPO、AGV actor。

## §111 2026-10-26 01:10 顶层回到 RS 规则先验

用户决定：顶层回到 **3 头 RS 规则先验**。

- job 4 路
- machine 3 路
- AGV 3 路

λ / 目标权重 agent 放弃：

- 每 phase 权重 pilot 不显著（§110）。
- 低层对 λ 几乎不敏感。
- 不再实现学出来的 λ′ 控制器。

### RS 现状

6×6 s1（seed1 验证集）：

| 对比 | RS off | RS on | 差 |
|---|---:|---:|---:|
| 无 AGV（E1 vs E0） | 582.21 | 599.89 | +17.68（更差） |
| AGV 规则（E4a vs E2） | 585.11 | 571.23 | −13.88（更好） |
| AGV actor（E4b vs E3） | 583.71 | 564.48 | −19.23（更好） |

熵：1.30 / 1.53 / 1.96，已坍缩。

10×6 s1：

- E1 588.53，熵 0.506。
- E4a 573.94，熵 0.565。
- E4b 未跑。
- 10×6 RS s1 的 E1/E4a 在停 RS 时被杀。

### 状态

- 6×6 RS s1 checkpoint 已保存。
- 6×6 RS s2/s3 已停。
- 10×6 RS s1 E1/E4a 已杀，无 checkpoint。
- RS 队列仍停止，等非 RS 跑完再恢复。

### 论文口径

- 顶层 = 3 头 RS 规则先验。
- RS 是**条件性**贡献：
  - AGV 臂为正；
  - 无 AGV 为负；
  - 熵坍缩。
- 如果最终 3 seed 仍是这样，RS 放结果/限制，不作为主贡献。
- 核心贡献仍是 C1–C3。

### 代码

- RS 3 头代码保留。
- `phase_weights` 只是评测诊断，不在方法里。
- 不在 RS 路径上使用。

## §112 2026-10-26 01:35 RS 动作直方图

6×6 s1，100 实例，500 个顶层决策，greedy。

| 臂 | job 头 | machine 头 | AGV 头 |
|---|---|---|---|
| E1 | 500× spt | 500× min_pt | — |
| E4a | 499× spt，1× est | **409× min_trans，91× min_t** | 500× nearest |
| E4b | 500× mwko | 500× min_t | 500× nearest |

marginal 熵：

- E1：job 0，machine 0。
- E4a：job 0.014，machine **0.474**，AGV 0。
- E4b：job 0，machine 0，AGV 0。

E4a 的 machine 头：

- first action：76× min_t，24× min_trans。
- 全段：409× min_trans，91× min_t。
- 说明它在“前段 min_t / 后段 min_trans”之间有一定切换。

其他头全部固定。

### 解释

- 「熵 1.30 / 1.53 / 1.96」是**每步熵**，说明每步很确定。
- marginal 熵 0 说明**跨状态动作也固定**。
- 所以 RS 实际是**固定规则选择器**，不是状态相关策略。
- E4a 的 machine 头有一点状态/阶段相关；job / AGV 固定。

### 重要推论

- E1 固定为 **SPT + min_pt**，比 E0 差 17.68。
- E4a 固定为 **SPT + min_trans（多数）+ nearest**，比 E2 好 13.88。
  - E2 的 AGV 是 earliest-free；E4a 是 nearest。
  - E4a 的收益可能主要来自**AGV 规则从 earliest-free 换成 nearest**，不是顶层自适应。
- E4b 固定为 **mwko + min_t + nearest 先验**，比 E3 好 19.23。
  - 收益可能来自 job/machine 规则或 AGV 先验，不是动态选择。

### 下一步诊断

1. 固定 nearest AGV 规则 + E2 的 job/machine 策略，和 E2（earliest-free）比。
   - 如果 nearest 本身就赢，E4a 的“RS 收益”其实是 AGV 规则收益。
2. 单头实验：固定 job / AGV，只学 machine。
3. 若固定规则已经最优，状态相关策略没有收益；RS 按固定规则选择器写。

### 论文口径

- 不能写「状态相关规则选择」。
- 只能写「学习到的固定规则选择」或「固定规则 + 条件性收益」。
- 顶层不作为主贡献。

### 当前非 RS 进度

| 任务 | episode | 验证 |
|---|---:|---:|
| E0 s2 | 39 | 606.10 |
| E0 s3 | 39 | 619.14 |
| E2 s1 | 49 | 596.64 |
| E2 s2 | 49 | 595.14 |
| E3 s1 | 39 | 628.10 |

- E0 s2/s3、E2 s1/s2、E3 s1 在跑。
- E2 s3、E3 s2/s3 待启动。
- GPU 12.1 / 24.6 GB，80%。
- RS 仍停止。

### Smoke

三条路径都 rc=0：

- E1 2 头，无 AGV：熵 2.483。
- E4a 3 头，顶层 AGV 规则：熵 3.58。
- E4b 3 头，AGV actor + 顶层先验：熵 3.58。

### 下一步

- 等 6×6 RS s2 / s3。
- E2 / E3 修复 s2 / s3。
- 6×6 RS 完成后启动 10×6 RS 队列。
- 最后跑 RS λ / 状态 / 超参扫。

## §113 2026-10-26 02:10 顶层信用分配：实现 + 排队

用户要求：非 RS 实验完成后，测试两种方案。

1. Per-head 局部奖励。
2. COMA 反事实 advantage。

### 实现

`--top_credit {team,local,coma}`，默认 `team`。

`team` 路径与原来一致：

- 共享 team reward；
- 标量 critic；
- 三个头的 log-prob 相加。

`local`：

| 头 | 局部成本 |
|---|---|
| job | `mk + idleT` |
| machine | `mean PEC + transT` |
| AGV | `transT + 累计 AGV wait` |

- `wait` 用环境的 non-loaded delay。
- 它含 deadhead 和跨车间固定延时。
- 跨车间固定延时对所有 AGV 选择是常数，不影响排序。
- deadhead 与 AGV 位置相关，是真实成本。

- 每个头单独算增量奖励。
- 每个头单独归一化。
- 每个头单独 GAE / advantage。
- 每个头单独 value baseline（`local_critic` 输出 3 维）。
- actor loss 按头相加。

`coma`：

- 共享 team reward。
- 新增 joint-action Q-critic。
- 输入 = 18 维状态 + 4/3/3 one-hot 联合动作。
- `A_i = Q(s,a) - Σ_{a_i'} π_i(a_i'|s) Q(s,(a_i',a_-i))`。
- 每个头单独归一化 advantage。
- Q 用 episode 的折扣 return-to-go 训练。
- Q 保留最近 32 个 episode 的 replay。
  - 单局只有 4–7 个决策，Q 太吵。
- 参考 Foerster et al., “Counterfactual Multi-Agent Policy Gradients,” AAAI 2018。

### 环境改动

`parallel_env.py` 新增 `agv_wait_total`：

- 每 episode 在 `_init_agv_states` 清零。
- 每次 AGV 运输累加 `timing["wait"]`。
- 仅 `local` 的 AGV 头使用；team / coma 不受影响。

### 代码改动

- `plcsp_v5/hierarchy/top_ppo.py`
- `plcsp_v5/Run.py`
- `plcsp_v5/parameters.py`
- `plcsp_v5/eval_st_seed3.py`
- `plcsp_v5/trainer/parallel_env.py`
- `plcsp_v5/tests/test_hierarchy.py`

### 验证

- `plcsp_v5/tests`：43 passed。
- 远程 CPU smoke：
  - `local`：rc=0，`hierarchy/update` 正常。
  - `coma`：rc=0，`hierarchy/update` 正常。
- 默认 `team` 路径未改。
- 已在跑的非 RS 任务继续用默认口径。

### 队列

`experiment-queue-credit.json`：

- 臂：E4b（3 头，AGV actor + 顶层先验）。
- 实例：6×6×2。
- 种子：1 / 2 / 3。
- 训练：200 ep。
- 每个训练后接 seed3 评测。
- 共 9 训练 + 9 评测。

| 组 | 顶层 | 目的 |
|---|---|---|
| team | 共享 team reward | 3 seed 对照 |
| local | per-head 局部奖励 | 方案 1 |
| coma | COMA 反事实 | 方案 2 |

- team 三点先跑，作为 3 seed 对照。
- 旧 E4b s1 564.48 只作历史参考。
- AGV 状态特征与 eval trend 已修（§115），不能直接对齐旧值。
- local / coma 随后跑。

启动方式：

- `run_credit_after.sh` 已在远程后台运行。
- 它等待当前所有 `Run.py` / `run_queue.py` 退出。
- 然后启动 `experiment-queue-credit.json`。

### 当前非 RS 进度

- E0 10×6 s2 / s3：训练已结束，评测进行中。
- E2 10×6 s1：训练结束。
- E2 s2、E3 s1、E3 s2：在跑。
- E2 s3、E3 s3：排队。

### 判读口径

- `local` / `coma` 与 E4b team（s1 564.48）直接比。
- 需要 3 seed 同向才算正结果。
- 若仍固定规则，说明信用分配不是瓶颈。
- 若 entropy 仍接近 0，记录为负结果，不移入主表。

### 代码审查修正

独立审查未发现主路径的数学错误。  
已修正以下健壮性问题：

1. `T*B=1` 时 advantage 标准差为 0。
   - 现在 `std <= 1e-8` 时不做归一化，保留原值。
   - 作用于 team / local / COMA。
2. `close_segment` 不校验 batch 维。
   - 现在 team / COMA 要求 `(B,)`。
   - local 要求 `(B,H)`；允许 B=1 的 `(H,)` 和 H=1 的 `(B,)`。
3. `final_cost` 不校验 batch 维。
   - 现在必须是 `(B,)`；B=1 允许标量。
4. `epochs=0` 时 stats 为空。
   - 现在 `epochs = max(1, epochs)`。
5. COMA 允许 `final_cost=None`。
   - 现在直接报错，避免 Q replay 混入未归一化目标。
6. COMA 的 Q replay 不在 checkpoint。
   - `full_ckpt` 现在保存 `hier_top_q_replay`。
   - resume 时恢复到 `top_ctl.q_replay`。
7. `act` 在 local / COMA 返回未训练的 team critic。
   - local 现在返回各头 value 的均值。
   - COMA 现在返回 `Q(s,a)`。

剩余限制：

- COMA replay 的 return-to-go 由旧策略产生，没有重要性修正。
- 顶层策略接近固定，故按“近似 on-policy”处理。
- 若要严格，应做 off-policy 修正或缩短 replay。

### 训练后的动作诊断

`diag_rs_actions.py` 已支持非 team 模式：

```bash
/root/venv/bin/python plcsp_v5/baselines/diag_rs_actions.py \
    --arm E4b --credit local \
    --run-tag plancredit_local_E4b_J6M6E2_s1_remote \
    --limit 100
```

- 对 `coma` 换成 `--credit coma`。
- 看 marginal 熵是否仍为 0。
- 若仍为 0，信用分配没有产生状态相关策略。

## §114 2026-10-26 02:35 源 AGV 边界位置：建模近似（更正）

### 更正

这里原先写成“bug”，不准确。  
`§3` 已明确登记：

- 边界位置没有建模；
- 源 AGV 送到边界后 `location = -1`；
- 下一次取货的空驶时间近似为 0。

所以它是 Stage-0 的**有记录的建模近似**，不是未记录的代码缺陷。  
审查把它标成高置信 bug；我直接采信，判断过重。

### 对之前训练的影响

- 所有 AGV 训练都用这个近似。
- 它们与该模型一致。
- E2 vs E3 的对照仍有效。
- 不能说它们“报废”。
- 如果采用更强的边界模型，旧 AGV 结果与新结果不可混用。
- 论文若用新模型，AGV 臂要重跑。
- 论文若沿用 Stage-0 模型，旧结果可用；只需在口径里写明这个近似。

### 物理含义

- 源 AGV 跨车间后瞬移回取货点。
- 少算一次空驶。
- 调度偏乐观。
- 这是模型精度问题，不是实现错误。

### 本地修复已回退

- 曾本地实现 `src_boundary_machine`。
- 已全部回退。
- 远程与本地都保持 Stage-0 近似。
- 不采用更强边界模型。
- 旧 AGV 结果继续按 Stage-0 口径使用。

### 次要问题（仍未修）

1. 空驶总加在 `max(ready, free_time)` 之后。
   - 若允许 AGV 提前空驶，wait 会被高估。
   - 当前 §54 写明 wait 含空驶，可能是有意的保守口径。
2. `parallel_env` 的 AGV 状态推进发生在 `env.step` 之前。
   - 守卫未检查当前节点已调度。
   - 正常掩码下不触发。
3. `get_agv_decision_batch` / `get_agv_dst_decision_batch` 的 `valid` 未检查前序已调度。
4. 目标车间边界预测漏了 `src_repair`。
   - 仅 `--agv_failure 1` 时触发。
5. `find_max_arrivaTime` 的提前返回依赖 override 存在。
   - 只影响 left-shift 一致性。

## §115 2026-10-26 02:40 RS 状态特征与 eval trend 修复

第三轮审查发现两个问题。  
它们影响所有 `--hier 1 --use_agv 1` 实验。

### 1. AGV 状态特征是常数 0

- `_agv_top_stats` 用 `makespan_previous_step` 当 `now`。
- 它是全任务 horizon，恒大于所有 AGV 的 `free_time`。
- 所以：
  - `busy` 恒为 0；
  - `wait` 恒为 0；
  - 顶层状态第 12 / 13 维是死特征。
- 顶层无法根据 AGV 拥塞做条件。
- 影响 E4a / E4b / 所有 credit 臂。

修法：

- `now` 改成“已调度节点的最大完工时间”。
- 与 `parallel_env.get_agv_decision_batch` 的口径一致。
- 已改 `Run.py` 的 `_agv_top_stats`。
- 已改 `validate.py` 的 `_top_agv_stats`。

### 2. eval 的 cost_trend 恒为 0

- 训练时 trend = `(cost_now - last) / max(cost_now, 1)`。
- 验证时 `_trend = np.zeros(1)`，从不更新。
- 结果：
  - 第一段一致；
  - 后续段的 eval 状态与训练分布不一致；
  - eval 数字不能反映训练到的条件策略。

修法：

- `validate.py` 新增 `_weighted_cost_env`。
- 初始决策记录 `_top_last_cost`。
- 每个顶层段重新计算 trend。
- 与 `Run.py` 同口径。

### 3. 防御性检查

- `hier_interval < 1` 现在直接报错。
- 避免 `% 0` 崩溃。

### 验证

- 本地编译通过。
- 远程 CPU smoke：`local` credit，1 episode，rc=0。
- 远程 `Run.py` / `validate.py` 编译通过。
- 非 RS 任务都是 `hier 0`，不受影响。
- 已上传修复后的代码。

### 影响

- 旧 6×6 RS 读数（§106 / §112）用的是死 AGV 特征 + eval trend 0。
- 那些读数只作历史记录。
- 信用分配实验的 team 对照必须用**新跑的 3 seed**。
- 旧 E4b s1 564.48 不与新结果直接比。
- 修复后 RS 可能不再固定；动作直方图要重跑。

### 仍未修

- §114 的源 AGV 边界近似（已回退修复）。
- resume 时不校验 credit mode / head layout。
- `--use_agv 1 --agv_rule 1 --agv_top_rule 0` 时 AGV 头被采样但不影响动作。



## §116 2026-10-26 07:56 Credit 队列重启

远程 06:36 断开，进程全部退出。

已完成：

- 10×6 非 RS 9/9 训练 + 评测。
- 6×6 RS 受影响，待重跑。

Credit 状态：

- 05:46 启动。
- 只跑了 3 个 team 对照。
- 停在约 60/200。
- 没有 top1，没有评测。
- local / coma 未启动。

处理：

- 3 个部分 run 移到 `trained_model/_partial_ep60/`，未删除。
- 07:56 重启 `experiment-queue-credit.json`。
- 仍是 9 训练 + 9 评测，3 seed。
- 环境保持 Stage-0，§115 的 state / trend 修复保留。

预计：

- 3 个 team 对照约 2.8 小时。
- 全量约 8–9 小时。
- 若只要 s1 筛选，可随时停队列。

## §117 2026-10-26 09:35 10×6 AGV 同策略对照

用 E3 的 10×6 checkpoint，评测时把 AGV 切回规则。

| seed | E3 actor | E3 + 规则 | 差（actor − rule） |
|---|---:|---:|---:|
| 1 | 587.64 | 575.22 | +12.42 |
| 2 | 571.56 | 573.12 | −1.55 |
| 3 | 573.93 | 585.14 | −11.22 |
| 均值 | 577.71 | 577.83 | **−0.12** |

结论：

- 10×6 actor 与规则基本持平。
- 2/3 seed actor 更好，但 s1 差 12.42。
- 6×6 的同策略增益是 −9.52，3/3。
- 10×6 没有复现。

对 E3 vs E2 的解释：

- E3 vs E2 在 10×6 是 +2.86（actor 更差）。
- 同策略对照显示 actor 本身约等于规则。
- 所以端到端差距主要来自 job/machine 策略方差。

论文口径：

- AGV actor 的稳定增益只在 6×6。
- 10×6 只能写“与规则持平”。
- 不能写“跨规模稳定优于规则”。

## §118 2026-10-26 13:15 Hybrid credit（两个机制都加）

用户要求：per-head 局部奖励和 COMA 两个机制都加。

### 做法

不是两个 advantage 相加。

每个头一个局部 Q：

```text
Q_i(s,a) ≈ 第 i 个头的局部回报
A_i = Q_i(s,a) − Σ_{a_i'} π_i(a_i'|s) Q_i(s,(a_i',a_-i))
```

- 局部奖励定义 Q_i 的目标。
- COMA 提供反事实基线。
- 没有新的 advantage 权重 α。

### 代码

- `--top_credit hybrid`。
- `TopLevelActorCritic.q_critic` 输出 H 维。
- `Run.py` 的 hybrid 分支使用局部奖励和 `final_local`。
- `q_replay` 保存 per-head return。
- `full_ckpt` 保存 hybrid 的 Q replay。

测试：

- `plcsp_v5/tests`：**51 passed**。
- `top_ppo.py` / `Run.py` / `parameters.py` 远程编译通过。

### 队列

`experiment-queue-credit-hybrid.json`：

- E4b，6×6×2。
- 3 seed。
- 200 ep。
- 3 训练 + 3 评测。

### 启动策略

先启动过 3 个 hybrid。

- 9 个任务同时跑。
- hybrid 每个只分到约 38% CPU。
- 首 episode 3 分钟没完成。
- 会拖慢 local / coma。

处理：

- 停止 hybrid 队列。
- 代码保留。
- `run_hybrid_after.sh` 等待当前 credit / PPO 队列结束。
- 之后自动启动 hybrid。

### ETA

- local / coma 第一波约 14:20。
- 第二波约 18:00。
- hybrid 约 18:00 启动。
- hybrid 三 seed 约 21:00。
- hybrid 评测约 21:30。

## §119 2026-10-26 16:10 MIP-AGV：CP-SAT 底座 + 两阶段 AGV

### 动机

用户要求：不管进不进论文，先把 MIP-AGV 做出来。

### 实现

`plcsp_v5/baselines/mip_agv/`：

| 文件 | 作用 |
|---|---|
| `model_noagv.py` | CP-SAT：job / machine，no-AGV |
| `validate_env.py` | 成本口径对齐；工序序列导出 |
| `agv_sim.py` | AGV Stage-0 仿真器；和 `Parallel_env` 对齐 |
| `agv_routing.py` | 两阶段：Stage A 定 job/machine，Stage B 解 AGV 路径 |

目标口径：

`0.4*mk + 0.4*(mean PEC + idleT) + 0.2*transT`

### 口径验证

no-AGV：同一调度送入 `Parallel_env`，模型重算四项分量。

- makespan、mean PEC、idleT、transT 差值均为 0.0000。
- objective 409.6773 vs 409.6773。

AGV：PDR 调度 + 最早空闲规则，和 `Parallel_env` 比。

- makespan 521.0160 / 521.0160
- idleT 807.1012 / 807.1011
- transT 191.9256 / 191.9256
- objective 751.3344 / 751.3344

实现要点：

- 左移（left-shift）必须照搬。只追加会差 36.63 的 idle。
- idle 按机器路由的调度顺序累加，不按开始时间排序。

### 结果 1：no-AGV CP-SAT vs 交付的 MIP CSV

6×6，5 个实例，各 20 秒。

| 实例 | CP-SAT | MIP CSV | 差 |
|---|---:|---:|---:|
| 0 | 441.61 | 604.98 | −163.37 |
| 1 | 445.91 | 629.72 | −183.80 |
| 2 | 477.49 | 672.00 | −194.52 |
| 3 | 464.48 | 486.23 | −21.75 |
| 4 | 444.68 | 566.94 | −122.26 |
| 均值 | — | — | **−137.14** |

5/5 更好。实例 0 最好一次 409.68。

结论：

- 交付的 MIP CSV 是弱 incumbent。
- 很多行只跑了约 1 秒；中位约 14 秒。
- 论文不能把它当上限，也不能当强基线。

### 结果 2：两阶段 MIP-AGV

Stage A：CP-SAT job/machine，10 秒。
Stage B：AGV 路径 MILP，8 秒，最多 3 轮，取最好。

| 实例 | 规则 AGV | 两阶段 | 差 |
|---|---:|---:|---:|
| 0 | 427.87 | 427.87 | 0.00 |
| 1 | 510.78 | 510.78 | 0.00 |
| 2 | 489.25 | 481.59 | −7.66 |
| 3 | 486.14 | 482.57 | −3.57 |
| 4 | 507.45 | 507.45 | 0.00 |
| 均值 | — | — | **−2.25** |

5 个实例的两阶段均值：**482.05**。

结论：

- 6×6、6 台 AGV 下，AGV 路径不是主要瓶颈。
- 3/5 个实例上，路径优化没有改善。
- 主要增益来自 job/machine 层。

### 口径边界

- 两阶段不是联合 MILP。Stage B 用 Stage A 的释放时间做代理目标。
- 重放时左移会改变调度；迭代 3 轮缓解，但不保证最优。
- Stage B 的目标是最后一段腿的完成时间之和，不是仿真器的目标。
- AGV 与 no-AGV 的 transT 口径不同，绝对值不能跨表比较。
- 5 个实例的均值和 100 实例的 RGA / DRL 数字不能直接并列。

### 对论文的影响

- 两阶段 MIP-AGV 的 5 实例均值 482.05。
- DRL E3 的 100 实例均值 564.05；RGA AGV 的 100 实例均值 478.04。
- 样本不同，只能作参考。
- 方向一致：质量上 DRL 不是最强，和 §90 的 RGA 结论一致。
- MIP 的绝对值必须重写。旧的 585.83 不能再用。

### 待办

- 两阶段扩到 100 个实例，给出均值。
- 若要真联合 MILP，需要把 AGV 路径和 job/machine 放进同一个模型。
- 联合模型的 6×6 求解时间未知。

### 100 实例完成（2026-10-26 18:10）

- 100/100 解出。
- 均值 505.05，中位 500.66，标准差 72.09，四分位 464.59 / 548.50。
- 规则 AGV 均值 508.38，中位 502.53。
- 路径 MILP 比规则好 50 个，打平 50 个，更差 0 个。
- 平均改善 3.33，最大 21.24。
- 每实例约 12.3 秒。
- 实例 78 无解原因：空闲 AGV 没有回路。
- 已修：允许车库自环。
- 结果文件：`plcsp_v5/results/mip_agv_100/shard{0..3}.csv`。
- 已写入论文 §5 的基线表。

## §120 2026-10-26 17:05 C4 测试：3 seed 全出

### seed3 测试（100 实例，greedy）

| 臂 | s1 | s2 | s3 | 均值 |
|---|---:|---:|---:|---:|
| local | 570.84 | 555.49 | 543.76 | **556.70** |
| RS off | 558.11 | 558.35 | 575.69 | 564.05 |
| coma | 547.60 | 567.12 | 579.07 | 564.60 |
| team | 554.14 | 597.82 | 566.57 | 572.85 |

### 逐 seed 对 RS off

| 臂 | s1 | s2 | s3 | 好 / 差 |
|---|---:|---:|---:|---|
| local | +12.73 | −2.86 | −31.93 | 2 好 1 差 |
| coma | −10.51 | +8.77 | +3.38 | 1 好 2 差 |
| team | −3.97 | +39.47 | −9.12 | 2 好 1 差 |

### 结果

- local 是全表最好：556.70，比 RS off 低 7.35（1.3%）。
- local 2 好 1 差；coma 1 好 2 差；team 2 好 1 差。
- 三个臂都没有 3/3。

### 写法

四个臂全报。  
最好的是 local，但只有 2/3，幅度 1.3%。  
RS 当前不作为主贡献，待补实验后定。

### 状态

- hybrid 3 seed 于 17:06 启动。
- e2e 10×6 s2/s3 在 hybrid 之后。

## §121 2026-10-27 01:40 Hybrid encoder 完成 + E2E+AGV 已启动

### Hybrid encoder（job GIN + machine ST，6×6，GRPO，no-AGV，no-RS）

| seed | obj |
|---|---:|
| 1 | 576.9986 |
| 2 | 580.9906 |
| 3 | 582.6431 |
| 均值 | **580.21** |

2×2 因子（6×6）：

| job | machine | obj |
|---|---|---:|
| ST | ST | 559.31 |
| GNN | ST | 580.21 |
| ST | GNN | 648.81 |
| GNN | GNN | 676.99 |

读数：

- 四格排序在 3 个 seed 内一致。
- 机器编码器是主效应：job 固定 ST 时换机器 ST→GNN 代价 89.50；
  job 固定 GNN 时代价 96.78。
- job 编码器换 GNN 的代价 20.90 / 28.18。
- 交互小。
- 已写入论文 §5 编码器表与 §1 C1。

### 队列状态（01:16）

- `queue_hybrid_encoder.log`：6/6 完成，0 失败。
- 随后自动拉起 `experiment-queue-e2e-agv.json`：3 个 E2E+AGV（6×6，PPO）。
- 10×6 RS 主模型 3 seed 仍在跑（10×6 local credit，seed 1/2/3）。

### 待办

- 10×6 RS 主模型测试完成后填 §5 的 RS 行。
- E2E+AGV 完成后填基线表一行（Obj/Std/Time）。

## §122 2026-10-27 02:10 10×10×2 实验排队

### 配置

`experiment-queue-10x10.json`（已上传 `/root/plcsp_v5/`）：

- 12 个 job：6 训练 + 6 评测。
- 臂：RS（`--top_credit local`）与 no-RS（`--hier 0`），各 3 seed。
- 规模：`--n_job 10 --n_machine 10 --n_edge 2`，`--n_agv 6`。
- 其余口径与 10×6 主模型一致：200 ep，G=8，resample 8，LR 1e-3，
  `--random_weight_type eval`，100 实例评测。
- `max_parallel 6`。
- 实例：`test_Instance_J10M10E2.pkl`（100 个，已有）。

### 状态

- 配置已上传。
- 启动命令已发出，但 SSH 通道超时，未取回启动确认。
- 随后 sshd 返回 `Exceeded MaxStartups`，暂时无法登录。
- **待 SSH 恢复后确认队列是否在跑**；若未启动，重新启动。

### 备注

- 10×6 主模型 s3 在 01:41 已到 ep167/200，预计 02:10 前完成。
- E2E+AGV 3 seed 于 01:16 启动，仍在跑。
- 1050 个 CPU 线程的机器上 load 约 22，容量足够并行。

## §123 2026-10-27 03:10 10×6 RS 完成 + 10×10 OOM

### 10×6 RS 主模型（3 seed，100 实例）

| seed | obj |
|---|---:|
| 1 | 584.04 |
| 2 | 572.71 |
| 3 | 574.00 |
| 均值 | **576.92** |

10×6 四配置：

| 配置 | obj |
|---|---:|
| E0 no RS, no AGV | 575.27 |
| E2 no RS, AGV rule | 574.85 |
| E3 no RS, AGV head | 577.71 |
| RS full | 576.92 |

- 四者极差 2.86，规则选择与 AGV 在 10×6 都不改变目标。
- RS 比 E3 低 0.79，小于 seed 极差（11.33）。

### 10×10 第一批：OOM

- 01:42 启动 12 job（6 训练 + 6 评测），max_parallel 6。
- 5 个训练在 16 分钟内失败：`torch.OutOfMemoryError: CUDA out of memory`。
- GPU 24 GB；同时跑 3 个 10×6 + 3 个 E2E + 6 个 10×10，显存不足。
- 唯一完成的 `nors s1`：训练 5.8 小时，测试 **1034.87**。
- 处理：改成 max_parallel 3，只重跑剩余 5 个训练 + 评测；
  同时排 PDR / RGA / MIP-AGV 基线（CPU）。
- 配置：`experiment-queue-10x10b.json`、`experiment-queue-10x10-baselines.json`。

### E2E+AGV

- 队列 03:25 完成，6/6（3 训练 + 3 评测）。
- 测试均值 **743.62**（695.35 / 696.80 / 838.71），实例 std 126.31。
- 已写入论文表 4。

## §124 2026-10-27 10:05 论文删 AGV 消融 + 10×10 基线完成

### 论文改动（按用户要求）

- 删掉 AGV 头 vs 规则的全部单独呈现：
  - §5.5 的 AGV 头段落、表 10、图（前一轮已删图）。
  - 10×6 表只留 E0 与 RS 两行。
  - §5.6 的“为什么 AGV 头在 6×6 有用、10×6 无用”段落删除。
  - 摘要不再提 AGV 头的规则对照。
- 贡献列表改为定性写法（对齐 E2E / AEI 2025）：
  1 多智能体跨车间生产物流联合调度；2 双 ST 编码器；3 多 actor 组相对更新；4 分层规则选择器。
  贡献里不再有结果数字。
- 摘要去掉结果数字（6×6、10×6、3/3、seed）。
- 删除与表重复的图：编码器热图、GRPO/PPO 柱状、RS 折线、验证曲线。
- 现留 5 张图：图 1 场景、图 2 方法、图 3 分布、图 4 甘特、图 5 规则概率热图。
- 编译：16 页，0 Overfull。

### 10×10 基线（100 实例）

| 方法 | 均值 | std |
|---|---:|---:|
| PDR（mor/spt） | 2552.77 | 252.66 |
| RGA（3 seed） | 1536.27 | 151.86（实例 std 105.44） |

- RGA 用时 336 s。
- 10×10 no-RS s1：1034.87。比 RGA 低 501。
- RS 主模型 3 seed 在跑（09:19 启动）。

### 远程状态

- 运行中：`experiment-queue-10x10b.json`，3 个 RS 训练。
- 无任何 AGV agent vs 规则的任务在跑或在队列里，无需取消。
- 10×10 PDR / RGA 队列已完成（2/2）。

## §125 2026-10-27 11:20 主表按实例族改版

- 用户澄清“按实例”= 6×6×2 / 10×6×2 / 10×10×2 三个实例族，不是 100 个单实例。
- 表 4 改为：行 = 实例族，列 = PDR / B3 / E2E / Ours / MIP / RGA。
  - 6×6：951.20 / 707.21 / 743.62 / 556.70 / 505.05 / 478.04
  - 10×6：1101.30 / 651.30 / -- / 576.92 / -- / 565.63
  - 10×10：2552.77 / -- / -- / 待 RS / -- / 1536.27
- 表 5 仍是 6×6 的 12 条 PDR 明细；E2E Table V 覆盖全部规模，后续可扩。
- 方法论名去掉 AGV 后缀：E2E / B3 / MIP。图 3 横轴同步。
- 论文 16 页，0 Overfull。

### 缺的数据

- 10×10 RS 主模型（3 seed 在跑）。
- 10×10 PDR 的 12 对明细（远端 `agv_pdr_100_10x10.json`，SSH 暂不可取）。
- 10×10 RGA 的逐实例 std。

## §126 2026-10-27 12:10 补齐 Table 4 的空格

### 已上传并启动

- `baselines/mip_agv/` 整个包 + `run_mip_batch.py` 启动器。
- `experiment-queue-fill-sizes.json`：B3 10×10、MIP-AGV 10×6（4 shard）、MIP-AGV 10×10（4 shard）。
- `experiment-queue-e2e-sizes.json`：E2E（PPO + 规则派车）10×6 与 10×10，各 3 seed + 评测。
- `launch_e2e_after_10x10.sh`：等 `queue_10x10b.log` 出现 queue done 后自动启动 E2E 队列。

### 为什么要等

- 10×10 RS 三个训练占 19.5 / 24 GB 显存。
- 直接加 E2E 会重演 CUDA OOM。

### 论文

- 结论改为定性，无结果数字。
- 删除规则概率热图（跨栏浮动导致结论页标题下空白）；数字保留在正文。
- `\needspace` 防止标题落在页底。
- 16 页，0 Overfull。


