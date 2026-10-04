"""策略网络——机台选择（S 层）+ AGV 派车（L 层）+ **路线选择（R 层）** + **维护时机（M 层）**四个头。

**无 critic**：组内相对优势用组内基线（见 `group_rel.py`），不需要价值网络。
2026-10-02：分批（B 层）头已删（分批环节砍除）；2026-10-03：critic 头 `v_head` 随 PPO 变体一并删除。
2026-10-03（P2 Task 4）：**两个头一律走编码器 token 嵌入**——L 头不再吃 `des.py` 手搓的
扁平向量（旧路径 `l_head` / `agv_logits` 与 S 头的 MLP 回退 `s_head` / `mach_logits` 一并删除）。
此前 L 头**不走编码器**，对生产侧结构性失明（看不到机台状态/计划/布局，`n_agv≥3` 时看不到
2 号以后的车）——这也解释了"L 在随机计划下无信号"的实测（见 spec §5.3.1 #4）。
2026-10-04：**R 头（`route_logits_emb`）恢复**——`route_logits` 当初随 ① 拥堵一并被砍
（"无拥堵时选远路严格更差"，spec §5.3），但 ① 后来在 `eb1d1da` 恢复而路线头漏恢复
（`docs/progress-log.md` §27.3/§28）。R 头是那次遗漏的补建，不是新发明。
2026-10-04（⑫ 维护头）：**M 头（`pm_logits_emb`）新增**——把"何时停机保养"从
`MachineSim` 的自动规则（`pm_clock >= pm_interval`）交还策略：候选 = {现在保养, 不保养}，
两个候选同属**一台**机台（候选是**动作**不是实体，故 `cand_idx` 两个候选共用该机台的
M token，见该方法的说明）。默认关闭，逐位等于今日行为。
2026-10-04（⑪ 充电头）：**C 头（`charge_logits_emb`）新增**——把"去哪充 / 充不充"从
`AgvSim._maybe_charge` 的规则（低电 → 最近**空闲**桩）交还策略：候选 =
{不去充} ∪ {各充电桩}，**变长候选集**（随布局桩数变）。⚠️ 充电桩在 token 序列里**没有
token**（序列只有 M/B/V/G 四段），"不去充"更不是实体——故全部候选共用**本车**的 V token
（与 R 头"候选是路径、广播本车 V token"同型，见该方法）。默认关闭，逐位等于今日行为。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..nn.encoder import LayoutEncoder


def v_token_index(seg: tuple[int, int, int, int]) -> list[int]:
    """V 段 token 在序列中的位置（= M 段 + B 段之后）。有序，第 i 个 = 第 i 台车。"""
    n_m, n_b, n_v, _ = seg
    return list(range(n_m + n_b, n_m + n_b + n_v))


def _to_dev(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """把打分输入搬到 `like`（token 嵌入）所在设备——五个头共用的**唯一搬运点**。

    见 `PolicyNet.device`：策略可以整体跑在 CUDA 上（`m13_train_a --device cuda`），而决策
    记录里的 token/特征一律是 **CPU numpy**（`roll_chain` 采样时存的），重算时在
    `_decision_logp_terms` 统一搬到参数设备。头里的这一层是**兜底**：外部调用方（测试、
    诊断脚本）手搓张量时不至于因设备不匹配而报错——`.to()` 在同设备时是 no-op。
    """
    return t.to(like.device)


class PolicyNet(nn.Module):
    """π = π_S(机台候选) · π_L(AGV 派车) · π_R(路线候选) · π_M(⑫ 何时保养) · π_C(⑪ 充电)；无 critic。

    五个头读**同一份** token 嵌入（spec §5.3.1）：S 头 `mach_logits_emb` / L 头
    `agv_logits_emb` / R 头 `route_logits_emb` / M 头 `pm_logits_emb` / C 头
    `charge_logits_emb`，嵌入均由 `forward_enc` 产出。
    ⚠️ `enc=None` 时**没有可用的头**——旧的扁平特征 MLP 回退路径（`s_head` / `mach_logits`）
    已在 P2 Task 4 删除，不存在第二条打分通路。

    五个头的打分输入同构：`token 嵌入 ⊕ 决策特征 ⊕ **候选特征**`。
    - **决策特征**（`feat_op` / `feat_task` / `feat_route` / `feat_mach` / `feat_agv`，形参
      `n_feat_op` / `n_feat_task` / `n_feat_route` / `n_feat_pm` / `n_feat_charge`）与候选
      **无关**，broadcast 给所有候选；
    - **候选特征**（`feat_cand`，形参 `n_feat_cand` / `n_feat_route_cand` / `n_feat_pm_cand` /
      `n_feat_charge_cand`）**逐候选**——S 头放换型代价 `setup(prev_job_of_m, j)`、L 头放"该车
      到取货点的预计行驶时长"、R 头放"该路径的长度比 / 区段数 / 当前争用"、M 头放两个**动作**的
      后果（现在保养 vs 不保养的停机余量/代价/进度）、C 头放"到该桩的行驶时长 / 该桩占用排队 /
      是不是不去充"。spec §5.3.1②：换型是 `(机台, 作业)` 的**交互量**，塞不进 M 段 token，
      只能走这个槽（旧 MLP 路径本有 `feat_cand`，重写时不可丢）；R 头同理，候选是路径，
      在序列里没有 token（见 `route_logits_emb`）；M 头候选是**动作码**，两次候选同属一台机台
      （见 `pm_logits_emb`）；C 头候选是**动作码 + 桩**，桩同样没有 token（见
      `charge_logits_emb`）。

    ⚠️ **无 `n_agv` 形参**（评审 M-4 删）：车队规模由 `seg` 的 V 段长度定（`v_token_index`），
    网络结构里没有任何一处随车队规模变——旧的 `n_agv` 形参与其 `self.n_agv` 属性**全仓零
    读取方**，只会给读者"车队规模进网络"的错觉（要理解车队规模如何进网，看 V 段 token）。
    ⚠️ **候选集变长不进网络结构**：五个头的输出长度由 `cand_idx` 的长度定，与权重形状无关
    （C 头的桩数、R 头的 k、S 头的候选机台数都是运行期量）。
    """
    def __init__(self, n_feat_op: int = 3,
                 hidden: int = 64, enc: LayoutEncoder | None = None,
                 n_feat_task: int = 4, n_feat_cand: int = 1,
                 n_feat_route: int = 4, n_feat_route_cand: int = 3,
                 n_feat_pm: int = 3, n_feat_pm_cand: int = 3,
                 n_feat_charge: int = 3, n_feat_charge_cand: int = 3):
        super().__init__()
        self.enc = enc
        self.optim: torch.optim.Optimizer | None = None   # 由训练器在首步惰性创建（Adam）
        if enc is not None:
            self.s_head_tok = nn.Sequential(          # 编码器 token 打分头（候选机台）
                nn.Linear(enc.d_model + n_feat_op + n_feat_cand, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
            # 编码器 token 打分头（候选车辆）：任务特征 = 起送机台/目标机台/工序序号/该机是否需换型
            self.l_head_tok = nn.Sequential(
                nn.Linear(enc.d_model + n_feat_task + n_feat_cand, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
            # 编码器打分头（候选路径，R）：行驶特征 = 起点/终点节点/是否负载/本车号；
            # 逐候选特征 = 长度比/区段数/争用（见 `route_logits_emb` 的"候选没有 token"说明）
            self.r_head_tok = nn.Sequential(
                nn.Linear(enc.d_model + n_feat_route + n_feat_route_cand, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
            # 编码器打分头（⑫ 维护，M）：决策特征 = 该机台状态摘要；逐候选特征 =
            # {现在保养, 不保养} 两个**动作**的后果（见 `pm_logits_emb`）。
            # ⚠️ 建在最后：既有三头的初始化抽签次序不得变（默认关闭档的黄金摘要靠它）。
            self.pm_head_tok = nn.Sequential(
                nn.Linear(enc.d_model + n_feat_pm + n_feat_pm_cand, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
            # 编码器打分头（⑪ 充电，C）：决策特征 = 本车状态摘要（电量/待办/进度）；
            # 逐候选特征 = {不去充} 与各桩的"行驶时长 / 占用排队 / 是不是不去充"
            # （见 `charge_logits_emb` 的"桩没有 token"说明）。
            # ⚠️ **必须建在 `pm_head_tok` 之后**：加新头不得改变既有头的初始化抽签次序
            # （黄金摘要含策略参数的随机初值，M 头当年也是为此建在最后）。
            self.c_head_tok = nn.Sequential(
                nn.Linear(enc.d_model + n_feat_charge + n_feat_charge_cand, hidden), nn.GELU(),
                nn.Linear(hidden, 1))

    def forward_enc(self, tok_feat: torch.Tensor | np.ndarray,
                    seg: tuple[int, int, int, int]
                    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """编码器前向（**五个**头共用）——**唯一的 numpy→torch 转换点**；无编码器返回 (None, None)。

        `build_tok`（Task 2）产 numpy `(N, F_MAX)`，编码器要 torch **(B, N, F_MAX)**：
        numpy/列表 → float32 张量，2 维 → 补 batch 维，**最后搬到参数设备**，在此**一处**统一
        （其余调用方只传 torch）。
        ⚠️ **三维即原样透传**（2026-10-04 批量重算批次）：单条在线决策传 `(1,N,F)`，重算路径
        把整组决策堆成 `(B,N,F)` 一次前向——`B` 由调用方决定，本函数不做任何跨批次的合并。
        ⚠️ **设备跟随参数**（2026-10-04 设备批次）：不硬编码 cpu/cuda——`--device cuda` 时在线
        前向与批重算都在 CUDA 上；`tok_feat` 是 CPU numpy，故这里必须搬（`_to_dev` 与之同理）。
        """
        if self.enc is None:
            return None, None
        x = (tok_feat.float() if torch.is_tensor(tok_feat)
             else torch.tensor(np.asarray(tok_feat, dtype=np.float32)))  # 复制：不共享上游 numpy 内存
        if x.dim() == 2:                                    # (N, F_MAX) → (1, N, F_MAX)
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"forward_enc 要 (N,F)、(1,N,F) 或 (B,N,F) 的 token 特征，"
                             f"收到 shape={tuple(x.shape)}")
        return self.enc(x.to(self.device), seg)

    @property
    def device(self) -> torch.device:
        """参数所在设备（**唯一真相**）——重算/采样一律跟随它，不硬编码 cpu/cuda。

        无参数的退化策略（`enc=None` 且未建头）回落到 cpu：那种策略没有任何张量语义，
        调用方本就走不到前向。
        """
        p = next(self.parameters(), None)
        return p.device if p is not None else torch.device("cpu")

    def mach_logits_emb(self, tok: torch.Tensor, feat_op: torch.Tensor,
                        feat_cand: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        """(1,N,d) × (1,1,F_op) × (1,n_cand,F_cand) × (n_cand,) → (1,1,n_cand) 候选分数（M3c）。

        M 段位于序列前部：cand_idx = 机台编号即序列位置（B/V 段追加于后）。
        `feat_op` 与候选**无关**（broadcast 给所有候选）；`feat_cand` **逐候选**
        （也收 `(n_cand, F_cand)`）——⑤ 换型代价是 `(机台, 作业)` 的交互量，只能走这个槽
        （spec §5.3.1②），塞不进 M token。
        """
        tok_c = tok[0, _to_dev(cand_idx, tok).long()]        # (n_cand, d)
        op = _to_dev(feat_op.expand(1, tok_c.shape[0], -1)[0], tok)   # (n_cand, F_op)
        cand = _to_dev(feat_cand[0] if feat_cand.dim() == 3 else feat_cand, tok)  # (n_cand,F_cand)
        return self.s_head_tok(torch.cat([tok_c, op, cand], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)

    def agv_logits_emb(self, tok: torch.Tensor, feat_task: torch.Tensor,
                       feat_cand: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        """(1,N,d) × (1,1,F_task) × (1,n_cand,F_cand) × (n_cand,) → (1,1,n_cand)。

        与 `mach_logits_emb` **对称**：候选对象的 token 嵌入 ⊕ 决策特征 ⊕ **候选特征** → 打分。
        `feat_cand` = "该车到取货点的预计行驶时长"（spec §5.3.1 V 段坐标维的既定用途：
        派最近的车）——没有这个槽，L 头落不了该语义。
        2026-10-03 新增：此前 L 头吃 `des.py` 手搓的 11 维扁平向量、**不走编码器**，
        导致它对生产侧结构性失明（看不到机台状态/计划/布局，n_agv≥3 时看不到 2 号以后的车）。
        """
        tok_c = tok[0, _to_dev(cand_idx, tok).long()]                 # (n_cand, d)
        ft = _to_dev(feat_task.expand(1, tok_c.shape[0], -1)[0], tok)  # (n_cand, F_task)
        cand = _to_dev(feat_cand[0] if feat_cand.dim() == 3 else feat_cand, tok)  # (n_cand,F_cand)
        return self.l_head_tok(torch.cat([tok_c, ft, cand], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)

    def route_logits_emb(self, tok: torch.Tensor, feat_drive: torch.Tensor,
                         feat_cand: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        """(1,N,d) × (1,1,F_route) × (1,k,F_cand) × (k,) → (1,1,k) 候选**路径**分数（R 头）。

        ⚠️ **与 S/L 两头的关键差别——路线候选在 token 序列里没有自己的 token**：
        序列只有 M（机台）/B（作业）/V（车辆）/G（全局）四段，候选是**路径/区段实体**，
        不是这三种实体中的任何一种（给区段加 token 是另一条机制 R2，不在本次恢复范围）。
        故本头的 `cand_idx` 不逐候选区分，它取**本车自己的 V token 下标**（k 个候选传同一个值）：
        - 语义 = "这辆车在做什么决定"，与 L 头（车辆实体）同源；
        - 作用 = 上下文 + 到编码器的梯度通路（R 头不是脱离编码器的第二条打分通路）；
        - **候选之间的分数差只能来自 `feat_cand`**（逐候选的长度比/区段数/当前争用）——
          `tok` 与 `feat_drive` 对 k 个候选是同一份输入，不携带候选间差异。
        ⚠️ **已知限度**（如实记下，不许含糊）：本头的"偏好哪条路"因此基本由逐候选特征决定，
        上下文只经 GELU 的非线性调节对各维的敏感度；要表达"同一辆车在不同状态下偏好不同路"，
        须等 R2 给区段/路径加 token（`progress-log.md` §27.2 的配对项）。
        """
        cand = _to_dev(feat_cand[0] if feat_cand.dim() == 3 else feat_cand, tok)  # (k,F_cand)
        tok_c = tok[0, _to_dev(cand_idx, tok).long()]                  # (k, d)
        fd = _to_dev(feat_drive.expand(1, tok_c.shape[0], -1)[0], tok)  # (k, F_route)
        return self.r_head_tok(torch.cat([tok_c, fd, cand], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)

    def pm_logits_emb(self, tok: torch.Tensor, feat_mach: torch.Tensor,
                      feat_cand: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        """(1,N,d) × (1,1,F_dec) × (2,F_cand) × (2,) → (1,1,2) 候选**动作**分数（⑫ 维护头，M）。

        ⚠️ **与 S/L 的关键差别——候选不是实体，是同一台机台的两个动作**：
        {0 = 现在保养, 1 = 不保养}（动作码见 `des.PM_CANDS`）。故：
        - `cand_idx` 对两个候选**取同一个下标** = 该机台的 M 段 token（机台号 = 序列位置，
          与 S 头同源）；**不是** `cand`（0/1 是动作码，拿去索引会读到 0/1 号机台的 token
          ——维护头就会给别的机器打分，与 2026-10-04 修的 L 头缺陷同型，见 progress-log §31）；
        - 候选之间的分数差只能来自 `feat_cand`：两个动作的"距停机余量 / 停机代价 / 进度"
          逐行不同（见 `group_rel._pm_cand_feat`）。若两行相同，两个候选的分数**恒等**，
          决策退化成不可学的掷硬币。

        `feat_mach` = 该机台的状态摘要（与候选无关，broadcast 给两个动作）。
        """
        tok_c = tok[0, _to_dev(cand_idx, tok).long()]                  # (2, d)
        fm = _to_dev(feat_mach.expand(1, tok_c.shape[0], -1)[0], tok)  # (2, F_dec)
        cand = _to_dev(feat_cand[0] if feat_cand.dim() == 3 else feat_cand, tok)  # (2, F_cand)
        return self.pm_head_tok(torch.cat([tok_c, fm, cand], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)

    def charge_logits_emb(self, tok: torch.Tensor, feat_agv: torch.Tensor,
                          feat_cand: torch.Tensor, cand_idx: torch.Tensor) -> torch.Tensor:
        """(1,N,d) × (1,1,F_dec) × (m,F_cand) × (m,) → (1,1,m) 候选**动作**分数（⑪ 充电头，C）。

        ⚠️ **与 R 头同型的关键差别——候选在 token 序列里没有 token**：候选 = {不去充} ∪
        {各充电桩}（动作码见 `des.charge_cands`），而序列只有 M/B/V/G 四段——充电桩不是其中
        任何一种实体，"不去充"更不是。故本头的 `cand_idx` **不逐候选区分**，它取**决定方那台车
        自己的 V token 下标**（m 个候选传同一个值）：
        - 语义 = "这辆车在做什么决定"，与 L 头（车辆实体）同源；
        - 作用 = 上下文 + 到编码器的梯度通路（C 头不是脱离编码器的第二条打分通路）；
        - **候选之间的分数差只能来自 `feat_cand`**（逐候选的"到该桩的行驶时长 / 该桩占用排队 /
          是不是不去充"）——`tok` 与 `feat_agv` 对 m 个候选是同一份输入，不携带候选间差异。
        ⚠️ **守卫后果（必须写明）**：正因为 m 个候选共用**同一个** token 下标，
        "扰动本车 V token ⟹ 分数必变"这条判据**不是恒真**——若头的第一层落在 GELU 的线性区，
        所有候选的 logit 会被同一个量平移，`log_softmax` 对其**完全不变**（同一候选行加同一
        常数 = 无变化）。故守卫（`test_charge_head.test_c_head_reads_the_deciding_agvs_token_
        on_the_production_path`）用**逐 token 恒等嵌入**切断注意力泄漏并让扰动逐列不同，
        且必须**实地跑一遍错误实现**确认判据会失败（§31 的教训：恒真的守卫等于没有守卫）。
        ⚠️ **已知限度**（如实记下）：与 R 头相同——"同一辆车在不同状态下偏好不同桩"只能由
        `feat_cand` + 上下文的非线性调节表达；要给桩加 token 是另一条机制（本任务不在范围）。

        `feat_agv` = 本车状态摘要（电量占比 / 待办任务占比 / episode 进度，与候选无关）。
        """
        tok_c = tok[0, _to_dev(cand_idx, tok).long()]                  # (m, d)
        fa = _to_dev(feat_agv.expand(1, tok_c.shape[0], -1)[0], tok)   # (m, F_dec)
        cand = _to_dev(feat_cand[0] if feat_cand.dim() == 3 else feat_cand, tok)  # (m, F_cand)
        return self.c_head_tok(torch.cat([tok_c, fa, cand], dim=-1)).squeeze(-1).unsqueeze(0).unsqueeze(1)
