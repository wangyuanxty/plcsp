"""M3 长训练 runner（M5 前置工程）：训练循环 + checkpoint 保存/恢复。

为什么需要：SA-GRPO 长训练（G/J 扫描 × 种子矩阵，见《方法设计文档》§5 消融⑨）单次数小时；
中断必须可续（教训：fjsp-gnnrl MK01 训练首次中断于 Episode 2399/10k，无 ckpt → 只能重跑）。

设计（KISS/工程层，不混淆理论）：
- 每步仍调用纯函数（`joint_chain_step`，P2 Task 7 起），runner 只负责循环/保存/恢复；
- ckpt = {"step": int, "model": state_dict, "best": float, "seed0": int}
  每 save_every 步写 checkpoints/<run_id>/ckpt.pt；metrics.ndjson 每步追加一行（增量，可断外部分析）；
- 恢复：resume_training(run_dir) → (policy, ckpt, step)；step 序列续跑（不追求逐位复现：
  ckpt 只存模型权重与步号、**不还原 Adam 状态**，续跑以当次 lr 重建优化器 ⇒ "续跑段"与
  "一次跑完"不同。单次连续运行本身是**逐位可复现**的（动作采样自评审 I-3 起由 seed 派生）。

用法：
    run_id = f"{algo}-G{G}-s{seed}"
    run_training(policy, inst, steps=200, step_fn=joint_chain_step, seed0=0,
                 step_kwargs=dict(layout=lay, dm=dm, cfg=cfg, ctx=ctx, ref=ref, G=8),
                 run_dir="checkpoints/" + run_id)
    policy, ck, _ = resume_training("checkpoints/" + run_id)

⚠️ `step_fn` 默认已是联合链（`joint_chain_step`）——它的**环境参数**（布局 / 距离矩阵 /
`SimConfig` / `NormContext` / 参考三目标 `ref`）一律经 `step_kwargs` 传入（runner 不自己造环境）。
⚠️ `ref` 必须是 `ReferenceObjectives.of(inst, cfg)`（与当前 cfg 同源）——`joint_chain_step`
入口校验指纹，不同源直接报错（评审 F5）。
⚠️ 训练用的 `PolicyNet` **必须带编码器**（`enc=LayoutEncoder()`）：两个打分头只在
`enc is not None` 时存在，无编码器的 policy 调 `joint_chain_step` 会 AttributeError。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .group_rel import joint_chain_step
from .policy import PolicyNet
from ..nn.encoder import LayoutEncoder
from ..env.instances import Instance

CKPT_NAME = "ckpt.pt"
METRICS_NAME = "metrics.ndjson"


def run_training(policy: PolicyNet, inst: Instance, steps: int,
                 step_fn=joint_chain_step, step_kwargs: dict | None = None,
                 seed0: int = 0, run_dir: str = "checkpoints/run_0",
                 save_every: int = 10, resume: bool = False,
                 eval_fn=None, eval_every: int | None = None) -> dict:
    """训练循环+保存。step_fn(policy, inst, seed=..., **step_kwargs) → (r_mean, diag)。

    resume=True：run_dir 存在 ckpt.pt → 恢复续跑（覆盖 policy 参数与起始 step）。
    eval_fn(policy) → dict（可选）；eval_every 步调用并追加到 metrics.jsonl（{"step","eval",...}）。
    返回最终 {"step", "best", "r_last"}。
    """
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    step_kwargs = dict(step_kwargs or {})
    start = 0
    best = 1e9
    if resume and (rd / CKPT_NAME).exists():
        ck = torch.load(rd / CKPT_NAME, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["model"])
        start, best = int(ck["step"]), float(ck["best"])
        print(f"[runner] resumed: step={start} best={best}")
    mf = open(rd / METRICS_NAME, "a", encoding="utf-8")
    t0 = time.time()
    try:
        for s in range(start, start + steps):
            r, diag = step_fn(policy, inst, seed=seed0 + s, **step_kwargs)
            rec = {"step": s, "r": float(r), "t": time.time() - t0,
                   **{k: v for k, v in diag.items() if isinstance(v, (int, float, str))}}
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            if eval_fn and eval_every and (s + 1) % eval_every == 0:
                ev = eval_fn(policy)                     # {"key": val, ...}
                rec2 = {"step": s + 1, "eval": ev}
                mf.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                mf.flush()
                print(f"[runner] step {s+1} eval={ev}", flush=True)
            if (s + 1) % save_every == 0 or s == start + steps - 1:
                torch.save({"step": s + 1, "model": policy.state_dict(),
                            "best": best, "seed0": seed0},
                           rd / CKPT_NAME)
                print(f"[runner] {s+1}/{start+steps} r={r:.2f} saved")
    finally:
        mf.close()
    return {"step": start + steps, "best": best, "r_last": float(r)}


def resume_training(run_dir: str) -> tuple[PolicyNet, dict, int]:
    """从 run_dir 恢复：返回 (policy(参数已加载), ckpt dict, step)。

    ckpt 键含 enc.* → 自动构造 PolicyNet(enc=LayoutEncoder()) 恢复。

    ⚠️ **前缀检查的语义（Task 7 同步）**：`enc.*` 是"这个 ckpt 能不能继续**训练**"的判据——
    P2 起两个打分头只在 `enc is not None` 时存在，无 `enc.*` 的 ckpt 只能当推理用的裸
    policy（`joint_chain_step` 会用 AttributeError 拒收，不会静默退化）。

    ⚠️ 2026-10-02：**P0 前的旧 checkpoint 全部作废**——`checkpoints/` 下 19 个 `.pt` 均含已删除的
    `b_head.*`（分批头）键，且训练于 bug#12 的错误实例（非官方 Brandimarte）。加载会明确报错。
    """
    rd = Path(run_dir)
    ck = torch.load(rd / CKPT_NAME, map_location="cpu", weights_only=False)
    pol = PolicyNet(enc=LayoutEncoder() if any(k.startswith("enc.") for k in ck["model"]) else None)
    try:
        pol.load_state_dict(ck["model"])
    except RuntimeError as e:
        raise RuntimeError(
            "checkpoint 与当前 PolicyNet 不兼容（P0 已删除 batch 头 b_head.*）。"
            "checkpoints/ 下 19 个旧 .pt 全部作废——它们训练于 bug#12 的错误实例，"
            "且含已删除的 batch 头。请勿复用。"
        ) from e
    return pol, ck, int(ck["step"])


if __name__ == "__main__":
    # 自检：跑 5 步 → ckpt；模拟恢复进程 → 续 3 步（验证保存/恢复管线）
    import shutil
    import tempfile
    from .setup import build_setup
    from ..env.des import SimConfig
    from ..env.instances import gen_random
    from ..env.reward import ReferenceObjectives
    inst = gen_random(4, 3, seed=0)
    cfg = SimConfig()
    lay, dm, ctx = build_setup(inst, cfg)   # ⚠️ 环境三件套唯一入口（评审 F4）
    ref = ReferenceObjectives.of(inst, cfg)  # ref 进训练入口（评审 F5：w 由 ref 派生）
    pol = PolicyNet(enc=LayoutEncoder())                           # 训练必须带编码器
    kw = dict(layout=lay, dm=dm, cfg=cfg, ctx=ctx, ref=ref, G=4)
    d = Path(tempfile.mkdtemp()) / "run_selfcheck"
    run_training(pol, inst, steps=5, step_fn=joint_chain_step, step_kwargs=kw,
                 seed0=0, run_dir=str(d), save_every=2)
    assert (d / CKPT_NAME).exists(), "ckpt 未写出"
    pol2, ck2, s2 = resume_training(str(d))
    r2 = run_training(pol2, inst, steps=3, step_fn=joint_chain_step, step_kwargs=kw,
                      seed0=0, run_dir=str(d), save_every=2, resume=True)
    print(f"[runner-selfcheck] resumed step={s2} → final={r2['step']} r_last={r2['r_last']:.2f}")
    shutil.rmtree(Path(d).parent, ignore_errors=True)
    print("RUNNER-SELFCHECK-OK")
