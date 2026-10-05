# -*- coding: utf-8 -*-
"""`m17_reeval` 的**口径钉**：日志解析 → cfg / 链级开关**逐字相同**。

## 为什么单独一个测试文件

重评脚本的全部风险集中在一处：**它自己重建口径**。手写开关表、或解析漏一个键，都会让
重评跑的不是训练时那个策略，而且**不报错**（读数看起来完全正常）。故本文件钉三件事：

1. **解析**：日志两行（开关行 + 约束组行）+ 首行（`inst/steps/seed/route_k`）里的每一个值
   都进 `RunSpec`；缺行、缺键、组名查不到 ⟹ 显式报错（**不猜默认值**）。
2. **重建**：`RunSpec.cfg()` / `RunSpec.chain_kwargs()` 与实跑参数一一对应。
3. **透传**：`m17` 建出的评估回调把**每一个**链级开关与**评估种子**（`EVAL_SEED_BASE + s`）
   送进 `roll_chain`——这一条走 `m13._make_eval_fn`（复用，不另写一份），故这里同时钉住了
   "重评与训练内联评估是同一个口径"。

⚠️ 本文件**不训练、不跑仿真**：替身 `roll_chain` 只记录 kwargs。
"""
from __future__ import annotations

import json

import pytest

from plcsp import m13_train_a as m13
from plcsp import m17_reeval as m17
from plcsp.env.constraints import ABLATION_GROUPS
from plcsp.env.des import SimConfig
from plcsp.env.instances import load_mk

# 逐字复制 `m13_train_a.main` 的打印格式（`full` run，见 D:/Temp/phase1/logs/full.log）。
_META_FULL = (
    "[m13] inst=mk01 作业10×机台6 车队3｜steps=300 G=8 lr=0.0003 seed=0｜route_k=2｜"
    "pm_head=True｜device=cuda:0｜parallel=True(workers=8,worker_device=cpu)")
_SW_FULL = (
    "[m13] 开关（训练=评估，同源）：route_zones=True geom_bias=True pm_head=True "
    "charge_head=True batch_head=True multi_drop=True agv_failover=True "
    "machine_age_failure=True t3=True")
_GRP_FULL = (
    "[m13] 约束组=Full｜优势口径=scalar（这两项不在上面的同源清单里——评估不跑优势，"
    "约束组由 build_training_setup 三处同源）")


def _log(meta: str = _META_FULL, sw: str = _SW_FULL, grp: str = _GRP_FULL) -> str:
    return "\n".join([meta, sw, grp, "[runner] 10/300 r=-113.15 saved"]) + "\n"


def test_parse_full_arm_spec_is_verbatim():
    """`full` run：九个开关全 True、约束组 Full、route_k=2、种子/步数逐字读出。"""
    spec = m17.parse_run_spec(_log())

    assert (spec.inst, spec.steps, spec.seed, spec.route_k) == ("mk01", 300, 0, 2)
    assert spec.constraint_group == "Full" and spec.adv_mode == "scalar"
    for key in m17._SWITCH_KEYS:
        assert getattr(spec, key) is True, f"{key} 没被解析成 True"


def test_parse_minus_production_arm():
    """`-生产` run：守卫强制的连带项（pm/役龄关）必须**如实**解析出来。

    ⚠️ 若把 pm_head 解析成 True，重评会调一个训练时根本不存在的维护头 ⟹ 用另一个策略评估。
    """
    sw = ("[m13] 开关（训练=评估，同源）：route_zones=True geom_bias=True pm_head=False "
          "charge_head=True batch_head=True multi_drop=True agv_failover=True "
          "machine_age_failure=False t3=True")
    grp = ("[m13] 约束组=-生产｜优势口径=scalar（这两项不在上面的同源清单里——评估不跑优势，"
           "约束组由 build_training_setup 三处同源）")
    spec = m17.parse_run_spec(_log(sw=sw, grp=grp))

    assert spec.constraint_group == "-生产"
    assert spec.pm_head is False and spec.machine_age_failure is False
    assert spec.charge_head is True and spec.route_zones is True and spec.route_k == 2


def test_parse_none_arm_with_route_k_1():
    """`c-none` run：`route_k=1` + 五个头全关 + T3 关（无可控约束）——一条都不许漏。"""
    meta = ("[m13] inst=mk01 作业10×机台6 车队3｜steps=300 G=8 lr=0.0003 seed=0｜route_k=1｜"
            "pm_head=False｜device=cuda:0｜parallel=True(workers=8,worker_device=cpu)")
    sw = ("[m13] 开关（训练=评估，同源）：route_zones=False geom_bias=True pm_head=False "
          "charge_head=False batch_head=False multi_drop=True agv_failover=True "
          "machine_age_failure=False t3=False")
    grp = ("[m13] 约束组=None｜优势口径=scalar（这两项不在上面的同源清单里——评估不跑优势，"
           "约束组由 build_training_setup 三处同源）")
    spec = m17.parse_run_spec(_log(meta=meta, sw=sw, grp=grp))

    assert spec.route_k == 1 and spec.constraint_group == "None"
    assert spec.route_zones is False and spec.pm_head is False
    assert spec.charge_head is False and spec.batch_head is False and spec.t3 is False
    # `geom_bias` / `multi_drop` / `agv_failover` 在这一 run 照基线开着——别"顺手"关掉
    assert spec.geom_bias is True and spec.multi_drop is True and spec.agv_failover is True


def test_parse_reinforce_arm_drops_t3():
    """`adv-reinforce` run：优势口径是 reinforce、T3 **关**（只支持 scalar）。"""
    sw = ("[m13] 开关（训练=评估，同源）：route_zones=True geom_bias=True pm_head=True "
          "charge_head=True batch_head=True multi_drop=True agv_failover=True "
          "machine_age_failure=True t3=False")
    grp = ("[m13] 约束组=Full｜优势口径=reinforce（这两项不在上面的同源清单里——评估不跑优势，"
           "约束组由 build_training_setup 三处同源）")
    spec = m17.parse_run_spec(_log(sw=sw, grp=grp))

    assert spec.adv_mode == "reinforce" and spec.t3 is False


def test_missing_switch_line_raises():
    """日志里没有开关行 ⟹ 报错（**不许**退回默认值：默认全关 = 另一个策略）。"""
    text = "\n".join([_META_FULL, _GRP_FULL]) + "\n"
    with pytest.raises(ValueError, match="开关行"):
        m17.parse_run_spec(text)


def test_missing_switch_key_raises():
    """开关行少一个键（日志截断 / 格式变了）⟹ 报错，不静默按默认值补。"""
    sw = _SW_FULL.replace(" charge_head=True", "")
    with pytest.raises(ValueError, match="charge_head"):
        m17.parse_run_spec(_log(sw=sw))


def test_missing_meta_line_raises():
    """首行缺 `route_k=`（它改的是**动作空间**）⟹ 报错。"""
    meta = _META_FULL.replace("｜route_k=2", "")
    with pytest.raises(ValueError, match="首行"):
        m17.parse_run_spec(_log(meta=meta))


def test_unknown_constraint_group_raises():
    """约束组名不在 `ABLATION_GROUPS` ⟹ 报错（约束口径重建不出来）。"""
    grp = _GRP_FULL.replace("约束组=Full", "约束组=-拼写错的组")
    with pytest.raises(ValueError, match="ABLATION_GROUPS"):
        m17.parse_run_spec(_log(grp=grp))


def test_optional_mechanism_arm_lines_are_parsed():
    """机制验证档的两行（小电池 / 短保养间隔）**改动力学** ⟹ 有就得解析出来。

    ⚠️ 漏掉它们 = 重评跑在另一种动力学上，且没有任何提示。
    """
    extra = ("[m13] ⚠️ 小电池验证档：车队电池全换 0.1 kWh、battery_low=0 "
             "（电池是**布局**属性 ⟹ 改动力学，与默认配置读数不可比；⑪ 这才可能跑到耗尽）\n"
             "[m13] ⚠️ 短保养间隔验证档：pm_interval=30.0（改动力学；⑫ 的被迫激活量"
             " `pm_events_forced` 这才可能非零）")
    spec = m17.parse_run_spec(_log() + extra + "\n")

    assert spec.agv_battery_kwh == 0.1 and spec.pm_interval == 30.0
    c = spec.cfg()
    assert c.battery_low == 0.0 and c.pm_interval == 30.0


def test_spec_cfg_matches_parsed_switches():
    """`RunSpec.cfg()` 的三个动力学开关与解析值一一对应（默认配置 = False）。"""
    spec = m17.parse_run_spec(_log())
    c = spec.cfg()
    assert c.multi_drop is True and c.agv_failover is True and c.machine_age_failure is True
    assert c.battery_low == SimConfig().battery_low and c.pm_interval == SimConfig().pm_interval
    # 链级开关**不进 cfg**（它们走 `chain_kwargs`，由 `_make_eval_fn` 透传）
    for k in ("route_zones", "geom_bias", "pm_head", "charge_head", "batch_head"):
        assert not hasattr(c, k), f"{k} 不该出现在 SimConfig 上——它由链级开关透传"


def test_spec_chain_kwargs_verbatim():
    """`chain_kwargs()` 六项与解析值逐字相同（多一项/少一项都会让评估跑另一个策略）。"""
    spec = m17.parse_run_spec(_log())
    assert spec.chain_kwargs() == {"route_k": 2, "route_zones": True, "geom_bias": True,
                                   "pm_head": True, "charge_head": True, "batch_head": True,
                                   "t3": True}


def _spy(monkeypatch) -> dict:
    """把 `m13.roll_chain` 换成只记录 kwargs 的替身（照 `test_m13_wiring._spy` 的口径）。"""
    seen: dict = {}

    def fake(*_a, **kw):
        seen.clear()
        seen.update(kw)
        n_jobs = load_mk("mk01").n_jobs
        return [], {"makespan": 1.0, "jobs_done": n_jobs, "horizon_hit": False,
                    "energy": 0.0, "completes": {j: 1.0 for j in range(n_jobs)}}

    monkeypatch.setattr(m13, "roll_chain", fake)
    return seen


def test_eval_fn_forwards_parsed_switches_and_eval_seeds(monkeypatch):
    """**本文件的核心**：解析出的口径 → `roll_chain` 的 kwargs，逐字相同。

    含 `seed=EVAL_SEED_BASE+s`（与训练同一批评估种子）与 `sample=False`（argmax）。
    """
    seen = _spy(monkeypatch)
    spec = m17.parse_run_spec(_log())
    fn = m17.make_eval_fn(spec, load_mk("mk01"), None, None, spec.cfg(), None,
                          ABLATION_GROUPS[spec.constraint_group], seeds=3, rule=1.0)

    fn(policy=None)     # 最后一次调用 = 最后一个评估种子

    assert seen["route_k"] == 2
    for k in ("route_zones", "geom_bias", "pm_head", "charge_head", "batch_head"):
        assert seen[k] is True, f"评估侧没透传 {k}——重评跑的不是训练时那个策略"
    assert seen["sample"] is False, "评估必须走 argmax（sample=False）"
    assert seen["seed"] == m13.EVAL_SEED_BASE + 2, "评估种子 = EVAL_SEED_BASE + s（同一批）"


def test_make_eval_fn_carries_the_constraint_group(monkeypatch):
    """约束组不同 ⟹ 传给 `roll_chain` 的 `constraints` 必须跟着不同（否则动力学口径错）。"""
    seen = _spy(monkeypatch)
    sw = ("[m13] 开关（训练=评估，同源）：route_zones=True geom_bias=True pm_head=False "
          "charge_head=True batch_head=True multi_drop=True agv_failover=True "
          "machine_age_failure=False t3=True")
    grp = ("[m13] 约束组=-生产｜优势口径=scalar（这两项不在上面的同源清单里——评估不跑优势，"
           "约束组由 build_training_setup 三处同源）")
    spec = m17.parse_run_spec(_log(sw=sw, grp=grp))
    cons = ABLATION_GROUPS[spec.constraint_group]
    fn = m17.make_eval_fn(spec, load_mk("mk01"), None, None, spec.cfg(), None, cons,
                          seeds=1, rule=1.0)
    fn(policy=None)

    assert seen["constraints"] is cons
    assert cons.maintenance is False, "`-生产` run 没有 ⑫——评估的动力学里也不该有"


def test_count_lines_and_last_inline_eval(tmp_path):
    """完成判据只看行数；内联评估取**最后一个** `eval` 行（中间值会误导）。"""
    p = tmp_path / "metrics.ndjson"
    rows = [{"step": 1, "r": -1.0},
            {"step": 30, "eval": {"makespan_mean": 88.0, "makespan_per_seed": [1.0]}},
            {"step": 31, "r": -2.0},
            {"step": 60, "eval": {"makespan_mean": 73.0}}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert m17.count_lines(p) == 4
    assert m17.last_inline_eval(p)["step"] == 60
    assert m17.last_inline_eval(p)["makespan_mean"] == 73.0


def test_parse_default_seeds_is_the_phase1_batch_value():
    """默认评估种子数 = 期① 批次的公共 `--eval-seeds 5`（它不在日志里，只能这样钉）。"""
    assert m17.DEFAULT_EVAL_SEEDS == 5
    assert m17.COMPLETE_MIN_LINES == 310       # 300 步 + 10 次评估
