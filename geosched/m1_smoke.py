"""M1 冒烟测试：MK01 + 三种布局拓扑 → 一次 episode 指标（rollout API 与拓扑轴对比）。

运行：python -m geosched.m1_smoke
"""
from .env.instances import load_mk, MK_OPTIMAL, load_kacem_8x8
from .env.des import rollout, SimConfig


def main() -> None:
    inst = load_mk("mk01")
    cfg = SimConfig(n_agv=2, agv_speed_mps=20.0, zone_hold=0.3, n_zones=inst.n_machines)
    opt = MK_OPTIMAL["mk01"]
    for ltype in ("line", "U", "island"):
        res = rollout(inst, layout_type=ltype, seed_layout=1, seed_chain=42, cfg=cfg)
        gap = 100 * (res["makespan"] - opt) / opt
        print(f"[smoke] {ltype:6s} makespan={res['makespan']:6.1f} gap={gap:6.1f}% "
              f"jobs={res['jobs_done']}/{inst.n_jobs} ops={res['ops_done']} moves={res['moves']} "
              f"fails={res['fail_events']} horizon_hit={res['horizon_hit']}")
    try:
        load_kacem_8x8()
    except NotImplementedError as e:
        print(f"[smoke] Kacem 入口：{e}")


if __name__ == "__main__":
    main()
