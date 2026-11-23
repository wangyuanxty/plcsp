"""Case-study Gantt chart: learned controller against PDR, machines + AGVs.

Both panels use the station instance and the M2 transport model, so the two
schedules are scored by the same objective as the main tables.
"""

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plcsp_v5"))
sys.path.insert(0, str(ROOT / "plcsp_v5" / "graph-jsp-env" / "src"))

from baselines.check_agv_env import run_env_m2  # noqa: E402
from baselines.agv_baselines import pdr_schedule  # noqa: E402
from plcsp_v5.agv import state as agv_state  # noqa: E402

OUT = Path(__file__).resolve().parent
OURS = Path("C:/Users/evari/AppData/Local/Temp/figrefs/ours_schedule.json")

# Okabe-Ito, colour-blind safe. Jobs and vehicles use separate sets.
JOB_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
              "#56B4E9", "#D55E00"]
AGV_COLORS = ["#332288", "#88CCEE", "#44AA99", "#117733",
              "#999933", "#AA4499"]

from pubstyle import FigureStyle, apply_publication_style, finalize_figure  # noqa: E402


def draw(ax, rows, legs, n_op, n_machine, n_agv, title):
    for i in range(n_machine + n_agv):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color="#E9EEF3", zorder=0, lw=0)
    for r in rows:
        job = int(r["task"]) // n_op
        ax.barh(int(r["machine"]),
                float(r["finish"]) - float(r["start"]),
                left=float(r["start"]), height=0.62,
                color=JOB_COLORS[job % len(JOB_COLORS)],
                edgecolor="white", linewidth=0.3)
    for leg in legs:
        ax.barh(n_machine + int(leg["agv"]),
                float(leg["end"]) - float(leg["start"]),
                left=float(leg["start"]), height=0.5,
                color=AGV_COLORS[int(leg["agv"]) % len(AGV_COLORS)],
                alpha=0.9, edgecolor="white", linewidth=0.3)
    labels = [f"M{i+1}" for i in range(n_machine)]
    labels += [f"A{i+1}" for i in range(n_agv)]
    ax.set_yticks(range(n_machine + n_agv))
    ax.set_yticklabels(labels)
    ax.axhline(n_machine - 0.5, color="black", lw=0.6)
    ax.set_xlabel("Time")
    ax.set_title(title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.6, n_machine + n_agv - 0.4)


def main():
    import pickle
    raw = pickle.load(open(ROOT / "plcsp_v5/instance/test_Instance_J6M6E2_lu.pkl",
                           "rb"))
    tm, pw, tt, edge, lu_m, lu_e = [np.asarray(x[0], dtype=np.float64)
                                    for x in raw[:6]]
    edge = edge.astype(np.int64)
    n_task, n_machine = tm.shape
    n_op = n_machine

    ours = json.loads(OURS.read_text())
    os_seq, ms = pdr_schedule(tm, pw, "mor", "spt")

    pdr_legs = []

    def _make(orig):
        def _tt(src, *a, **kw):
            out = orig(src, *a, **kw)
            ready = float(a[2]) if len(a) > 2 else float(kw.get("ready", 0.0))
            pdr_legs.append({"agv": int(src.idx),
                             "start": max(ready, float(src.free_time)),
                             "end": float(out["src_free"])})
            dst = kw.get("dst")
            if dst is not None:
                pdr_legs.append({"agv": int(dst.idx),
                                 "start": float(out["dst_free"]) - float(out["dst_travelled"]),
                                 "end": float(out["dst_free"])})
            return out
        return _tt

    _mods = [agv_state]
    try:
        import agv.state as _state_b
        if _state_b is not agv_state:
            _mods.append(_state_b)
    except ImportError:
        pass
    _saved = []
    for _mod in _mods:
        for _name in ("transport_timing", "transport_timing_m2"):
            _fn = getattr(_mod, _name, None)
            if _fn is not None:
                _saved.append((_mod, _name, _fn))
                setattr(_mod, _name, _make(_fn))
    env = run_env_m2(tm, pw, tt, edge, lu_m, lu_e, os_seq, list(ms), n_agv=6)
    for _mod, _name, _fn in _saved:
        setattr(_mod, _name, _fn)

    pdr_rows = []
    for t in range(n_task):
        m = int(ms[t])
        pdr_rows.append({"task": t, "start": float(env["starts"][t]),
                         "finish": float(env["finishes"][t]), "machine": m})

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    draw(axes[0], ours["rows"], ours["agv_rows"], n_op,
         n_machine, 6, "(a) HMAGRA")
    draw(axes[1], pdr_rows, pdr_legs, n_op, n_machine, 6,
         f"(b) PDR, objective {env['obj']:.1f}")
    handles = [Patch(facecolor=JOB_COLORS[j], edgecolor="white",
                     label=f"J{j+1}") for j in range(6)]
    handles += [Patch(facecolor=AGV_COLORS[a], label=f"A{a+1}")
                for a in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=7.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    finalize_figure(fig, OUT / "fig_gantt", formats=["pdf"])
    print("wrote fig_gantt.pdf", "agv legs ours/pdr", len(ours["agv_rows"]),
          len(pdr_legs), "pdr obj", round(env["obj"], 2))


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=8, axes_linewidth=0.9))
    main()