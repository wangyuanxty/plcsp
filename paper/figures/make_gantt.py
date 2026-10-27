"""Case-study Gantt chart: learned controller against PDR, machines + AGVs."""

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

from baselines.check_agv_env import run_env  # noqa: E402
from baselines.agv_baselines import pdr_schedule  # noqa: E402
import agv.state as agv_state  # noqa: E402

OUT = Path(__file__).resolve().parent
OURS = Path("C:/Users/evari/AppData/Local/Temp/figrefs/ours_schedule.json")

JOB_COLORS = ["#0F4D92", "#3775BA", "#42949E", "#8BCF8B", "#FFD700",
              "#B64342", "#9A4D8E", "#CFCECE", "#E9A6A1", "#AADCA9"]
AGV_COLOR = "#42949E"
HATCHES = ["", "//", "xx", "..", "++", "oo"]

from pubstyle import FigureStyle, apply_publication_style, finalize_figure


def draw(ax, rows, legs, n_op, n_machine, n_agv, title):
    for r in rows:
        job = int(r["task"]) // n_op
        bar = ax.barh(int(r["machine"]),
                      float(r["finish"]) - float(r["start"]),
                      left=float(r["start"]), height=0.62,
                      color=JOB_COLORS[job % len(JOB_COLORS)],
                      edgecolor="white", linewidth=0.3)[0]
        bar.set_hatch(HATCHES[job % len(HATCHES)])
    for leg in legs:
        ax.barh(n_machine + int(leg["agv"]),
                float(leg["end"]) - float(leg["start"]),
                left=float(leg["start"]), height=0.5,
                color=AGV_COLOR, alpha=0.75, edgecolor="white",
                linewidth=0.3)
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
    raw = pickle.load(open(ROOT / "plcsp_v5/instance/test_Instance_J6M6E2.pkl",
                           "rb"))
    tm = np.asarray(raw[0][0], dtype=np.float64)
    pw = np.asarray(raw[1][0], dtype=np.float64)
    tt = np.asarray(raw[2][0], dtype=np.float64)
    edge = np.asarray(raw[3][0], dtype=np.int64)
    n_task, n_machine = tm.shape
    n_op = n_machine

    ours = json.loads(OURS.read_text())
    os_seq, ms = pdr_schedule(tm, pw, "mor", "spt")

    pdr_legs = []
    _orig = agv_state.transport_timing

    def _tt(src, pickup, dropoff, ready, total, travel, **kw):
        out = _orig(src, pickup, dropoff, ready, total, travel, **kw)
        pdr_legs.append({"agv": int(src.idx),
                         "start": max(float(ready), float(src.free_time)),
                         "end": float(out["src_free"])})
        dst = kw.get("dst")
        if dst is not None:
            pdr_legs.append({"agv": int(dst.idx),
                             "start": float(out["dst_free"]) - float(out["dst_travelled"]),
                             "end": float(out["dst_free"])})
        return out

    agv_state.transport_timing = _tt
    env = run_env(tm, pw, tt, edge, os_seq, list(ms), 1, 6)
    agv_state.transport_timing = _orig

    pdr_rows = []
    for t in range(n_task):
        m = int(ms[t])
        pdr_rows.append({"task": t, "start": float(env["starts"][t]),
                         "finish": float(env["finishes"][t]), "machine": m})

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True)
    draw(axes[0], ours["rows"], ours["agv_rows"], n_op,
         n_machine, 6, "(a) H-ST-GRPO")
    draw(axes[1], pdr_rows, pdr_legs, n_op, n_machine, 6,
         f"(b) PDR, objective {env['obj']:.1f}")
    handles = [Patch(facecolor=JOB_COLORS[j], hatch=HATCHES[j],
                     edgecolor="white", label=f"J{j+1}") for j in range(6)]
    handles.append(Patch(facecolor=AGV_COLOR, label="AGV busy"))
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=5.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    finalize_figure(fig, OUT / "fig_gantt", formats=["pdf"])
    print("wrote fig_gantt.pdf", "agv legs ours/pdr", len(ours["agv_rows"]),
          len(pdr_legs))


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.9))
    main()
