"""Paired significance tests for the main comparison of the v5 paper.

Every method is evaluated on the same 100 test instances, so the comparison is
paired by instance. The learned methods are averaged over their three training
seeds on each instance; the deterministic baselines have one value per
instance. The test is the two-sided Wilcoxon signed-rank test on the
per-instance objective. The family of comparisons is corrected with the
Holm-Bonferroni procedure at alpha = 0.05.

Before any test the script recomputes the mean of every method and checks it
against the value printed in Table 4 of the paper. A mismatch stops the run.

Usage:
    python -m plcsp_v5.analysis_stat_tests
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "trained_model"

SIZES = ("6x6", "10x6", "10x10")

# Values printed in Table 4 of the paper. Used as a guard only.
TABLE4 = {
    ("ours", "6x6"): 867.75,
    ("ours", "10x6"): 808.67,
    ("ours", "10x10"): 1836.64,
    ("rga", "6x6"): 762.04,
    ("rga", "10x6"): 796.70,
    ("rga", "10x10"): 2298.63,
    ("pdr", "6x6"): 1319.24,
    ("pdr", "10x6"): 1461.63,
    ("pdr", "10x10"): 3288.35,
    ("e2e", "6x6"): 1075.64,
    ("e2e", "10x6"): 1066.24,
    ("e2e", "10x10"): 3153.81,
    ("li", "6x6"): 1160.59,
    ("li", "10x6"): 1198.09,
    ("li", "10x10"): 2738.91,
    ("mip", "6x6"): 605.77,
    ("mip", "10x6"): 516.43,
    ("mip", "10x10"): 1242.46,
}

OURS_NPZ = {
    "6x6": "m3_agv6_ours_s{seed}/eval_st_seed3_seed3.npz",
    "10x6": "m3_agv_s{seed}_J10M6/eval_st_seed3_seed3.npz",
    "10x10": "m3_agv_s{seed}_J10M10/eval_st_seed3_seed3.npz",
}

NO_RS_NPZ = {
    "6x6": "m3_noRS_s{seed}/eval_st_seed3_seed3.npz",
    "10x6": "m3_noRS_J10M6_s{seed}/eval_st_seed3.npz",
}

PPO_NPZ = {
    "6x6": "m3_ppo_s{seed}/eval_st_seed3_seed3.npz",
    "10x6": "m3_ppo_J10M6_s{seed}/eval_st_seed3.npz",
}

E2E_NPZ = {
    "6x6": "m3_agv6_e2e_s{seed}/eval_st_seed3_seed3.npz",
    "10x6": "m3_e2e_s{seed}_J10M6/eval_st_seed3_seed3.npz",
    "10x10": "m3_e2e_s{seed}_J10M10/eval_st_seed3_seed3.npz",
}

RGA_JSON = {
    "6x6": "rga_m2_simeval_6x6.json",
    "10x6": "rga_m2_simeval_J10M6E2.json",
    "10x10": "rga_m2_simeval_J10M10E2.json",
}

PDR_JSON = {
    "6x6": "pdr_m2_trailing_agvrule3_6x6.json",
    "10x6": "pdr_m2_trailing_agvrule3_10x6.json",
    "10x10": "pdr_m2_trailing_agvrule3_10x10.json",
}

LI_JSON = "baselines/li2025/results/m2_m3_li_J{size}E2_s{seed}.json"

MIP_CSV = "joint_m2_{size}_v2/shard*.csv"

LI_SIZE_TAG = {"6x6": "6M6", "10x6": "10M6", "10x10": "10M10"}


def npz_per_instance(path: Path) -> np.ndarray:
    """Return the objective of each instance, indexed by instance id."""
    z = np.load(path)
    obj = np.asarray(z["obj"], dtype=float)
    idx = np.asarray(z["idx"], dtype=int)
    out = np.full(int(idx.max()) + 1, np.nan)
    out[idx] = obj
    return out


def seed_mean_npz(pattern: str, size: str) -> np.ndarray:
    """Average the per-instance objective over the three training seeds."""
    runs = [npz_per_instance(MODELS / pattern.format(seed=s))
            for s in (1, 2, 3)]
    stack = np.vstack(runs)
    return np.nanmean(stack, axis=0)


def json_records_mean(path: Path, value: str = "obj") -> np.ndarray:
    """Average records over seeds, indexed by instance id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    n = max(int(r["instance"]) for r in data["records"]) + 1
    total = np.zeros(n)
    count = np.zeros(n)
    for r in data["records"]:
        i = int(r["instance"])
        total[i] += float(r[value])
        count[i] += 1
    out = np.full(n, np.nan)
    seen = count > 0
    out[seen] = total[seen] / count[seen]
    return out


def _best_rule(path: Path) -> tuple[str, str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    best = min(data["summary"]["per_rule"], key=lambda r: r["mean"])
    return best["job_rule"], best["machine_rule"], best["agv_rule"]


def pdr_best_rule(size: str) -> np.ndarray:
    """Per-instance objective of the best fixed rule triple of that size."""
    path = RESULTS / PDR_JSON[size]
    job, mach, agv = _best_rule(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 100
    out = np.full(n, np.nan)
    for r in data["records"]:
        if (r["job_rule"], r["machine_rule"], r["agv_rule"]) == (job, mach, agv):
            out[int(r["instance"])] = float(r["obj"])
    return out


def lis(size: str) -> np.ndarray:
    """Li et al. adapted baseline, averaged over the three seeds."""
    tag = LI_SIZE_TAG[size]
    runs = []
    for s in (1, 2, 3):
        path = ROOT / LI_JSON.format(size=tag, seed=s)
        runs.append(json_records_mean(path))
    return np.nanmean(np.vstack(runs), axis=0)


def mip(size: str) -> np.ndarray:
    """Joint CP-SAT baseline, per instance, from the shard CSV files.

    An instance the solver did not solve is written with ``joint_obj`` 0, so
    a non-positive value is treated as missing rather than as a perfect score.
    """
    out = np.full(100, np.nan)
    for path in sorted((RESULTS / MIP_CSV.format(size=size)).parent.glob("shard*.csv")):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    value = float(row["joint_obj"])
                except (KeyError, ValueError):
                    continue
                if value > 0:
                    out[int(row["idx"])] = value
    return out


def holm(pvals: list[float], alpha: float = 0.05) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def main() -> None:
    ours = {s: seed_mean_npz(OURS_NPZ[s], s) for s in SIZES}
    e2e = {s: seed_mean_npz(E2E_NPZ[s], s) for s in SIZES}
    rga = {s: json_records_mean(RESULTS / RGA_JSON[s]) for s in SIZES}
    pdr = {s: pdr_best_rule(s) for s in SIZES}
    li = {s: lis(s) for s in SIZES}
    mip_vals = {s: mip(s) for s in SIZES}

    series: dict[str, dict[str, np.ndarray]] = {
        "ours": ours, "e2e": e2e, "rga": rga, "pdr": pdr, "li": li, "mip": mip_vals,
    }

    print("== guard: recomputed mean against Table 4 ==")
    for (name, size), expected in TABLE4.items():
        arr = series[name][size]
        got = float(np.nanmean(arr))
        flag = "ok" if abs(got - expected) < 0.02 else "MISMATCH"
        print(f"  {name:5s} {size:6s} recomputed {got:9.2f}  table {expected:9.2f}  {flag}")
        if flag != "ok":
            raise SystemExit(f"guard failed for {name} {size}")

    tests = []
    for size in SIZES:
        for base in ("rga", "pdr", "e2e", "li"):
            tests.append((size, base))
    for size in SIZES:
        tests.append((size, "mip"))

    rows = []
    for size, base in tests:
        a = ours[size]
        b = series[base][size]
        mask = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[mask], b[mask]
        diff = a - b
        stat, p = wilcoxon(a, b, alternative="two-sided")
        rows.append({
            "size": size,
            "base": base,
            "n": int(mask.sum()),
            "ours": float(np.mean(a)),
            "base_mean": float(np.mean(b)),
            "median_diff": float(np.median(diff)),
            "wins": int((diff < 0).sum()),
            "p": float(p),
        })

    adjusted = holm([r["p"] for r in rows])
    for r, adj in zip(rows, adjusted):
        r["p_holm"] = adj

    print("\n== paired Wilcoxon signed-rank, Holm over the family ==")
    print(f"{'size':6s} {'base':5s} {'n':>4s} {'ours':>9s} {'base':>9s} "
          f"{'med.diff':>9s} {'wins':>5s} {'p':>10s} {'p_holm':>10s}")
    for r in rows:
        print(f"{r['size']:6s} {r['base']:5s} {r['n']:4d} {r['ours']:9.2f} "
              f"{r['base_mean']:9.2f} {r['median_diff']:9.2f} {r['wins']:5d} "
              f"{r['p']:10.3e} {r['p_holm']:10.3e}")

    # Second family: the rule-selector ablation. Two comparisons, so Holm is
    # applied within this family only and the two families are not pooled.
    ablation = []
    for size in ("6x6", "10x6"):
        with_rs = ours[size]
        without = seed_mean_npz(NO_RS_NPZ[size], size)
        mask = ~np.isnan(with_rs) & ~np.isnan(without)
        diff = with_rs[mask] - without[mask]
        _, p = wilcoxon(with_rs[mask], without[mask], alternative="two-sided")
        ablation.append({"size": size, "n": int(mask.sum()),
                         "with_rs": float(np.mean(with_rs[mask])),
                         "no_rs": float(np.mean(without[mask])),
                         "median_diff": float(np.median(diff)),
                         "wins": int((diff < 0).sum()), "p": float(p)})
    for row, adj in zip(ablation, holm([r["p"] for r in ablation])):
        row["p_holm"] = adj

    print("\n== rule-selector ablation, separate family ==")
    for r in ablation:
        print(f"{r['size']:6s} RS {r['with_rs']:8.2f}  noRS {r['no_rs']:8.2f}  "
              f"med.diff {r['median_diff']:8.2f}  wins {r['wins']:3d}  "
              f"p {r['p']:.4f}  p_holm {r['p_holm']:.4f}")

    # Third family: GRPO against PPO, full protocol.
    estimator = []
    for size in ("6x6", "10x6"):
        grpo = ours[size]
        ppo = seed_mean_npz(PPO_NPZ[size], size)
        mask = ~np.isnan(grpo) & ~np.isnan(ppo)
        diff = grpo[mask] - ppo[mask]
        _, p = wilcoxon(grpo[mask], ppo[mask], alternative="two-sided")
        estimator.append({"size": size, "n": int(mask.sum()),
                          "grpo": float(np.mean(grpo[mask])),
                          "ppo": float(np.mean(ppo[mask])),
                          "median_diff": float(np.median(diff)),
                          "wins": int((diff < 0).sum()), "p": float(p)})
    for row, adj in zip(estimator, holm([r["p"] for r in estimator])):
        row["p_holm"] = adj

    print("\n== GRPO vs PPO, full protocol, separate family ==")
    for r in estimator:
        print(f"{r['size']:6s} GRPO {r['grpo']:8.2f}  PPO {r['ppo']:8.2f}  "
              f"med.diff {r['median_diff']:8.2f}  wins {r['wins']:3d}  "
              f"p {r['p']:.3e}  p_holm {r['p_holm']:.3e}")

    out = ROOT / "results" / "stat_tests_main.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"alpha": 0.05, "test": "wilcoxon two-sided",
                               "correction": "holm", "rows": rows,
                               "rule_selector": ablation,
                               "grpo_vs_ppo": estimator}, indent=1),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
