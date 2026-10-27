"""Re-solve the failed 10x10 MIP-AGV instances with a longer routing limit."""

import csv
import glob
import types

from plcsp_v5.baselines.mip_agv.run_batch import FIELDS, run_one

INST = "plcsp_v5/instance/test_Instance_J10M10E2.pkl"
OUT_DIR = "plcsp_v5/results/mip_agv_100_10x10"

rows = {}
for path in sorted(glob.glob(OUT_DIR + "/shard*.csv")):
    for row in csv.DictReader(open(path)):
        rows[int(row["idx"])] = row

args = types.SimpleNamespace(
    n_agv=6, stage_a_time=10.0, routing_time=60.0, iters=3, workers=8)

done = 0
for idx in sorted(rows):
    if not rows[idx].get("stage_a_status", "").startswith("error"):
        continue
    new = run_one(INST, idx, args)
    rows[idx] = {k: new.get(k, "") for k in FIELDS}
    done += 1
    print(f"redo {idx} {rows[idx].get('best_obj')} "
          f"{rows[idx].get('stage_a_status')}", flush=True)

for shard in range(4):
    path = f"{OUT_DIR}/shard{shard}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for idx in sorted(rows):
            if idx % 4 == shard:
                w.writerow({k: rows[idx].get(k, "") for k in FIELDS})

print("redone", done)
