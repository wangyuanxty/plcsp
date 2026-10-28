"""Wait for the running 10x10 RS jobs, then run their evals (4 shards each),
the no-RS queue, and the E2E queue."""

import json
import os
import subprocess
import time

CFG = "experiment-queue-10x10b.json"
PY = "/root/venv/bin/python"
NSHARD = 4


def wait_for(pattern):
    while subprocess.run(["pgrep", "-f", pattern],
                         capture_output=True).stdout.strip():
        time.sleep(120)


cfg = json.load(open(CFG))
evals = {j["id"]: j for j in cfg["jobs"]
         if j["script"] == "eval_st_seed3.py"}

wait_for("plan10_local_J10M10E2")
print("training finished", flush=True)

for tag in ["s1", "s2", "s3"]:
    job = evals[f"E_local_J10M10E2_{tag}_10x10"]
    base_env = os.environ.copy()
    base_env.update(job.get("env", {}))
    suffix = base_env.get("EVAL_OUTPUT_SUFFIX", "")
    procs = []
    for k in range(NSHARD):
        env = base_env.copy()
        env["EVAL_SHARD"] = str(k)
        env["EVAL_N_SHARDS"] = str(NSHARD)
        env["EVAL_OUTPUT_SUFFIX"] = f"{suffix}_shard{k}of{NSHARD}"
        procs.append(subprocess.Popen([PY, job["script"]] + job["args"],
                                      env=env))
    for p in procs:
        p.wait()
    print("eval", tag, "shards done", flush=True)

    run_tag = job["args"][job["args"].index("--run_tag") + 1]
    model_dir = os.path.join("trained_model", run_tag)
    pattern = os.path.join(
        model_dir, f"eval_st_seed3{suffix}_shard*of{NSHARD}.npz")
    out = os.path.join(model_dir, f"eval_st_seed3{suffix}.npz")
    subprocess.run([PY, "merge_shards.py", pattern, out], check=False)

# no-RS 10x10 skipped for the draft: not used in Table 4.
print("E2E queue", flush=True)
subprocess.run([PY, "run_queue.py",
                "--config", "experiment-queue-e2e-sizes.json"], check=False)

print("all done", flush=True)
