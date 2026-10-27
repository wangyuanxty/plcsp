#!/usr/bin/env python
"""Batch experiment queue for plcsp_v5.

The queue reads a JSON job list, runs jobs with a bounded parallelism, writes
one log per job, and records completion markers.  It is designed for an
unattended overnight run:

    cd /root/plcsp_v5
    PYTHONPATH=/root/plcsp_v5/graph-jsp-env/src \
      nohup /root/venv/bin/python run_queue.py \
      --config experiment-queue.json > queue_logs/queue.log 2>&1 &

Each job has:
    id       unique queue id
    script   Run.py or eval_st_seed3.py
    args     command-line arguments
    env      optional environment overrides
    enabled  default true
    priority lower runs first
    after    optional list of job ids that must finish first
    retries  optional retry count

Completion marker: queue_logs/<id>.done
Status file:       queue_logs/status.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

V5_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = V5_DIR / "experiment-queue.json"
QUEUE_DIR = V5_DIR / "queue_logs"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _done_path(job_id: str) -> Path:
    return QUEUE_DIR / f"{job_id}.done"


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "jobs" not in cfg or not isinstance(cfg["jobs"], list):
        raise ValueError("config needs a 'jobs' list")
    ids = [j["id"] for j in cfg["jobs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("job ids are not unique")
    for job in cfg["jobs"]:
        if "id" not in job or "script" not in job:
            raise ValueError("every job needs id and script")
        job.setdefault("args", [])
        job.setdefault("env", {})
        job.setdefault("enabled", True)
        job.setdefault("priority", 100)
        job.setdefault("after", [])
        job.setdefault("retries", 0)
    return cfg


def _gpu_free_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=10)
    except Exception:
        return None
    values = [float(x) for x in out.replace(",", " ").split() if x.strip()]
    return min(values) / 1024.0 if values else None


def _job_env(job: dict) -> dict:
    env = os.environ.copy()
    graph_src = str(V5_DIR / "graph-jsp-env" / "src")
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = graph_src + (os.pathsep + old if old else "")
    env["WANDB_MODE"] = "disabled"
    env["PYTHONIOENCODING"] = "utf-8"
    env.update({str(k): str(v) for k, v in job.get("env", {}).items()})
    return env


def _write_status(status: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False, default=str)


def _start(job: dict, python: str, records: dict, status_path: Path):
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = QUEUE_DIR / f"{stamp}_{job['id']}.log"
    script = V5_DIR / job["script"]
    if not script.exists():
        raise FileNotFoundError(script)
    cmd = [python, str(script)] + [str(a) for a in job["args"]]
    log = open(log_path, "w", encoding="utf-8")
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        cmd, cwd=str(V5_DIR), env=_job_env(job),
        stdout=log, stderr=subprocess.STDOUT, **kwargs)
    rec = {
        "id": job["id"], "pid": proc.pid, "proc": proc, "log": log,
        "log_path": str(log_path), "start": _now(), "args": job["args"],
    }
    records[job["id"]] = rec
    status = _read_status(status_path)
    status["jobs"][job["id"]] = {
        "state": "running", "pid": proc.pid, "start": rec["start"],
        "log": str(log_path),
    }
    _write_status(status, status_path)
    print(f"[{_now()}] START {job['id']} pid={proc.pid} log={log_path}",
          flush=True)


def _read_status(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"started": _now(), "jobs": {}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=None)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--only", default="", help="comma-separated ids")
    parser.add_argument("--force", action="store_true",
                        help="ignore completion markers")
    parser.add_argument("--no-gpu-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    max_parallel = int(args.max_parallel if args.max_parallel is not None
                       else cfg.get("max_parallel", 2))
    min_free_gb = float(args.min_free_gb if args.min_free_gb is not None
                        else cfg.get("min_free_gb", 3.0))
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    jobs = [j for j in cfg["jobs"]
            if j.get("enabled", True) and (not only or j["id"] in only)]
    by_id = {j["id"]: j for j in cfg["jobs"]}

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    status_path = QUEUE_DIR / "status.json"
    status = {"started": _now(), "config": str(args.config), "jobs": {}}
    _write_status(status, status_path)

    done = {j["id"] for j in jobs
            if not args.force and _done_path(j["id"]).exists()}
    pending = [j for j in jobs if j["id"] not in done]
    completed = set(done)
    failed = set()
    running = {}
    retries_left = {j["id"]: int(j.get("retries", 0)) for j in jobs}

    for job in pending:
        for dep in job.get("after", []):
            if dep not in by_id:
                raise ValueError(f"{job['id']} depends on unknown {dep}")

    if args.dry_run:
        for job in sorted(pending, key=lambda j: j["priority"]):
            deps = job.get("after", [])
            print(f"[dry-run] {job['id']} priority={job['priority']} "
                  f"after={deps}")
            print("  ", " ".join([args.python, str(V5_DIR / job["script"])]
                                + [str(a) for a in job["args"]]))
        return 0

    print(f"[{_now()}] queue start: {len(pending)} pending, "
          f"max_parallel={max_parallel}, min_free_gb={min_free_gb}",
          flush=True)

    try:
        while True:
            # start runnable jobs
            for job in sorted(pending, key=lambda j: j["priority"]):
                if job["id"] in running:
                    continue
                if any(dep not in completed for dep in job.get("after", [])):
                    continue
                if len(running) >= max_parallel:
                    break
                if not args.no_gpu_check:
                    free = _gpu_free_gb()
                    if free is not None and free < min_free_gb:
                        break
                _start(job, args.python, running, status_path)

            # poll running jobs
            finished = []
            for job_id, rec in list(running.items()):
                rc = rec["proc"].poll()
                if rc is None:
                    continue
                rec["log"].close()
                rec["end"] = _now()
                rec["rc"] = rc
                finished.append(job_id)
                status = _read_status(status_path)
                if rc == 0:
                    _done_path(job_id).touch()
                    completed.add(job_id)
                    state = "done"
                else:
                    if retries_left.get(job_id, 0) > 0:
                        retries_left[job_id] -= 1
                        state = "retry"
                    else:
                        failed.add(job_id)
                        state = "failed"
                status["jobs"][job_id] = {
                    "state": state, "pid": rec.get("pid"),
                    "start": rec.get("start"), "end": rec["end"],
                    "rc": rc, "log": rec.get("log_path"),
                }
                _write_status(status, status_path)
                print(f"[{_now()}] {state.upper()} {job_id} rc={rc} "
                      f"log={rec.get('log_path')}", flush=True)
                del running[job_id]
                if state == "done":
                    pending = [j for j in pending if j["id"] != job_id]
                elif state == "retry":
                    pass  # stays pending
                elif state == "failed":
                    pending = [j for j in pending if j["id"] != job_id]
                    if cfg.get("stop_on_failure", False):
                        raise SystemExit(2)

            if not running:
                startable = []
                for job in pending:
                    if any(dep not in completed for dep in job.get("after", [])):
                        continue
                    startable.append(job)
                if not startable:
                    break
            if not pending and not running:
                break
            time.sleep(max(1.0, args.poll_seconds))
    except KeyboardInterrupt:
        print(f"[{_now()}] interrupted; terminating children", flush=True)
        for rec in running.values():
            try:
                rec["proc"].terminate()
            except Exception:
                pass
        return 130

    status = _read_status(status_path)
    status["finished"] = _now()
    status["completed"] = sorted(completed)
    status["failed"] = sorted(failed)
    status["blocked"] = sorted(j["id"] for j in pending)
    _write_status(status, status_path)
    print(f"[{_now()}] queue done: completed={len(completed)} "
          f"failed={len(failed)} blocked={len(pending)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
