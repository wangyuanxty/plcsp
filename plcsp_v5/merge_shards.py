"""Merge sharded eval npz files into one, ordered by instance index."""

import glob
import sys

import numpy as np

pattern, out = sys.argv[1], sys.argv[2]
objs, cost4, idx = [], [], []
for path in glob.glob(pattern):
    d = np.load(path)
    objs.append(d["obj"])
    cost4.append(d["cost4"])
    idx.append(d["idx"])
obj = np.concatenate(objs)
cost = np.concatenate(cost4)
ids = np.concatenate(idx)
order = np.argsort(ids)
np.savez(out, obj=obj[order], cost4=cost[order], idx=ids[order])
print(f"merged {len(ids)} instances -> {out}")
