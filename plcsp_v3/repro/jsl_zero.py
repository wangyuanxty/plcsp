# -*- coding: utf-8 -*-
"""**把 JobShopLab 的 ft06 运输时间清零** → 纯 JSSP 基准。

为什么要清：`data/jssp_instances/transport/ft06.yaml` 的 `logistics` 矩阵带非零行程时间
（跑出来 fifo 83 / spt 85 / mwkr 85）。**要和我的仿真比，两边必须用同一个运输口径。**
零运输是唯一能让两边语义一致的配置（瞬时运输 ⟹ AGV 模型差异消失）。

⚠️ 这是**测试用改写**，不是他们的原始数据。原文件不动，改写版落临时目录。
"""
import re
import sys
from pathlib import Path

SRC = Path("data/jssp_instances/transport/ft06.yaml")
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ft06_zero.yaml")

txt = SRC.read_text(encoding="utf-8")
lines = txt.splitlines()
out, in_mat = [], False
for ln in lines:
    if ln.strip().startswith("specification:"):
        in_mat = "logistics" in "\n".join(out[-4:]) or True
    # `specification:` 段里的数据行：`m-x| <数字> <数字> ...`
    # ⚠️ **表头行长得一模一样**（`m-0|m-1|m-2|...`），必须靠"后面是不是数字"区分。
    body = "|".join(ln.split("|")[1:])
    # ⚠️ 格式是 `<行名>|` 后面跟**空格分隔**的数字（不是 `|` 分隔）⟹ 数**数字**的个数，
    # 不是数 cell 的个数。表头行 `m-0|m-1|...` 拆开后含字母 ⟹ 自然被排除。
    nums = body.split()
    is_data = bool(nums) and all(re.fullmatch(r"-?\d+(\.\d+)?", x) for x in nums)
    if re.match(r"^\s*(m-\d+|b-\d+|in-buf|out-buf)\|", ln) and is_data:
        out.append(ln.split("|")[0] + "|" + " ".join(["0"] * len(nums)))
        continue
    out.append(ln)
DST.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"改写版写到 {DST}")
for ln in out:
    if re.match(r"^\s*(m-\d+)\|", ln):
        print("   ", ln[:90])
