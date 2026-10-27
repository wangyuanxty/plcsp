# -*- coding: utf-8 -*-
"""v4 —— 从 DFJSPT 起步的干净实现（2026-10-20 起）。

══ 来源 ══

代码复制自 `third_party/DFJSPT_code/DFJSPT/`（commit `7ad75b4c59cc7402f844eea32712be05df0c0785`）。
对应论文：Wang, W.; Zhang, Y.; Wang, Y.; Pan, G.; Feng, Y.
*Hierarchical multi-agent deep reinforcement learning for dynamic
flexible job-shop scheduling with transportation.*
**International Journal of Production Research**, 2025.
DOI: 10.1080/00207543.2025.2511239

══ 许可 ══

上游为 **GPL-3.0**（见 `LICENSE`）。本目录是它的衍生副本，
**发布时同样受 GPL-3.0 约束**。

本次复制只改了一处：把 `from DFJSPT` / `import DFJSPT` 改为
`from plcsp_v4` / `import plcsp_v4`。其余代码未动。

══ 与原仓库的差异 ══

- 未复制 `training_results/`（预训练权重，约 72 MB）。
  原路径：`third_party/DFJSPT_code/DFJSPT/training_results/`。
- 未复制上游根目录的 `Run.py` / `test_all.py`（它们属于其它仓库）。
"""
