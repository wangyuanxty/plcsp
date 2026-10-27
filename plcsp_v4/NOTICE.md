# NOTICE — v4 代码来源与许可

## 上游

| 项 | 内容 |
|---|---|
| 仓库 | https://github.com/ClouDaDaDa/DFJSPT_code |
| 本地副本 | `third_party/DFJSPT_code` |
| commit | `7ad75b4c59cc7402f844eea32712be05df0c0785` |
| 许可 | **GPL-3.0** |

## 论文

Wang, W.; Zhang, Y.; Wang, Y.; Pan, G.; Feng, Y.
*Hierarchical multi-agent deep reinforcement learning for dynamic flexible
job-shop scheduling with transportation.*
**International Journal of Production Research**, 2025.
DOI: 10.1080/00207543.2025.2511239

## 本目录的改动

只做了一处机械替换：

```
from DFJSPT...   →  from plcsp_v4...
import DFJSPT    →  import plcsp_v4
```

其余代码与上游一致。

## 未复制的部分

- `training_results/`：上游预训练权重（约 72 MB）。仍保留在
  `third_party/DFJSPT_code/DFJSPT/training_results/`。
- 上游根目录的 `Run.py` / `test_all.py` / `model/` / `trainer/`：属于 E2E-MAPPO 仓库，不在 DFJSPT 中。

## 许可证义务

`plcsp_v4/` 是 GPL-3.0 的衍生作品。**对外发布时，本目录及其衍生部分
必须按 GPL-3.0 提供源码。** 如果最终需要非 GPL 的发布许可，
必须重写这些文件，不能继续复制。
