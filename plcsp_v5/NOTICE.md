# NOTICE — plcsp_v5

## 来源

| 项 | 值 |
|---|---|
| 上游仓库 | https://github.com/RKWin93/E2E-MAPPO-for-MT-FJSP.git |
| commit | `bdd8719144a62f66c6c16c4243056047c050dc6d` |
| 许可证 | MIT，Copyright (c) 2024 RKWin（见 `LICENSE`） |
| 论文 | R. Wang, Y. Jing, C. Gu, S. He, J. Chen, "End-to-End Multitarget Flexible Job Shop Scheduling With Deep Reinforcement Learning", *IEEE Internet of Things Journal*, 12(4): 4420–4434, 2025. doi:10.1109/JIOT.2024.3485748 |

`graph-jsp-env/` 是上游附带的 MIT 依赖包。

## 相对上游的修改

代码中搜索 `[local fix]` 可定位每一处。

| # | 文件 | 修改 | 原因 |
|---|---|---|---|
| 1 | `trainer/train_device.py` | `cuda:1` → `cuda:0`；单卡守卫 | 本机只有 1 张 GPU |
| 2 | `trainer/parallel_env.py` | 日志里的 `np.array(ragged_list)` 改为记录长度 | numpy ≥ 2 拒绝不规则数组 |
| 3 | `Run.py` | `os.rename` → `os.replace` | Windows 下目标存在时 `os.rename` 失败 |
| 4 | `test_all.py` | `samples=2` → `samples=100` | 论文 Table V 用 100 个测试实例 |
| 5 | `test_all.py` | `torch.load` 默认 `map_location` | 上游权重保存在 8 卡机器的 `cuda:4` |
| 6 | 多处 | 绝对路径 `third_party/E2E-MAPPO-for-MT-FJSP` → `plcsp_v5` | 脱离 third_party |

其他环境修复（gym 0.26.2、simpy、pynvml、rich、kaleido、wandb offline、
Linux 路径等）在 `third_party` 阶段已完成，拷贝时一并带入。

## 数据

`instance/` 下为 6×6×2 合成实例：

| 文件 | 数量 | 生成 seed |
|---|---:|---:|
| `Instance_J6M6E2.pkl` | 12,800（训练） | 0 |
| `eval_Instance_J6M6E2.pkl` | 100（验证） | 1 |
| `test_Instance_J6M6E2.pkl` | 100（测试） | 3 |

单个实例含 4 个张量：`t` (36×6)、`p` (36×6)、`transT` (6×6)、`edge` (2×3)。

## 基线复现

用 `test_all.py` + `tester/IoTJ_MAPPO/PPO_*_J6M6E2_1000.pth`，100 个测试实例：

| 指标 | 论文 Table V | 本仓库 |
|---|---:|---:|
| P-G Obj. | 732.5 | **732.5** |
| Std | 96.8 | **96.8** |
| Time | 0.529 s | 0.533 s |

## 使用提示

- 运行目录：`plcsp_v5` 必须在 `sys.path` 上（直接 `python plcsp_v5/Run.py` 即可）。
- `graph-jsp-env` 已复制到本目录；若需完全脱离 `third_party` 的 editable 安装，
  在 `e2e` 环境中执行
  `pip install -e plcsp_v5/graph-jsp-env --no-deps`。
- 训练输出写入 `trained_model/`、`trajectory/`、`results/`；这些目录已建好。
