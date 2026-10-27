# PLCSP — hierarchical multi-agent RL for flexible job shop scheduling with AGV transport

This repository holds the code, experiment records, and manuscript for
**H-ST-GRPO**: a hierarchical multi-agent controller for a flexible job shop
with per-workshop AGV fleets. Three policy heads make the operation, machine,
and vehicle decisions; the low-level heads use dual Set Transformer encoders
and are trained with group-relative policy optimization (GRPO) without a value
critic; a top-level policy selects a dispatching-rule prior for each head every
few steps.

The paper draft is in [`paper/`](paper/). The current experiment line is
`plcsp_v5/`.

## Repository layout

| Path | Contents |
|---|---|
| `plcsp/` | v1 line (archived; tests kept runnable) |
| `plcsp_v2/` | v2 line (archived; tests kept runnable) |
| `plcsp_v3/` | v3 line (PA rule heads) |
| `plcsp_v4/` | exploratory line (DFJSPT) |
| `plcsp_v5/` | current line: v5 environment, agents, baselines, paper experiments |
| `paper/` | Elsevier LaTeX manuscript (`main.tex`, `sections/`, `figures/`) |
| `docs/` | design records, experiment plans, progress logs, citation cards |
| `third_party/` | external repositories (not tracked; see `.gitignore`) |

## v5 in brief

- **Shop**: two workshops, `M` machines, one AGV fleet per workshop. A
  cross-workshop transport is a source leg, a fixed handover delay (`5`), and a
  target leg (`cross_leg_ratio 0.5`).
- **Actors**: job head (which operation), machine head (which machine), AGV head
  (which vehicle; the same head selects the source and target vehicle).
- **Encoders**: job and AGV heads use a Set Transformer; the machine encoder
  splits features into three groups, applies intra-machine attention, then
  inter-machine attention.
- **Training**: GRPO with group size 8, clipped ratio, shared advantage, no
  critic. The top level is PPO over segments of 8 low-level steps.
- **Objective**: `0.4 * makespan + 0.4 * (mean processing energy + idle time) +
  0.2 * loaded transport time`.

## Environments

| Interpreter | Use |
|---|---|
| `D:/anaconda/python.exe` | CPU-only gate: `plcsp*/tests/`, ruff, figure scripts |
| `D:/anaconda/envs/py312/python.exe` | GPU and OR-Tools: training, MIP-AGV baselines |

## Training and evaluation

Run from `plcsp_v5/`:

```bash
python Run.py --n_job 6 --n_machine 6 --n_edge 2 \
  --encoder_type settrans --machine_intra_attn 1 \
  --hier 1 --hier_interval 8 --hier_alpha 0.5 --top_credit local \
  --use_agv 1 --n_agv 6 --cross_delay 5 --cross_leg_ratio 0.5 \
  --agv_dst_actor 1 --agv_rule 0 --agv_top_rule 0 \
  --st_token_context 1 --st_task_features 1 --st_edge_features 0 \
  --algo grpo --episode_num 200 --buffer_size 8 --resample_freq 8 \
  --LR 1e-3 --use_lr_decay 0 --eval_sample 100 \
  --random_weight_type eval --seed 1 --train_seed 1 --run_tag demo
```

Evaluation of a trained checkpoint:

```bash
python eval_st_seed3.py --n_job 6 --n_machine 6 --n_edge 2 \
  --encoder_type settrans --machine_intra_attn 1 \
  --hier 1 --hier_interval 8 --hier_alpha 0.5 --top_credit local \
  --use_agv 1 --n_agv 6 --cross_delay 5 --cross_leg_ratio 0.5 \
  --agv_dst_actor 1 --agv_rule 0 --agv_top_rule 0 \
  --run_tag demo
```

## Baselines

```bash
# Twelve dispatching-rule pairs and RGA (from plcsp_v5/)
python baselines/agv_baselines.py --instances instance/test_Instance_J6M6E2.pkl \
  --mode pdr --agv 1 --out results/agv_pdr_100.json
python baselines/agv_baselines.py --instances instance/test_Instance_J6M6E2.pkl \
  --mode rga --seeds 3 --out results/agv_rga_100x3.json

# Two-stage MIP-AGV, one shard (from the repository root; needs OR-Tools)
D:/anaconda/envs/py312/python.exe -m plcsp_v5.baselines.mip_agv.run_batch \
  --instance plcsp_v5/instance/test_Instance_J6M6E2.pkl \
  --shard 0 --n_shards 4 --out plcsp_v5/results/mip_agv_100/shard0.csv
```

## Manuscript

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Selected v5 results (100 test instances)

| Size | Method | Objective |
|---|---|---:|
| 6×6×2 | PDR (mor/spt) | 951.20 |
| 6×6×2 | B3 (Lei et al. 2022, adapted) | 707.21 |
| 6×6×2 | E2E (Wang et al. 2025 architecture, retrained) | 743.62 |
| 6×6×2 | H-ST-GRPO | 556.70 |
| 6×6×2 | MIP-AGV (two-stage CP-SAT) | 505.05 |
| 6×6×2 | RGA | 478.04 |
| 10×6×2 | PDR (mor/seyc) | 1101.30 |
| 10×6×2 | B3 | 651.30 |
| 10×6×2 | H-ST-GRPO | 576.92 |
| 10×6×2 | MIP-AGV | 503.22 |
| 10×6×2 | RGA | 565.63 |

The 10×10×2 experiments are in progress.

## Data and checkpoints

Instance pickles, trained models, and run outputs are not tracked; see
`plcsp_v5/.gitignore`. The test sets used by the manuscript are
`plcsp_v5/instance/test_Instance_J{6,10}M{6,10}E2.pkl` on the experiment host.
