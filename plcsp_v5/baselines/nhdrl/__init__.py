# -*- coding: utf-8 -*-
"""Adapted NHDRL baseline for the v5 production-logistics shop.

Source paper
------------
J. Shi, F. Qiao, J. Liu, Y. Ma, D. Wang, C. Ding,
"Production-logistics collaborative scheduling in dynamic flexible job
shops using nested-hierarchical deep reinforcement learning",
Advanced Engineering Informatics 65 (2025) 103195.
doi:10.1016/j.aei.2025.103195

Status
------
**Adapted baseline, not a reproduction.**  No official implementation was
found.  GitHub repository/code search and a web search on 2026-10-27 returned
no code link.  The paper's data availability statement says "Data will be
made available on request."  The implementation below follows the paper's
Section 4.2, Table 3 (actions), Table 4 (network widths), Table 6
(hyper-parameters), and Algorithm 1, with the deviations listed here.

Nested hierarchy implemented
----------------------------
* OA (objective agent): 2 actions, select MS or the v5 non-makespan cost.
* PA (production agent): 9 composite dispatching rules (3 job rules x
  3 machine rules).
* LA (logistics agent): 3 AGV dispatching rules.
* At each decision point OA acts first, its action enters the PA and LA
  states (feature f32), then PA selects an operation and machine, then LA
  selects an AGV.  All three agents receive the same step reward.

Deviations from the paper
-------------------------
1. No official code.  Implemented from the paper text.
2. Static instances.  The paper's disturbance is AGV breakdown.  v5 has no
   AGV failure in this baseline.  Decision points are operation assignments
   only; there are no failure-triggered decisions.
3. Two workshops with per-workshop AGV fleets.  The paper's AGVs can serve
   the whole shop.  For a cross-workshop move, the source AGV and the
   destination AGV are selected separately by applying the same LA rule to
   each fleet.  The LA context features f33-f48 use the source fleet.
4. No load/unload station and no return trip.  The first operation of a job
   and a move to the same machine need no transport, so LA does not act at
   those decision points.  The paper gives every operation a logistics task
   (origin station and return).
5. Objective.  MS = makespan.  The paper's TC = c * sum(y_v) +
   uc * transport time.  v5 has no AGV commissioning cost and no transport
   energy.  The second objective is therefore the non-makespan part of the
   v5 objective:
       C = 0.4 * (processing energy / n_task + machine idle energy)
           + 0.2 * loaded transport time
       J = 0.4 * MS + C
   Machine idle follows the published definition sum_k (MKT - P_k);
   --idle-mode gap restores the old leading-idle-plus-gaps reading.
   This keeps OA's two-objective trade-off over the v5 cost components.
6. Reward.  w1 = 0.8, w2 = 0.2; neutral preferences p_ms = p_tc = 5 with
   maxp = 10, as in the paper's non-sensitivity setting.  The MS signal is
   the partial-makespan gain (MS_before - MS_after, <= 0); the cost signal
   is the change of C (C_after - C_before, which can be negative because
   filling an idle gap reduces machine idle energy).  Both are normalised
   to [0, 1]; by default they share one standardised scale (see deviation
   19c).  The paper does not state the sign convention of mst; this
   implementation chooses the convention under which a smaller makespan
   growth and a smaller cost increase give a higher reward.
   The normalised cost reward is clipped to [0, 1].
7. State features.  The dimensions are 31 / 24 / 30 as in the paper, but:
   (a) there is no dummy machine and no L/U station;
   (b) machine completion time MCT is the end of the last operation on the
       machine (v5 insertion scheduling);
   (c) f9 "activated AGVs" counts AGVs that have carried at least one task;
   (d) f17 is the cumulative adapted non-makespan cost C, not the paper's
       pure logistics cost;
   (e) all time-like features are divided by the instance's mean compatible
       processing time, and count features are scaled; the paper uses raw
       values;
   (f) the LA context features use the source fleet, as explained in
       deviation 3.
8. PA rules.  SPT and SRPT use the mean compatible processing time (APT).
   EAJ estimates the job's earliest start with direct machine-to-machine
   travel; it ignores AGV deadhead and waiting because PA acts before LA.
   EAM uses the earliest feasible insertion start rather than the paper's
   append-only machine completion time.  MINU uses processing time divided
   by machine completion time.
9. Ties are broken deterministically by the lowest job/machine/AGV index.
   The paper selects randomly among ties.
10. Networks.  Actor 256 hidden ReLU, critic 128 hidden ReLU (paper Table
    4).  The output layers are linear (categorical logits and scalar
    value); the paper's Tanh output cannot produce a categorical policy.
11. Training.  Independent PPO per agent with its own critic on its own
    state and a shared reward.  This matches the paper's Table 4 per-agent
    critic input; no centralised joint critic or mixing network is used.
    PPO epochs = 20, minibatch = 32, clip = 0.2, GAE lambda = 0.95,
    gamma = 0.99, entropy coefficient = 0.01, value coefficient = 0.5,
    gradient norm clip = 0.5.  The paper does not report the number of PPO
    epochs; 20 is used because 4 underfits at a 100-episode budget.
12. Learning rates.  The paper's tuned level 2 is OA/PA 1e-4 and LA 1e-3.
    The default here is 1e-3 for all agents because the training budget is
    100 episodes.  Use --lr-oa 1e-4 --lr-pa 1e-4 --lr-la 1e-3 for the
    paper rates.
13. Memory and updates.  The paper uses capacities 32 / 32 / 500 and
    updates when a minibatch is full.  Here OA and PA are updated at the end
    of each episode on the whole episode (cap 1000; 36 steps for 6x6), and
    LA on all transport decisions of the episode (<= 100).  This keeps PPO
    updates on-policy; the paper's short upper-level memory is not copied
    because it is too small for 100-episode learning.
14. Exploration.  epsilon-greedy with a uniform random action, epsilon from
    0.9 to 0.1 with 0.99 decay per episode.  The paper adds noise to the
    actor's action distribution.  The CLI exposes the epsilon parameters.
15. Terminal reward.  The paper uses per-step rewards only.  A terminal
    reward ``-terminal_coef * J / (t_scale * n_task)`` is added to the last
    step, with ``terminal_coef = 1.0`` by default.  Without it the
    100-episode policy often fails to learn a competitive rule.  Set
    ``--terminal-coef 0`` for the paper's reward design.
16. Training budget and data.  100 episodes by default (paper: 1000 epochs,
    about 6.78 s each).  By default the policy is trained on the 6x6
    training split (``Instance_J6M6E2.pkl``).  The CLI accepts
    ``--train-mix 6x6,10x6,10x10``; 10x6 and 10x10 instances are generated
    with the v5 generator into a temporary directory.  The JSON records the
    exact pool sizes.  A 6x6-trained policy is also evaluated zero-shot on
    10x6 and 10x10.
17. Evaluation.  One deterministic greedy schedule per instance, scored by
    the v5 weighted objective.  The paper reports 20 repeated runs and
    Pareto HV/SP indicators.  A publishable comparison must add multiple
    seeds, Pareto/HV/SP, and the paper's rule/NSGA-II/Gurobi baselines.
18. The simulator is checked against
    `plcsp_v5.baselines.agv_baselines.decode`, which matches the v5
    `Parallel_env` to about 1e-4.  Run `--check` to repeat the check.
19. Implementation fixes (on by default; disable each flag to restore the
    old behaviour):
    (a) PPO stored the actor log probability for epsilon-greedy random
        actions, although the behaviour policy is
        ``(1 - eps) * pi + eps / |A|``.  ``--fix-behaviour-logprob 1`` stores
        the behaviour-policy log probability;
    (b) the reward min/max only widened, so early extremes fixed the range.
        ``--fix-reward-window 1`` keeps the last ``--reward-window`` (2000)
        observations;
    (c) the MS and cost OA branches used separate running ranges.  With
        ``--fix-common-reward-scale 1`` each branch's "badness" signal (the
        negated MS gain and the cost delta) is divided by its own running
        standard deviation over the same window, so one standard deviation
        of either branch maps to the same reward change before the weighted
        sum.

Entry point
-----------
    D:/anaconda/python.exe plcsp_v5/baselines/nhdrl_baseline.py \
        --instances plcsp_v5/instance/test_Instance_J6M6E2.pkl \
        --limit 100 --epochs 100 --out <temporary path>/nhdrl.json

The v5 `baselines` folder is a namespace package; the thin wrapper
`plcsp_v5/baselines/nhdrl_baseline.py` adds the needed paths.
"""
