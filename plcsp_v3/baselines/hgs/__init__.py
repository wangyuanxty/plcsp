"""HGS 对位基线（spec §6.1 的 ④ 层）——从零实现。

出处：Moon, Lee, Park. *Learning-enabled Flexible Job-shop Scheduling for Scalable Smart
Manufacturing.* Journal of Manufacturing Systems 77:356–367, 2024（arXiv:2402.08979）。
**原文未提供代码**（citation-cards §2.7）。本包按原文 §IV–V 实现。

三个模块：

- `env.py`：FJSPT 环境（原文 §IV-B 的 MDP）。**独立仿真器**，不是本仓 `des.py`——见其 docstring。
- `model.py`：HGS 网络（异构编码器 + 三阶段解码器，原文 §IV-C/D）。
- `train.py`：REINFORCE + greedy rollout baseline（原文 Alg.1）。
"""
