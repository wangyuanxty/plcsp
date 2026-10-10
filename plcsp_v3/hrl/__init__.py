# -*- coding: utf-8 -*-
"""`hrl/`：超启发式的**规则集**与（步 4 起）分层策略的网络。

`rules.py` 只依赖 `env/rule_views.py` 的纯数据视图，**不 import `env/des.py`**——
层次纪律：规则层不认识 SimPy 对象，`env/` 也不认识策略。
"""
