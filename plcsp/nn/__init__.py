"""plcsp.nn：token 表征层。

- `encoder.py`   —— 编码器（当前为双轴掩码注意力；P2 将改为标准 Transformer）
- `features.py`  —— token 特征定义
- `state_emb.py` —— Instance + Layout → 编码器输入

🔴 **当前 token 特征全为零桩**——动态量从未接入过，几何特征已删（`progress-log.md` §12.6）。
   **P2 必须重新设计 token 特征**；在此之前编码器路径无意义。
"""
