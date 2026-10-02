"""Token 特征构造（《方法设计文档》§2.2）。

⚠️ 2026-10-02 状态（**必读**）：
  **几何特征已整条删除**（几何/度量感知路线砍除，见 `progress-log.md` §12.6）。
  删除后本模块**尚无替代实现**——`state_emb.encode_state` 目前产出的 token 特征**全为零向量**。

  **P2（骨架）必须重新设计 token 特征。** 依据：旧版除 4 位静态几何外，其余全是桩 0——
  动态量（in/out 缓冲占用、占道、负载、电量、等待）**从未接入过**（旧 docstring 标"② 接入"，
  但一直是 `np.zeros`）。**因此在 P2 完成前，编码器等价于吃零输入。**
"""
from __future__ import annotations

F_DYN = 6      # 动态特征位数（具体语义待 P2 定义）


def token_feature_dim(n_dyn: int = F_DYN) -> int:
    """token 特征维度。

    2026-10-02：原为 `n_geomfeat(4) + n_dyn(6) = 10`；几何 4 位已删，故 = n_dyn。
    """
    return n_dyn
