"""几何不变特征与布局描述器（《方法设计文档》§2.1–§2.2）。

不变性四规则：平移（只相对量）/ 旋转（PCA 主轴系）/ 缩放（char_len 归一）/ 拓扑（可变长）。
特征函数训练与部署共用（同一函数、来源无关——描述器 schema 保证，零样本机制基石）。
动态量（缓冲占用/占道计数）由 M3 的 state 快照提供（本节含静态几何侧与偏置相似度函数）。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ..env.layout import Layout


def pca_axes(xy: np.ndarray) -> np.ndarray:
    """机台坐标系主轴（2×2，列=主轴，按奇异值降序）。旋转不变（工厂转 90°→轴随之旋转）。"""
    c = np.asarray(xy, float) - np.asarray(xy, float).mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt.T


def project_to_principal(xy: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """投影到主轴系：相对坐标 (u,v)（无绝对位置）。"""
    return (np.asarray(xy, float) - np.asarray(xy, float).mean(0)) @ axes


@dataclass
class GeometryFeatures:
    char_len: float            # §2.2 归一化长度（包围盒对角线）
    axes: np.ndarray           # (2,2) 主轴
    proj: np.ndarray           # (n_m,2) 主轴相对坐标（u,v）
    dist_norm: np.ndarray      # (n_m,n_m) 装卸点距离 / char_len
    corridor_deg: np.ndarray   # (n_m,) 走廊瓶颈度（第 k 邻距离倒数均值）
    shape: tuple = (0, 0)


def geometry_features(layout: Layout, k_neigh: int = 3) -> GeometryFeatures:
    xy = np.array([m.dock for m in layout.machines], dtype=float)
    axes = pca_axes(xy)
    proj = project_to_principal(xy, axes)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    dn = d / max(layout.char_len, 1e-9)
    if len(xy) > k_neigh:
        kth = np.sort(np.where(dn > 0, dn, np.inf), axis=1)[:, k_neigh - 1]
        corridor_deg = 1.0 / np.maximum(kth, 1e-9)
    else:
        corridor_deg = np.zeros(len(xy), dtype=float)
    return GeometryFeatures(char_len=layout.char_len, axes=axes, proj=proj,
                            dist_norm=dn, corridor_deg=corridor_deg, shape=xy.shape)


def conf_sim(occ: np.ndarray) -> np.ndarray:
    """拥堵相似度：资产占道/拥堵程度越接近、越应互相注意（几何偏置第 2 项）。

    conf_sim(i,j) = exp(-|occ_i - occ_j| / (occ_max - occ_min + eps))：同拥堵→1，最悬殊→~e^-1。
    """
    a = np.asarray(occ, float)
    m = float(a.max() - a.min()) if a.size else 0.0
    return np.exp(-np.abs(a[:, None] - a[None, :]) / (m + 1e-9))


def token_feature_dim(n_geomfeat: int = 4, n_dyn: int = 6) -> int:
    """机器 token 特征维度 = 静态几何（4）+ 动态（6：in/out 缓冲占用、占道、负载、电量、等待）。"""
    return n_geomfeat + n_dyn
