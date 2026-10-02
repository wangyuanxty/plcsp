"""标准实例加载器：Brandimarte MK 系列（.fjs 文本格式）+ 最优解表。

文档依据：《方法设计文档》§1.1 实例规格（S=Kacem 8×8 / M=MK01 / L=MK08, 以官方文件为准）。
来源：third_party/fjsp-gnnrl/evaluations/standard/brandimarte/（标准 .fjs + 最优解）。
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

# MK 最优解（Brandimarte；来源 brandimarte_optimal.txt）
MK_OPTIMAL: dict[str, int] = {
    "mk01": 40, "mk02": 27, "mk03": 204, "mk04": 60, "mk05": 172,
    "mk06": 58, "mk07": 139, "mk08": 523, "mk09": 299, "mk10": 205,
}
def load_kacem_8x8():
    """Kacem 2002 8×8 实例——**数据待录入**（本地克隆库无此文件；论文数据必须真实，禁止编造）。

    录入路径：从 Kacem et al. 2002 原表转录（多篇文献 Table 1 载有全表），或提供文件后经
    parse_fjs_text 加载。S 档调试先以 gen_random() 代替（仅调试，不进入论文数据）。
    """
    raise NotImplementedError(
        "Kacem 8×8 数据待录入（见 docstring）；调试请用 gen_random()")


# Kacem 8×8 经典最优（Kacem et al. 2002 报告值，多重最优之一）
KACEM_8X8_OPTIMAL = 14


@dataclass
class Instance:
    n_jobs: int
    n_machines: int
    jobs: list[list[list[tuple[int, float]]]]   # job -> op -> [(mach, time), ...]
    source: str = ""


def _parse_fjs_body(toks: list[str], header_len: int) -> tuple[int, int, list, int]:
    """从 token 流解析实例主体。header_len = 表头 token 数（2 或 3）。返回 (n, m, jobs, end_idx)。"""
    idx = header_len
    n = int(toks[0])
    m = int(toks[1])
    jobs = []
    for _ in range(n):
        n_ops = int(toks[idx]); idx += 1
        job_ops = []
        for _ in range(n_ops):
            n_alt = int(toks[idx]); idx += 1
            alts = []
            for _ in range(n_alt):
                mach = int(toks[idx]) - 1
                t = float(toks[idx + 1]); idx += 2
                alts.append((mach, t))
            job_ops.append(alts)
        jobs.append(job_ops)
    return n, m, jobs, idx


def parse_fjs_text(text: str) -> tuple[int, int, list[list[list[tuple[int, float]]]]]:
    """解析标准 Brandimarte .fjs 文本。

    **表头格式：`n_jobs n_machines [avg_ops]`——第三个 token 必须被消费**
    （bug#12：不消费则后续 token 全部错位；Mk02 的该 token 是 `3.5`，会直接崩）。
    无第三 token 的变体亦兼容（按"必须恰好消费完 token 流"回退判定）。

    主体格式：每作业先给工序数 K，随后 K 组；每组先给候选机台数 A，再给 A 对
    `(机台号 工时)`，机台号为 **1 基**（此处转为 0 基）。
    """
    toks = text.split()
    if len(toks) < 2:
        raise ValueError("fjs 文本格式错误（首行缺少 n m）")

    last_err: Exception | None = None
    for header_len in (3, 2):        # 先试带 avg_ops 的三 token 头（Brandimarte 标准）
        try:
            n, m, jobs, end = _parse_fjs_body(toks, header_len)
        except (ValueError, IndexError) as e:
            last_err = e
            continue
        if end == len(toks):         # 判据：必须恰好消费完，无残留 token
            return n, m, jobs
        last_err = ValueError(f"token 未消费完（{end}/{len(toks)}）")
    raise ValueError(f"fjs 解析失败：{last_err}")


def find_fjs(name: str, base: Path) -> Path | None:
    hits = list(base.glob(f"*{name.lower()}*.fjs")) + list(base.glob(f"*{name.upper()}*.fjs"))
    return hits[0] if hits else None


def load_mk(name: str = "mk01", base: Path | None = None) -> Instance:
    """加载 MK 系列实例——**读官方 `.fjs` 原文**。

    ⚠️ bug#12（2026-10-02）：此前读 `brandimarte_dataset_numpy/*.npy`，该转换产物
    的 shape 与官方不符（Mk03: 10 vs 8 机、Mk05: 9 vs 4、Mk08/09: 14 vs 10），
    内容亦被错位——MK01 逐工序对比 **54/55 不符**。**npy 路径已废弃**。
    官方 `.fjs` 经回归测试核对：10/10 实例的维度与总工序数全部正确。
    见 `plcsp/tests/test_instances.py`。
    """
    # 夹具来源（P0 整支评审发现 #3 的修复）：10 个 .fjs 共约 20 KB，随包存于
    # plcsp/data/brandimarte/，**不再依赖被 .gitignore 排除的 third_party/**（1.5 GB / 33 嵌套 .git）。
    # 原始出处：third_party/fjsp-gnnrl/evaluations/standard/brandimarte/brandimarte_dataset/
    if base is None:
        base = Path(__file__).resolve().parents[1] / "data" / "brandimarte"
    f = base / f"Mk{int(name[2:]):02d}.fjs" if name[2:].isdigit() else None
    if f is None or not f.exists():
        avail = sorted(p.name for p in base.glob("Mk*.fjs"))[:12] if base.exists() else []
        raise FileNotFoundError(f"未找到 {name}.fjs 于 {base}；可用：{avail}")
    n, m, jobs = parse_fjs_text(f.read_text())
    return Instance(n_jobs=n, n_machines=m, jobs=jobs, source=str(f))


def gen_random(n_jobs: int, n_machines: int, seed: int, n_alt: int = 2,
               t_lo: float = 2.0, t_hi: float = 8.0) -> Instance:
    """合成实例（调试用）：每作业 3~5 工序，每工序最多 n_alt 个候选机台。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(n_jobs):
        n_ops = int(rng.integers(3, 6))
        job_ops = []
        for _ in range(n_ops):
            picked = rng.choice(n_machines, size=min(n_alt, n_machines), replace=False)
            job_ops.append([(int(m), float(rng.uniform(t_lo, t_hi))) for m in picked])
        jobs.append(job_ops)
    return Instance(n_jobs=n_jobs, n_machines=n_machines, jobs=jobs,
                    source=f"random n={n_jobs} m={n_machines} seed={seed}")
