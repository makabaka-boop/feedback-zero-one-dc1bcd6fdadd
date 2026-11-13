"""部分识别（随机鼓励 / 二元处理 / 二元结果）纯后端精确求解。

数学设定
========
观测变量均为二元：鼓励 Z、处理 D、结果 Y。
潜在类型为四元组 (d0, d1, y0, y1)，共 16 种（字典序）：
    d0/d1 为 Z=0/1 下接受的处理；y0/y1 为 D=0/1 下的潜在结果。
不假定处理单调，因此 16 种类型全部保留。
共同权重 w_t >= 0、sum_t w_t = 1 描述总体中的类型分布（两组共享同一 w）。

矩阵 A 按行 (z, d, y) 字典序展平（8 行）、列按潜在类型字典序（16 列），
    A[(z,d,y), t] = 1 当且仅当该类型在鼓励 z 下 d_z = d 且 y_{d_z} = y。
p 以同序展平，约束为 A w = p、w >= 0。
目标 c_t = y1 - y0（个体处理效应），求 c·w 的精确最小/最大值。

全部运算使用 fractions.Fraction，不使用浮点、不调用任何优化求解器。
核心为带 Bland 规则的两阶段单纯形（字典形式），第一阶段同时产出
Farkas 不可行证书 h：A^T h >= 0 且 p·h = -1。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from fractions import Fraction
from itertools import product

# ---------------------------------------------------------------------------
# 常量：16 种潜在类型（字典序）与目标系数 c_t = y1 - y0
# ---------------------------------------------------------------------------
TYPES = list(product((0, 1), repeat=4))  # (d0, d1, y0, y1)
N_TYPES = len(TYPES)  # 16
C = [Fraction(t[3] - t[2]) for t in TYPES]  # y1 - y0
ROWS = list(product((0, 1), repeat=3))  # (z, d, y)，与 p 展平顺序一致
N_ROWS = len(ROWS)  # 8

_FRACTION_RE = re.compile(r"^-?[0-9]+/[0-9]+$")


class PIDError(ValueError):
    """输入或求解过程中的可预期错误。"""


# ---------------------------------------------------------------------------
# 矩阵 A 与严格输入解析
# ---------------------------------------------------------------------------
def build_matrix_a() -> list[list[int]]:
    """构造 8x16 的 0/1 矩阵 A，行 (z,d,y) 字典序、列类型字典序。"""
    a = [[0] * N_TYPES for _ in range(N_ROWS)]
    for ri, (z, d, y) in enumerate(ROWS):
        for ci, (d0, d1, y0, y1) in enumerate(TYPES):
            dz = d1 if z == 1 else d0
            ydz = y1 if dz == 1 else y0
            if dz == d and ydz == y:
                a[ri][ci] = 1
    return a


A = build_matrix_a()


def parse_probability(text: object, path: str) -> Fraction:
    """解析非负 p/q 字符串：分子为（可带负号的）整数，分母为正整数。

    禁止小数、空白、多余符号、零分母等。路径用于精确定位非法元素。
    """
    if not isinstance(text, str) or not _FRACTION_RE.match(text):
        raise PIDError(
            f"非法概率（应为 p/q 字符串），路径 {path}，实际值: {text!r}"
        )
    num_s, den_s = text.split("/")
    numerator = int(num_s)
    denominator = int(den_s)
    if denominator <= 0:
        raise PIDError(f"分母必须为正整数，路径 {path}，实际值: {text!r}")
    if numerator < 0:
        raise PIDError(f"概率必须非负，路径 {path}，实际值: {text!r}")
    return Fraction(numerator, denominator)


def _require_exact_shape(obj: object, dims: tuple[int, ...], path: str) -> None:
    if not isinstance(obj, list) or len(obj) != dims[0]:
        raise PIDError(
            f"非法结构：路径 {path} 需要长度为 {dims[0]} 的数组，"
            f"实际: {type(obj).__name__}"
        )
    if len(dims) == 1:
        return
    for i, item in enumerate(obj):
        _require_exact_shape(item, dims[1:], f"{path}[{i}]")


def parse_input(payload: object) -> list[Fraction]:
    """校验顶层 JSON 与 2x2x2 的 p，返回按 (z,d,y) 字典序展平的 Fraction 列表。"""
    if not isinstance(payload, dict):
        raise PIDError("顶层必须为 JSON 对象，且仅含成员 \"p\"")
    if set(payload.keys()) != {"p"}:
        extra = sorted(k for k in payload.keys() if k != "p")
        raise PIDError(f"拒绝未知/多余 JSON 成员: {extra}（仅允许 \"p\"）")

    p3 = payload["p"]
    _require_exact_shape(p3, (2, 2, 2), "p")

    flat: list[Fraction] = []
    for z in range(2):
        slice_sum = Fraction(0)
        for d in range(2):
            for y in range(2):
                value = parse_probability(p3[z][d][y], f"p[{z}][{d}][{y}]")
                flat.append(value)
                slice_sum += value
        if slice_sum != 1:
            raise PIDError(f"给定 Z={z} 的切片概率和必须为 1，实际和为 {slice_sum}")
    return flat


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
    """object_pairs_hook：重复键直接报错（含 NaN/Infinity 常量一并拒绝）。"""
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise PIDError(f"拒绝重复 JSON 成员: {key!r}")
        result[key] = value
    return result


def load_input_file(path: str) -> object:
    """以 UTF-8 严格模式读取并解析 JSON，拒绝 NaN/Infinity 与重复键。"""
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as fh:
            text = fh.read()
    except OSError as exc:
        raise PIDError(f"无法读取输入文件 {path}: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda c: (_ for _ in ()).throw(
                PIDError(f"拒绝 JSON 常量 {c}，仅允许普通 JSON 数值/字符串/结构")
            ),
        )
    except json.JSONDecodeError as exc:
        raise PIDError(f"JSON 解析失败（{path}）: {exc}") from exc


# ---------------------------------------------------------------------------
# 精确线性代数辅助
# ---------------------------------------------------------------------------
def _solve_square(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """对非奇异方阵做精确高斯-若尔当消元，返回解；奇异则报错。"""
    m = len(matrix)
    if m == 0:
        return []
    if any(len(row) != m for row in matrix) or len(rhs) != m:
        raise PIDError("内部错误：方阵求解维度不一致")
    tab = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(m):
        pivot = next((r for r in range(col, m) if tab[r][col] != 0), None)
        if pivot is None:
            raise PIDError("内部错误：基矩阵奇异")
        if pivot != col:
            tab[col], tab[pivot] = tab[pivot], tab[col]
        piv = tab[col][col]
        tab[col] = [v / piv for v in tab[col]]
        for r in range(m):
            if r == col or tab[r][col] == 0:
                continue
            factor = tab[r][col]
            tab[r] = [rv - factor * cv for rv, cv in zip(tab[r], tab[col])]
    return [tab[r][m] for r in range(m)]


def _pivot(
    basis: list[int],
    nonbasis: list[int],
    tab: list[list[Fraction]],
    rhs: list[Fraction],
    cbar: list[Fraction],
    obj: Fraction,
    enter: int,
    leave_row: int,
) -> tuple[list[int], list[int], list[list[Fraction]], list[Fraction], list[Fraction], Fraction]:
    """规范表（dictionary）转轴。

    规范形式：sum_j tab[i][j] * x_{N,j} + x_{B,i} = rhs[i]，rhs >= 0，
    即 x_{B,i} = rhs[i] - tab[i]·x_N。目标 z = obj + cbar·x_N
    （最小化问题中 cbar_j < 0 表示 x_j 进基可改进）。
    进基变量在离基行满足 tab[leave_row][enter] > 0。
    """
    leave = basis[leave_row]
    enter_pos = nonbasis.index(enter)
    a = tab[leave_row][enter_pos]  # 严格为正

    new_nonbasis = [leave if name == enter else name for name in nonbasis]
    new_tab = [[Fraction(0) for _ in nonbasis] for _ in basis]
    new_rhs = [Fraction(0) for _ in basis]

    # 离基行归一化：x_enter = rhs/a - (1/a) x_leave
    #             - sum_{j != enter} (tab_lj/a) x_j
    new_rhs[leave_row] = rhs[leave_row] / a
    for j in range(len(nonbasis)):
        if j == enter_pos:
            new_tab[leave_row][j] = Fraction(1) / a  # 新非基变量 x_leave 的系数
        else:
            new_tab[leave_row][j] = tab[leave_row][j] / a

    # 其余行消去进基列
    for r in range(len(basis)):
        if r == leave_row:
            continue
        factor = tab[r][enter_pos]
        new_rhs[r] = rhs[r] - factor * rhs[leave_row] / a
        for j in range(len(nonbasis)):
            if j == enter_pos:
                new_tab[r][j] = -factor / a
            else:
                new_tab[r][j] = tab[r][j] - factor * tab[leave_row][j] / a

    # 目标行消去进基列
    alpha = cbar[enter_pos]
    new_cbar = [Fraction(0) for _ in nonbasis]
    for j in range(len(nonbasis)):
        if j == enter_pos:
            new_cbar[j] = -alpha / a
        else:
            new_cbar[j] = cbar[j] - alpha * tab[leave_row][j] / a
    new_obj = obj + alpha * rhs[leave_row] / a

    new_basis = basis[:]
    new_basis[leave_row] = enter
    return new_basis, new_nonbasis, new_tab, new_rhs, new_cbar, new_obj


# ---------------------------------------------------------------------------
# 两阶段单纯形（最小化 c·x，约束 M x = b，x >= 0；全部精确分数）
# ---------------------------------------------------------------------------
def simplex_min(
    mat: list[list[Fraction]],
    b: list[Fraction],
    c: list[Fraction],
) -> dict:
    """求解 LP。返回 {"status": "optimal", ...} 或 {"status": "infeasible", "h": ...}。

    不可行证书 h 针对调用方给出的原系统 mat x = b：
        mat^T h >= 0 且 b·h = -1。
    """
    m = len(mat)
    n = len(mat[0]) if m else 0
    if any(len(row) != n for row in mat) or len(b) != m or len(c) != n:
        raise PIDError("内部错误：LP 维度不一致")

    # 全部元素强制转为 Fraction，避免 int/int 触发 Python 浮点除法
    work_mat = [[Fraction(v) for v in row] for row in mat]
    work_b = [Fraction(v) for v in b]
    c_frac = [Fraction(v) for v in c]
    # b 中的负分量整行取反，记录翻转符号（证书需还原）
    flip = [Fraction(1)] * m
    for i in range(m):
        if work_b[i] < 0:
            work_b[i] = -work_b[i]
            work_mat[i] = [-v for v in work_mat[i]]
            flip[i] = Fraction(-1)

    # 初始基：全部人工变量 a_i = n+i；非基仅为 n 个原变量
    # 规范行：x_{a_i} + sum_j M_ij x_j = b_i，即 tab = M
    basis = [n + i for i in range(m)]
    nonbasis = list(range(n))
    tab = [work_mat[i][:] for i in range(m)]
    rhs = work_b[:]

    # ---- 第一阶段目标：最小化 sum_i a_i ----
    # 代入规范行 a_i = b_i - M_i·x：z = sum b - (sum_i M_i)·x
    cbar = [Fraction(0) for _ in nonbasis]
    for i in range(m):
        for j in range(len(nonbasis)):
            cbar[j] -= tab[i][j]
    obj = sum(rhs, Fraction(0))

    def bland_step(
        basis: list[int],
        nonbasis: list[int],
        tab: list[list[Fraction]],
        rhs: list[Fraction],
        cbar: list[Fraction],
        obj: Fraction,
    ) -> tuple | None:
        # Bland 进基：非基变量编号最小且简约费用严格为负
        enter_pos = None
        for j, name in enumerate(nonbasis):
            if cbar[j] < 0:
                if enter_pos is None or name < nonbasis[enter_pos]:
                    enter_pos = j
        if enter_pos is None:
            return None
        enter = nonbasis[enter_pos]
        # 规范行 x_B = rhs - tab·x_N：进基要求 tab_i,enter > 0，
        # 比值测试 rhs_i / tab_i,enter；平局取基变量编号最小（Bland）
        best = None
        for i in range(len(basis)):
            coef = tab[i][enter_pos]
            if coef <= 0:
                continue
            ratio = rhs[i] / coef
            tie_break = basis[i]
            if best is None or ratio < best[0] or (ratio == best[0] and tie_break < best[1]):
                best = (ratio, tie_break, i)
        if best is None:
            raise PIDError("内部错误：第一/二阶段出现无界（本问题含等式和约束，不应发生）")
        leave_row = best[2]
        return _pivot(basis, nonbasis, tab, rhs, cbar, obj, enter, leave_row)

    # 第一阶段迭代
    while True:
        step = bland_step(basis, nonbasis, tab, rhs, cbar, obj)
        if step is None:
            break
        basis, nonbasis, tab, rhs, cbar, obj = step

    if obj != 0:
        # 不可行：构造第一阶段对偶证书。
        # 第一阶段 LP 为 min 1^T x_a s.t. work_mat x + I x_a = work_b、x,x_a >= 0。
        # 其对偶：max work_b·h s.t. work_mat^T h <= 0、h <= 1。
        # 由当前基解对偶 h：对全部基变量（原变量或人工变量）取紧约束。
        #   人工基变量 a_i 在基 -> h_i = 1；
        #   原变量 x_j 在基 -> (work_mat^T h)_j = 0。
        # 得到的 h 满足 work_mat^T h <= 0、h <= 1 且 work_b·h = obj > 0。
        rhs_h: list[Fraction] = []
        mat_h: list[list[Fraction]] = []
        for i, bname in enumerate(basis):
            row = [Fraction(0)] * m
            if bname >= n:  # 基变量为人工变量 a_{bname-n}
                row[bname - n] = Fraction(1)
                rhs_h.append(Fraction(1))
            else:  # 基变量为原变量
                for rr in range(m):
                    row[rr] = work_mat[rr][bname]
                rhs_h.append(Fraction(0))
            mat_h.append(row)
        h_work = _solve_square(mat_h, rhs_h)
        # 独立验证 work_mat^T h <= 0 与 h <= 1
        for j in range(n):
            if sum(work_mat[i][j] * h_work[i] for i in range(m)) > 0:
                raise PIDError("内部错误：不可行证书违反 work_mat^T h <= 0")
        for i in range(m):
            if h_work[i] > 1:
                raise PIDError("内部错误：不可行证书违反 h <= 1")
        dot = sum(work_b[i] * h_work[i] for i in range(m))
        if dot != obj or dot <= 0:
            raise PIDError("内部错误：不可行证书构造失败")
        # 缩放并按行翻转还原为原系统证书：A^T h >= 0、p·h = -1
        h_orig = [-flip[i] * h_work[i] / dot for i in range(m)]
        if sum(b[i] * h_orig[i] for i in range(m)) != -1:
            raise PIDError("内部错误：不可行证书点积校验失败")
        return {"status": "infeasible", "h": h_orig}

    # ---- 可行：把仍在基中的人工变量（必为退化 0 值）转轴出去 ----
    # 若某行原变量系数全为 0，则该行冗余，直接删除。
    keep_rows: list[int] = []
    for i in range(m):
        if basis[i] < n:
            keep_rows.append(i)
            continue
        # 基变量是人工变量：尝试用该行上任意非零原变量替换
        enter_pos = next(
            (j for j, name in enumerate(nonbasis) if name < n and tab[i][j] != 0),
            None,
        )
        if enter_pos is None:
            continue  # 冗余行，稍后删除
        enter = nonbasis[enter_pos]
        basis, nonbasis, tab, rhs, cbar, obj = _pivot(
            basis, nonbasis, tab, rhs, cbar, obj, enter, i
        )
        keep_rows.append(i)

    # 保留行下标与原约束行号一一对应（转轴不改变行的顺序）
    row_map = keep_rows[:]
    basis = [basis[i] for i in keep_rows]
    rhs = [rhs[i] for i in keep_rows]
    tab = [tab[i] for i in keep_rows]
    # 剔除全部人工变量列
    keep_cols = [j for j, name in enumerate(nonbasis) if name < n]
    nonbasis = [nonbasis[j] for j in keep_cols]
    tab = [[row[j] for j in keep_cols] for row in tab]
    mr = len(basis)

    # ---- 第二阶段：换回原目标 c ----
    cbar2 = [Fraction(0) for _ in nonbasis]
    for j, name in enumerate(nonbasis):
        cbar2[j] = c_frac[name]
    for i, bname in enumerate(basis):
        for j in range(len(nonbasis)):
            cbar2[j] -= c_frac[bname] * tab[i][j]
    obj2 = sum(
        (c_frac[bname] * rhs[i] for i, bname in enumerate(basis)),
        Fraction(0),
    )

    while True:
        step = bland_step(basis, nonbasis, tab, rhs, cbar2, obj2)
        if step is None:
            break
        basis, nonbasis, tab, rhs, cbar2, obj2 = step

    # 原始变量解
    x = [Fraction(0) for _ in range(n)]
    for i, name in enumerate(basis):
        x[name] = rhs[i]

    # 对偶乘子：由基矩阵 B（使用 work_mat 的保留行）解 B^T lambda = c_B
    if mr == 0:
        lam_work = [Fraction(0) for _ in range(m)]
    else:
        bmat = [[work_mat[row_map[i]][bj] for i in range(mr)] for bj in basis]
        c_b = [c_frac[bj] for bj in basis]
        lam_kept = _solve_square(bmat, c_b)
        lam_work = [Fraction(0) for _ in range(m)]
        for i, row_idx in enumerate(row_map):
            lam_work[row_idx] = lam_kept[i]

    # 行翻转还原到调用方原系统
    lam = [flip[i] * lam_work[i] for i in range(m)]
    return {
        "status": "optimal",
        "x": x,
        "lam": lam,
        "obj": obj2,
    }


# ---------------------------------------------------------------------------
# 问题封装与输出构造
# ---------------------------------------------------------------------------
def _frac_str(value: Fraction) -> str:
    """约分后的 p/q 字符串；分母恒正，负号落在分子上。"""
    return f"{value.numerator}/{value.denominator}"


def solve_partial_identification(p_flat: list[Fraction]) -> dict:
    """求解完整问题并在返回前重算全部等式/不等式与证书（独立自检）。"""
    if len(p_flat) != N_ROWS:
        raise PIDError("内部错误：p 展平长度应为 8")

    lower = simplex_min(A, p_flat, C)
    if lower["status"] == "infeasible":
        h = lower["h"]
        # 独立重算证书：A^T h >= 0、p·h = -1
        for t in range(N_TYPES):
            dot = sum(A[r][t] * h[r] for r in range(N_ROWS))
            if dot < 0:
                raise PIDError("内部错误：不可行证书违反 A^T h >= 0")
        if sum(p_flat[r] * h[r] for r in range(N_ROWS)) != -1:
            raise PIDError("内部错误：不可行证书违反 p·h = -1")
        return {"infeasible": True, "h": [_frac_str(v) for v in h]}

    upper = simplex_min(A, p_flat, [-v for v in C])
    if upper["status"] != "optimal":
        raise PIDError("内部错误：上界 LP 意外不可行")

    w_l, lam = lower["x"], lower["lam"]
    w_u, lam_neg = upper["x"], upper["lam"]
    mu = [-v for v in lam_neg]  # 最大化 c·w 的对偶 = 最小化 -c·w 对偶取负
    L = lower["obj"]
    U = -upper["obj"]

    # ---- 独立自检：重算全部约束 ----
    for tag, w in (("下界", w_l), ("上界", w_u)):
        if any(v < 0 for v in w):
            raise PIDError(f"内部错误：{tag}权重含负值")
        if sum(w) != 1:
            raise PIDError(f"内部错误：{tag}权重和不为 1")
        for r in range(N_ROWS):
            if sum(A[r][t] * w[t] for t in range(N_TYPES)) != p_flat[r]:
                raise PIDError(f"内部错误：{tag}权重不满足 A w = p（行 {r}）")
    if sum(C[t] * w_l[t] for t in range(N_TYPES)) != L:
        raise PIDError("内部错误：下界目标值不一致")
    if sum(C[t] * w_u[t] for t in range(N_TYPES)) != U:
        raise PIDError("内部错误：上界目标值不一致")
    if L > U:
        raise PIDError("内部错误：下界大于上界")

    # 对偶：A^T lambda <= c、A^T mu >= c；p·lambda = L、p·mu = U
    for t in range(N_TYPES):
        at_lam = sum(A[r][t] * lam[r] for r in range(N_ROWS))
        at_mu = sum(A[r][t] * mu[r] for r in range(N_ROWS))
        if at_lam > C[t]:
            raise PIDError(f"内部错误：下界乘子违反 A^T λ <= c（类型 {t}）")
        if at_mu < C[t]:
            raise PIDError(f"内部错误：上界乘子违反 A^T μ >= c（类型 {t}）")
    if sum(p_flat[r] * lam[r] for r in range(N_ROWS)) != L:
        raise PIDError("内部错误：p·λ != L")
    if sum(p_flat[r] * mu[r] for r in range(N_ROWS)) != U:
        raise PIDError("内部错误：p·μ != U")

    return {
        "infeasible": False,
        "bounds": [_frac_str(L), _frac_str(U)],
        "lower_weights": [_frac_str(v) for v in w_l],
        "upper_weights": [_frac_str(v) for v in w_u],
        "lambda": [_frac_str(v) for v in lam],
        "mu": [_frac_str(v) for v in mu],
    }


# ---------------------------------------------------------------------------
# 入口：UTF-8 JSON 输入 -> 成功才原子替换输出，失败保留旧文件
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="随机鼓励设计二元处理/二元结果的部分识别精确求解器"
    )
    parser.add_argument("input", help="仅含成员 p 的 UTF-8 JSON 文件路径")
    parser.add_argument("output", help="结果 JSON 输出路径（成功才替换，失败保留旧文件）")
    args = parser.parse_args(argv)

    try:
        payload = load_input_file(args.input)
        p_flat = parse_input(payload)
        result = solve_partial_identification(p_flat)
    except PIDError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".pid_out_", suffix=".tmp", dir=out_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp_path, args.output)  # 原子替换
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    except OSError as exc:
        print(f"错误: 写出结果失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
