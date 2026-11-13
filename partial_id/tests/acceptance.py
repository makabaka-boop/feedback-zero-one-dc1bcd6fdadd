"""独立验收测试：不调用 pid 的求解逻辑，仅复用其数据结构定义。

独立内容：
  1. 用 RREF + 枚举 rank(A) 个列子集的方法独立重写 LP 求解（可行域枚举全部
     基可行解），重算上下界并与 pid 的单纯形结果逐项比对；
  2. 重算 A w = p、w >= 0、sum w = 1、c·w = L/U；
  3. 重算对偶 A^T λ <= c、A^T μ >= c、p·λ = L、p·μ = U；
  4. 不可行实例重算 Farkas 证书 A^T h >= 0、p·h = -1；
  5. 校验全部输出均为约分、正分母的 p/q 字符串；
  6. 覆盖退化可行域、共享权重导致的不可行、负下界、随机模糊测试、
     非法输入拒绝、旧输出保留、CLI 端到端。

运行: python3 -I tests/acceptance.py
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from fractions import Fraction
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pid  # noqa: E402  仅复用 TYPES/A/C/解析，不复用求解函数

FAILURES: list[str] = []


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"  PASS {message}")
    else:
        FAILURES.append(message)
        print(f"  FAIL {message}")


# ---------------------------------------------------------------------------
# 独立 RREF 工具与枚举求解器（与 pid.simplex_min 完全独立的实现）
# ---------------------------------------------------------------------------
def rref_augmented(matrix: list[list[Fraction]], rhs: list[Fraction]):
    """对增广矩阵做 RREF，返回 (主元列, 结果表)；矛盾行返回 None。"""
    m = len(matrix)
    n = len(matrix[0])
    # 全部元素强制 Fraction，杜绝 int/int 浮点除法
    tab = [[Fraction(matrix[i][j]) for j in range(n)] + [Fraction(rhs[i])] for i in range(m)]
    pivot_cols: list[int] = []
    r = 0
    for col in range(n):
        pr = next((i for i in range(r, m) if tab[i][col] != 0), None)
        if pr is None:
            continue
        tab[r], tab[pr] = tab[pr], tab[r]
        d = tab[r][col]
        tab[r] = [v / d for v in tab[r]]
        for i in range(m):
            if i != r and tab[i][col] != 0:
                f = tab[i][col]
                tab[i] = [a - f * c for a, c in zip(tab[i], tab[r])]
        pivot_cols.append(col)
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if tab[i][n] != 0:
            return None
    return pivot_cols, tab


def matrix_rank(matrix: list[list]) -> int:
    m = len(matrix)
    n = len(matrix[0])
    tab = [[Fraction(v) for v in row] for row in matrix]
    r = 0
    for col in range(n):
        pr = next((i for i in range(r, m) if tab[i][col] != 0), None)
        if pr is None:
            continue
        tab[r], tab[pr] = tab[pr], tab[r]
        d = tab[r][col]
        tab[r] = [v / d for v in tab[r]]
        for i in range(m):
            if i != r and tab[i][col] != 0:
                f = tab[i][col]
                tab[i] = [a - f * c for a, c in zip(tab[i], tab[r])]
        r += 1
    return r


def enumerate_basic_solutions(p_flat: list[Fraction]):
    """枚举 A 的 rank(A) 列子集，返回全部可行权重 w（独立枚举法）。"""
    rank = matrix_rank([[Fraction(v) for v in row] for row in pid.A])
    solutions: list[list[Fraction]] = []
    for cols in combinations(range(pid.N_TYPES), rank):
        sub = [[pid.A[i][j] for j in cols] for i in range(pid.N_ROWS)]
        result = rref_augmented(sub, p_flat)
        if result is None:
            continue
        pivot_cols, tab = result
        if len(pivot_cols) != rank:
            continue  # 列子矩阵秩不足，跳过（其它子集覆盖等价基）
        w = [Fraction(0)] * pid.N_TYPES
        for ri, col_in_sub in enumerate(pivot_cols):
            w[cols[col_in_sub]] = tab[ri][rank]
        if any(v < 0 for v in w):
            continue
        # 独立重算等式
        if all(
            sum(pid.A[i][j] * w[j] for j in range(pid.N_TYPES)) == p_flat[i]
            for i in range(pid.N_ROWS)
        ):
            solutions.append(w)
    return solutions


def independent_bounds(p_flat: list[Fraction]):
    sols = enumerate_basic_solutions(p_flat)
    if not sols:
        return None
    vals = [sum(pid.C[j] * w[j] for j in range(pid.N_TYPES)) for w in sols]
    return min(vals), max(vals), sols


# ---------------------------------------------------------------------------
# 输出字符串与约束重算
# ---------------------------------------------------------------------------
def parse_out_fraction(text: object) -> Fraction:
    assert isinstance(text, str) and "/" in text, f"非 p/q 字符串: {text!r}"
    num, den = text.split("/")
    value = Fraction(int(num), int(den))
    # 约分校验：Fraction 已约分；分母正；字符串与规范形式一致
    assert int(den) > 0, f"分母非正: {text}"
    assert f"{value.numerator}/{value.denominator}" == text, f"未约分: {text}"
    return value


def verify_feasible_output(result: dict, p_flat: list[Fraction]) -> None:
    L = parse_out_fraction(result["bounds"][0])
    U = parse_out_fraction(result["bounds"][1])
    check(L <= U, f"下界 <= 上界（{L} <= {U}）")

    for key in ("lower_weights", "upper_weights"):
        vec = [parse_out_fraction(s) for s in result[key]]
        check(len(vec) == 16, f"{key} 恰有 16 项")
        check(all(v >= 0 for v in vec), f"{key} 全部非负")
        check(sum(vec, Fraction(0)) == 1, f"{key} 和为 1")
        for i in range(pid.N_ROWS):
            dot = sum(pid.A[i][j] * vec[j] for j in range(pid.N_TYPES))
            check(dot == p_flat[i], f"{key} 重算 A w = p（行 {i}）")

    wl = [parse_out_fraction(s) for s in result["lower_weights"]]
    wu = [parse_out_fraction(s) for s in result["upper_weights"]]
    check(sum(pid.C[j] * wl[j] for j in range(16)) == L, "下界权重目标值 = L")
    check(sum(pid.C[j] * wu[j] for j in range(16)) == U, "上界权重目标值 = U")

    lam = [parse_out_fraction(s) for s in result["lambda"]]
    mu = [parse_out_fraction(s) for s in result["mu"]]
    check(len(lam) == 8 and len(mu) == 8, "乘子各有 8 项（与行同序）")
    for t in range(16):
        at_lam = sum(pid.A[i][t] * lam[i] for i in range(8))
        at_mu = sum(pid.A[i][t] * mu[i] for i in range(8))
        check(at_lam <= pid.C[t], f"A^T λ <= c（类型 {t}）")
        check(at_mu >= pid.C[t], f"A^T μ >= c（类型 {t}）")
    check(sum(p_flat[i] * lam[i] for i in range(8)) == L, "p·λ = L")
    check(sum(p_flat[i] * mu[i] for i in range(8)) == U, "p·μ = U")


def verify_infeasible_output(result: dict, p_flat: list[Fraction]) -> None:
    check(set(result.keys()) == {"infeasible", "h"}, "不可行输出仅含两个成员")
    check(result["infeasible"] is True, "infeasible 标志为 true")
    h = [parse_out_fraction(s) for s in result["h"]]
    check(len(h) == 8, "证书 h 有 8 项")
    for t in range(16):
        dot = sum(pid.A[i][t] * h[i] for i in range(8))
        check(dot >= 0, f"Farkas 证书 A^T h >= 0（类型 {t}）")
    check(sum(p_flat[i] * h[i] for i in range(8)) == -1, "Farkas 证书 p·h = -1")


# ---------------------------------------------------------------------------
# 夹具与模糊测试
# ---------------------------------------------------------------------------
FIXTURE_DIR = os.path.join(tempfile.gettempdir(), "pid_acceptance_fixtures")


def write_fixture(name: str, payload: dict) -> str:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    path = os.path.join(FIXTURE_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def run_cli(input_path: str, output_path: str) -> int:
    return subprocess.call(
        [sys.executable, "-I", os.path.join(ROOT, "pid.py"), input_path, output_path]
    )


def fixture_payload(p3: list[list[list[str]]]) -> dict:
    return {"p": p3}


def test_named_fixtures() -> None:
    print("\n[1] 命名夹具（含退化可行域、不可行、负下界、题面示例）")
    cases = {
        "example": (
            [[["1/1", "0/1"], ["0/1", "0/1"]],
             [["1/1", "0/1"], ["0/1", "0/1"]]],
            ("0/1", "1/1"),
        ),
        "degenerate": (
            [[["0/1", "0/1"], ["0/1", "1/1"]],
             [["0/1", "0/1"], ["0/1", "1/1"]]],
            None,
        ),
        "negative": (
            [[["1/4", "3/4"], ["0/1", "0/1"]],
             [["0/1", "0/1"], ["3/4", "1/4"]]],
            None,
        ),
        "mixed": (
            [[["1/3", "1/6"], ["1/6", "1/3"]],
             [["1/6", "1/3"], ["1/3", "1/6"]]],
            None,
        ),
    }
    for name, (p3, expected) in cases.items():
        print(f" -- {name}")
        inp = write_fixture(name, fixture_payload(p3))
        out = os.path.join(FIXTURE_DIR, f"{name}.out.json")
        if os.path.exists(out):
            os.unlink(out)
        rc = run_cli(inp, out)
        check(rc == 0, f"{name} CLI 返回 0")
        with open(out, encoding="utf-8") as fh:
            result = json.load(fh)
        check(result["infeasible"] is False, f"{name} 判定可行")
        p_flat = pid.parse_input(fixture_payload(p3))
        verify_feasible_output(result, p_flat)
        independent = independent_bounds(p_flat)
        check(independent is not None, f"{name} 枚举法同样可行")
        lo, hi, _ = independent
        check(parse_out_fraction(result["bounds"][0]) == lo, f"{name} 下界与枚举一致 ({lo})")
        check(parse_out_fraction(result["bounds"][1]) == hi, f"{name} 上界与枚举一致 ({hi})")
        if expected is not None:
            check(result["bounds"] == list(expected), f"{name} 题面界限 {expected}")

    print(" -- infeasible（共享权重导致的不可行）")
    p3 = [[["0/1", "0/1"], ["0/1", "1/1"]],
          [["1/9", "4/9"], ["4/9", "0/1"]]]
    inp = write_fixture("infeasible", fixture_payload(p3))
    out = os.path.join(FIXTURE_DIR, "infeasible.out.json")
    if os.path.exists(out):
        os.unlink(out)
    rc = run_cli(inp, out)
    check(rc == 0, "infeasible CLI 返回 0")
    with open(out, encoding="utf-8") as fh:
        result = json.load(fh)
    p_flat = pid.parse_input(fixture_payload(p3))
    verify_infeasible_output(result, p_flat)
    check(independent_bounds(p_flat) is None, "枚举法同样判定不可行")


def test_fuzz() -> None:
    print("\n[2] 随机模糊测试（可行/不可行混合，全部与枚举法对照）")
    rng = random.Random(20261112)
    n_feasible = 0
    n_infeasible = 0
    trial = 0
    while n_feasible < 8 or n_infeasible < 8:
        trial += 1
        slices: list[list[str]] = []
        p_flat: list[Fraction] = []
        ok = True
        for _ in range(2):
            vals = [rng.randint(0, 6) for _ in range(4)]
            total = sum(vals)
            if total == 0:
                ok = False
                break
            for v in vals:
                p_flat.append(Fraction(v, total))
                slices.append(f"{v}/{total}")
        if not ok:
            continue
        p3 = [
            [[slices[0], slices[1]], [slices[2], slices[3]]],
            [[slices[4], slices[5]], [slices[6], slices[7]]],
        ]
        independent = independent_bounds(p_flat)
        name = f"fuzz_{trial}"
        inp = write_fixture(name, fixture_payload(p3))
        out = os.path.join(FIXTURE_DIR, f"{name}.out.json")
        rc = run_cli(inp, out)
        check(rc == 0, f"{name} CLI 返回 0")
        with open(out, encoding="utf-8") as fh:
            result = json.load(fh)
        if independent is None:
            n_infeasible += 1
            check(result["infeasible"] is True, f"{name} 与枚举法一致：不可行")
            verify_infeasible_output(result, p_flat)
        else:
            n_feasible += 1
            lo, hi, _ = independent
            check(result["infeasible"] is False, f"{name} 与枚举法一致：可行")
            verify_feasible_output(result, p_flat)
            check(parse_out_fraction(result["bounds"][0]) == lo, f"{name} 下界一致 ({lo})")
            check(parse_out_fraction(result["bounds"][1]) == hi, f"{name} 上界一致 ({hi})")
    print(f"  （模糊测试：{n_feasible} 个可行、{n_infeasible} 个不可行）")


def test_invalid_inputs() -> None:
    print("\n[3] 非法输入拒绝与路径定位")
    bad_cases = {
        "not_object": [1, 2, 3],
        "unknown_member": {"p": [[["1/1", "0/1"], ["0/1", "0/1"]],
                                 [["1/1", "0/1"], ["0/1", "0/1"]]], "x": 1},
        "missing_p": {},
        "bad_shape": {"p": [[["1/1", "0/1"]], [["1/1", "0/1"]]]},
        "bad_fraction": {"p": [[["0.5", "0/1"], ["0/1", "0/1"]],
                               [["1/1", "0/1"], ["0/1", "0/1"]]]},
        "zero_den": {"p": [[["1/0", "0/1"], ["0/1", "0/1"]],
                           [["1/1", "0/1"], ["0/1", "0/1"]]]},
        "negative_prob": {"p": [[["-1/2", "0/1"], ["0/1", "0/1"]],
                                [["1/1", "0/1"], ["0/1", "0/1"]]]},
        "bad_slice_sum": {"p": [[["1/2", "1/2"], ["0/1", "0/1"]],
                                [["1/2", "1/3"], ["0/1", "0/1"]]]},
        "non_string": {"p": [[[1, "0/1"], ["0/1", "0/1"]],
                             [["1/1", "0/1"], ["0/1", "0/1"]]]},
        "nan": '{"p": [[[NaN, "0/1"], ["0/1", "0/1"]], [["1/1", "0/1"], ["0/1", "0/1"]]]}',
    }
    for name, payload in bad_cases.items():
        inp = os.path.join(FIXTURE_DIR, f"bad_{name}.json")
        if isinstance(payload, str):
            with open(inp, "w", encoding="utf-8") as fh:
                fh.write(payload)
        else:
            with open(inp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        out = os.path.join(FIXTURE_DIR, f"bad_{name}.out.json")
        sentinel = "KEEP_OLD_OUTPUT"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(sentinel)
        rc = run_cli(inp, out)
        check(rc == 1, f"{name} 被拒绝（退出码 1）")
        with open(out, encoding="utf-8") as fh:
            check(fh.read() == sentinel, f"{name} 失败时保留旧输出文件")

    # 重复键
    dup = os.path.join(FIXTURE_DIR, "dup.json")
    with open(dup, "w", encoding="utf-8") as fh:
        fh.write('{"p": [[["1/1","0/1"],["0/1","0/1"]],'
                 '[["1/1","0/1"],["0/1","0/1"]]], '
                 '"p": [[["1/1","0/1"],["0/1","0/1"]],'
                 '[["1/1","0/1"],["0/1","0/1"]]]}')
    out = os.path.join(FIXTURE_DIR, "dup.out.json")
    rc = run_cli(dup, out)
    check(rc == 1, "重复 JSON 成员被拒绝")

    # 路径定位检查：非法元素信息包含具体零基数组路径
    inp = os.path.join(FIXTURE_DIR, "pathloc.json")
    with open(inp, "w", encoding="utf-8") as fh:
        json.dump({"p": [[["1/1", "0/1"], ["0/1", "0/1"]],
                         [["1/1", "0/1"], ["0/1", "not-a-fraction"]]]}, fh)
    proc = subprocess.run(
        [sys.executable, "-I", os.path.join(ROOT, "pid.py"), inp,
         os.path.join(FIXTURE_DIR, "pathloc.out.json")],
        capture_output=True, text=True,
    )
    check(proc.returncode == 1, "路径定位用例被拒绝")
    check("p[1][1][1]" in proc.stderr, f"错误信息定位零基数组路径 p[1][1][1]：{proc.stderr.strip()}")


def test_matrix_structure() -> None:
    print("\n[4] 矩阵 A 与类型定义结构核验")
    check(len(pid.TYPES) == 16, "恰有 16 种潜在类型")
    check(pid.TYPES == sorted(pid.TYPES), "潜在类型按字典序排列")
    check(len(pid.A) == 8 and all(len(r) == 16 for r in pid.A), "A 为 8x16")
    check(all(v in (0, 1) for row in pid.A for v in row), "A 为 0/1 矩阵")
    # 每个类型在每个 z 切片恰落入一个 (d,y) 单元 -> A 每列和为 2
    for t in range(16):
        check(sum(pid.A[i][t] for i in range(8)) == 2, f"类型 {t} 每切片一个观测单元（列和 2）")
    # A 的定义独立重算
    for ri, (z, d, y) in enumerate(pid.ROWS):
        for ci, (d0, d1, y0, y1) in enumerate(pid.TYPES):
            dz = d1 if z else d0
            ydz = y1 if dz else y0
            expect = 1 if (dz == d and ydz == y) else 0
            check(pid.A[ri][ci] == expect, f"A[{ri}][{ci}] 定义正确")
    check(pid.C == [Fraction(t[3] - t[2]) for t in pid.TYPES], "c_t = y1 - y0")


def main() -> int:
    test_matrix_structure()
    test_named_fixtures()
    test_fuzz()
    test_invalid_inputs()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"验收失败：{len(FAILURES)} 项")
        for msg in FAILURES:
            print(f"  - {msg}")
        return 1
    print("全部验收通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
