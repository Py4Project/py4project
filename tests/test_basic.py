import re

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import pytest

from py4project import regression, regression_plot, scatter_plot, bar_plot, box_plot


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    x = rng.uniform(10, 100, size=10)
    y = 1.5 * x + rng.normal(0, 5, size=10)
    return pd.DataFrame({
        "産業コード": range(1, 11),
        "産業": [f"産業{i}" for i in range(1, 11)],
        "全国成長効果": x,
        "産業構成効果": y,
        "地域固有効果": x - y,
    })


def test_regression_runs(sample_df, capsys):
    """regression がエラーなく実行され、結果を出力することの確認

    ラベルの文言は変わりうるので、特定の語句ではなく
    「数値を含む出力があること」を確かめる。
    """
    regression("全国成長効果", "産業構成効果", sample_df)
    out = capsys.readouterr().out
    assert out.strip() != "", "regression が何も出力していません。"
    assert re.search(r"\d", out), "regression の出力に数値が含まれていません。"


def test_regression_plot_runs(sample_df):
    regression_plot("全国成長効果", "産業構成効果", sample_df, text=True)


def test_scatter_plot_runs(sample_df):
    scatter_plot("全国成長効果", "産業構成効果", sample_df)


def test_scatter_plot_log_runs(sample_df):
    df = sample_df.copy()
    df["全国成長効果"] = df["全国成長効果"].abs() + 1
    df["産業構成効果"] = df["産業構成効果"].abs() + 1
    scatter_plot("全国成長効果", "産業構成効果", df, xylog=True)


def test_bar_plot_single_column(sample_df):
    bar_plot("全国成長効果", sample_df)


def test_bar_plot_multi_column(sample_df):
    bar_plot(["全国成長効果", "産業構成効果", "地域固有効果"], sample_df)


def test_box_plot_runs(sample_df):
    box_plot("全国成長効果", sample_df)


def test_box_plot_xlog_runs(sample_df):
    # 修正前は NameError: name 'pd' is not defined で失敗していた箇所
    df = sample_df.copy()
    df["全国成長効果"] = df["全国成長効果"].abs() + 1
    box_plot(["全国成長効果"], df, xlog=True)


# ============================================================
# エラーメッセージのテスト
# ============================================================

def test_error_column_not_found(sample_df):
    with pytest.raises(ValueError, match="見つかりません"):
        regression("全国成長効果", "存在しない列", sample_df)


def test_error_argument_order(sample_df):
    with pytest.raises(TypeError, match="データフレームではありません"):
        regression(sample_df, "全国成長効果", "産業構成効果")


def test_error_non_numeric_column(sample_df):
    with pytest.raises(TypeError, match="数値ではない"):
        regression("全国成長効果", "産業", sample_df)


def test_error_missing_values(sample_df):
    df = sample_df.copy()
    df.loc[0, "全国成長効果"] = np.nan
    with pytest.raises(ValueError, match="欠損値"):
        regression("全国成長効果", "産業構成効果", df)


def test_error_log_non_positive(sample_df):
    df = sample_df.copy()
    df.loc[0, "全国成長効果"] = 0.0
    with pytest.raises(ValueError, match="対数をとれません"):
        scatter_plot("全国成長効果", "産業構成効果", df, xlog=True)


def test_error_sort_by_missing(sample_df):
    df = sample_df.drop(columns=["産業コード"])
    with pytest.raises(ValueError, match="並べ替えに使う列"):
        bar_plot("全国成長効果", df)


def test_error_unit_zero(sample_df):
    with pytest.raises(ValueError, match="unit に 0"):
        bar_plot("全国成長効果", sample_df, unit=0)


def test_error_xlabel_length_mismatch(sample_df):
    with pytest.raises(ValueError, match="xlabel の数"):
        box_plot(["全国成長効果", "産業構成効果"], sample_df, xlabel=["A"])


def test_numeric_text_col_does_not_crash(sample_df):
    # 修正前は IndexError で落ちていた
    regression_plot("全国成長効果", "産業構成効果", sample_df, text_col="産業コード")


def test_error_messages_are_multiline(sample_df):
    # KeyError だと改行が潰れるため ValueError を使っていることの確認
    with pytest.raises(ValueError) as exc:
        regression("全国成長効果", "存在しない列", sample_df)
    assert "\n" in str(exc.value)
    assert "対処：" in str(exc.value)


# ============================================================
# 信頼区間の臨界値に関するテスト
# ============================================================

def _extract_ci(text):
    """regression の出力から信頼区間の2つの数値を取り出す。

    print の書式（空白の有無、ラベルの文言）に左右されないよう、
    「[数値, 数値]」の形を正規表現で探して数値として返す。
    """
    number = r"[-+]?\d+(?:\.\d+)?"
    pattern = re.compile(rf"\[\s*({number})\s*,\s*({number})\s*\]")

    # 「信頼区間」を含む行を優先して探す
    for line in text.splitlines():
        if "信頼区間" in line:
            m = pattern.search(line)
            if m:
                return float(m.group(1)), float(m.group(2))

    # 見つからなければ出力全体から探す
    m = pattern.search(text)
    assert m is not None, (
        "出力の中に「[下限, 上限]」の形の信頼区間が見つかりませんでした。\n"
        f"実際の出力:\n{text}"
    )
    return float(m.group(1)), float(m.group(2))


def test_ci_uses_correct_critical_value(capsys):
    """信頼区間が自由度に応じたt分布の臨界値を使っていることの確認

    表示された数値そのものを取り出して比べるので、
    print の文言や空白の入れ方を変えてもこのテストは壊れない。
    """
    from scipy import stats as _stats

    rng = np.random.default_rng(1)
    n = 47
    xv = rng.uniform(10, 100, n)
    yv = 1.5 * xv + rng.normal(0, 5, n)
    df = pd.DataFrame({"x": xv, "y": yv, "産業": [f"i{i}" for i in range(n)]})

    regression("x", "y", df)
    left, right = _extract_ci(capsys.readouterr().out)

    res = _stats.linregress(xv, yv)
    t_crit = _stats.t.ppf(0.975, n - 2)

    # 小数第3位まで表示されるため、丸め誤差ぶんの許容幅をとる
    tol = 0.001
    assert abs(left - (res.slope - t_crit * res.stderr)) <= tol
    assert abs(right - (res.slope + t_crit * res.stderr)) <= tol

    # 旧実装の固定値 2.131 を使っていないことの確認。
    # 「その文字列が出力に無いこと」ではなく
    # 「数値が2.131版とは違うこと」を確かめる。
    wrong_left = res.slope - 2.131 * res.stderr
    assert abs(left - wrong_left) > tol, (
        "信頼区間が固定値 2.131 を使ったときの値と一致しています。"
        "t_crit による計算に戻っていないか確認してください。"
    )


def test_extract_ci_is_format_independent():
    """_extract_ci が print の書式変更に耐えることの確認"""
    patterns = [
        "信頼区間 (95%) = [1.511,1.597]",
        "b₁ の95%信頼区間 : [1.511, 1.597]",
        "信頼区間　:　[ 1.511 ,  1.597 ]",
        "推定式: y = b0 + b1x\n標本の大きさ(n): 47\nb₁ の95%信頼区間 : [1.511, 1.597]\n",
    ]
    for text in patterns:
        assert _extract_ci(text) == (1.511, 1.597), f"抽出に失敗: {text!r}"

    # 負の値を含む場合
    assert _extract_ci("信頼区間 : [-0.250, 1.030]") == (-0.250, 1.030)


def test_ci_consistent_with_beta_one_test():
    """信頼区間が1を含まないことと、β=1検定の棄却が一致することの確認"""
    from scipy import stats as _stats

    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(5, 60))
        xv = rng.uniform(10, 100, n)
        yv = rng.uniform(0.5, 2.0) * xv + rng.normal(0, float(rng.uniform(1, 30)), n)
        res = _stats.linregress(xv, yv)
        t_crit = _stats.t.ppf(0.975, n - 2)
        lo = res.slope - t_crit * res.stderr
        hi = res.slope + t_crit * res.stderr
        t_stat = (res.slope - 1) / res.stderr
        p = 2 * (1 - _stats.t.cdf(abs(t_stat), n - 2))
        assert (not (lo <= 1 <= hi)) == (p < 0.05)


def test_docstring_no_longer_mentions_n17():
    """docstringからn=17に関する注意書きが削除されていることの確認"""
    for fn in (regression, regression_plot):
        assert "2.131" not in fn.__doc__
        assert "n = 17" not in fn.__doc__
