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
    regression("全国成長効果", "産業構成効果", sample_df)
    captured = capsys.readouterr()
    assert "決定係数" in captured.out


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

def test_ci_uses_correct_critical_value(capsys):
    """信頼区間が自由度に応じたt分布の臨界値を使っていることの確認"""
    from scipy import stats as _stats

    rng = np.random.default_rng(1)
    n = 47
    xv = rng.uniform(10, 100, n)
    yv = 1.5 * xv + rng.normal(0, 5, n)
    df = pd.DataFrame({"x": xv, "y": yv, "産業": [f"i{i}" for i in range(n)]})

    regression("x", "y", df)
    out = capsys.readouterr().out

    res = _stats.linregress(xv, yv)
    t_crit = _stats.t.ppf(0.975, n - 2)
    expected_left = res.slope - t_crit * res.stderr
    expected_right = res.slope + t_crit * res.stderr

    assert f"[{expected_left:.3f},{expected_right:.3f}]" in out
    # 旧実装の固定値 2.131 を使っていないことの確認
    wrong_left = res.slope - 2.131 * res.stderr
    assert f"[{wrong_left:.3f}," not in out


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
