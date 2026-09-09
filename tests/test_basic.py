import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from py4project import (
    regression, regression_plot, scatter_plot, bar_plot, box_plot,
    line_plot, pie_plot,
)
from py4project.core import RegressionResult


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    x = rng.uniform(10, 100, size=20)
    y = 1.5 * x + rng.normal(0, 5, size=20)
    return pd.DataFrame({
        "産業コード": range(1, 21),
        "産業": [f"産業{i}" for i in range(1, 21)],
        "全国成長効果": x,
        "産業構成効果": y,
        "地域固有効果": x - y,
        "区分": ["A", "B"] * 10,
    })


# ============================================================
# 基本動作
# ============================================================

def test_regression_returns_wrapper(sample_df):
    res = regression("全国成長効果", "産業構成効果", sample_df)
    assert isinstance(res, RegressionResult)


def test_regression_plot_runs(sample_df):
    regression_plot("全国成長効果", "産業構成効果", sample_df)


def test_regression_plot_with_text(sample_df):
    regression_plot("全国成長効果", "産業構成効果", sample_df, label_col="産業")


def test_regression_plot_from_result(sample_df):
    res = regression("全国成長効果", "産業構成効果", sample_df)
    regression_plot(ols_res=res)


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


def test_title_default_is_none_everywhere():
    """4つの作図関数すべてで title の既定値が None であることの確認"""
    import inspect
    for fn in (regression_plot, scatter_plot, bar_plot, box_plot):
        assert inspect.signature(fn).parameters["title"].default is None, fn.__name__


def test_omitted_title_never_shows_none(sample_df):
    """title を省略したときに文字列 'None' が表示されないことの確認"""
    calls = [
        lambda: regression_plot("全国成長効果", "産業構成効果", sample_df),
        lambda: scatter_plot("全国成長効果", "産業構成効果", sample_df),
        lambda: bar_plot("全国成長効果", sample_df),
        lambda: box_plot("全国成長効果", sample_df),
    ]
    for call in calls:
        call()
        assert "None" not in matplotlib.pyplot.gcf().axes[0].get_title()
        matplotlib.pyplot.close("all")


def test_given_title_is_displayed(sample_df):
    for call in (
        lambda: scatter_plot("全国成長効果", "産業構成効果", sample_df, title="大阪府"),
        lambda: bar_plot("全国成長効果", sample_df, title="大阪府"),
        lambda: box_plot("全国成長効果", sample_df, title="大阪府"),
    ):
        call()
        assert matplotlib.pyplot.gcf().axes[0].get_title() == "大阪府"
        matplotlib.pyplot.close("all")


def test_box_plot_defaults(sample_df):
    """title と label_col の既定値が None であることの確認"""
    import inspect
    p = inspect.signature(box_plot).parameters
    assert p["title"].default is None
    assert p["label_col"].default is None


def test_box_plot_no_labels_without_text_col(sample_df):
    """label_col を省略すると名前が表示されないことの確認"""
    df = sample_df.copy()
    df.loc[0, "全国成長効果"] = 1000.0          # 外れ値をつくる
    box_plot("全国成長効果", df)
    ax = matplotlib.pyplot.gcf().axes[0]
    assert len(ax.texts) == 0
    assert ax.get_title() == ""                 # None と表示されない
    matplotlib.pyplot.close("all")


def test_box_plot_labels_with_text_col(sample_df):
    """label_col を指定すると外れ値に名前がつくことの確認"""
    df = sample_df.copy()
    df.loc[0, "全国成長効果"] = 1000.0
    box_plot("全国成長効果", df, label_col="産業")
    ax = matplotlib.pyplot.gcf().axes[0]
    assert "産業1" in [t.get_text() for t in ax.texts]
    matplotlib.pyplot.close("all")


def test_box_plot_title(sample_df):
    box_plot("全国成長効果", sample_df, title="大阪府")
    assert matplotlib.pyplot.gcf().axes[0].get_title() == "大阪府"
    matplotlib.pyplot.close("all")


def test_box_plot_xlog_runs(sample_df):
    df = sample_df.copy()
    df["全国成長効果"] = df["全国成長効果"].abs() + 1
    box_plot(["全国成長効果"], df, xlog=True)


# ============================================================
# 回帰分析の数値（D-1：printではなく戻り値を検証する）
# ============================================================

def test_regression_matches_statsmodels(sample_df):
    """係数と信頼区間が statsmodels の計算と一致することの確認"""
    res = regression("全国成長効果", "産業構成効果", sample_df)

    Y = sample_df["産業構成効果"]
    X = sm.add_constant(sample_df[["全国成長効果"]])
    expected = sm.OLS(Y, X).fit()

    assert res.params.equals(expected.params)
    assert res.rsquared == pytest.approx(expected.rsquared)

    ci = res.conf_int(alpha=0.05)
    exp_ci = expected.conf_int(alpha=0.05)
    assert np.allclose(ci.to_numpy(), exp_ci.to_numpy())


def test_regression_alpha_changes_interval(sample_df):
    """alpha を小さくすると信頼区間が広がることの確認"""
    r95 = regression("全国成長効果", "産業構成効果", sample_df, alpha=0.05)
    r99 = regression("全国成長効果", "産業構成効果", sample_df, alpha=0.01)

    w95 = r95.table["信頼区間上限"] - r95.table["信頼区間下限"]
    w99 = r99.table["信頼区間上限"] - r99.table["信頼区間下限"]
    assert (w99 > w95).all()
    assert ("99 ％") in repr(r99)


def test_regression_robust_changes_se_not_coefficients(sample_df):
    """robust=True で係数は変わらず、標準誤差だけが変わることの確認"""
    plain = regression("全国成長効果", "産業構成効果", sample_df, robust=False)
    rob = regression("全国成長効果", "産業構成効果", sample_df, robust=True)

    assert np.allclose(plain.params.to_numpy(), rob.params.to_numpy())
    assert not np.allclose(plain.bse.to_numpy(), rob.bse.to_numpy())
    assert "不均一分散頑健(HC3)" in repr(rob)
    assert "非頑健" in repr(plain)


def test_regression_multiple_x(sample_df):
    """x にリストを渡すと重回帰になることの確認"""
    res = regression(["全国成長効果", "地域固有効果"], "産業構成効果", sample_df)
    assert list(res.table.index) == ["定数項", "全国成長効果", "地域固有効果"]


def test_regression_stars_toggle(sample_df):
    with_stars = regression("全国成長効果", "産業構成効果", sample_df, stars=True)
    without = regression("全国成長効果", "産業構成効果", sample_df, stars=False)
    assert "有意性" in with_stars.table.columns
    assert "有意性" not in without.table.columns


def test_regression_drops_missing_rows(sample_df):
    df = sample_df.copy()
    df.loc[0, "全国成長効果"] = np.nan
    res = regression("全国成長効果", "産業構成効果", df)
    assert int(res.nobs) == len(sample_df) - 1
    assert "欠損値のため除外" in repr(res)


# ============================================================
# 案3：表示の仕組み
# ============================================================

def test_result_has_html_repr(sample_df):
    res = regression("全国成長効果", "産業構成効果", sample_df)
    html = res._repr_html_()
    assert "<table" in html
    assert "被説明変数" in html


def test_footnote_escaped_only_in_html(sample_df):
    """脚注の「<」がテキストではそのまま、HTMLではエスケープされることの確認"""
    res = regression("全国成長効果", "産業構成効果", sample_df, stars=True)
    assert "p<0.01" in repr(res)
    assert "&lt;" not in repr(res)
    assert "p&lt;0.01" in res._repr_html_()


def test_no_scientific_notation_in_display(sample_df):
    """桁の大きな値でも指数表記にならないことの確認"""
    df = sample_df.copy()
    df["産業構成効果"] = df["産業構成効果"] * 1_000_000   # 桁を大きくする
    res = regression("全国成長効果", "産業構成効果", df)

    assert "e+" not in repr(res)
    assert "e+" not in res._repr_html_()


def test_table_stays_numeric(sample_df):
    """表示を整えても表の列が数値型のままであることの確認"""
    res = regression("全国成長効果", "産業構成効果", sample_df)
    width = res.table["信頼区間上限"] - res.table["信頼区間下限"]
    assert (width > 0).all()


def test_result_forwards_attributes(sample_df):
    """statsmodels の属性が転送されることの確認"""
    res = regression("全国成長効果", "産業構成効果", sample_df)
    assert res.nobs == 20
    assert hasattr(res, "pvalues")
    assert type(res.result).__name__ == "RegressionResultsWrapper"


# ============================================================
# regression_plot の仕様（A-1, A-2, A-4）
# ============================================================

def test_plot_uses_passed_result_without_refitting(sample_df):
    """A-2：渡した結果の係数がそのまま使われることの確認"""
    res = regression("全国成長効果", "産業構成効果", sample_df)
    regression_plot(ols_res=res)

    ax = matplotlib.pyplot.gcf().axes[0]
    line = [ln for ln in ax.lines if ln.get_label() == "回帰直線"][0]
    xs, ys = line.get_xdata(), line.get_ydata()
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    assert slope == pytest.approx(float(res.params.iloc[1]))
    matplotlib.pyplot.close("all")


def test_plot_uses_variable_names_from_result(sample_df):
    """A-1：endog / exog の名前がタイトルと軸ラベルに使われることの確認"""
    res = regression("全国成長効果", "産業構成効果", sample_df)
    regression_plot(ols_res=res)

    ax = matplotlib.pyplot.gcf().axes[0]
    assert "None" not in ax.get_title()
    assert "産業構成効果" in ax.get_title()
    assert ax.get_xlabel() == "全国成長効果"
    assert ax.get_ylabel() == "産業構成効果"
    matplotlib.pyplot.close("all")


def test_empty_title_still_draws_line(sample_df):
    """A-4：title='' でも回帰直線が消えないことの確認"""
    regression_plot("全国成長効果", "産業構成効果", sample_df, title="")
    ax = matplotlib.pyplot.gcf().axes[0]
    assert any(ln.get_label() == "回帰直線" for ln in ax.lines)
    matplotlib.pyplot.close("all")


def test_line_false_draws_no_line(sample_df):
    regression_plot("全国成長効果", "産業構成効果", sample_df, line=False)
    ax = matplotlib.pyplot.gcf().axes[0]
    assert not any(ln.get_label() == "回帰直線" for ln in ax.lines)
    matplotlib.pyplot.close("all")


def test_plot_rejects_multiple_regression_result(sample_df):
    res = regression(["全国成長効果", "地域固有効果"], "産業構成効果", sample_df)
    with pytest.raises(ValueError, match="重回帰"):
        regression_plot(ols_res=res)


# ============================================================
# 引数の既定値（A-5, A-6）
# ============================================================

def test_no_text_parameter():
    """text 引数が廃止され、label_col に統合されていることの確認"""
    import inspect
    for fn in (regression_plot, scatter_plot, box_plot):
        assert "text" not in inspect.signature(fn).parameters, fn.__name__


def test_text_col_defaults_to_none():
    """label_col の既定値が3関数すべてで None であることの確認"""
    import inspect
    for fn in (regression_plot, scatter_plot, box_plot):
        assert inspect.signature(fn).parameters["label_col"].default is None, fn.__name__


def test_labels_shown_only_when_text_col_given(sample_df):
    """label_col の指定有無で名前の表示が切り替わることの確認"""
    for fn in (regression_plot, scatter_plot):
        fn("全国成長効果", "産業構成効果", sample_df)
        assert len(matplotlib.pyplot.gcf().axes[0].texts) == 0, fn.__name__
        matplotlib.pyplot.close("all")

        fn("全国成長効果", "産業構成効果", sample_df, label_col="産業")
        assert len(matplotlib.pyplot.gcf().axes[0].texts) == len(sample_df), fn.__name__
        matplotlib.pyplot.close("all")


def test_ols_res_is_keyword_only():
    import inspect
    p = inspect.signature(regression_plot).parameters["ols_res"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY


def test_argument_order_is_x_y_data():
    """全関数で (x, y, data) の順であることの確認"""
    import inspect
    for fn in (regression, regression_plot, scatter_plot):
        names = list(inspect.signature(fn).parameters)[:3]
        assert names == ["x", "y", "data"], f"{fn.__name__}: {names}"


# ============================================================
# エラーメッセージ（B-4, B-5）
# ============================================================

def test_error_column_not_found(sample_df):
    with pytest.raises(ValueError, match="見つかりません"):
        regression("全国成長効果", "存在しない列", sample_df)


def test_error_argument_order(sample_df):
    with pytest.raises(TypeError, match="データフレームではありません"):
        regression(sample_df, "全国成長効果", "産業構成効果")


def test_error_non_numeric_column(sample_df):
    with pytest.raises(TypeError, match="数値ではない"):
        regression("全国成長効果", "区分", sample_df)


def test_error_text_col_not_found(sample_df):
    for fn in (regression_plot, scatter_plot, box_plot):
        with pytest.raises(ValueError, match="見つかりません"):
            if fn is box_plot:
                fn("全国成長効果", sample_df, label_col="存在しない列")
            else:
                fn("全国成長効果", "産業構成効果", sample_df, label_col="存在しない列")


def test_error_ols_res_with_text_col(sample_df):
    res = regression("全国成長効果", "産業構成効果", sample_df)
    with pytest.raises(ValueError, match="label_col は一緒に使えません"):
        regression_plot(ols_res=res, label_col="産業")


def test_error_ols_res_with_robust(sample_df):
    """⑥：ols_res と robust の同時指定はエラーになる"""
    res = regression("全国成長効果", "産業構成効果", sample_df)
    with pytest.raises(ValueError, match="robust は一緒に使えません"):
        regression_plot(ols_res=res, robust=True)


def test_error_ols_res_without_constant(sample_df):
    """⑦：定数項のない結果は「重回帰」ではなく専用のメッセージになる"""
    import statsmodels.api as _sm
    res = _sm.OLS(sample_df["産業構成効果"], sample_df[["全国成長効果"]]).fit()
    with pytest.raises(ValueError, match="定数項のない"):
        regression_plot(ols_res=res)


# ============================================================
# 欠損値・標本の大きさ・説明変数の指定（①〜⑤）
# ============================================================

def test_missing_values_dropped_consistently(sample_df, capsys):
    """①：3関数すべてが欠損行を除いて動作する"""
    df = sample_df.copy()
    df.loc[0, "全国成長効果"] = np.nan

    res = regression("全国成長効果", "産業構成効果", df)
    assert int(res.nobs) == len(sample_df) - 1

    for fn in (regression_plot, scatter_plot):
        fn("全国成長効果", "産業構成効果", df)
        assert "欠損値のため" in capsys.readouterr().out, fn.__name__
        matplotlib.pyplot.close("all")


def test_sample_size_checked_after_dropna():
    """②：行数の確認は欠損値を除いたあとでおこなう"""
    df = pd.DataFrame({"x": [1.0, 2.0] + [np.nan] * 18,
                       "y": [1.0, 3.0] + list(np.arange(18.0))})
    with pytest.raises(ValueError, match="回帰分析を計算できません"):
        regression("x", "y", df)


def test_sample_size_depends_on_number_of_x():
    """③：必要な行数が説明変数の数に応じて変わる"""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0],
                       "b": [2.0, 1.0, 4.0],
                       "y": [1.0, 2.0, 3.0]})
    # 説明変数1個なら n=3 で足りる
    regression("a", "y", df)
    # 説明変数2個には n=4 必要
    with pytest.raises(ValueError, match="少なくとも4行"):
        regression(["a", "b"], "y", df)


def test_no_nan_pvalues_displayed(sample_df):
    """③：自由度が確保されるため p値が nan にならない"""
    res = regression(["全国成長効果", "地域固有効果"], "産業構成効果", sample_df)
    assert "nan" not in repr(res)


def test_error_duplicate_x(sample_df):
    """④：説明変数の重複はエラー"""
    with pytest.raises(ValueError, match="重複"):
        regression(["全国成長効果", "全国成長効果"], "産業構成効果", sample_df)


def test_error_x_equals_y(sample_df):
    """⑤：x と y が同じ列ならエラー"""
    with pytest.raises(ValueError, match="両方に指定"):
        regression("全国成長効果", "全国成長効果", sample_df)
    with pytest.raises(ValueError, match="両方に指定"):
        regression_plot("全国成長効果", "全国成長効果", sample_df)


def test_log_option_names_match_function(sample_df):
    """⑧：対数エラーの案内が、その関数が持つオプション名だけになる"""
    df = sample_df.copy()
    df.loc[0, "全国成長効果"] = 0.0

    with pytest.raises(ValueError) as exc:
        box_plot("全国成長効果", df, xlog=True)
    assert "ylog" not in str(exc.value)
    assert "xylog" not in str(exc.value)

    with pytest.raises(ValueError) as exc:
        scatter_plot("全国成長効果", "産業構成効果", df, xlog=True)
    assert "xylog" in str(exc.value)


def test_error_log_non_positive(sample_df):
    df = sample_df.copy()
    df.loc[0, "全国成長効果"] = 0.0
    with pytest.raises(ValueError, match="対数をとれません"):
        scatter_plot("全国成長効果", "産業構成効果", df, xlog=True)


def test_error_sort_by_missing(sample_df):
    """sort_by に存在しない列を指定したときだけエラーになる"""
    with pytest.raises(ValueError, match="並べ替えに使う列"):
        bar_plot("全国成長効果", sample_df, sort_by="存在しない列")


def test_bar_plot_works_without_industry_columns(sample_df):
    """産業コードや産業の列がなくても bar_plot は動く（他の授業でも使えるように）"""
    df = sample_df.drop(columns=["産業コード", "産業"])
    bar_plot("全国成長効果", df)
    matplotlib.pyplot.close("all")


def test_error_unit_zero(sample_df):
    with pytest.raises(ValueError, match="unit に 0"):
        bar_plot("全国成長効果", sample_df, unit=0)


def test_error_xlabel_length_mismatch(sample_df):
    with pytest.raises(ValueError, match="xlabel の数"):
        box_plot(["全国成長効果", "産業構成効果"], sample_df, xlabel=["A"])


def test_error_messages_are_multiline(sample_df):
    """エラーメッセージが【エラー】形式の複数行であることの確認"""
    with pytest.raises(ValueError) as exc:
        regression("全国成長効果", "存在しない列", sample_df)
    msg = str(exc.value)
    assert "【エラー】" in msg
    assert "原因：" in msg
    assert "対処：" in msg


def test_error_example_uses_x_y_order(sample_df):
    """エラー文中の例が (x, y, data) の順になっていることの確認"""
    with pytest.raises(TypeError) as exc:
        regression(sample_df, "全国成長効果", "産業構成効果")
    assert 'regression("事業所数", "従業者数", df)' in str(exc.value)


def test_numeric_text_col_does_not_crash(sample_df):
    regression_plot("全国成長効果", "産業構成効果", sample_df, label_col="産業コード")


# ============================================================
# 依存関係（C-2）
# ============================================================

def test_core_does_not_import_scipy_directly():
    """C-2：core.py が scipy を直接 import していないことの確認"""
    import inspect
    import py4project.core as core
    src = inspect.getsource(core)
    assert "from scipy" not in src
    assert "import scipy" not in src


# ============================================================
# line_plot
# ============================================================

@pytest.fixture
def series_df():
    return pd.DataFrame({
        "年": [2021, 2016, 2019, 2018],
        "A": [10.0, 4.0, 8.0, 6.0],
        "B": [1.0, 2.0, 3.0, 4.0],
        "全国成長効果": [5.0, 6.0, 7.0, 8.0],
        "地域固有効果": [-1.0, 0.0, 2.0, 3.0],
    })


def test_line_plot_runs(series_df):
    line_plot("年", "A", series_df)
    matplotlib.pyplot.close("all")


def test_line_plot_sorts_by_x(series_df):
    """横軸の値で昇順に並べ替えてから線を引く"""
    line_plot("年", "A", series_df)
    xs = list(matplotlib.pyplot.gcf().axes[0].lines[0].get_xdata())
    assert xs == sorted(xs)
    matplotlib.pyplot.close("all")


def test_line_plot_does_not_convert_year_to_date(series_df):
    """年の整数は日付に変換せず、そのまま使う"""
    line_plot("年", "A", series_df)
    xs = list(matplotlib.pyplot.gcf().axes[0].lines[0].get_xdata())
    assert xs[0] == 2016 and xs[-1] == 2021
    matplotlib.pyplot.close("all")


def test_line_plot_keeps_rows_with_missing_y():
    """yの欠損は行ごと消さず、その箇所だけ線を途切れさせる"""
    df = pd.DataFrame({"年": [2016, 2017, 2018, 2019],
                       "A": [1.0, np.nan, 3.0, 4.0],
                       "B": [2.0, 3.0, np.nan, 5.0]})
    line_plot("年", ["A", "B"], df)
    for line in matplotlib.pyplot.gcf().axes[0].lines:
        assert len(line.get_xdata()) == 4
    matplotlib.pyplot.close("all")


def test_line_plot_drops_rows_with_missing_x(capsys):
    df = pd.DataFrame({"年": [2016.0, np.nan, 2018.0], "A": [1.0, 2.0, 3.0]})
    line_plot("年", "A", df)
    assert "欠損値のため" in capsys.readouterr().out
    matplotlib.pyplot.close("all")


def test_line_plot_ylabel_depends_on_number_of_series(series_df):
    line_plot("年", "A", series_df)
    assert matplotlib.pyplot.gcf().axes[0].get_ylabel() == "A"
    matplotlib.pyplot.close("all")

    line_plot("年", ["A", "B"], series_df)
    assert matplotlib.pyplot.gcf().axes[0].get_ylabel() == ""
    matplotlib.pyplot.close("all")


def test_line_plot_legend(series_df):
    line_plot("年", ["A", "B"], series_df)
    labels = [t.get_text() for t in matplotlib.pyplot.gcf().axes[0].get_legend().get_texts()]
    assert labels == ["A", "B"]
    matplotlib.pyplot.close("all")

    line_plot("年", ["A", "B"], series_df, legend=["系列A", "系列B"])
    labels = [t.get_text() for t in matplotlib.pyplot.gcf().axes[0].get_legend().get_texts()]
    assert labels == ["系列A", "系列B"]
    matplotlib.pyplot.close("all")


def test_line_plot_legend_for_single_series(series_df):
    """yが1つでも legend を指定すれば凡例を出す（黙って無視しない）"""
    line_plot("年", "A", series_df, legend=["系列A"])
    assert matplotlib.pyplot.gcf().axes[0].get_legend() is not None
    matplotlib.pyplot.close("all")


def test_line_plot_legend_length_mismatch(series_df):
    with pytest.raises(ValueError, match="legend の数"):
        line_plot("年", ["A", "B"], series_df, legend=["一つだけ"])


def test_line_plot_shares_color_map(series_df):
    """bar_plot と同じ配色を使う"""
    line_plot("年", ["全国成長効果", "地域固有効果"], series_df)
    colors = [ln.get_color() for ln in matplotlib.pyplot.gcf().axes[0].lines[:2]]
    assert colors == ["#5B7C99", "#1F7A5C"]
    matplotlib.pyplot.close("all")


def test_line_plot_log_options(series_df):
    line_plot("年", "A", series_df, ylog=True)
    matplotlib.pyplot.close("all")
    line_plot("年", "A", series_df, xylog=True)
    matplotlib.pyplot.close("all")


def test_line_plot_log_rejects_non_positive(series_df):
    with pytest.raises(ValueError, match="対数をとれません"):
        line_plot("年", "地域固有効果", series_df, ylog=True)


def test_line_plot_accepts_string_x():
    """文字列の横軸はカテゴリとして描ける"""
    df = pd.DataFrame({"期": ["2016Q1", "2016Q2"], "値": [1.0, 2.0]})
    line_plot("期", "値", df)
    matplotlib.pyplot.close("all")


def test_line_plot_string_x_rejected_for_log():
    df = pd.DataFrame({"期": ["2016Q1", "2016Q2"], "値": [1.0, 2.0]})
    with pytest.raises(TypeError, match="数値ではない"):
        line_plot("期", "値", df, xlog=True)


# ============================================================
# pie_plot
# ============================================================

@pytest.fixture
def pie_df():
    return pd.DataFrame({
        "産業": ["製造業", "卸売業", "小売業", "建設業"],
        "従業者数": [1500.0, 900.0, 700.0, 380.0],
        "地域固有効果": [30.0, -5.0, 15.0, 40.0],
    })


def test_pie_plot_runs(pie_df):
    pie_plot("従業者数", pie_df, "産業")
    matplotlib.pyplot.close("all")


def test_pie_plot_shows_percentages(pie_df):
    pie_plot("従業者数", pie_df, "産業")
    texts = [t.get_text() for t in matplotlib.pyplot.gcf().axes[0].texts]
    assert any(t.endswith("%") for t in texts)
    matplotlib.pyplot.close("all")


def test_pie_plot_is_circular(pie_df):
    pie_plot("従業者数", pie_df, "産業")
    assert matplotlib.pyplot.gcf().axes[0].get_aspect() == 1.0
    matplotlib.pyplot.close("all")


def test_pie_plot_legend_replaces_wedge_labels(pie_df):
    """legend=True のときは扇形の名前を出さず、凡例だけにする"""
    pie_plot("従業者数", pie_df, "産業", legend=False)
    ax = matplotlib.pyplot.gcf().axes[0]
    names = [t.get_text() for t in ax.texts if not t.get_text().endswith("%")]
    assert "製造業" in names
    assert ax.get_legend() is None
    matplotlib.pyplot.close("all")

    pie_plot("従業者数", pie_df, "産業", legend=True)
    ax = matplotlib.pyplot.gcf().axes[0]
    names = [t.get_text() for t in ax.texts if not t.get_text().endswith("%")]
    assert all(n == "" for n in names)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == list(pie_df["産業"])
    matplotlib.pyplot.close("all")


def test_pie_plot_rejects_negative_values(pie_df):
    with pytest.raises(ValueError, match="負の値"):
        pie_plot("地域固有効果", pie_df, "産業")


def test_pie_plot_rejects_zero_total(pie_df):
    df = pie_df.copy()
    df["従業者数"] = 0.0
    with pytest.raises(ValueError, match="合計が0"):
        pie_plot("従業者数", df, "産業")


def test_pie_plot_drops_missing(pie_df, capsys):
    df = pie_df.copy()
    df.loc[0, "従業者数"] = np.nan
    pie_plot("従業者数", df, "産業")
    assert "欠損値のため" in capsys.readouterr().out
    matplotlib.pyplot.close("all")


def test_pie_plot_title(pie_df):
    pie_plot("従業者数", pie_df, "産業", title="産業別構成")
    assert matplotlib.pyplot.gcf().axes[0].get_title() == "産業別構成"
    matplotlib.pyplot.close("all")


def test_new_functions_follow_style():
    """新しい2関数も既存のスタイルに沿っていることの確認"""
    import inspect
    # title の既定値は None
    for fn in (line_plot, pie_plot):
        assert inspect.signature(fn).parameters["title"].default is None, fn.__name__
    # 引数は (x, ..., data) の順
    assert list(inspect.signature(line_plot).parameters)[:3] == ["x", "y", "data"]
    assert list(inspect.signature(pie_plot).parameters)[:3] == ["x", "data", "label_col"]



# ============================================================
# 引数名の統一（label_col）
# ============================================================

def test_label_col_name_is_unified():
    """「名前が入った列」の引数名が全関数で label_col に統一されている"""
    import inspect
    for fn in (bar_plot, box_plot, scatter_plot, regression_plot, pie_plot):
        names = set(inspect.signature(fn).parameters)
        assert "label_col" in names, fn.__name__
        assert "text_col" not in names, fn.__name__
        assert "tick_label" not in names, fn.__name__


# ============================================================
# pie_plot の other / top
# ============================================================

@pytest.fixture
def abcd_df():
    return pd.DataFrame({"区分": ["A", "B", "C", "D"],
                         "値": [40.0, 30.0, 20.0, 10.0]})


def _pie_names_and_pcts():
    ax = matplotlib.pyplot.gcf().axes[0]
    names = [t.get_text() for t in ax.texts if not t.get_text().endswith("%")]
    pcts = [t.get_text() for t in ax.texts if t.get_text().endswith("%")]
    return names, pcts


def test_pie_other_merges_listed_labels(abcd_df):
    pie_plot("値", abcd_df, "区分", other=["C", "D"])
    names, pcts = _pie_names_and_pcts()
    assert names == ["A", "B", "その他"]
    assert pcts == ["40.0%", "30.0%", "30.0%"]
    matplotlib.pyplot.close("all")


def test_pie_other_is_placed_last(abcd_df):
    """まとめた扇形は必ず最後に置かれる"""
    pie_plot("値", abcd_df, "区分", other=["B"])
    names, _ = _pie_names_and_pcts()
    assert names[-1] == "その他"
    matplotlib.pyplot.close("all")


def test_pie_other_accepts_string(abcd_df):
    pie_plot("値", abcd_df, "区分", other="D")
    names, _ = _pie_names_and_pcts()
    assert names == ["A", "B", "C", "その他"]
    matplotlib.pyplot.close("all")


def test_pie_other_label_can_be_changed(abcd_df):
    pie_plot("値", abcd_df, "区分", other=["C", "D"], other_label="その他（合計）")
    names, _ = _pie_names_and_pcts()
    assert names[-1] == "その他（合計）"
    matplotlib.pyplot.close("all")


def test_pie_other_is_grey(abcd_df):
    pie_plot("値", abcd_df, "区分", other=["C", "D"])
    ax = matplotlib.pyplot.gcf().axes[0]
    rgb = ax.patches[-1].get_facecolor()[:3]
    assert all(abs(v - 0.6196) < 0.01 for v in rgb)
    matplotlib.pyplot.close("all")


def test_pie_top_keeps_largest(abcd_df):
    pie_plot("値", abcd_df, "区分", top=2)
    names, pcts = _pie_names_and_pcts()
    assert names == ["A", "B", "その他"]
    assert pcts == ["40.0%", "30.0%", "30.0%"]
    matplotlib.pyplot.close("all")


def test_pie_top_larger_than_categories(abcd_df):
    """top が項目数以上ならまとめない"""
    pie_plot("値", abcd_df, "区分", top=10)
    names, _ = _pie_names_and_pcts()
    assert names == ["A", "B", "C", "D"]
    matplotlib.pyplot.close("all")


def test_pie_negative_checked_before_merging():
    """負の値の検査はまとめる前におこなう（正負の相殺を防ぐ）"""
    df = pd.DataFrame({"区分": ["A", "B", "C", "D"],
                       "値": [40.0, 30.0, -5.0, 10.0]})
    with pytest.raises(ValueError, match="負の値"):
        pie_plot("値", df, "区分", other=["C", "D"])


def test_pie_aggregates_duplicate_labels():
    """同じ名前の行は合計する"""
    df = pd.DataFrame({"区分": ["A", "B", "A", "C"],
                       "値": [10.0, 30.0, 30.0, 30.0]})
    pie_plot("値", df, "区分")
    names, pcts = _pie_names_and_pcts()
    assert names == ["A", "B", "C"]
    assert pcts == ["40.0%", "30.0%", "30.0%"]
    matplotlib.pyplot.close("all")


def test_pie_labels_are_not_truncated():
    """円グラフの名前は途中で切らない（凡例に使うため）"""
    long_name = "サービス業（他に分類されないもの）"
    df = pd.DataFrame({"区分": [long_name, "A"], "値": [1.0, 1.0]})
    pie_plot("値", df, "区分", legend=True)
    labels = [t.get_text()
              for t in matplotlib.pyplot.gcf().axes[0].get_legend().get_texts()]
    assert long_name in labels
    matplotlib.pyplot.close("all")


def test_pie_error_unknown_other_label(abcd_df):
    with pytest.raises(ValueError, match="見つかりません"):
        pie_plot("値", abcd_df, "区分", other=["存在しない"])


def test_pie_error_merging_everything(abcd_df):
    with pytest.raises(ValueError, match="すべての項目"):
        pie_plot("値", abcd_df, "区分", other=["A", "B", "C", "D"])


def test_pie_error_other_label_collision():
    df = pd.DataFrame({"区分": ["A", "B", "その他"], "値": [40.0, 30.0, 10.0]})
    with pytest.raises(ValueError, match="すでにあります"):
        pie_plot("値", df, "区分", other=["B"])


def test_pie_error_other_and_top_together(abcd_df):
    with pytest.raises(ValueError, match="一緒に使えません"):
        pie_plot("値", abcd_df, "区分", other=["C"], top=2)


def test_pie_error_invalid_top(abcd_df):
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="1以上の整数"):
            pie_plot("値", abcd_df, "区分", top=bad)


def test_bar_plot_label_col_defaults_to_none():
    """bar_plot の label_col の既定値は None（他の授業でも使えるように）"""
    import inspect
    assert inspect.signature(bar_plot).parameters["label_col"].default is None


def test_bar_plot_uses_index_without_label_col():
    df = pd.DataFrame({"産業コード": [1, 2, 3],
                       "産業": ["製造業", "卸売業", "小売業"],
                       "値": [10.0, 20.0, 30.0]})
    bar_plot("値", df)
    ticks = [t.get_text() for t in matplotlib.pyplot.gcf().axes[0].get_yticklabels()]
    assert ticks == ["0", "1", "2"]
    matplotlib.pyplot.close("all")

    bar_plot("値", df, label_col="産業")
    ticks = [t.get_text() for t in matplotlib.pyplot.gcf().axes[0].get_yticklabels()]
    assert ticks == ["製造業", "卸売業", "小売業"]
    matplotlib.pyplot.close("all")


def test_pie_top_sorts_descending():
    """top を使うと、残した項目は値の大きい順に並ぶ"""
    df = pd.DataFrame({"区分": ["A", "B", "C", "D", "E"],
                       "値": [10.0, 40.0, 20.0, 5.0, 25.0]})
    pie_plot("値", df, "区分", top=3)
    names, pcts = _pie_names_and_pcts()
    assert names == ["B", "E", "C", "その他"]
    assert pcts == ["40.0%", "25.0%", "20.0%", "15.0%"]
    matplotlib.pyplot.close("all")


def test_pie_other_keeps_data_order():
    """other のときはデータの並び順を保つ（その他だけ末尾）"""
    df = pd.DataFrame({"区分": ["A", "B", "C", "D", "E"],
                       "値": [10.0, 40.0, 20.0, 5.0, 25.0]})
    pie_plot("値", df, "区分", other=["A", "D"])
    names, _ = _pie_names_and_pcts()
    assert names == ["B", "C", "E", "その他"]
    matplotlib.pyplot.close("all")


# ============================================================
# 全体の整合性
# ============================================================

def test_bar_plot_sort_by_defaults_to_none():
    """sort_by の既定値は None（省略時はデータ順）"""
    import inspect
    assert inspect.signature(bar_plot).parameters["sort_by"].default is None


def test_bar_plot_keeps_data_order_by_default():
    df = pd.DataFrame({"産業コード": [3, 1, 2],
                       "産業": ["小売業", "製造業", "卸売業"],
                       "値": [30.0, 10.0, 20.0]})
    bar_plot("値", df, label_col="産業")
    ticks = [t.get_text() for t in matplotlib.pyplot.gcf().axes[0].get_yticklabels()]
    assert ticks == ["小売業", "製造業", "卸売業"]
    matplotlib.pyplot.close("all")


def test_bar_plot_sort_by_defaults_to_descending():
    df = pd.DataFrame({"産業": ["小売業", "製造業", "卸売業"], "値": [30.0, 10.0, 20.0]})
    bar_plot("値", df, label_col="産業", sort_by="値")
    ticks = [t.get_text() for t in matplotlib.pyplot.gcf().axes[0].get_yticklabels()]
    assert ticks == ["小売業", "卸売業", "製造業"]
    matplotlib.pyplot.close("all")

    bar_plot("値", df, label_col="産業", sort_by="値", ascending=True)
    ticks = [t.get_text() for t in matplotlib.pyplot.gcf().axes[0].get_yticklabels()]
    assert ticks == ["製造業", "卸売業", "小売業"]
    matplotlib.pyplot.close("all")


def test_dataframe_error_shows_function_specific_example():
    """引数の順番を間違えたとき、その関数の正しい呼び方が示される"""
    cases = [
        (regression, ("a", "b", "notdf"), "regression("),
        (scatter_plot, ("a", "b", "notdf"), "scatter_plot("),
        (bar_plot, ("a", "notdf"), "bar_plot("),
        (box_plot, ("a", "notdf"), "box_plot("),
        (pie_plot, ("a", "notdf", "b"), "pie_plot("),
        (line_plot, ("a", "b", "notdf"), "line_plot("),
    ]
    for fn, args, expected in cases:
        with pytest.raises(TypeError) as exc:
            fn(*args)
        msg = str(exc.value)
        assert expected in msg, fn.__name__
        assert "説明変数x、被説明変数y" not in msg, fn.__name__


def test_line_plot_legend_rejects_bool():
    """pie_plot の legend=True と取り違えたときに分かるようにする"""
    df = pd.DataFrame({"年": [2016, 2017], "A": [1.0, 2.0], "B": [2.0, 3.0]})
    with pytest.raises(TypeError, match="True や False"):
        line_plot("年", ["A", "B"], df, legend=True)


def test_scatter_plot_rejects_same_x_and_y():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="両方に指定"):
        scatter_plot("a", "a", df)


def test_sample_size_message_matches_context():
    """標本不足のメッセージが用途に合っている"""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="トレンド線"):
        scatter_plot("a", "b", df)
    with pytest.raises(ValueError, match="回帰分析"):
        regression("a", "b", df)


def test_no_stale_argument_names_in_docstrings():
    """docstring に古い引数名が残っていない"""
    import py4project.core as core
    import inspect
    src = inspect.getsource(core)
    assert "text_col" not in src
    assert "tick_label" not in src.replace("tick_labels", "")


def test_package_description_mentions_all_plots():
    """パッケージ説明が実際の関数一覧と食い違っていない"""
    import py4project
    doc = py4project.__doc__
    for word in ("回帰分析", "散布図", "折れ線", "バープロット", "ボックスプロット", "円グラフ"):
        assert word in doc, word


# ============================================================
# 公開前に見つかった境界的なケース
# ============================================================

def test_const_column_name_is_rejected():
    """'const' という列名は、定数項と区別できなくなるためエラーにする"""
    df = pd.DataFrame({"const": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 9.0]})
    with pytest.raises(ValueError, match="const"):
        regression("const", "y", df)
    with pytest.raises(ValueError, match="const"):
        regression("y", "const", df)


def test_line_plot_empty_data_gives_japanese_error():
    df = pd.DataFrame({"x": pd.Series(dtype=float), "y": pd.Series(dtype=float)})
    with pytest.raises(ValueError, match="折れ線グラフにできるデータ"):
        line_plot("x", "y", df)


def test_pie_plot_single_category():
    df = pd.DataFrame({"区分": ["A"], "値": [10.0]})
    pie_plot("値", df, "区分")
    pcts = [t.get_text() for t in matplotlib.pyplot.gcf().axes[0].texts
            if t.get_text().endswith("%")]
    assert pcts == ["100.0%"]
    matplotlib.pyplot.close("all")
