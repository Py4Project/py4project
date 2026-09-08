import japanize_matplotlib_jlite
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ============================================================
# エラーメッセージ用の内部ヘルパー
#   学生が実行時に出会うエラーを、原因と対処法が分かる日本語にする。
#   先頭に _ が付いた関数は「内部用」であり、学生が直接使うものではない。
# ============================================================

def _err(title, cause, hint):
    """学生向けエラーメッセージの共通フォーマットを組み立てる（内部用）"""
    return (f'\n【エラー】{title}\n'
            f'  原因：{cause}\n'
            f'  対処：{hint}\n')


def _check_dataframe(data, arg_name='data'):
    """dataがDataFrameかどうかを確認する（内部用）"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(_err(
            f'引数 {arg_name} がデータフレームではありません。',
            f'{arg_name} に {type(data).__name__} 型のものが渡されました。',
            '引数の順番を確認してください。'
            'この関数は「説明変数x、被説明変数y、データフレーム」の順に書きます。'
            '例： regression("人口", "所得", df) '
            '（人口が説明変数、所得が被説明変数）'))


def _check_column(data, col, arg_name):
    """指定された列がdataに存在するかを確認する（内部用）"""
    if col not in data.columns:
        columns = '、'.join(str(c) for c in data.columns)
        raise ValueError(_err(
            f'列「{col}」がデータフレームの中に見つかりません。',
            f'引数 {arg_name} に「{col}」が指定されましたが、この列は存在しません。',
            f'使える列名は次の{len(data.columns)}個です → {columns}\n'
            '        （スペルの間違い、全角と半角の違い、'
            '余分なスペースがないか確認してください）'))


def _check_numeric(data, col, arg_name):
    """指定された列が数値かどうかを確認する（内部用）"""
    if not pd.api.types.is_numeric_dtype(data[col]):
        example = data[col].iloc[0] if len(data) > 0 else '（データが空です）'
        raise TypeError(_err(
            f'列「{col}」は数値ではないため、計算やグラフに使えません。',
            f'引数 {arg_name} に指定された列「{col}」には'
            f'文字列などが入っています（例：{example!r}）。',
            '数値が入っている列を指定してください。'
            '数値のはずなのにこのエラーが出る場合は、'
            'データに「-」「※」「秘匿」などの記号が混じっている可能性があります。'))


def _check_sample_size(data, n_x=1):
    """説明変数の数に対して標本の大きさが足りているかを確認する（内部用）

    定数項と説明変数の分を推定したうえで、誤差を評価する余地
    （自由度1以上）が残る行数を必要とする。"""
    n_need = n_x + 2
    if len(data) < n_need:
        raise ValueError(_err(
            f'データが{len(data)}行しかなく、回帰分析ができません。',
            f'説明変数が{n_x}個の回帰分析には、少なくとも{n_need}行のデータが必要です。',
            '説明変数を減らすか、データの絞り込み条件を見直してください。'
            '（欠損値のある行は計算から除かれるため、'
            '見た目の行数より少なくなることがあります）'))


def _check_duplicate_x(x):
    """説明変数に同じ列が重複していないかを確認する（内部用）"""
    dup = sorted({col for col in x if x.count(col) > 1})
    if dup:
        raise ValueError(_err(
            f'説明変数に同じ列が重複しています：{"、".join(dup)}',
            '同じ列を2回入れても新しい情報は増えず、計算結果が定まりません。',
            'x に指定する列名が重複していないか確認してください。'))


def _check_x_not_y(x, y):
    """被説明変数が説明変数に含まれていないかを確認する（内部用）"""
    if y in x:
        raise ValueError(_err(
            f'列「{y}」が x と y の両方に指定されています。',
            '同じ変数で自分自身を説明することはできません。',
            'x と y には別の列を指定してください。'))


def _dropna_for_plot(data, cols):
    """作図に使う列の欠損値を除く（内部用）

    regression と同じように、欠損値のある行は計算・作図から除く。
    除いた場合は、黙って減らさずに件数を知らせる。"""
    n_before = len(data)
    data = data.dropna(subset=list(cols))
    n_dropped = n_before - len(data)
    if n_dropped > 0:
        print(f'※ 欠損値のため {n_dropped} 行を除いて作図しました。')
    return data


def _check_positive(data, col, arg_name, options='xlog'):
    """対数をとる列に0以下の値がないかを確認する（内部用）

    options には、その関数が実際に持っている対数化オプション名を渡す。
    box_plot には xlog しかないため、ylog や xylog を案内しないようにする。"""
    n_bad = (data[col] <= 0).sum()
    if n_bad > 0:
        raise ValueError(_err(
            f'列「{col}」に0以下の値が{n_bad}個あるため、対数をとれません。',
            '常用対数（log10）は、0や負の数に対しては定義されていません。',
            f'対数化のオプション（{options}）をFalseに戻すか、'
            '0以下の値を含む行を除いてから実行してください。'
            '（このまま計算すると、エラーにならないまま -inf や nan が'
            '混じった誤ったグラフになります）'))


def _label(value):
    """マーカー横に表示する文字列を安全に作る（内部用）

    数値の列がtext_colに指定された場合でもエラーにせず文字列にする。"""
    return str(value)[:6]


def _format_p(p):
    """p値を、小さすぎて 0 に見えないように整形する（内部用）"""
    if p < 0.0001:
        return '<0.0001'
    return f'{p:.4f}'


def _stars(p):
    """有意性の星印を返す（内部用）"""
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.1:
        return '*'
    return ''


def _float_format(v):
    """表の数値を小数第4位までの通常表記にする（内部用）

    pandas は桁の開きが大きい列を自動的に科学記法（1.948712e+06）にしてしまうため、
    表示のときだけ書式を固定する。表そのものは数値のままなので計算にも使える。"""
    return f'{v:.4f}'


def _unwrap(res):
    """RegressionResult でも statsmodels の結果でも、statsmodels の結果を返す（内部用）"""
    return getattr(res, 'result', res)


def _fit_ols(Y, X, robust):
    """OLS推定をおこない、結果と標準誤差の種類の説明を返す（内部用）"""
    if robust:
        return sm.OLS(Y, X).fit(cov_type='HC3'), '不均一分散頑健(HC3)'
    return sm.OLS(Y, X).fit(), '非頑健'


# ============================================================
# 回帰分析の結果オブジェクト
# ============================================================

class RegressionResult:
    """回帰分析の結果。

    Colab や Jupyter のセルの最後に置くと、結果表として表示される。
    statsmodels の結果オブジェクトが持つ属性（params、rsquared など）も
    そのまま使うことができる。

    公開している属性
      .table  : 係数の表（DataFrame）。列は数値のままなので計算にも使える。
      .result : 元の statsmodels の結果オブジェクト。
    """

    def __init__(self, result, header, table, footnote=''):
        # __getattr__ が無限に呼ばれるのを避けるため object.__setattr__ を使う
        object.__setattr__(self, 'result', result)
        object.__setattr__(self, 'table', table)
        object.__setattr__(self, '_header', header)
        object.__setattr__(self, '_footnote', footnote)

    def __getattr__(self, name):
        # 自分が持っていない属性は statsmodels の結果オブジェクトに転送する
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, 'result'), name)

    def __dir__(self):
        return sorted(set(list(super().__dir__()) + dir(self.result)))

    def __repr__(self):
        """文字だけで表示するとき（端末、print(...) など）"""
        lines = [f'{k}：{v}' for k, v in self._header]
        body = ('\n'.join(lines) + '\n\n'
                + self.table.to_string(float_format=_float_format))
        if self._footnote:
            body += '\n\n' + self._footnote
        return body

    def _repr_html_(self):
        """Colab や Jupyter で表示するとき"""
        from html import escape
        head = ''.join(
            f'<tr>'
            f'<td style="padding:1px 12px 1px 0; white-space:nowrap;">{escape(str(k))}</td>'
            f'<td style="padding:1px 0;">{escape(str(v))}</td>'
            f'</tr>'
            for k, v in self._header
        )
        # 脚注には「<」が含まれるため、HTMLにするときだけエスケープする
        foot = (f'<div style="margin-top:6px; font-size:90%;">'
                f'{escape(self._footnote)}</div>'
                if self._footnote else '')
        return (
            '<div>'
            f'<table style="border:none; margin-bottom:8px;"><tbody>{head}</tbody></table>'
            f'{self.table.to_html(float_format=_float_format)}'
            f'{foot}'
            '</div>'
        )


# ============================================================
# 学生が使う関数
# ============================================================

def regression(x, y, data, robust=False, alpha=0.05, stars=True):
    """最小二乗法（OLS）による回帰分析をおこなう。

    xを説明変数、yを被説明変数として回帰分析をおこない、
    結果を表にして返します。Colabのセルの最後に置くと表が表示されます。

    引数
    ----
    x : 文字列 または 文字列のリスト
        説明変数（yを説明するために使う変数）の列名。
        リストで複数指定すると重回帰分析になります。
        同じ列を重ねて指定したり、yと同じ列を指定したりはできません。
        例： x='特化係数'
        例： x=['特化係数', '事業所数']
    y : 文字列
        被説明変数（説明したい変数）の列名。
    data : DataFrame
        xとyの列を含むデータフレーム。
        xまたはyが欠けている行は、計算から自動的に除かれます。
    robust : True または False
        Falseなら通常の標準誤差（表には「非頑健」と表示）、
        Trueなら不均一分散に強い標準誤差（表には「不均一分散頑健(HC3)」と表示）
        を使います。
    alpha : 数値
        有意水準。信頼区間は (1 - alpha) × 100 ％ で計算されます。
        初期値0.05は95％信頼区間にあたります。
    stars : True または False
        Trueで「有意性」の列に星印を追加します。

    戻り値
    ------
    RegressionResult
        セルの最後に置くと結果表として表示されます。
        result.params のように statsmodels の属性も使えます。
        元のstatsmodelsのオブジェクトは result.result で取り出せます。

    表示される内容
    --------------
    サンプルの大きさ       : 計算に使った行数。
    欠損値のため除外       : 欠けている値があって除いた行数（あった場合のみ）。
    決定係数 R²            : xがyの変動をどれくらい説明できているかを示す0〜1の値。
                             1に近いほどよく説明できている。
    自由度調整済み決定係数 : 説明変数の数が増えた分を割り引いた決定係数。
                             説明変数の数が違うモデルを見比べるときに使う。
    標準誤差の種類         : robust の設定に応じて「非頑健」または
                             「不均一分散頑健(HC3)」と表示される。
    信頼区間の水準         : alpha の設定に応じた水準（初期値は95％）。

    表の各列
    --------
    推定値   : 推定された係数。
    p値      : その係数が0であると考えたときに、
               これほどの値が偶然得られる確率。小さいほど0とは考えにくい。
    信頼区間 : 係数の値がこの範囲に入ると考えられる区間。
    有意性   : p値の小ささに応じた星印（stars=True のときだけ表示）。

    使用例
    ------
    >>> regression('特化係数', '地域固有効果', df)
    >>> regression(['特化係数', '事業所数'], '地域固有効果', df)
    >>> regression('特化係数', '地域固有効果', df, robust=True)

    注意点
    ------
    セルの最後に置かないと結果表は表示されません。
    変数に入れた場合（res = regression(...)）は、
    次の行に res と書くと表示されます。
    """
    _check_dataframe(data)

    # 文字列1つで渡された場合もリストとして扱う
    if isinstance(x, str):
        x = [x]
    x = list(x)

    _check_duplicate_x(x)
    _check_x_not_y(x, y)

    _check_column(data, y, 'y')
    _check_numeric(data, y, 'y')
    for col in x:
        _check_column(data, col, 'x')
        _check_numeric(data, col, 'x')

    # 使用する列だけ取り出し、欠損値のある行を除く
    df = data[[y] + x].dropna()
    n_dropped = len(data) - len(df)

    # 行数の確認は、欠損値を除いたあとの行数でおこなう
    _check_sample_size(df, len(x))

    Y = df[y]
    X = sm.add_constant(df[x], has_constant='add')   # 定数項（切片）を必ず追加

    result, se_type = _fit_ols(Y, X, robust)

    # ---- 見出し ----
    header = [('被説明変数', y),
              ('サンプルの大きさ', int(result.nobs))]
    if n_dropped > 0:
        header.append(('欠損値のため除外', f'{n_dropped} 行'))
    header += [('決定係数', f'{result.rsquared:.4f}'),
               ('自由度調整済み決定係数', f'{result.rsquared_adj:.4f}'),
               ('標準誤差の種類', se_type),
               ('信頼区間の水準', f'{(1 - alpha) * 100:.0f} ％')]

    # ---- 係数の表 ----
    ci = result.conf_int(alpha=alpha)
    table = pd.DataFrame({
        '推定値':      result.params.round(4),
        'p値':         result.pvalues.map(_format_p),
        '信頼区間下限': ci[0].round(4),
        '信頼区間上限': ci[1].round(4),
    })

    footnote = ''
    if stars:
        table['有意性'] = result.pvalues.map(_stars)
        footnote = '星印：*** p<0.01　** p<0.05　* p<0.1'

    table.index.name = '変数'
    table = table.rename(index={'const': '定数項'})

    return RegressionResult(result, header, table, footnote)


def regression_plot(x=None, y=None, data=None, title=None, xlabel=None, ylabel=None,
                    line=True, text_col=None, robust=False, *, ols_res=None):
    """散布図に回帰直線を重ねて表示する。

    引数
    ----
    x : 文字列
        横軸にする変数（説明変数）の列名。
    y : 文字列
        縦軸にする変数（被説明変数）の列名。
    data : DataFrame
        xとyの列を含むデータフレーム。
        xまたはyが欠けている行は、作図と計算から自動的に除かれます。
    title : 文字列
        グラフの上に表示するタイトル。
        省略した場合、line=True なら推定式だけが、
        line=False ならタイトルなしになります。
    xlabel : 文字列
        横軸のラベル。省略するとxの列名がそのまま使われます。
    ylabel : 文字列
        縦軸のラベル。省略するとyの列名がそのまま使われます。
    line : True または False
        Trueで回帰直線を引き、推定式をタイトルに表示します。
    text_col : 文字列
        各点の横に表示する名前が入っている列名。
        省略すると名前は表示されません。
        例： text_col='産業'、text_col='都道府県'
        名前は先頭6文字までが表示されます。
    robust : True または False
        Falseなら通常の標準誤差、Trueなら不均一分散に強い標準誤差（HC3）
        を使います。回帰直線そのものはどちらでも変わりません。
    ols_res : regression の結果（キーワード指定のみ）
        regression で計算した結果を渡すと、計算し直さずにその結果で
        図を描きます。この場合 x、y、data の指定は不要です。
        text_col と robust は一緒に使えません。
        robust を変えたい場合は regression の側で指定してください。

    使用例
    ------
    >>> regression_plot('特化係数', '地域固有効果', df, title='大阪府')
    >>> regression_plot('特化係数', '地域固有効果', df, text_col='産業')
    >>> res = regression('特化係数', '地域固有効果', df)
    >>> regression_plot(ols_res=res)

    注意点
    ------
    ols_res を渡す場合、説明変数が1つのときだけ図を描けます。
    重回帰の結果は横軸を1つに決められないため、図にできません。

    欠損値のある行を除いた結果、データが3行に満たない場合はエラーになります。
    """
    if ols_res is not None:
        # ---- regression の結果をそのまま使う ----
        # 使えない組み合わせを先に弾く
        if text_col is not None:
            raise ValueError(_err(
                'ols_res と text_col は一緒に使えません。',
                '回帰分析の結果には、点の横に表示する名前が入っていません。',
                'x、y、data を直接渡してください。'))
        if robust:
            raise ValueError(_err(
                'ols_res と robust は一緒に使えません。',
                '標準誤差の種類は、regression を実行した時点で決まっています。',
                'regression(..., robust=True) の結果を渡してください。'))

        result = _unwrap(ols_res)
        exog_names = list(result.model.exog_names)

        if 'const' not in exog_names:
            raise ValueError(_err(
                '定数項のない回帰分析の結果は図にできません。',
                '切片が求まらないため、直線を引くことができません。',
                'regression で計算した結果を渡してください。'))

        if len(exog_names) != 2:
            raise ValueError(_err(
                '重回帰の結果は散布図にできません。',
                f'渡された結果には説明変数が{len(exog_names) - 1}個あり、'
                '横軸をどれにすればよいか決められません。',
                '説明変数が1つだけの回帰分析の結果を渡すか、'
                'regression_plot(x=..., y=..., data=...) の形で'
                '図にしたい2つの変数を指定してください。'))

        # 定数項ではないほうの列を横軸に使う
        const_pos = exog_names.index('const')
        x_pos = 1 - const_pos

        _y = pd.Series(np.asarray(result.model.endog))
        _x = pd.Series(np.asarray(result.model.exog)[:, x_pos])

        y_name = result.model.endog_names
        x_name = exog_names[x_pos]

        intercept = float(result.params.iloc[const_pos])
        slope = float(result.params.iloc[x_pos])

        plot_data = None

    else:
        # ---- x、y、data から計算する ----
        _check_dataframe(data)
        _check_x_not_y([x], y)
        _check_column(data, x, 'x')
        _check_column(data, y, 'y')
        _check_numeric(data, x, 'x')
        _check_numeric(data, y, 'y')
        if text_col is not None:
            _check_column(data, text_col, 'text_col')

        # regression と同じように、欠損値のある行を除いてから推定する
        plot_data = _dropna_for_plot(data.copy(), [x, y])
        _check_sample_size(plot_data, 1)

        _x = plot_data[x]
        _y = plot_data[y]
        y_name, x_name = y, x

        X = sm.add_constant(_x, has_constant='add')
        result, _ = _fit_ols(_y, X, robust)
        intercept = float(result.params.iloc[0])
        slope = float(result.params.iloc[1])

    if xlabel is None:
        xlabel = x_name
    if ylabel is None:
        ylabel = y_name

    fig, ax = plt.subplots()
    ax.scatter(_x, _y, s=60, color='steelblue', zorder=3)

    if text_col is not None and plot_data is not None:
        for _, row in plot_data.iterrows():
            ax.annotate(_label(row[text_col]), (row[x], row[y]), fontsize=8, alpha=0.85)

    # 回帰直線とタイトルは別々に判定する
    # （title を省略したときに回帰直線が消えてしまわないようにするため）
    if line:
        xs = np.linspace(_x.min(), _x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color='crimson', linewidth=2,
                label='回帰直線')
        ax.legend()

    title_lines = []
    if title:
        title_lines.append(title)
    if line:
        title_lines.append(
            f'推定式: {ylabel} = {intercept:.4f} + {slope:.4f} × {xlabel}')
    if title_lines:
        ax.set_title('\n'.join(title_lines))

    if _y.min() < 0 < _y.max():
        ax.axhline(0, color='gray', linewidth=0.8)
    if _x.min() < 0 < _x.max():
        ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel(f'{ylabel}')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def scatter_plot(x, y, data, title=None, xlabel=None, ylabel=None,
                 line=True, text_col=None,
                 xlog=False, ylog=False, xylog=False):
    """散布図にトレンド線を重ねて表示する。

    regression_plotとよく似ていますが、こちらは推定式を表示せず、
    データの傾向を目で確かめることを目的としています。
    値の大きさが極端に違う変数を扱うときは、対数化のオプションが使えます。

    引数
    ----
    x : 文字列
        横軸にする変数の列名。
    y : 文字列
        縦軸にする変数の列名。
    data : DataFrame
        xとyの列を含むデータフレーム。
        xまたはyが欠けている行は、作図と計算から自動的に除かれます。
    title : 文字列
        グラフの上に表示するタイトル。省略するとタイトルなしになります。
    xlabel : 文字列
        横軸のラベル。省略するとxの列名がそのまま使われます。
    ylabel : 文字列
        縦軸のラベル。省略するとyの列名がそのまま使われます。
    line : True または False
        Trueでトレンド線（当てはめた直線）を引きます。
    text_col : 文字列
        各点の横に表示する名前が入っている列名。
        省略すると名前は表示されません。
        例： text_col='産業'、text_col='都道府県'
        名前は先頭6文字までが表示されます。
    xlog : True または False
        Trueで横軸の変数を常用対数（log10）に変換します。
    ylog : True または False
        Trueで縦軸の変数を常用対数（log10）に変換します。
    xylog : True または False
        Trueで横軸と縦軸の両方を常用対数に変換します。

    使用例
    ------
    >>> scatter_plot('従業者数', '売上高', df)
    >>> scatter_plot('従業者数', '売上高', df, xylog=True)
    >>> scatter_plot('従業者数', '売上高', df, text_col='産業')

    注意点
    ------
    対数をとる列に0以下の値が含まれているとエラーになります。
    トレンド線を引く場合（line=True）、欠損値を除いたあとのデータが
    3行に満たないとエラーになります。
    """
    _check_dataframe(data)
    _check_column(data, x, 'x')
    _check_column(data, y, 'y')
    _check_numeric(data, x, 'x')
    _check_numeric(data, y, 'y')
    if text_col is not None:
        _check_column(data, text_col, 'text_col')

    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = y

    # regression と同じように、欠損値のある行を除いてから作図する
    data = _dropna_for_plot(data.copy(), [x, y])
    if line:
        _check_sample_size(data, 1)

    if xlog or xylog:
        _check_positive(data, x, 'x', options='xlog / xylog')
    if ylog or xylog:
        _check_positive(data, y, 'y', options='ylog / xylog')

    if xlog or xylog:
        data[x + '_log'] = np.log10(data[x])
        x_plot = x + '_log'
    else:
        x_plot = x

    if ylog or xylog:
        data[y + '_log'] = np.log10(data[y])
        y_plot = y + '_log'
    else:
        y_plot = y

    _x = data[x_plot]
    _y = data[y_plot]

    fig, ax = plt.subplots()
    ax.scatter(_x, _y, s=60, color='steelblue', zorder=3)

    if text_col is not None:
        for _, row in data.iterrows():
            ax.annotate(_label(row[text_col]), (row[x_plot], row[y_plot]),
                        fontsize=8, alpha=0.85)

    if line:
        X = sm.add_constant(_x, has_constant='add')
        res = sm.OLS(_y, X).fit()
        intercept = float(res.params.iloc[0])
        slope = float(res.params.iloc[1])
        xs = np.linspace(_x.min(), _x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color='crimson', linewidth=2,
                label='トレンド線')
        ax.legend()

    if _y.min() < 0 < _y.max():
        ax.axhline(0, color='gray', linewidth=0.8)
    if _x.min() < 0 < _x.max():
        ax.axvline(0, color='gray', linewidth=0.8)

    if title:
        ax.set_title(title)
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel(f'{ylabel}')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# シフトシェア分析の3つの効果に対応する色。
# 授業で使うスライドや資料と同じ配色になるように固定してある。
_COLOR_MAP = {
    '全国成長効果': '#5B7C99',
    '産業構成効果': '#C9873A',
    '地域固有効果': '#1F7A5C',
    '実際の変化': '#3D4551',
}


def bar_plot(x, data, title=None, xlabel=None, unit=1000,
             sort_by='産業コード', tick_label='産業', ascending=None):
    """横向きの棒グラフを表示する。

    産業ごとの値を横に並べた棒グラフを描きます。
    複数の列をリストで渡すと、それぞれを並べて比較できます。
    シフトシェア分析の3つの効果（全国成長効果・産業構成効果・地域固有効果）は
    授業資料と同じ色で自動的に塗り分けられます。

    引数
    ----
    x : 文字列 または 文字列のリスト
        棒の長さにする変数の列名。
        リストで複数渡すと、産業ごとに並べて表示されます。
        例： x='地域固有効果'
        例： x=['全国成長効果', '産業構成効果', '地域固有効果']
    data : DataFrame
        xの列を含むデータフレーム。
    title : 文字列
        グラフの上に表示するタイトル。省略するとタイトルなしになります。
    xlabel : 文字列
        横軸のラベル。省略すると列名と単位から自動でつくられます。
    unit : 数値 または 辞書
        値を割る数。表示の桁を調整するために使います。
        1000 なら千単位、0.01 なら百分率（％）の表示になります。
        列ごとに変えたい場合は辞書で渡します。
        例： unit={'地域固有効果': 1000, '特化係数': 1}
    sort_by : 文字列 または None
        棒を並べる順番を決める列名。初期値は '産業コード' です。
        Noneを渡すと、xの最初の列の値で並べ替えます。
    tick_label : 文字列
        縦軸に表示する名前が入っている列名。初期値は '産業' です。
        都道府県ごとのグラフを描く場合は tick_label='都道府県' と指定します。
    ascending : True または False（省略可）
        Trueで昇順（小さい順）、Falseで降順（大きい順）に並べます。
        省略した場合は次のように自動で決まります。
          ・sort_by が '産業コード' のまま  → 昇順（産業コード順に並ぶ）
          ・sort_by を別の列に変えた場合    → 降順（値の大きい順に並ぶ）

    使用例
    ------
    >>> bar_plot('地域固有効果', df, title='大阪府')
    >>> bar_plot(['全国成長効果', '産業構成効果', '地域固有効果'], df)
    >>> bar_plot('地域固有効果', df, sort_by='地域固有効果')
    """
    _check_dataframe(data)

    if isinstance(x, pd.DataFrame):
        raise TypeError(_err(
            '引数 x にデータフレームが渡されています。',
            'この関数は「列名を先、データフレームを後」に書きます。',
            '例： bar_plot("地域固有効果", df) '
            '（bar_plot(df, "地域固有効果") ではありません）'))

    x_list = [x] if isinstance(x, str) else list(x)

    for col in x_list:
        _check_column(data, col, 'x')
        _check_numeric(data, col, 'x')
    _check_column(data, tick_label, 'tick_label')

    # unit を列ごとの辞書に正規化
    if isinstance(unit, dict):
        unit_map = {col: unit.get(col, 1000) for col in x_list}
    else:
        unit_map = {col: unit for col in x_list}

    for col, u in unit_map.items():
        if u == 0:
            raise ValueError(_err(
                'unit に 0 が指定されています。',
                f'列「{col}」の unit が 0 になっており、0で割ることはできません。',
                'unit には 1（そのまま）、1000（千単位）、'
                '0.01（％表示）などの0以外の数値を指定してください。'))

    if xlabel is None:
        distinct_units = set(unit_map.values())
        if len(distinct_units) == 1:
            xlabel = '・'.join(x_list) + f' (単位：{unit_map[x_list[0]]})'
        else:
            xlabel = '・'.join(f'{col}(単位：{unit_map[col]})' for col in x_list)

    # ソート列の決定：sort_by=Noneが明示された場合のみxの最初の要素にフォールバック
    effective_sort_col = sort_by if sort_by is not None else x_list[0]
    if effective_sort_col not in data.columns:
        columns = '、'.join(str(c) for c in data.columns)
        extra = ''
        if sort_by == '産業コード':
            extra = ('\n        （sort_by は指定しなければ "産業コード" が使われます。'
                     'この列がデータにない場合は、sort_by で別の列名を指定してください）')
        raise ValueError(_err(
            f'並べ替えに使う列「{effective_sort_col}」が見つかりません。',
            f'sort_by に「{effective_sort_col}」が指定されましたが、'
            f'この列は存在しません。{extra}',
            f'使える列名は次の{len(data.columns)}個です → {columns}'))

    # ascending の決定
    if ascending is None:
        ascending = (sort_by == '産業コード')

    data = data.sort_values(effective_sort_col, ascending=ascending)

    fig, ax = plt.subplots(figsize=(9, 6))

    n = len(x_list)
    y = np.arange(len(data))
    bar_height = 0.8 / n

    for i, col in enumerate(x_list):
        offset = (i - (n - 1) / 2) * bar_height
        color = _COLOR_MAP.get(col, plt.cm.tab10(i / max(n - 1, 1)))
        ax.barh(y + offset, data[col] / unit_map[col], height=bar_height,
                color=color, label=col)

    if (data[x_list] < 0).any().sum() > 0:
        ax.axvline(0, color='gray', linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(data[tick_label])
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    if n > 1:
        ax.legend()
    plt.tight_layout()
    plt.show()


def box_plot(x, data, title=None, text_col=None, xlabel=None, xlog=False):
    """箱ひげ図を表示する。

    データの散らばり方（中央値・四分位数・外れ値）を確認するための図です。
    箱の中の線が中央値、三角の印が平均値を表します。
    text_col を指定すると、外れ値の横に名前が表示されるので、
    どの産業や都道府県が極端な値をとっているかが分かります。

    引数
    ----
    x : 文字列 または 文字列のリスト
        箱ひげ図にする変数の列名。
        リストで複数渡すと、横に並べて比較できます。
        例： x='地域固有効果'
        例： x=['全国成長効果', '産業構成効果', '地域固有効果']
    data : DataFrame
        xの列を含むデータフレーム。
    title : 文字列
        グラフの上に表示するタイトル。省略するとタイトルなしになります。
    text_col : 文字列
        外れ値の横に表示する名前が入っている列名。
        省略すると名前は表示されず、外れ値は点だけで表示されます。
        例： text_col='産業'、text_col='都道府県'
        名前は先頭6文字までが表示されます。
    xlabel : 文字列のリスト
        横軸に表示するラベルのリスト。
        省略するとxの列名がそのまま使われます。
    xlog : True または False
        Trueでデータを常用対数（log10）に変換してから箱ひげ図を描きます。

    使用例
    ------
    >>> box_plot('地域固有効果', df)
    >>> box_plot(['全国成長効果', '産業構成効果', '地域固有効果'], df)
    >>> box_plot('地域固有効果', df, text_col='産業')
    >>> box_plot('売上高', df, xlog=True)

    注意点
    ------
    xlog=True にする場合、対象の列に0以下の値が含まれているとエラーになります。
    box_plot に ylog や xylog はありません。
    """
    _check_dataframe(data)

    if isinstance(x, pd.DataFrame):
        raise TypeError(_err(
            '引数 x にデータフレームが渡されています。',
            'この関数は「列名を先、データフレームを後」に書きます。',
            '例： box_plot("地域固有効果", df) '
            '（box_plot(df, "地域固有効果") ではありません）'))

    if (xlabel is None) and (not isinstance(x, list)):
        x = [x]
        xlabel = x
    elif (xlabel is None) and isinstance(x, list):
        xlabel = x
    elif not isinstance(x, list):
        x = [x]

    if not isinstance(xlabel, list):
        xlabel = [xlabel]

    for col in x:
        _check_column(data, col, 'x')
        _check_numeric(data, col, 'x')
    if text_col is not None:
        _check_column(data, text_col, 'text_col')

    if len(xlabel) != len(x):
        raise ValueError(_err(
            'xlabel の数と x の数が合っていません。',
            f'x には{len(x)}個の列名が指定されましたが、'
            f'xlabel には{len(xlabel)}個のラベルが指定されています。',
            'xlabel は省略できます。指定する場合は、'
            'x と同じ数のラベルをリストで渡してください。'))

    if xlog:
        for col in x:
            _check_positive(data, col, 'x', options='xlog')

    data = data.copy()

    fig, ax = plt.subplots()

    if xlog:
        x_log = [s + '_log' for s in x]
        tmp = np.log10(data[x])
        tmp.columns = x_log
        data = pd.concat([data, tmp], axis='columns')
        ax.boxplot(data[x_log], tick_labels=xlabel, showmeans=True, showfliers=False)
        cols = x_log
    else:
        ax.boxplot(data[x], tick_labels=xlabel, showmeans=True)
        cols = x

    for i, col in enumerate(cols, start=1):
        _data = data[col]
        q1, q3 = _data.quantile(0.25), _data.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cond = (data[col] < lower) | (data[col] > upper)
        outliers = data.loc[cond, :]

        # 外れ値の点は常に表示する
        # （xlog=True のときは showfliers=False にしているため、ここで描く必要がある）
        ax.scatter([i] * len(outliers), outliers[col], color='C0', zorder=3)

        # 名前は text_col が指定されたときだけ添える
        if text_col is not None:
            for _, row in outliers.iterrows():
                ax.annotate(_label(row[text_col]), (i, row[col]),
                            textcoords="offset points", xytext=(6, 0),
                            fontsize=8, ha='left')

    if title:
        ax.set_title(title)
    plt.show()
