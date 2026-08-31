import japanize_matplotlib_jlite
import numpy as np
import pandas as pd
from scipy import stats
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
            'この関数は「列名を先、データフレームを後」に書きます。'
            '例： regression("人口", "所得", df) '
            '（regression(df, "人口", "所得") ではありません）'))


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


def _check_missing(data, col, arg_name):
    """指定された列に欠損値がないかを確認する（内部用）"""
    n_missing = data[col].isna().sum()
    if n_missing > 0:
        raise ValueError(_err(
            f'列「{col}」に欠けている値（欠損値）が{n_missing}個あります。',
            f'引数 {arg_name} に指定された列「{col}」に欠損値が含まれているため、'
            '計算結果がすべて nan（数値でない）になってしまいます。',
            'df = df.dropna(subset=["' + str(col) + '"]) のように'
            '欠損値のある行を除いてから、もう一度実行してください。'))


def _check_sample_size(data, n_min=3):
    """回帰分析に必要な標本の大きさがあるかを確認する（内部用）"""
    if len(data) < n_min:
        raise ValueError(_err(
            f'データの行数が{len(data)}行しかなく、回帰分析ができません。',
            f'回帰分析には最低でも{n_min}行のデータが必要です。',
            'データの絞り込み条件が厳しすぎないか確認してください。'))


def _check_positive(data, col, arg_name):
    """対数をとる列に0以下の値がないかを確認する（内部用）"""
    n_bad = (data[col] <= 0).sum()
    if n_bad > 0:
        raise ValueError(_err(
            f'列「{col}」に0以下の値が{n_bad}個あるため、対数をとれません。',
            '常用対数（log10）は、0や負の数に対しては定義されていません。',
            '対数化のオプション（xlog / ylog / xylog）をFalseに戻すか、'
            '0以下の値を含む行を除いてから実行してください。'
            '（このまま計算すると、エラーにならないまま -inf や nan が'
            '混じった誤ったグラフになります）'))


def _label(value):
    """マーカー横に表示する文字列を安全に作る（内部用）

    数値の列がtext_colに指定された場合でもエラーにせず文字列にする。"""
    return str(value)[:6]


# ============================================================
# 学生が使う関数
# ============================================================

def regression(x, y, data):
    """回帰分析の結果を表示する。

    xを説明変数、yを被説明変数として単回帰分析をおこない、
    標本の大きさ・決定係数・傾きの95%信頼区間・回帰式を画面に表示します。

    ＊注意＊ 引数は「x（説明変数）→ y（被説明変数）→ data」の順です。
    回帰式 y = a × x + b の見た目とは順番が逆になるので気をつけてください。

    引数
    ----
    x : 文字列
        説明変数（横軸にあたる変数）の列名。
    y : 文字列
        被説明変数（縦軸にあたる変数）の列名。
    data : DataFrame
        xとyの列を含むデータフレーム。

    表示される内容
    --------------
    標本の大きさ n : 分析に使ったデータの行数。
    決定係数 R²    : xがyの変動をどれくらい説明できているかを示す0〜1の値。
                     1に近いほどよく説明できている。
    信頼区間 (95%) : 傾きの値がこの範囲に入ると考えられる区間。
                     この区間が1をまたいでいなければ、傾きは1と異なると判断できる。
    回帰式         : 推定された直線の式。

    使用例
    ------
    >>> regression('全国成長効果', '実際の変化', df)
    """
    _check_dataframe(data)
    _check_column(data, x, 'x')
    _check_column(data, y, 'y')
    _check_numeric(data, x, 'x')
    _check_numeric(data, y, 'y')
    _check_missing(data, x, 'x')
    _check_missing(data, y, 'y')
    _check_sample_size(data)

    d = data.copy()
    n = len(d)
    _x = d[x]
    _y = d[y]
    r, pp = stats.pearsonr(_x, _y)
    res = stats.linregress(_x, _y)
    t = (res.slope - 1) / res.stderr
    p = 2 * (1 - stats.t.cdf(abs(t), n-2))
    t_crit = stats.t.ppf(0.975, n-2)
    ci = (res.slope - t_crit*res.stderr, res.slope + t_crit*res.stderr)
    ci_left = ci[0]
    ci_right = ci[1]

    print('推定式:  yᵢ = b₀ + b₁·xᵢ + uᵢ,  i = 1, 2, …, n')
    print(f'標本の大きさ(n): {n}')
    print(f'決定係数(R²) : {r**2:.3f}')
    print(f'b₀ の推定値  : {res.intercept:.3f}')
    print(f'b₁ の推定値  : {res.slope:.3f}')
    print(f'b₁ の95%信頼区間 : [{ci_left:.3f}, {ci_right:.3f}]')


def regression_plot(x, y, data, title='', xlabel=None, ylabel=None,
                    line=True, text=True, text_col='産業'):
    """散布図に回帰直線を重ねて表示する。

    点の散らばり方と、そこに当てはめた直線を同時に確認できます。
    タイトル部分には回帰式と傾きの95%信頼区間も表示されます。

    ＊注意＊ 引数は「x（説明変数）→ y（被説明変数）→ data」の順です。

    引数
    ----
    x : 文字列
        横軸にする変数（説明変数）の列名。
    y : 文字列
        縦軸にする変数（被説明変数）の列名。
    data : DataFrame
        xとyの列を含むデータフレーム。
    title : 文字列
        グラフの上に表示するタイトル。省略すると回帰式だけが表示されます。
    xlabel : 文字列
        横軸のラベル。省略するとxの列名がそのまま使われます。
    ylabel : 文字列
        縦軸のラベル。省略するとyの列名がそのまま使われます。
    line : True または False
        Trueで回帰直線を引きます。Falseにすると点だけの散布図になります。
    text : True または False
        Trueで各点の横に名前を表示します。点が多くて重なる場合はFalseにしてください。
    text_col : 文字列
        点の横に表示する名前が入っている列名。初期値は '産業' です。
        都道府県名を表示したい場合は text_col='都道府県' のように指定します。
        名前は先頭6文字までが表示されます。

    使用例
    ------
    >>> regression_plot('特化係数', '地域固有効果', df, title='大阪府')
    >>> regression_plot('特化係数', '地域固有効果', df, text=False)
    """
    _check_dataframe(data)
    _check_column(data, x, 'x')
    _check_column(data, y, 'y')
    _check_numeric(data, x, 'x')
    _check_numeric(data, y, 'y')
    _check_missing(data, x, 'x')
    _check_missing(data, y, 'y')
    _check_sample_size(data)
    if text:
        _check_column(data, text_col, 'text_col')

    if xlabel == None:
        xlabel = x
    if ylabel == None:
        ylabel = y

    data = data.copy()
    n = len(data)
    _x = data[x]
    _y = data[y]
    r, p = stats.pearsonr(_x, _y)
    res = stats.linregress(_x, _y)
    t = (res.slope - 1) / res.stderr
    p = 2 * (1 - stats.t.cdf(abs(t), n-2))
    t_crit = stats.t.ppf(0.975, n-2)
    ci = (res.slope - t_crit*res.stderr, res.slope + t_crit*res.stderr)
    ci_left = ci[0]
    ci_right = ci[1]

    fig, ax = plt.subplots()
    ax.scatter(_x, _y, s=60, color='steelblue', zorder=3)
    if text:
        for _, row in data.iterrows():
            ax.annotate(_label(row[text_col]), (row[x], row[y]), fontsize=8, alpha=0.85)
    if line:
        xs = np.linspace(_x.min(), _x.max(), 50)
        ax.plot(xs, res.slope*xs + res.intercept, color='crimson', linewidth=2,
                # label=f'回帰直線 (r={r:+.3f}, p値={p:.3f})')
            label=f'回帰直線')
        ax.legend()
        ax.set_title(f'{title}\n'+
                     f'推定式: {y} = {res.intercept:.2f}  {res.slope:+.2f} × {x}\n'+
                     f'スロープ係数の信頼区間 (95%)： [{ci_left:.2f}, {ci_right:.2f}]')
    else:
        ax.set_title(f'{title}')
    if _y.min() < 0 < _y.max():
        ax.axhline(0, color='gray', linewidth=0.8)
    if _x.min() < 0 < _x.max():
        ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel(f'{ylabel}')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def scatter_plot(x, y, data, title='', xlabel=None, ylabel=None,
                 line=True, text=True, text_col='産業', xlog=False, ylog=False, xylog=False):
    """散布図にトレンド線を重ねて表示する。

    regression_plotとよく似ていますが、こちらは回帰式や信頼区間を表示せず、
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
    title : 文字列
        グラフの上に表示するタイトル。
    xlabel : 文字列
        横軸のラベル。省略するとxの列名がそのまま使われます。
    ylabel : 文字列
        縦軸のラベル。省略するとyの列名がそのまま使われます。
    line : True または False
        Trueでトレンド線（当てはめた直線）を引きます。
    text : True または False
        Trueで各点の横に名前を表示します。点が多くて重なる場合はFalseにしてください。
    text_col : 文字列
        点の横に表示する名前が入っている列名。初期値は '産業' です。
        名前は先頭6文字までが表示されます。
    xlog : True または False
        Trueで横軸の変数を常用対数（log10）に変換します。
    ylog : True または False
        Trueで縦軸の変数を常用対数（log10）に変換します。
    xylog : True または False
        Trueで横軸と縦軸の両方を常用対数に変換します。
        xlog=True, ylog=True と書くのと同じ意味です。

    使用例
    ------
    >>> scatter_plot('従業者数', '売上高', df)
    >>> scatter_plot('従業者数', '売上高', df, xylog=True, text=False)

    注意点
    ------
    対数をとる列に0以下の値が含まれているとエラーになります。
    従業者数や売上高が0の行がある場合は、あらかじめ取り除いてください。
    """
    _check_dataframe(data)
    _check_column(data, x, 'x')
    _check_column(data, y, 'y')
    _check_numeric(data, x, 'x')
    _check_numeric(data, y, 'y')
    _check_missing(data, x, 'x')
    _check_missing(data, y, 'y')
    if text:
        _check_column(data, text_col, 'text_col')
    if xlog or xylog:
        _check_positive(data, x, 'x')
    if ylog or xylog:
        _check_positive(data, y, 'y')

    if xlabel == None:
        xlabel = x
    if ylabel == None:
        ylabel = y

    data = data.copy()

    if xlog or xylog:
        data[x+'_log'] = np.log10(data[x])
        _x = data[x+'_log']
    else:
        _x = data[x]

    if ylog or xylog:
        data[y+'_log'] = np.log10(data[y])
        _y = data[y+'_log']
    else:
        _y = data[y]

    fig, ax = plt.subplots()
    ax.scatter(_x, _y, s=60, color='steelblue', zorder=3)

    if (text and  xlog and ylog) or (text and xylog):
        for _, row in data.iterrows():
            ax.annotate(_label(row[text_col]), (row[x+'_log'], row[y+'_log']), fontsize=8, alpha=0.85)
    elif text and xlog and (not ylog):
        for _, row in data.iterrows():
            ax.annotate(_label(row[text_col]), (row[x+'_log'], row[y]), fontsize=8, alpha=0.85)
    elif text and (not xlog) and ylog:
        for _, row in data.iterrows():
            ax.annotate(_label(row[text_col]), (row[x], row[y+'_log']), fontsize=8, alpha=0.85)
    else:
        for _, row in data.iterrows():
            ax.annotate(_label(row[text_col]), (row[x], row[y]), fontsize=8, alpha=0.85)

    if line:
        res = stats.linregress(_x, _y)
        xs = np.linspace(_x.min(), _x.max(), 50)
        ax.plot(xs, res.slope*xs + res.intercept, color='crimson', linewidth=2, label=f'トレンド線')
        ax.legend()

    if _y.min() < 0 < _y.max():
        ax.axhline(0, color='gray', linewidth=0.8)
    if _x.min() < 0 < _x.max():
        ax.axvline(0, color='gray', linewidth=0.8)

    ax.set_title(f'{title}')
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


def bar_plot(x, data, title='', xlabel=None, unit=1000,
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
        グラフの上に表示するタイトル。
    xlabel : 文字列
        横軸のラベル。省略すると列名と単位から自動でつくられます。
    unit : 数値 または 辞書
        値を割る数。表示の桁を調整するために使います。
        1000 なら千単位、0.01 なら百分率（％）の表示になります。
        すべての列に同じ単位を使う場合は数値をひとつ渡します（初期値は1000）。
        列ごとに変えたい場合は辞書で渡します。
        例： unit={'地域固有効果': 1000, '特化係数': 1}
    sort_by : 文字列 または None
        棒を並べる順番を決める列名。初期値は '産業コード' です。
        値の大きい順に並べたいときは、その列名を指定します。
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

    if ( data[x_list] < 0 ).any().sum() > 0:
        ax.axvline(0, color='gray', linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(data[tick_label])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    if n > 1:
        ax.legend()
    plt.tight_layout()
    plt.show()


def box_plot(x, data, title='', text_col='産業', xlabel=None, xlog=False):
    """箱ひげ図を表示する。

    データの散らばり方（中央値・四分位数・外れ値）を確認するための図です。
    箱の中の線が中央値、三角の印が平均値を表します。
    外れ値には自動的に名前が表示されるので、
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
        グラフの上に表示するタイトル。
    text_col : 文字列
        外れ値の横に表示する名前が入っている列名。初期値は '産業' です。
        名前は先頭6文字までが表示されます。
    xlabel : 文字列のリスト
        横軸に表示するラベルのリスト。
        省略するとxの列名がそのまま使われます。
        xに渡した列の数と同じ数のラベルを指定してください。
    xlog : True または False
        Trueでデータを常用対数（log10）に変換してから箱ひげ図を描きます。
        値の大きさが極端に違う場合に使います。

    使用例
    ------
    >>> box_plot('地域固有効果', df)
    >>> box_plot(['全国成長効果', '産業構成効果', '地域固有効果'], df)
    >>> box_plot('売上高', df, xlog=True)

    注意点
    ------
    xlog=True にする場合、対象の列に0以下の値が含まれているとエラーになります。
    """
    _check_dataframe(data)

    if isinstance(x, pd.DataFrame):
        raise TypeError(_err(
            '引数 x にデータフレームが渡されています。',
            'この関数は「列名を先、データフレームを後」に書きます。',
            '例： box_plot("地域固有効果", df) '
            '（box_plot(df, "地域固有効果") ではありません）'))

    if (xlabel == None) and ( not isinstance(x, list) ):
        x = [x]
        xlabel = x
    elif (xlabel == None) and isinstance(x, list):
        xlabel = x
    elif not isinstance(x, list):
        x = [x]

    if not isinstance(xlabel, list):
        xlabel = [xlabel]

    for col in x:
        _check_column(data, col, 'x')
        _check_numeric(data, col, 'x')
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
            _check_positive(data, col, 'x')

    data = data.copy()

    fig, ax = plt.subplots()

    if xlog:
        x_log = [s+'_log' for s in x]
        tmp = np.log10(data[x])
        tmp.columns = x_log
        data = pd.concat([data, tmp], axis='columns')
        ax.boxplot(data[x_log], tick_labels=xlabel, showmeans=True, showfliers=False)

        for i, col in enumerate(x_log, start=1):
                _data = data[col]
                q1, q3 = _data.quantile(0.25), _data.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                cond = ( data[col] < lower ) | ( data[col] > upper )
                outliers = data.loc[cond,:]

                ax.scatter([i] * len(outliers), outliers[col], color='C0', zorder=3)
                for _, row in outliers.iterrows():
                    ax.annotate(_label(row[text_col]), (i, row[col]),
                                textcoords="offset points", xytext=(6, 0),
                                fontsize=8, ha='left')

    else:
        ax.boxplot(data[x], tick_labels=xlabel, showmeans=True)

        for i, col in enumerate(x, start=1):
            _data = data[col]
            q1, q3 = _data.quantile(0.25), _data.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            cond = ( data[col] < lower ) | ( data[col] > upper )
            outliers = data.loc[cond,:]

            ax.scatter([i] * len(outliers), outliers[col], color='C0', zorder=3)
            for _, row in outliers.iterrows():
                ax.annotate(_label(row[text_col]), (i, row[col]),
                            textcoords="offset points", xytext=(6, 0),
                            fontsize=8, ha='left')

    ax.set_title(f'{title}')
    plt.show()
