# 変更履歴

## 0.3.0（2026-09-09）

### 破壊的変更

- 「名前が入った列」を指定する引数名を `label_col` に統一した。
  `bar_plot` の `tick_label` と、`box_plot`・`scatter_plot`・`regression_plot` の
  `text_col` が、すべて `label_col` になる。
- `bar_plot` の `label_col` の既定値を `None` にした。
  省略した場合は行の名前（インデックス）が縦軸に表示される。
  産業名を出すには `label_col='産業'` と指定する。
- `bar_plot` の `sort_by` の既定値を `None` にした。
  省略した場合はデータの並び順のまま表示する。
  `sort_by` を指定したときの向きは、既定で降順（値の大きい順）になる。
  これで「産業コード」列のないデータでも bar_plot が使える。

### 追加

- `line_plot` を追加した。折れ線グラフを描く。`y` にリストを渡すと複数系列を
  重ねて表示し、凡例を出す。横軸の値で昇順に並べ替えてから線を引く。
  年や四半期を日付へ自動変換することはしない。
- `pie_plot` を追加した。円グラフをパーセント表示で描く。
  `legend=True` で凡例を右側に置く。負の値や合計0のデータは
  日本語のエラーで知らせる。`label_col` が同じ行は合計する。
  `top=5` で大きいほうから5項目を残し（残した項目は値の大きい順に並ぶ）、
  `other=['A','B']` で指定した項目をまとめられる。
  まとめた扇形は円の最後に灰色で置かれ、
  名前は `other_label`（初期値 'その他'）で変更できる。

### 変更

- 引数の順番を間違えたときのエラーで、その関数の正しい呼び方を示すようにした。
  以前はどの関数でも regression の例が表示されていた。
- 標本の大きさが足りないときのメッセージを用途に合わせた
  （`scatter_plot` では「トレンド線」と表示する）。

### 修正

- `scatter_plot` でも `x` と `y` に同じ列を指定できないようにした。
  以前は意味のない対角線が描かれていた。
- `line_plot` の `legend` に `True` / `False` を渡したときに、
  `pie_plot` の `legend` との違いを説明するエラーを出すようにした。
  以前は英語の TypeError が出ていた。
- 内部ヘルパー `_label` の説明に残っていた古い引数名を修正した。
- `const` という名前の列を回帰分析に使うと、結果表に「定数項」が
  2行できてしまう問題を修正した。日本語のエラーで知らせる。
- `line_plot` で欠損値を除いた結果データが残らないとき、
  英語のエラーが出ていた問題を修正した。

### その他

- パッケージの説明文（README、`__init__.py`、`pyproject.toml`）に
  折れ線グラフと円グラフを加えた。
- 使い方のノートブック `examples/example.ipynb` に、折れ線グラフ・円グラフ・
  `sort_by`・`regression` の結果の使い方の節を追加した。
  これで公開している7つの関数すべてが収録された。
- README の `examples/example.ipynb` へのリンクを絶対URLにした。
  PyPI の説明欄では相対リンクが機能しないため。
- GitHub Actions を最新版に更新した（checkout@v6、setup-python@v6、
  upload-artifact@v5、download-artifact@v6）。Node.js 20 の非推奨警告が出なくなる。

## 0.2.2（2026-09-08）

### 破壊的変更

- `regression_plot` と `scatter_plot` の `text` 引数を廃止し、`text_col` に統合した。
  `text_col` を省略すると名前は表示されず、指定すると表示される。
  これで `box_plot` を含む3つの作図関数が同じ指定方法になった。

### 変更

- 欠損値の扱いを統一した。`regression_plot` と `scatter_plot` はエラーにせず、
  `regression` と同じように欠損値のある行を除いて作図するようにした。
  除いた場合は行数を知らせる。
- `ols_res` と `robust` の同時指定をエラーにした。
  以前は `robust` が黙って無視されていた。

### 修正

- 標本の大きさの確認を、欠損値を除いたあとの行数でおこなうようにした。
  除いた結果2行しか残らない場合でも計算が実行されてしまう問題を修正。
- 必要な行数を説明変数の数に応じて判定するようにした。
  自由度が足りず p値が `nan` になる結果が表示される問題を修正。
- 説明変数の重複（`x=['a','a']`）と、`x` と `y` への同じ列の指定を
  日本語のエラーで知らせるようにした。
  以前は壊れた表や内部エラーがそのまま出ていた。
- 定数項のない結果を `ols_res` に渡したとき、「重回帰」という誤った説明が
  表示される問題を修正。
- 対数化のエラーで、その関数にないオプション名（`box_plot` の `ylog` など）を
  案内しないようにした。

### その他

- README を現在の動作に合わせて修正した。
  依存パッケージの記述（`scipy` → `statsmodels`）、`regression` と `box_plot` の説明、
  および実行するとエラーになっていた使用例を修正した。
- docstring を実際の動作に合わせて修正した。
  `regression_plot` の `title` の説明、`regression` の説明変数の説明、
  結果表の各項目の説明、欠損値の扱いなど。
- 未使用になった内部ヘルパー `_check_missing` を削除した。

## 0.2.1（2026-09-06）

### 変更

- `box_plot` の既定値を `title=None`、`text_col=None` に変更した。
  `text_col` を省略した場合、外れ値は点だけで表示され、名前は表示されない。
- 作図関数4つ（`regression_plot`、`scatter_plot`、`bar_plot`、`box_plot`）の
  `title` の既定値を `None` に統一した。省略時はタイトルなしになる。
- 結果表に表示する標準誤差の種類の表記を「非頑健」「不均一分散頑健(HC3)」に変更した。

### 修正

- `scatter_plot` で `title` を省略するとタイトルに文字列 `None` が表示される問題を修正。
- 回帰分析の結果表で、桁の大きな値が科学記法（`1.948712e+06`）で表示される問題を修正。
  表示のみ書式を固定したので、表の列は数値のままで計算にも使える。

### その他

- パッケージのメタデータに作者名とメールアドレスを設定した。
- 使い方を示す `examples/`（`example.ipynb`、`data_project.csv`）を追加した。

## 0.2.0（2026-09-05）

### 破壊的変更

- `regression` を scipy から statsmodels ベースに変更し、戻り値として結果オブジェクト
  `RegressionResult` を返すようにした。Colab のセルの最後に置くと結果表が表示される。
- 全関数で引数の順番を `(x, y, data)` に統一した。
- `regression_plot` の `ols_res` をキーワード専用引数にした。
- `regression_plot` と `scatter_plot` の既定値を `text=False`、`text_col=None` に変更した。
  `text=True` にする場合は `text_col` の指定が必要。

### 追加

- `regression` に `robust`（HC3 標準誤差）、`alpha`（有意水準）、`stars`（有意性の星印）を追加。
- `regression` の `x` にリストを渡せるようにし、重回帰に対応した。
- `regression_plot` に `robust` を追加し、`regression` と揃えた。
- `regression_plot` に `ols_res` を追加。`regression` の結果を渡すと、再推定せずにその結果で作図する。

### 修正

- `regression_plot` で `title=''` を渡すと回帰直線が描かれなかった問題を修正。
- `regression_plot` に `ols_res` を渡したとき、タイトルの変数名が `None` と表示される問題を修正。
  推定結果の endog / exog 名を使うようにした。
- 数値の列を `text_col` に指定すると `IndexError` になる問題を修正。
- エラーメッセージを全関数で【エラー】形式の日本語に統一した。

### その他

- 依存に `statsmodels` を追加し、`scipy` への直接依存をなくした。
- テストを、表示される文字列ではなく戻り値の数値を検証する方式に変更した。

## 0.1.0（2026-08-31）

- 初回リリース。`regression`、`regression_plot`、`scatter_plot`、`bar_plot`、`box_plot` を収録。
