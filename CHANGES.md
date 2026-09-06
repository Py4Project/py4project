# 変更履歴

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
