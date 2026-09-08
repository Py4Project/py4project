# py4project

授業向けの分析・可視化ユーティリティです。pandas DataFrame から回帰分析・散布図・横バープロット・ボックスプロットを簡単に作成できます。

## インストール

```bash
pip install py4project
```

## 提供している関数

- `regression(x, y, data, ...)` — 回帰分析（`x` にリストを渡すと重回帰）。決定係数・係数・p値・信頼区間を表にして表示し、結果オブジェクトを返します
- `regression_plot(x, y, data, ...)` — 散布図＋回帰直線（`regression` の結果を渡すこともできます）
- `scatter_plot(x, y, data, ...)` — 散布図＋トレンド線（対数軸オプションあり）
- `bar_plot(x, data, ...)` — 横バープロット（複数系列・シフトシェア分解の配色に対応）
- `box_plot(x, data, ...)` — ボックスプロット（対数化オプションあり。`text_col` を指定すると外れ値に名前を表示）

## 使用例

```python
import pandas as pd
from py4project import regression, bar_plot

df = pd.DataFrame({
    "産業コード": [1, 2, 3, 4],
    "産業": ["製造業", "卸売業", "小売業", "建設業"],
    "事業所数": [120, 80, 60, 40],
    "従業者数": [1500, 900, 700, 380],
})

# 回帰分析（セルの最後に置くと結果表が表示されます）
regression("事業所数", "従業者数", df)

# 横バープロット
bar_plot("従業者数", df, title="産業別の従業者数")
```

より詳しい使い方は [`examples/example.ipynb`](examples/example.ipynb) を参照してください。

## 依存パッケージ

`numpy`, `pandas`, `statsmodels`, `matplotlib>=3.9`, `japanize_matplotlib_jlite`（日本語フォントを自動設定します）

## ライセンス

MIT License
