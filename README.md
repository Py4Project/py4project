# py4project

授業用分析・可視化ユーティリティです。pandas DataFrame から回帰分析・散布図・横バープロット・ボックスプロットを簡単に作成できます。

## インストール

```bash
pip install py4project
```

## 提供している関数

- `regression(x, y, data)` — 単回帰分析（R²、95%信頼区間、回帰式を表示）
- `regression_plot(x, y, data, ...)` — 散布図＋回帰直線
- `scatter_plot(x, y, data, ...)` — 散布図＋トレンド線（対数軸オプションあり）
- `bar_plot(x, data, ...)` — 横バープロット（複数系列・シフトシェア分解の配色に対応）
- `box_plot(x, data, ...)` — ボックスプロット（外れ値の自動ラベル表示、対数化オプションあり）

## 使用例

```python
import pandas as pd
from py4project import regression, bar_plot

df = pd.DataFrame({
    "産業": ["製造業", "卸売業", "小売業"],
    "全国成長効果": [120, 80, 60],
    "産業構成効果": [-20, 10, 5],
    "地域固有効果": [30, -5, 15],
})

bar_plot(["全国成長効果", "産業構成効果", "地域固有効果"], df, title="シフトシェア分解")
```

## 依存パッケージ

`numpy`, `pandas`, `scipy`, `matplotlib`, `japanize_matplotlib_jlite`（日本語フォントを自動設定します）

## ライセンス

MIT License
