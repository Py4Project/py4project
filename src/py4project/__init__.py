"""py4project: 授業向けの分析・可視化ユーティリティ

pandas DataFrame から回帰分析・散布図・折れ線グラフ・横バープロット・
ボックスプロット・円グラフを簡単に作成するための関数群です。
"""

from .core import (
    regression,
    regression_plot,
    scatter_plot,
    bar_plot,
    box_plot,
    line_plot,
    pie_plot,
)

__version__ = "0.3.0"

__all__ = [
    "regression",
    "regression_plot",
    "scatter_plot",
    "bar_plot",
    "box_plot",
    "line_plot",
    "pie_plot",
]
