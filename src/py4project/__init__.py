"""py4project: 授業用分析・可視化ユーティリティ

pandas DataFrame から回帰分析・散布図・横バープロット・
ボックスプロットを簡単に作成するための関数群です。
"""

from .core import (
    regression,
    regression_plot,
    scatter_plot,
    bar_plot,
    box_plot,
)

__version__ = "0.1.0"

__all__ = [
    "regression",
    "regression_plot",
    "scatter_plot",
    "bar_plot",
    "box_plot",
]
