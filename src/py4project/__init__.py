"""py4project: 「研究プロジェクト2」向けの分析・可視化ユーティリティ

学生・教員が pandas DataFrame から回帰分析・散布図・横バープロット・
ボックスプロットを簡単に作成するための関数群です。
"""

from .core import (
    regression,
    regression_plot,
    scatter_plot,
    bar_plot,
    box_plot,
)

__version__ = "0.2.1"

__all__ = [
    "regression",
    "regression_plot",
    "scatter_plot",
    "bar_plot",
    "box_plot",
]
