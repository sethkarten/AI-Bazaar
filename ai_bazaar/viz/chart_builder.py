"""
Altair chart builder: constructs charts from a DataFrame passed at construction.
Data extraction and reshaping are done by the caller; this class only builds the chart.
"""

from typing import Any, List, Optional

import altair as alt
import pandas as pd


class AltairChartBuilder:
    """
    Builds an Altair chart from a pre-prepared DataFrame.
    Caller is responsible for melting/aggregating data; use .x(), .y(), .color(), then .build().
    """

    def __init__(self, df: pd.DataFrame):
        """Store the DataFrame to chart. No copying or reshaping is done here."""
        self._df = df
        self._x_col: Optional[str] = None
        self._y_col: Optional[str] = None
        self._color_col: Optional[str] = None
        self._color_domain: Optional[List[Any]] = None
        self._color_range: Optional[List[str]] = None
        self._x_title: Optional[str] = None
        self._y_title: Optional[str] = None
        self._color_legend_title: Optional[str] = None
        self._mark: str = "line"
        self._mark_opts: dict = {}

    def x(self, column: str, title: Optional[str] = None) -> "AltairChartBuilder":
        """Set the x-axis column (and optional axis title)."""
        self._x_col = column
        self._x_title = title
        return self

    def y(self, column: str, title: Optional[str] = None) -> "AltairChartBuilder":
        """Set the y-axis column (and optional axis title)."""
        self._y_col = column
        self._y_title = title
        return self

    def color(
        self,
        column: str,
        domain: Optional[List[Any]] = None,
        range_: Optional[List[str]] = None,
        legend_title: Optional[str] = None,
    ) -> "AltairChartBuilder":
        """
        Encode color by column. If domain and range_ are provided, they define the
        scale (e.g. domain=["Metric A", "Metric B"], range_=["#blue", "#orange"]).
        Domain must match the actual values in the column or those series won't get a color.
        """
        self._color_col = column
        self._color_domain = domain
        self._color_range = range_
        self._color_legend_title = legend_title
        return self

    def mark_line(self, **kwargs) -> "AltairChartBuilder":
        """Use line mark; kwargs (e.g. strokeWidth=2) are passed to Altair."""
        self._mark = "line"
        self._mark_opts = kwargs
        return self

    def mark_point(self, **kwargs) -> "AltairChartBuilder":
        """Use point mark; kwargs are passed to Altair."""
        self._mark = "point"
        self._mark_opts = kwargs
        return self

    def mark_bar(self, **kwargs) -> "AltairChartBuilder":
        """Use bar mark; kwargs are passed to Altair."""
        self._mark = "bar"
        self._mark_opts = kwargs
        return self

    def mark_area(self, **kwargs) -> "AltairChartBuilder":
        """Use area mark; kwargs are passed to Altair."""
        self._mark = "area"
        self._mark_opts = kwargs
        return self

    def build(self) -> alt.Chart:
        """Produce the Altair chart. X and y must have been set. Axis types are inferred from dtypes (numeric -> quantitative)."""
        if self._x_col is None or self._y_col is None:
            raise ValueError("x and y columns must be set before build()")

        if self._mark == "line":
            chart = alt.Chart(self._df).mark_line(**self._mark_opts)
        elif self._mark == "point":
            chart = alt.Chart(self._df).mark_point(**self._mark_opts)
        elif self._mark == "bar":
            chart = alt.Chart(self._df).mark_bar(**self._mark_opts)
        elif self._mark == "area":
            chart = alt.Chart(self._df).mark_area(**self._mark_opts)
        else:
            chart = alt.Chart(self._df).mark_line(**self._mark_opts)

        # Infer quantitative (numbers) vs nominal (categories) from pandas dtype
        x_enc = alt.X(
            self._x_col,
            type="quantitative" if self._df[self._x_col].dtype.kind in "iufc" else "nominal",
            title=self._x_title or self._x_col,
        )
        y_enc = alt.Y(
            self._y_col,
            type="quantitative" if self._df[self._y_col].dtype.kind in "iufc" else "nominal",
            title=self._y_title or self._y_col,
        )

        encoding = {"x": x_enc, "y": y_enc}

        if self._color_col is not None:
            color_scale = None
            if self._color_domain is not None and self._color_range is not None:
                color_scale = alt.Scale(domain=self._color_domain, range=self._color_range)
            color_enc = alt.Color(
                self._color_col,
                type="nominal",
                scale=color_scale,
                legend=alt.Legend(title=self._color_legend_title or self._color_col),
            )
            encoding["color"] = color_enc

        return chart.encode(**encoding)
