"""Reusable Streamlit presentation components."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st
from pandas.api.types import is_object_dtype

from src.dashboard.formatting import format_number, format_status, metric_label


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def final_test_banner(result: dict[str, Any] | None = None) -> None:
    if result and not result.get("_error") and result.get("status") == "completed":
        st.success("✅ Final test evaluation: COMPLETED ONCE — model selection remains frozen.")
    else:
        st.success("🔒 Final test evaluation: LOCKED — reserved final test has not been used for model selection.")


def artifact_warning(value: pd.DataFrame | dict[str, Any], fallback: str) -> bool:
    error = value.attrs.get("error") if isinstance(value, pd.DataFrame) else value.get("_error")
    if error:
        st.warning(str(error or fallback))
        return True
    return False


def metric_cards(row: pd.Series | dict[str, Any], metrics: tuple[str, ...]) -> None:
    columns = st.columns(min(4, len(metrics)))
    for index, metric in enumerate(metrics):
        value = row.get(f"{metric}_mean", row.get(metric))
        columns[index % len(columns)].metric(metric_label(metric), format_number(value))


def status_pill(label: str, value: Any) -> None:
    st.markdown(f"**{label}:** `{format_status(value)}`")


def arrow_safe_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy whose object columns serialize consistently."""
    def display_value(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, sort_keys=True, default=str)
        if pd.isna(value):
            return value
        return str(value)

    result = frame.copy()
    for column in result.columns:
        if not is_object_dtype(result[column].dtype):
            continue
        values = result[column].dropna().tolist()
        value_types = {type(value) for value in values}
        contains_structures = any(isinstance(value, (dict, list, tuple, set)) for value in values)
        if len(value_types) > 1 or contains_structures:
            result[column] = result[column].map(display_value)
    return result


def dataframe(frame: pd.DataFrame, *, hide_index: bool = True) -> None:
    if artifact_warning(frame, "Data are not available."):
        return
    st.dataframe(arrow_safe_dataframe(frame), hide_index=hide_index, width="stretch")
