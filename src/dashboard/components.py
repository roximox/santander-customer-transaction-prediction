"""Reusable Streamlit presentation components."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.formatting import format_number, format_status, metric_label


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def final_test_banner() -> None:
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


def dataframe(frame: pd.DataFrame, *, hide_index: bool = True) -> None:
    if artifact_warning(frame, "Data are not available."):
        return
    st.dataframe(frame, hide_index=hide_index, width="stretch")
