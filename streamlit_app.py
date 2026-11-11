from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="SydneyTransitBench",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DELAY_DIR = ROOT / "delay_results"
OCCUPANCY_DIR = ROOT / "occupancy_results"
BENCHMARK_DIR = ROOT / "benchmark_results"

COLORS = {
    "primary": "#176B87",
    "secondary": "#C77C32",
    "green": "#2A9D8F",
    "red": "#D95D39",
    "purple": "#6C5CE7",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1450px;}
    [data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        background: #ffffff;
    }
    [data-testid="stMetricLabel"] {color: #667085;}
    .small-note {color: #667085; font-size: 0.9rem;}
    .section-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        background: #fafafa;
        min-height: 120px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(
    frame: pd.DataFrame,
    exact: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
) -> str | None:
    lookup = {normalized(column): column for column in frame.columns}
    for candidate in exact:
        if normalized(candidate) in lookup:
            return lookup[normalized(candidate)]
    for column in frame.columns:
        name = normalized(column)
        if all(token in name for token in contains):
            return column
    return None


def numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def format_integer(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def format_number(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def show_missing_file(path: Path, purpose: str) -> None:
    st.warning(f"Missing file for {purpose}: `{path.relative_to(ROOT)}`")


def plotly_chart(figure: go.Figure) -> None:
    figure.update_layout(
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        font=dict(family="Arial"),
    )
    st.plotly_chart(figure, use_container_width=True)


def page_header(title: str, description: str) -> None:
    st.header(title)
    st.caption(description)


def overview_page() -> None:
    page_header(
        "Dataset Overview",
        "Final unified V4 dataset, pipeline summary and interactive record explorer.",
    )

    summary_path = DATA_DIR / "v4_dataset_summary.json"
    sample_path = DATA_DIR / "v4_dashboard_sample.parquet"

    if not summary_path.is_file():
        show_missing_file(summary_path, "dataset overview")
        return

    summary = load_json(str(summary_path))
    metric_columns = st.columns(5)
    metric_columns[0].metric("Records", format_integer(summary.get("row_count")))
    metric_columns[1].metric("Transport modes", format_integer(summary.get("mode_count")))
    metric_columns[2].metric("Service dates", format_integer(summary.get("service_date_count")))
    metric_columns[3].metric("Routes", format_integer(summary.get("route_count")))
    metric_columns[4].metric("Stops", format_integer(summary.get("stop_count")))

    st.markdown(
        f"**Coverage:** {summary.get('first_service_date', '—')} to "
        f"{summary.get('last_service_date', '—')}  ·  "
        f"**Common-schema fields:** {summary.get('column_count', '—')}"
    )

    st.subheader("Unified data pipeline")
    pipeline = [
        ("1", "Source datasets", "BOAM, FOAM, LOAM and ROAM"),
        ("2", "Standardisation", "Common 16-field schema"),
        ("3", "Data quality", "Missingness and duplicate checks"),
        ("4", "GTFS linkage", "Trip and stop alignment"),
        ("5", "Analytical tasks", "Delay, occupancy and storage benchmarks"),
    ]
    columns = st.columns(5)
    for column, (number, title, text) in zip(columns, pipeline):
        column.markdown(
            f"<div class='section-card'><b>{number}. {title}</b><br><br>"
            f"<span class='small-note'>{text}</span></div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    mode_data = pd.DataFrame(summary.get("records_by_mode", []))
    if not mode_data.empty:
        figure = px.bar(
            mode_data,
            x="mode",
            y="row_count",
            title="Records by transport mode",
            color_discrete_sequence=[COLORS["primary"]],
        )
        figure.update_yaxes(title="Records")
        figure.update_xaxes(title="Mode")
        with left:
            plotly_chart(figure)

date_data = pd.DataFrame(summary.get("records_by_service_date", []))
if not date_data.empty:
    date_data["service_date"] = pd.to_datetime(
        date_data["service_date"],
        errors="coerce"
    )
    date_data = date_data.dropna(
        subset=["service_date"]
    ).sort_values("service_date")

    date_data["service_date"] = date_data[
        "service_date"
    ].dt.strftime("%Y-%m-%d")

    figure = px.bar(
        date_data,
        x="service_date",
        y="row_count",
        title="Record coverage by service date",
        color_discrete_sequence=[COLORS["secondary"]],
    )
    figure.update_yaxes(title="Records")
    figure.update_xaxes(
        title="Service date",
        type="category"
    )

    with right:
        plotly_chart(figure)
    
    st.subheader("Record explorer")
    st.caption(
        "The explorer uses the reproducible dashboard sample; headline statistics above use the complete V4 dataset."
    )
    if not sample_path.is_file():
        show_missing_file(sample_path, "record explorer")
        return

    sample = load_parquet(str(sample_path))
    filtered = sample.copy()
    filter_columns = st.columns(3)

    if "mode" in filtered.columns:
        modes = sorted(filtered["mode"].dropna().astype(str).unique())
        selected_modes = filter_columns[0].multiselect("Mode", modes, default=modes)
        filtered = filtered[filtered["mode"].astype(str).isin(selected_modes)]

    if "service_date" in filtered.columns:
        dates = sorted(filtered["service_date"].dropna().astype(str).unique())
        selected_dates = filter_columns[1].multiselect("Service date", dates)
        if selected_dates:
            filtered = filtered[filtered["service_date"].astype(str).isin(selected_dates)]

    if "route_id" in filtered.columns:
        routes = sorted(filtered["route_id"].dropna().astype(str).unique())
        selected_routes = filter_columns[2].multiselect("Route", routes)
        if selected_routes:
            filtered = filtered[filtered["route_id"].astype(str).isin(selected_routes)]

    st.write(f"Showing {min(len(filtered), 500):,} of {len(filtered):,} matching sampled records")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)


def data_quality_page() -> None:
    page_header(
        "Data Quality and GTFS Matching",
        "Column completeness, logical duplicate checks and GTFS linkage coverage for V4.",
    )

    quality_path = DATA_DIR / "v4_data_quality.csv"
    matching_path = DATA_DIR / "v4_gtfs_matching.csv"
    summary_path = DATA_DIR / "v4_dataset_summary.json"

    if quality_path.is_file():
        quality = load_csv(str(quality_path))
        missing_column = find_column(quality, exact=("missing_or_blank_count", "missing_count"))
        missing_pct_column = find_column(quality, exact=("missing_pct", "missing_rate"))
        name_column = find_column(quality, exact=("column_name", "column"))

        if missing_column:
            missing_values = numeric(quality, missing_column)
            cards = st.columns(3)
            cards[0].metric("Fields checked", format_integer(len(quality)))
            cards[1].metric("Fields with missing values", format_integer((missing_values > 0).sum()))
            cards[2].metric("Total missing/blank cells", format_integer(missing_values.sum()))

        if name_column and missing_pct_column:
            chart_data = quality[[name_column, missing_pct_column]].copy()
            chart_data[missing_pct_column] = numeric(chart_data, missing_pct_column)
            chart_data = chart_data.sort_values(missing_pct_column, ascending=False)
            figure = px.bar(
                chart_data,
                x=name_column,
                y=missing_pct_column,
                title="Missing or blank values by field",
                color_discrete_sequence=[COLORS["secondary"]],
            )
            figure.update_yaxes(title="Missing values (%)")
            figure.update_xaxes(title="Field")
            plotly_chart(figure)

        with st.expander("Column-quality table", expanded=False):
            st.dataframe(quality, use_container_width=True, hide_index=True)
    else:
        show_missing_file(quality_path, "column-quality analysis")

    if summary_path.is_file():
        summary = load_json(str(summary_path))
        duplicate = summary.get("logical_duplicate_check", {})
        if duplicate:
            st.subheader("Logical duplicate check")
            duplicate_columns = st.columns(3)
            duplicate_columns[0].metric(
                "Complete-key records", format_integer(duplicate.get("complete_key_rows"))
            )
            duplicate_columns[1].metric(
                "Rows beyond first", format_integer(duplicate.get("duplicate_rows_beyond_first"))
            )
            duplicate_columns[2].metric(
                "Logical duplicate rate", f"{format_number(duplicate.get('duplicate_pct'), 4)}%"
            )
            key = " + ".join(duplicate.get("logical_key", []))
            st.caption(
                f"Logical key: {key}. This diagnostic does not by itself imply that operational records should be deleted."
            )

    st.subheader("GTFS matching")
    if not matching_path.is_file():
        show_missing_file(matching_path, "GTFS matching")
        return

    matching = load_csv(str(matching_path))
    mode_column = find_column(matching, exact=("mode",))
    row_rate = find_column(matching, exact=("row_match_rate_pct",))
    trip_rate = find_column(matching, exact=("trip_match_rate_pct",))

    overall = matching
    if mode_column:
        all_rows = matching[matching[mode_column].astype(str).str.upper() == "ALL"]
        if not all_rows.empty:
            overall = all_rows
    if not overall.empty:
        cards = st.columns(4)
        for index, (label, column) in enumerate(
            [
                ("Matched rows", "matched_rows"),
                ("Unmatched rows", "unmatched_rows"),
                ("Row match rate", row_rate),
                ("Trip match rate", trip_rate),
            ]
        ):
            if column and column in overall.columns:
                value = overall.iloc[0][column]
                suffix = "%" if "rate" in label.lower() else ""
                cards[index].metric(label, f"{format_number(value) if suffix else format_integer(value)}{suffix}")

    if mode_column and row_rate and trip_rate:
        chart_data = matching[matching[mode_column].astype(str).str.upper() != "ALL"].copy()
        chart_data = chart_data.melt(
            id_vars=[mode_column],
            value_vars=[row_rate, trip_rate],
            var_name="measure",
            value_name="match_rate_pct",
        )
        chart_data["measure"] = chart_data["measure"].map(
            {row_rate: "Row match rate", trip_rate: "Trip-instance match rate"}
        )
        figure = px.bar(
            chart_data,
            x=mode_column,
            y="match_rate_pct",
            color="measure",
            barmode="group",
            title="GTFS matching rate by mode",
            color_discrete_sequence=[COLORS["primary"], COLORS["green"]],
        )
        figure.update_yaxes(title="Match rate (%)", range=[0, 100])
        figure.update_xaxes(title="Mode")
        plotly_chart(figure)

    st.dataframe(matching, use_container_width=True, hide_index=True)


def delay_page() -> None:
    page_header(
        "Delay Prediction",
        "Comparison of delay baselines and LightGBM, with split audit and held-out error analysis.",
    )
    st.warning(
        "Preliminary delay benchmark based on the frozen V1 dataset. "
        "Replace these files when the multi-date V4 delay evaluation is complete."
    )

    metrics_path = DELAY_DIR / "v2_model_metrics_all_modes.csv"
    predictions_path = DELAY_DIR / "v2_all_heldout_predictions.csv.gz"
    split_path = first_existing(
        DELAY_DIR / "v2_split_audit.csv",
        DELAY_DIR / "v2_split_audit (1).csv",
    )

    if not metrics_path.is_file():
        show_missing_file(metrics_path, "delay model comparison")
        return

    metrics = load_csv(str(metrics_path))
    model_column = find_column(metrics, exact=("model", "model_name"))
    mode_column = find_column(metrics, exact=("mode",))
    mae_column = find_column(metrics, exact=("mae_minutes", "mae_min", "mae"), contains=("mae",))
    rmse_column = find_column(metrics, exact=("rmse_minutes", "rmse_min", "rmse"), contains=("rmse",))
    within_column = find_column(
        metrics,
        exact=("within_5_minutes_pct", "within_5_min_pct", "within_5_pct"),
        contains=("within", "5"),
    )

    display_metrics = metrics.copy()
    if mode_column:
        mode_values = display_metrics[mode_column].dropna().astype(str).unique().tolist()
        preferred = [value for value in mode_values if value.lower() in {"all", "overall", "all modes"}]
        selected_mode = st.selectbox(
            "Metric scope",
            mode_values,
            index=mode_values.index(preferred[0]) if preferred else 0,
        )
        display_metrics = display_metrics[
            display_metrics[mode_column].astype(str) == str(selected_mode)
        ]

    if model_column and mae_column:
        best = display_metrics.loc[numeric(display_metrics, mae_column).idxmin()]
        cards = st.columns(4)
        cards[0].metric("Best model", str(best[model_column]))
        cards[1].metric("Best MAE", f"{format_number(best[mae_column], 3)} min")
        if rmse_column:
            cards[2].metric("RMSE", f"{format_number(best[rmse_column], 3)} min")
        if within_column:
            cards[3].metric("Within ±5 minutes", f"{format_number(best[within_column])}%")

        values = [column for column in [mae_column, rmse_column] if column]
        chart_data = display_metrics[[model_column] + values].melt(
            id_vars=[model_column],
            value_vars=values,
            var_name="metric",
            value_name="minutes",
        )
        figure = px.bar(
            chart_data,
            x=model_column,
            y="minutes",
            color="metric",
            barmode="group",
            title="Delay model error comparison",
            color_discrete_sequence=[COLORS["primary"], COLORS["secondary"]],
        )
        figure.update_yaxes(title="Error (minutes)")
        figure.update_xaxes(title="Model")
        plotly_chart(figure)

    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    if split_path and split_path.is_file():
        with st.expander("Train–test split audit", expanded=False):
            st.dataframe(load_csv(str(split_path)), use_container_width=True, hide_index=True)

    st.subheader("Held-out prediction analysis")
    if not predictions_path.is_file():
        show_missing_file(predictions_path, "delay prediction analysis")
        return

    predictions = load_csv(str(predictions_path), nrows=150_000)
    actual_column = find_column(
        predictions,
        exact=("actual_delay", "actual_delay_minutes", "y_true", "actual"),
        contains=("actual", "delay"),
    )
    predicted_column = find_column(
        predictions,
        exact=("predicted_delay", "predicted_delay_minutes", "y_pred", "prediction"),
        contains=("pred", "delay"),
    )
    prediction_model = find_column(predictions, exact=("model", "model_name"))
    prediction_mode = find_column(predictions, exact=("mode",))

    filtered = predictions.copy()
    filter_columns = st.columns(2)
    if prediction_model:
        models = sorted(filtered[prediction_model].dropna().astype(str).unique())
        selected_model = filter_columns[0].selectbox("Model", models)
        filtered = filtered[filtered[prediction_model].astype(str) == selected_model]
    if prediction_mode:
        modes = sorted(filtered[prediction_mode].dropna().astype(str).unique())
        selected_prediction_mode = filter_columns[1].selectbox("Mode", modes)
        filtered = filtered[filtered[prediction_mode].astype(str) == selected_prediction_mode]

    if actual_column and predicted_column:
        filtered = filtered.copy()
        filtered[actual_column] = numeric(filtered, actual_column)
        filtered[predicted_column] = numeric(filtered, predicted_column)
        filtered = filtered.dropna(subset=[actual_column, predicted_column])
        filtered["absolute_error_minutes"] = (
            filtered[actual_column] - filtered[predicted_column]
        ).abs()

        left, right = st.columns(2)
        plot_sample = filtered.sample(min(5_000, len(filtered)), random_state=42) if len(filtered) else filtered
        scatter = px.scatter(
            plot_sample,
            x=actual_column,
            y=predicted_column,
            opacity=0.35,
            title="Actual vs predicted delay",
            color_discrete_sequence=[COLORS["primary"]],
        )
        if not plot_sample.empty:
            minimum = min(plot_sample[actual_column].min(), plot_sample[predicted_column].min())
            maximum = max(plot_sample[actual_column].max(), plot_sample[predicted_column].max())
            scatter.add_shape(type="line", x0=minimum, y0=minimum, x1=maximum, y1=maximum, line=dict(dash="dash"))
        with left:
            plotly_chart(scatter)

        histogram = px.histogram(
            filtered,
            x="absolute_error_minutes",
            nbins=50,
            title="Absolute-error distribution",
            color_discrete_sequence=[COLORS["secondary"]],
        )
        histogram.update_xaxes(title="Absolute error (minutes)")
        with right:
            plotly_chart(histogram)

        with st.expander("Largest held-out errors", expanded=False):
            st.dataframe(
                filtered.sort_values("absolute_error_minutes", ascending=False).head(100),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Prediction columns were loaded, but their names could not be identified automatically.")
        st.dataframe(predictions.head(500), use_container_width=True, hide_index=True)


def occupancy_page() -> None:
    page_header(
        "Occupancy Prediction",
        "LOAM V4 validation results using multi-date historical occupancy features.",
    )
    st.warning(
        "Validation results only — independent testing on unseen service dates is still outstanding. "
        "Performance on rare high-occupancy bands remains limited."
    )

    metrics_path = OCCUPANCY_DIR / "validation_metrics.csv"
    daily_path = OCCUPANCY_DIR / "daily_validation_metrics.csv"
    predictions_path = OCCUPANCY_DIR / "validation_predictions.parquet"
    report_path = OCCUPANCY_DIR / "best_classification_report.csv"
    summary_path = OCCUPANCY_DIR / "summary.json"
    audit_path = OCCUPANCY_DIR / "history_audit.json"

    if not metrics_path.is_file():
        show_missing_file(metrics_path, "occupancy model comparison")
        return

    metrics = load_csv(str(metrics_path))
    model_column = find_column(metrics, exact=("model", "model_name", "variant"))
    macro_column = find_column(metrics, exact=("macro_f1", "macro_f1_score"), contains=("macro", "f1"))
    accuracy_column = find_column(metrics, exact=("accuracy", "accuracy_score"), contains=("accuracy",))
    band_column = find_column(metrics, exact=("band_mae", "band_error", "mae_band"), contains=("band", "mae"))

    if model_column and macro_column:
        best_index = numeric(metrics, macro_column).idxmax()
        best = metrics.loc[best_index]
        cards = st.columns(4)
        cards[0].metric("Selected model", str(best[model_column]))
        cards[1].metric("Macro F1", format_number(best[macro_column], 4))
        if accuracy_column:
            accuracy = float(best[accuracy_column])
            accuracy = accuracy * 100 if accuracy <= 1 else accuracy
            cards[2].metric("Accuracy", f"{accuracy:.2f}%")
        if band_column:
            cards[3].metric("Band MAE", format_number(best[band_column], 4))

        values = [column for column in [macro_column, accuracy_column] if column]
        chart_data = metrics[[model_column] + values].melt(
            id_vars=[model_column],
            value_vars=values,
            var_name="metric",
            value_name="score",
        )
        for metric in values:
            mask = chart_data["metric"] == metric
            if chart_data.loc[mask, "score"].max() <= 1:
                chart_data.loc[mask, "score"] *= 100
        figure = px.bar(
            chart_data,
            x=model_column,
            y="score",
            color="metric",
            barmode="group",
            title="Occupancy model comparison",
            color_discrete_sequence=[COLORS["primary"], COLORS["green"]],
        )
        figure.update_yaxes(title="Score (%)")
        figure.update_xaxes(title="Model")
        plotly_chart(figure)

    st.dataframe(metrics, use_container_width=True, hide_index=True)

    if daily_path.is_file():
        daily = load_csv(str(daily_path))
        date_column = find_column(daily, exact=("service_date", "validation_date", "date"), contains=("date",))
        daily_macro = find_column(daily, exact=("macro_f1", "macro_f1_score"), contains=("macro", "f1"))
        if date_column and daily_macro:
            daily[date_column] = pd.to_datetime(daily[date_column], errors="coerce")
            figure = px.line(
                daily,
                x=date_column,
                y=daily_macro,
                color=find_column(daily, exact=("model", "model_name", "variant")),
                markers=True,
                title="Validation performance by service date",
            )
            figure.update_yaxes(title="Macro F1")
            plotly_chart(figure)

    left, right = st.columns(2)
    if report_path.is_file():
        report = load_csv(str(report_path))
        with left:
            st.subheader("Best-model classification report")
            st.dataframe(report, use_container_width=True, hide_index=True)

    if predictions_path.is_file():
        predictions = load_parquet(str(predictions_path))
        actual_column = find_column(
            predictions,
            exact=("actual_band", "true_band", "y_true", "actual", "occupancy_range"),
            contains=("actual",),
        )
        predicted_column = find_column(
            predictions,
            exact=("predicted_band", "prediction", "y_pred", "predicted"),
            contains=("pred",),
        )
        if actual_column and predicted_column:
            matrix = pd.crosstab(
                predictions[actual_column].astype(str),
                predictions[predicted_column].astype(str),
            )
            heatmap = px.imshow(
                matrix,
                text_auto=True,
                aspect="auto",
                title="Occupancy-band confusion matrix",
                labels=dict(x="Predicted band", y="Actual band", color="Records"),
                color_continuous_scale="Blues",
            )
            with right:
                plotly_chart(heatmap)
        else:
            with right:
                st.info("Prediction columns could not be identified automatically.")

    if summary_path.is_file():
        with st.expander("Experiment summary", expanded=False):
            st.json(load_json(str(summary_path)))
    if audit_path.is_file():
        with st.expander("Historical-feature audit", expanded=False):
            st.json(load_json(str(audit_path)))


def storage_page() -> None:
    page_header(
        "Storage and Query Benchmark",
        "Reproducible V4 comparison of equivalent CSV and Parquet representations.",
    )

    storage_path = BENCHMARK_DIR / "storage_benchmark_results.csv"
    query_path = BENCHMARK_DIR / "query_benchmark_results.csv"
    validation_path = BENCHMARK_DIR / "query_validation.csv"
    environment_path = BENCHMARK_DIR / "benchmark_environment.csv"

    if not storage_path.is_file():
        show_missing_file(storage_path, "storage benchmark")
        return

    storage = load_csv(str(storage_path))
    format_column = find_column(storage, exact=("format", "file_format"))
    size_column = find_column(storage, exact=("file_size_mb", "size_mb"), contains=("file", "size"))
    load_column = find_column(storage, exact=("median_load_seconds",), contains=("median", "load"))

    if format_column and size_column and load_column:
        parquet_rows = storage[storage[format_column].astype(str).str.lower() == "parquet"]
        cards = st.columns(4)
        if not parquet_rows.empty:
            parquet_row = parquet_rows.iloc[0]
            cards[0].metric("Parquet size", f"{format_number(parquet_row[size_column])} MB")
            cards[1].metric("Parquet load time", f"{format_number(parquet_row[load_column])} s")
            size_reduction = find_column(storage, contains=("size", "reduction", "pct"))
            load_reduction = find_column(storage, contains=("load", "reduction", "pct"))
            if size_reduction:
                cards[2].metric("Storage reduction", f"{format_number(parquet_row[size_reduction])}%")
            if load_reduction:
                cards[3].metric("Load-time reduction", f"{format_number(parquet_row[load_reduction])}%")

        left, right = st.columns(2)
        size_figure = px.bar(
            storage,
            x=format_column,
            y=size_column,
            text_auto=".1f",
            title="V4 storage-size comparison",
            color=format_column,
            color_discrete_map={"CSV": COLORS["secondary"], "Parquet": COLORS["primary"]},
        )
        size_figure.update_yaxes(title="File size (MB)")
        size_figure.update_xaxes(title="File format")
        with left:
            plotly_chart(size_figure)

        load_figure = px.bar(
            storage,
            x=format_column,
            y=load_column,
            text_auto=".2f",
            title="V4 complete-loading time",
            color=format_column,
            color_discrete_map={"CSV": COLORS["secondary"], "Parquet": COLORS["primary"]},
        )
        load_figure.update_yaxes(title="Median load time (seconds)")
        load_figure.update_xaxes(title="File format")
        with right:
            plotly_chart(load_figure)

    st.dataframe(storage, use_container_width=True, hide_index=True)

    st.subheader("Query performance")
    if query_path.is_file():
        query = load_csv(str(query_path))
        query_name = find_column(query, exact=("query_name", "query"))
        query_format = find_column(query, exact=("format", "file_format"))
        query_time = find_column(query, exact=("median_query_seconds",), contains=("median", "query"))
        if query_name and query_format and query_time:
            figure = px.bar(
                query,
                x=query_name,
                y=query_time,
                color=query_format,
                barmode="group",
                title="V4 query-performance comparison",
                color_discrete_map={"CSV": COLORS["secondary"], "Parquet": COLORS["primary"]},
                log_y=True,
            )
            figure.update_yaxes(title="Median query time (seconds, log scale)")
            figure.update_xaxes(title="Analytical query", tickangle=-25)
            plotly_chart(figure)
        st.dataframe(query, use_container_width=True, hide_index=True)
    else:
        show_missing_file(query_path, "query benchmark")

    if validation_path.is_file():
        validation = load_csv(str(validation_path))
        result_column = find_column(validation, exact=("results_equal", "result_equal", "equal"))
        if result_column:
            values = validation[result_column].astype(str).str.lower().isin({"true", "1", "yes"})
            if values.all():
                st.success("All CSV and Parquet benchmark queries returned identical results.")
            else:
                st.error("At least one CSV/Parquet query validation check failed.")
        with st.expander("Query-result validation", expanded=False):
            st.dataframe(validation, use_container_width=True, hide_index=True)

    if environment_path.is_file():
        with st.expander("Benchmark environment", expanded=False):
            st.dataframe(load_csv(str(environment_path)), use_container_width=True, hide_index=True)


st.title("SydneyTransitBench")
st.markdown(
    "**Interactive companion for the unified multimodal dataset, data-quality checks, "
    "predictive benchmarks and storage evaluation.**"
)

page = st.radio(
    "Navigation",
    [
        "1 · Overview",
        "2 · Data Quality",
        "3 · Delay Prediction",
        "4 · Occupancy Prediction",
        "5 · Storage Benchmark",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

if page.startswith("1"):
    overview_page()
elif page.startswith("2"):
    data_quality_page()
elif page.startswith("3"):
    delay_page()
elif page.startswith("4"):
    occupancy_page()
else:
    storage_page()

st.divider()
st.caption(
    "SydneyTransitBench · Dashboard figures are computed from the supplied V4 summaries, "
    "benchmark outputs and model-evaluation files."
)

