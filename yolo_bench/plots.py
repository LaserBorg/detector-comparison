"""Plotly chart builders for benchmark results.

Design goal
-----------
Answer four distinct questions, each with the chart shape that actually suits it.
Bar charts were the starting point, but they answer "what is the value at this
category?" and most of the interesting questions here are about *relationships*:

  1. **Runtime comparison** — how do backends rank, and does the ranking hold
     across model sizes?  -> :func:`runtime_slope` (slope/dumbbell per runtime)
     or :func:`runtime_ranking_heatmap` for the full matrix.
  2. **Performance vs model quality/size** — the trade-off curve. Needs a quality
     metric, so :func:`quality_vs_speed` plots *anything* on x (GFLOPs, params,
     latency), and pairs with ``accuracy.py`` for the quantization-delta variant
     (:func:`quantization_tradeoff`).
  3. **Quantization advantage** — FP16 vs FP32 per cell, and what it cost in
     detections. -> :func:`quantization_gain_heatmap` plus
     :func:`quantization_tradeoff`.
  4. **GPU comparison** — the same measurement across machines. -> faceted
     scatter/lines grouped by ``gpu_name``.

Two structural notes:

* **Small vs large models are compared separately** by default
  (``split="size"``), because n/s models and m/l/x models sit on different ends of
  the memory-bandwidth curve; a 4-model bar group mixes two questions. The
  yolo11-vs-yolo26 question is answered by the *colour* within a size class, not by
  putting all four side by side.
* Missing measurements are never rendered as zero. Bars/lines are skipped and
  heatmap cells are left blank, so "not measured" cannot be misread as "0".

Every builder returns a ``plotly.graph_objects.Figure`` and works headless, so the
same code serves the notebook, a script and a test.
"""

from __future__ import annotations

# Colour per model so the same model keeps the same colour across every figure,
# which is what makes visually scanning "same family, different size" work.
MODEL_COLORS: dict[str, str] = {
    "yolo11s": "#636EFA",   # blue
    "yolo11l": "#EF553B",   # red
    "yolo26s": "#00CC96",   # green
    "yolo26l": "#AB63FA",   # purple
}

# Fallbacks for models not in the map above (e.g. RF-DETR later).
_EXTRA_COLORS = ("#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF")

PRECISION_OPACITY = {"fp32": 0.55, "fp16": 1.0}

# Dash per precision, so a black-and-white print or a colour-blind reader can
# still separate the two series.
PRECISION_DASH = {"fp32": "dot", "fp16": "solid"}

# Marker per runtime family where a chart needs to distinguish several runtimes in
# one legend (a scatter cannot rely on dash style alone).
RUNTIME_SYMBOLS = {
    "tensorrt": "circle", "ort_trt": "square", "ort_cuda": "diamond",
    "pytorch": "triangle-up", "openvino": "pentagon",
    "ort_cpu": "circle-open", "pytorch_cpu": "triangle-down-open",
    "openvino_cpu": "square-open",
}


def _color_for(model: str, index: int = 0) -> str:
    return MODEL_COLORS.get(model, _EXTRA_COLORS[index % len(_EXTRA_COLORS)])


def model_order(models) -> list[str]:
    """Sort models family-major then size-ascending: yolo11s, yolo11l, yolo26s, yolo26l.

    Sorting alphabetically would interleave families and put large before small
    (11l, 11s, 26l, 26s), the opposite of what a size comparison wants.
    """
    SIZE_ORDER = "nsmlx"  # nano < small < medium < large < x — note m before l

    def key(m: str):
        # Split the name into its family prefix (letters/digits) and size letter.
        stem = m
        size = stem[-1] if stem and stem[-1] in SIZE_ORDER else ""
        if size:
            stem = stem[:-1]
        family = "".join(c for c in stem if not c.isdigit())
        digits = "".join(c for c in stem if c.isdigit())
        size_rank = SIZE_ORDER.index(size) if size else 99
        # n < s < m < l < x is the actual complexity order.
        return (family, int(digits) if digits else 0, size_rank)

    return sorted(set(models), key=key)


def _pivot(df, metric: str):
    """Index (runtime, model) x precision -> metric.

    Missing combinations come back as NaN, which Plotly renders as a gap rather
    than a zero — important, since "not measured" must not look like "0 fps".
    """
    import pandas as pd

    return df.pivot_table(
        index=["runtime", "model"], columns="precision",
        values=metric, aggfunc="mean",
    ).sort_index()


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def ok_rows(df):
    """Only successfully measured rows."""
    return df[df["status"] == "ok"] if "status" in df.columns else df


def split_models(models, split: str = "size"):
    """Group models for a comparison.

    ``split="size"``  -> ``{"small": [n/s models], "large": [m/l/x models]}``
    ``split="family"``-> ``{"yolo11": [...], "yolo26": [...]}``
    ``split="all"``   -> ``{"all models": [...]}``

    These are different questions and should not share an axis: comparing
    yolo11l against yolo26s answers neither "which family is better at a given
    cost" nor "how does cost scale".
    """
    from .utils import is_small_model

    models = list(models)
    if split == "all":
        return [("all models", models)]
    if split == "family":
        groups: dict[str, list[str]] = {}
        for m in models:
            fam = "".join(c for c in m if not c.isdigit())
            groups.setdefault(fam or m, []).append(m)
        return [(k, model_order(v)) for k, v in groups.items()]
    if split == "size":
        small = [m for m in models if is_small_model(m)]
        large = [m for m in models if not is_small_model(m)]
        # "large" label covers m/l/x; name it honestly.
        out = []
        if small:
            out.append(("small (n/s)", model_order(small)))
        if large:
            out.append(("large (m/l/x)", model_order(large)))
        return out
    raise ValueError(f"unknown split {split!r}; use 'size', 'family' or 'all'")


def _resolve_runtimes(data, runtimes):
    present = list(data["runtime"].dropna().unique())
    return [r for r in (runtimes or present) if r in present]


# ---------------------------------------------------------------------------
# 1. Runtime comparison
# ---------------------------------------------------------------------------

def runtime_slope(
    df,
    *,
    metric: str = "fps",
    precision: str = "fp16",
    runtimes=None,
    split: str = "size",
    title: str | None = None,
    log_y: bool = True,
    height: int = 560,
    show_values: bool = True,
):
    """Slope chart: one line per runtime, one x-slot per model, grouped by size.

    Chosen over bars because the question here is a *ranking* that may reorder as
    the model grows. A slope line makes a crossing obvious ("openvino overtakes
    pytorch at L") while a bar group hides it, and it removes the zero baseline
    that makes CPU and CUDA backends impossible to show together.

    Log y is the default for the same reason: backends here span ~20x.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = ok_rows(df)
    data = data[data["precision"] == precision] if precision else data
    runtimes = _resolve_runtimes(data, runtimes)
    if not runtimes:
        raise ValueError("no runtimes with data")
    groups = split_models(data["model"].dropna().unique(), split)

    fig = make_subplots(
        rows=1, cols=len(groups),
        column_titles=[name for name, _ in groups],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    ylabel = _metric_label(metric)
    for col, (_, models) in enumerate(groups, start=1):
        for runtime in runtimes:
            xs, ys, texts = [], [], []
            for model in models:
                sel = data[(data["runtime"] == runtime) & (data["model"] == model)]
                if sel.empty:
                    continue
                value = float(sel[metric].mean())
                if value != value:
                    continue
                xs.append(model)
                ys.append(value)
                texts.append(f"{value:.1f}")
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, name=_runtime_label(runtime),
                    mode="lines+markers+text" if show_values else "lines+markers",
                    text=texts, textposition="middle right",
                    textfont=dict(size=9),
                    line=dict(width=2.4, color=_runtime_color(runtime, runtimes)),
                    marker=dict(size=11, symbol=RUNTIME_SYMBOLS.get(runtime, "circle"),
                                line=dict(width=1, color="white")),
                    showlegend=(col == 1),
                    legendgroup=runtime,
                    hovertemplate=(f"<b>%{{x}}</b><br>{_runtime_label(runtime)}"
                                   f"<br>{ylabel}: %{{y:.2f}}<extra></extra>"),
                ),
                row=1, col=col,
            )
        fig.update_xaxes(categoryorder="array",
                         categoryarray=models, tickangle=0, row=1, col=col)

    fig.update_yaxes(title_text=ylabel, row=1, col=1,
                     type="log" if log_y else "linear")
    fig.update_layout(
        title=title or (f"{ylabel} by runtime, split by model size "
                        f"({precision.upper() if precision else 'all precisions'})"),
        height=height, width=max(700, 380 * len(groups)),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0, title=None),
        margin=dict(t=130, b=60, l=80, r=60),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    return fig


def runtime_ranking_heatmap(
    df,
    *,
    metric: str = "fps",
    precision: str = "fp16",
    runtimes=None,
    models=None,
    title: str | None = None,
    height: int = 460,
    annotate: bool = True,
):
    """Runtime x model heatmap, coloured by ``metric``.

    The full matrix in one glance: colour answers "which runtime is fast for this
    model" without six separate bar groups, and blank cells are combinations that
    were skipped. Uses a log-scaled colour axis because CUDA and CPU backends
    differ by ~20x and a linear scale would flatten the CPU rows to one shade.
    """
    import numpy as np
    import plotly.graph_objects as go

    data = ok_rows(df)
    if precision:
        data = data[data["precision"] == precision]
    runtimes = _resolve_runtimes(data, runtimes)
    models = [m for m in (models or model_order(data["model"].unique()))
              if m in set(data["model"])]

    grid = np.full((len(runtimes), len(models)), np.nan)
    for i, runtime in enumerate(runtimes):
        for j, model in enumerate(models):
            sel = data[(data["runtime"] == runtime) & (data["model"] == model)]
            if not sel.empty:
                grid[i, j] = float(sel[metric].mean())

    text = [[("" if np.isnan(v) else f"{v:.1f}") for v in row] for row in grid]
    # log colours, but annotated with the real values.
    z = grid.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        zlog = np.log10(np.where(z > 0, z, np.nan))

    fig = go.Figure(go.Heatmap(
        z=zlog if metric == "fps" else grid,
        x=models, y=[_runtime_label(r) for r in runtimes],
        text=text if annotate else None,
        texttemplate="%{text}" if annotate else None,
        textfont=dict(size=11),
        colorscale="Viridis",
        colorbar=dict(title=(f"log10 {metric}" if metric == "fps" else metric)),
        hovertemplate=("<b>%{y}</b> · %{x}<br>"
                       + _metric_label(metric) + ": %{text}<extra></extra>"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title=title or (f"{_metric_label(metric)} by runtime and model"
                        f" ({precision.upper() if precision else 'all'})"),
        height=height, width=max(560, 190 * len(models) + 260),
        xaxis_title=None, yaxis_title=None,
        margin=dict(t=100, b=60, l=170, r=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def fps_nested_bars(
    df,
    *,
    metric: str = "fps",
    runtimes=None,
    models=None,
    title: str | None = None,
    log_y: bool = False,
    height: int = 420 + 60,
    show_text: bool = True,
):
    """Nested FP32/FP16 bar chart, one subplot per runtime.

    Args:
        df: results DataFrame (from orchestrate.read_csv or Job.to_dataframe).
        metric: column to plot (default ``fps``; ``infer_ms`` invites ``log_y``).
        runtimes: restrict/order runtimes (default: all present).
        models: restrict/order models (default: all present, size-ordered).
        log_y: log-scale the value axis — useful when CPU and CUDA backends are
            in the same figure and differ by 100x.

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = df[df["status"] == "ok"] if "status" in df.columns else df
    if metric not in data.columns:
        raise KeyError(f"metric {metric!r} not in results; have {list(data.columns)[:20]}")

    present = data["runtime"].dropna().unique().tolist()
    runtimes = [r for r in (runtimes or present) if r in present]
    if not runtimes:
        raise ValueError("no runtimes with data")

    present_models = data["model"].dropna().unique().tolist()
    models = [m for m in (models or model_order(present_models)) if m in present_models]
    if not models:
        raise ValueError("no models with data")

    grid = _pivot(data, metric)

    fig = make_subplots(
        rows=1, cols=len(runtimes),
        subplot_titles=[_runtime_label(r) for r in runtimes],
        shared_yaxes=True,
        horizontal_spacing=min(0.12, 0.6 / max(1, len(runtimes))),
    )

    ylabel = {"fps": "frames / second", "infer_ms": "ms (raw forward)",
              "e2e_ms": "ms (full loop)"}.get(metric, metric)

    # Geometric encoding, chosen so the chart survives the fp16-wins case:
    #
    #   FP32 = WIDE bar, quotient = FP16/FP32
    #   FP16 = NARROW bar laid over its centre, quotient = 1.0
    #
    # Both bars are sized as a *quotient of the cell maximum*, so their ratio is
    # preserved; the tall/short relationship never changes. The wide FP32 bar is
    # the reference; the narrow FP16 bar sits inside it when fp16 is slower and
    # pokes out of the top when fp16 is faster. Either way the visible area is
    # directly the fp16 effect — with the naive "both bars full width" version, a
    # fill to the top is ambiguous between "equal" and "regressed", which defeats
    # the chart.
    WIDTH_OUTER = 0.64
    WIDTH_INNER = 0.28

    for col, runtime in enumerate(runtimes, start=1):
        cell = {}
        for model in models:
            cell[model] = {}
            for precision in ("fp32", "fp16"):
                try:
                    value = float(grid.loc[(runtime, model), precision])
                except KeyError:
                    value = float("nan")
                if value != value:
                    value = None
                cell[model][precision] = value
        # Per-cell maximum so the narrow bar still fills most of the plot height.
        cell_max = max(
            (v for m in models for v in cell[m].values() if v is not None),
            default=0.0,
        ) or 1.0

        legend_shown: set[str] = set()
        traces = []
        # FP32 reference, wide.
        for model in models:
            v32 = cell[model]["fp32"]
            if v32 is None:
                continue
            ratio = max(0.0, min(1.0, v32 / cell_max))
            traces.append(go.Bar(
                x=[model], y=[ratio], name="FP32",
                legendgroup="fp32",
                showlegend=(col == 1 and "fp32" not in legend_shown),
                marker_color=_color_for(model, models.index(model)),
                marker_opacity=PRECISION_OPACITY["fp32"],
                marker_line=dict(width=0.6, color="rgba(0,0,0,0.3)"),
                width=WIDTH_OUTER, cliponaxis=False,
                customdata=[[v32]],
                hovertemplate=(f"<b>{model}</b><br>{runtime} · FP32"
                               f"<br>{ylabel}: %{{customdata[0]:.2f}}<extra></extra>"),
            ))
            legend_shown.add("fp32")
        # FP16 inset, narrow, on top. Its visible sliver (or overshoot) is the
        # fp16 effect.
        for model in models:
            v16 = cell[model]["fp16"]
            if v16 is None:
                continue
            ratio = max(0.0, min(1.0, v16 / cell_max))
            v32 = cell[model]["fp32"]
            if v32 is not None:
                delta_pct = (v16 / v32 - 1.0) * 100.0
                delta = f"  ({delta_pct:+.0f}% vs FP32)"
            else:
                delta = ""
            traces.append(go.Bar(
                x=[model], y=[ratio], name="FP16",
                legendgroup="fp16",
                showlegend=(col == 1 and "fp16" not in legend_shown),
                marker_color=_color_for(model, models.index(model)),
                marker_opacity=PRECISION_OPACITY["fp16"],
                marker_line=dict(width=1.4, color="#222"),
                width=WIDTH_INNER, cliponaxis=False,
                customdata=[[v16, delta]],
                hovertemplate=(f"<b>{model}</b><br>{runtime} · FP16"
                               f"<br>{ylabel}: %{{customdata[0]:.2f}}"
                               f"%{{customdata[1]}}<extra></extra>"),
            ))
            legend_shown.add("fp16")
        for trace in traces:
            fig.add_trace(trace, row=1, col=col)

        # One compact label per cell in the headroom above the bars, reading
        # "FP32 → FP16" so the comparison is explicit ("13" when there is no FP16
        # row). Two stacked labels per cell collided horizontally with the
        # neighbouring model's, since every cell is the same width.
        for i, model in enumerate(models):
            v32 = cell[model]["fp32"]
            v16 = cell[model]["fp16"]
            if v32 is None and v16 is None:
                continue
            if v32 is not None and v16 is not None:
                body = f"{v32:.0f}<span style='color:#999'>→</span>{v16:.0f}"
            else:
                body = f"{(v32 if v32 is not None else v16):.0f}"
            # Place inside the axes (axis coords, not domain) at y≈1.02. The
            # y-range is [0, 1.16] while the tallest bar is 1.0, so this lands in
            # the headroom just above the bars — above the plot area it collides
            # with the subplot titles.
            yref = "y" if col == 1 else f"y{col}"
            fig.add_annotation(
                x=model, y=1.02, xref=("x" if col == 1 else f"x{col}"),
                yref=yref, text=body, showarrow=False,
                font=dict(size=11), align="center",
            )

    # Model names are short but crowded on a multi-subplot figure; a tilt plus an
    # explicit category order keeps them legible (auto-order would reorder
    # categories and desync the per-bar traces).
    fig.update_xaxes(
        tickangle=-35, tickfont=dict(size=10),
        categoryorder="array", categoryarray=models,
    )
    # The y axis carries a normalised quotient, so hide the numbers and keep only
    # the metric name; the real values are in the labels and hovers.
    fig.update_yaxes(title_text=ylabel, range=[0, 1.16], showticklabels=False,
                     showgrid=True, row=1, col=1)

    fig.update_layout(
        title=title or (
            f"{ylabel} — wide bar = FP32, narrow bar = FP16 "
            f"(narrow overflows the wide bar when FP16 is faster)"
        ),
        barmode="overlay",           # <- the "bar in the bar" effect
        bargap=0.30,
        height=height,
        width=max(760, 320 * len(runtimes)),
        # Two-row top area: legend row above the subplot titles, so a long legend
        # cannot overlap them (which it does if both sit at y~1).
        legend=dict(orientation="h", yanchor="bottom", y=1.10,
                    xanchor="left", x=0, title=None),
        margin=dict(t=150, b=70, l=70, r=30),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        uniformtext=dict(minsize=8, mode="hide"),
    )
    return fig


def fps_grouped_bars(
    df,
    *,
    metric: str = "fps",
    runtimes=None,
    models=None,
    title: str | None = None,
    log_y: bool = False,
    height: int = 520,
    show_text: bool = True,
):
    """Conventional grouped bars: x = runtime, one bar per (model, precision).

    The complement to :func:`fps_nested_bars`. Nested bars answer "did FP16 help?";
    grouped bars answer "which runtime wins for this model?" without the inset
    hiding anything. Having both is deliberate — they make different things easy.
    """
    import plotly.graph_objects as go

    data = df[df["status"] == "ok"] if "status" in df.columns else df
    present = data["runtime"].dropna().unique().tolist()
    runtimes = [r for r in (runtimes or present) if r in present]
    present_models = data["model"].dropna().unique().tolist()
    models = [m for m in (models or model_order(present_models)) if m in present_models]

    fig = go.Figure()
    legend_seen: set[str] = set()
    for model in models:
        for precision in ("fp32", "fp16"):
            xs, ys, texts = [], [], []
            for runtime in runtimes:
                sel = data[(data["runtime"] == runtime)
                           & (data["model"] == model)
                           & (data["precision"] == precision)]
                if sel.empty:
                    continue
                value = float(sel[metric].mean())
                if value != value:
                    continue
                xs.append(runtime)
                ys.append(value)
                texts.append(f"{value:.1f}")

            if not xs:
                continue
            label = f"{model} {precision.upper()}"
            fig.add_trace(go.Bar(
                x=xs, y=ys, name=label,
                marker_color=_color_for(model, models.index(model)),
                marker_opacity=PRECISION_OPACITY[precision],
                marker_line=dict(width=1.0,
                                 color="#222" if precision == "fp16" else "rgba(0,0,0,0.2)"),
                text=texts if show_text else None,
                textposition="outside",
                textfont=dict(size=9),
                showlegend=label not in legend_seen,
                hovertemplate=f"<b>{label}</b><br>%{{x}}<br>{metric}: %{{y:.2f}}<extra></extra>",
            ))
            legend_seen.add(label)

    fig.update_layout(
        title=title or f"{metric} by runtime, grouped by model and precision",
        barmode="group", bargap=0.18, bargroupgap=0.06,
        height=height, xaxis_title=None,
        yaxis_title={"fps": "frames / second"}.get(metric, metric),
        yaxis=dict(rangemode="tozero", type="log" if log_y else "linear"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=10)),
        margin=dict(t=110, b=60, l=70, r=30),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    return fig


def speedup_vs_reference(
    df,
    *,
    metric: str = "fps",
    reference: str = "tensorrt",
    precision: str = "fp16",
    models=None,
    title: str | None = None,
    height: int = 480,
):
    """Relative speed of each runtime vs ``reference`` (1.0 = parity).

    A log/relative axis is the only way to compare a 170 fps CUDA backend and a
    7 fps CPU backend on one scale without the CPU bars vanishing.
    """
    import numpy as np
    import plotly.graph_objects as go

    data = df[(df["status"] == "ok") & (df["precision"] == precision)]
    present_models = data["model"].dropna().unique().tolist()
    models = [m for m in (models or model_order(present_models)) if m in present_models]

    runtimes = data["runtime"].dropna().unique().tolist()
    fig = go.Figure()
    for model in models:
        ratios, names = [], []
        base_row = data[(data["model"] == model) & (data["runtime"] == reference)]
        if base_row.empty or metric not in base_row:
            continue
        base = float(base_row[metric].mean())
        if base != base or base == 0:
            continue
        for runtime in runtimes:
            sel = data[(data["model"] == model) & (data["runtime"] == runtime)]
            if sel.empty:
                continue
            value = float(sel[metric].mean())
            if value != value or value == 0:
                continue
            ratios.append((value / base) if metric == "fps" else (base / value))
            names.append(runtime)
        if not ratios:
            continue
        fig.add_trace(go.Bar(
            x=names, y=ratios, name=model,
            marker_color=_color_for(model, models.index(model)),
            text=[f"{r:.2f}x" for r in ratios],
            textposition="outside", textfont=dict(size=10),
            hovertemplate=f"<b>{model}</b><br>%{{x}}<br>vs {reference}: %{{y:.2f}}x<extra></extra>",
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="#666",
                  annotation_text=f"{reference} = 1.0x", annotation_position="top left")
    fig.update_layout(
        title=title or f"{metric.upper()} relative to `{reference}` ({precision.upper()})",
        barmode="group", height=height,
        yaxis_title=f"x  (1.0 = {reference})",
        yaxis=dict(type="log" if metric == "fps" else "linear"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=110, b=60, l=70, r=30),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _runtime_label(kind: str) -> str:
    """Short axis label for a runtime, from the registry where possible."""
    try:
        from .runtimes import spec_for

        return spec_for(kind).label
    except Exception:
        return kind


def _runtime_color(kind: str, all_kinds) -> str:
    """Stable colour per runtime (from the registry order, so it never shifts)."""
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
    try:
        idx = list(all_kinds).index(kind)
    except ValueError:
        idx = 0
    return palette[idx % len(palette)]


def _metric_label(metric: str) -> str:
    return {
        "fps": "frames / second (higher is better)",
        "infer_ms": "raw forward latency, ms (lower is better)",
        "e2e_ms": "full-loop latency, ms (lower is better)",
        "e2e_p95_ms": "full-loop p95 latency, ms (lower is better)",
        "peak_vram_gb": "peak device memory, GB (lower is better)",
        "weights_mb": "artifact size on disk, MB (lower is better)",
        "host_rss_mb": "host RSS, MB (lower is better)",
    }.get(metric, metric)


# ---------------------------------------------------------------------------
# 2. Quality / cost trade-off
# ---------------------------------------------------------------------------

def quality_vs_speed(
    df,
    *,
    quality: str = "gflops",
    metric: str = "fps",
    precision: str = "fp16",
    runtimes=None,
    split: str = "all",
    color_by: str = "model",
    title: str | None = None,
    log_x: bool = True,
    log_y: bool = True,
    height: int = 560,
):
    """Scatter of a quality/cost axis against a performance axis.

    This is the chart that answers the trade-off question, and it is a scatter
    because both axes are *quantitative*. Bars force one axis to be categorical,
    which is exactly the wrong shape here.

    ``quality`` is deliberately generic — the caller decides what "quality" means:

    * ``gflops`` / ``params_m`` — model complexity (cost; lower-left is cheaper).
      The useful reading is a *Pareto frontier*: models on the upper-left edge are
      the best throughput per unit of compute.
    * ``recall`` / ``mean_iou`` from ``accuracy.py`` — actual quantized quality.
      Then the chart answers "what throughput did this quantization buy, and at
      what measured cost in detections?", which is the real FP16 question.

    ``split`` facets by size or family so the two size classes are never read as
    one curve.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = ok_rows(df)
    if precision:
        data = data[data["precision"] == precision]
    if quality not in data.columns:
        raise KeyError(
            f"quality column {quality!r} not in results. For model cost run "
            f"`export --meta-only` to record params/gflops; for quantized quality "
            f"merge in accuracy.py output (recall/mean_iou)."
        )
    data = data[data[quality].notna()]
    if data.empty:
        raise ValueError(f"no rows carry {quality!r}")
    runtimes = _resolve_runtimes(data, runtimes)
    groups = split_models(data["model"].dropna().unique(), split)

    fig = make_subplots(
        rows=1, cols=len(groups),
        column_titles=[name for name, _ in groups],
        shared_yaxes=True, horizontal_spacing=0.06,
    )

    for col, (_, models) in enumerate(groups, start=1):
        for runtime in runtimes:
            sub = data[(data["runtime"] == runtime) & (data["model"].isin(models))]
            if sub.empty:
                continue
            grouped = sub.groupby("model", as_index=False).agg(
                {quality: "mean", metric: "mean", "model_size": "first"}
            )
            grouped = grouped.sort_values(quality)
            fig.add_trace(go.Scatter(
                x=grouped[quality], y=grouped[metric],
                name=_runtime_label(runtime),
                mode="markers+lines+text",
                text=grouped["model"], textposition="top center",
                textfont=dict(size=9),
                line=dict(width=1.6, color=_runtime_color(runtime, runtimes)),
                marker=dict(
                    size=13 if color_by == "model" else 15,
                    symbol=RUNTIME_SYMBOLS.get(runtime, "circle"),
                    color=[_color_for(m, models.index(m)) for m in grouped["model"]]
                    if color_by == "model" else _runtime_color(runtime, runtimes),
                    line=dict(width=1, color="white"),
                ),
                showlegend=(col == 1),
                legendgroup=runtime,
                hovertemplate=(
                    "<b>%{text}</b><br>" + _runtime_label(runtime)
                    + f"<br>{quality}: %{{x:.2f}}"
                    + f"<br>{_metric_label(metric)}: %{{y:.2f}}<extra></extra>"
                ),
            ), row=1, col=col)
        fig.update_xaxes(type="log" if log_x else "linear", row=1, col=col,
                         title_text="GFLOPs" if quality == "gflops" else quality)

    fig.update_yaxes(title_text=_metric_label(metric), row=1, col=1,
                     type="log" if log_y else "linear")
    fig.update_layout(
        title=title or (f"{_metric_label(metric)} vs {quality.replace('_', ' ')}"
                        f" ({precision.upper() if precision else 'all'})"),
        height=height, width=max(700, 420 * len(groups)),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0, title=None),
        margin=dict(t=130, b=70, l=80, r=50),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Quantization advantage
# ---------------------------------------------------------------------------

def quantization_gain_heatmap(
    df,
    *,
    metric: str = "fps",
    runtimes=None,
    models=None,
    title: str | None = None,
    height: int = 460,
    annotate: bool = True,
    clamp: float = 2.0,
):
    """FP16 speedup as a diverging heatmap (``fp16 / fp32``, 1.0 = no change).

    A diverging scale centred on 1.0 with a white midpoint is the honest encoding:
    "no effect" is visually neutral, a gain and a *regression* are opposite
    colours. A bar chart of the same data makes a regression look like a small
    gain. Clamping at ``clamp`` keeps one large outlier from washing out the
    scale; the annotated number is always the true, unclamped value.
    """
    import numpy as np
    import plotly.graph_objects as go

    data = ok_rows(df)
    runtimes = _resolve_runtimes(data, runtimes)
    models = [m for m in (models or model_order(data["model"].unique()))
              if m in set(data["model"])]

    grid = np.full((len(runtimes), len(models)), np.nan)
    for i, runtime in enumerate(runtimes):
        for j, model in enumerate(models):
            sel = data[(data["runtime"] == runtime) & (data["model"] == model)]
            lo = sel[sel["precision"] == "fp32"][metric]
            hi = sel[sel["precision"] == "fp16"][metric]
            if lo.empty or hi.empty:
                continue
            base = float(lo.mean())
            val = float(hi.mean())
            if base == 0 or base != base or val != val:
                continue
            # fps higher-is-better -> ratio; latency lower-is-better -> inverse.
            ratio = val / base if metric in ("fps",) else base / val
            grid[i, j] = ratio

    text = [[("" if np.isnan(v) else f"{v:.2f}x") for v in row] for row in grid]
    plot_z = np.clip(grid, 1.0 / clamp, clamp)

    fig = go.Figure(go.Heatmap(
        z=plot_z,
        x=models, y=[_runtime_label(r) for r in runtimes],
        text=text if annotate else None,
        texttemplate="%{text}" if annotate else None,
        textfont=dict(size=11, color="#111"),
        # 1.0 = white/neutral, green = gain, red = regression.
        zmid=1.0,
        colorscale=[
            [0.0, "#b2182b"], [0.25, "#ef8a62"], [0.5, "#f7f7f7"],
            [0.75, "#67a9cf"], [1.0, "#2166ac"],
        ],
        zmin=1.0 / clamp, zmax=clamp,
        colorbar=dict(title=f"{metric} ratio<br>(fp16 / fp32)",
                      tickvals=[1 / clamp, 0.75, 1.0, 1.25, clamp],
                      ticktext=[f"≤{1/clamp:.2f}x", "0.75x", "1.00x",
                                "1.25x", f"≥{clamp:.1f}x"]),
        hoverongaps=False,
        hovertemplate=("<b>%{y}</b> · %{x}<br>fp16/fp32: %{text}"
                       "<extra></extra>"),
    ))
    fig.update_layout(
        title=title or (f"Quantization advantage — {metric} ratio "
                        f"(white ≈ 1.0x = FP16 bought nothing)"),
        height=height, width=max(560, 190 * len(models) + 260),
        xaxis_title=None, yaxis_title=None,
        margin=dict(t=100, b=60, l=170, r=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def quantization_tradeoff(
    agreement_df,
    *,
    speed_df=None,
    metric: str = "fps",
    x: str = "mean_iou",
    runtimes=None,
    title: str | None = None,
    height: float | None = None,
):
    """What FP16 cost in detections vs what it bought in speed.

    Two panels sharing a y axis of the quantization *cost*:

    * left  — ``x`` quality metric (``mean_iou`` or ``recall``) against FP32,
    * right — ``metric`` (speed) gained.

    Reading it: points high-and-right on the quality panel are "no measurable
    cost"; points far right on the speed panel are "big win". A runtime that
    appears at recall 1.0 *and* a big speedup is a free lunch; one at lower recall
    is where the accuracy/speed decision actually has to be made.

    ``agreement_df`` comes from ``accuracy.precision_agreement`` (or its CSV).
    ``speed_df`` is the benchmark results frame; when omitted only the cost panel
    is drawn.
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ag = agreement_df
    if not isinstance(ag, pd.DataFrame):
        ag = pd.DataFrame(ag)
    if "status" in ag.columns:
        ag = ag[ag["status"] == "ok"]
    runtimes = [r for r in (runtimes or list(ag["runtime"].unique()))
                if r in set(ag["runtime"])]

    has_speed = speed_df is not None
    cols = 2 if has_speed else 1
    fig = make_subplots(
        rows=1, cols=cols,
        column_titles=([f"quantization cost ({x})", f"speed gained ({metric})"]
                       if has_speed else [f"quantization cost ({x})"]),
        shared_yaxes=True, horizontal_spacing=0.08,
    )

    for i, runtime in enumerate(runtimes):
        sub = ag[ag["runtime"] == runtime].sort_values("model")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub["model"],
            name=_runtime_label(runtime), mode="markers+text",
            text=sub["model"], textposition="middle right", textfont=dict(size=9),
            marker=dict(size=15, color=_runtime_color(runtime, runtimes),
                        symbol=RUNTIME_SYMBOLS.get(runtime, "circle"),
                        line=dict(width=1, color="white")),
            showlegend=(i == 0), legendgroup=runtime,
            hovertemplate=(f"<b>%{{y}}</b><br>{_runtime_label(runtime)}"
                           f"<br>{x}: %{{x:.4f}}<extra></extra>"),
        ), row=1, col=1)

        if has_speed:
            sp = ok_rows(speed_df)
            sp = sp[(sp["runtime"] == runtime) & (sp["precision"] == "fp16")]
            agg = (sp.groupby("model", as_index=False)[metric].mean()
                   .sort_values("model"))
            if agg.empty:
                continue
            fig.add_trace(go.Scatter(
                x=agg[metric], y=agg["model"],
                name=_runtime_label(runtime), mode="markers+text",
                text=agg["model"], textposition="middle right",
                textfont=dict(size=9),
                marker=dict(size=15, color=_runtime_color(runtime, runtimes),
                            symbol=RUNTIME_SYMBOLS.get(runtime, "circle"),
                            line=dict(width=1, color="white")),
                showlegend=False, legendgroup=runtime,
                hovertemplate=(f"<b>%{{y}}</b><br>{_runtime_label(runtime)}"
                               f"<br>{metric}: %{{x:.1f}}<extra></extra>"),
            ), row=1, col=2)

    if x in ("recall", "mean_iou"):
        # A vertical reference at "no divergence" makes the cost instantly legible.
        ref = 1.0
        fig.add_vline(x=ref, line_dash="dash", line_color="#888",
                      annotation_text="reference (no divergence)",
                      annotation_position="top", row=1, col=1)
    fig.update_layout(
        title=title or "Quantization trade-off: detection cost vs speed gained",
        height=height or (320 + 44 * max(1, len(ag))),
        width=1000 if has_speed else 560,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0, title=None),
        margin=dict(t=120, b=60, l=150, r=50),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. GPU / machine comparison
# ---------------------------------------------------------------------------

def gpu_comparison(
    df,
    *,
    metric: str = "fps",
    precision: str = "fp16",
    runtimes=None,
    split: str = "size",
    facet: str = "gpu_name",
    log_y: bool = True,
    title: str | None = None,
    height: int = 620,
):
    """Same measurement across machines: one line per (machine, runtime).

    Requires a **merged** frame (concatenate the ``results/*.csv`` from each
    machine) — every row already carries ``gpu_name``/``gpu_arch``/``cpu``, which
    is what makes this possible at all. Lines rather than bars because the
    interesting output is the *gap between machines* at each model size, and a
    line pair makes both the gap and the trend readable at once.

    ⚠️ Rows from different TensorRT majors are not version-comparable; the legend
    and hover carry the runtime's own version so a mixed comparison is visible
    rather than implied.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = ok_rows(df)
    if precision:
        data = data[data["precision"] == precision]
    if facet not in data.columns:
        raise KeyError(f"facet column {facet!r} missing; merged CSVs carry gpu_name/cpu")
    machines = [m for m in data[facet].dropna().unique()]
    if not machines:
        raise ValueError(f"no values in {facet!r} — did you merge CSVs?")
    runtimes = _resolve_runtimes(data, runtimes)
    groups = split_models(data["model"].dropna().unique(), split)

    fig = make_subplots(
        rows=1, cols=len(groups),
        column_titles=[name for name, _ in groups],
        shared_yaxes=True, horizontal_spacing=0.06,
    )

    for col, (_, models) in enumerate(groups, start=1):
        for machine in machines:
            for runtime in runtimes:
                xs, ys, versions = [], [], []
                for model in models:
                    sel = data[(data["model"] == model)
                               & (data["runtime"] == runtime)
                               & (data[facet] == machine)]
                    if sel.empty:
                        continue
                    value = float(sel[metric].mean())
                    if value != value:
                        continue
                    xs.append(model)
                    ys.append(value)
                    versions.append(str(sel["torch"].iloc[0]) if "torch" in sel else "")
                if not xs:
                    continue
                # Truncate the machine name so the legend stays readable.
                short = str(machine).replace("NVIDIA GeForce ", "").replace(" (dGPU)", "")
                version = ""
                if "runtime_version" in data.columns:
                    rv = data[(data["runtime"] == runtime)
                              & (data[facet] == machine)]["runtime_version"]
                    if not rv.empty and rv.notna().any():
                        version = f" v{rv.dropna().iloc[0]}"
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    name=f"{short} · {runtime}{version}",
                    mode="lines+markers",
                    line=dict(width=2.2,
                              color=_runtime_color(runtime, runtimes),
                              dash=("solid" if machine == machines[0] else "dash")),
                    marker=dict(
                        size=10, symbol=RUNTIME_SYMBOLS.get(runtime, "circle"),
                        color=_runtime_color(runtime, runtimes),
                        line=dict(width=1, color="white"),
                    ),
                    showlegend=(col == 1),
                    legendgroup=f"{machine}-{runtime}",
                    hovertemplate=(f"<b>%{{x}}</b><br>{short} · "
                                   f"{_runtime_label(runtime)}{version}"
                                   f"<br>{_metric_label(metric)}: %{{y:.2f}}"
                                   "<extra></extra>"),
                ), row=1, col=col)
        fig.update_xaxes(categoryorder="array", categoryarray=models, row=1, col=col)

    fig.update_yaxes(title_text=_metric_label(metric), row=1, col=1,
                     type="log" if log_y else "linear")
    fig.update_layout(
        title=title or (f"{_metric_label(metric)} across machines, by model size"
                        f" ({precision.upper() if precision else 'all'})"),
        height=height, width=max(720, 430 * len(groups)),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0,
                    font=dict(size=10), title=None),
        margin=dict(t=150, b=60, l=80, r=50),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    return fig


def machine_header(df) -> str:
    """Markdown machine header for a results frame (matches compare's report)."""
    from .env import FINGERPRINT_FIELDS, human_summary

    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if ok.empty:
        return "_no successful rows_"
    row = ok.iloc[0]
    fp = {k: row.get(k) for k in FINGERPRINT_FIELDS}
    return "```\n" + human_summary(fp) + "\n```"


def write_html(fig, path) -> str:
    """Save a figure as a self-contained HTML file (for sharing a result)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return str(path)