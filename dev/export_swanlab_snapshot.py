#!/usr/bin/env python3
"""
Fetch latest successful SwanLab runs for nanochat-{train,sft,rl}, export metrics,
and render Plotly charts for README snapshots.

Usage:
    python dev/export_swanlab_snapshot.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import swanlab


PROJECTS = ("nanochat-train", "nanochat-sft", "nanochat-rl")
SUCCESS_STATES = {"FINISHED", "SUCCESS", "COMPLETED"}

METRIC_KEYS: dict[str, list[str]] = {
    "nanochat-train": [
        "step",
        "train/loss",
        "train/mfu",
        "train/tok_per_sec",
        "train/epoch",
        "val/bpb",
        "core_metric",
        "total_training_time",
        "total_training_flops",
    ],
    "nanochat-sft": [
        "step",
        "train/loss",
        "train/mfu",
        "train/tok_per_sec",
        "train/epoch",
        "val/bpb",
        "total_training_time",
        "total_training_flops",
    ],
    "nanochat-rl": [
        "step",
        "reward",
        "sequence_length",
        "lrm",
        "pass@1",
        "pass@2",
        "pass@4",
        "pass@8",
        "pass@16",
    ],
}


@dataclass
class RunSelection:
    workspace: str
    project: str
    run_id: str
    run_name: str
    state: str
    created_at: str | None
    finished_at: str | None


def _workspace_candidates() -> tuple[str, ...]:
    vals: list[str] = []
    primary = os.environ.get("SWANLAB_WORKSPACE", "").strip()
    if primary:
        vals.append(primary)
    extra = os.environ.get("SWANLAB_WORKSPACE_CANDIDATES", "").strip()
    if extra:
        for item in extra.split(","):
            item = item.strip()
            if item and item not in vals:
                vals.append(item)
    return tuple(vals)


def _to_jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return str(v)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _duration_hms(created_at: str | None, finished_at: str | None) -> str | None:
    a = _parse_iso(created_at)
    b = _parse_iso(finished_at)
    if not a or not b:
        return None
    secs = int((b - a).total_seconds())
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _metric_timestamp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).endswith("_timestamp"):
            return str(c)
    return None


def _metric_value_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).endswith("_timestamp"):
            continue
        return str(c)
    return None


def _fetch_metric_df(run, key: str) -> pd.DataFrame | None:
    try:
        df = run.metrics(keys=[key])
    except Exception as e:
        if "404" in str(e):
            return None
        raise
    if df is None or df.empty:
        return None
    df = df.copy()
    tcol = _metric_timestamp_col(df)
    if tcol:
        df["timestamp_ms"] = pd.to_numeric(df[tcol], errors="coerce")
        df["time_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    else:
        df["time_utc"] = pd.NaT
    return df


def _latest_successful_run(api: swanlab.Api, workspace: str, project_name: str) -> RunSelection | None:
    project = api.project(f"{workspace}/{project_name}")
    runs = list(project.runs())
    finished = [r for r in runs if str(getattr(r, "state", "")).upper() in SUCCESS_STATES]
    if not finished:
        return None

    def sort_key(r):
        return (str(getattr(r, "finished_at", "") or ""), str(getattr(r, "created_at", "") or ""), str(getattr(r, "id", "") or ""))

    target = sorted(finished, key=sort_key, reverse=True)[0]
    return RunSelection(
        workspace=workspace,
        project=project_name,
        run_id=str(getattr(target, "id", "")),
        run_name=str(getattr(target, "name", "")),
        state=str(getattr(target, "state", "")),
        created_at=_to_jsonable(getattr(target, "created_at", None)),
        finished_at=_to_jsonable(getattr(target, "finished_at", None)),
    )


def _select_runs(api: swanlab.Api) -> dict[str, RunSelection]:
    workspace_candidates = _workspace_candidates()
    if not workspace_candidates:
        raise SystemExit("Set SWANLAB_WORKSPACE (or SWANLAB_WORKSPACE_CANDIDATES) before running this script.")
    selected: dict[str, RunSelection] = {}
    for project_name in PROJECTS:
        for ws in workspace_candidates:
            try:
                sel = _latest_successful_run(api, ws, project_name)
            except Exception:
                continue
            if sel is not None:
                selected[project_name] = sel
                break
    return selected


def _ensure_dirs(repo_root: Path) -> tuple[Path, Path]:
    data_dir = repo_root / "docs" / "data" / "swanlab"
    asset_dir = repo_root / "docs" / "assets" / "swanlab"
    data_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, asset_dir


def _clear_old_metric_csvs(data_dir: Path) -> None:
    for p in data_dir.glob("*.csv"):
        p.unlink()


def _save_fig(fig: go.Figure, base_path: Path) -> dict[str, str]:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=70, r=30, t=90, b=50),
        font=dict(size=16),
        title=dict(font=dict(size=24)),
        showlegend=False,
    )
    fig.update_xaxes(
        tickformat="%m-%d %H:%M",
        tickangle=-25,
        tickfont=dict(size=12),
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        automargin=True,
    )
    fig.update_yaxes(
        tickfont=dict(size=12),
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        automargin=True,
    )
    for ann in fig.layout.annotations or []:
        ann.font = dict(size=16)
    out = {}
    png_path = base_path.with_suffix(".png")
    try:
        width = 1600
        height = int(fig.layout.height) if fig.layout.height else 1400
        fig.write_image(str(png_path), width=width, height=height, scale=2)
        repo_root = base_path.parents[3]
        out["png"] = str(png_path.relative_to(repo_root))
    except Exception as e:
        out["png_error"] = str(e)
    return out


def _add_line(fig, row, col, df: pd.DataFrame | None, name: str, color: str | None = None, mode: str = "lines"):
    if df is None or df.empty:
        return
    vcol = _metric_value_col(df)
    if vcol is None:
        return
    fig.add_trace(
        go.Scatter(
            x=df["time_utc"],
            y=df[vcol],
            mode=mode,
            name=name,
            line=dict(color=color, width=3) if color else dict(width=3),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _build_train_fig(run_meta: RunSelection, dfs: dict[str, pd.DataFrame | None]) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=("Train Loss", "Validation BPB", "CORE Metric", "Train MFU (%)"),
        vertical_spacing=0.08,
        shared_xaxes=True,
    )
    _add_line(fig, 1, 1, dfs.get("train/loss"), "train/loss", "#d62728")
    _add_line(fig, 2, 1, dfs.get("val/bpb"), "val/bpb", "#1f77b4", mode="lines+markers")
    core_df = dfs.get("core_metric")
    if core_df is not None and not core_df.empty:
        vcol = _metric_value_col(core_df)
        fig.add_trace(
            go.Scatter(
                x=core_df["time_utc"],
                y=core_df[vcol],
                mode="markers+lines",
                name="core_metric",
                line=dict(color="#2ca02c", dash="dot", width=3),
                marker=dict(size=10, symbol="diamond"),
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    _add_line(fig, 4, 1, dfs.get("train/mfu"), "train/mfu", "#9467bd")
    fig.update_layout(
        title="Pretraining Metrics Snapshot (SwanLab)",
        height=2000,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for r in (1, 2, 3):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    return fig


def _build_sft_fig(run_meta: RunSelection, dfs: dict[str, pd.DataFrame | None]) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Train Loss", "Validation BPB", "Train MFU (%)"),
        vertical_spacing=0.09,
        shared_xaxes=True,
    )
    _add_line(fig, 1, 1, dfs.get("train/loss"), "train/loss", "#d62728")
    _add_line(fig, 2, 1, dfs.get("val/bpb"), "val/bpb", "#1f77b4", mode="lines+markers")
    _add_line(fig, 3, 1, dfs.get("train/mfu"), "train/mfu", "#9467bd")
    fig.update_layout(
        title="SFT Metrics Snapshot (SwanLab)",
        height=1600,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for r in (1, 2):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    return fig


def _build_rl_fig(run_meta: RunSelection, dfs: dict[str, pd.DataFrame | None]) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Reward", "Sequence Length", "Pass@k (latest snapshot)"),
        vertical_spacing=0.1,
        shared_xaxes=False,
    )
    _add_line(fig, 1, 1, dfs.get("reward"), "reward", "#2ca02c")
    _add_line(fig, 2, 1, dfs.get("sequence_length"), "sequence_length", "#1f77b4")

    pass_vals = {}
    for k in ("pass@1", "pass@2", "pass@4", "pass@8", "pass@16"):
        df = dfs.get(k)
        if df is None or df.empty:
            continue
        vcol = _metric_value_col(df)
        if vcol is None:
            continue
        val = df[vcol].dropna()
        if len(val):
            pass_vals[k] = float(val.iloc[-1])
    if pass_vals:
        fig.add_trace(
            go.Bar(
                x=list(pass_vals.keys()),
                y=list(pass_vals.values()),
                name="pass@k",
                marker_color="#9467bd",
                text=[f"{v:.3f}" for v in pass_vals.values()],
                textposition="outside",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title="RL Metrics Snapshot (SwanLab)",
        height=1700,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for r in (1, 2):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    return fig


def _latest_scalar(metrics: dict[str, pd.DataFrame | None], key: str) -> Any:
    df = metrics.get(key)
    if df is None or df.empty:
        return None
    vcol = _metric_value_col(df)
    if vcol is None:
        return None
    series = pd.to_numeric(df[vcol], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1]) if pd.api.types.is_float_dtype(series) else series.iloc[-1]


def _build_summary_json(generated_at: str, metrics_by_project: dict[str, dict[str, pd.DataFrame | None]]) -> dict[str, Any]:
    projects = []
    for project in PROJECTS:
        metrics = metrics_by_project.get(project)
        if metrics is None:
            projects.append({"project": project, "found": False})
            continue
        metric_summary: dict[str, Any] = {}
        for key, df in metrics.items():
            if df is None or df.empty:
                continue
            vcol = _metric_value_col(df)
            if vcol is None:
                continue
            row = df.tail(1).iloc[0].to_dict()
            metric_summary[key] = {
                "rows": int(len(df)),
                "last_value": _to_jsonable(row.get(vcol)),
            }
        projects.append({"project": project, "found": True, "metrics": metric_summary})
    return {"generated_at": generated_at, "projects": projects}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir, asset_dir = _ensure_dirs(repo_root)
    _clear_old_metric_csvs(data_dir)
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    api = swanlab.Api()
    selected = _select_runs(api)
    if len(selected) < 1:
        raise SystemExit("No SwanLab runs found. Check SWANLAB_API_KEY / workspace / network.")

    metrics_by_project: dict[str, dict[str, pd.DataFrame | None]] = {}
    chart_outputs: dict[str, dict[str, str]] = {}

    for project, sel in selected.items():
        run = api.run(f"{sel.workspace}/{project}/{sel.run_id}")
        project_metrics: dict[str, pd.DataFrame | None] = {}
        for key in METRIC_KEYS.get(project, []):
            df = _fetch_metric_df(run, key)
            project_metrics[key] = df
            if df is not None:
                safe_key = key.replace("/", "__").replace("@", "_at_")
                csv_path = data_dir / f"{project}__{safe_key}.csv"
                df.to_csv(csv_path, index=False)
        metrics_by_project[project] = project_metrics

        if project == "nanochat-train":
            fig = _build_train_fig(sel, project_metrics)
            base_path = asset_dir / "nanochat-train-latest"
        elif project == "nanochat-sft":
            fig = _build_sft_fig(sel, project_metrics)
            base_path = asset_dir / "nanochat-sft-latest"
        else:
            fig = _build_rl_fig(sel, project_metrics)
            base_path = asset_dir / "nanochat-rl-latest"
        chart_outputs[project] = _save_fig(fig, base_path)

    summary = _build_summary_json(generated_at, metrics_by_project)
    summary["charts"] = chart_outputs
    with open(data_dir / "latest_runs_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
