"""
data_profiler.py

Field-level profiler for tabular data (CSV / Excel).

For every column, computes:
  - % populated
  - number of distinct values
  - inferred type: int, float, str, date
  - int/float: min, max, distribution (histogram bins)
  - date: min, max, distribution (by month or year, depending on range)
  - str: full value/frequency list if distinct count < 50, else top 50 by frequency

Usage:
    from data_profiler import profile_file, render_html

    profile = profile_file("mydata.csv")          # or .xlsx
    render_html(profile, "report.html")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------
# Type inference
# --------------------------------------------------------------------------

def _infer_column_type(series: pd.Series) -> str:
    """Infer one of: int, float, date, str -- based on the non-null values."""
    non_null = series.dropna()
    if non_null.empty:
        return "str"

    # Already a proper dtype?
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"

    # Try numeric coercion (handles numbers stored as text)
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().mean() > 0.98:  # allow a tiny fraction of junk
        # int if every coercible value has no fractional part
        as_int = numeric.dropna()
        if (as_int % 1 == 0).all():
            return "int"
        return "float"

    # Try date coercion
    try:
        dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        dates = pd.to_datetime(non_null, errors="coerce")
    if dates.notna().mean() > 0.98:
        return "date"

    return "str"


# --------------------------------------------------------------------------
# Per-type stat builders
# --------------------------------------------------------------------------

def _numeric_stats(series: pd.Series, as_int: bool, bins: int = 10) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "max": None, "distribution": []}

    vmin, vmax = float(values.min()), float(values.max())

    if vmin == vmax:
        distribution = [{"range": f"{vmin:g}", "count": int(len(values))}]
    else:
        cats = pd.cut(values, bins=bins, include_lowest=True)
        vc = cats.value_counts(sort=False)
        distribution = []
        for interval, count in vc.items():
            lo, hi = interval.left, interval.right
            if as_int:
                label = f"{math.floor(lo)}–{math.ceil(hi)}"
            else:
                label = f"{lo:.2f}–{hi:.2f}"
            distribution.append({"range": label, "count": int(count)})

    return {
        "min": int(vmin) if as_int else round(vmin, 4),
        "max": int(vmax) if as_int else round(vmax, 4),
        "distribution": distribution,
    }


def _date_stats(series: pd.Series) -> dict[str, Any]:
    dates = pd.to_datetime(series, errors="coerce", format="mixed")
    dates = dates.dropna()
    if dates.empty:
        return {"min": None, "max": None, "distribution": []}

    dmin, dmax = dates.min(), dates.max()
    span_days = (dmax - dmin).days

    if span_days > 730:
        buckets = dates.dt.to_period("Y").astype(str)
    elif span_days > 60:
        buckets = dates.dt.to_period("M").astype(str)
    else:
        buckets = dates.dt.to_period("D").astype(str)

    vc = buckets.value_counts().sort_index()
    distribution = [{"range": idx, "count": int(cnt)} for idx, cnt in vc.items()]

    return {
        "min": dmin.date().isoformat(),
        "max": dmax.date().isoformat(),
        "distribution": distribution,
    }


def _string_stats(series: pd.Series, distinct_count: int) -> dict[str, Any]:
    values = series.dropna().astype(str)
    vc = values.value_counts()
    if distinct_count < 50:
        top = vc
        truncated = False
    else:
        top = vc.head(50)
        truncated = True
    values_freq = [{"value": v, "count": int(c)} for v, c in top.items()]
    return {"values_freq": values_freq, "truncated": truncated}


# --------------------------------------------------------------------------
# Main entry points
# --------------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    total_rows = len(df)
    columns = []

    for col in df.columns:
        series = df[col]
        non_null = series.notna().sum()
        pct_populated = round((non_null / total_rows) * 100, 2) if total_rows else 0.0
        distinct_count = int(series.dropna().nunique())
        col_type = _infer_column_type(series)

        entry: dict[str, Any] = {
            "name": str(col),
            "type": col_type,
            "pct_populated": pct_populated,
            "distinct_count": distinct_count,
        }

        if col_type in ("int", "float"):
            entry.update(_numeric_stats(series, as_int=(col_type == "int")))
        elif col_type == "date":
            entry.update(_date_stats(series))
        else:  # str
            entry.update(_string_stats(series, distinct_count))

        columns.append(entry)

    return {
        "row_count": total_rows,
        "column_count": len(df.columns),
        "columns": columns,
    }


def profile_file(path: str | Path, sheet_name: str | int = 0) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return profile_dataframe(df)


def save_profile_json(profile: dict[str, Any], out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps(profile, indent=2, default=str))


# --------------------------------------------------------------------------
# HTML report rendering
# --------------------------------------------------------------------------

import html as _html
from datetime import datetime

_TYPE_META = {
    "int": ("numeric", "#1F6F5C"),
    "float": ("numeric", "#1F6F5C"),
    "date": ("date", "#33587A"),
    "str": ("categorical", "#8A5A2B"),
}


def _esc(x: Any) -> str:
    return _html.escape(str(x))


def _bars(distribution: list[dict[str, Any]], color: str) -> str:
    if not distribution:
        return '<p class="empty">No values to distribute.</p>'
    max_count = max(d["count"] for d in distribution) or 1
    rows = []
    for d in distribution:
        pct = round((d["count"] / max_count) * 100, 1)
        label = _esc(d.get("value", d.get("range", "")))
        rows.append(
            f'<div class="bar-row">'
            f'<div class="bar-label" title="{label}">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{pct}%;background:{color}"></div></div>'
            f'<div class="bar-count">{d["count"]:,}</div>'
            f"</div>"
        )
    return "".join(rows)


def _stat_line(label: str, value: Any) -> str:
    if value is None:
        return ""
    return f'<div class="stat"><span>{_esc(label)}</span><b>{_esc(value)}</b></div>'


def _field_block(col: dict[str, Any], index: int) -> str:
    kind, color = _TYPE_META.get(col["type"], ("categorical", "#8A5A2B"))
    anchor = f"field-{index}"

    stats = (
        _stat_line("Populated", f'{col["pct_populated"]}%')
        + _stat_line("Distinct", f'{col["distinct_count"]:,}')
        + _stat_line("Min", col.get("min"))
        + _stat_line("Max", col.get("max"))
    )

    if col["type"] == "str":
        distribution = col.get("values_freq", [])
        note = ""
        if col.get("truncated"):
            note = (
                f'<p class="note">Top 50 of {col["distinct_count"]:,} '
                f"distinct values shown.</p>"
            )
        dist_html = _bars(distribution, color) + note
    else:
        dist_html = _bars(col.get("distribution", []), color)

    zebra = " alt" if index % 2 else ""
    return f"""
    <section class="field{zebra}" id="{anchor}">
      <div class="field-meta">
        <div class="field-name">{_esc(col['name'])}</div>
        <div class="field-type"><span class="dot" style="background:{color}"></span>{_esc(col['type'])} &middot; {_esc(kind)}</div>
        <div class="stats">{stats}</div>
      </div>
      <div class="field-dist">{dist_html}</div>
    </section>"""


def render_html(
    profile: dict[str, Any], out_path: str | Path, source_name: str = "dataset"
) -> None:
    columns = profile["columns"]
    nav_links = "".join(
        f'<a href="#field-{i}">{_esc(c["name"])}</a>' for i, c in enumerate(columns)
    )
    field_sections = "".join(
        _field_block(c, i) for i, c in enumerate(columns)
    )
    generated = datetime.now().strftime("%d %b %Y, %H:%M")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Field Profile — {_esc(source_name)}</title>
<style>
  :root {{
    --bg: #FAFAF8;
    --bg-alt: #F1F1EE;
    --ink: #14171C;
    --ink-muted: #6B7178;
    --rule: #DEDEDA;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.45;
  }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 48px 24px 96px; }}
  header {{ border-bottom: 1px solid var(--rule); padding-bottom: 24px; margin-bottom: 32px; }}
  h1 {{ font-size: 28px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; }}
  .subhead {{ color: var(--ink-muted); font-size: 14px; }}
  .subhead b {{ color: var(--ink); font-weight: 600; }}
  nav {{
    columns: 3 220px;
    column-gap: 24px;
    border-bottom: 1px solid var(--rule);
    padding-bottom: 28px;
    margin-bottom: 8px;
    font-size: 13px;
  }}
  nav a {{
    display: block;
    color: var(--ink-muted);
    text-decoration: none;
    padding: 3px 0;
    break-inside: avoid;
    font-family: 'Consolas', 'SFMono-Regular', Menlo, monospace;
  }}
  nav a:hover {{ color: var(--ink); text-decoration: underline; }}
  .field {{
    display: grid;
    grid-template-columns: 260px 1fr;
    gap: 32px;
    padding: 28px 16px;
    border-bottom: 1px solid var(--rule);
    scroll-margin-top: 16px;
  }}
  .field.alt {{ background: var(--bg-alt); }}
  .field-name {{ font-size: 17px; font-weight: 600; margin-bottom: 4px; }}
  .field-type {{
    font-family: 'Consolas', 'SFMono-Regular', Menlo, monospace;
    font-size: 12px;
    color: var(--ink-muted);
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 14px;
  }}
  .dot {{ width: 8px; height: 8px; display: inline-block; }}
  .stats {{ font-size: 13px; }}
  .stat {{ display: flex; justify-content: space-between; padding: 2px 0; max-width: 200px; }}
  .stat span {{ color: var(--ink-muted); }}
  .stat b {{ font-family: 'Consolas', 'SFMono-Regular', Menlo, monospace; font-weight: 600; }}
  .bar-row {{ display: grid; grid-template-columns: 130px 1fr 60px; align-items: center; gap: 10px; padding: 3px 0; font-size: 12px; }}
  .bar-label {{ font-family: 'Consolas', 'SFMono-Regular', Menlo, monospace; color: var(--ink-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .bar-track {{ background: var(--rule); height: 10px; }}
  .bar-fill {{ height: 100%; }}
  .bar-count {{ font-family: 'Consolas', 'SFMono-Regular', Menlo, monospace; text-align: right; color: var(--ink-muted); }}
  .note, .empty {{ font-size: 12px; color: var(--ink-muted); margin: 8px 0 0; }}
  @media (max-width: 640px) {{
    .field {{ grid-template-columns: 1fr; }}
    nav {{ columns: 2 140px; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Field Profile</h1>
    <div class="subhead">
      <b>{_esc(source_name)}</b> &nbsp;·&nbsp;
      {profile['row_count']:,} rows &nbsp;·&nbsp;
      {profile['column_count']} columns &nbsp;·&nbsp;
      generated {generated}
    </div>
  </header>
  <nav>{nav_links}</nav>
  {field_sections}
</div>
</body>
</html>"""

    Path(out_path).write_text(html_doc, encoding="utf-8")
from data_profiler import profile_file, render_html

profile = profile_file("trades.xlsx")   # or .csv
render_html(profile, "report.html", source_name="trades.xlsx")





# NewsBasedClassification
Identification of news category based on headlines and short description
