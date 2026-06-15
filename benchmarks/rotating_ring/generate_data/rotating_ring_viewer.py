"""Compact Plotly HTML viewer for rotating-ring graph datasets."""

from __future__ import annotations

import json
from html import escape
from math import pi
from pathlib import Path
from typing import Any

import torch

try:
    from .rotating_ring_dataset import (
        CENTER_ROLE,
        INNER_ROLE,
        OUTER_ROLE,
        RING_GRAPH_CLASS_NAMES,
        RingGraphEnvironment,
        undirected_edge_pairs,
    )
except ImportError:
    from rotating_ring_dataset import (
        CENTER_ROLE,
        INNER_ROLE,
        OUTER_ROLE,
        RING_GRAPH_CLASS_NAMES,
        RingGraphEnvironment,
        undirected_edge_pairs,
    )


VIEWER_VERSION = "single-slider-smooth-outer3d-camera-min-distance-v1"
VIEWER_TEMPLATE_VERSION = VIEWER_VERSION
DEGREES = 180.0 / pi
ROLE_COLORS = {CENTER_ROLE: "#f47c20", INNER_ROLE: "#145f7a", OUTER_ROLE: "#44a9cc"}
ROLE_NAMES = {CENTER_ROLE: "center", INNER_ROLE: "inner", OUTER_ROLE: "outer"}


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_finite(value: Any) -> bool:
    value = _finite(value)
    return value == value


def _summary(values: list[float]) -> dict[str, float]:
    clean = [_finite(value) for value in values if _is_finite(value)]
    if not clean:
        nan = float("nan")
        return {"min": nan, "mean": nan, "max": nan}
    return {"min": min(clean), "mean": sum(clean) / len(clean), "max": max(clean)}


def _padded_range(values: list[float], *, pad_fraction: float, lower_bound: float | None = None) -> list[float]:
    clean = [_finite(value) for value in values if _is_finite(value)]
    if not clean:
        return [0.0, 1.0]
    low, high = min(clean), max(clean)
    pad = pad_fraction * (high - low) if high > low else max(1.0e-6, 0.05 * abs(high), 0.05)
    low -= pad
    high += pad
    if lower_bound is not None:
        low = max(lower_bound, low)
    return [low, high]


def _wrapped_angle_difference_degrees(angle_a: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
    diff = torch.atan2(torch.sin(angle_a - angle_b), torch.cos(angle_a - angle_b)).abs()
    return diff * DEGREES


def _closest_distance_angle_pairs(env: RingGraphEnvironment) -> tuple[list[float], list[float]]:
    """For each inner node, return nearest outer distance and xy-angle gap."""

    inner = env.R[env.node_role == INNER_ROLE]
    outer = env.R[env.node_role == OUTER_ROLE]
    if inner.numel() == 0 or outer.numel() == 0:
        return [], []

    distances = torch.cdist(inner, outer, p=2.0)
    closest, outer_index = distances.min(dim=1)
    inner_angles = torch.atan2(inner[:, 1], inner[:, 0])
    outer_angles = torch.atan2(outer[outer_index, 1], outer[outer_index, 0])
    gaps = _wrapped_angle_difference_degrees(inner_angles, outer_angles)
    return [float(x) for x in closest.tolist()], [float(x) for x in gaps.tolist()]


def _edge_coordinates(env: RingGraphEnvironment) -> tuple[list[float | None], list[float | None], list[float | None]]:
    coords: list[list[float | None]] = [[], [], []]
    for src, dst in undirected_edge_pairs(env.edge_index):
        for axis in range(3):
            coords[axis].extend([float(env.R[src, axis]), float(env.R[dst, axis]), None])
    return coords[0], coords[1], coords[2]


def _node_hover(env: RingGraphEnvironment) -> list[str]:
    hover = []
    for index, role in enumerate(env.node_role.tolist()):
        role_name = ROLE_NAMES.get(int(role), f"role {int(role)}")
        hover.append(
            "<br>".join(
                [
                    f"node {index}",
                    f"role={role_name}",
                    f"x={float(env.R[index, 0]):.3f}",
                    f"y={float(env.R[index, 1]):.3f}",
                    f"z={float(env.R[index, 2]):.3f}",
                ]
            )
        )
    return hover


def _sort_key(item: tuple[int, RingGraphEnvironment]) -> tuple[float, int]:
    original_index, env = item
    meta = dict(env.metadata)
    for key in ("class_index", "distance_split_rank", "variation_t", "generation_index"):
        if key in meta:
            value = _finite(meta[key])
            if value == value:
                return value, original_index
    return float(original_index), original_index


def _select_by_label(
    environments: list[RingGraphEnvironment],
    max_graphs: int | None,
) -> tuple[list[int], dict[int, list[tuple[int, RingGraphEnvironment]]]]:
    by_label: dict[int, list[tuple[int, RingGraphEnvironment]]] = {}
    for index, env in enumerate(environments):
        by_label.setdefault(int(env.label), []).append((index, env))

    labels = sorted(by_label)
    for label in labels:
        by_label[label].sort(key=_sort_key)

    if max_graphs is None:
        return labels, {label: list(by_label[label]) for label in labels}

    requested = max(1, int(max_graphs))
    per_label = max(1, requested // max(1, len(labels)))
    remainder = max(0, requested - per_label * len(labels))
    return labels, {
        label: by_label[label][: per_label + (1 if offset < remainder else 0)]
        for offset, label in enumerate(labels)
    }


def _class_name(label: int, meta: dict[str, Any]) -> str:
    if 0 <= label < len(RING_GRAPH_CLASS_NAMES):
        return str(meta.get("class_name", RING_GRAPH_CLASS_NAMES[label]))
    return str(meta.get("class_name", f"label {label}"))


def _split_cutoff(meta: dict[str, Any]) -> tuple[float, float, float]:
    low = _finite(meta.get("distance_split_threshold_low"))
    high = _finite(meta.get("distance_split_threshold_high"))
    cutoff = 0.5 * (low + high) if low == low and high == high else float("nan")
    return low, high, cutoff


def _record(
    *,
    label: int,
    class_position: int,
    class_total: int,
    original_index: int,
    env: RingGraphEnvironment,
) -> dict[str, Any]:
    meta = dict(env.metadata)
    fallback_t = 0.0 if class_total <= 1 else class_position / float(class_total - 1)
    variation_t = _finite(meta.get("variation_t"), fallback_t)
    if variation_t != variation_t:
        variation_t = fallback_t

    class_name = _class_name(label, meta)
    class_display = f"Class {label + 1}"
    distances, angle_gaps = _closest_distance_angle_pairs(env)
    distance_summary = _summary(distances)
    angle_summary = _summary(angle_gaps)
    threshold_low, threshold_high, cutoff = _split_cutoff(meta)

    inner_radius = _finite(meta.get("inner_radius"))
    outer_radius = _finite(meta.get("outer_radius"))
    outer_rotation = _finite(meta.get("outer_rotation_clockwise"))
    outer_phase = _finite(meta.get("outer_phase_clockwise"))
    outer_3d_rotation = _finite(meta.get("outer_3d_rotation"), 0.0)
    outer_3d_axis_angle = _finite(meta.get("outer_3d_axis_angle"), 0.0)
    global_rotation = _finite(meta.get("global_rotation"))
    edge_x, edge_y, edge_z = _edge_coordinates(env)

    title = (
        f"{class_display} ({class_name}) | variation {100.0 * variation_t:.1f}% | "
        f"graph {class_position + 1}/{class_total}<br>"
        f"r_inner={inner_radius:.3f}, r_outer={outer_radius:.3f}, "
        f"outer rotation={outer_rotation * DEGREES:.1f} deg, "
        f"outer phase={outer_phase * DEGREES:.1f} deg, "
        f"outer 3D tilt={outer_3d_rotation * DEGREES:.1f} deg, "
        f"global rotation={global_rotation * DEGREES:.1f} deg"
    )

    return {
        "label": int(label),
        "classDisplay": class_display,
        "className": class_name,
        "classGraphIndex": int(class_position),
        "classGraphCount": int(class_total),
        "originalDatasetIndex": int(original_index),
        "name": env.name,
        "variationT": float(variation_t),
        "title": title,
        "innerRadius": inner_radius,
        "outerRadius": outer_radius,
        "outerRotationDeg": outer_rotation * DEGREES,
        "outerPhaseDeg": outer_phase * DEGREES,
        "outer3DRotationDeg": outer_3d_rotation * DEGREES,
        "outer3DAxisDeg": outer_3d_axis_angle * DEGREES,
        "globalRotationDeg": global_rotation * DEGREES,
        "closestInnerOuterDistances": distances,
        "closestInnerOuterAngleDeg": angle_gaps,
        "closestInnerOuterDistanceMin": distance_summary["min"],
        "closestInnerOuterDistanceMean": distance_summary["mean"],
        "closestInnerOuterDistanceMax": distance_summary["max"],
        "distanceSplitThresholdLow": threshold_low,
        "distanceSplitThresholdHigh": threshold_high,
        "distanceSplitCutoff": cutoff,
        "closestInnerOuterAngleMinDeg": angle_summary["min"],
        "closestInnerOuterAngleMeanDeg": angle_summary["mean"],
        "closestInnerOuterAngleMaxDeg": angle_summary["max"],
        "nodeX": [float(x) for x in env.R[:, 0].tolist()],
        "nodeY": [float(y) for y in env.R[:, 1].tolist()],
        "nodeZ": [float(z) for z in env.R[:, 2].tolist()],
        "nodeColor": [ROLE_COLORS.get(int(role), "#999999") for role in env.node_role.tolist()],
        "nodeHover": _node_hover(env),
        "edgeX": edge_x,
        "edgeY": edge_y,
        "edgeZ": edge_z,
    }


def _payload(environments: list[RingGraphEnvironment], max_graphs: int | None) -> dict[str, Any]:
    labels, selected_by_label = _select_by_label(environments, max_graphs)
    selected_items = [item for label in labels for item in selected_by_label[label]]
    if not selected_items:
        raise ValueError("No graphs selected for the viewer.")

    records: list[dict[str, Any]] = []
    for label in labels:
        group = selected_by_label[label]
        for i, (original_index, env) in enumerate(group):
            records.append(
                _record(
                    label=label,
                    class_position=i,
                    class_total=len(group),
                    original_index=original_index,
                    env=env,
                )
            )

    max_radius = max(float(torch.linalg.vector_norm(env.R, dim=1).max().item()) for _, env in selected_items)
    axis_range = [-1.15 * max_radius, 1.15 * max_radius]
    min_values = [r["closestInnerOuterDistanceMin"] for r in records] + [r["distanceSplitCutoff"] for r in records]
    min_range = _padded_range(min_values, pad_fraction=0.10)

    return {
        "viewerVersion": VIEWER_VERSION,
        "graphs": records,
        "axisRange": axis_range,
        "zRange": list(axis_range),
        "distanceRange": _padded_range(
            [x for r in records for x in r["closestInnerOuterDistances"]],
            pad_fraction=0.05,
        ),
        "minDistanceRange": min_range,
        "distanceBinSize": max(1.0e-9, (min_range[1] - min_range[0]) / 80.0),
        "angleRangeDeg": _padded_range(
            [x for r in records for x in r["closestInnerOuterAngleDeg"]],
            pad_fraction=0.05,
            lower_bound=0.0,
        ),
        "viewerTemplateVersion": VIEWER_TEMPLATE_VERSION,
    }


def _plotly_loader(include_plotlyjs: bool | str) -> str:
    if include_plotlyjs is True:
        from plotly.offline import get_plotlyjs

        return f'<script type="text/javascript">{get_plotlyjs()}</script>'
    if include_plotlyjs == "cdn":
        return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    if include_plotlyjs:
        return f'<script src="{escape(str(include_plotlyjs))}"></script>'
    return ""


def _validate_generated_viewer_html(html: str) -> None:
    required = [
        VIEWER_VERSION,
        'id="graph-slider"',
        "allGraphs",
        "class-boundary",
        "outer3DRotationDeg",
        "closestInnerOuterDistances",
        'id="distance-histogram"',
        "updateDistanceHistogram",
        "closestInnerOuterAngleDeg",
        'id="distance-angle-scatter"',
        "updateDistanceAngleScatter",
        "getCurrentCamera",
        "rememberCurrentCamera",
        "distanceSplitCutoff",
        "selected graph minimum",
        "class cutoff",
    ]
    missing = [marker for marker in required if marker not in html]
    if missing:
        raise RuntimeError("Generated HTML is missing required viewer markers: " + ", ".join(missing))


def _render_html(payload_json: str, plotly_loader: str) -> str:
    return f"""<!doctype html>
<!-- rotating-ring-viewer-template: {VIEWER_VERSION} -->
<html lang="en">
<!-- viewer-version: {VIEWER_VERSION} -->
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rotating Ring Graph Viewer</title>
  <meta name="ring-viewer-version" content="{VIEWER_VERSION}">
  {plotly_loader}
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #fafafa;
      color: #202020;
    }}
    #controls {{
      padding: 14px 18px 8px 18px;
      border-bottom: 1px solid #d6d6d6;
      background: white;
      position: sticky;
      top: 0;
      z-index: 10;
      box-shadow: 0 1px 6px rgba(0, 0, 0, 0.06);
    }}
    .control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }}
    button {{
      border: 1px solid #bbbbbb;
      background: #f5f5f5;
      border-radius: 8px;
      padding: 7px 11px;
      cursor: pointer;
      font-size: 14px;
    }}
    button.active {{
      background: #202020;
      color: white;
      border-color: #202020;
    }}
    button:hover {{
      filter: brightness(0.96);
    }}
    #graph-slider {{
      flex: 1 1 420px;
      min-width: 240px;
    }}
    #status {{
      font-size: 14px;
      font-weight: 600;
    }}
    #class-boundary {{
      font-size: 14px;
    }}
    .current-class-boundary {{
      font-weight: 700;
    }}
    #details {{
      font-size: 13px;
      color: #555555;
    }}
    #viewer-body {{
      display: grid;
      grid-template-columns: minmax(520px, 2fr) minmax(360px, 1fr);
      gap: 12px;
      padding: 12px;
      box-sizing: border-box;
      height: calc(100vh - 118px);
      min-height: 560px;
    }}
    #plot {{
      width: 100%;
      height: 100%;
      min-height: 560px;
      background: white;
      border: 1px solid #e1e1e1;
      border-radius: 10px;
      overflow: hidden;
    }}
    #distance-panel {{
      background: white;
      border: 1px solid #e1e1e1;
      border-radius: 10px;
      padding: 12px;
      min-width: 0;
      overflow-y: auto;
      overflow-x: hidden;
    }}
    #distance-panel h2 {{
      margin: 0 0 6px 0;
      font-size: 16px;
    }}
    .scatter-heading {{
      margin-top: 18px !important;
      padding-top: 6px;
    }}
    #distance-summary {{
      font-size: 12px;
      color: #444444;
      line-height: 1.45;
      margin-bottom: 8px;
    }}
    #distance-histogram {{
      width: 100%;
      height: 260px;
      min-height: 240px;
    }}
    #distance-angle-scatter {{
      width: 100%;
      height: 300px;
      min-height: 280px;
      margin-top: 22px;
      border-top: 1px solid #eeeeee;
      padding-top: 10px;
    }}
    .distance-note {{
      font-size: 12px;
      color: #666666;
      line-height: 1.4;
      margin-top: 8px;
    }}
    @media (max-width: 1100px) {{
      #viewer-body {{
        grid-template-columns: 1fr;
        height: auto;
      }}
      #plot {{
        height: 560px;
      }}
      #distance-panel {{
        overflow: visible;
      }}
      #distance-histogram {{
        height: 260px;
      }}
      #distance-angle-scatter {{
        height: 320px;
      }}
    }}
    .hint {{
      font-size: 12px;
      color: #666666;
    }}
  </style>
</head>
<body>
  <div id="controls">
    <div class="control-row">
      <span id="class-boundary"></span>
      <button id="prev-button" type="button">Previous</button>
      <button id="next-button" type="button">Next</button>
      <button id="play-button" type="button">Play</button>
    </div>
    <div class="control-row">
      <input id="graph-slider" type="range" min="0" max="0" step="1" value="0">
      <span id="status"></span>
    </div>
    <div id="details"></div>
    <div class="hint">The single slider walks through all selected graphs: class 0 first, then class 1. When the slider crosses the class boundary, the status/header changes to show the active class. Rotating or zooming the 3D view is preserved while the slider changes. Viewer template: {VIEWER_VERSION}.</div>
  </div>
  <div id="viewer-body">
    <div id="plot"></div>
    <section id="distance-panel" aria-label="Closest inner-to-outer distance diagnostics">
      <h2>Minimum closest inner-to-outer distance</h2>
      <div id="distance-summary"></div>
      <div id="distance-histogram"></div>
      <h2 class="scatter-heading">Angle gap vs closest distance</h2>
      <div id="distance-angle-scatter"></div>
      <div class="distance-note">The histogram shows one value per graph: the minimum nearest outer-ring distance across inner-ring nodes. The dotted vertical line is the class cutoff. The dashed vertical line is the selected graph minimum. The scatter plot below is unchanged: it shows the raw nearest-node pairs as angle gap vs distance.</div>
    </section>
  </div>

  <script type="text/javascript">
    const payload = {payload_json};
    window.RING_VIEWER_VERSION = payload.viewerVersion;
    const graphsByClass = {{}};
    for (const graph of payload.graphs) {{
      const key = String(graph.label);
      if (!graphsByClass[key]) graphsByClass[key] = [];
      graphsByClass[key].push(graph);
    }}
    for (const key of Object.keys(graphsByClass)) {{
      graphsByClass[key].sort((a, b) => {{
        const aIndex = Number.isFinite(Number(a.classGraphIndex)) ? Number(a.classGraphIndex) : Number(a.variationT);
        const bIndex = Number.isFinite(Number(b.classGraphIndex)) ? Number(b.classGraphIndex) : Number(b.variationT);
        return aIndex - bIndex;
      }});
    }}

    const classLabels = Object.keys(graphsByClass).sort((a, b) => Number(a) - Number(b));
    const allGraphs = [];
    for (const label of classLabels) {{
      allGraphs.push(...graphsByClass[label]);
    }}
    let activeClass = allGraphs.length > 0 ? String(allGraphs[0].label) : (classLabels[0] || "0");
    let playTimer = null;
    let storedCamera = null;

    const plotDiv = document.getElementById("plot");
    const histogramDiv = document.getElementById("distance-histogram");
    const angleDistanceDiv = document.getElementById("distance-angle-scatter");
    const distanceSummary = document.getElementById("distance-summary");
    const slider = document.getElementById("graph-slider");
    const status = document.getElementById("status");
    const details = document.getElementById("details");
    const classBoundary = document.getElementById("class-boundary");
    const playButton = document.getElementById("play-button");

    const config = {{responsive: true, displaylogo: false}};
    const classPlotColors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"];
    function colorForClassIndex(index) {{
      return classPlotColors[index % classPlotColors.length];
    }}
    let isProgrammaticCameraUpdate = false;

    function cloneObject(value) {{
      if (value === null || value === undefined) return null;
      return JSON.parse(JSON.stringify(value));
    }}

    function getCurrentCamera() {{
      try {{
        if (plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.scene && plotDiv._fullLayout.scene.camera) {{
          return cloneObject(plotDiv._fullLayout.scene.camera);
        }}
      }} catch (error) {{
        // Ignore camera reads before Plotly has finished initializing.
      }}
      return storedCamera ? cloneObject(storedCamera) : null;
    }}

    function rememberCurrentCamera() {{
      const camera = getCurrentCamera();
      if (camera) storedCamera = camera;
      return storedCamera ? cloneObject(storedCamera) : null;
    }}

    function attachCameraListener() {{
      if (!plotDiv || plotDiv.__ringCameraListenerAttached || typeof plotDiv.on !== "function") return;
      plotDiv.__ringCameraListenerAttached = true;
      plotDiv.on("plotly_relayout", (eventData) => {{
        if (isProgrammaticCameraUpdate) return;

        if (eventData && eventData["scene.camera"]) {{
          storedCamera = cloneObject(eventData["scene.camera"]);
        }} else {{
          const camera = getCurrentCamera();
          if (camera) storedCamera = camera;
        }}
      }});
    }}

    function currentSliderFraction() {{
      const maxIndex = Math.max(allGraphs.length - 1, 1);
      return Number(slider.value) / maxIndex;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function updateClassBoundaryText(graph, index) {{
      const boundaryPieces = [];
      let offset = 0;
      for (const label of classLabels) {{
        const graphs = graphsByClass[label] || [];
        if (graphs.length === 0) continue;
        const first = graphs[0];
        const start = offset + 1;
        const end = offset + graphs.length;
        const labelText = `${{first.classDisplay}}: ${{first.className}} ${{start}}-${{end}}`;
        if (String(graph.label) === String(label)) {{
          boundaryPieces.push(`<span class="current-class-boundary">${{escapeHtml(labelText)}} current</span>`);
        }} else {{
          boundaryPieces.push(escapeHtml(labelText));
        }}
        offset += graphs.length;
      }}
      classBoundary.innerHTML = boundaryPieces.join(" | ");
    }}

    function finiteNumberArray(values) {{
      if (!Array.isArray(values)) return [];
      return values.map(Number).filter((value) => Number.isFinite(value));
    }}

    function finiteNumber(value) {{
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : NaN;
    }}

    function formatDistance(value) {{
      return Number.isFinite(value) ? value.toFixed(4) : "n/a";
    }}

    function summarizeDistances(values) {{
      const clean = finiteNumberArray(values);
      if (clean.length === 0) {{
        return {{count: 0, min: NaN, mean: NaN, max: NaN}};
      }}
      let minValue = clean[0];
      let maxValue = clean[0];
      let total = 0.0;
      for (const value of clean) {{
        if (value < minValue) minValue = value;
        if (value > maxValue) maxValue = value;
        total += value;
      }}
      return {{
        count: clean.length,
        min: minValue,
        mean: total / clean.length,
        max: maxValue
      }};
    }}

    function minDistancesForClass(label) {{
      const graphs = graphsByClass[String(label)] || [];
      const values = [];
      for (const graph of graphs) {{
        const value = finiteNumber(graph.closestInnerOuterDistanceMin);
        if (Number.isFinite(value)) values.push(value);
      }}
      return values;
    }}

    function cutoffForGraph(graph) {{
      const directCutoff = finiteNumber(graph.distanceSplitCutoff);
      if (Number.isFinite(directCutoff)) return directCutoff;

      const low = finiteNumber(graph.distanceSplitThresholdLow);
      const high = finiteNumber(graph.distanceSplitThresholdHigh);
      if (Number.isFinite(low) && Number.isFinite(high)) {{
        return 0.5 * (low + high);
      }}

      return NaN;
    }}

    function cutoffForAllGraphs() {{
      const cutoffs = [];
      for (const graph of allGraphs) {{
        const cutoff = cutoffForGraph(graph);
        if (Number.isFinite(cutoff)) cutoffs.push(cutoff);
      }}
      if (cutoffs.length === 0) return NaN;

      const first = cutoffs[0];
      for (const cutoff of cutoffs) {{
        if (Math.abs(cutoff - first) > 1.0e-9) {{
          // If multiple cutoffs somehow appear, use the mean. This keeps a
          // visible cutoff while avoiding arbitrary first-value behavior.
          let total = 0.0;
          for (const value of cutoffs) total += value;
          return total / cutoffs.length;
        }}
      }}
      return first;
    }}

    function updateDistanceHistogram(graph) {{
      const traces = [];
      const summaryLines = [];
      const classSummaries = [];
      const cutoff = cutoffForAllGraphs();

      for (const label of classLabels) {{
        const graphs = graphsByClass[label] || [];
        if (graphs.length === 0) continue;
        const first = graphs[0];
        const minDistances = minDistancesForClass(label);
        const classSummary = summarizeDistances(minDistances);
        classSummaries.push({{label: label, first: first, distances: minDistances, summary: classSummary}});
        summaryLines.push(
          `${{first.classDisplay}} (${{first.className}}): graphs=${{classSummary.count}}, min=${{formatDistance(classSummary.min)}}, mean=${{formatDistance(classSummary.mean)}}, max=${{formatDistance(classSummary.max)}}`
        );
      }}

      const shapes = [];
      const annotations = [];

      if (Number.isFinite(cutoff)) {{
        shapes.push({{
          type: "line",
          xref: "x",
          yref: "paper",
          x0: cutoff,
          x1: cutoff,
          y0: 0,
          y1: 1,
          line: {{width: 2, dash: "dot", color: "#666666"}}
        }});
        annotations.push({{
          x: cutoff,
          y: 1.08,
          xref: "x",
          yref: "paper",
          text: "class cutoff",
          showarrow: false,
          font: {{size: 11, color: "#666666"}}
        }});
      }}

      const selectedMinimum = finiteNumber(graph.closestInnerOuterDistanceMin);
      if (Number.isFinite(selectedMinimum)) {{
        shapes.push({{
          type: "line",
          xref: "x",
          yref: "paper",
          x0: selectedMinimum,
          x1: selectedMinimum,
          y0: 0,
          y1: 1,
          line: {{width: 3, dash: "dash", color: "#202020"}}
        }});
        annotations.push({{
          x: selectedMinimum,
          y: 1.02,
          xref: "x",
          yref: "paper",
          text: "selected graph minimum",
          showarrow: false,
          font: {{size: 11, color: "#202020"}}
        }});
      }}

      for (let classIndex = 0; classIndex < classSummaries.length; classIndex += 1) {{
        const item = classSummaries[classIndex];
        traces.push({{
          type: "histogram",
          x: item.distances,
          name: `${{item.first.classDisplay}}: ${{item.first.className}}`,
          histnorm: "probability density",
          opacity: 0.88,
          marker: {{color: colorForClassIndex(classIndex), line: {{color: "#202020", width: 1}}}},
          xbins: {{
            start: payload.minDistanceRange[0],
            end: payload.minDistanceRange[1],
            size: payload.distanceBinSize
          }},
          hovertemplate: "graph minimum distance=%{{x:.4f}}<br>density=%{{y:.4f}}<extra>%{{fullData.name}}</extra>"
        }});
      }}

      const selectedRawSummary = summarizeDistances(graph.closestInnerOuterDistances);
      const cutoffLine = Number.isFinite(cutoff)
        ? `<strong>Class cutoff</strong>: minimum distance = ${{formatDistance(cutoff)}}`
        : `<strong>Class cutoff</strong>: n/a`;
      const selectedLine = `<strong>Selected graph minimum</strong>: ${{formatDistance(selectedMinimum)}}`;
      const rawLine = `<strong>Selected raw nearest distances</strong>: n=${{selectedRawSummary.count}}, min=${{formatDistance(selectedRawSummary.min)}}, mean=${{formatDistance(selectedRawSummary.mean)}}, max=${{formatDistance(selectedRawSummary.max)}}`;

      distanceSummary.innerHTML = [
        selectedLine,
        cutoffLine,
        rawLine,
        ...summaryLines
      ].filter((line) => line.length > 0).join("<br>");

      const layout = {{
        xaxis: {{
          title: "minimum closest distance from any inner node to the outer ring",
          range: payload.minDistanceRange
        }},
        yaxis: {{title: "probability density", rangemode: "tozero"}},
        barmode: "overlay",
        bargap: 0.08,
        bargroupgap: 0.02,
        margin: {{l: 56, r: 16, t: 72, b: 56}},
        legend: {{orientation: "h", y: 1.18, x: 0.0}},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {{color: "#202020"}},
        shapes: shapes,
        annotations: annotations
      }};
      Plotly.react(histogramDiv, traces, layout, config);
    }}

    function updateDistanceAngleScatter(graph) {{
      const traces = [];
      for (let classIndex = 0; classIndex < classLabels.length; classIndex += 1) {{
        const label = classLabels[classIndex];
        const graphs = graphsByClass[label] || [];
        if (graphs.length === 0) continue;
        const first = graphs[0];
        const xAngles = [];
        const yDistances = [];
        const hoverText = [];
        for (const classGraph of graphs) {{
          const angles = finiteNumberArray(classGraph.closestInnerOuterAngleDeg);
          const distances = finiteNumberArray(classGraph.closestInnerOuterDistances);
          const n = Math.min(angles.length, distances.length);
          for (let i = 0; i < n; i += 1) {{
            xAngles.push(angles[i]);
            yDistances.push(distances[i]);
            hoverText.push(`${{classGraph.classDisplay}} (${{classGraph.className}})<br>graph=${{classGraph.name}}<br>angle gap=${{angles[i].toFixed(3)}} deg<br>closest distance=${{distances[i].toFixed(4)}}`);
          }}
        }}
        traces.push({{
          type: "scatter",
          mode: "markers",
          x: xAngles,
          y: yDistances,
          name: `${{first.classDisplay}}: ${{first.className}}`,
          marker: {{
            color: colorForClassIndex(classIndex),
            size: 10,
            opacity: 0.85,
            line: {{color: "#202020", width: 1}}
          }},
          text: hoverText,
          hoverinfo: "text"
        }});
      }}

      const selectedAngles = finiteNumberArray(graph.closestInnerOuterAngleDeg);
      const selectedDistances = finiteNumberArray(graph.closestInnerOuterDistances);
      const selectedN = Math.min(selectedAngles.length, selectedDistances.length);
      if (selectedN > 0) {{
        traces.push({{
          type: "scatter",
          mode: "markers",
          x: selectedAngles.slice(0, selectedN),
          y: selectedDistances.slice(0, selectedN),
          name: "selected graph",
          marker: {{
            color: "#000000",
            size: 15,
            symbol: "x",
            line: {{color: "#000000", width: 2}}
          }},
          hovertemplate: "selected graph<br>angle=%{{x:.3f}} deg<br>distance=%{{y:.4f}}<extra></extra>"
        }});
      }}

      const layout = {{
        title: {{text: "Angle gap vs closest distance", x: 0.02, xanchor: "left"}},
        xaxis: {{
          title: "nearest inner-to-outer angle gap (degrees)",
          range: payload.angleRangeDeg,
          gridcolor: "#e5e5e5",
          zerolinecolor: "#999999"
        }},
        yaxis: {{
          title: "closest distance",
          range: payload.distanceRange,
          gridcolor: "#e5e5e5",
          zerolinecolor: "#999999"
        }},
        margin: {{l: 56, r: 16, t: 52, b: 56}},
        legend: {{orientation: "h", y: 1.18, x: 0.0}},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {{color: "#202020"}}
      }};
      Plotly.react(angleDistanceDiv, traces, layout, config);
    }}

    function makeTraces(graph) {{
      return [
        {{
          type: "scatter3d",
          mode: "lines",
          x: graph.edgeX,
          y: graph.edgeY,
          z: graph.edgeZ,
          line: {{width: 5, color: "rgba(120,120,120,0.75)"}},
          hoverinfo: "skip",
          showlegend: false,
          name: "edges"
        }},
        {{
          type: "scatter3d",
          mode: "markers",
          x: graph.nodeX,
          y: graph.nodeY,
          z: graph.nodeZ,
          marker: {{
            size: 9,
            color: graph.nodeColor,
            line: {{width: 1.5, color: "#202020"}}
          }},
          hovertext: graph.nodeHover,
          hoverinfo: "text",
          showlegend: false,
          name: "nodes"
        }}
      ];
    }}

    function makeLayout(graph, camera) {{
      const layout = {{
        title: {{text: graph.title, x: 0.02, xanchor: "left"}},
        scene: {{
          xaxis: {{title: "x", range: payload.axisRange}},
          yaxis: {{title: "y", range: payload.axisRange}},
          zaxis: {{title: "z", range: payload.zRange}},
          aspectmode: "cube",
          uirevision: "keep-camera"
        }},
        uirevision: "keep-camera",
        margin: {{l: 0, r: 0, t: 78, b: 0}},
        showlegend: false
      }};
      if (camera) {{
        layout.scene.camera = cloneObject(camera);
      }}
      return layout;
    }}

    function updatePlot() {{
      if (allGraphs.length === 0) return;
      const maxIndex = Math.max(allGraphs.length - 1, 0);
      slider.max = String(maxIndex);
      let index = Number(slider.value);
      if (!Number.isFinite(index)) index = 0;
      index = Math.max(0, Math.min(maxIndex, Math.round(index)));
      slider.value = String(index);

      const graph = allGraphs[index];
      activeClass = String(graph.label);
      const camera = storedCamera ? cloneObject(storedCamera) : getCurrentCamera();
      Plotly.react(plotDiv, makeTraces(graph), makeLayout(graph, camera), config).then(() => {{
        attachCameraListener();
        if (camera) {{
          storedCamera = cloneObject(camera);
          isProgrammaticCameraUpdate = true;
          const relayoutPromise = Plotly.relayout(plotDiv, {{"scene.camera": cloneObject(camera)}});
          Promise.resolve(relayoutPromise).then(() => {{
            isProgrammaticCameraUpdate = false;
          }}, () => {{
            isProgrammaticCameraUpdate = false;
          }});
        }}
      }});
      status.textContent = `${{graph.classDisplay}}: ${{graph.className}} | graph ${{index + 1}}/${{allGraphs.length}} | class graph ${{graph.classGraphIndex + 1}}/${{graph.classGraphCount}} | variation ${{(100 * graph.variationT).toFixed(1)}}%`;
      details.textContent = `dataset index ${{graph.originalDatasetIndex}} | ${{graph.name}} | inner radius ${{graph.innerRadius.toFixed(3)}} | outer radius ${{graph.outerRadius.toFixed(3)}} | outer rotation ${{graph.outerRotationDeg.toFixed(1)}} deg | outer phase ${{graph.outerPhaseDeg.toFixed(1)}} deg | outer 3D tilt ${{graph.outer3DRotationDeg.toFixed(1)}} deg | tilt axis ${{graph.outer3DAxisDeg.toFixed(1)}} deg | closest inner-outer min ${{formatDistance(graph.closestInnerOuterDistanceMin)}}`;
      updateClassBoundaryText(graph, index);
      updateDistanceHistogram(graph);
      updateDistanceAngleScatter(graph);
    }}

    function stopPlay() {{
      if (playTimer !== null) {{
        window.clearInterval(playTimer);
        playTimer = null;
      }}
      playButton.textContent = "Play";
    }}

    function togglePlay() {{
      if (playTimer !== null) {{
        stopPlay();
        return;
      }}
      playButton.textContent = "Pause";
      playTimer = window.setInterval(() => {{
        const maxIndex = Math.max(allGraphs.length - 1, 0);
        const next = Number(slider.value) + 1;
        if (next > maxIndex) {{
          stopPlay();
          return;
        }}
        slider.value = String(next);
        updatePlot();
      }}, 130);
    }}

    slider.addEventListener("input", () => {{ stopPlay(); updatePlot(); }});
    document.getElementById("prev-button").addEventListener("click", () => {{
      stopPlay();
      slider.value = String(Math.max(0, Number(slider.value) - 1));
      updatePlot();
    }});
    document.getElementById("next-button").addEventListener("click", () => {{
      stopPlay();
      const maxIndex = Number(slider.max);
      slider.value = String(Math.min(maxIndex, Number(slider.value) + 1));
      updatePlot();
    }});
    playButton.addEventListener("click", togglePlay);

    window.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft") {{
        stopPlay();
        slider.value = String(Math.max(0, Number(slider.value) - 1));
        updatePlot();
      }} else if (event.key === "ArrowRight") {{
        stopPlay();
        const maxIndex = Number(slider.max);
        slider.value = String(Math.min(maxIndex, Number(slider.value) + 1));
        updatePlot();
      }}
    }});

    slider.max = String(Math.max(allGraphs.length - 1, 0));
    updatePlot();
  </script>
</body>
</html>
"""



def write_ring_graph_viewer(
    environments: list[RingGraphEnvironment],
    html_path: str | Path,
    *,
    max_graphs: int | None = 500,
    include_plotlyjs: bool | str = True,
) -> None:
    """Write the single-slider 3D graph viewer with distance diagnostics."""

    if not environments:
        raise ValueError("Cannot write a viewer for an empty environment list.")

    output_path = Path(html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _render_html(json.dumps(_payload(environments, max_graphs), allow_nan=True), _plotly_loader(include_plotlyjs))
    _validate_generated_viewer_html(html)
    output_path.write_text(html, encoding="utf-8")
