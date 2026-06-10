"""Plotly HTML viewer for rotating-ring graph datasets."""

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


VIEWER_TEMPLATE_VERSION = "class-filtered-smooth-outer3d-camera-v4"
VIEWER_VERSION = "class-filtered-smooth-slider-outer3d-camera-v4"


# -----------------------------------------------------------------------------
# Plotly HTML viewer
# -----------------------------------------------------------------------------


def _edge_trace_for_env(env: RingGraphEnvironment) -> Any:
    import plotly.graph_objects as go

    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for src, dst in undirected_edge_pairs(env.edge_index):
        x.extend([float(env.R[src, 0]), float(env.R[dst, 0]), None])
        y.extend([float(env.R[src, 1]), float(env.R[dst, 1]), None])
        z.extend([float(env.R[src, 2]), float(env.R[dst, 2]), None])

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"width": 5, "color": "rgba(120,120,120,0.75)"},
        hoverinfo="skip",
        showlegend=False,
        name="edges",
    )


def _node_trace_for_env(env: RingGraphEnvironment) -> Any:
    import plotly.graph_objects as go

    role_to_color = {
        CENTER_ROLE: "#f47c20",
        INNER_ROLE: "#145f7a",
        OUTER_ROLE: "#44a9cc",
    }
    role_to_name = {
        CENTER_ROLE: "center",
        INNER_ROLE: "inner",
        OUTER_ROLE: "outer",
    }
    colors = [role_to_color[int(role)] for role in env.node_role.tolist()]
    hover = [
        f"node {i}<br>role={role_to_name[int(role)]}<br>x={env.R[i,0]:.3f}<br>y={env.R[i,1]:.3f}<br>z={env.R[i,2]:.3f}"
        for i, role in enumerate(env.node_role.tolist())
    ]

    return go.Scatter3d(
        x=env.R[:, 0].tolist(),
        y=env.R[:, 1].tolist(),
        z=env.R[:, 2].tolist(),
        mode="markers",
        marker={
            "size": 9,
            "color": colors,
            "line": {"width": 1.5, "color": "#202020"},
        },
        text=[str(i) for i in range(env.n_nodes)],
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
        name="nodes",
    )


def _viewer_title(env: RingGraphEnvironment, index: int, total: int) -> str:
    meta = dict(env.metadata)
    deg = 180.0 / pi
    return (
        f"Graph {index + 1}/{total}: {env.name} | "
        f"label={env.label} ({meta.get('class_name', 'unknown')}) | "
        f"r_inner={meta.get('inner_radius', float('nan')):.3f}, "
        f"r_outer={meta.get('outer_radius', float('nan')):.3f}, "
        f"outer phase={meta.get('outer_phase_clockwise', float('nan')) * deg:.1f} deg, "
        f"outer 3D tilt={meta.get('outer_3d_rotation', 0.0) * deg:.1f} deg"
    )


def _validate_generated_viewer_html(html: str) -> None:
    """Fail fast if this file ever emits the older single-slider viewer."""

    required_markers = [
        VIEWER_VERSION,
        'id="class-buttons"',
        'graphsByClass',
        'setActiveClass',
        'The slider is ordered by the smooth variation coordinate inside the selected class',
        'outer3DRotationDeg',
        'getCurrentCamera',
        'rememberCurrentCamera',
        'scene.camera',
        'uirevision',
    ]
    missing = [marker for marker in required_markers if marker not in html]
    if missing:
        raise RuntimeError(
            "Generated HTML is missing the updated class-filtered viewer markers: "
            + ", ".join(missing)
        )


def write_ring_graph_viewer(
    environments: list[RingGraphEnvironment],
    html_path: str | Path,
    *,
    max_graphs: int | None = 500,
    include_plotlyjs: bool | str = True,
) -> None:
    """Write an interactive 3D Plotly viewer with class filtering.

    The viewer keeps the classes separate. The class buttons switch between
    Class 1 (label 0) and Class 2 (label 1), and the slider then walks only
    through the selected class. Within each class, graphs are sorted by
    `metadata["variation_t"]` when available, so neighboring slider positions
    correspond to neighboring points along the same smooth parameter path.
    """

    if not environments:
        raise ValueError("Cannot write a viewer for an empty environment list.")

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    def sort_value(item: tuple[int, RingGraphEnvironment]) -> tuple[float, int]:
        original_index, env = item
        meta = dict(env.metadata)
        for key in ("variation_t", "class_index", "generation_index"):
            if key in meta:
                try:
                    return float(meta[key]), int(original_index)
                except (TypeError, ValueError):
                    pass
        return float(original_index), int(original_index)

    items_by_label: dict[int, list[tuple[int, RingGraphEnvironment]]] = {}
    for original_index, env in enumerate(environments):
        items_by_label.setdefault(int(env.label), []).append((original_index, env))

    labels = sorted(items_by_label)
    for label in labels:
        items_by_label[label].sort(key=sort_value)

    # Keep the viewer balanced across classes when max_graphs is smaller than
    # the full dataset. This preserves the class switch even for small previews.
    selected_by_label: dict[int, list[tuple[int, RingGraphEnvironment]]] = {}
    if max_graphs is None:
        selected_by_label = {label: list(items_by_label[label]) for label in labels}
    else:
        requested = max(1, int(max_graphs))
        per_label = max(1, requested // max(1, len(labels)))
        remainder = max(0, requested - per_label * len(labels))
        for label_i, label in enumerate(labels):
            limit = per_label + (1 if label_i < remainder else 0)
            selected_by_label[label] = list(items_by_label[label][:limit])

    selected_items = [item for label in labels for item in selected_by_label[label]]
    if not selected_items:
        raise ValueError("No graphs selected for the viewer.")

    max_radius = max(
        float(torch.linalg.vector_norm(env.R, dim=1).max().item())
        for _, env in selected_items
    )
    axis_range = [-1.15 * max_radius, 1.15 * max_radius]
    # Use the same range for z so tilted outer-ring nodes are always visible
    # and the sphere interpretation is not visually compressed.
    z_range = list(axis_range)

    role_to_color = {
        CENTER_ROLE: "#f47c20",
        INNER_ROLE: "#145f7a",
        OUTER_ROLE: "#44a9cc",
    }
    role_to_name = {
        CENTER_ROLE: "center",
        INNER_ROLE: "inner",
        OUTER_ROLE: "outer",
    }

    records: list[dict[str, Any]] = []
    deg = 180.0 / pi
    for label in labels:
        group = selected_by_label[label]
        class_total = len(group)
        if class_total == 0:
            continue

        for class_position, (original_index, env) in enumerate(group):
            meta = dict(env.metadata)
            fallback_t = 0.0 if class_total <= 1 else class_position / float(class_total - 1)
            try:
                variation_t = float(meta.get("variation_t", fallback_t))
            except (TypeError, ValueError):
                variation_t = fallback_t

            if 0 <= label < len(RING_GRAPH_CLASS_NAMES):
                class_name = str(meta.get("class_name", RING_GRAPH_CLASS_NAMES[label]))
            else:
                class_name = str(meta.get("class_name", f"label {label}"))
            class_display = f"Class {label + 1}"

            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            edge_z: list[float | None] = []
            for src, dst in undirected_edge_pairs(env.edge_index):
                edge_x.extend([float(env.R[src, 0]), float(env.R[dst, 0]), None])
                edge_y.extend([float(env.R[src, 1]), float(env.R[dst, 1]), None])
                edge_z.extend([float(env.R[src, 2]), float(env.R[dst, 2]), None])

            node_x = [float(x) for x in env.R[:, 0].tolist()]
            node_y = [float(y) for y in env.R[:, 1].tolist()]
            node_z = [float(z) for z in env.R[:, 2].tolist()]
            node_colors = [role_to_color.get(int(role), "#999999") for role in env.node_role.tolist()]
            node_hover = []
            for node_i, role in enumerate(env.node_role.tolist()):
                role_name = role_to_name.get(int(role), f"role {int(role)}")
                node_hover.append(
                    "<br>".join(
                        [
                            f"node {node_i}",
                            f"role={role_name}",
                            f"x={float(env.R[node_i, 0]):.3f}",
                            f"y={float(env.R[node_i, 1]):.3f}",
                            f"z={float(env.R[node_i, 2]):.3f}",
                        ]
                    )
                )

            inner_radius = float(meta.get("inner_radius", float("nan")))
            outer_radius = float(meta.get("outer_radius", float("nan")))
            outer_phase = float(meta.get("outer_phase_clockwise", float("nan")))
            outer_rotation = float(meta.get("outer_rotation_clockwise", float("nan")))
            outer_3d_rotation = float(meta.get("outer_3d_rotation", 0.0))
            outer_3d_axis_angle = float(meta.get("outer_3d_axis_angle", 0.0))
            global_rotation = float(meta.get("global_rotation", float("nan")))
            title = (
                f"{class_display} ({class_name}) | variation {100.0 * variation_t:.1f}% | "
                f"graph {class_position + 1}/{class_total}<br>"
                f"r_inner={inner_radius:.3f}, r_outer={outer_radius:.3f}, "
                f"outer rotation={outer_rotation * deg:.1f} deg, "
                f"outer phase={outer_phase * deg:.1f} deg, "
                f"outer 3D tilt={outer_3d_rotation * deg:.1f} deg, "
                f"global rotation={global_rotation * deg:.1f} deg"
            )

            records.append(
                {
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
                    "outerRotationDeg": outer_rotation * deg,
                    "outerPhaseDeg": outer_phase * deg,
                    "outer3DRotationDeg": outer_3d_rotation * deg,
                    "outer3DAxisDeg": outer_3d_axis_angle * deg,
                    "globalRotationDeg": global_rotation * deg,
                    "nodeX": node_x,
                    "nodeY": node_y,
                    "nodeZ": node_z,
                    "nodeColor": node_colors,
                    "nodeHover": node_hover,
                    "edgeX": edge_x,
                    "edgeY": edge_y,
                    "edgeZ": edge_z,
                }
            )

    payload = {
        "viewerVersion": VIEWER_VERSION,
        "graphs": records,
        "axisRange": axis_range,
        "zRange": z_range,
        "viewerTemplateVersion": VIEWER_TEMPLATE_VERSION,
    }
    payload_json = json.dumps(payload, allow_nan=True)

    if include_plotlyjs is True:
        from plotly.offline import get_plotlyjs

        plotly_loader = f'<script type="text/javascript">{get_plotlyjs()}</script>'
    elif include_plotlyjs == "cdn":
        plotly_loader = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    elif include_plotlyjs:
        plotly_loader = f'<script src="{escape(str(include_plotlyjs))}"></script>'
    else:
        plotly_loader = ""

    html = f"""<!doctype html>
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
    #details {{
      font-size: 13px;
      color: #555555;
    }}
    #plot {{
      width: 100vw;
      height: calc(100vh - 118px);
      min-height: 560px;
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
      <span id="class-buttons"></span>
      <button id="prev-button" type="button">Previous</button>
      <button id="next-button" type="button">Next</button>
      <button id="play-button" type="button">Play</button>
    </div>
    <div class="control-row">
      <input id="graph-slider" type="range" min="0" max="0" step="1" value="0">
      <span id="status"></span>
    </div>
    <div id="details"></div>
    <div class="hint">The slider is ordered by the smooth variation coordinate inside the selected class, so adjacent positions should show small geometry changes. Rotating or zooming the 3D view is preserved while the slider/class changes. Set outer 3D rotation to 0 for the original planar dataset, or a positive value to tilt only the outer ring. Viewer template: {VIEWER_VERSION}.</div>
  </div>
  <div id="plot"></div>

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
      graphsByClass[key].sort((a, b) => a.variationT - b.variationT);
    }}

    const classLabels = Object.keys(graphsByClass).sort((a, b) => Number(a) - Number(b));
    let activeClass = classLabels[0];
    let playTimer = null;
    let storedCamera = null;

    const plotDiv = document.getElementById("plot");
    const slider = document.getElementById("graph-slider");
    const status = document.getElementById("status");
    const details = document.getElementById("details");
    const classButtons = document.getElementById("class-buttons");
    const playButton = document.getElementById("play-button");

    const config = {{responsive: true, displaylogo: false}};

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

    function currentClassGraphs() {{
      return graphsByClass[activeClass] || [];
    }}

    function currentSliderFraction() {{
      const graphs = currentClassGraphs();
      const maxIndex = Math.max(graphs.length - 1, 1);
      return Number(slider.value) / maxIndex;
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
        // Keep user-driven scene state, especially the 3D camera, across Plotly.react calls.
        uirevision: "keep-camera",
        margin: {{l: 0, r: 0, t: 78, b: 0}},
        showlegend: false
      }};
      if (camera) {{
        layout.scene.camera = cloneObject(camera);
      }}
      return layout;
    }}

    function refreshClassButtons() {{
      classButtons.innerHTML = "";
      for (const label of classLabels) {{
        const graphs = graphsByClass[label];
        if (!graphs || graphs.length === 0) continue;
        const first = graphs[0];
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = `${{first.classDisplay}}: ${{first.className}} (${{graphs.length}})`;
        button.className = label === activeClass ? "active" : "";
        button.addEventListener("click", () => setActiveClass(label));
        classButtons.appendChild(button);
      }}
    }}

    function setActiveClass(label) {{
      const fraction = currentSliderFraction();
      activeClass = String(label);
      const graphs = currentClassGraphs();
      const maxIndex = Math.max(graphs.length - 1, 0);
      slider.max = String(maxIndex);
      slider.value = String(Math.round(fraction * maxIndex));
      updatePlot();
    }}

    function updatePlot() {{
      const graphs = currentClassGraphs();
      if (graphs.length === 0) return;
      const maxIndex = Math.max(graphs.length - 1, 0);
      slider.max = String(maxIndex);
      let index = Number(slider.value);
      if (!Number.isFinite(index)) index = 0;
      index = Math.max(0, Math.min(maxIndex, Math.round(index)));
      slider.value = String(index);

      const graph = graphs[index];
      const camera = rememberCurrentCamera();
      Plotly.react(plotDiv, makeTraces(graph), makeLayout(graph, camera), config).then(() => {{
        if (storedCamera) {{
          Plotly.relayout(plotDiv, {{"scene.camera": cloneObject(storedCamera)}});
        }}
      }});
      status.textContent = `${{graph.classDisplay}} graph ${{index + 1}}/${{graphs.length}} | variation ${{(100 * graph.variationT).toFixed(1)}}%`;
      details.textContent = `dataset index ${{graph.originalDatasetIndex}} | ${{graph.name}} | inner radius ${{graph.innerRadius.toFixed(3)}} | outer radius ${{graph.outerRadius.toFixed(3)}} | outer rotation ${{graph.outerRotationDeg.toFixed(1)}} deg | outer phase ${{graph.outerPhaseDeg.toFixed(1)}} deg | outer 3D tilt ${{graph.outer3DRotationDeg.toFixed(1)}} deg | tilt axis ${{graph.outer3DAxisDeg.toFixed(1)}} deg`;
      refreshClassButtons();
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
        const graphs = currentClassGraphs();
        const maxIndex = Math.max(graphs.length - 1, 0);
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

    refreshClassButtons();
    updatePlot();
  </script>
</body>
</html>
"""

    _validate_generated_viewer_html(html)
    html_path.write_text(html, encoding="utf-8")

