"""Create rotating camera videos for the two rotating-ring graph classes.

Place this file next to rotating_ring_dataset.py, then run it from that folder or
as a module. It generates MP4 videos that automatically move through the graphs
inside each class while the camera orbits in azimuth, elevation, and roll.

Examples
--------
Create the default combined video plus one video per class:

    python generate_data/rotating_ring_video.py \
        --n-graphs 500 \
        --outer-3d-rotation-deg 70 \
        --global-rotation-frac-max 8

Create a quick preview with fewer graphs per class:

    python generate_data/rotating_ring_video.py \
        --n-graphs 500 \
        --max-graphs-per-class 30 \
        --frames-per-graph 30

The script uses matplotlib for rendering. By default it writes MP4 when ffmpeg
is available and falls back to GIF otherwise. Use --format mp4 to require MP4,
or --format gif to force the no-ffmpeg path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import pi, sin
from pathlib import Path
from typing import Iterable, Sequence

import torch

try:
    from .rotating_ring_dataset import (
        CENTER_ROLE,
        INNER_ROLE,
        OUTER_ROLE,
        RING_GRAPH_CLASS_NAMES,
        RingGraphEnvironment,
        create_rotating_ring_dataset,
        undirected_edge_pairs,
    )
except ImportError:
    from rotating_ring_dataset import (
        CENTER_ROLE,
        INNER_ROLE,
        OUTER_ROLE,
        RING_GRAPH_CLASS_NAMES,
        RingGraphEnvironment,
        create_rotating_ring_dataset,
        undirected_edge_pairs,
    )


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "original_visualizations" / "videos"

ROLE_COLORS = {
    CENTER_ROLE: "#f47c20",
    INNER_ROLE: "#145f7a",
    OUTER_ROLE: "#44a9cc",
}
ROLE_NAMES = {
    CENTER_ROLE: "center",
    INNER_ROLE: "inner",
    OUTER_ROLE: "outer",
}
EDGE_COLOR = "#6f6f6f"
NODE_EDGE_COLOR = "#202020"


@dataclass(frozen=True)
class RenderSelection:
    """One contiguous section of a video."""

    label: int
    graphs: list[RingGraphEnvironment]


@dataclass(frozen=True)
class CameraPath:
    """Smooth camera path that moves in several viewing dimensions."""

    azimuth_start_deg: float = 35.0
    azimuth_turns: float = 0.5
    elevation_mid_deg: float = 28.0
    elevation_amplitude_deg: float = 12.0
    elevation_cycles: float = 0.5
    roll_amplitude_deg: float = 4.0
    roll_cycles: float = 0.5

    def angles_for_fraction(self, t: float) -> tuple[float, float, float]:
        """Return (elevation, azimuth, roll) in degrees for a 0..1 fraction."""

        t = max(0.0, min(1.0, float(t)))
        azimuth = self.azimuth_start_deg + 360.0 * self.azimuth_turns * t
        elevation = self.elevation_mid_deg + self.elevation_amplitude_deg * sin(
            2.0 * pi * self.elevation_cycles * t
        )
        roll = self.roll_amplitude_deg * sin(2.0 * pi * self.roll_cycles * t)
        return elevation, azimuth, roll


def _sort_graphs_for_video(graphs: Iterable[RingGraphEnvironment]) -> list[RingGraphEnvironment]:
    """Sort graphs by the same smooth per-class order used by the HTML slider."""

    def sort_key(env: RingGraphEnvironment) -> tuple[int, float, int]:
        meta = dict(env.metadata)
        label = int(env.label)
        for key in ("variation_t", "class_index", "generation_index"):
            if key in meta:
                try:
                    return label, float(meta[key]), int(meta.get("generation_index", 0))
                except (TypeError, ValueError):
                    pass
        return label, 0.0, 0

    return sorted(graphs, key=sort_key)


def _group_by_class(environments: Sequence[RingGraphEnvironment]) -> list[RenderSelection]:
    """Return class groups in label order."""

    grouped: dict[int, list[RingGraphEnvironment]] = {}
    for env in environments:
        grouped.setdefault(int(env.label), []).append(env)

    selections: list[RenderSelection] = []
    for label in sorted(grouped):
        selections.append(RenderSelection(label=label, graphs=_sort_graphs_for_video(grouped[label])))
    return selections


def _evenly_limit_graphs(
    graphs: Sequence[RingGraphEnvironment],
    max_graphs: int | None,
) -> list[RingGraphEnvironment]:
    """Keep all graphs, or an evenly spaced preview across the full class range."""

    if max_graphs is None or max_graphs <= 0 or len(graphs) <= max_graphs:
        return list(graphs)
    if max_graphs == 1:
        return [graphs[0]]

    indices = torch.linspace(0, len(graphs) - 1, steps=max_graphs).round().long().tolist()
    unique_indices: list[int] = []
    seen: set[int] = set()
    for index in indices:
        index = int(index)
        if index not in seen:
            unique_indices.append(index)
            seen.add(index)
    return [graphs[index] for index in unique_indices]


def _class_name_for_label(label: int, env: RingGraphEnvironment | None = None) -> str:
    """Return a stable class name for labels 0 and 1."""

    if env is not None:
        meta_name = dict(env.metadata).get("class_name")
        if meta_name:
            return str(meta_name)
    if 0 <= int(label) < len(RING_GRAPH_CLASS_NAMES):
        return str(RING_GRAPH_CLASS_NAMES[int(label)])
    return f"label {int(label)}"


def _axis_limit_for_graphs(selections: Sequence[RenderSelection]) -> float:
    """Compute a shared cubic axis limit for every graph in a video."""

    max_radius = 1.0
    for selection in selections:
        for env in selection.graphs:
            if env.R.numel() == 0:
                continue
            max_radius = max(max_radius, float(torch.linalg.vector_norm(env.R, dim=1).max().item()))
    return 1.18 * max_radius


def _set_camera(ax: object, *, elevation: float, azimuth: float, roll: float) -> None:
    """Set the 3D camera and gracefully handle older matplotlib versions."""

    try:
        ax.view_init(elev=elevation, azim=azimuth, roll=roll)
    except TypeError:
        ax.view_init(elev=elevation, azim=azimuth)


def _style_3d_axes(ax: object) -> None:
    """Style the 3D axes to resemble the HTML/Plotly viewer."""

    ax.set_xlabel("x", labelpad=6)
    ax.set_ylabel("y", labelpad=6)
    ax.set_zlabel("z", labelpad=6)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=8, pad=1, colors="#555555")

    pane_color = (0.96, 0.97, 0.98, 0.38)
    pane_edge_color = (0.68, 0.72, 0.78, 0.85)
    grid_color = (0.72, 0.76, 0.82, 0.72)
    axis_color = (0.42, 0.46, 0.52, 0.9)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_color)
        axis.pane.set_edgecolor(pane_edge_color)
        axis._axinfo["grid"]["color"] = grid_color
        axis._axinfo["grid"]["linewidth"] = 0.8
        axis._axinfo["axisline"]["color"] = axis_color
        axis._axinfo["tick"]["color"] = axis_color


def _draw_environment(
    ax: object,
    env: RingGraphEnvironment,
    *,
    graph_index: int,
    graph_count: int,
    axis_limit: float,
    elevation: float,
    azimuth: float,
    roll: float,
    show_axes: bool,
    show_node_indices: bool,
) -> None:
    """Draw one graph frame."""

    ax.cla()
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_facecolor("white")
    try:
        ax.set_proj_type("persp")
    except AttributeError:
        pass

    if show_axes:
        _style_3d_axes(ax)
    else:
        ax.set_axis_off()

    for src, dst in undirected_edge_pairs(env.edge_index):
        ax.plot(
            [float(env.R[src, 0]), float(env.R[dst, 0])],
            [float(env.R[src, 1]), float(env.R[dst, 1])],
            [float(env.R[src, 2]), float(env.R[dst, 2])],
            color=EDGE_COLOR,
            linewidth=2.8,
            alpha=0.78,
        )

    roles = env.node_role.tolist()
    for role in (CENTER_ROLE, INNER_ROLE, OUTER_ROLE):
        indices = [i for i, value in enumerate(roles) if int(value) == role]
        if not indices:
            continue
        size = 105 if role == CENTER_ROLE else 75
        ax.scatter(
            [float(env.R[i, 0]) for i in indices],
            [float(env.R[i, 1]) for i in indices],
            [float(env.R[i, 2]) for i in indices],
            s=size,
            c=ROLE_COLORS.get(role, "#999999"),
            edgecolors=NODE_EDGE_COLOR,
            linewidths=0.85,
            alpha=1.0,
            depthshade=False,
            label=ROLE_NAMES.get(role, f"role {role}"),
        )

    if show_node_indices:
        for node_index in range(env.n_nodes):
            ax.text(
                float(env.R[node_index, 0]),
                float(env.R[node_index, 1]),
                float(env.R[node_index, 2]),
                f" {node_index}",
                fontsize=8,
                color="#202020",
            )

    class_number = int(env.label) + 1

    ax.text2D(
        0.5,
        0.955,
        f"Class {class_number}",
        transform=ax.transAxes,
        fontsize=21,
        fontweight="bold",
        ha="center",
        va="top",
    )
    _set_camera(ax, elevation=elevation, azimuth=azimuth, roll=roll)


def _frame_lookup(
    selections: Sequence[RenderSelection],
    frames_per_graph: int,
) -> list[tuple[int, int, int]]:
    """Map video frames to (selection_index, graph_index, graph_count)."""

    lookup: list[tuple[int, int, int]] = []
    for selection_index, selection in enumerate(selections):
        graph_count = len(selection.graphs)
        for graph_index in range(graph_count):
            for _ in range(frames_per_graph):
                lookup.append((selection_index, graph_index, graph_count))
    return lookup


def _ffmpeg_available() -> bool:
    """Return whether matplotlib can find an ffmpeg writer."""

    import matplotlib.animation as animation

    return bool(animation.writers.is_available("ffmpeg"))


def _resolve_output_format(requested_format: str) -> str:
    """Resolve auto/mp4/gif into a concrete format before output paths are built."""

    requested_format = requested_format.lower().lstrip(".")
    if requested_format == "gif":
        return "gif"
    if requested_format == "mp4":
        if not _ffmpeg_available():
            raise RuntimeError(
                "Matplotlib could not find ffmpeg, which is required for --format mp4. "
                "Install ffmpeg, omit --format to use automatic GIF fallback, or rerun with --format gif."
            )
        return "mp4"
    if requested_format == "auto":
        if _ffmpeg_available():
            return "mp4"
        print("Matplotlib could not find ffmpeg; writing GIF output instead of MP4.")
        return "gif"
    raise ValueError(f"Expected --format auto, mp4, or gif, got {requested_format!r}.")


def _make_writer(output_path: Path, *, fps: int, bitrate: int, codec: str):
    """Return a matplotlib animation writer for MP4 or GIF output."""

    import matplotlib.animation as animation

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix != ".mp4":
        raise ValueError(f"Unsupported video extension {output_path.suffix!r}. Use .mp4 or .gif.")
    if not _ffmpeg_available():
        raise RuntimeError(
            "Matplotlib could not find ffmpeg, which is required for MP4 output. "
            "Install ffmpeg or rerun with --format gif."
        )
    return animation.FFMpegWriter(
        fps=fps,
        codec=codec,
        bitrate=bitrate,
        extra_args=["-pix_fmt", "yuv420p"],
    )


def render_rotating_graph_video(
    selections: Sequence[RenderSelection],
    output_path: str | Path,
    *,
    fps: int = 30,
    frames_per_graph: int = 4,
    dpi: int = 130,
    width: int = 1280,
    height: int = 720,
    camera_path: CameraPath | None = None,
    show_axes: bool = True,
    show_node_indices: bool = False,
    bitrate: int = 3000,
    codec: str = "libx264",
    progress_every_graphs: int = 10,
) -> Path:
    """Render a camera-orbit video that steps through the provided graph selections.

    The selections can contain one class or multiple classes. If multiple classes
    are supplied, the video plays the first class group, then the second, while
    updating the on-screen class label.
    """

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if frames_per_graph <= 0:
        raise ValueError(f"frames_per_graph must be positive, got {frames_per_graph}.")

    non_empty = [selection for selection in selections if selection.graphs]
    if not non_empty:
        raise ValueError("No graphs were provided for rendering.")

    camera_path = camera_path or CameraPath()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a non-interactive backend so this works on a server or cluster node.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    frame_map = _frame_lookup(non_empty, int(frames_per_graph))
    total_frames = len(frame_map)
    total_graphs = sum(len(selection.graphs) for selection in non_empty)
    axis_limit = _axis_limit_for_graphs(non_empty)

    figsize = (float(width) / float(dpi), float(height) / float(dpi))
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    print(
        f"Rendering {output_path} ({total_graphs} graphs, {total_frames} frames, "
        f"{width}x{height}, {fps} fps)...",
        flush=True,
    )
    graph_offsets: list[int] = []
    running_graph_count = 0
    for selection in non_empty:
        graph_offsets.append(running_graph_count)
        running_graph_count += len(selection.graphs)
    last_reported_graph = {"index": -1}

    def update(frame_index: int) -> None:
        selection_index, graph_index, graph_count = frame_map[frame_index]
        env = non_empty[selection_index].graphs[graph_index]
        overall_graph_index = graph_offsets[selection_index] + graph_index
        should_report = (
            frame_index % frames_per_graph == 0
            and overall_graph_index != last_reported_graph["index"]
            and (
                overall_graph_index == 0
                or overall_graph_index == total_graphs - 1
                or progress_every_graphs <= 1
                or overall_graph_index % progress_every_graphs == 0
            )
        )
        if should_report:
            print(
                f"  graph {overall_graph_index + 1}/{total_graphs} "
                f"(class {int(env.label) + 1}, frame {frame_index + 1}/{total_frames})",
                flush=True,
            )
            last_reported_graph["index"] = overall_graph_index
        t = 0.0 if total_frames <= 1 else float(frame_index) / float(total_frames - 1)
        elevation, azimuth, roll = camera_path.angles_for_fraction(t)
        _draw_environment(
            ax,
            env,
            graph_index=graph_index,
            graph_count=graph_count,
            axis_limit=axis_limit,
            elevation=elevation,
            azimuth=azimuth,
            roll=roll,
            show_axes=show_axes,
            show_node_indices=show_node_indices,
        )

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000.0 / fps)
    writer = _make_writer(output_path, fps=fps, bitrate=bitrate, codec=codec)
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def _with_suffix_format(path_without_suffix: Path, output_format: str) -> Path:
    output_format = output_format.lower().lstrip(".")
    if output_format not in {"mp4", "gif"}:
        raise ValueError(f"Expected a resolved output format of mp4 or gif, got {output_format!r}.")
    return path_without_suffix.with_suffix(f".{output_format}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rotating camera MP4/GIF videos for both rotating-ring graph classes."
    )

    # Dataset arguments mirror rotating_ring_dataset.py so the video can show the
    # same smooth graph path as the HTML viewer.
    parser.add_argument("--n-graphs", type=int, default=500, help="Total number of graphs across both classes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-inner", type=int, default=8, help="Number of inner-ring nodes.")
    parser.add_argument("--n-outer", type=int, default=8, help="Number of outer-ring nodes.")
    parser.add_argument("--inner-radius-min", type=float, default=1.0)
    parser.add_argument("--inner-radius-max", type=float, default=1.8)
    parser.add_argument("--outer-gap-min", type=float, default=0.8)
    parser.add_argument("--outer-gap-max", type=float, default=1.6)
    parser.add_argument("--outer-rotation-frac-min", type=float, default=0.0)
    parser.add_argument("--outer-rotation-frac-max", type=float, default=0.45)
    parser.add_argument(
        "--outer-3d-rotation-deg",
        type=float,
        default=None,
        help=(
            "Convenience maximum outer-ring 3D tilt in degrees. If supplied, it overrides "
            "--outer-3d-rotation-deg-max."
        ),
    )
    parser.add_argument("--outer-3d-rotation-deg-min", type=float, default=0.0)
    parser.add_argument("--outer-3d-rotation-deg-max", type=float, default=0.0)
    parser.add_argument("--outer-3d-axis-deg", type=float, default=0.0)
    parser.add_argument("--global-rotation-frac-min", type=float, default=0.0)
    parser.add_argument("--global-rotation-frac-max", type=float, default=0.0)
    parser.add_argument("--class-phase-offset-frac", type=float, default=0.5)
    parser.add_argument(
        "--random-parameters",
        action="store_true",
        help="Sample graph parameters randomly instead of using the smooth class path.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle graph order after generation.")
    parser.add_argument("--add-inner-ring-edges", action="store_true")
    parser.add_argument("--add-outer-ring-edges", action="store_true")
    parser.add_argument("--add-center-outer-edges", action="store_true")

    # Video output and camera arguments.
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="rotating_ring")
    parser.add_argument(
        "--format",
        choices=("auto", "mp4", "gif"),
        default="auto",
        help="Video format. auto writes MP4 when ffmpeg is available, otherwise GIF.",
    )
    parser.add_argument(
        "--max-graphs-per-class",
        type=int,
        default=None,
        help="Optional preview limit. Graphs are sampled evenly across each class range.",
    )
    parser.add_argument(
        "--frames-per-graph",
        type=int,
        default=30,
        help="Number of frames to hold each graph before advancing to the next variation.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--bitrate", type=int, default=3000)
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument(
        "--progress-every-graphs",
        type=int,
        default=10,
        help="Print render progress every N graphs. Use 1 for every graph.",
    )
    parser.add_argument("--camera-turns", type=float, default=0.5, help="Number of azimuth orbits across each video.")
    parser.add_argument("--azimuth-start-deg", type=float, default=35.0)
    parser.add_argument("--elevation-mid-deg", type=float, default=28.0)
    parser.add_argument("--elevation-amplitude-deg", type=float, default=12.0)
    parser.add_argument("--elevation-cycles", type=float, default=0.5)
    parser.add_argument("--roll-amplitude-deg", type=float, default=4.0)
    parser.add_argument("--roll-cycles", type=float, default=0.5)
    parser.set_defaults(show_axes=True)
    parser.add_argument("--show-axes", action="store_true", help="Show the 3D axis box, labels, panes, and grid.")
    parser.add_argument("--hide-axes", action="store_false", dest="show_axes", help="Hide the 3D axis box and grid.")
    parser.add_argument("--show-node-indices", action="store_true")
    parser.add_argument(
        "--skip-combined-video",
        action="store_true",
        help="Only write the per-class videos.",
    )
    parser.add_argument(
        "--skip-separate-class-videos",
        action="store_true",
        help="Only write the combined two-class video.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outer_3d_rotation_deg_max = (
        args.outer_3d_rotation_deg
        if args.outer_3d_rotation_deg is not None
        else args.outer_3d_rotation_deg_max
    )

    environments = create_rotating_ring_dataset(
        n_graphs=args.n_graphs,
        seed=args.seed,
        n_inner=args.n_inner,
        n_outer=args.n_outer,
        inner_radius_range=(args.inner_radius_min, args.inner_radius_max),
        outer_gap_range=(args.outer_gap_min, args.outer_gap_max),
        outer_rotation_fraction_range=(args.outer_rotation_frac_min, args.outer_rotation_frac_max),
        outer_3d_rotation_range=(
            args.outer_3d_rotation_deg_min * pi / 180.0,
            outer_3d_rotation_deg_max * pi / 180.0,
        ),
        outer_3d_axis_angle=args.outer_3d_axis_deg * pi / 180.0,
        global_rotation_fraction_range=(args.global_rotation_frac_min, args.global_rotation_frac_max),
        class_phase_offset_fraction=args.class_phase_offset_frac,
        smooth_order=not args.random_parameters,
        shuffle=args.shuffle,
        add_inner_ring_edges=args.add_inner_ring_edges,
        add_outer_ring_edges=args.add_outer_ring_edges,
        add_center_outer_edges=args.add_center_outer_edges,
    )

    selections = _group_by_class(environments)
    selections = [
        RenderSelection(
            label=selection.label,
            graphs=_evenly_limit_graphs(selection.graphs, args.max_graphs_per_class),
        )
        for selection in selections
    ]

    if args.skip_combined_video and args.skip_separate_class_videos:
        raise ValueError("Both combined and separate videos were skipped; nothing to render.")

    camera_path = CameraPath(
        azimuth_start_deg=args.azimuth_start_deg,
        azimuth_turns=args.camera_turns,
        elevation_mid_deg=args.elevation_mid_deg,
        elevation_amplitude_deg=args.elevation_amplitude_deg,
        elevation_cycles=args.elevation_cycles,
        roll_amplitude_deg=args.roll_amplitude_deg,
        roll_cycles=args.roll_cycles,
    )

    output_format = _resolve_output_format(args.format)
    output_paths: list[Path] = []
    if not args.skip_combined_video:
        output_path = _with_suffix_format(args.output_dir / f"{args.output_prefix}_both_classes", output_format)
        output_paths.append(
            render_rotating_graph_video(
                selections,
                output_path,
                fps=args.fps,
                frames_per_graph=args.frames_per_graph,
                dpi=args.dpi,
                width=args.width,
                height=args.height,
                camera_path=camera_path,
                show_axes=args.show_axes,
                show_node_indices=args.show_node_indices,
                bitrate=args.bitrate,
                codec=args.codec,
                progress_every_graphs=args.progress_every_graphs,
            )
        )

    if not args.skip_separate_class_videos:
        for selection in selections:
            class_number = int(selection.label) + 1
            class_name = _class_name_for_label(selection.label, selection.graphs[0] if selection.graphs else None)
            safe_class_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in class_name).strip("_")
            output_path = _with_suffix_format(
                args.output_dir / f"{args.output_prefix}_class_{class_number}_{safe_class_name}",
                output_format,
            )
            output_paths.append(
                render_rotating_graph_video(
                    [selection],
                    output_path,
                    fps=args.fps,
                    frames_per_graph=args.frames_per_graph,
                    dpi=args.dpi,
                    width=args.width,
                    height=args.height,
                    camera_path=camera_path,
                    show_axes=args.show_axes,
                    show_node_indices=args.show_node_indices,
                    bitrate=args.bitrate,
                    codec=args.codec,
                    progress_every_graphs=args.progress_every_graphs,
                )
            )

    print("Rendered videos:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
