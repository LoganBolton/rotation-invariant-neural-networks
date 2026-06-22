"""Configuration parsing helpers for benchmark sweeps."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

# model type, l_max, n_max
# DEFAULT_MODEL_CONFIGS = (
#     ("hiphop", 0, 1),
#     ("hiphop", 1, 4),
#     ("hiphop", 2, 4),
#     ("hiphop", 3, 4),
# )

DEFAULT_MODEL_CONFIGS = (
    ("hiphop", 0, 1),
    ("hiphop", 1, 2),
    ("hiphop", 1, 3),
    ("hiphop", 1, 4),
    ("hiphop", 2, 2),
    ("hiphop", 2, 3),
    ("hiphop", 2, 4),
    ("hiphop", 3, 2),
    ("hiphop", 3, 3),
    ("hiphop", 3, 4),
)

# num inner nodes, num outer nodes
DEFAULT_2D_RING_GRAPH_CONFIGS = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
)
DEFAULT_3D_RING_GRAPH_CONFIGS = (
    (3, 3),
    (4, 4),
)
RING_GRAPH_CONFIG_RE = re.compile(r"^(?P<dimension>[23]d)_(?P<inner>\d+)inner_(?P<outer>\d+)_outer$")


@dataclass(frozen=True)
class RingGraphConfig:
    name: str
    n_inner: int
    n_outer: int
    outer_3d_rotation_deg: float
    outer_3d_axis_deg: float


def parse_model_configs(configs: list[str] | None) -> list[tuple[str, int, int]]:
    if configs is None:
        return []
    if configs in (["default"], ["all"]):
        return list(DEFAULT_MODEL_CONFIGS)

    parsed = []
    for config in configs:
        if config in {"default", "all"}:
            parsed.extend(DEFAULT_MODEL_CONFIGS)
        elif config == "hipnn":
            parsed.append(("hipnn", 0, 1))
        elif config.startswith("l") and "_n" in config:
            l_text, n_text = config.removeprefix("l").split("_n", maxsplit=1)
            parsed.append(("hiphop", int(l_text), int(n_text)))
        elif config.startswith("hiphop:"):
            _model, l_text, n_text = config.split(":", maxsplit=2)
            parsed.append(("hiphop", int(l_text), int(n_text)))
        else:
            raise ValueError(
                f"Unknown model config {config!r}. Use 'default', 'all', 'hipnn', 'l2_n3', or 'hiphop:2:3'."
            )

    return parsed


def parse_ring_graph_configs(configs: list[str] | None, output_dir: Path | None) -> list[RingGraphConfig]:
    if configs is None:
        return []

    parsed = []
    for config in configs:
        normalized = config.lower().replace("-", "_")
        if normalized in {"all", "all_2d", "2d", "all_3d", "3d"}:
            dimension = None if normalized == "all" else normalized.removeprefix("all_")
            parsed.extend(_discover_ring_graph_configs(output_dir, dimension) or _default_ring_graph_configs(dimension))
            continue

        match = RING_GRAPH_CONFIG_RE.match(normalized)
        if match:
            parsed.append(_make_ring_graph_config(match.group("dimension"), int(match.group("inner")), int(match.group("outer"))))
            continue

        parts = normalized.split(":")
        if len(parts) == 3 and parts[0] in {"2d", "3d"}:
            parsed.append(_make_ring_graph_config(parts[0], int(parts[1]), int(parts[2])))
            continue

        raise ValueError(
            f"Unknown rotating-ring graph config {config!r}. "
            "Use 'all_2d', 'all_3d', 'all', '2d_3inner_4_outer', or '2d:3:4'."
        )

    deduped = []
    seen = set()
    for config in parsed:
        if config.name not in seen:
            seen.add(config.name)
            deduped.append(config)
    return deduped


def config_log_name(model: str, l_max: int, n_max: int) -> str:
    if model == "hipnn":
        return "l0_n1.md"
    return f"l{l_max}_n{n_max}.md"


def args_for_config(args: argparse.Namespace, model: str, l_max: int, n_max: int) -> argparse.Namespace:
    config_args = argparse.Namespace(**vars(args))
    config_args.model = model
    config_args.l_max = l_max
    config_args.n_max = n_max
    return config_args


def args_for_ring_graph_config(
    args: argparse.Namespace,
    graph_config: RingGraphConfig,
    output_dir: Path | None = None,
) -> argparse.Namespace:
    config_args = argparse.Namespace(**vars(args))
    config_args.ring_n_inner = graph_config.n_inner
    config_args.ring_n_outer = graph_config.n_outer
    config_args.ring_outer_3d_rotation_deg = graph_config.outer_3d_rotation_deg
    config_args.ring_outer_3d_axis_deg = graph_config.outer_3d_axis_deg
    if output_dir is not None:
        config_args.output_dir = output_dir
    return config_args


def _make_ring_graph_config(dimension: str, n_inner: int, n_outer: int) -> RingGraphConfig:
    if dimension not in {"2d", "3d"}:
        raise ValueError(f"Unknown rotating-ring graph dimension {dimension!r}. Expected '2d' or '3d'.")
    rotation = 360.0 if dimension == "3d" else 0.0
    return RingGraphConfig(
        name=f"{dimension}_{n_inner}inner_{n_outer}_outer",
        n_inner=n_inner,
        n_outer=n_outer,
        outer_3d_rotation_deg=rotation,
        outer_3d_axis_deg=rotation,
    )


def _discover_ring_graph_configs(output_dir: Path | None, dimension: str | None) -> list[RingGraphConfig]:
    if output_dir is None or not output_dir.exists():
        return []

    configs = []
    for child in output_dir.iterdir():
        match = RING_GRAPH_CONFIG_RE.match(child.name) if child.is_dir() else None
        if match is None or (dimension is not None and match.group("dimension") != dimension):
            continue
        configs.append(_make_ring_graph_config(match.group("dimension"), int(match.group("inner")), int(match.group("outer"))))

    return sorted(configs, key=lambda config: (config.name.startswith("3d_"), config.n_inner, config.n_outer))


def _default_ring_graph_configs(dimension: str | None) -> list[RingGraphConfig]:
    configs = []
    if dimension in {None, "2d"}:
        configs.extend(_make_ring_graph_config("2d", n_inner, n_outer) for n_inner, n_outer in DEFAULT_2D_RING_GRAPH_CONFIGS)
    if dimension in {None, "3d"}:
        configs.extend(_make_ring_graph_config("3d", n_inner, n_outer) for n_inner, n_outer in DEFAULT_3D_RING_GRAPH_CONFIGS)
    return configs
