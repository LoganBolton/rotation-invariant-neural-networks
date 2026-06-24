"""Capture and plot HIP-HOP invariant activations on the rotating-ring dataset."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "run_models" / ".matplotlib-cache"))
os.environ.setdefault("HIPPYNN_USE_CUSTOM_KERNELS", "False")

BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from run_models.train import load_dataset, make_model, model_forward_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/rotating_ring/results/show_invariants"))
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--n-graphs", type=int, default=100)
    parser.add_argument("--n-inner", type=int, default=3)
    parser.add_argument("--n-outer", type=int, default=3)
    parser.add_argument("--outer-3d-rotation-deg", type=float, default=0.0)
    parser.add_argument("--outer-3d-axis-deg", type=float, default=0.0)
    parser.add_argument("--n-interaction-layers", type=int, default=1)
    parser.add_argument("--n-atom-layers", type=int, default=1)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-sensitivities", type=int, default=16)
    parser.add_argument("--l-max", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--dist-hard-max", type=float, default=6.5)
    parser.add_argument("--success-margin", type=float, default=0.1)
    parser.add_argument("--stop-at-accuracy", type=float, default=1.0)
    parser.add_argument("--plot-radial-functions", action="store_true", help="Write radial sensitivity-function plots in a separate subdirectory.")
    parser.add_argument("--radial-r-min", type=float, default=0.0)
    parser.add_argument("--radial-r-max", type=float, default=None)
    parser.add_argument("--radial-n-points", type=int, default=500)
    parser.add_argument(
        "--no-channel-average",
        action="store_true",
        help=(
            "Also write per-feature-channel invariant class deltas instead of only "
            "the feature-averaged center-node summary."
        ),
    )
    parser.add_argument(
        "--invariant-axis-contractions",
        action="store_true",
        help="Label invariant plot axes by the tensor contraction for each invariant column.",
    )
    return parser.parse_args()


def training_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="rotating_ring",
        counterexample="all",
        k=4,
        model="hiphop",
        neighborhood_cutoff="edges",
        learning_rate=args.learning_rate,
        n_interaction_layers=args.n_interaction_layers,
        n_atom_layers=args.n_atom_layers,
        n_features=args.n_features,
        n_sensitivities=args.n_sensitivities,
        dist_soft_min=None,
        dist_soft_max=None,
        dist_hard_max=args.dist_hard_max,
        l_max=args.l_max,
        n_max=args.n_max,
        ring_n_graphs=args.n_graphs,
        ring_seed=args.seed,
        ring_n_inner=args.n_inner,
        ring_n_outer=args.n_outer,
        ring_outer_3d_rotation_deg=args.outer_3d_rotation_deg,
        ring_outer_3d_axis_deg=args.outer_3d_axis_deg,
        epochs=args.epochs,
        seed=args.seed,
        stop_at_accuracy=args.stop_at_accuracy,
        success_margin=args.success_margin,
    )


def graph_label(args: argparse.Namespace) -> str:
    return f"{args.n_inner} inner {args.n_outer} outer"


def radial_reference_distances() -> dict[str, float]:
    inner_radius = 1.0
    outer_radius = inner_radius + 1.2
    return {"inner": inner_radius, "outer": outer_radius}


def train_model(model: torch.nn.Module, forward_args: tuple[torch.Tensor, ...], targets: torch.Tensor, args: argparse.Namespace) -> dict[str, float]:
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    final = {"epoch": 0, "loss": float("nan"), "accuracy": 0.0, "margin_accuracy": 0.0}

    for epoch in range(1, args.epochs + 1):
        (logits,) = model(*forward_args)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = (logits >= 0).to(targets.dtype)
            accuracy = float((predictions == targets).to(torch.float32).mean().item())
            signed_targets = targets.mul(2).sub(1)
            margin_accuracy = float((signed_targets * logits >= args.success_margin).to(torch.float32).mean().item())

        final = {
            "epoch": float(epoch),
            "loss": float(loss.item()),
            "accuracy": accuracy,
            "margin_accuracy": margin_accuracy,
        }
        if margin_accuracy >= args.stop_at_accuracy:
            break

    return final


def capture_invariants(model: torch.nn.Module, forward_args: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captures[name] = output.detach().cpu()

        return hook

    for name, module in model.named_modules():
        if type(module).__name__ == "HopInvariantLayer":
            handles.append(module.register_forward_hook(make_hook(name)))

    if not handles:
        raise RuntimeError("No HopInvariantLayer modules found in the model.")

    model.eval()
    with torch.no_grad():
        model(*forward_args)

    for handle in handles:
        handle.remove()
    return captures


def sensitivity_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, "sensitivity"):
            layers.append((name, module.sensitivity))
    return layers


def evaluate_sensitivity(sensitivity: torch.nn.Module, distances: torch.Tensor) -> torch.Tensor:
    try:
        return sensitivity(distances, warn_low_distances=False).detach().cpu()
    except TypeError:
        return sensitivity(distances).detach().cpu()


def write_radial_function_csv(path: Path, distances: torch.Tensor, sensitivity_values: torch.Tensor) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["distance"]
        header.extend(f"sensitivity_{index}" for index in range(sensitivity_values.shape[1]))
        header.append("sum")
        writer.writerow(header)
        for row_index in range(distances.numel()):
            values = sensitivity_values[row_index]
            writer.writerow(
                [
                    f"{float(distances[row_index]):.8g}",
                    *[f"{float(value):.8g}" for value in values],
                    f"{float(values.sum()):.8g}",
                ]
            )


def plot_radial_functions(
    path: Path,
    distances: torch.Tensor,
    sensitivity_values: torch.Tensor,
    *,
    title: str,
    reference_distances: dict[str, float] | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    for sensitivity_index in range(sensitivity_values.shape[1]):
        axis.plot(
            distances,
            sensitivity_values[:, sensitivity_index],
            color="#b33a3a",
            alpha=0.35,
            linewidth=1.3,
            label=f"S{sensitivity_index}",
        )
    axis.plot(distances, sensitivity_values.sum(dim=1), color="#145f7a", linewidth=2.4, label="sum")
    if reference_distances:
        colors = ["#2f7d32", "#f47c20", "#555555"]
        for index, (label, distance) in enumerate(reference_distances.items()):
            axis.axvline(
                distance,
                color=colors[index % len(colors)],
                linestyle="--",
                linewidth=1.8,
                label=f"{label} r={distance:g}",
            )
    axis.set_title(title)
    axis.set_xlabel("pair distance r")
    axis.set_ylabel("radial sensitivity")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=4, fontsize=7.5)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_radial_function_outputs(
    output_dir: Path,
    model: torch.nn.Module,
    *,
    title_prefix: str,
    r_min: float,
    r_max: float,
    n_points: int,
    reference_distances: dict[str, float] | None = None,
) -> list[tuple[Path, Path]]:
    radial_dir = output_dir / "radial_functions"
    radial_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for layer_index, (name, sensitivity) in enumerate(sensitivity_layers(model)):
        first_param = next(sensitivity.parameters())
        distances = torch.linspace(r_min, r_max, n_points, dtype=first_param.dtype, device=first_param.device)
        sensitivity_values = evaluate_sensitivity(sensitivity, distances)
        distances_cpu = distances.detach().cpu()

        safe_name = f"layer{layer_index}"
        csv_path = radial_dir / f"{safe_name}_radial_functions.csv"
        png_path = radial_dir / f"{safe_name}_radial_functions.png"
        write_radial_function_csv(csv_path, distances_cpu, sensitivity_values)
        plot_radial_functions(
            png_path,
            distances_cpu,
            sensitivity_values,
            title=f"{title_prefix}: radial sensitivity functions",
            reference_distances=reference_distances,
        )
        outputs.append((csv_path, png_path))

    if not outputs:
        raise RuntimeError("No model modules with sensitivity functions were found.")
    return outputs


CONTRACTION_LABELS = {
    0: r"$s$",
    1: r"$v_i v_i$",
    2: r"$q_{ij}q_{ij}$",
    3: r"$q_{ij}q_{jk}q_{ki}$",
    4: r"$t_{ijk}t_{ijk}$",
    5: r"$(t_{ikl}t_{jkl})(t_{imn}t_{jmn})$",
    6: r"$q_{ij}v_i v_j$",
    7: r"$(q_{ij}v_j)(q_{ik}v_k)$",
    8: r"$t_{ijk}v_i v_j v_k$",
    9: r"$(t_{ikl}t_{jkl})v_i v_j$",
    10: r"$(t_{ikl}t_{jkl})q_{ij}$",
    11: r"$(t_{ikl}t_{jkl})(q_{im}q_{jm})$",
    12: r"$(t_{ijk}q_{jk})(t_{ilm}q_{lm})$",
}


def active_invariant_ids(l_max: int, n_max: int) -> list[int]:
    invariant_ids = [0]
    if l_max >= 1 and n_max >= 2:
        invariant_ids.append(1)
    if l_max >= 2:
        if n_max >= 2:
            invariant_ids.append(2)
        if n_max >= 3:
            invariant_ids.extend([3, 6])
        if n_max >= 4:
            invariant_ids.append(7)
    if l_max >= 3:
        if n_max >= 2:
            invariant_ids.append(4)
        if n_max >= 3:
            invariant_ids.append(10)
        if n_max >= 4:
            invariant_ids.extend([5, 8, 9, 11, 12])
    return sorted(invariant_ids)


def invariant_axis_labels(l_max: int, n_max: int, n_invariants: int, *, use_contractions: bool) -> list[str]:
    if not use_contractions:
        return [f"I{i}" for i in range(n_invariants)]

    invariant_ids = active_invariant_ids(l_max, n_max)
    if len(invariant_ids) != n_invariants:
        return [f"I{i}" for i in range(n_invariants)]
    return [CONTRACTION_LABELS[invariant_id] for invariant_id in invariant_ids]


def set_invariant_xticks(axis: plt.Axes, invariant_labels: list[str], *, rotation: float = 0.0) -> None:
    axis.set_xticks(torch.arange(len(invariant_labels)))
    axis.set_xticklabels(invariant_labels, rotation=rotation, ha="right" if rotation else "center")


def summarize_capture(
    capture: torch.Tensor,
    *,
    node_counts: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    n_atoms = int(node_counts.sum().item())
    if capture.shape[0] % n_atoms != 0:
        raise ValueError(f"Cannot reshape capture with shape {tuple(capture.shape)} over {n_atoms} atoms.")

    n_features = capture.shape[0] // n_atoms
    n_invariants = capture.shape[1]
    inv = capture.reshape(n_atoms, n_features, n_invariants)

    starts = torch.cat([torch.zeros(1, dtype=torch.long), node_counts.cumsum(dim=0)[:-1]])
    center_by_graph_feature = inv[starts]
    center_by_graph = center_by_graph_feature.mean(dim=1)

    class_means = torch.stack([center_by_graph[labels == label].mean(dim=0) for label in (0, 1)])
    class_stds = torch.stack([center_by_graph[labels == label].std(dim=0, unbiased=False) for label in (0, 1)])
    pooled_std = torch.sqrt((class_stds[0].pow(2) + class_stds[1].pow(2)) / 2.0).clamp_min(1.0e-12)
    standardized_delta = (class_means[1] - class_means[0]) / pooled_std

    feature_class_means = torch.stack([center_by_graph_feature[labels == label].mean(dim=0) for label in (0, 1)])
    feature_class_stds = torch.stack([center_by_graph_feature[labels == label].std(dim=0, unbiased=False) for label in (0, 1)])
    feature_pooled_std = torch.sqrt((feature_class_stds[0].pow(2) + feature_class_stds[1].pow(2)) / 2.0).clamp_min(1.0e-12)
    feature_standardized_delta = (feature_class_means[1] - feature_class_means[0]) / feature_pooled_std

    return (
        center_by_graph,
        class_means,
        class_stds,
        standardized_delta,
        feature_class_means,
        feature_class_stds,
        feature_standardized_delta,
    )


def write_summary_csv(path: Path, class_means: torch.Tensor, class_stds: torch.Tensor, standardized_delta: torch.Tensor) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "invariant_index",
                "class0_center_mean",
                "class0_center_std",
                "class0_center_variance",
                "class1_center_mean",
                "class1_center_std",
                "class1_center_variance",
                "class1_minus_class0",
                "standardized_delta",
            ]
        )
        for i in range(class_means.shape[1]):
            diff = class_means[1, i] - class_means[0, i]
            writer.writerow(
                [
                    i,
                    f"{float(class_means[0, i]):.8g}",
                    f"{float(class_stds[0, i]):.8g}",
                    f"{float(class_stds[0, i].pow(2)):.8g}",
                    f"{float(class_means[1, i]):.8g}",
                    f"{float(class_stds[1, i]):.8g}",
                    f"{float(class_stds[1, i].pow(2)):.8g}",
                    f"{float(diff):.8g}",
                    f"{float(standardized_delta[i]):.8g}",
                ]
            )


def write_center_values_csv(path: Path, values: torch.Tensor, labels: torch.Tensor) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["graph_index", "label", "invariant_index", "center_value"])
        for graph_index in range(values.shape[0]):
            label = int(labels[graph_index].item())
            for invariant_index in range(values.shape[1]):
                writer.writerow(
                    [
                        graph_index,
                        label,
                        invariant_index,
                        f"{float(values[graph_index, invariant_index]):.8g}",
                    ]
                )


def plot_invariants(
    path: Path,
    values: torch.Tensor,
    labels: torch.Tensor,
    standardized_delta: torch.Tensor,
    title: str,
    invariant_labels: list[str],
) -> None:
    n_invariants = values.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, n_invariants * 1.0), 7.5), constrained_layout=True)

    x = torch.arange(n_invariants)
    for label, color, marker in [(0, "#145f7a", "o"), (1, "#f47c20", "s")]:
        class_values = values[labels == label]
        mean = class_values.mean(dim=0)
        std = class_values.std(dim=0, unbiased=False)
        axes[0].errorbar(x, mean, yerr=std, fmt=marker + "-", color=color, capsize=3, label=f"class {label}")

    axes[0].set_title(title)
    axes[0].set_xlabel("invariant contraction" if invariant_labels[0].startswith("$") else "invariant index")
    axes[0].set_ylabel("center-node mean over feature channels")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(x, standardized_delta, color=["#145f7a" if v < 0 else "#f47c20" for v in standardized_delta.tolist()])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("invariant contraction" if invariant_labels[0].startswith("$") else "invariant index")
    axes[1].set_ylabel("standardized class delta")
    axes[1].grid(True, axis="y", alpha=0.25)

    for axis in axes:
        set_invariant_xticks(axis, invariant_labels, rotation=30.0 if invariant_labels[0].startswith("$") else 0.0)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_feature_delta_csv(path: Path, feature_standardized_delta: torch.Tensor) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_index", "invariant_index", "standardized_delta"])
        for feature_index in range(feature_standardized_delta.shape[0]):
            for invariant_index in range(feature_standardized_delta.shape[1]):
                writer.writerow(
                    [
                        feature_index,
                        invariant_index,
                        f"{float(feature_standardized_delta[feature_index, invariant_index]):.8g}",
                    ]
                )


def write_channel_summary_csv(
    path: Path,
    feature_class_means: torch.Tensor,
    feature_class_stds: torch.Tensor,
    feature_standardized_delta: torch.Tensor,
) -> None:
    raw_delta = feature_class_means[1] - feature_class_means[0]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feature_index",
                "invariant_index",
                "class0_center_mean",
                "class0_center_std",
                "class0_center_variance",
                "class1_center_mean",
                "class1_center_std",
                "class1_center_variance",
                "class1_minus_class0",
                "abs_class1_minus_class0",
                "standardized_delta",
                "abs_standardized_delta",
            ]
        )
        for feature_index in range(feature_class_means.shape[1]):
            for invariant_index in range(feature_class_means.shape[2]):
                diff = raw_delta[feature_index, invariant_index]
                standardized = feature_standardized_delta[feature_index, invariant_index]
                writer.writerow(
                    [
                        feature_index,
                        invariant_index,
                        f"{float(feature_class_means[0, feature_index, invariant_index]):.8g}",
                        f"{float(feature_class_stds[0, feature_index, invariant_index]):.8g}",
                        f"{float(feature_class_stds[0, feature_index, invariant_index].pow(2)):.8g}",
                        f"{float(feature_class_means[1, feature_index, invariant_index]):.8g}",
                        f"{float(feature_class_stds[1, feature_index, invariant_index]):.8g}",
                        f"{float(feature_class_stds[1, feature_index, invariant_index].pow(2)):.8g}",
                        f"{float(diff):.8g}",
                        f"{float(diff.abs()):.8g}",
                        f"{float(standardized):.8g}",
                        f"{float(standardized.abs()):.8g}",
                    ]
                )


def plot_feature_delta_heatmap(
    path: Path,
    feature_standardized_delta: torch.Tensor,
    title: str,
    invariant_labels: list[str],
) -> None:
    vmax = max(1.0, float(feature_standardized_delta.abs().max().item()))
    fig_width = max(10.0, feature_standardized_delta.shape[1] * (1.25 if invariant_labels[0].startswith("$") else 0.75))
    fig, axis = plt.subplots(figsize=(fig_width, max(4.0, feature_standardized_delta.shape[0] * 0.33)), constrained_layout=True)
    image = axis.imshow(
        feature_standardized_delta,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("invariant contraction" if invariant_labels[0].startswith("$") else "invariant index")
    axis.set_ylabel("feature channel")
    set_invariant_xticks(axis, invariant_labels, rotation=35.0 if invariant_labels[0].startswith("$") else 0.0)
    axis.set_yticks(torch.arange(feature_standardized_delta.shape[0]))
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("standardized class delta")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_channel_delta_magnitude_heatmap(
    path: Path,
    feature_standardized_delta: torch.Tensor,
    title: str,
    invariant_labels: list[str],
) -> None:
    delta_magnitude = feature_standardized_delta.abs()
    vmax = max(1.0, float(torch.quantile(delta_magnitude.reshape(-1), 0.98).item()))
    fig_width = max(9.0, feature_standardized_delta.shape[1] * (1.25 if invariant_labels[0].startswith("$") else 0.75))
    fig_height = max(4.0, feature_standardized_delta.shape[0] * 0.36)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = axis.imshow(
        delta_magnitude,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("invariant contraction" if invariant_labels[0].startswith("$") else "invariant index k")
    axis.set_ylabel("feature channel a")
    set_invariant_xticks(axis, invariant_labels, rotation=35.0 if invariant_labels[0].startswith("$") else 0.0)
    axis.set_yticks(torch.arange(feature_standardized_delta.shape[0]))
    axis.set_yticklabels([f"a = {i}" for i in range(feature_standardized_delta.shape[0])])
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("|standardized class delta|")

    for feature_index in range(feature_standardized_delta.shape[0]):
        for invariant_index in range(feature_standardized_delta.shape[1]):
            signed_value = float(feature_standardized_delta[feature_index, invariant_index])
            if abs(signed_value) >= 1.0:
                text_color = "white" if abs(signed_value) > 0.65 * vmax else "black"
                axis.text(
                    invariant_index,
                    feature_index,
                    f"{signed_value:+.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_args = training_namespace(args)
    arrays, description = load_dataset(train_args)
    forward_args = model_forward_args(train_args, arrays)
    targets = arrays["T"]
    labels = targets.squeeze(1).to(torch.long)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="HIP-HOP-NN is still in a beta state.*")
        model = make_model(train_args)

    metrics = train_model(model, forward_args, targets, args)
    captures = capture_invariants(model, forward_args)
    plot_title_prefix = graph_label(args)
    radial_r_max = args.radial_r_max if args.radial_r_max is not None else args.dist_hard_max

    summary_lines = [
        f"dataset: {description}",
        f"model: hiphop l_max={args.l_max} n_max={args.n_max}",
        f"epochs: {int(metrics['epoch'])}",
        f"loss: {metrics['loss']:.6f}",
        f"accuracy: {metrics['accuracy']:.3f}",
        f"margin_accuracy: {metrics['margin_accuracy']:.3f}",
        "",
    ]
    if args.plot_radial_functions:
        radial_outputs = write_radial_function_outputs(
            args.output_dir,
            model,
            title_prefix=plot_title_prefix,
            r_min=args.radial_r_min,
            r_max=radial_r_max,
            n_points=args.radial_n_points,
            reference_distances=radial_reference_distances(),
        )
        summary_lines.append(f"radial_function_range: {args.radial_r_min:g} to {radial_r_max:g}")
        for radial_csv_path, radial_png_path in radial_outputs:
            summary_lines.append(f"radial_function_csv: {radial_csv_path}")
            summary_lines.append(f"radial_function_plot: {radial_png_path}")
        summary_lines.append("")

    for layer_index, (name, capture) in enumerate(captures.items()):
        (
            values,
            class_means,
            class_stds,
            standardized_delta,
            feature_class_means,
            feature_class_stds,
            feature_standardized_delta,
        ) = summarize_capture(
            capture,
            node_counts=arrays["node_counts"].cpu(),
            labels=labels.cpu(),
        )
        invariant_labels = invariant_axis_labels(
            args.l_max,
            args.n_max,
            values.shape[1],
            use_contractions=args.invariant_axis_contractions,
        )
        safe_name = f"layer{layer_index}"
        csv_path = args.output_dir / f"{safe_name}_center_invariants.csv"
        values_csv_path = args.output_dir / f"{safe_name}_center_invariant_values.csv"
        png_path = args.output_dir / f"{safe_name}_center_invariants.png"
        feature_csv_path = args.output_dir / f"{safe_name}_feature_invariant_deltas.csv"
        feature_png_path = args.output_dir / f"{safe_name}_feature_invariant_deltas.png"
        channel_csv_path = args.output_dir / f"{safe_name}_channel_invariant_class_deltas.csv"
        channel_png_path = args.output_dir / f"{safe_name}_channel_invariant_delta_magnitude.png"
        write_summary_csv(csv_path, class_means, class_stds, standardized_delta)
        write_center_values_csv(values_csv_path, values, labels.cpu())
        write_feature_delta_csv(feature_csv_path, feature_standardized_delta)
        plot_invariants(
            png_path,
            values,
            labels.cpu(),
            standardized_delta,
            f"{plot_title_prefix}: center node invariant means by class",
            invariant_labels,
        )
        plot_feature_delta_heatmap(
            feature_png_path,
            feature_standardized_delta,
            f"{plot_title_prefix}: center node feature x invariant class deltas",
            invariant_labels,
        )
        if args.no_channel_average:
            write_channel_summary_csv(
                channel_csv_path,
                feature_class_means,
                feature_class_stds,
                feature_standardized_delta,
            )
            plot_channel_delta_magnitude_heatmap(
                channel_png_path,
                feature_standardized_delta,
                f"{plot_title_prefix}: center node invariant class deltas",
                invariant_labels,
            )

        top = torch.argsort(standardized_delta.abs(), descending=True)[: min(5, standardized_delta.numel())]
        top_text = ", ".join(f"I{int(i)}={float(standardized_delta[i]):+.2f}" for i in top)
        flat_top = torch.argsort(feature_standardized_delta.abs().reshape(-1), descending=True)[: min(8, feature_standardized_delta.numel())]
        feature_top_parts = []
        for flat_index in flat_top:
            feature_index = int(flat_index // feature_standardized_delta.shape[1])
            invariant_index = int(flat_index % feature_standardized_delta.shape[1])
            value = float(feature_standardized_delta[feature_index, invariant_index])
            feature_top_parts.append(f"F{feature_index}/I{invariant_index}={value:+.2f}")
        summary_lines.append(f"{safe_name}: {name}")
        summary_lines.append(f"  capture_shape: {tuple(capture.shape)}")
        if args.invariant_axis_contractions:
            summary_lines.append(f"  invariant_axis_labels: {', '.join(invariant_labels)}")
        summary_lines.append(f"  top standardized deltas: {top_text}")
        summary_lines.append(f"  top feature/invariant deltas: {', '.join(feature_top_parts)}")
        summary_lines.append(f"  csv: {csv_path}")
        summary_lines.append(f"  values_csv: {values_csv_path}")
        summary_lines.append(f"  plot: {png_path}")
        summary_lines.append(f"  feature_delta_csv: {feature_csv_path}")
        summary_lines.append(f"  feature_delta_plot: {feature_png_path}")
        if args.no_channel_average:
            summary_lines.append(f"  channel_delta_csv: {channel_csv_path}")
            summary_lines.append(f"  channel_delta_magnitude_plot: {channel_png_path}")
        summary_lines.append("")

    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
