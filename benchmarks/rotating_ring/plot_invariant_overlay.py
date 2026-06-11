"""Overlay ring-node HIP-HOP invariant summaries."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


RUN_DIRS = {
    1: "show_invariants_n1",
    2: "show_invariants_n2",
    3: "show_invariants",
    4: "show_invariants_n4",
}
COLORS = {
    1: "#555555",
    2: "#2f7d32",
    3: "#006bb6",
    4: "#f47c20",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("benchmarks/rotating_ring/results/l3_n2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--l-max", type=int, default=3)
    parser.add_argument("--n-max", type=int, default=2)
    parser.add_argument("--epochs-requested", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-sensitivities", type=int, default=16)
    parser.add_argument("--write-log-y", action="store_true", help="Also write a symlog-y overlay plot.")
    return parser.parse_args()


def run_label(n_ring: int) -> str:
    return f"{n_ring} inner / {n_ring} outer"


def read_center_csv(path: Path) -> dict[str, torch.Tensor]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    return {
        "index": torch.tensor([int(row["invariant_index"]) for row in rows], dtype=torch.long),
        "class0": torch.tensor([float(row["class0_center_mean"]) for row in rows]),
        "class1": torch.tensor([float(row["class1_center_mean"]) for row in rows]),
        "delta": torch.tensor([float(row["standardized_delta"]) for row in rows]),
    }


def read_feature_csv(path: Path) -> torch.Tensor:
    triples = []
    max_feature = 0
    max_invariant = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            feature = int(row["feature_index"])
            invariant = int(row["invariant_index"])
            value = float(row["standardized_delta"])
            triples.append((feature, invariant, value))
            max_feature = max(max_feature, feature)
            max_invariant = max(max_invariant, invariant)

    values = torch.zeros((max_feature + 1, max_invariant + 1))
    for feature, invariant, value in triples:
        values[feature, invariant] = value
    return values


def read_summary(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    fields = {}
    for key in ("epochs", "loss", "accuracy", "margin_accuracy"):
        match = re.search(rf"^{key}: ([^\n]+)", text, flags=re.MULTILINE)
        fields[key] = match.group(1) if match else "unknown"
    match = re.search(r"capture_shape: ([^\n]+)", text)
    fields["capture_shape"] = match.group(1) if match else "unknown"
    return fields


def write_overlay_csv(path: Path, centers: dict[int, dict[str, torch.Tensor]], feature_deltas: dict[int, torch.Tensor]) -> None:
    n_ring_values = sorted(centers)
    n_invariants = int(centers[n_ring_values[0]]["index"].numel())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["invariant_index"]
        for n_ring in n_ring_values:
            header.extend(
                [
                    f"n{n_ring}_class0_mean",
                    f"n{n_ring}_class1_mean",
                    f"n{n_ring}_standardized_delta",
                    f"n{n_ring}_max_abs_feature_delta",
                ]
            )
        writer.writerow(header)
        for invariant in range(n_invariants):
            row = [invariant]
            for n_ring in n_ring_values:
                row.extend(
                    [
                        f"{float(centers[n_ring]['class0'][invariant]):.8g}",
                        f"{float(centers[n_ring]['class1'][invariant]):.8g}",
                        f"{float(centers[n_ring]['delta'][invariant]):.8g}",
                        f"{float(feature_deltas[n_ring][:, invariant].abs().max()):.8g}",
                    ]
                )
            writer.writerow(row)


def plot_overlay(
    path: Path,
    centers: dict[int, dict[str, torch.Tensor]],
    feature_deltas: dict[int, torch.Tensor],
    *,
    log_y: bool = False,
) -> None:
    n_invariants = int(next(iter(centers.values()))["index"].numel())
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    for n_ring, data in centers.items():
        x = data["index"]
        axes[0].plot(
            x,
            data["class0"],
            marker="o",
            color=COLORS[n_ring],
            linestyle="-",
            linewidth=2.1,
            markersize=8.0,
            label=f"{run_label(n_ring)}, class 0",
        )
        axes[0].plot(
            x,
            data["class1"],
            marker="s",
            color=COLORS[n_ring],
            linestyle="--",
            linewidth=2.1,
            markersize=8.0,
            label=f"{run_label(n_ring)}, class 1",
        )
        axes[1].plot(
            x,
            data["delta"],
            marker="o",
            color=COLORS[n_ring],
            linewidth=2.2,
            markersize=8.5,
            label=run_label(n_ring),
        )
        axes[2].plot(
            x,
            feature_deltas[n_ring].abs().max(dim=0).values,
            marker="o",
            color=COLORS[n_ring],
            linewidth=2.2,
            markersize=8.5,
            label=run_label(n_ring),
        )

    scale_suffix = " (symlog y scale)" if log_y else ""
    axes[0].set_title(f"Center-node invariant means, averaged over feature channels{scale_suffix}")
    axes[0].set_ylabel("mean invariant value")
    axes[0].legend(ncol=2)

    axes[1].set_title(f"Class separation after averaging feature channels{scale_suffix}")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("standardized class delta")
    axes[1].legend()

    axes[2].set_title(f"Strongest feature-specific class separation per invariant{scale_suffix}")
    axes[2].set_ylabel("max |standardized delta|")
    axes[2].set_xlabel("invariant index")
    axes[2].legend()

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xticks(torch.arange(n_invariants))
        if log_y:
            axis.set_yscale("symlog", linthresh=1.0e-3, linscale=0.6)
            axis.set_ylim(bottom=0.0)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def dataset_row(n_ring: int, capture_shape: str) -> str:
    nodes_per_graph = 1 + 2 * n_ring
    total_nodes = nodes_per_graph * 100
    directed_edges = 4 * n_ring
    sector = 360.0 / n_ring
    label0_high = 0.45 * sector
    offset = 0.5 * sector
    return (
        f"| {run_label(n_ring)} | {nodes_per_graph} | {total_nodes} | {2 * n_ring} | "
        f"{directed_edges} | {sector:g} deg | 0 to {label0_high:g} deg | {offset:g} deg | "
        f"{offset:g} to {offset + label0_high:g} deg | `(100, {nodes_per_graph})` | "
        f"`(100, {nodes_per_graph}, 3)` | `(100, 2, {directed_edges})` | `{capture_shape}` |"
    )


def top_feature_text(feature_delta: torch.Tensor, limit: int = 4) -> str:
    max_by_invariant = feature_delta.abs().max(dim=0).values
    top = torch.argsort(max_by_invariant, descending=True)[: min(limit, max_by_invariant.numel())]
    return ", ".join(f"I{int(i)}={float(max_by_invariant[i]):.3g}" for i in top)


def write_report(
    path: Path,
    *,
    overlay_csv: Path,
    overlay_png: Path,
    overlay_log_png: Path | None,
    args: argparse.Namespace,
    summaries: dict[int, dict[str, str]],
    centers: dict[int, dict[str, torch.Tensor]],
    feature_deltas: dict[int, torch.Tensor],
) -> None:
    n_ring_values = sorted(summaries)
    run_list = ", ".join(str(n_ring) for n_ring in n_ring_values)
    title = " vs ".join(f"{n_ring}-Node" for n_ring in n_ring_values)
    training_rows = "\n".join(
        f"| {run_label(n_ring)} | {summaries[n_ring]['epochs']} | {summaries[n_ring]['loss']} | "
        f"{summaries[n_ring]['accuracy']} | {summaries[n_ring]['margin_accuracy']} |"
        for n_ring in n_ring_values
    )
    dataset_rows = "\n".join(
        dataset_row(n_ring, summaries[n_ring]["capture_shape"])
        for n_ring in n_ring_values
    )
    top_rows = "\n".join(
        f"| {run_label(n_ring)} | {top_feature_text(feature_deltas[n_ring])} | "
        f"{top_feature_text(centers[n_ring]['delta'].abs().view(1, -1))} |"
        for n_ring in n_ring_values
    )

    path.write_text(
        f"""# {title} Rotating-Ring Invariants

This compares HIP-HOP invariant runs for {run_list} inner/outer ring-node settings on the `show-invariants` branches of both local repos.

## Shared Model Parameters

| parameter | value |
| --- | --- |
| model | HIP-HOP |
| `l_max` | `{args.l_max}` |
| `n_max` | `{args.n_max}` |
| interaction layers | `1` |
| atom layers | `1` |
| feature channels | `{args.n_features}` |
| sensitivity functions | `{args.n_sensitivities}` |
| neighborhood cutoff mode | `edges` |
| epochs requested | `{args.epochs_requested}` |
| learning rate | `{args.learning_rate:g}` |
| graph count | `100` |
| class balance | `50` aligned, `50` interleaved |
| seed | `0` |
| outer 3D tilt | `0` degrees |
| global rotation | `0` |

## Dataset Differences

| run | nodes per graph | total real nodes | undirected edges per graph | directed edges per graph | one inner-ring sector | label 0 outer phase range | label 1 phase offset | label 1 outer phase range | `Z` shape | `R` shape | `edge_indices` shape | invariant capture shape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
{dataset_rows}

## Training Outcome

| run | epochs used | loss | accuracy | margin accuracy |
| --- | ---: | ---: | ---: | ---: |
{training_rows}

## Strongest Invariant Positions

| run | strongest feature-specific positions | strongest feature-averaged positions |
| --- | --- | --- |
{top_rows}

## Overlay Outputs

- Linear-y overlay plot: `{overlay_png}`
{f"- Symlog-y overlay plot: `{overlay_log_png}`" if overlay_log_png is not None else ""}
- Overlay CSV: `{overlay_csv}`

## Main Read

The bottom panel is usually the most useful one: it keeps the learned feature-channel axis and reports the strongest class separation available for each invariant index. The middle panel averages across feature channels, which can hide useful feature-specific signal.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_root / "show_invariants_overlay"
    output_dir.mkdir(parents=True, exist_ok=True)

    centers = {}
    feature_deltas = {}
    summaries = {}
    for n_ring, dirname in RUN_DIRS.items():
        run_dir = args.results_root / dirname
        centers[n_ring] = read_center_csv(run_dir / "layer0_center_invariants.csv")
        feature_deltas[n_ring] = read_feature_csv(run_dir / "layer0_feature_invariant_deltas.csv")
        summaries[n_ring] = read_summary(run_dir / "summary.txt")

    overlay_name = "_".join(f"n{n_ring}" for n_ring in sorted(RUN_DIRS)) + "_invariant_overlay"
    overlay_csv = output_dir / f"{overlay_name}.csv"
    overlay_png = output_dir / f"{overlay_name}.png"
    overlay_log_png = output_dir / f"{overlay_name}_logy.png" if args.write_log_y else None
    overlay_report = output_dir / f"{overlay_name}.md"

    write_overlay_csv(overlay_csv, centers, feature_deltas)
    plot_overlay(overlay_png, centers, feature_deltas)
    if overlay_log_png is not None:
        plot_overlay(overlay_log_png, centers, feature_deltas, log_y=True)
    write_report(
        overlay_report,
        overlay_csv=overlay_csv,
        overlay_png=overlay_png,
        overlay_log_png=overlay_log_png,
        args=args,
        summaries=summaries,
        centers=centers,
        feature_deltas=feature_deltas,
    )

    print(f"wrote {overlay_png}")
    if overlay_log_png is not None:
        print(f"wrote {overlay_log_png}")
    print(f"wrote {overlay_csv}")
    print(f"wrote {overlay_report}")


if __name__ == "__main__":
    main()
