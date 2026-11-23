"""Calculate benchmark summaries, paper tables, and the recourse figure."""

import os
import re
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from solver import DISTRIBUTIONS, FAMILIES, TARGETS, find_benchmark, read_csv, read_instance, write_csv

RESULTS = Path(__file__).resolve().parents[1] / "results"

def run_characteristics(args):
    """Summarize the instances used in Table 1."""
    import numpy as np

    def cv(values):
        values = np.asarray(values, dtype=float)
        return float(100 * values.std(ddof=0) / values.mean())

    def nearest_distances(points, neighbors, exclude_self=False):
        differences = points[:, None, :] - neighbors[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        if exclude_self:
            np.fill_diagonal(distances, np.inf)
        return distances.min(axis=1)

    summaries = []
    for family, count in (("STD", 512), ("VAR", 512), ("LRG", 128)):
        for distribution in DISTRIBUTIONS:
            path = find_benchmark(args.instances_dir, family, distribution, count)
            group = []
            for index in TARGETS:
                instance = read_instance(path, index)
                if (instance["num_scenarios"] != count
                    or set(instance["scenarios_n2"]) != set(range(1, count + 1))):
                    raise ValueError(f"Unexpected scenario count in {path.name}, instance {index}")
                if any(s["vehicles"] != 3 or s["time_limit"] != 2000 for s in instance["solutions"].values()):
                    raise ValueError(f"Unexpected vehicle count or time limit in {path.name}, instance {index}")
                depot = next(node for node in instance["n1_nodes"] if node["node_id"] == 0)
                origin = (depot["x"], depot["y"])
                customers = [node for node in instance["n1_nodes"] if node["node_id"] != 0]
                xy1 = np.array([(node["x"], node["y"]) for node in customers], dtype=float)
                p1 = np.array([node["profit"] for node in customers], dtype=float)
                static = {"n1_count": len(customers), "n1_prize_mean": float(p1.mean()),
                          "n1_prize_cv": cv(p1), "n1_prize_min": float(p1.min()),
                          "n1_prize_max": float(p1.max()),
                          "n1_depot_mean": float(np.linalg.norm(xy1 - origin, axis=1).mean())}
                within = []
                for _, nodes in sorted(instance["scenarios_n2"].items()):
                    xy2 = np.array([(node["x"], node["y"]) for node in nodes], dtype=float)
                    p2 = np.array([node["profit"] for node in nodes], dtype=float)
                    row = {"n2_count": len(nodes),
                           "n2_prize_mean": float(p2.mean()), "n2_prize_cv": cv(p2),
                           "n2_prize_total": float(p2.sum()),
                           "n2_depot_mean": float(np.linalg.norm(xy2 - origin, axis=1).mean()),
                           "n2_nearest_mean": float(nearest_distances(xy2, xy2, exclude_self=True).mean()),
                           "n1_nearest_n2_mean": float(nearest_distances(xy1, xy2).mean())}
                    within.append(row)
                for field in ("n2_count", "n2_prize_mean", "n2_prize_cv", "n2_depot_mean",
                              "n2_nearest_mean", "n1_nearest_n2_mean"):
                    static[field] = float(np.mean([row[field] for row in within]))
                static["n2_count_min"] = min(row["n2_count"] for row in within)
                static["n2_count_max"] = max(row["n2_count"] for row in within)
                static["n2_count_scenario_cv"] = cv([row["n2_count"] for row in within])
                static["n2_total_prize_scenario_cv"] = cv([row["n2_prize_total"] for row in within])
                group.append(static)
            summary = {"family": family, "distribution": distribution,
                       "instances": len(group), "scenarios": count}
            summary.update({field: float(np.mean([row[field] for row in group])) for field in group[0]})
            summary["n1_count_min"] = min(row["n1_count"] for row in group)
            summary["n1_count_max"] = max(row["n1_count"] for row in group)
            summary["n1_count_instance_cv"] = cv([row["n1_count"] for row in group])
            summary["n1_prize_min"] = min(row["n1_prize_min"] for row in group)
            summary["n1_prize_max"] = max(row["n1_prize_max"] for row in group)
            summary["n2_count_min"] = min(row["n2_count_min"] for row in group)
            summary["n2_count_max"] = max(row["n2_count_max"] for row in group)
            summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "instance_characteristics.csv"
    write_csv(output, summaries)
    print(f"Wrote characteristics for {len(summaries)} benchmark groups to {output}.")

# Tables 1–6 and the recourse-budget figure data

REPORT_METHODS = ("GNN16", "P16", "P128", "P512")
STUDENT_T_975_DF9 = 2.262157162740992

def targets_by_distribution(rows, name):
    scenario_count = {"STD": 384, "VAR": 384, "LRG": 112}.get(name)
    if scenario_count and any("scenarios" in row and int(row["scenarios"]) != scenario_count for row in rows):
        raise ValueError(f"{name}: expected {scenario_count} evaluation scenarios per target")
    if name == "LRG" and any("seeds" in row and int(row["seeds"]) != 3 for row in rows):
        raise ValueError("LRG: expected three training seeds per target")
    groups = {}
    for distribution in DISTRIBUTIONS:
        selected = [row for row in rows if row["distribution"] == distribution]
        instances = {int(row["instance"]) for row in selected}
        if len(selected) != 10 or instances != set(TARGETS):
            raise ValueError(f"{name}: expected targets 1–10 for {distribution}")
        groups[distribution] = sorted(selected, key=lambda row: int(row["instance"]))
    if len(rows) != 20:
        raise ValueError(f"{name}: expected exactly 20 targets")
    return groups

def column_mean(rows, column):
    return mean(float(row[column]) for row in rows)

def gap(value, reference):
    return 100 * (value / reference - 1)

def interval(values):
    center = mean(values)
    half_width = STUDENT_T_975_DF9 * stdev(values) / sqrt(len(values))
    return center, center - half_width, center + half_width

def benchmark_characteristics(rows):
    group_names = {"standard": "STD", "var": "VAR", "large": "LRG"}
    values = {(group_names.get(row["family"], row["family"]), row["distribution"]): row for row in rows}
    characteristics = (
        ("Available scenarios", "scenarios"),
        ("N1 count range", ("n1_count_min", "n1_count_max")),
        ("N1 count mean", "n1_count"),
        ("Across-instance CV of N1 count (%)", "n1_count_instance_cv"),
        ("N2 count range", ("n2_count_min", "n2_count_max")),
        ("N2 count mean", "n2_count"),
        ("Scenario CV of N2 count (%)", "n2_count_scenario_cv"),
        ("Mean N1 prize", "n1_prize_mean"),
        ("CV of N1 prizes (%)", "n1_prize_cv"),
        ("Mean N2 prize", "n2_prize_mean"),
        ("CV of N2 prizes (%)", "n2_prize_cv"),
        ("Scenario CV of total N2 prize (%)", "n2_total_prize_scenario_cv"),
        ("Mean N1–depot distance", "n1_depot_mean"),
        ("Mean N2–depot distance", "n2_depot_mean"),
        ("Mean N2 nearest-neighbor distance", "n2_nearest_mean"),
        ("Mean N1-to-nearest-N2 distance", "n1_nearest_n2_mean"),
    )
    table = []
    for title, column in characteristics:
        output = {"characteristic": title}
        for family in FAMILIES:
            for distribution in DISTRIBUTIONS:
                source = values[family, distribution]
                if isinstance(column, tuple):
                    lower, upper = column
                    value = f"{int(float(source[lower]))}–{int(float(source[upper]))}"
                elif column == "scenarios":
                    value = int(float(source[column]))
                else:
                    value = f"{float(source[column]):.1f}"
                output[f"{family} {distribution}"] = value
        table.append(output)
    return table

def configuration_results(rows):
    configurations = {}
    for row in rows:
        config = (int(row["neighbors"]), int(row["hidden_dim"]),
                  int(row["layers"]), float(row["dropout"]))
        configurations.setdefault(config, []).append(row)

    table = []
    for config, selected in configurations.items():
        targets_by_distribution(selected, f"configuration {config}")
        neighbors, hidden, layers, dropout = config
        initially_feasible = sum(row["initial_feasible"].strip().lower() == "true"
                                 for row in selected)
        table.append({
            "neighbors": neighbors,
            "hidden_dim": hidden,
            "layers": layers,
            "dropout": format(dropout, "g"),
            "mean_ap": f"{column_mean(selected, 'average_precision'):.4f}",
            "initially_feasible": f"{initially_feasible}/20",
            "mean_construction_profit": f"{column_mean(selected, 'construction_profit'):.1f}",
            "mean_train_seconds": f"{column_mean(selected, 'train_seconds'):.2f}",
            "mean_total_seconds": f"{column_mean(selected, 'construction_seconds'):.2f}",
        })
    if len(table) != 5:
        raise ValueError("Table 2 requires all five shortlisted configurations")
    return sorted(table, key=lambda row: float(row["mean_ap"]), reverse=True)

def performance_results(family, groups):
    table = []
    for distribution, rows in groups.items():
        output = {
            "distribution": distribution,
            "ap": f"{column_mean(rows, 'average_precision'):.4f}",
            "p_ref": f"{column_mean(rows, 'p_ref'):.4f}",
        }
        if family == "VAR":
            output["within_var_ap"] = f"{column_mean(rows, 'within_var_ap'):.4f}"
            output["within_var_p_ref"] = f"{column_mean(rows, 'within_var_p_ref'):.4f}"

        methods = ("GNN16", "P128") if family == "LRG" else REPORT_METHODS
        profits = {}
        for method in methods:
            profits[method] = column_mean(rows, f"{method.lower()}_profit")
            output[f"{method}_profit"] = f"{profits[method]:.1f}"

        references = ("P128",) if family == "LRG" else ("P16", "P512")
        for reference in references:
            # The current manuscript mixes paired gaps and gaps between means.
            # Preserve the convention of each printed cell.
            paired = family == "LRG" or (family, distribution, reference) == (
                "VAR", "clustered", "P512"
            )
            if paired:
                ref_column = f"{reference.lower()}_profit"
                value = mean(gap(float(row["gnn16_profit"]), float(row[ref_column]))
                             for row in rows)
            else:
                value = gap(profits["GNN16"], profits[reference])
            output[f"GNN16_gap_to_{reference}_percent"] = f"{value:+.2f}"
        table.append(output)
    return table

def timing_results(target_groups, author_rows):
    author_times = {(row["family"], row["distribution"], row["method"]):
                    float(row["seconds"]) for row in author_rows}
    table = []
    for clock in ("first_stage", "oos_per_scenario"):
        for family, groups in target_groups.items():
            for distribution, rows in groups.items():
                output = {"clock": clock, "family": family, "distribution": distribution}
                for method in REPORT_METHODS:
                    if family == "LRG" and method == "P512":
                        output[method] = ""
                    elif clock == "first_stage":
                        seconds = (column_mean(rows, "gnn16_construction_seconds")
                                   if method == "GNN16"
                                   else author_times[family, distribution, method])
                        output[method] = f"{seconds:.1f}"
                    else:
                        column = f"{method.lower()}_oos_seconds"
                        recorded = [float(row[column]) for row in rows if row[column] != ""]
                        if len(recorded) == 10:
                            output[method] = f"{mean(recorded):.2f}"
                        elif family == "LRG" and method == "P16" and not recorded:
                            # P16 is optional in the LRG rerun. Never fill its clock
                            # with a different run's OOS timing.
                            output[method] = ""
                        else:
                            raise ValueError(f"Incomplete OOS times for {family} {distribution} {method}")
                table.append(output)
    return table

def run_report(args):
    """Write Tables 1–6 from complete result CSVs."""
    input_dir = args.input_dir or RESULTS
    characteristics = input_dir / "instance_characteristics.csv"
    author_times = args.author_times
    if author_times is None:
        author_times = input_dir / "pisinger_times.csv"
        if not author_times.is_file():
            author_times = RESULTS / "pisinger_times.csv"

    tables = {
        "table1.csv": benchmark_characteristics(read_csv(characteristics)),
        "table2.csv": configuration_results(read_csv(input_dir / "gnn_configuration_validation.csv")),
    }
    oos = read_csv(input_dir / "oos_results.csv")
    target_groups = {}
    for table_number, family in enumerate(FAMILIES, start=3):
        rows = [row for row in oos if row["family"] == family]
        target_groups[family] = targets_by_distribution(rows, family)
        tables[f"table{table_number}.csv"] = performance_results(family, target_groups[family])
    tables["table6.csv"] = timing_results(target_groups, read_csv(author_times))
    output_dir = args.output_dir / "paper_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output_dir / name, rows)
    print(f"Wrote Tables 1–6 to {output_dir}.")

def run_plot(args):
    """Plot the matched fast and intensive recourse comparison."""
    output = args.output_dir / "recourse_budget.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budget = read_csv((args.input_dir or RESULTS) / "recourse_budget_results.csv")
    if any("scenarios" in row and int(row["scenarios"]) != 384 for row in budget):
        raise ValueError("Recourse budget figure requires 384 evaluation scenarios per target and policy")
    cases = {(row["family"], row["distribution"], int(row["instance"]), row["method"]): row
             for row in budget}
    if len(cases) != 160 or len(budget) != len(cases):
        raise ValueError("The recourse comparison requires 40 targets and four policies")

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), sharex=True)
    groups = [(family, distribution) for family in ("STD", "VAR") for distribution in DISTRIBUTIONS]
    references = REPORT_METHODS[1:]
    profiles = (("fast", -.12, "#777777", "white"), ("intensive", .12, "#176B9C", "#176B9C"))
    for index, (axis, (family, distribution)) in enumerate(zip(axes.flat, groups)):
        gnn = [cases[family, distribution, target, "GNN16"] for target in TARGETS]
        for profile, offset, color, fill in profiles:
            column = f"{profile}_profit"
            intervals = []
            for method in references:
                reference = [cases[family, distribution, target, method] for target in TARGETS]
                intervals.append(interval([gap(float(a[column]), float(b[column]))
                                           for a, b in zip(gnn, reference)]))
            means = [center for center, _, _ in intervals]
            low = [center - lower for center, lower, _ in intervals]
            high = [upper - center for center, _, upper in intervals]
            axis.errorbar(means, [i + offset for i in range(3)], xerr=[low, high], fmt="o",
                          color=color, markerfacecolor=fill, markersize=4, capsize=2.5,
                          label=f"{profile.capitalize()} recourse")
        axis.axvline(0, color="#333333", linewidth=.8, linestyle="--")
        axis.set_yticks(range(3), references)
        axis.set_ylim(2.62, -.72)
        title = "Standard" if family == "STD" else "VAR"
        axis.set_title(f"({chr(97 + index)}) {title}, {distribution}", loc="left", pad=18)
        gain = mean(gap(float(row["intensive_profit"]), float(row["fast_profit"])) for row in gnn)
        ratio = column_mean(gnn, "intensive_seconds") / column_mean(gnn, "fast_seconds")
        axis.text(0, 1.025, f"GNN profit {gain:+.2f}%, recourse time {ratio:.1f}×",
                  transform=axis.transAxes, fontsize=8, color="#444444")
        axis.grid(axis="x", color="#e6e6e6", linewidth=.6)
        if index >= 2:
            axis.set_xlabel("Mean paired GNN profit gap (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.subplots_adjust(left=.085, right=.985, bottom=.16, top=.89, hspace=.72, wspace=.2)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}.")

# Reference wall times

def run_author_times(args):
    """Extract mean construction wall times from the sixteen supplied solver logs."""
    if args.log_dir is None or not args.log_dir.is_dir():
        raise ValueError("Choose a folder containing the reference logs with --log-dir.")

    logs_by_name = {}
    for path in sorted(args.log_dir.rglob("pisinger-team-orienteering-*.txt")):
        logs_by_name.setdefault(path.name, []).append(path)
    rows = []
    for family in FAMILIES:
        for distribution in DISTRIBUTIONS:
            for scenarios in ((16, 128) if family == "LRG" else (16, 128, 512)):
                nodes = 400 if family == "LRG" else 100
                suffix = "-VAR" if family == "VAR" else ""
                name = f"pisinger-team-orienteering-{distribution}-{nodes}nodes-{scenarios}scen{suffix}.txt"
                logs = logs_by_name.get(name)
                if not logs:
                    raise ValueError(f"Missing log {name}")
                totals = set()
                for log in logs:
                    text = log.read_text(encoding="utf-8")
                    seconds = re.findall(r"^\s*totwalltime\s*=\s*([\d.]+)\s*$", text, re.MULTILINE)
                    instances = set(re.findall(r"^INSTANCE\s+(\d+)\s*$", text, re.MULTILINE))
                    if len(seconds) != 1 or instances != {str(target) for target in TARGETS}:
                        raise ValueError(f"Expected one totwalltime and ten instances in {log}")
                    totals.add(float(seconds[0]))
                if len(totals) != 1:
                    raise ValueError(f"Different timings in multiple copies of {name}")
                rows.append({"family": family, "distribution": distribution,
                             "method": f"P{scenarios}", "seconds": totals.pop()})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "pisinger_times.csv"
    write_csv(output, rows)
    print(f"Wrote {len(rows)} reference timing means to {output}.")
