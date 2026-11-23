"""Run the journal's GNN screening, out-of-sample tests, and recourse comparison."""

import json
import os
import random
import sys
import time
from functools import lru_cache
from itertools import product
from multiprocessing import get_context
from statistics import mean

from solver import (
    CONSENSUS_LAMBDA, DISTRIBUTIONS, FAMILIES, TARGETS,
    build_feasibility_fallback_candidates, build_gnn_candidates, build_graph,
    build_greedy_scenario_states, build_scenario_node_sets, ensure_solver_metadata,
    evaluate_candidates, evaluate_fixed_n1_candidate, find_benchmark,
    get_reference_selected_n1_node_ids, get_selected_n1_node_ids_from_state,
    improve_fixed_set, parse_selected_instance, predict_probabilities, read_csv,
    read_instance, routes_from_state, score_routes, train_model, write_csv,
)

OOS_CSV = "oos_results.csv"
OUTPUT_FILES = {
    "oos": OOS_CSV,
    "recourse-budget": "recourse_budget_results.csv",
    "screen": "gnn_configuration_screen.csv",
    "validation": "gnn_configuration_validation.csv",
}

# Keep the numerical libraries single-threaded in each training process.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def run_configuration_test(args, targets):
    screen = args.experiment == "screen"
    configurations = SCREENING_GRID if screen else SHORTLIST
    rows = []
    for configuration in configurations:
        config_rows = []
        for distribution, target in targets:
            print(f"{args.experiment}: {distribution} {target}, {configuration}", flush=True)
            config_rows.append(fit_fold(args.instances_dir, "STD", "STD", distribution, target,
                                        configuration, construct=not screen))
        if screen:
            score = mean(row["average_precision"] for row in config_rows)
            for row in config_rows:
                row["configuration_mean_ap"] = score
        rows.extend(config_rows)
    if screen:
        rows.sort(key=lambda row: row["configuration_mean_ap"], reverse=True)
    return rows

def run_std_var_oos(args, family, targets, pool):
    count = args.scenario_count or 384
    scenarios = range(129, 129 + count)
    profile = "original" if family == "STD" else "fast"
    rows = []
    for distribution, instance in targets:
        print(f"{family}: training for {distribution} instance {instance}", flush=True)
        fold = fit_fold(args.instances_dir, "STD", family, distribution, instance,
                        SELECTED_CONFIGURATION, construct=True)
        selected = [int(node) for node in fold["selected_ids"].split()]
        row = {
            "family": family, "distribution": distribution, "instance": instance,
            "benchmark_scenarios": 512, "scenarios": count, "seeds": 1,
            "average_precision": fold["average_precision"], "p_ref": fold["p_ref"],
            "gnn16_construction_seconds": fold["construction_seconds"],
        }
        if family == "VAR":
            within = fit_fold(args.instances_dir, "VAR", "VAR", distribution, instance,
                              SELECTED_CONFIGURATION, construct=False)
            row.update(within_var_ap=within["average_precision"], within_var_p_ref=within["p_ref"])

        benchmark, sets = fixed_sets(args.instances_dir, family, distribution, instance, selected)
        for method in STD_VAR_METHODS:
            print(f"{family}: evaluating {method}, {distribution} instance {instance}", flush=True)
            cases, elapsed_per_scenario = evaluate_policy(
                pool, family, distribution, instance, benchmark, method, sets[method], scenarios, profile,
            )
            row[f"{method.lower()}_profit"] = mean(case["fast_profit"] for case in cases)
            row[f"{method.lower()}_oos_seconds"] = (elapsed_per_scenario if family == "STD"
                                                   else mean(case["fast_seconds"] for case in cases))
        row["fixed_sets"] = json.dumps(sets, separators=(",", ":"))
        rows.append(row)
    return rows

def run_recourse_budget(args, targets, pool):
    source = (args.input_dir or args.output_dir) / OOS_CSV
    if not source.is_file():
        raise ValueError(f"Run STD and VAR OOS first to write the fixed sets to {source}.")
    saved = {(row["family"], row["distribution"], int(row["instance"])): row
             for row in read_csv(source)}
    families = (args.family,) if args.family else ("STD", "VAR")
    keys = [(family, distribution, instance) for family in families for distribution, instance in targets]
    for key in keys:
        row = saved.get(key)
        if row is None or not row.get("fixed_sets"):
            raise ValueError(f"Missing fixed sets for {key}. Run OOS for these targets first.")

    rows = []
    scenarios = range(129, 129 + (args.scenario_count or 384))
    for family, distribution, instance in keys:
        sets = json.loads(saved[family, distribution, instance]["fixed_sets"])
        benchmark, current_sets = fixed_sets(args.instances_dir, family, distribution, instance, sets["GNN16"])
        if current_sets != sets:
            raise ValueError(f"Reference selections changed for {family} {distribution} instance {instance}.")
        evaluated = {}
        for method in STD_VAR_METHODS:
            selected = tuple(sets[method])
            if selected not in evaluated:
                print(f"budget: {family} {distribution} instance {instance}, {method}", flush=True)
                evaluated[selected], _ = evaluate_policy(
                    pool, family, distribution, instance, benchmark, method, selected, scenarios, "budget",
                )
            cases = evaluated[selected]
            row = {"family": family, "distribution": distribution, "instance": instance,
                   "method": method, "scenarios": len(cases)}
            for column in ("fast_profit", "intensive_profit", "fast_seconds", "intensive_seconds"):
                row[column] = mean(case[column] for case in cases)
            rows.append(row)
    return rows

def lrg_target_result(distribution, target, ranking, construction, evaluations):
    gnn_rows, p128_rows, p16_rows = [], [], []
    for _, group in evaluations:
        for method, row in group["methods"].items():
            if method == "P128":
                p128_rows.append(row)
            elif method == "P16":
                p16_rows.append(row)
            else:
                gnn_rows.append(row)
    return {
        "family": "LRG", "distribution": distribution, "instance": target,
        "benchmark_scenarios": 128, "scenarios": len(evaluations), "seeds": len(ranking),
        "average_precision": mean(row["average_precision"] for row in ranking),
        "p_ref": mean(row["p_ref"] for row in ranking),
        "gnn16_profit": mean(row["profit"] for row in gnn_rows),
        "p128_profit": mean(row["profit"] for row in p128_rows),
        "gnn16_construction_seconds": mean(
            rank["train_seconds"] + rank["predict_seconds"] + result["construction_seconds"]
            for rank, result in zip(ranking, construction)
        ),
        "gnn16_oos_seconds": mean(row["seconds"] for row in gnn_rows),
        "p16_oos_seconds": mean(row["seconds"] for row in p16_rows),
        "p128_oos_seconds": mean(row["seconds"] for row in p128_rows),
    }

def run_lrg_experiment(args, targets, pool):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    count = args.scenario_count or 112
    files = {}
    for distribution, _ in targets:
        if distribution not in files:
            files[distribution] = (
                find_benchmark(args.instances_dir, "LRG", distribution, 128),
                find_benchmark(args.instances_dir, "LRG", distribution, 16),
            )

    # Save costly intermediate routes so interrupted LRG runs can resume.
    checkpoint = args.output_dir / "lrg_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    manifest_path = checkpoint / "manifest.json"
    inputs = {}
    for pair in files.values():
        for path in pair:
            stat = path.stat()
            inputs[path.name] = [stat.st_size, stat.st_mtime_ns]
    manifest = {
        "inputs": inputs,
        "targets": [list(target) for target in targets], "seeds": args.seeds, "scenarios": count,
        "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
    }
    if manifest_path.is_file():
        if not args.resume or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("LRG checkpoints already exist or differ. Use a fresh output directory.")
    else:
        write_json(manifest_path, manifest)

    # Keep PyTorch in the parent. Spawned workers run only the routing code.
    tasks = []
    for distribution, target in targets:
        p128, p16 = files[distribution]
        ranking = rank_target(distribution, target, p128, checkpoint, args.seeds)
        tasks.append((distribution, target, p128, p16, checkpoint, ranking, count))

    results = map(route_target, tasks) if pool is None else pool.map(route_target, tasks)
    return [lrg_target_result(*result) for result in results]

def run_experiment(args):
    distributions = [args.distribution] if args.distribution else DISTRIBUTIONS
    instances = [args.instance] if args.instance else TARGETS
    targets = list(product(distributions, instances))
    experiment = args.experiment
    output = args.output_dir / OUTPUT_FILES[experiment]
    lock = output.with_suffix(".lock")
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise ValueError(f"Another run may be writing {output}. Choose a separate output directory.") from None
    pool = None
    try:
        rows = read_csv(output) if output.is_file() else []
        families = (args.family,) if args.family else FAMILIES
        if experiment in ("oos", "recourse-budget"):
            keys = {(family, distribution, target) for family in families for distribution, target in targets}
            overlap = any((row["family"], row["distribution"], int(row["instance"])) in keys for row in rows)
        else:
            overlap = bool(rows)
        if overlap:
            raise ValueError(f"Requested rows already exist in {output}. Choose a fresh output directory.")

        if experiment in ("screen", "validation"):
            rows.extend(run_configuration_test(args, targets))
        else:
            if args.workers > 1:
                pool = get_context("spawn").Pool(args.workers)
            if experiment == "recourse-budget":
                rows.extend(run_recourse_budget(args, targets, pool))
            else:
                for family in families:
                    if family == "LRG":
                        rows.extend(run_lrg_experiment(args, targets, pool))
                    else:
                        rows.extend(run_std_var_oos(args, family, targets, pool))
                    # Keep finished families if later runs are interrupted.
                    if family != families[-1]:
                        write_csv(output, rows)
            rows.sort(key=lambda row: (FAMILIES.index(row["family"]),
                                       DISTRIBUTIONS.index(row["distribution"]), int(row["instance"])))
        write_csv(output, rows)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        lock.unlink()
    print(f"Wrote {len(rows)} rows to {output}.")

# ===== STD and VAR experiments =====

STD_VAR_METHODS = ("P16", "P128", "P512", "GNN16")
SELECTED_CONFIGURATION = (5, 64, 4, 0.0)
SHORTLIST = (
    SELECTED_CONFIGURATION,
    (5, 128, 3, 0.0),
    (5, 128, 3, 0.10),
    (5, 64, 3, 0.0),
    (10, 64, 3, 0.10),
)
SCREENING_GRID = tuple(product((5, 10), (64, 128), (3, 4), (0.0, 0.10)))

def load_instance(data_dir, family, distribution, scenarios, instance, distances):
    path = find_benchmark(data_dir, family, distribution, scenarios)
    if distances:
        result = parse_selected_instance(path, instance)
    else:
        result = read_instance(path, instance)
    if set(result["scenarios_n2"]) != set(range(1, scenarios + 1)):
        raise ValueError(f"Unexpected scenario IDs in {path}, instance {instance}")
    return result

@lru_cache(maxsize=40)
def training_instance(data_dir, family, distribution, instance):
    return load_instance(data_dir, family, distribution, 16, instance, distances=True)

def fit_fold(data_dir, source, target, distribution, instance, config, construct):
    # Load PyTorch only for training so spawned routing workers stay lightweight.
    import numpy as np
    import torch

    torch.set_num_threads(1)
    neighbors, hidden_dim, layers, dropout = config
    seed = 11 + 17 * instance
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Hold out the matching instance ID in both within-family and transfer folds.
    training_graphs = []
    for other in TARGETS:
        if other == instance:
            continue
        data = training_instance(data_dir, source, distribution, other)
        ensure_solver_metadata(data, 0, 0)
        training_graphs.append(build_graph(other, data, neighbors, True))

    target_data = training_instance(data_dir, target, distribution, instance)
    ensure_solver_metadata(target_data, 0, 0)
    target_graph = build_graph(instance, target_data, neighbors, True)
    features = np.vstack([graph["features"] for graph in training_graphs])
    feature_mean = features.mean(axis=0)
    feature_std = np.maximum(features.std(axis=0), 1e-6)
    device = torch.device("cpu")
    started = time.perf_counter()
    model = train_model(training_graphs, feature_mean, feature_std, device,
                        seed=seed, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    train_seconds = time.perf_counter() - started
    probabilities = predict_probabilities(model, target_graph, feature_mean, feature_std, device)
    order = np.argsort(-probabilities.astype(float), kind="stable")
    labels = target_graph["labels"][order].astype(int)
    reference_size = int(labels.sum())
    if reference_size == 0:
        raise ValueError(f"Empty P16 selection for {target} {distribution} instance {instance}.")
    precision_at_rank = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    row = {
        "family": target, "distribution": distribution, "instance": instance,
        "neighbors": neighbors, "hidden_dim": hidden_dim, "layers": layers, "dropout": dropout,
        "average_precision": float((precision_at_rank * labels).sum() / reference_size),
        "p_ref": float(labels[:reference_size].mean()),
        "train_seconds": train_seconds,
    }
    if not construct:
        return row

    # Use labels to score the prediction above. Routing receives no stored routes.
    if target == "VAR":
        # The VAR experiment read its saved probability column as double precision.
        probabilities = probabilities.astype(float)
    blind_data = {**target_data, "solutions": {
        scenario: {**metadata, "routes": {}}
        for scenario, metadata in target_data["solutions"].items()
    }}
    blind_graph = {**target_graph, "instance_data": blind_data, "labels": None}
    candidates = build_gnn_candidates(blind_graph, probabilities, training_graphs, 5)
    fallback = build_feasibility_fallback_candidates(blind_graph, probabilities, candidates)
    started = time.perf_counter()
    best, initially_feasible = evaluate_candidates(blind_graph, candidates, seed, fallback)
    candidate_seconds = time.perf_counter() - started
    if not best["is_feasible"]:
        raise RuntimeError(f"Construction failed for {target} {distribution} instance {instance}")
    row.update(
        initial_feasible=initially_feasible,
        construction_profit=best["expected_profit"],
        construction_seconds=train_seconds + candidate_seconds,
        selected_ids=" ".join(map(str, best["selected_n1_node_ids"])),
    )
    return row

def fixed_sets(data_dir, family, distribution, instance, selected):
    policies = {"GNN16": selected}
    layout = None
    for count in (16, 128, 512):
        benchmark = load_instance(data_dir, family, distribution, count, instance, distances=False)
        choices = get_reference_selected_n1_node_ids(benchmark)
        if not choices["all_same"] or not choices["common"]:
            raise ValueError(f"{family} {distribution} P{count} instance {instance} has no common stored set.")
        if layout is not None and benchmark["n1_nodes"] != layout:
            raise ValueError(f"Inconsistent first-stage layout for {family} {distribution} instance {instance}.")
        layout = benchmark["n1_nodes"]
        policies[f"P{count}"] = choices["common"]
    return benchmark, policies

def evaluate_std_var_scenario(task):
    family, distribution, instance, method, scenario, selected, raw, profile = task
    case_started = time.perf_counter()
    scene = build_scenario_node_sets({instance: raw})[instance]
    if profile == "original":
        # STD Table 3 uses the original one-pass records and their method-specific seeds.
        method_rank = STD_VAR_METHODS.index(method) + 1
        seed = 11 + instance * 10007 + method_rank * 9173 + scenario * 1009
        options = {}
    else:
        family_seed = 0 if family == "STD" else 10_000_000
        distribution_seed = 0 if distribution == "uniform" else 1_000_000
        seed = 20261006 + family_seed + distribution_seed + instance * 10007 + scenario * 1009
        options = {"lower_stagnation_limit": 10, "replacement_candidate_limit": 20}

    started = time.perf_counter()
    result = evaluate_fixed_n1_candidate(
        scene, selected, method, CONSENSUS_LAMBDA, 1, random.Random(seed), **options,
    )
    routes = routes_from_state(result["scenario_states"][scenario])
    initial = score_routes(raw, scenario, routes, selected)
    fast_seconds = time.perf_counter() - started
    if not initial["feasible"] or initial["profit"] != result["scenario_totals"][scenario]["total_profit"]:
        raise RuntimeError(f"Route audit failed for {family} {distribution} instance {instance}, "
                           f"{method}, scenario {scenario}: {initial['errors']}")
    row = {"fast_profit": initial["profit"], "fast_seconds": fast_seconds}
    if profile == "budget":
        improved = improve_fixed_set(
            scene, raw, scenario, selected, routes, seed + 500_000_000,
            passes=3, starts=2, seconds=120, replacement_limit=50,
        )
        row.update(intensive_profit=improved["profit"], intensive_seconds=time.perf_counter() - case_started)
    return row

def evaluate_policy(pool, family, distribution, instance, benchmark, method, selected, scenarios, profile):
    tasks = []
    for scenario in scenarios:
        metadata = benchmark["solutions"][scenario]
        blind = {
            "instance_id": instance, "num_scenarios": 1, "n1_nodes": benchmark["n1_nodes"],
            "scenarios_n2": {scenario: benchmark["scenarios_n2"][scenario]},
            "solutions": {scenario: {**metadata, "routes": {}}},
        }
        tasks.append((family, distribution, instance, method, scenario, selected, blind, profile))
    started = time.perf_counter()
    if pool is None:
        rows = list(map(evaluate_std_var_scenario, tasks))
    else:
        rows = pool.map(evaluate_std_var_scenario, tasks, chunksize=4)
    elapsed = time.perf_counter() - started
    return rows, elapsed / len(rows)

# ===== LRG experiments =====

CONSTRUCTION_SCENARIOS = range(1, 17)
EVALUATION_SCENARIOS = range(17, 129)

def read_lrg_instance(path, target, scenario_count):
    # Keep raw node records and avoid building all 128 distance matrices here.
    raw = read_instance(path, target)
    expected = set(range(1, scenario_count + 1))
    if (raw["num_scenarios"] != scenario_count
            or set(raw["scenarios_n2"]) != expected
            or set(raw["solutions"]) != expected):
        raise ValueError(f"{path.name}, target {target}: unexpected scenario numbers")
    return raw

def reference_selection(raw):
    reference = get_reference_selected_n1_node_ids(raw)
    if not reference["all_same"] or not reference["common"]:
        raise ValueError(f"Target {raw['instance_id']}: expected a common reference selection")
    return reference["common"]

def scenario_subset(raw, scenario_ids):
    # Preserve the problem data while removing stored vehicle routes from search.
    solutions = {}
    for scenario in scenario_ids:
        solution = raw["solutions"][scenario]
        solutions[scenario] = {
            "time_limit": solution["time_limit"],
            "vehicles": solution["vehicles"],
            "routes": {},
        }
    return {
        "instance_id": raw["instance_id"],
        "num_scenarios": len(scenario_ids),
        "n1_nodes": raw["n1_nodes"],
        "scenarios_n2": {s: raw["scenarios_n2"][s] for s in scenario_ids},
        "scenarios": {s: {} for s in scenario_ids},
        "solutions": solutions,
    }

@lru_cache(maxsize=2)
def lrg_training_graphs(p128_path):
    """Build graph features from scenarios 1–16 and labels from P128 selections."""
    import numpy as np

    graphs = {}
    seen_customers = set()
    for target in TARGETS:
        raw = read_lrg_instance(p128_path, target, 128)
        customers = {(n["x"], n["y"], n["profit"]) for n in raw["n1_nodes"] if n["node_id"]}
        if seen_customers.intersection(customers):
            raise ValueError(f"{p128_path.name}: shared first-stage records across targets")
        seen_customers.update(customers)
        observed = scenario_subset(raw, CONSTRUCTION_SCENARIOS)
        graph = build_graph(target, observed, 5, False)
        selected = set(reference_selection(raw))
        graph["labels"] = np.array(
            [node["node_id"] in selected for node in graph["nodes"]], dtype=np.float32
        )
        graphs[target] = graph
    return graphs

def train_and_rank(target, repeat, graph, train_graphs):
    import numpy as np
    import torch

    seed = 11 + 17 * target + 100000 * repeat
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    features = np.vstack([g["features"] for g in train_graphs])
    feature_mean = features.mean(axis=0)
    feature_std = np.maximum(features.std(axis=0), 1e-6)
    device = torch.device("cpu")
    started = time.perf_counter()
    model = train_model(train_graphs, feature_mean, feature_std, device,
                        seed=seed, hidden_dim=64, layers=4, dropout=0.0)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = predict_probabilities(model, graph, feature_mean, feature_std, device)
    predict_seconds = time.perf_counter() - started

    # Target labels are used here to score the ranking after prediction.
    labels = graph["labels"].astype(int)
    order = np.argsort(-probabilities, kind="stable")
    ranked_labels = labels[order]
    precision = np.cumsum(ranked_labels) / np.arange(1, len(order) + 1)
    positives = int(labels.sum())

    # The original pipeline read predictions back from CSV as float64.
    probabilities = probabilities.astype(np.float64)
    candidates = build_gnn_candidates(graph, probabilities, train_graphs, 5)
    fallback = build_feasibility_fallback_candidates(graph, probabilities, candidates)
    return {
        "seed": seed,
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "average_precision": float((precision * ranked_labels).sum() / positives),
        "p_ref": float(ranked_labels[:positives].mean()),
        "node_ids": [node["node_id"] for node in graph["nodes"]],
        "probabilities": probabilities.tolist(),
        "candidates": list(candidates.values()),
        "fallback_candidates": list(fallback.values()),
    }

def routing_instance(raw, scenario):
    target = raw["instance_id"]
    instance = scenario_subset(raw, [scenario])
    return build_scenario_node_sets({target: instance})[target]

def evaluate_lrg_scenario(instance, scenario, selected, passes, greedy):
    seed = 20261005 + instance["instance_id"] * 10007 + scenario * 1009
    started = time.perf_counter()
    result = evaluate_fixed_n1_candidate(
        instance, selected, "400-node held-out validation", CONSENSUS_LAMBDA,
        passes, random.Random(seed), base_states=greedy,
        lower_stagnation_limit=10, replacement_candidate_limit=20,
    )
    seconds = time.perf_counter() - started
    state = result["scenario_states"][scenario]
    totals = result["scenario_totals"][scenario]
    actual_ids = set(get_selected_n1_node_ids_from_state(state, len(instance["n1_nodes"])))
    visits = [node for route in state["routes"] for node in route[1:-1]]
    feasible = (result["is_feasible"] and actual_ids == set(selected)
                and len(visits) == len(set(visits)) and totals["total_overtime"] == 0)
    return {
        "profit": totals["total_profit"], "objective": totals["total_objective"],
        "overtime": totals["total_overtime"], "feasible": bool(feasible), "seconds": seconds,
        "routes": [[state["nodes"][node]["node_id"] for node in route] for route in state["routes"]],
    }

def screen_candidate(raw, candidate, stop_on_failure=False):
    values = []
    for scenario in CONSTRUCTION_SCENARIOS:
        instance = routing_instance(raw, scenario)
        greedy = build_greedy_scenario_states(instance)
        value = evaluate_lrg_scenario(instance, scenario, candidate["selected_ids"], 0, greedy)
        values.append(value)
        if stop_on_failure and not value["feasible"]:
            break
    return {
        "sources": candidate["sources"],
        "selected_ids": candidate["selected_ids"],
        "feasible": len(values) == 16 and all(v["feasible"] for v in values),
        "mean_profit": mean(v["profit"] for v in values),
        "mean_objective": mean(v["objective"] for v in values),
    }

def construct(raw, candidates, fallback_candidates):
    started = time.perf_counter()
    screened = [screen_candidate(raw, candidate) for candidate in candidates]
    feasible = [result for result in screened if result["feasible"]]
    used_fallback = not feasible
    if used_fallback:
        for candidate in fallback_candidates:
            result = screen_candidate(raw, candidate, stop_on_failure=True)
            screened.append(result)
            if result["feasible"]:
                feasible.append(result)
                break
    if not feasible:
        raise RuntimeError(f"Target {raw['instance_id']}: no feasible first-stage candidate")
    best = max(feasible, key=lambda row: (
        row["mean_objective"], row["mean_profit"], -len(row["selected_ids"])
    ))
    return {
        "selected_ids": best["selected_ids"],
        "construction_profit": best["mean_profit"],
        "construction_seconds": time.perf_counter() - started,
        "used_fallback": used_fallback,
        "candidates": screened,
    }

def p16_selection(p16_path, p128_raw):
    p16_raw = read_lrg_instance(p16_path, p128_raw["instance_id"], 16)
    if p16_raw["n1_nodes"] != p128_raw["n1_nodes"]:
        raise ValueError("P16 and P128 first-stage node records differ")
    for scenario in CONSTRUCTION_SCENARIOS:
        if p16_raw["scenarios_n2"][scenario] != p128_raw["scenarios_n2"][scenario]:
            raise ValueError(f"P16 and P128 nodes differ on scenario {scenario}")
        for key in ("time_limit", "vehicles"):
            if p16_raw["solutions"][scenario][key] != p128_raw["solutions"][scenario][key]:
                raise ValueError(f"P16 and P128 metadata differ on scenario {scenario}")
    return reference_selection(p16_raw)

def read_or_run(path, run):
    if path.exists():
        return json.loads(path.read_text())
    value = run()
    write_json(path, value)
    return value

def evaluate_policies(raw, scenario, policies):
    instance = routing_instance(raw, scenario)
    greedy = build_greedy_scenario_states(instance)
    evaluations = {}
    previous_selections = {}
    for method, selected in policies.items():
        # The original GNN/P128 evaluation reuses routing for identical policies.
        # P16 was measured separately for the timing comparison.
        key = tuple(sorted(selected))
        if method == "P16" or key not in previous_selections:
            previous_selections[key] = evaluate_lrg_scenario(instance, scenario, selected, 1, greedy)
        evaluations[method] = previous_selections[key]
        if not evaluations[method]["feasible"]:
            raise RuntimeError(f"Target {raw['instance_id']}, scenario {scenario}: infeasible {method}")
    stored_profit = sum(r["metrics"]["profit"] for r in raw["solutions"][scenario]["routes"].values())
    return {"methods": evaluations, "p128_saved_profit": stored_profit}

def rank_target(distribution, target, p128_path, output, repeats):
    folder = output / f"{distribution}_{target}"
    folder.mkdir(exist_ok=True)
    print(f"{distribution} target {target}: ranking", flush=True)
    graphs = lrg_training_graphs(p128_path)
    train_graphs = [graphs[i] for i in TARGETS if i != target]
    target_graph = graphs[target]
    ranking = []
    for repeat in range(repeats):
        row = read_or_run(folder / f"ranking_{repeat}.json",
                          lambda: train_and_rank(target, repeat, target_graph, train_graphs))
        ranking.append(row)
    return ranking

def route_target(task):
    distribution, target, p128_path, p16_path, output, ranking, oos_count = task
    folder = output / f"{distribution}_{target}"
    raw = read_lrg_instance(p128_path, target, 128)
    reference_p16 = p16_selection(p16_path, raw) if p16_path else None
    policies = {}
    construction = []
    print(f"{distribution} target {target}: first-stage screening", flush=True)
    for repeat, row in enumerate(ranking):
        result = read_or_run(folder / f"construction_{repeat}.json",
                             lambda: construct(raw, row["candidates"], row["fallback_candidates"]))
        construction.append(result)
        policies[f"GNN{repeat}"] = result["selected_ids"]
    policies["P128"] = reference_selection(raw)
    if reference_p16:
        policies["P16"] = reference_p16

    evaluations = []
    print(f"{distribution} target {target}: evaluating scenarios 17–{16 + oos_count}", flush=True)
    for scenario in EVALUATION_SCENARIOS[:oos_count]:
        result = read_or_run(folder / f"evaluation_{scenario}.json",
                             lambda: evaluate_policies(raw, scenario, policies))
        evaluations.append((scenario, result))
        if len(evaluations) % 16 == 0:
            print(f"{distribution} target {target}: {len(evaluations)}/{oos_count} scenarios done", flush=True)
    return distribution, target, ranking, construction, evaluations
