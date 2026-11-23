"""Benchmark input, GNN training, and routing routines used by the experiments."""

import csv
import math
import random
import re
import time
from math import exp, sqrt
from pathlib import Path

FAMILIES = ("STD", "VAR", "LRG")
DISTRIBUTIONS = ("uniform", "clustered")
TARGETS = range(1, 11)

def find_benchmark(instances_dir, family, distribution, scenarios):
    path = Path(instances_dir) / f"{family.upper()}_{distribution}_{scenarios}scen.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark: {path}")
    return path

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

def write_csv(path, rows):
    """Keep one CSV per experiment and use blank cells for inapplicable fields."""
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

# Read benchmark instances

def parse_selected_instance(filepath: Path, instance_id: int):
    """Read an instance and build distance matrices for its scenarios."""
    return build_scenario_node_sets({instance_id: read_instance(filepath, instance_id)})[instance_id]

def read_instance(filepath: Path, instance_id: int):
    """Read customers and stored routes without building distance matrices."""
    data = None
    current_instance_id = None
    state = None
    current_scenario_id = None

    with filepath.open() as input_file:
        for raw_line in input_file:
            line = raw_line.strip().replace("\xa0", " ")
            if not line or line.startswith("#"):
                continue

            if line.startswith("instance ") and line.endswith(" start"):
                next_instance_id = int(line.split()[1])
                if (
                    data is not None
                    and current_instance_id == instance_id
                    and next_instance_id != instance_id
                ):
                    break
                current_instance_id = next_instance_id
                if current_instance_id == instance_id:
                    data = {
                        "instance_id": instance_id,
                        "num_scenarios": 0,
                        "n1_nodes": [],
                        "scenarios_n2": {},
                        "solutions": {},
                    }
                state = None
                continue

            if line.startswith("instance ") and line.endswith(" end"):
                if current_instance_id == instance_id:
                    state = "post_instance"
                    continue
                current_instance_id = None
                state = None
                current_scenario_id = None
                continue

            if current_instance_id != instance_id or data is None:
                continue

            if line.startswith("S "):
                data["num_scenarios"] = int(line.split()[1])
                continue

            if line.startswith("n1 "):
                state = "n1"
                continue

            if line.startswith("n2 "):
                current_scenario_id = int(line.split()[1])
                data["scenarios_n2"][current_scenario_id] = []
                state = "n2"
                continue

            if line.startswith("SOLUTION SCENARIO"):
                parts = line.split()
                current_scenario_id = int(parts[2])
                data["solutions"][current_scenario_id] = {
                    "scenario_id": current_scenario_id,
                    "time_limit": int(parts[5]),
                    "vehicles": int(parts[7]),
                    "routes": {},
                }
                state = "solution"
                continue

            if line.startswith("route ") and state == "solution":
                parts = line.split(":")
                route_id = int(parts[0].split()[1])
                route_nodes = [int(value) for value in parts[1].strip().split()]
                data["solutions"][current_scenario_id]["routes"][route_id] = {
                    "path": route_nodes,
                    "metrics": {},
                }
                continue

            if line.startswith("length ") and state == "solution":
                match = re.match(
                    r"length\s+(\d+)\s+of\s+(\d+)\s+prof\s+(\d+)\s+used\s+time\s+(\d+)\s+T\s+(\d+)",
                    line,
                )
                if match:
                    routes = data["solutions"][current_scenario_id]["routes"]
                    route_id = list(routes)[-1]
                    routes[route_id]["metrics"] = {
                        "length": int(match.group(1)),
                        "nodes_available": int(match.group(2)),
                        "profit": int(match.group(3)),
                        "used_time": int(match.group(4)),
                        "T": int(match.group(5)),
                    }
                continue

            parts = line.split()
            if len(parts) == 5 and all(part.lstrip("-").isdigit() for part in parts):
                node = {
                    "node_id": int(parts[1]),
                    "x": int(parts[2]),
                    "y": int(parts[3]),
                    "profit": int(parts[4]),
                }
                if state == "n1":
                    data["n1_nodes"].append(node)
                elif state == "n2":
                    data["scenarios_n2"][current_scenario_id].append(node)

    if data is None:
        raise ValueError(f"Instance {instance_id} not found in {filepath}")
    return data

# Build and improve vehicle routes

LAMBDA = 1.0
CONSENSUS_LAMBDA = 5.0
DSALNS_LOWER_PASSES = 3
LOWER_ALNS_REMOVE_MIN = 4
LOWER_ALNS_REMOVE_MAX = 14
LOWER_ALNS_ITERATIONS_PER_PASS = 35
LOWER_ALNS_STAGNATION_LIMIT = 0
LOWER_REPLACEMENT_CANDIDATE_LIMIT = 0
LOWER_ALNS_SEGMENT_LENGTH = 20
LOWER_ALNS_START_TEMPERATURE_FACTOR = 0.03
LOWER_ALNS_COOLING_RATE = 0.995
VEHICLES = 3
DESTROY_REPAIR_MAX_REMOVE = 8
ALNS_REACTION_FACTOR = 0.2

def deadline_reached(deadline):
    return deadline is not None and time.perf_counter() >= deadline

# Benchmark distances and input format

def calculate_distance_matrix(nodes):
    distance_matrix = []
    for node_i in nodes:
        row = []
        for node_j in nodes:
            dx = node_i["x"] - node_j["x"]
            dy = node_i["y"] - node_j["y"]
            row.append(round(sqrt(dx * dx + dy * dy)))
        distance_matrix.append(row)
    return distance_matrix

def build_scenario_node_sets(instances):
    for data in instances.values():
        data["scenarios"] = {}
        for scen_id, n2_nodes in data["scenarios_n2"].items():
            nodes = data["n1_nodes"] + n2_nodes
            data["scenarios"][scen_id] = {
                "nodes": nodes,
                "distance_matrix": calculate_distance_matrix(nodes)
            }
    return instances

def get_time_limit(instance_data, scenario_id):
    solution = instance_data["solutions"].get(scenario_id)
    if solution:
        return solution["time_limit"]
    return 0

# Greedy route construction and basic route operations

def calculate_route_travel_time(route, distance_matrix):
    if len(route) < 2:
        return 0

    travel_time = 0
    for current_index, next_index in zip(route, route[1:]):
        travel_time += distance_matrix[current_index][next_index]

    if route[-1] != route[0]:
        travel_time += distance_matrix[route[-1]][route[0]]

    return travel_time

def evaluate_solution(route, nodes, distance_matrix, time_limit, lambda_value):
    total_profit = sum(nodes[index]["profit"] for index in route)
    total_travel_time = calculate_route_travel_time(route, distance_matrix)
    overtime = max(0, total_travel_time - time_limit)
    objective = total_profit - lambda_value * overtime

    return {
        "objective": objective,
        "total_profit": total_profit,
        "total_travel_time": total_travel_time,
        "overtime": overtime
    }

def find_depot_index(nodes):
    for index, node in enumerate(nodes):
        if node["node_id"] == 0:
            return index
    return 0

def calculate_insertion_delta(route, position, node_index, distance_matrix):
    previous_index = route[position]
    next_index = route[position + 1]
    return (
        distance_matrix[previous_index][node_index]
        + distance_matrix[node_index][next_index]
        - distance_matrix[previous_index][next_index]
    )

def find_best_feasible_insertion(routes, route_times, unvisited, nodes, distance_matrix, time_limit):
    best_insertion = None

    for node_index in unvisited:
        profit = nodes[node_index]["profit"]
        if profit <= 0:
            continue

        for vehicle_index, route in enumerate(routes):
            for position in range(len(route) - 1):
                added_time = calculate_insertion_delta(route, position, node_index, distance_matrix)
                new_route_time = route_times[vehicle_index] + added_time
                if new_route_time > time_limit:
                    continue

                score = profit / max(added_time, 1)
                score_key = (
                    score,
                    profit,
                    -added_time,
                    -new_route_time,
                    -vehicle_index
                )

                if best_insertion is None or score_key > best_insertion["score_key"]:
                    best_insertion = {
                        "vehicle_index": vehicle_index,
                        "position": position,
                        "node_index": node_index,
                        "node": nodes[node_index],
                        "added_time": added_time,
                        "new_route_time": new_route_time,
                        "score": score,
                        "score_key": score_key
                    }

    return best_insertion

def build_parallel_greedy_insertion_solution(nodes, distance_matrix, time_limit, vehicle_count=VEHICLES):
    depot_index = find_depot_index(nodes)
    routes = [[depot_index, depot_index] for _ in range(vehicle_count)]
    route_times = [0 for _ in range(vehicle_count)]
    unvisited = set(range(len(nodes)))
    unvisited.remove(depot_index)
    insertion_steps = []

    while True:
        best_insertion = find_best_feasible_insertion(
            routes,
            route_times,
            unvisited,
            nodes,
            distance_matrix,
            time_limit
        )

        if best_insertion is None:
            break

        vehicle_index = best_insertion["vehicle_index"]
        insert_position = best_insertion["position"] + 1
        node_index = best_insertion["node_index"]

        routes[vehicle_index].insert(insert_position, node_index)
        route_times[vehicle_index] = best_insertion["new_route_time"]
        unvisited.remove(node_index)

        insertion_steps.append({
            "step": len(insertion_steps) + 1,
            "vehicle_index": vehicle_index,
            "node": best_insertion["node"],
            "added_time": best_insertion["added_time"],
            "route_time_after": route_times[vehicle_index],
            "score": best_insertion["score"]
        })

    vehicle_solutions = []
    for route in routes:
        metrics = evaluate_solution(route, nodes, distance_matrix, time_limit, LAMBDA)
        vehicle_solutions.append({
            "route": route,
            "nodes": [nodes[index] for index in route],
            "metrics": metrics
        })

    return {
        "vehicle_solutions": vehicle_solutions,
        "insertion_steps": insertion_steps,
        "unvisited_indexes": sorted(unvisited),
        "unvisited_nodes": [nodes[index] for index in sorted(unvisited)],
        "total_profit": sum(solution["metrics"]["total_profit"] for solution in vehicle_solutions),
        "total_travel_time": sum(solution["metrics"]["total_travel_time"] for solution in vehicle_solutions),
        "total_overtime": sum(solution["metrics"]["overtime"] for solution in vehicle_solutions),
        "total_objective": sum(solution["metrics"]["objective"] for solution in vehicle_solutions)
    }

def calculate_routes_totals(routes, route_times, nodes, time_limit, lambda_value=LAMBDA):
    depot_index = find_depot_index(nodes)
    total_profit = 0

    for route in routes:
        for node_index in route:
            if node_index != depot_index:
                total_profit += nodes[node_index]["profit"]

    total_travel_time = sum(route_times)
    total_overtime = sum(max(0, route_time - time_limit) for route_time in route_times)
    total_objective = total_profit - lambda_value * total_overtime

    return {
        "total_profit": total_profit,
        "total_travel_time": total_travel_time,
        "total_overtime": total_overtime,
        "total_objective": total_objective
    }

def two_opt_route(route, distance_matrix):
    route = route[:]
    total_saved = 0
    move_count = 0

    while True:
        best_delta = 0
        best_i = None
        best_k = None

        for i in range(1, len(route) - 2):
            for k in range(i + 1, len(route) - 1):
                a = route[i - 1]
                b = route[i]
                c = route[k]
                d = route[k + 1]
                delta = (
                    distance_matrix[a][c]
                    + distance_matrix[b][d]
                    - distance_matrix[a][b]
                    - distance_matrix[c][d]
                )

                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_k = k

        if best_i is None:
            break

        route[best_i:best_k + 1] = reversed(route[best_i:best_k + 1])
        total_saved += -best_delta
        move_count += 1

    return route, total_saved, move_count

def get_selected_customer_indexes(routes, nodes):
    depot_index = find_depot_index(nodes)
    selected = set()
    for route in routes:
        for node_index in route:
            if node_index != depot_index:
                selected.add(node_index)
    return selected

def find_destroy_candidates(routes, nodes, distance_matrix):
    candidates = []

    for vehicle_index, route in enumerate(routes):
        current_time = calculate_route_travel_time(route, distance_matrix)
        for position in range(1, len(route) - 1):
            node_index = route[position]
            candidate_route = route[:position] + route[position + 1:]
            new_time = calculate_route_travel_time(candidate_route, distance_matrix)
            saved_time = current_time - new_time
            profit = nodes[node_index]["profit"]
            weakness_score = profit / max(saved_time, 1)
            candidates.append({
                "vehicle_index": vehicle_index,
                "node_index": node_index,
                "node": nodes[node_index],
                "saved_time": saved_time,
                "profit": profit,
                "weakness_score": weakness_score
            })

    return sorted(
        candidates,
        key=lambda item: (
            item["weakness_score"],
            item["profit"],
            -item["saved_time"],
            item["node"]["node_id"]
        )
    )

def clone_routes(routes):
    return [route[:] for route in routes]

def calculate_route_times(routes, distance_matrix):
    return [
        calculate_route_travel_time(route, distance_matrix)
        for route in routes
    ]

def choose_weighted(weights, rng):
    total_weight = sum(weights.values())
    pick = rng.random() * total_weight
    cumulative = 0

    for name, weight in weights.items():
        cumulative += weight
        if pick <= cumulative:
            return name

    return next(reversed(weights))

def apply_two_opt_to_routes(routes, route_times, distance_matrix):
    saved_time = 0
    move_count = 0

    for vehicle_index, route in enumerate(routes):
        improved_route, route_saved_time, route_move_count = two_opt_route(route, distance_matrix)
        if route_move_count:
            routes[vehicle_index] = improved_route
            route_times[vehicle_index] -= route_saved_time
            saved_time += route_saved_time
            move_count += route_move_count

    return saved_time, move_count

def refresh_operator_weights(weights, scores, counts):
    for name in weights:
        if counts[name]:
            average_score = scores[name] / counts[name]
            weights[name] = (
                (1 - ALNS_REACTION_FACTOR) * weights[name]
                + ALNS_REACTION_FACTOR * max(average_score, 0.1)
            )
        scores[name] = 0
        counts[name] = 0

# Scenario state and first-stage feasibility

def get_n1_customer_indexes(instance_data):
    return [
        index
        for index, node in enumerate(instance_data["n1_nodes"])
        if node["node_id"] != 0
    ]

def normalize_route(route, depot_index):
    route = route[:]
    if not route:
        return [depot_index, depot_index]
    if route[0] != depot_index:
        route.insert(0, depot_index)
    if route[-1] != depot_index:
        route.append(depot_index)
    return route

def normalize_routes(routes, depot_index):
    return [normalize_route(route, depot_index) for route in routes]

def build_greedy_scenario_state(instance_data, scenario_id, vehicle_count):
    scenario = instance_data["scenarios"][scenario_id]
    nodes = scenario["nodes"]
    distance_matrix = scenario["distance_matrix"]
    time_limit = get_time_limit(instance_data, scenario_id)
    solution = build_parallel_greedy_insertion_solution(
        nodes,
        distance_matrix,
        time_limit,
        vehicle_count
    )
    depot_index = find_depot_index(nodes)
    routes = normalize_routes(
        [vehicle_solution["route"] for vehicle_solution in solution["vehicle_solutions"]],
        depot_index
    )

    return {
        "scenario_id": scenario_id,
        "nodes": nodes,
        "distance_matrix": distance_matrix,
        "time_limit": time_limit,
        "routes": routes,
        "route_times": calculate_route_times(routes, distance_matrix)
    }

def clone_scenario_state(state):
    return {
        "scenario_id": state["scenario_id"],
        "nodes": state["nodes"],
        "distance_matrix": state["distance_matrix"],
        "time_limit": state["time_limit"],
        "routes": clone_routes(state["routes"]),
        "route_times": state["route_times"][:]
    }

def calculate_state_totals(state, lambda_value):
    return calculate_routes_totals(
        state["routes"],
        state["route_times"],
        state["nodes"],
        state["time_limit"],
        lambda_value
    )

def build_solution_from_state(state, lambda_value):
    nodes = state["nodes"]
    depot_index = find_depot_index(nodes)
    selected = get_selected_customer_indexes(state["routes"], nodes)
    unvisited = set(range(len(nodes))) - selected - {depot_index}
    vehicle_solutions = []

    for route in state["routes"]:
        metrics = evaluate_solution(
            route,
            nodes,
            state["distance_matrix"],
            state["time_limit"],
            lambda_value
        )
        vehicle_solutions.append({
            "route": route,
            "nodes": [nodes[index] for index in route],
            "metrics": metrics
        })

    totals = calculate_state_totals(state, lambda_value)
    return {
        "vehicle_solutions": vehicle_solutions,
        "unvisited_indexes": sorted(unvisited),
        "unvisited_nodes": [nodes[index] for index in sorted(unvisited)],
        "total_profit": totals["total_profit"],
        "total_travel_time": totals["total_travel_time"],
        "total_overtime": totals["total_overtime"],
        "total_objective": totals["total_objective"]
    }

def find_node_in_routes(routes, node_index):
    for vehicle_index, route in enumerate(routes):
        for position in range(1, len(route) - 1):
            if route[position] == node_index:
                return vehicle_index, position
    return None

def insertion_objective_delta(route_time, added_time, profit, time_limit, lambda_value):
    old_overtime = max(0, route_time - time_limit)
    new_overtime = max(0, route_time + added_time - time_limit)
    return profit - lambda_value * (new_overtime - old_overtime)

def removal_objective_delta(route, position, route_time, nodes, distance_matrix, time_limit, lambda_value):
    node_index = route[position]
    candidate_route = route[:position] + route[position + 1:]
    new_time = calculate_route_travel_time(candidate_route, distance_matrix)
    old_overtime = max(0, route_time - time_limit)
    new_overtime = max(0, new_time - time_limit)
    return (
        -nodes[node_index]["profit"]
        - lambda_value * (new_overtime - old_overtime),
        candidate_route,
        new_time
    )

def find_best_feasible_state_insertion_for_node(state, node_index, mode="min_added"):
    nodes = state["nodes"]
    profit = nodes[node_index]["profit"]
    best = None

    for vehicle_index, route in enumerate(state["routes"]):
        route_time = state["route_times"][vehicle_index]
        for position in range(len(route) - 1):
            added_time = calculate_insertion_delta(
                route,
                position,
                node_index,
                state["distance_matrix"]
            )
            new_route_time = route_time + added_time
            if new_route_time > state["time_limit"]:
                continue

            slack = state["time_limit"] - new_route_time
            if mode == "balance":
                key = (-new_route_time, -added_time, profit, slack)
            elif mode == "profit":
                key = (profit, -added_time, slack)
            elif mode == "ratio":
                key = (profit / max(added_time, 1), profit, -added_time, slack)
            else:
                key = (-added_time, slack, profit)

            if best is None or key > best["key"]:
                best = {
                    "vehicle_index": vehicle_index,
                    "position": position,
                    "added_time": added_time,
                    "new_route_time": new_route_time,
                    "delta": profit,
                    "key": key
                }

    return best

def find_best_feasible_state_insertion(state, candidate_indexes, mode="ratio", selected=None):
    if selected is None:
        selected = get_selected_customer_indexes(state["routes"], state["nodes"])

    best = None
    for node_index in candidate_indexes:
        if node_index in selected:
            continue
        if state["nodes"][node_index]["profit"] <= 0:
            continue
        insertion = find_best_feasible_state_insertion_for_node(state, node_index, mode)
        if insertion is None:
            continue
        key = (
            insertion["key"],
            state["nodes"][node_index]["profit"],
            -node_index
        )
        if best is None or key > best["key"]:
            best = dict(insertion)
            best["node_index"] = node_index
            best["key"] = key

    return best

def insert_node_in_state(state, node_index, insertion):
    vehicle_index = insertion["vehicle_index"]
    insert_position = insertion["position"] + 1
    state["routes"][vehicle_index].insert(insert_position, node_index)
    state["route_times"][vehicle_index] = insertion["new_route_time"]

def remove_node_from_state(state, node_index):
    location = find_node_in_routes(state["routes"], node_index)
    if location is None:
        return False

    vehicle_index, position = location
    route = state["routes"][vehicle_index]
    route.pop(position)
    state["route_times"][vehicle_index] = calculate_route_travel_time(
        route,
        state["distance_matrix"]
    )
    return True

def repair_state_with_profitable_insertions(state, n1_count, fixed_decisions, lambda_value):
    depot_index = find_depot_index(state["nodes"])
    selected = get_selected_customer_indexes(state["routes"], state["nodes"])
    candidates = []

    for node_index in range(len(state["nodes"])):
        if node_index == depot_index or node_index in selected:
            continue
        if node_index < n1_count and fixed_decisions.get(node_index) == 0:
            continue
        candidates.append(node_index)

    while True:
        insertion = find_best_feasible_state_insertion(
            state,
            candidates,
            "ratio",
            selected
        )
        if insertion is None:
            break
        insert_node_in_state(state, insertion["node_index"], insertion)
        candidates.remove(insertion["node_index"])
        selected.add(insertion["node_index"])

def recover_state_feasibility(state, n1_count, protected_n1, lambda_value):
    while True:
        overfull_routes = [
            vehicle_index
            for vehicle_index, route_time in enumerate(state["route_times"])
            if route_time > state["time_limit"]
        ]
        if not overfull_routes:
            break

        best_removal = None
        for vehicle_index in overfull_routes:
            route = state["routes"][vehicle_index]
            route_time = state["route_times"][vehicle_index]
            for position in range(1, len(route) - 1):
                node_index = route[position]
                if node_index in protected_n1:
                    continue
                delta, candidate_route, new_time = removal_objective_delta(
                    route,
                    position,
                    route_time,
                    state["nodes"],
                    state["distance_matrix"],
                    state["time_limit"],
                    lambda_value
                )
                is_second_stage = node_index >= n1_count
                saved_time = route_time - new_time
                key = (
                    1 if is_second_stage else 0,
                    delta,
                    saved_time,
                    -state["nodes"][node_index]["profit"]
                )
                if best_removal is None or key > best_removal["key"]:
                    best_removal = {
                        "vehicle_index": vehicle_index,
                        "route": candidate_route,
                        "new_time": new_time,
                        "key": key
                    }

        if best_removal is None:
            break

        vehicle_index = best_removal["vehicle_index"]
        state["routes"][vehicle_index] = best_removal["route"]
        state["route_times"][vehicle_index] = best_removal["new_time"]

def improve_state_routes(state):
    saved_time, move_count = apply_two_opt_to_routes(
        state["routes"],
        state["route_times"],
        state["distance_matrix"]
    )
    return saved_time, move_count

def state_has_overtime(state):
    return any(route_time > state["time_limit"] for route_time in state["route_times"])

def missing_protected_nodes(state, protected_n1):
    selected = get_selected_customer_indexes(state["routes"], state["nodes"])
    return [node_index for node_index in sorted(protected_n1) if node_index not in selected]

def order_protected_nodes_for_rebuild(state, protected_n1, order_name):
    depot_index = find_depot_index(state["nodes"])
    if order_name == "profit_desc":
        return sorted(
            protected_n1,
            key=lambda node_index: (
                -state["nodes"][node_index]["profit"],
                state["distance_matrix"][depot_index][node_index],
                node_index
            )
        )
    if order_name == "farthest":
        return sorted(
            protected_n1,
            key=lambda node_index: (
                -state["distance_matrix"][depot_index][node_index],
                -state["nodes"][node_index]["profit"],
                node_index
            )
        )
    if order_name == "nearest":
        return sorted(
            protected_n1,
            key=lambda node_index: (
                state["distance_matrix"][depot_index][node_index],
                -state["nodes"][node_index]["profit"],
                node_index
            )
        )
    return sorted(protected_n1)

def rebuild_state_with_fixed_decisions(state, n1_count, fixed_decisions, lambda_value):
    protected_n1 = {
        node_index
        for node_index, value in fixed_decisions.items()
        if value == 1
    }
    depot_index = find_depot_index(state["nodes"])
    candidate_specs = (
        ("profit_desc", "balance"),
        ("node_id", "balance"),
        ("profit_desc", "min_added"),
        ("farthest", "min_added"),
        ("nearest", "balance")
    )
    best_candidate = None

    for order_name, insertion_mode in candidate_specs:
        candidate = clone_scenario_state(state)
        candidate["routes"] = [
            [depot_index, depot_index]
            for _ in candidate["routes"]
        ]
        candidate["route_times"] = [0 for _ in candidate["routes"]]
        failed = False

        for node_index in order_protected_nodes_for_rebuild(
            candidate,
            protected_n1,
            order_name
        ):
            insertion = find_best_feasible_state_insertion_for_node(
                candidate,
                node_index,
                insertion_mode
            )
            if insertion is None:
                failed = True
                break
            insert_node_in_state(candidate, node_index, insertion)
            improve_state_routes(candidate)

        if failed:
            continue

        improve_state_with_relocation(candidate, lambda_value, max_moves=100)
        improve_state_routes(candidate)
        recover_state_feasibility(candidate, n1_count, protected_n1, lambda_value)
        if missing_protected_nodes(candidate, protected_n1) or state_has_overtime(candidate):
            continue

        repair_state_with_profitable_insertions(
            candidate,
            n1_count,
            fixed_decisions,
            lambda_value
        )
        improve_state_routes(candidate)
        improve_state_with_relocation(candidate, lambda_value, max_moves=100)
        improve_state_routes(candidate)
        recover_state_feasibility(candidate, n1_count, protected_n1, lambda_value)

        if missing_protected_nodes(candidate, protected_n1) or state_has_overtime(candidate):
            continue

        totals = calculate_state_totals(candidate, lambda_value)
        key = (
            totals["total_objective"],
            totals["total_profit"],
            -totals["total_travel_time"]
        )
        if best_candidate is None or key > best_candidate["key"]:
            best_candidate = {
                "routes": clone_routes(candidate["routes"]),
                "route_times": candidate["route_times"][:],
                "key": key
            }

    if best_candidate is None:
        return False

    state["routes"] = best_candidate["routes"]
    state["route_times"] = best_candidate["route_times"]
    return True

def reoptimize_state_with_fixed_decisions(state, n1_count, fixed_decisions, lambda_value):
    for node_index, value in fixed_decisions.items():
        if value == 0:
            remove_node_from_state(state, node_index)

    protected_n1 = {
        node_index
        for node_index, value in fixed_decisions.items()
        if value == 1
    }

    selected = get_selected_customer_indexes(state["routes"], state["nodes"])
    for node_index in sorted(protected_n1):
        if node_index in selected:
            continue
        insertion = find_best_feasible_state_insertion_for_node(
            state,
            node_index,
            "balance"
        )
        if insertion is not None:
            insert_node_in_state(state, node_index, insertion)
            selected.add(node_index)

    improve_state_routes(state)
    recover_state_feasibility(state, n1_count, protected_n1, lambda_value)
    if missing_protected_nodes(state, protected_n1) or state_has_overtime(state):
        rebuilt = rebuild_state_with_fixed_decisions(
            state,
            n1_count,
            fixed_decisions,
            lambda_value
        )
        if rebuilt:
            return

    repair_state_with_profitable_insertions(state, n1_count, fixed_decisions, lambda_value)
    improve_state_routes(state)
    recover_state_feasibility(state, n1_count, protected_n1, lambda_value)
    if missing_protected_nodes(state, protected_n1) or state_has_overtime(state):
        rebuild_state_with_fixed_decisions(
            state,
            n1_count,
            fixed_decisions,
            lambda_value
        )

# Stored reference selections

def get_reference_selected_n1_node_ids(instance_data):
    n1_node_ids = {
        node["node_id"]
        for node in instance_data["n1_nodes"]
        if node["node_id"] != 0
    }
    selections = {}
    for scenario_id, solution in instance_data["solutions"].items():
        selected = set()
        for route_info in solution["routes"].values():
            selected.update(
                node_id
                for node_id in route_info["path"]
                if node_id in n1_node_ids
            )
        selections[scenario_id] = selected

    common = set.intersection(*selections.values()) if selections else set()
    union = set.union(*selections.values()) if selections else set()
    return {
        "common": sorted(common),
        "union": sorted(union),
        "all_same": all(selected == common for selected in selections.values()),
        "by_scenario": selections
    }

def build_stochastic_result(
    strategy,
    fixed_decisions,
    states,
    instance_data,
    lambda_value,
    history=None,
    search_history=None,
    algorithm_name="D-SALNS"
):
    scenario_solutions = {
        scenario_id: build_solution_from_state(state, lambda_value)
        for scenario_id, state in states.items()
    }
    scenario_totals = {
        scenario_id: calculate_state_totals(state, lambda_value)
        for scenario_id, state in states.items()
    }
    expected_profit = (
        sum(totals["total_profit"] for totals in scenario_totals.values())
        / len(scenario_totals)
    )
    expected_objective = (
        sum(totals["total_objective"] for totals in scenario_totals.values())
        / len(scenario_totals)
    )
    selected_n1_indexes = [
        node_index
        for node_index, value in sorted(fixed_decisions.items())
        if value == 1
    ]
    selected_n1_node_ids = [
        instance_data["n1_nodes"][node_index]["node_id"]
        for node_index in selected_n1_indexes
    ]

    return {
        "algorithm_name": algorithm_name,
        "strategy": strategy,
        "fixed_decisions": fixed_decisions,
        "selected_n1_indexes": selected_n1_indexes,
        "selected_n1_node_ids": selected_n1_node_ids,
        "scenario_states": states,
        "scenario_solutions": scenario_solutions,
        "scenario_totals": scenario_totals,
        "expected_profit": expected_profit,
        "expected_objective": expected_objective,
        "history": history or [],
        "search_history": search_history or []
    }

def get_lower_level_removal_candidates(state, n1_count, fixed_decisions, unlocked):
    protected = {
        node_index
        for node_index, value in fixed_decisions.items()
        if value == 1
    }
    candidates = []
    for candidate in find_destroy_candidates(state["routes"], state["nodes"], state["distance_matrix"]):
        node_index = candidate["node_index"]
        if node_index in protected:
            continue
        if node_index < n1_count and node_index not in unlocked:
            continue
        candidates.append(candidate)
    return candidates

def destroy_lower_level_nodes(state, n1_count, fixed_decisions, unlocked, rng, remove_count, mode):
    candidates = get_lower_level_removal_candidates(state, n1_count, fixed_decisions, unlocked)
    if not candidates:
        return []

    remove_count = min(remove_count, len(candidates))
    if mode == "random":
        selected = rng.sample(candidates, remove_count)
    elif mode == "high_detour":
        selected = sorted(
            candidates,
            key=lambda item: (-item["saved_time"], item["profit"], item["node"]["node_id"])
        )[:remove_count]
    elif mode == "near":
        seed = rng.choice(candidates)["node_index"]
        selected = sorted(
            candidates,
            key=lambda item: (
                state["distance_matrix"][seed][item["node_index"]],
                item["profit"],
                -item["saved_time"]
            )
        )[:remove_count]
    elif mode == "similar_detour":
        seed_candidate = rng.choice(candidates)
        seed_detour = seed_candidate["saved_time"]
        selected = sorted(
            candidates,
            key=lambda item: (
                abs(item["saved_time"] - seed_detour),
                state["distance_matrix"][seed_candidate["node_index"]][item["node_index"]],
                item["profit"]
            )
        )[:remove_count]
    elif mode == "overfull":
        overfull_vehicles = {
            vehicle_index
            for vehicle_index, route_time in enumerate(state["route_times"])
            if route_time > state["time_limit"]
        }
        overfull_candidates = [
            candidate
            for candidate in candidates
            if candidate["vehicle_index"] in overfull_vehicles
        ]
        selected = (
            sorted(overfull_candidates, key=lambda item: (item["weakness_score"], item["profit"]))[:remove_count]
            if overfull_candidates
            else sorted(candidates, key=lambda item: (item["weakness_score"], item["profit"]))[:remove_count]
        )
    else:
        selected = sorted(candidates, key=lambda item: (item["weakness_score"], item["profit"]))[:remove_count]

    removed = sorted({candidate["node_index"] for candidate in selected})
    for node_index in removed:
        remove_node_from_state(state, node_index)
    return removed

def score_lower_level_insertion(mode, delta, profit, added_time, new_route_time):
    if mode == "profit":
        return (profit, delta, -added_time, -new_route_time)
    if mode == "min_distance":
        return (-added_time, delta, profit, -new_route_time)
    if mode == "ratio":
        return (delta / max(added_time, 1), delta, profit, -added_time)
    return (delta, profit, -added_time, -new_route_time)

def find_best_lower_level_insertion(state, candidate_indexes, lambda_value, mode, selected=None):
    if selected is None:
        selected = get_selected_customer_indexes(state["routes"], state["nodes"])

    best = None
    for node_index in candidate_indexes:
        if node_index in selected:
            continue
        profit = state["nodes"][node_index]["profit"]
        if profit <= 0:
            continue

        for vehicle_index, route in enumerate(state["routes"]):
            route_time = state["route_times"][vehicle_index]
            for position in range(len(route) - 1):
                added_time = calculate_insertion_delta(
                    route,
                    position,
                    node_index,
                    state["distance_matrix"]
                )
                new_route_time = route_time + added_time
                if new_route_time > state["time_limit"]:
                    continue
                delta = insertion_objective_delta(
                    route_time,
                    added_time,
                    profit,
                    state["time_limit"],
                    lambda_value
                )
                if delta <= 0:
                    continue
                key = score_lower_level_insertion(mode, delta, profit, added_time, new_route_time)
                if best is None or key > best["key"]:
                    best = {
                        "node_index": node_index,
                        "vehicle_index": vehicle_index,
                        "position": position,
                        "added_time": added_time,
                        "new_route_time": new_route_time,
                        "delta": delta,
                        "key": key
                    }
    return best

def repair_state_with_mode(state, n1_count, fixed_decisions, lambda_value, mode):
    depot_index = find_depot_index(state["nodes"])
    selected = get_selected_customer_indexes(state["routes"], state["nodes"])
    candidates = []
    for node_index in range(len(state["nodes"])):
        if node_index == depot_index or node_index in selected:
            continue
        if node_index < n1_count and fixed_decisions.get(node_index) == 0:
            continue
        candidates.append(node_index)

    inserted = []
    while True:
        insertion = find_best_lower_level_insertion(
            state,
            candidates,
            lambda_value,
            mode,
            selected
        )
        if insertion is None:
            break
        insert_node_in_state(state, insertion["node_index"], insertion)
        selected.add(insertion["node_index"])
        candidates.remove(insertion["node_index"])
        inserted.append(insertion["node_index"])
    return inserted

def get_lower_level_insertion_candidates(state, n1_count, fixed_decisions, selected=None):
    depot_index = find_depot_index(state["nodes"])
    if selected is None:
        selected = get_selected_customer_indexes(state["routes"], state["nodes"])

    candidates = []
    for node_index in range(len(state["nodes"])):
        if node_index == depot_index or node_index in selected:
            continue
        if node_index < n1_count and fixed_decisions.get(node_index) == 0:
            continue
        if state["nodes"][node_index]["profit"] <= 0:
            continue
        candidates.append(node_index)
    return candidates

def cap_replacement_candidates(state, candidates, candidate_limit):
    if candidate_limit is None or candidate_limit <= 0 or len(candidates) <= candidate_limit:
        return candidates

    scored = []
    for node_index in candidates:
        best_added_time = None
        for route in state["routes"]:
            for position in range(len(route) - 1):
                added_time = calculate_insertion_delta(
                    route,
                    position,
                    node_index,
                    state["distance_matrix"]
                )
                if best_added_time is None or added_time < best_added_time:
                    best_added_time = added_time

        if best_added_time is None:
            best_added_time = state["time_limit"] * max(1, len(state["routes"]))

        profit = state["nodes"][node_index]["profit"]
        scored.append((
            (
                profit / max(best_added_time, 1),
                profit,
                -best_added_time,
                -node_index
            ),
            node_index
        ))

    scored.sort(reverse=True)
    return [node_index for _, node_index in scored[:candidate_limit]]

def find_best_lower_level_regret_insertion(state, candidate_indexes, selected=None):
    if selected is None:
        selected = get_selected_customer_indexes(state["routes"], state["nodes"])

    best_choice = None
    for node_index in candidate_indexes:
        if node_index in selected:
            continue
        profit = state["nodes"][node_index]["profit"]
        if profit <= 0:
            continue

        insertions = []
        for vehicle_index, route in enumerate(state["routes"]):
            route_time = state["route_times"][vehicle_index]
            for position in range(len(route) - 1):
                added_time = calculate_insertion_delta(
                    route,
                    position,
                    node_index,
                    state["distance_matrix"]
                )
                new_route_time = route_time + added_time
                if new_route_time > state["time_limit"]:
                    continue
                insertions.append({
                    "node_index": node_index,
                    "vehicle_index": vehicle_index,
                    "position": position,
                    "added_time": added_time,
                    "new_route_time": new_route_time
                })

        if not insertions:
            continue

        insertions.sort(key=lambda item: (item["added_time"], item["new_route_time"]))
        best = insertions[0]
        second_added_time = insertions[1]["added_time"] if len(insertions) > 1 else state["time_limit"]
        regret = second_added_time - best["added_time"]
        key = (profit + regret, profit, regret, -best["added_time"], -best["new_route_time"])
        if best_choice is None or key > best_choice["key"]:
            best_choice = dict(best)
            best_choice["key"] = key

    return best_choice

def repair_state_with_regret_insertions(state, n1_count, fixed_decisions):
    selected = get_selected_customer_indexes(state["routes"], state["nodes"])
    candidates = get_lower_level_insertion_candidates(
        state,
        n1_count,
        fixed_decisions,
        selected
    )
    inserted = []
    while True:
        insertion = find_best_lower_level_regret_insertion(state, candidates, selected)
        if insertion is None:
            break
        insert_node_in_state(state, insertion["node_index"], insertion)
        selected.add(insertion["node_index"])
        candidates.remove(insertion["node_index"])
        inserted.append(insertion["node_index"])
    return inserted

def repair_state_with_full_insertions(state, n1_count, fixed_decisions):
    selected = get_selected_customer_indexes(state["routes"], state["nodes"])
    candidates = get_lower_level_insertion_candidates(
        state,
        n1_count,
        fixed_decisions,
        selected
    )
    inserted = []

    for vehicle_index in range(len(state["routes"])):
        while True:
            best = None
            route = state["routes"][vehicle_index]
            route_time = state["route_times"][vehicle_index]
            for node_index in candidates:
                profit = state["nodes"][node_index]["profit"]
                for position in range(len(route) - 1):
                    added_time = calculate_insertion_delta(
                        route,
                        position,
                        node_index,
                        state["distance_matrix"]
                    )
                    new_route_time = route_time + added_time
                    if new_route_time > state["time_limit"]:
                        continue
                    key = (profit, profit / max(added_time, 1), -added_time, -new_route_time)
                    if best is None or key > best["key"]:
                        best = {
                            "node_index": node_index,
                            "vehicle_index": vehicle_index,
                            "position": position,
                            "added_time": added_time,
                            "new_route_time": new_route_time,
                            "key": key
                        }

            if best is None:
                break
            insert_node_in_state(state, best["node_index"], best)
            selected.add(best["node_index"])
            candidates.remove(best["node_index"])
            inserted.append(best["node_index"])
            route = state["routes"][vehicle_index]
            route_time = state["route_times"][vehicle_index]

    return inserted

def repair_state_with_lower_operator(state, n1_count, fixed_decisions, lambda_value, operator_name):
    if operator_name == "regret2":
        return repair_state_with_regret_insertions(state, n1_count, fixed_decisions)
    if operator_name == "full":
        return repair_state_with_full_insertions(state, n1_count, fixed_decisions)
    return repair_state_with_mode(state, n1_count, fixed_decisions, lambda_value, operator_name)

def improve_state_with_replacement(
    state,
    n1_count,
    fixed_decisions,
    protected_n1,
    max_moves=30,
    deadline=None,
    candidate_limit=LOWER_REPLACEMENT_CANDIDATE_LIMIT
):
    move_count = 0
    depot_index = find_depot_index(state["nodes"])

    while move_count < max_moves:
        if deadline_reached(deadline):
            break
        selected = get_selected_customer_indexes(state["routes"], state["nodes"])
        candidates = get_lower_level_insertion_candidates(
            state,
            n1_count,
            fixed_decisions,
            selected
        )
        if not candidates:
            break
        candidates = cap_replacement_candidates(state, candidates, candidate_limit)

        best_move = None
        timed_out = False
        for from_vehicle, route in enumerate(state["routes"]):
            if deadline_reached(deadline):
                timed_out = True
                break
            if len(route) <= 2:
                continue
            old_from_time = state["route_times"][from_vehicle]
            for from_position in range(1, len(route) - 1):
                if deadline_reached(deadline):
                    timed_out = True
                    break
                removed_index = route[from_position]
                if removed_index == depot_index or removed_index in protected_n1:
                    continue
                removed_profit = state["nodes"][removed_index]["profit"]
                base_from_route = route[:from_position] + route[from_position + 1:]
                base_from_time = calculate_route_travel_time(
                    base_from_route,
                    state["distance_matrix"]
                )

                for insert_index in candidates:
                    if deadline_reached(deadline):
                        timed_out = True
                        break
                    insert_profit = state["nodes"][insert_index]["profit"]
                    profit_gain = insert_profit - removed_profit
                    if profit_gain < 0:
                        continue

                    for to_vehicle, target_route in enumerate(state["routes"]):
                        if deadline_reached(deadline):
                            timed_out = True
                            break
                        target_base_route = (
                            base_from_route
                            if to_vehicle == from_vehicle
                            else target_route
                        )
                        target_base_time = (
                            base_from_time
                            if to_vehicle == from_vehicle
                            else state["route_times"][to_vehicle]
                        )
                        for insert_position in range(len(target_base_route) - 1):
                            added_time = calculate_insertion_delta(
                                target_base_route,
                                insert_position,
                                insert_index,
                                state["distance_matrix"]
                            )
                            new_target_time = target_base_time + added_time
                            if new_target_time > state["time_limit"]:
                                continue

                            if to_vehicle == from_vehicle:
                                time_delta = new_target_time - old_from_time
                                feasible = True
                            else:
                                feasible = base_from_time <= state["time_limit"]
                                time_delta = (
                                    base_from_time - old_from_time
                                    + new_target_time - state["route_times"][to_vehicle]
                                )
                            if not feasible:
                                continue

                            if profit_gain == 0 and time_delta >= 0:
                                continue

                            key = (
                                profit_gain,
                                -time_delta,
                                insert_profit,
                                -removed_profit,
                                -new_target_time
                            )
                            if best_move is None or key > best_move["key"]:
                                best_move = {
                                    "from_vehicle": from_vehicle,
                                    "from_position": from_position,
                                    "to_vehicle": to_vehicle,
                                    "insert_position": insert_position,
                                    "insert_index": insert_index,
                                    "base_from_route": base_from_route,
                                    "base_from_time": base_from_time,
                                    "target_base_route": target_base_route,
                                    "new_target_time": new_target_time,
                                    "key": key
                                }
                    if timed_out:
                        break
                if timed_out:
                    break
            if timed_out:
                break

        if timed_out or best_move is None:
            break

        from_vehicle = best_move["from_vehicle"]
        to_vehicle = best_move["to_vehicle"]
        if from_vehicle == to_vehicle:
            new_route = best_move["target_base_route"][:]
            new_route.insert(best_move["insert_position"] + 1, best_move["insert_index"])
            state["routes"][from_vehicle] = new_route
            state["route_times"][from_vehicle] = best_move["new_target_time"]
        else:
            state["routes"][from_vehicle] = best_move["base_from_route"]
            state["route_times"][from_vehicle] = best_move["base_from_time"]
            new_target_route = state["routes"][to_vehicle][:]
            new_target_route.insert(best_move["insert_position"] + 1, best_move["insert_index"])
            state["routes"][to_vehicle] = new_target_route
            state["route_times"][to_vehicle] = best_move["new_target_time"]
        move_count += 1

    return move_count

def lower_level_solution_key(state, lambda_value):
    totals = calculate_state_totals(state, lambda_value)
    return (
        totals["total_objective"],
        totals["total_profit"],
        -totals["total_travel_time"]
    )

def lower_level_state_is_feasible(state, protected_n1):
    return (
        not state_has_overtime(state)
        and not missing_protected_nodes(state, protected_n1)
    )

def copy_state_routes(source, target):
    target["routes"] = clone_routes(source["routes"])
    target["route_times"] = source["route_times"][:]

def improve_state_with_relocation(state, lambda_value, max_moves=20):
    move_count = 0
    while move_count < max_moves:
        best_move = None
        for vehicle_index, route in enumerate(state["routes"]):
            if len(route) <= 3:
                continue
            for position in range(1, len(route) - 1):
                node_index = route[position]
                base_route = route[:position] + route[position + 1:]
                base_time = calculate_route_travel_time(base_route, state["distance_matrix"])
                old_time = state["route_times"][vehicle_index]
                old_overtime = max(0, old_time - state["time_limit"])

                for target_vehicle_index, target_route in enumerate(state["routes"]):
                    target_base_route = base_route if target_vehicle_index == vehicle_index else target_route
                    target_base_time = base_time if target_vehicle_index == vehicle_index else state["route_times"][target_vehicle_index]
                    for insert_position in range(len(target_base_route) - 1):
                        if target_vehicle_index == vehicle_index and insert_position in (position - 1, position):
                            continue
                        added_time = calculate_insertion_delta(
                            target_base_route,
                            insert_position,
                            node_index,
                            state["distance_matrix"]
                        )
                        new_target_time = target_base_time + added_time
                        if target_vehicle_index == vehicle_index:
                            time_delta = new_target_time - old_time
                            overtime_delta = max(0, new_target_time - state["time_limit"]) - old_overtime
                        else:
                            time_delta = base_time - old_time + new_target_time - state["route_times"][target_vehicle_index]
                            overtime_delta = (
                                max(0, base_time - state["time_limit"]) - old_overtime
                                + max(0, new_target_time - state["time_limit"])
                                - max(0, state["route_times"][target_vehicle_index] - state["time_limit"])
                            )
                        objective_delta = -lambda_value * overtime_delta
                        key = (objective_delta, -time_delta)
                        if overtime_delta < 0 or (overtime_delta == 0 and time_delta < 0):
                            if best_move is None or key > best_move["key"]:
                                best_move = {
                                    "from_vehicle": vehicle_index,
                                    "from_position": position,
                                    "to_vehicle": target_vehicle_index,
                                    "to_position": insert_position,
                                    "node_index": node_index,
                                    "key": key
                                }

        if best_move is None:
            break

        from_vehicle = best_move["from_vehicle"]
        to_vehicle = best_move["to_vehicle"]
        node_index = best_move["node_index"]
        state["routes"][from_vehicle].pop(best_move["from_position"])
        insert_position = best_move["to_position"]
        state["routes"][to_vehicle].insert(insert_position + 1, node_index)
        state["route_times"][from_vehicle] = calculate_route_travel_time(
            state["routes"][from_vehicle],
            state["distance_matrix"]
        )
        if to_vehicle != from_vehicle:
            state["route_times"][to_vehicle] = calculate_route_travel_time(
                state["routes"][to_vehicle],
                state["distance_matrix"]
            )
        move_count += 1
    return move_count

# One scenario: repair fixed membership, improve feasible routes, then run ALNS

def solve_lower_level_state(
    state,
    n1_count,
    fixed_decisions,
    unlocked,
    lambda_value,
    rng,
    passes=DSALNS_LOWER_PASSES,
    remove_count=DESTROY_REPAIR_MAX_REMOVE,
    deadline=None,
    stagnation_limit=LOWER_ALNS_STAGNATION_LIMIT,
    replacement_candidate_limit=LOWER_REPLACEMENT_CANDIDATE_LIMIT
):
    for node_index in unlocked:
        remove_node_from_state(state, node_index)

    reoptimize_state_with_fixed_decisions(state, n1_count, fixed_decisions, lambda_value)
    if deadline_reached(deadline):
        return False

    protected_n1 = {
        node_index
        for node_index, value in fixed_decisions.items()
        if value == 1
    }
    if not lower_level_state_is_feasible(state, protected_n1):
        rebuild_state_with_fixed_decisions(state, n1_count, fixed_decisions, lambda_value)

    repair_state_with_lower_operator(state, n1_count, fixed_decisions, lambda_value, "full")
    improve_state_routes(state)
    improve_state_with_relocation(state, lambda_value, max_moves=50)
    improve_state_with_replacement(
        state,
        n1_count,
        fixed_decisions,
        protected_n1,
        max_moves=30,
        deadline=deadline,
        candidate_limit=replacement_candidate_limit
    )
    if deadline_reached(deadline):
        return False
    improve_state_routes(state)
    recover_state_feasibility(state, n1_count, protected_n1, lambda_value)

    if not lower_level_state_is_feasible(state, protected_n1):
        rebuild_state_with_fixed_decisions(state, n1_count, fixed_decisions, lambda_value)

    current = clone_scenario_state(state)
    best = clone_scenario_state(state)
    current_key = lower_level_solution_key(current, lambda_value)
    best_key = lower_level_solution_key(best, lambda_value)

    destroy_weights = {
        "worst": 1.0,
        "random": 1.0,
        "high_detour": 1.0,
        "near": 1.0,
        "similar_detour": 1.0,
        "overfull": 1.0
    }
    repair_weights = {
        "lagrangian": 1.0,
        "profit": 1.0,
        "min_distance": 1.0,
        "ratio": 1.0,
        "regret2": 1.0,
        "full": 1.0
    }
    destroy_scores = {name: 0 for name in destroy_weights}
    repair_scores = {name: 0 for name in repair_weights}
    destroy_counts = {name: 0 for name in destroy_weights}
    repair_counts = {name: 0 for name in repair_weights}

    iterations = max(0, passes) * LOWER_ALNS_ITERATIONS_PER_PASS
    temperature = max(1.0, calculate_state_totals(best, lambda_value)["total_profit"] * LOWER_ALNS_START_TEMPERATURE_FACTOR)

    timed_out = False
    non_improving_iterations = 0
    for iteration in range(1, iterations + 1):
        if deadline_reached(deadline):
            timed_out = True
            break
        if stagnation_limit and non_improving_iterations >= stagnation_limit:
            break
        destroy_name = choose_weighted(destroy_weights, rng)
        repair_name = choose_weighted(repair_weights, rng)
        destroy_counts[destroy_name] += 1
        repair_counts[repair_name] += 1

        candidate = clone_scenario_state(current)
        removable_count = len(
            get_lower_level_removal_candidates(
                candidate,
                n1_count,
                fixed_decisions,
                unlocked
            )
        )
        removed = []
        if removable_count:
            dynamic_upper = min(
                removable_count,
                max(LOWER_ALNS_REMOVE_MAX, removable_count // 3, remove_count)
            )
            dynamic_lower = min(LOWER_ALNS_REMOVE_MIN, dynamic_upper)
            destroy_count = rng.randint(dynamic_lower, dynamic_upper)
            removed = destroy_lower_level_nodes(
                candidate,
                n1_count,
                fixed_decisions,
                unlocked,
                rng,
                destroy_count,
                destroy_name
            )

        if not removed and not get_lower_level_insertion_candidates(candidate, n1_count, fixed_decisions):
            non_improving_iterations += 1
            continue

        repair_state_with_lower_operator(
            candidate,
            n1_count,
            fixed_decisions,
            lambda_value,
            repair_name
        )
        improve_state_routes(candidate)
        improve_state_with_relocation(candidate, lambda_value, max_moves=40)
        improve_state_with_replacement(
            candidate,
            n1_count,
            fixed_decisions,
            protected_n1,
            max_moves=25,
            deadline=deadline,
            candidate_limit=replacement_candidate_limit
        )
        if deadline_reached(deadline):
            timed_out = True
            break
        improve_state_routes(candidate)
        repair_state_with_lower_operator(
            candidate,
            n1_count,
            fixed_decisions,
            lambda_value,
            "ratio"
        )
        improve_state_routes(candidate)
        recover_state_feasibility(candidate, n1_count, protected_n1, lambda_value)

        if not lower_level_state_is_feasible(candidate, protected_n1):
            reward = 0
            improved_best = False
        else:
            candidate_key = lower_level_solution_key(candidate, lambda_value)
            delta_current = candidate_key[0] - current_key[0]
            accepted = False
            reward = 0
            improved_best = False
            if candidate_key >= current_key:
                accepted = True
                reward = 4
            else:
                acceptance_probability = exp(max(-50, delta_current / max(temperature, 0.1)))
                if rng.random() < acceptance_probability:
                    accepted = True
                    reward = 1

            if accepted:
                current = candidate
                current_key = candidate_key

            if candidate_key > best_key:
                best = clone_scenario_state(candidate)
                best_key = candidate_key
                reward = 10
                improved_best = True

        if improved_best:
            non_improving_iterations = 0
        else:
            non_improving_iterations += 1

        destroy_scores[destroy_name] += reward
        repair_scores[repair_name] += reward

        if iteration % LOWER_ALNS_SEGMENT_LENGTH == 0:
            refresh_operator_weights(destroy_weights, destroy_scores, destroy_counts)
            refresh_operator_weights(repair_weights, repair_scores, repair_counts)

        temperature = max(0.1, temperature * LOWER_ALNS_COOLING_RATE)

    copy_state_routes(best, state)
    if timed_out or deadline_reached(deadline):
        return False
    improve_state_routes(state)
    improve_state_with_relocation(state, lambda_value, max_moves=100)
    improve_state_with_replacement(
        state,
        n1_count,
        fixed_decisions,
        protected_n1,
        max_moves=50,
        deadline=deadline,
        candidate_limit=replacement_candidate_limit
    )
    if deadline_reached(deadline):
        return False
    improve_state_routes(state)
    recover_state_feasibility(state, n1_count, protected_n1, lambda_value)
    if not lower_level_state_is_feasible(state, protected_n1):
        copy_state_routes(best, state)
    return True

# Evaluate a fixed subscription set across scenarios

def build_fixed_n1_decisions(instance_data, selected_n1_node_ids):
    """Build the Pisinger-style common first-stage decision.

    The stored Pisinger solutions use the same selected N1 customer set in
    every scenario. Therefore a candidate set is interpreted as the complete
    common N1 subset: selected nodes are fixed to 1, and all other N1
    customers are fixed to 0 during recourse evaluation.
    """
    selected_n1_node_ids = set(selected_n1_node_ids)
    fixed_decisions = {}
    for node_index in get_n1_customer_indexes(instance_data):
        node_id = instance_data["n1_nodes"][node_index]["node_id"]
        fixed_decisions[node_index] = 1 if node_id in selected_n1_node_ids else 0
    return fixed_decisions

def get_instance_vehicle_count(instance_data):
    first_solution = next(iter(instance_data["solutions"].values()), None)
    return first_solution["vehicles"] if first_solution else VEHICLES

def build_greedy_scenario_states(instance_data, vehicle_count=None):
    if vehicle_count is None:
        vehicle_count = get_instance_vehicle_count(instance_data)
    return {
        scenario_id: build_greedy_scenario_state(instance_data, scenario_id, vehicle_count)
        for scenario_id in sorted(instance_data["scenarios"])
    }

def clone_scenario_states(states):
    return {
        scenario_id: clone_scenario_state(state)
        for scenario_id, state in states.items()
    }

def get_selected_n1_node_ids_from_state(state, n1_count):
    selected = get_selected_customer_indexes(state["routes"], state["nodes"])
    return sorted(
        state["nodes"][node_index]["node_id"]
        for node_index in selected
        if node_index < n1_count
    )

def evaluate_fixed_n1_candidate(
    instance_data,
    selected_n1_node_ids,
    source,
    lambda_value,
    lower_passes,
    rng,
    base_states=None,
    lower_stagnation_limit=LOWER_ALNS_STAGNATION_LIMIT,
    replacement_candidate_limit=LOWER_REPLACEMENT_CANDIDATE_LIMIT
):
    vehicle_count = get_instance_vehicle_count(instance_data)
    n1_count = len(instance_data["n1_nodes"])
    fixed_decisions = build_fixed_n1_decisions(instance_data, selected_n1_node_ids)
    protected_n1 = {
        node_index
        for node_index, value in fixed_decisions.items()
        if value == 1
    }
    states = (
        clone_scenario_states(base_states)
        if base_states is not None
        else build_greedy_scenario_states(instance_data, vehicle_count)
    )

    feasible = True
    for state in states.values():
        solve_lower_level_state(
            state,
            n1_count,
            fixed_decisions,
            set(),
            lambda_value,
            rng,
            passes=lower_passes,
            stagnation_limit=lower_stagnation_limit,
            replacement_candidate_limit=replacement_candidate_limit
        )
        if not lower_level_state_is_feasible(state, protected_n1):
            feasible = False

    result = build_stochastic_result(
        source,
        fixed_decisions,
        states,
        instance_data,
        lambda_value,
        [],
        [],
        "Stochastic solver"
    )
    result["is_feasible"] = feasible
    result["candidate_source"] = source
    result["candidate_sources"] = [source]
    return result

def candidate_result_key(result):
    if not result.get("is_feasible", True):
        return (float("-inf"), float("-inf"), float("-inf"))
    return (
        result["expected_objective"],
        result["expected_profit"],
        -len(result["selected_n1_node_ids"])
    )

# Score and improve a fixed first-stage set

def score_routes(raw, scenario, routes, selected_ids):
    nodes = {node["node_id"]: node for node in raw["n1_nodes"] + raw["scenarios_n2"][scenario]}
    n1 = {node["node_id"] for node in raw["n1_nodes"] if node["node_id"] != 0}
    selected = set(selected_ids)
    metadata = raw["solutions"][scenario]
    errors = []
    if not selected <= n1:
        errors.append("fixed set contains a non-N1 customer")
    if len(routes) != metadata["vehicles"]:
        errors.append("wrong vehicle count")
    visited = []
    durations = []
    for route in routes:
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            errors.append("route does not start and end at the depot")
        if any(node not in nodes for node in route):
            errors.append("unknown route node")
            durations.append(None)
            continue
        if 0 in route[1:-1]:
            errors.append("depot appears inside a route")
        visited.extend(route[1:-1])
        durations.append(sum(round(math.hypot(nodes[a]["x"] - nodes[b]["x"],
                                             nodes[a]["y"] - nodes[b]["y"]))
                             for a, b in zip(route, route[1:])))
    if len(visited) != len(set(visited)):
        errors.append("repeated customer")
    if (set(visited) & n1) != selected:
        errors.append("first-stage membership differs from the fixed set")
    overtime = sum(max(0, value - metadata["time_limit"]) for value in durations if value is not None)
    if overtime:
        errors.append("route time limit exceeded")
    profit = sum(nodes[node]["profit"] for node in visited)
    return {"feasible": not errors, "profit": profit, "objective": profit if not errors else None,
            "overtime": overtime, "route_times": durations, "errors": errors}

def state_from_routes(instance, scenario, routes):
    scene = instance["scenarios"][scenario]
    index = {node["node_id"]: i for i, node in enumerate(scene["nodes"])}
    paths = [[index[node] for node in route] for route in routes]
    return {"scenario_id": scenario, "nodes": scene["nodes"],
            "distance_matrix": scene["distance_matrix"],
            "time_limit": instance["solutions"][scenario]["time_limit"],
            "routes": paths, "route_times": calculate_route_times(paths, scene["distance_matrix"])}

def routes_from_state(state):
    return [[state["nodes"][index]["node_id"] for index in route] for route in state["routes"]]

def improve_fixed_set(instance, raw, scenario, selected_ids, initial_routes,
                      seed, passes=3, starts=2, seconds=120, replacement_limit=50,
                      solve_state=None):
    initial = score_routes(raw, scenario, initial_routes, selected_ids)
    if not initial["feasible"]:
        raise ValueError("Initial incumbent must be feasible: " + "; ".join(initial["errors"]))
    if starts < 1 or passes < 0 or seconds <= 0:
        raise ValueError("Invalid recourse search budget")
    best = state_from_routes(instance, scenario, initial_routes)
    best_score = initial
    fixed = build_fixed_n1_decisions(instance, selected_ids)
    protected = [index for index, value in fixed.items() if value == 1]
    solve = solve_state or solve_lower_level_state
    start_time = time.perf_counter()
    deadline = start_time + seconds
    attempts = []
    for repeat in range(starts):
        if time.perf_counter() >= deadline:
            break
        candidate = clone_scenario_state(best)
        rng = random.Random(seed + repeat * 100003)
        if repeat:
            # Change the route backbone, not the fixed first-stage decision.
            for index in rng.sample(protected, min(10, len(protected))):
                remove_node_from_state(candidate, index)
            optional = get_lower_level_removal_candidates(candidate, len(raw["n1_nodes"]), fixed, set())
            for item in rng.sample(optional, min(15, len(optional))):
                remove_node_from_state(candidate, item["node_index"])
        attempt_start = time.perf_counter()
        attempt_deadline = attempt_start + (deadline - attempt_start) / (starts - repeat)
        completed = solve(candidate, len(raw["n1_nodes"]), fixed, set(), CONSENSUS_LAMBDA,
                          rng, passes=passes, deadline=attempt_deadline,
                          stagnation_limit=0, replacement_candidate_limit=replacement_limit)
        routes = routes_from_state(candidate)
        value = score_routes(raw, scenario, routes, selected_ids)
        improved = value["feasible"] and (value["profit"], -sum(value["route_times"])) > (
            best_score["profit"], -sum(best_score["route_times"]))
        if improved:
            best = clone_scenario_state(candidate)
            best_score = value
        attempts.append({"repeat": repeat, "completed_passes": bool(completed),
                         "feasible": value["feasible"], "profit": value["profit"],
                         "accepted": improved, "seconds": time.perf_counter() - attempt_start})
    final_routes = routes_from_state(best)
    final = score_routes(raw, scenario, final_routes, selected_ids)
    if not final["feasible"] or final["profit"] < initial["profit"]:
        raise AssertionError("Incumbent-preservation contract violated")
    return {**final, "initial_profit": initial["profit"], "routes": final_routes,
            "seconds": time.perf_counter() - start_time, "attempts": attempts}

# Build graph features, train the GNN and evaluate candidate sets

def customer_nodes(instance_data):
    return [node for node in instance_data["n1_nodes"] if node["node_id"] != 0]

def depot_node(instance_data):
    for node in instance_data["n1_nodes"]:
        if node["node_id"] == 0:
            return node
    raise ValueError("The instance has no depot with node_id 0.")

def has_stored_reference_routes(instance_data):
    if not instance_data.get("solutions"):
        return False
    for scenario_id in instance_data["scenarios"]:
        solution = instance_data["solutions"].get(scenario_id)
        if not solution or not solution.get("routes"):
            return False
    return True

def ensure_solver_metadata(instance_data, time_limit=0, vehicles=0):
    """Ensure the reused lower-level solver has T and vehicle count available."""
    instance_data.setdefault("solutions", {})
    for scenario_id in sorted(instance_data["scenarios"]):
        solution = instance_data["solutions"].get(scenario_id)
        inferred_time_limit = solution.get("time_limit", 0) if solution else 0
        inferred_vehicles = solution.get("vehicles", 0) if solution else 0
        effective_time_limit = time_limit or inferred_time_limit
        effective_vehicles = vehicles or inferred_vehicles

        if effective_time_limit <= 0:
            raise ValueError(
                "Time limit is missing. Provide --time-limit when the target "
                "file does not contain SOLUTION metadata."
            )
        if effective_vehicles <= 0:
            raise ValueError(
                "Vehicle count is missing. Provide --vehicles when the target "
                "file does not contain SOLUTION metadata."
            )

        if solution is None:
            solution = {
                "scenario_id": scenario_id,
                "time_limit": effective_time_limit,
                "vehicles": effective_vehicles,
                "routes": {},
            }
            instance_data["solutions"][scenario_id] = solution
        else:
            solution["time_limit"] = effective_time_limit
            solution["vehicles"] = effective_vehicles

def reference_common_labels(instance_data, nodes):
    import numpy as np

    if not has_stored_reference_routes(instance_data):
        raise ValueError(
            f"Instance {instance_data['instance_id']} has no stored reference "
            "routes, so it cannot be used as supervised training data."
        )
    selected_ids = set(get_reference_selected_n1_node_ids(instance_data)["common"])
    return np.array(
        [1.0 if node["node_id"] in selected_ids else 0.0 for node in nodes],
        dtype=np.float32,
    )

# Graph features and normalized adjacency

def normalized_knn_adjacency(coordinates, neighbor_count):
    import numpy as np

    count = len(coordinates)
    if count == 1:
        return np.ones((1, 1), dtype=np.float32)

    distances = np.linalg.norm(
        coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :],
        axis=2,
    )
    k = min(max(1, neighbor_count), count - 1)
    neighbor_indexes = np.argsort(distances, axis=1)[:, 1:k + 1]
    neighbor_distances = distances[np.arange(count)[:, None], neighbor_indexes]
    scale = max(float(np.median(neighbor_distances)), 1.0)

    adjacency = np.eye(count, dtype=np.float32)
    for node_index, neighbors in enumerate(neighbor_indexes):
        for neighbor_index in neighbors:
            weight = float(np.exp(-distances[node_index, neighbor_index] / scale))
            adjacency[node_index, neighbor_index] = max(
                adjacency[node_index, neighbor_index],
                weight,
            )
            adjacency[neighbor_index, node_index] = max(
                adjacency[neighbor_index, node_index],
                weight,
            )

    degrees = adjacency.sum(axis=1)
    inv_sqrt_degrees = 1.0 / np.sqrt(np.maximum(degrees, 1e-8))
    return (inv_sqrt_degrees[:, None] * adjacency * inv_sqrt_degrees[None, :]).astype(
        np.float32
    )

def n1_neighborhood_features(coordinates, profits, scale, neighbor_count):
    import numpy as np

    count = len(coordinates)
    if count == 1:
        zeros = np.zeros(1, dtype=np.float32)
        return zeros, zeros, profits / max(float(profits.max()), 1.0)

    distances = np.linalg.norm(
        coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :],
        axis=2,
    )
    np.fill_diagonal(distances, np.inf)
    nearest_distances = distances.min(axis=1)
    k = min(max(1, neighbor_count), count - 1)
    near_indexes = np.argsort(distances, axis=1)[:, :k]
    near_distances = distances[np.arange(count)[:, None], near_indexes]
    near_profits = profits[near_indexes]

    max_profit = max(float(profits.max()), 1.0)
    return (
        (nearest_distances / scale).astype(np.float32),
        (near_distances.mean(axis=1) / scale).astype(np.float32),
        (near_profits.mean(axis=1) / max_profit).astype(np.float32),
    )

def scenario_context_features(instance_data, coordinates, depot, scale):
    import numpy as np

    attraction_by_scenario = []
    nearest_by_scenario = []
    close_share_by_scenario = []
    radius = max(1.0, 0.45 * scale)
    decay = max(1.0, 0.50 * scale)

    for scenario_id in sorted(instance_data["scenarios"]):
        n2_nodes = instance_data["scenarios_n2"].get(scenario_id, [])
        if not n2_nodes:
            attraction_by_scenario.append(np.zeros(len(coordinates), dtype=np.float32))
            nearest_by_scenario.append(np.ones(len(coordinates), dtype=np.float32))
            close_share_by_scenario.append(np.zeros(len(coordinates), dtype=np.float32))
            continue

        n2_coordinates = np.array(
            [[node["x"], node["y"]] for node in n2_nodes],
            dtype=np.float32,
        )
        n2_profits = np.array([node["profit"] for node in n2_nodes], dtype=np.float32)
        distances = np.linalg.norm(
            coordinates[:, np.newaxis, :] - n2_coordinates[np.newaxis, :, :],
            axis=2,
        )
        total_n2_profit = max(float(n2_profits.sum()), 1.0)
        attraction = (np.exp(-distances / decay) @ n2_profits) / total_n2_profit
        nearest = distances.min(axis=1) / scale
        close_share = (distances <= radius).mean(axis=1)
        attraction_by_scenario.append(attraction.astype(np.float32))
        nearest_by_scenario.append(nearest.astype(np.float32))
        close_share_by_scenario.append(close_share.astype(np.float32))

    attraction_matrix = np.vstack(attraction_by_scenario)
    nearest_matrix = np.vstack(nearest_by_scenario)
    close_share_matrix = np.vstack(close_share_by_scenario)
    return [
        attraction_matrix.mean(axis=0),
        attraction_matrix.std(axis=0),
        nearest_matrix.mean(axis=0),
        nearest_matrix.std(axis=0),
        close_share_matrix.mean(axis=0),
        close_share_matrix.std(axis=0),
    ]

def build_graph(instance_id, instance_data, neighbor_count, include_labels):
    import numpy as np

    nodes = customer_nodes(instance_data)
    if not nodes:
        raise ValueError(f"Instance {instance_id} has no n1 customer nodes.")

    depot = depot_node(instance_data)
    coordinates = np.array([[node["x"], node["y"]] for node in nodes], dtype=np.float32)
    depot_coordinates = np.array([depot["x"], depot["y"]], dtype=np.float32)
    profits = np.array([node["profit"] for node in nodes], dtype=np.float32)
    max_profit = max(float(profits.max()), 1.0)

    depot_distances = np.linalg.norm(coordinates - depot_coordinates, axis=1)
    coordinate_span = np.ptp(
        np.vstack([coordinates, depot_coordinates.reshape(1, 2)]),
        axis=0,
    )
    scale = max(float(coordinate_span.max()), float(depot_distances.max()), 1.0)
    nearest_n1, mean_near_n1, near_profit = n1_neighborhood_features(
        coordinates,
        profits,
        scale,
        neighbor_count,
    )
    scenario_context = scenario_context_features(instance_data, coordinates, depot, scale)

    scenario_ids = sorted(instance_data["scenarios"])
    time_limits = [get_time_limit(instance_data, scenario_id) for scenario_id in scenario_ids]
    n2_counts = [len(instance_data["scenarios_n2"][scenario_id]) for scenario_id in scenario_ids]
    vehicles = get_instance_vehicle_count(instance_data)
    distance_normalizer = max(float(depot_distances.max()), 1.0)
    profit_per_depot_distance = profits / np.maximum(depot_distances, 1.0)
    profit_distance_normalizer = max(float(profit_per_depot_distance.max()), 1.0)

    features = np.column_stack(
        [
            (coordinates[:, 0] - depot["x"]) / scale,
            (coordinates[:, 1] - depot["y"]) / scale,
            profits / max_profit,
            depot_distances / distance_normalizer,
            nearest_n1,
            mean_near_n1,
            near_profit,
            profit_per_depot_distance / profit_distance_normalizer,
            *scenario_context,
            np.full(len(nodes), len(scenario_ids) / 16.0, dtype=np.float32),
            np.full(len(nodes), float(np.mean(n2_counts)) / max(len(nodes), 1), dtype=np.float32),
            np.full(len(nodes), float(np.mean(time_limits)) / 2000.0, dtype=np.float32),
            np.full(len(nodes), vehicles / 3.0, dtype=np.float32),
        ]
    ).astype(np.float32)

    labels = reference_common_labels(instance_data, nodes) if include_labels else None
    return {
        "instance_id": instance_id,
        "nodes": nodes,
        "features": features,
        "adjacency": normalized_knn_adjacency(coordinates, neighbor_count),
        "labels": labels,
        "instance_data": instance_data,
    }

def make_gnn_model(feature_count, hidden_dim, dropout, layers):
    """Create the graph neural network in the training process."""
    from torch import nn
    from torch.nn import functional as F

    class GraphConvolution(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.self_linear = nn.Linear(input_dim, output_dim)
            self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)

        def forward(self, features, normalized_adjacency):
            return self.self_linear(features) + self.neighbor_linear(
                normalized_adjacency @ features
            )

    class CommonN1Selector(nn.Module):
        def __init__(self, feature_count, hidden_dim, dropout, layers=3):
            super().__init__()
            if layers not in (2, 3, 4):
                raise ValueError(f"layers must be 2, 3, or 4, got {layers}")
            self.layers = layers
            self.first = GraphConvolution(feature_count, hidden_dim)
            self.first_norm = nn.LayerNorm(hidden_dim)
            self.second = GraphConvolution(hidden_dim, hidden_dim)
            self.second_norm = nn.LayerNorm(hidden_dim)
            if layers >= 3:
                self.third = GraphConvolution(hidden_dim, hidden_dim)
                self.third_norm = nn.LayerNorm(hidden_dim)
            if layers >= 4:
                self.fourth = GraphConvolution(hidden_dim, hidden_dim)
                self.fourth_norm = nn.LayerNorm(hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)
            self.dropout = dropout

        def forward(self, features, normalized_adjacency):
            hidden = F.relu(self.first_norm(self.first(features, normalized_adjacency)))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            first_residual = hidden
            hidden = F.relu(self.second_norm(self.second(hidden, normalized_adjacency)))
            if self.layers == 2:
                hidden = hidden + first_residual
                return self.output(hidden).squeeze(-1)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            second_residual = hidden
            hidden = F.relu(
                self.third_norm(self.third(hidden, normalized_adjacency)) + first_residual
            )
            if self.layers == 4:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
                hidden = F.relu(
                    self.fourth_norm(self.fourth(hidden, normalized_adjacency))
                    + second_residual
                )
            return self.output(hidden).squeeze(-1)

    return CommonN1Selector(feature_count, hidden_dim, dropout, layers)

# Training and prediction

def tensor_graph(graph, feature_mean, feature_std, device):
    import torch

    normalized_features = (graph["features"] - feature_mean) / feature_std
    tensors = {
        "features": torch.tensor(normalized_features, dtype=torch.float32, device=device),
        "adjacency": torch.tensor(graph["adjacency"], dtype=torch.float32, device=device),
    }
    if graph.get("labels") is not None:
        tensors["labels"] = torch.tensor(graph["labels"], dtype=torch.float32, device=device)
    return tensors

def train_model(train_graphs, feature_mean, feature_std, device, *, seed, hidden_dim, layers, dropout):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = make_gnn_model(
        train_graphs[0]["features"].shape[1],
        hidden_dim,
        dropout,
        layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.0001,
    )
    train_tensors = [tensor_graph(graph, feature_mean, feature_std, device) for graph in train_graphs]
    positive_count = sum(float(graph["labels"].sum().item()) for graph in train_tensors)
    total_count = sum(graph["labels"].numel() for graph in train_tensors)
    positive_weight = torch.tensor(
        [(total_count - positive_count) / max(positive_count, 1.0)],
        dtype=torch.float32,
        device=device,
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    for _ in range(600):
        model.train()
        optimizer.zero_grad()
        losses = []
        for graph in train_tensors:
            logits = model(graph["features"], graph["adjacency"])
            losses.append(loss_function(logits, graph["labels"]))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    return model

def predict_probabilities(model, graph, feature_mean, feature_std, device):
    import torch

    model.eval()
    graph_tensors = tensor_graph(graph, feature_mean, feature_std, device)
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(graph_tensors["features"], graph_tensors["adjacency"])
        )
    return probabilities.cpu().numpy()

# Candidate sets and feasibility fallback

def add_candidate(candidate_map, selected_ids, source):
    selected_ids = sorted({int(node_id) for node_id in selected_ids if int(node_id) != 0})
    if not selected_ids:
        return
    key = tuple(selected_ids)
    if key not in candidate_map:
        candidate_map[key] = {
            "selected_ids": selected_ids,
            "sources": [],
        }
    candidate_map[key]["sources"].append(source)

def training_selection_rate(train_graphs):
    import numpy as np

    rates = [float(graph["labels"].mean()) for graph in train_graphs]
    return float(np.median(rates))

def build_gnn_candidates(graph, probabilities, train_graphs, candidate_limit):
    import numpy as np

    candidate_map = {}
    ranked_indexes = np.argsort(-probabilities)
    node_count = len(graph["nodes"])
    expected_count = int(round(float(probabilities.sum())))
    median_count = int(round(training_selection_rate(train_graphs) * node_count))
    threshold_count = int(np.sum(probabilities >= 0.5))

    raw_counts = [
        expected_count,
        threshold_count,
        median_count,
        expected_count - 5,
        expected_count + 5,
        median_count - 5,
        median_count + 5,
        expected_count - 10,
        expected_count + 10,
    ]
    seen_counts = set()
    for raw_count in raw_counts:
        count = min(node_count, max(1, raw_count))
        if count in seen_counts:
            continue
        seen_counts.add(count)
        selected = [graph["nodes"][index]["node_id"] for index in ranked_indexes[:count]]
        add_candidate(candidate_map, selected, f"GNN top {count}")

    for threshold in (0.60, 0.50, 0.40):
        selected = [
            graph["nodes"][index]["node_id"]
            for index, probability in enumerate(probabilities)
            if probability >= threshold
        ]
        add_candidate(candidate_map, selected, f"GNN probability >= {threshold:.2f}")

    limited_map = {}
    for key, candidate in candidate_map.items():
        limited_map[key] = candidate
        if len(limited_map) >= max(1, candidate_limit):
            break
    return limited_map

def build_feasibility_fallback_candidates(graph, probabilities, evaluated_candidate_map):
    """Build smaller top-k GNN sets used only if all screened candidates fail."""
    import numpy as np

    ranked_indexes = np.argsort(-probabilities)
    existing_keys = set(evaluated_candidate_map)
    existing_counts = [
        len(candidate["selected_ids"])
        for candidate in evaluated_candidate_map.values()
    ]
    start_count = min(existing_counts) - 1 if existing_counts else len(graph["nodes"])

    fallback_map = {}
    for count in range(start_count, 0, -1):
        selected = [graph["nodes"][index]["node_id"] for index in ranked_indexes[:count]]
        key = tuple(sorted({int(node_id) for node_id in selected if int(node_id) != 0}))
        if key in existing_keys:
            continue
        add_candidate(fallback_map, selected, f"GNN feasibility fallback top {count}")
    return fallback_map

def evaluate_candidates(graph, candidate_map, seed, fallback_candidate_map):
    """Choose the best feasible candidate, trying smaller sets only if needed."""
    instance = graph["instance_data"]
    base_states = build_greedy_scenario_states(instance)

    def evaluate(candidate, rank):
        return evaluate_fixed_n1_candidate(
            instance, candidate["selected_ids"], " + ".join(candidate["sources"]),
            CONSENSUS_LAMBDA, 1, random.Random(seed + rank * 1009), base_states=base_states,
        )

    results = [evaluate(candidate, rank) for rank, candidate in enumerate(candidate_map.values(), start=1)]
    best = max(results, key=candidate_result_key)
    if best["is_feasible"]:
        return best, True

    for rank, candidate in enumerate(fallback_candidate_map.values(), start=len(results) + 1):
        result = evaluate(candidate, rank)
        if result["is_feasible"]:
            return result, False
    return best, False
