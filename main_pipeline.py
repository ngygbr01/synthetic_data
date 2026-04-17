import json
import time
import random
import os
import signal
from graph_engine import GraphColoringEngine

keep_running = True


def signal_handler(sig, frame):
    global keep_running
    print("\n[!] Stop signal (Ctrl+C). Saving and exiting...")
    keep_running = False


signal.signal(signal.SIGINT, signal_handler)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def validate_python_fast(graph, assignment):
    for u, v in graph.edges():
        if u in assignment and v in assignment and assignment[u] == assignment[v]:
            return False
    return len(assignment) == graph.number_of_nodes()


def get_existing_count(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ------------------------------------------------------------------
# Reasoning builders for non-coloring tasks
# (task_a reasoning comes directly from the engine strategies)
# ------------------------------------------------------------------

def build_reasoning_b(graph, assignment, facts, is_valid, conflict_u=None, conflict_v=None):
    """
    task_b_validation.
    VALID: confirms all edges are conflict-free, notes structural reason.
    INVALID: explains exactly which edge fails and why.
    """
    lines = [
        f"I need to verify whether this coloring is valid by checking every "
        f"edge for color conflicts. "
        f"The graph has {facts['n']} nodes and {facts['m']} edges."
    ]

    if is_valid:
        # Show a real sample of edge checks so the model sees actual verification work
        all_edges = list(graph.edges())
        sample_size = min(12, len(all_edges))
        sample_edges = random.sample(all_edges, sample_size)
        edge_checks = []
        for u, v in sorted(sample_edges):
            cu, cv = assignment[u], assignment[v]
            edge_checks.append(
                f"  Edge ({u},{v}): node {u} has color {cu}, "
                f"node {v} has color {cv} → {'OK ✓' if cu != cv else 'CONFLICT ✗'}"
            )
        lines.append(
            f"Spot-checking {sample_size} of {facts['m']} edges "
            f"(all edges verified internally):"
        )
        lines.extend(edge_checks)
        if facts['m'] > sample_size:
            lines.append(f"  … and {facts['m'] - sample_size} further edges — all conflict-free.")

        if facts['is_bipartite']:
            lines.append(
                "Structural note: the graph is bipartite — "
                "it can always be 2-colored, so a valid coloring was expected."
            )
        elif facts['clique_number'] > 1:
            lines.append(
                f"Structural note: the graph contains a clique of size "
                f"{facts['clique_number']} (nodes {facts['largest_clique']}), "
                f"so at least {facts['clique_number']} colors were necessary — "
                f"all properly assigned here."
            )
        lines.append("Verdict: VALID — no conflicts found.")
    else:
        color = assignment[conflict_u]
        u_neighbors = sorted(graph.neighbors(conflict_u))
        lines.append(
            f"Checking edges… conflict found at edge ({conflict_u}, {conflict_v}): "
            f"both node {conflict_u} and node {conflict_v} are assigned color {color}. "
            f"Node {conflict_u} has neighbors {u_neighbors}, and node {conflict_v} "
            f"is one of them — two adjacent nodes cannot share the same color."
        )
        lines.append("Verdict: INVALID — conflict detected, coloring rejected.")

    return "\n".join(lines)


def build_reasoning_c(graph, assignment, facts, target_node, correct_color, max_colors):
    """
    task_c_missing_color.
    Explains the constraint propagation: why each forbidden color is forbidden,
    then why the chosen color is safe.
    """
    neighbors = sorted(graph.neighbors(target_node))
    neighbor_colors = {nb: assignment[nb] for nb in neighbors if nb in assignment}
    forbidden = sorted(set(neighbor_colors.values()))
    available = [c for c in range(max_colors) if c not in forbidden]

    lines = [
        f"Node {target_node} is uncolored. I need to find a color from "
        f"0 to {max_colors - 1} that does not conflict with any neighbor.",
        f"Node {target_node} has degree {graph.degree(target_node)}, "
        f"with neighbors: {neighbors}.",
        f"Colors already used by these neighbors: {neighbor_colors}.",
        f"Forbidden colors (used by at least one neighbor): {forbidden}.",
        f"Available colors: {available}.",
        f"I choose color {correct_color} — it is not used by any neighbor "
        f"of node {target_node}, so no conflict arises.",
    ]

    # Add a structural note if helpful
    if facts['clique_number'] > 1 and target_node in facts['largest_clique']:
        lines.append(
            f"Note: node {target_node} is part of the largest clique "
            f"{facts['largest_clique']} (size {facts['clique_number']}), "
            f"which is why so many colors are forbidden here."
        )

    return "\n".join(lines)


def build_reasoning_d(graph, facts, k_min):
    """
    task_d_chromatic.
    Builds a genuine structural argument:
    lower bound (clique / odd cycle) + upper bound (Brooks) + solver confirmation.
    """
    lines = [
        f"I need to determine the exact chromatic number χ(G) for this graph "
        f"({facts['n']} nodes, {facts['m']} edges, density={facts['density']})."
    ]

    # Lower bound
    lower = 1
    lb_reasons = []
    if facts['clique_number'] > lower:
        lower = facts['clique_number']
        lb_reasons.append(
            f"A clique of size {facts['clique_number']} exists "
            f"(nodes {facts['largest_clique']}): all nodes in a clique are "
            f"mutually adjacent, so they all need distinct colors → χ(G)≥{lower}."
        )
    if not facts['is_bipartite'] and facts['odd_cycle_example']:
        if 3 > lower:
            lower = 3
        lb_reasons.append(
            f"An odd cycle {facts['odd_cycle_example']} exists: odd cycles "
            f"cannot be 2-colored → χ(G)≥3."
        )
    if not lb_reasons:
        lb_reasons.append("No clique larger than 1 found; lower bound is 1.")

    lines.append("Lower bound analysis:")
    lines.extend(f"  • {r}" for r in lb_reasons)

    # Upper bound
    lines.append("Upper bound analysis:")
    if facts['is_bipartite']:
        lines.append(
            f"  • Graph is bipartite (no odd cycles) → χ(G)≤2 by definition."
        )
    elif facts['is_tree']:
        lines.append(
            f"  • Graph is a tree → χ(G)=2 (trees are always bipartite)."
        )
    else:
        lines.append(
            f"  • Brooks' theorem: χ(G)≤max_degree={facts['max_degree']} "
            f"for any connected graph that is not a complete graph or odd cycle."
        )

    # Solver confirmation
    lines.append(
        f"Solver verification: trying k=1,2,…{k_min} colors. "
        f"With {k_min - 1} color(s) the constraints are unsatisfiable "
        f"(the clique/odd-cycle structure forces more). "
        f"With {k_min} color(s) a valid assignment exists."
    )
    lines.append(f"Conclusion: χ(G) = {k_min}.")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_pipeline(num_tasks=100_000, output_file="100000_dataset.jsonl"):
    global keep_running

    already_done = get_existing_count(output_file)
    if already_done >= num_tasks:
        print(f"--- Already complete! ({already_done}/{num_tasks}) ---")
        return

    print(f"--- SFT Data Factory: {num_tasks} tasks ---")
    if already_done > 0:
        print(f"--- Resuming from task {already_done + 1} ---")

    start_time = time.time()
    tasks_completed = already_done
    strategy_counts = {'backtracking': 0, 'welsh_powell': 0, 'dsatur': 0}

    with open(output_file, "a", encoding="utf-8") as f:
        while tasks_completed < num_tasks and keep_running:

            engine = GraphColoringEngine(min_nodes=10, max_nodes=65)
            G = engine.G
            difficulty = engine.get_difficulty()
            adj_str = engine.get_adjacency_list_string()
            facts = engine.get_graph_facts()

            task_type = random.choices(
                ['task_a_coloring', 'task_b_validation',
                 'task_c_missing_color', 'task_d_chromatic'],
                weights=[0.40, 0.30, 0.20, 0.10]
            )[0]

            k_min, is_exact = engine.get_chromatic_number_z3()

            if not is_exact and task_type == 'task_d_chromatic':
                continue

            max_colors = k_min + random.randint(0, 1)
            success, assignment, trace_text, strategy = engine.solve_with_trace(max_colors)

            if not success:
                continue

            if not validate_python_fast(G, assignment) or \
               not engine.verify_coloring_z3(assignment, max_colors):
                print("[!] Invalid coloring discarded.")
                continue

            strategy_counts[strategy] += 1

            entry = {
                "task_type": task_type,
                "graph_type": engine.g_type,
                "difficulty": difficulty,
                "strategy": strategy,          # NEW: which strategy was used
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "instruction": "",
                "reasoning": "",
                "solution": "",
            }

            # ---- A: full coloring ----
            if task_type == 'task_a_coloring':
                entry["instruction"] = (
                    f"Color this graph using at most {max_colors} color(s) "
                    f"so that no two adjacent nodes share the same color. "
                    f"Show your full reasoning before giving the final assignment.\n"
                    f"Adjacency list: {adj_str}"
                )
                # Prepend structural facts to the strategy trace
                preamble = _build_structural_preamble(facts, k_min, max_colors)
                entry["reasoning"] = preamble + "\n\n" + trace_text
                entry["solution"] = str(assignment)

            # ---- B: validation ----
            elif task_type == 'task_b_validation':
                is_valid = random.choice([True, False])
                test_assignment = assignment.copy()
                conflict_u = conflict_v = None

                if not is_valid:
                    conflict_u, conflict_v = random.choice(list(G.edges()))
                    test_assignment[conflict_u] = test_assignment[conflict_v]

                entry["instruction"] = (
                    f"Is the following coloring valid? "
                    f"(No two adjacent nodes may share a color.)\n"
                    f"Adjacency list: {adj_str}\n"
                    f"Proposed coloring: {test_assignment}"
                )
                entry["reasoning"] = build_reasoning_b(
                    G, test_assignment, facts, is_valid, conflict_u, conflict_v
                )
                entry["solution"] = (
                    "VALID" if is_valid
                    else f"INVALID: conflict at edge ({conflict_u},{conflict_v}) "
                         f"— both nodes assigned color {test_assignment[conflict_u]}."
                )

            # ---- C: missing color ----
            elif task_type == 'task_c_missing_color':
                target_node = random.choice(list(G.nodes()))
                correct_color = assignment[target_node]
                test_assignment = assignment.copy()
                del test_assignment[target_node]

                entry["instruction"] = (
                    f"Node {target_node} is uncolored in the partial assignment below. "
                    f"Determine a valid color (0 to {max_colors - 1}) for it, "
                    f"ensuring no conflict with its neighbors.\n"
                    f"Adjacency list: {adj_str}\n"
                    f"Partial coloring: {test_assignment}"
                )
                entry["reasoning"] = build_reasoning_c(
                    G, assignment, facts, target_node, correct_color, max_colors
                )
                entry["solution"] = f"Node {target_node} → color {correct_color}."

            # ---- D: chromatic number ----
            elif task_type == 'task_d_chromatic':
                entry["instruction"] = (
                    f"Determine the chromatic number χ(G) of this graph — "
                    f"the minimum number of colors needed so that no two adjacent "
                    f"nodes share the same color.\n"
                    f"Adjacency list: {adj_str}"
                )
                entry["reasoning"] = build_reasoning_d(G, facts, k_min)
                entry["solution"] = str(k_min)

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

            tasks_completed += 1
            if tasks_completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = (tasks_completed - already_done) / elapsed
                eta_min = (num_tasks - tasks_completed) / rate / 60 if rate > 0 else 0
                print(
                    f"[{tasks_completed}/{num_tasks}] "
                    f"{rate:.1f} tasks/s | ETA {eta_min:.1f} min | "
                    f"strategies: {strategy_counts}"
                )

    print(f"\n--- Done. Total saved: {get_existing_count(output_file)} ---")
    print(f"Elapsed: {time.time() - start_time:.1f}s")
    print(f"Strategy distribution: {strategy_counts}")


def _build_structural_preamble(facts, k_min, max_colors):
    """
    Graph-theory preamble prepended to every task_a reasoning.
    Establishes WHY a certain number of colors is needed before
    the strategy trace begins.
    `k_min`      — chromatic number (structural lower bound)
    `max_colors` — the budget given in the instruction (k_min or k_min+1)
    """
    lines = [
        f"Structural analysis:",
        f"  • {facts['n']} nodes, {facts['m']} edges, "
        f"density={facts['density']}, max_degree={facts['max_degree']}.",
    ]

    if facts['is_bipartite']:
        lines.append(
            f"  • Graph is bipartite → no odd cycles → χ(G)≤2."
        )
    elif facts['odd_cycle_example']:
        lines.append(
            f"  • Odd cycle {facts['odd_cycle_example']} → χ(G)≥3."
        )

    if facts['clique_number'] >= 2:
        lines.append(
            f"  • Largest clique: {facts['largest_clique']} "
            f"(size {facts['clique_number']}) → χ(G)≥{facts['clique_number']}. "
            f"These nodes must all receive different colors."
        )

    lines.append(f"  • Chromatic number: χ(G) = {k_min}.")
    if max_colors > k_min:
        lines.append(
            f"  • Budget: at most {max_colors} color(s) allowed "
            f"(1 extra over the minimum — any valid {k_min}-coloring also qualifies)."
        )
    else:
        lines.append(f"  • Budget: exactly {max_colors} color(s) — must achieve the minimum.")
    return "\n".join(lines)


if __name__ == "__main__":
    run_pipeline(num_tasks=5_000)