"""
main_pipeline.py  —  javított v4

Fő változások v3 → v4:
  - Kisebb gráfok: task_d → max 12 csúcs (Z3 exact), task_a/b/c → max 20
  - Instruction mezőbe SOHA nem kerül csonkított adjacency lista
  - Task-típusonként helyes gráfméret-hint átadása a GraphColoringEngine-nek
  - task_d: is_exact=False esetén automatikus skip (nem fallback save!)
  - Reasoning konzisztencia: task_a/b pontosabb ellenőrzés
  - Eloszlás tracking: automatikus figyelmeztetés, ha task_d alulreprezentált
  - max_colors logika: k_min alapján, de task_a-nál mindig k_min (nem k_min+1)
    → a reasoning és solution összehangolt lesz
  - Graceful shutdown: SIGINT esetén az aktuális entry befejezése után kilép
"""

import json
import time
import random
import os
import re
import signal
import requests
from graph_engine import GraphColoringEngine

# -----------------------------------------------------------------------
# Konfiguráció
# -----------------------------------------------------------------------

PROVIDERS = [
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "model": "llama-3.3-70b-versatile",
        "rpm_limit": 30,
        "daily_limit": 14400,
    },
    {
        "name": "mistral",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "api_key": os.getenv("MISTRAL_API_KEY", ""),
        "model": "mistral-small-latest",
        "rpm_limit": 5,
        "daily_limit": 500,
    },
]

provider_state = {
    p["name"]: {"daily_used": 0, "last_call_time": 0.0}
    for p in PROVIDERS
}

# Gráfméret korlátok task-típusonként
TASK_NODE_LIMITS = {
    "task_a_coloring":    {"min": 6,  "max": 20, "hint": "medium"},
    "task_b_validation":  {"min": 6,  "max": 20, "hint": "medium"},
    "task_c_missing_color": {"min": 5, "max": 15, "hint": "small"},
    "task_d_chromatic":   {"min": 5,  "max": 12, "hint": "small"},  # Z3 exact csak kis gráfon
}

# Prompt hossz limit — CSAK a LLM-promptban, instruction mezőbe NEM
MAX_LLM_PROMPT_CHARS = 2500

# Reasoning ellenőrzés: ennyi hibánál több → fallback
MAX_REASONING_ERRORS = 2

keep_running = True


def signal_handler(sig, frame):
    global keep_running
    print("\n[!] Stop signal received. Finishing current entry then exiting...")
    keep_running = False


signal.signal(signal.SIGINT, signal_handler)


# -----------------------------------------------------------------------
# Provider kezelés
# -----------------------------------------------------------------------

def get_available_provider():
    for p in PROVIDERS:
        name = p["name"]
        state = provider_state[name]

        if not p["api_key"]:
            continue
        if state["daily_used"] >= p["daily_limit"]:
            continue

        min_gap = 60.0 / p["rpm_limit"]
        elapsed = time.time() - state["last_call_time"]
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

        return p

    return None


def call_llm(prompt: str, max_tokens: int = 600) -> str | None:
    """LLM hívás. A prompt itt csonkítható — ez SOHA nem kerül instruction mezőbe."""
    if len(prompt) > MAX_LLM_PROMPT_CHARS:
        prompt = prompt[:MAX_LLM_PROMPT_CHARS] + "\n...[prompt truncated for length]"

    provider = get_available_provider()
    if provider is None:
        return None

    name = provider["name"]
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": provider["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(provider["url"], headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()

        provider_state[name]["daily_used"] += 1
        provider_state[name]["last_call_time"] = time.time()

        print(f"  [LLM:{name}] #{provider_state[name]['daily_used']} — {len(text)} chars")
        return text

    except requests.exceptions.HTTPError as e:
        status = resp.status_code
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text[:200]
        if status == 429:
            print(f"  [!] {name}: rate limit, waiting 60s.")
            provider_state[name]["last_call_time"] = time.time() + 60
        elif status == 400:
            print(f"  [!] {name}: 400 Bad Request — {err_body}")
        else:
            print(f"  [!] {name} HTTP error {status}: {e}")
        return None
    except Exception as e:
        print(f"  [!] {name} error: {e}")
        return None


# -----------------------------------------------------------------------
# Reasoning konzisztencia ellenőrzés
# -----------------------------------------------------------------------

def verify_reasoning(reasoning: str, assignment: dict, graph) -> tuple[bool, list]:
    """
    Ellenőrzött minták:
      • "node X (color Y)", "node X → color Y", stb. — szín állítások
      • "node X is adjacent to Y" — él-állítások

    FONTOS: task_d-nél az assignment a greedy eredmény, nem a "helyes" megoldás,
    ezért ott a szín-állításokat NEM ellenőrizzük (csak az él-állításokat).
    """
    errors = []

    color_patterns = [
        r'\bnode\s+(\d+)\s+\(color\s+(\d+)\)',
        r'\bnode\s+(\d+)\s+[→\-]+\s*color\s+(\d+)',
        r'\bnode\s+(\d+)\s+(?:has|with|assigned)\s+color\s+(\d+)',
        r'\bnode\s+(\d+):\s+color\s+(\d+)',
        r'\bassign\s+color\s+(\d+)\s+to\s+node\s+(\d+)',   # fordított sorrend
    ]
    for pat in color_patterns:
        for m in re.finditer(pat, reasoning, re.IGNORECASE):
            grps = m.groups()
            # "assign color Y to node X" esetén fordított sorrend
            if 'assign' in pat:
                color_str, node_str = grps[0], grps[1]
            else:
                node_str, color_str = grps[0], grps[1]
            node = int(node_str)
            claimed_color = int(color_str)
            if node in assignment and assignment[node] != claimed_color:
                errors.append(
                    f"Color mismatch — node {node}: reasoning claims {claimed_color}, "
                    f"actual {assignment[node]}"
                )

    adj_patterns = [
        r'\bnode\s+(\d+)\s+is\s+adjacent\s+to\s+(?:node\s+)?(\d+)',
        r'\bnodes?\s+(\d+)\s+and\s+(\d+)\s+are\s+adjacent',
        r'\bedge\s+\((\d+),\s*(\d+)\)',
    ]
    for pat in adj_patterns:
        for m in re.finditer(pat, reasoning, re.IGNORECASE):
            u, v = int(m.group(1)), int(m.group(2))
            if (u in graph.nodes() and v in graph.nodes()
                    and not graph.has_edge(u, v) and not graph.has_edge(v, u)):
                errors.append(
                    f"Hallucinated edge ({u},{v}): claimed adjacent but no edge exists"
                )

    return len(errors) <= MAX_REASONING_ERRORS, errors


# -----------------------------------------------------------------------
# Prompt csonkítás — CSAK LLM-prompthoz
# -----------------------------------------------------------------------

def truncate_adj_for_prompt(adj_str: str, n_nodes: int) -> str:
    """
    Adj lista csonkítása kizárólag a LLM-prompthoz.
    Az instruction mezőbe MINDIG a teljes adj_str kerül.
    """
    limits = {
        (0, 10): 1500,
        (10, 15): 1000,
        (15, 20): 700,
        (20, 99): 450,
    }
    max_chars = 450
    for (lo, hi), lim in limits.items():
        if lo <= n_nodes < hi:
            max_chars = lim
            break

    if len(adj_str) <= max_chars:
        return adj_str
    truncated = adj_str[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + " ...[truncated]"


def truncate_assign_for_prompt(assignment: dict, max_items: int = 15) -> str:
    items = list(assignment.items())
    if len(items) <= max_items:
        return str(dict(items))
    return str(dict(items[:max_items])) + f" ...[+{len(items) - max_items} more]"


# -----------------------------------------------------------------------
# Prompt építők  —  csonkított adj/assign CSAK a LLM-nek
# -----------------------------------------------------------------------

def build_prompt_a(adj_str, assignment, max_colors, facts, k_min):
    n = facts['n']
    adj_short = truncate_adj_for_prompt(adj_str, n)
    assign_str = truncate_assign_for_prompt(assignment)
    clique_str = str(facts['largest_clique'][:8]) + ('…' if len(facts['largest_clique']) > 8 else '')

    return f"""Graph coloring expert task. Explain HOW to solve this step by step.

GRAPH: {n} nodes, {facts['m']} edges, density={facts['density']}
Clique size: {facts['clique_number']} {clique_str}
Bipartite: {facts['is_bipartite']}, χ(G)={k_min}, budget={max_colors} colors

ADJACENCY: {adj_short}
SOLUTION: {assign_str}

Write 150-250 word reasoning trace:
1. Why χ(G)={k_min} (clique/odd-cycle lower bound, Brooks upper bound)
2. Coloring strategy and key node decisions
3. Confirm no conflicts — only mention adjacencies that actually exist in the graph above
Reasoning only, no preamble."""


def build_prompt_b(adj_str, test_assignment, facts, is_valid, conflict_u, conflict_v):
    n = facts['n']
    adj_short = truncate_adj_for_prompt(adj_str, n)
    verdict = "VALID" if is_valid else f"INVALID — conflict at edge ({conflict_u},{conflict_v})"
    conflict_info = ""
    if not is_valid:
        conflict_info = f"\nBoth node {conflict_u} and {conflict_v} have color {test_assignment[conflict_u]}."
    assign_str = truncate_assign_for_prompt(test_assignment)

    return f"""Graph coloring validation task.

GRAPH: {n} nodes, {facts['m']} edges
ADJACENCY: {adj_short}
COLORING: {assign_str}
VERDICT: {verdict}{conflict_info}

Write 100-180 word reasoning trace:
1. What to check (adjacent nodes must differ)
2. Sample 3-5 specific edge checks with colors — only use edges from the adjacency list above
3. {'Exact conflict explanation' if not is_valid else 'Why valid coloring expected structurally'}
4. Final verdict
Reasoning only."""


def build_prompt_c(adj_str, assignment, facts, target_node, correct_color, max_colors, graph):
    n = facts['n']
    adj_short = truncate_adj_for_prompt(adj_str, n)
    neighbors = sorted(graph.neighbors(target_node))
    neighbor_colors = {nb: assignment[nb] for nb in neighbors if nb in assignment}
    forbidden = sorted(set(neighbor_colors.values()))
    available = [c for c in range(max_colors) if c not in forbidden]

    return f"""Missing node color task.

GRAPH: {n} nodes, {facts['m']} edges
ADJACENCY: {adj_short}
TARGET NODE: {target_node} (neighbors: {neighbors[:10]}{'...' if len(neighbors) > 10 else ''})
NEIGHBOR COLORS: {neighbor_colors}
FORBIDDEN: {forbidden}, AVAILABLE: {available}
CORRECT COLOR: {correct_color}

Write 100-160 word reasoning trace:
1. List neighbors and their colors
2. Explain why each forbidden color is blocked
3. Why color {correct_color} is chosen
4. Confirm no conflict
Reasoning only."""


def build_prompt_d(adj_str, facts, k_min):
    n = facts['n']
    adj_short = truncate_adj_for_prompt(adj_str, n)
    clique_str = str(facts['largest_clique'][:8]) + ('…' if len(facts['largest_clique']) > 8 else '')
    odd_cycle = facts.get('odd_cycle_example', [])
    odd_cycle_str = str(odd_cycle[:6]) + ('…' if len(odd_cycle) > 6 else '')

    return f"""Chromatic number determination task.

GRAPH: {n} nodes, {facts['m']} edges, density={facts['density']}
Max degree: {facts['max_degree']}, Bipartite: {facts['is_bipartite']}, Tree: {facts['is_tree']}
Clique: {clique_str} (size {facts['clique_number']})
Odd cycle: {odd_cycle_str}
ADJACENCY: {adj_short}
ANSWER: χ(G) = {k_min}

Write 150-220 word reasoning trace:
1. Lower bound: clique → χ≥{facts['clique_number']}, odd cycle argument
2. Upper bound: Brooks theorem or bipartite/tree argument
3. Why {k_min - 1} colors insufficient
4. Why {k_min} colors sufficient
5. Conclusion: χ(G)={k_min}
Reasoning only."""


# -----------------------------------------------------------------------
# Fallback sablon reasoning-ek
# -----------------------------------------------------------------------

def build_fallback_a(facts, k_min, max_colors, trace_text):
    lines = [
        f"Structural analysis: {facts['n']} nodes, {facts['m']} edges, "
        f"density={facts['density']}, max_degree={facts['max_degree']}.",
    ]
    if facts['is_bipartite']:
        lines.append("Graph is bipartite → χ(G)=2.")
    elif facts.get('odd_cycle_example'):
        lines.append(f"Odd cycle {facts['odd_cycle_example']} → χ(G)≥3.")
    if facts['clique_number'] >= 2:
        lines.append(f"Largest clique: {facts['largest_clique']} → χ(G)≥{facts['clique_number']}.")
    lines.append(f"χ(G)={k_min}, budget={max_colors}.\n")
    return "\n".join(lines) + trace_text


def build_fallback_b(facts, is_valid, conflict_u, conflict_v, test_assignment):
    if is_valid:
        return (f"Checking all {facts['m']} edges for color conflicts. "
                f"All adjacent node pairs have different colors. Verdict: VALID.")
    else:
        return (f"Checking edges… conflict found at ({conflict_u},{conflict_v}): "
                f"both assigned color {test_assignment[conflict_u]}. Verdict: INVALID.")


def build_fallback_c(graph, assignment, target_node, correct_color, max_colors):
    neighbors = sorted(graph.neighbors(target_node))
    neighbor_colors = {nb: assignment[nb] for nb in neighbors if nb in assignment}
    forbidden = sorted(set(neighbor_colors.values()))
    available = [c for c in range(max_colors) if c not in forbidden]
    return (f"Node {target_node} neighbors: {neighbors}. "
            f"Neighbor colors: {neighbor_colors}. "
            f"Forbidden: {forbidden}. Available: {available}. "
            f"Assign color {correct_color} — no conflict.")


def build_fallback_d(facts, k_min):
    lines = [f"Lower bound: clique size {facts['clique_number']} → χ(G)≥{facts['clique_number']}."]
    if facts.get('odd_cycle_example'):
        lines.append(f"Odd cycle {facts['odd_cycle_example']} → χ(G)≥3.")
    lines.append(f"Upper bound (Brooks): χ(G)≤{facts['max_degree']}.")
    lines.append(f"Solver confirms: {k_min - 1} colors insufficient, {k_min} sufficient.")
    lines.append(f"Conclusion: χ(G) = {k_min}.")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Segédfüggvények
# -----------------------------------------------------------------------

def validate_python_fast(graph, assignment):
    """Gyors Python-alapú ellenőrzés Z3 nélkül."""
    if len(assignment) != graph.number_of_nodes():
        return False
    for u, v in graph.edges():
        if u in assignment and v in assignment and assignment[u] == assignment[v]:
            return False
    return True


def get_existing_count(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def print_status(tasks_completed, num_tasks, start_time, already_done,
                 llm_success, llm_fallback, reasoning_rejected, strategy_counts,
                 task_type_counts):
    elapsed = time.time() - start_time
    new_done = tasks_completed - already_done
    rate = new_done / max(elapsed, 1)
    eta_min = (num_tasks - tasks_completed) / max(rate, 0.001) / 60
    llm_pct = llm_success / max(1, llm_success + llm_fallback) * 100

    print(f"\n[{tasks_completed}/{num_tasks}] "
          f"{rate:.2f} entries/s | ETA {eta_min:.0f} min | "
          f"LLM: {llm_success} ({llm_pct:.0f}%) | "
          f"Fallback: {llm_fallback} | Rejected reasoning: {reasoning_rejected}")
    print(f"  Strategies: {strategy_counts}")
    print(f"  Task types: {task_type_counts}")

    # Figyelmeztetés ha task_d nagyon kevés
    total = sum(task_type_counts.values())
    d_count = task_type_counts.get('task_d_chromatic', 0)
    if total > 20 and d_count / total < 0.05:
        print(f"  [WARN] task_d underrepresented ({d_count}/{total}). "
              f"Consider reducing graph size for task_d or increasing its weight.")

    for p in PROVIDERS:
        name = p["name"]
        used = provider_state[name]["daily_used"]
        key_status = "✓" if p["api_key"] else "✗ (no key)"
        print(f"  {name}: {used}/{p['daily_limit']} {key_status}")


# -----------------------------------------------------------------------
# Fő pipeline
# -----------------------------------------------------------------------

def run_pipeline(num_tasks: int = 1000, output_file: str = "dataset.jsonl"):
    global keep_running

    already_done = get_existing_count(output_file)
    if already_done >= num_tasks:
        print(f"--- Already complete! ({already_done}/{num_tasks}) ---")
        return

    print(f"--- Graph Coloring SFT Factory: {num_tasks} tasks ---")
    print(f"--- Graph sizes: task_d≤12, task_a/b≤20, task_c≤15 nodes ---")
    print(f"--- Active providers: {[p['name'] for p in PROVIDERS if p['api_key']]} ---")
    if already_done > 0:
        print(f"--- Resuming from task {already_done + 1} ---")

    start_time = time.time()
    tasks_completed = already_done
    llm_success = 0
    llm_fallback = 0
    reasoning_rejected = 0
    strategy_counts = {'backtracking': 0, 'welsh_powell': 0, 'dsatur': 0}
    task_type_counts = {
        'task_a_coloring': 0,
        'task_b_validation': 0,
        'task_c_missing_color': 0,
        'task_d_chromatic': 0,
    }
    skipped = 0

    with open(output_file, "a", encoding="utf-8") as f:
        while tasks_completed < num_tasks and keep_running:

            # Task típus kiválasztása
            task_type = random.choices(
                ['task_a_coloring', 'task_b_validation',
                 'task_c_missing_color', 'task_d_chromatic'],
                weights=[0.35, 0.25, 0.25, 0.15]
            )[0]

            # Gráfméret meghatározása task-típus szerint
            limits = TASK_NODE_LIMITS[task_type]
            engine = GraphColoringEngine(
                min_nodes=limits["min"],
                max_nodes=limits["max"],
                graph_size_hint=limits["hint"]
            )
            G = engine.G
            n_nodes = G.number_of_nodes()
            difficulty = engine.get_difficulty()

            # Teljes adjacency lista — MINDIG az instruction mezőbe kerül
            adj_str_full = engine.get_adjacency_list_string()
            facts = engine.get_graph_facts()

            # Kromatikus szám meghatározása
            k_min, is_exact = engine.get_chromatic_number_z3(
                max_mem_mb=4096,
                timeout_ms=4000
            )

            # task_d CSAK pontos k_min-nel megy
            if task_type == 'task_d_chromatic' and not is_exact:
                skipped += 1
                if skipped % 20 == 0:
                    print(f"  [INFO] Skipped {skipped} task_d entries (Z3 not exact) — "
                          f"graph sizes might need further reduction.")
                continue

            # max_colors meghatározása — task_a-nál k_min (konzisztens reasoning!)
            if task_type == 'task_a_coloring':
                max_colors = k_min  # csak k_min — a reasoning és solution egyezik
            else:
                max_colors = k_min + random.randint(0, 1)

            # Megoldás generálása
            success, assignment, trace_text, strategy = engine.solve_with_trace(max_colors)

            if not success:
                continue

            # Kettős validáció
            if not validate_python_fast(G, assignment):
                print("[!] Invalid coloring (Python check), skipping.")
                continue

            if not engine.verify_coloring_z3(assignment, max_colors):
                print("[!] Invalid coloring (Z3 check), skipping.")
                continue

            strategy_counts[strategy] += 1
            task_type_counts[task_type] += 1

            entry = {
                "task_type": task_type,
                "graph_type": engine.g_type,
                "difficulty": difficulty,
                "strategy": strategy,
                "nodes": n_nodes,
                "edges": G.number_of_edges(),
                "chromatic_number": k_min,
                "is_exact_chromatic": is_exact,
                "llm_generated": False,
                "instruction": "",
                "reasoning": "",
                "solution": "",
            }

            # ---- A: teljes színezés ----
            if task_type == 'task_a_coloring':
                entry["instruction"] = (
                    f"Color this graph using at most {max_colors} color(s) "
                    f"so that no two adjacent nodes share the same color. "
                    f"Show your full reasoning before giving the final assignment.\n"
                    f"Adjacency list: {adj_str_full}"
                )
                prompt = build_prompt_a(adj_str_full, assignment, max_colors, facts, k_min)
                llm_trace = call_llm(prompt, max_tokens=500)

                if llm_trace:
                    ok, errs = verify_reasoning(llm_trace, assignment, G)
                    if ok:
                        entry["reasoning"] = llm_trace
                        entry["llm_generated"] = True
                        llm_success += 1
                    else:
                        print(f"  [!] Reasoning rejected ({len(errs)} errors): {errs[:2]}")
                        reasoning_rejected += 1
                        entry["reasoning"] = build_fallback_a(facts, k_min, max_colors, trace_text)
                        llm_fallback += 1
                else:
                    entry["reasoning"] = build_fallback_a(facts, k_min, max_colors, trace_text)
                    llm_fallback += 1
                entry["solution"] = str(assignment)

            # ---- B: validáció ----
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
                    f"Adjacency list: {adj_str_full}\n"
                    f"Proposed coloring: {test_assignment}"
                )
                prompt = build_prompt_b(
                    adj_str_full, test_assignment, facts, is_valid, conflict_u, conflict_v)
                llm_trace = call_llm(prompt, max_tokens=400)

                if llm_trace:
                    ok, errs = verify_reasoning(llm_trace, test_assignment, G)
                    if ok:
                        entry["reasoning"] = llm_trace
                        entry["llm_generated"] = True
                        llm_success += 1
                    else:
                        print(f"  [!] Reasoning rejected ({len(errs)} errors): {errs[:2]}")
                        reasoning_rejected += 1
                        entry["reasoning"] = build_fallback_b(
                            facts, is_valid, conflict_u, conflict_v, test_assignment)
                        llm_fallback += 1
                else:
                    entry["reasoning"] = build_fallback_b(
                        facts, is_valid, conflict_u, conflict_v, test_assignment)
                    llm_fallback += 1

                entry["solution"] = (
                    "VALID" if is_valid
                    else f"INVALID: conflict at edge ({conflict_u},{conflict_v}) "
                         f"— both nodes assigned color {test_assignment[conflict_u]}."
                )

            # ---- C: hiányzó szín ----
            elif task_type == 'task_c_missing_color':
                target_node = random.choice(list(G.nodes()))
                correct_color = assignment[target_node]
                test_assignment = assignment.copy()
                del test_assignment[target_node]

                entry["instruction"] = (
                    f"Node {target_node} is uncolored in the partial assignment below. "
                    f"Determine a valid color (0 to {max_colors - 1}) for it, "
                    f"ensuring no conflict with its neighbors.\n"
                    f"Adjacency list: {adj_str_full}\n"
                    f"Partial coloring: {test_assignment}"
                )
                prompt = build_prompt_c(
                    adj_str_full, assignment, facts, target_node, correct_color, max_colors, G)
                llm_trace = call_llm(prompt, max_tokens=400)

                if llm_trace:
                    ok, errs = verify_reasoning(llm_trace, assignment, G)
                    if ok:
                        entry["reasoning"] = llm_trace
                        entry["llm_generated"] = True
                        llm_success += 1
                    else:
                        print(f"  [!] Reasoning rejected ({len(errs)} errors): {errs[:2]}")
                        reasoning_rejected += 1
                        entry["reasoning"] = build_fallback_c(
                            G, assignment, target_node, correct_color, max_colors)
                        llm_fallback += 1
                else:
                    entry["reasoning"] = build_fallback_c(
                        G, assignment, target_node, correct_color, max_colors)
                    llm_fallback += 1
                entry["solution"] = f"Node {target_node} → color {correct_color}."

            # ---- D: kromatikus szám (csak is_exact=True esetén) ----
            elif task_type == 'task_d_chromatic':
                entry["instruction"] = (
                    f"Determine the chromatic number χ(G) of this graph — "
                    f"the minimum number of colors needed so that no two adjacent "
                    f"nodes share the same color.\n"
                    f"Adjacency list: {adj_str_full}"
                )
                prompt = build_prompt_d(adj_str_full, facts, k_min)
                llm_trace = call_llm(prompt, max_tokens=450)

                if llm_trace:
                    # task_d: csak él-állításokat ellenőrzünk, szín-állításokat nem
                    # (az assignment itt greedy, nem "a" helyes megoldás)
                    _, errs = verify_reasoning(llm_trace, {}, G)
                    if len(errs) <= MAX_REASONING_ERRORS:
                        entry["reasoning"] = llm_trace
                        entry["llm_generated"] = True
                        llm_success += 1
                    else:
                        print(f"  [!] Reasoning rejected ({len(errs)} errors): {errs[:2]}")
                        reasoning_rejected += 1
                        entry["reasoning"] = build_fallback_d(facts, k_min)
                        llm_fallback += 1
                else:
                    entry["reasoning"] = build_fallback_d(facts, k_min)
                    llm_fallback += 1
                entry["solution"] = str(k_min)

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            tasks_completed += 1

            if tasks_completed % 25 == 0:
                print_status(tasks_completed, num_tasks, start_time, already_done,
                             llm_success, llm_fallback, reasoning_rejected,
                             strategy_counts, task_type_counts)

    total = get_existing_count(output_file)
    llm_pct = llm_success / max(1, llm_success + llm_fallback) * 100
    print(f"\n--- Done. Total: {total} entries ---")
    print(f"LLM trace: {llm_success} ({llm_pct:.0f}%) | Fallback: {llm_fallback} | "
          f"Rejected: {reasoning_rejected}")
    print(f"Strategies: {strategy_counts}")
    print(f"Task types: {task_type_counts}")


if __name__ == "__main__":
    run_pipeline(num_tasks=1000, output_file="dataset.jsonl")