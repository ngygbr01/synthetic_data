"""
main_pipeline.py  —  LLM-trace verzió
Több provider (Groq, HuggingFace, Mistral) automatikus váltással.
A Python csak a gráfot + helyes megoldást generálja (Z3-verifikálva),
a reasoning trace-t az LLM írja — ez adja az értéket.
"""

import json
import time
import random
import os
import signal
import requests
from graph_engine import GraphColoringEngine

# -----------------------------------------------------------------------
# Konfiguráció — töltsd ki az API kulcsokat
# -----------------------------------------------------------------------

PROVIDERS = [
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": os.getenv("GROQ_API_KEY", ""),        # https://console.groq.com
        "model": "llama3-70b-8192",
        "rpm_limit": 30,          # ingyenes: 30 req/perc
        "daily_limit": 14400,
        "headers_extra": {},
    },
    {
        "name": "huggingface",
        "url": "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1/chat/completions",
        "api_key": os.getenv("HF_API_KEY", ""),           # https://huggingface.co/settings/tokens
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "rpm_limit": 10,
        "daily_limit": 1000,
        "headers_extra": {},
    },
    {
        "name": "mistral",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "api_key": os.getenv("MISTRAL_API_KEY", ""),      # https://console.mistral.ai
        "model": "mistral-small-latest",
        "rpm_limit": 5,
        "daily_limit": 500,
        "headers_extra": {},
    },
]

# Provider state — követi a napi használatot és az utolsó hívás idejét
provider_state = {
    p["name"]: {"daily_used": 0, "last_call_time": 0.0}
    for p in PROVIDERS
}

keep_running = True


def signal_handler(sig, frame):
    global keep_running
    print("\n[!] Stop signal. Mentés és kilépés...")
    keep_running = False


signal.signal(signal.SIGINT, signal_handler)


# -----------------------------------------------------------------------
# Provider kezelés
# -----------------------------------------------------------------------

def get_available_provider():
    """Visszaadja az első elérhető providert (daily limit + rate limit alapján)."""
    for p in PROVIDERS:
        name = p["name"]
        state = provider_state[name]

        if not p["api_key"]:
            continue  # nincs API kulcs beállítva

        if state["daily_used"] >= p["daily_limit"]:
            continue  # napi limit kimerült

        # Rate limit: várjuk meg ha túl hamar hívnánk
        min_gap = 60.0 / p["rpm_limit"]
        elapsed = time.time() - state["last_call_time"]
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)

        return p

    return None  # egyik sem elérhető


def call_llm(prompt: str, max_tokens: int = 900) -> str | None:
    """
    Meghívja az első elérhető providert.
    Visszatérési érték: a modell szöveges válasza, vagy None ha minden provider hibás.
    """
    provider = get_available_provider()
    if provider is None:
        print("[!] Egyik provider sem elérhető (limit vagy hiányzó kulcs).")
        return None

    name = provider["name"]
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
        **provider.get("headers_extra", {}),
    }
    payload = {
        "model": provider["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,   # alacsony = konzisztensebb reasoning
    }

    try:
        resp = requests.post(provider["url"], headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()

        provider_state[name]["daily_used"] += 1
        provider_state[name]["last_call_time"] = time.time()

        print(f"  [LLM:{name}] #{provider_state[name]['daily_used']} — {len(text)} karakter")
        return text

    except requests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            print(f"  [!] {name}: rate limit, provider kihagyva.")
            provider_state[name]["last_call_time"] = time.time() + 60
        else:
            print(f"  [!] {name} HTTP hiba: {e}")
        return None
    except Exception as e:
        print(f"  [!] {name} hiba: {e}")
        return None


# -----------------------------------------------------------------------
# Prompt építők — minden task-típushoz
# -----------------------------------------------------------------------

def build_prompt_a(adj_str, assignment, max_colors, facts, k_min):
    """Task A: teljes gráf-színezés reasoning."""
    return f"""You are an expert in graph theory. A graph coloring problem was solved correctly.
Your job: write a detailed, step-by-step reasoning trace that explains HOW to arrive at this solution.
Think like a human expert working through the problem — show your thought process, not just the result.

GRAPH FACTS:
- Nodes: {facts['n']}, Edges: {facts['m']}, Density: {facts['density']}
- Max degree: {facts['max_degree']}, Is bipartite: {facts['is_bipartite']}
- Largest clique: {facts['largest_clique']} (size {facts['clique_number']})
- Chromatic number χ(G) = {k_min}, color budget: {max_colors}
- Odd cycle example: {facts.get('odd_cycle_example', [])}

ADJACENCY LIST: {adj_str}

CORRECT SOLUTION (verified): {assignment}

Write a reasoning trace (200-350 words) that:
1. Analyzes the graph structure (clique, bipartite, density)
2. Explains why χ(G) = {k_min} (lower and upper bound argument)
3. Describes the coloring strategy chosen and why
4. Walks through the key coloring decisions node by node (focus on the hardest nodes)
5. Confirms no conflicts exist

Do NOT just describe the answer — reason through it as if solving it fresh.
Respond with the reasoning trace only, no preamble."""


def build_prompt_b(adj_str, test_assignment, facts, is_valid, conflict_u, conflict_v):
    """Task B: validáció reasoning."""
    verdict = "VALID" if is_valid else f"INVALID (conflict at edge ({conflict_u},{conflict_v}))"
    conflict_info = ""
    if not is_valid:
        conflict_info = f"\nConflict location: edge ({conflict_u},{conflict_v}) — both nodes have color {test_assignment[conflict_u]}."

    return f"""You are an expert in graph theory. A graph coloring validation was performed.
Your job: write a detailed reasoning trace explaining HOW to verify whether this coloring is valid.

GRAPH FACTS:
- Nodes: {facts['n']}, Edges: {facts['m']}
- Is bipartite: {facts['is_bipartite']}, Clique number: {facts['clique_number']}

ADJACENCY LIST: {adj_str}
PROPOSED COLORING: {test_assignment}
VERDICT: {verdict}{conflict_info}

Write a reasoning trace (150-250 words) that:
1. States what needs to be checked (every edge must connect differently-colored nodes)
2. Shows spot-checks of specific edges with their colors
3. If INVALID: explains exactly which edge fails and why it's a conflict
4. If VALID: notes structural reasons why a valid coloring was expected (bipartite, clique size, etc.)
5. States the final verdict clearly

Respond with the reasoning trace only, no preamble."""


def build_prompt_c(adj_str, assignment, facts, target_node, correct_color, max_colors):
    """Task C: hiányzó szín reasoning."""
    neighbors = sorted(__import__('networkx').Graph(
        [(u, v) for u, v in []]  # placeholder, nincs graph objektum itt
    ).neighbors(target_node)) if False else []

    # Rekonstruáljuk a szükséges infót az assignment-ből
    forbidden_colors = []
    neighbor_color_map = {}
    for node, color in assignment.items():
        if node != target_node:
            neighbor_color_map[node] = color

    return f"""You are an expert in graph theory. A missing node color must be determined.
Your job: write a detailed reasoning trace explaining HOW to find the correct color for the missing node.

GRAPH FACTS:
- Nodes: {facts['n']}, Edges: {facts['m']}
- Max degree: {facts['max_degree']}, Clique: {facts['largest_clique']}

ADJACENCY LIST: {adj_str}
PARTIAL COLORING (node {target_node} is missing): { {k:v for k,v in assignment.items() if k != target_node} }
COLOR BUDGET: 0 to {max_colors - 1}
CORRECT COLOR FOR NODE {target_node}: {correct_color}

Write a reasoning trace (150-250 words) that:
1. Identifies all neighbors of node {target_node} from the adjacency list
2. Lists the colors used by those neighbors (these are forbidden)
3. Lists the available colors
4. Explains why color {correct_color} is chosen (smallest available, or structural reason)
5. If node {target_node} is in the largest clique, notes why so many colors are forbidden

Respond with the reasoning trace only, no preamble."""


def build_prompt_d(adj_str, facts, k_min):
    """Task D: kromatikus szám reasoning."""
    return f"""You are an expert in graph theory. The chromatic number of a graph must be determined.
Your job: write a detailed reasoning trace explaining HOW to determine χ(G) = {k_min}.

GRAPH FACTS:
- Nodes: {facts['n']}, Edges: {facts['m']}, Density: {facts['density']}
- Max degree: {facts['max_degree']}, Min degree: {facts['min_degree']}
- Is bipartite: {facts['is_bipartite']}, Is tree: {facts['is_tree']}
- Largest clique: {facts['largest_clique']} (size {facts['clique_number']})
- Odd cycle example: {facts.get('odd_cycle_example', [])}

ADJACENCY LIST: {adj_str}
ANSWER: χ(G) = {k_min}

Write a reasoning trace (200-300 words) that:
1. Establishes the lower bound: clique number gives χ(G) ≥ {facts['clique_number']}, odd cycle (if exists) gives χ(G) ≥ 3
2. Establishes the upper bound: Brooks' theorem (χ(G) ≤ max_degree = {facts['max_degree']}), or bipartite/tree argument
3. Explains why {k_min - 1} colors are NOT sufficient (structural argument)
4. Explains why {k_min} colors ARE sufficient (existence of valid assignment)
5. States the conclusion: χ(G) = {k_min}

Respond with the reasoning trace only, no preamble."""


# -----------------------------------------------------------------------
# Fallback: ha az LLM nem elérhető, visszaesünk a sablon trace-re
# -----------------------------------------------------------------------

def build_fallback_reasoning_a(facts, k_min, max_colors, trace_text):
    preamble = (
        f"Structural analysis:\n"
        f"  • {facts['n']} nodes, {facts['m']} edges, density={facts['density']}, max_degree={facts['max_degree']}.\n"
    )
    if facts['is_bipartite']:
        preamble += "  • Graph is bipartite → χ(G)≤2.\n"
    elif facts.get('odd_cycle_example'):
        preamble += f"  • Odd cycle {facts['odd_cycle_example']} → χ(G)≥3.\n"
    if facts['clique_number'] >= 2:
        preamble += (f"  • Largest clique: {facts['largest_clique']} "
                     f"(size {facts['clique_number']}) → χ(G)≥{facts['clique_number']}.\n")
    preamble += f"  • χ(G) = {k_min}, budget: {max_colors} color(s)."
    return preamble + "\n\n" + trace_text


# -----------------------------------------------------------------------
# Segédfüggvények
# -----------------------------------------------------------------------

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


def print_provider_status():
    print("\n  Provider státusz:")
    for p in PROVIDERS:
        name = p["name"]
        used = provider_state[name]["daily_used"]
        limit = p["daily_limit"]
        key_ok = "✓" if p["api_key"] else "✗ (nincs kulcs)"
        print(f"    {name}: {used}/{limit} req {key_ok}")


# -----------------------------------------------------------------------
# Fő pipeline
# -----------------------------------------------------------------------

def run_pipeline(num_tasks=100_000, output_file="llm_dataset.jsonl"):
    global keep_running

    already_done = get_existing_count(output_file)
    if already_done >= num_tasks:
        print(f"--- Már kész! ({already_done}/{num_tasks}) ---")
        return

    print(f"--- LLM-trace SFT Factory: {num_tasks} feladat ---")
    print(f"--- Aktív providerek: {[p['name'] for p in PROVIDERS if p['api_key']]} ---")
    if already_done > 0:
        print(f"--- Folytatás: {already_done + 1}. feladattól ---")

    start_time = time.time()
    tasks_completed = already_done
    llm_success = 0
    llm_fallback = 0
    strategy_counts = {'backtracking': 0, 'welsh_powell': 0, 'dsatur': 0}

    with open(output_file, "a", encoding="utf-8") as f:
        while tasks_completed < num_tasks and keep_running:

            # --- Gráf generálás ---
            engine = GraphColoringEngine(min_nodes=10, max_nodes=65)
            G = engine.G
            difficulty = engine.get_difficulty()
            adj_str = engine.get_adjacency_list_string()
            facts = engine.get_graph_facts()

            task_type = random.choices(
                ['task_a_coloring', 'task_b_validation',
                 'task_c_missing_color', 'task_d_chromatic'],
                weights=[0.35, 0.25, 0.25, 0.15]
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
                print("[!] Érvénytelen színezés, kihagyva.")
                continue

            strategy_counts[strategy] += 1

            # --- Entry összeállítás ---
            entry = {
                "task_type": task_type,
                "graph_type": engine.g_type,
                "difficulty": difficulty,
                "strategy": strategy,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "llm_generated": False,   # frissítjük ha az LLM sikerül
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
                    f"Adjacency list: {adj_str}"
                )
                prompt = build_prompt_a(adj_str, assignment, max_colors, facts, k_min)
                llm_trace = call_llm(prompt, max_tokens=900)
                if llm_trace:
                    entry["reasoning"] = llm_trace
                    entry["llm_generated"] = True
                    llm_success += 1
                else:
                    entry["reasoning"] = build_fallback_reasoning_a(
                        facts, k_min, max_colors, trace_text)
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
                    f"Adjacency list: {adj_str}\n"
                    f"Proposed coloring: {test_assignment}"
                )
                prompt = build_prompt_b(
                    adj_str, test_assignment, facts, is_valid, conflict_u, conflict_v)
                llm_trace = call_llm(prompt, max_tokens=700)
                if llm_trace:
                    entry["reasoning"] = llm_trace
                    entry["llm_generated"] = True
                    llm_success += 1
                else:
                    # Sablon fallback B-hez
                    lines = [f"Checking {facts['m']} edges for color conflicts."]
                    if is_valid:
                        lines.append("All edges verified — no conflicts found.")
                        lines.append("Verdict: VALID")
                    else:
                        lines.append(
                            f"Conflict at edge ({conflict_u},{conflict_v}): "
                            f"both nodes have color {test_assignment[conflict_u]}.")
                        lines.append("Verdict: INVALID")
                    entry["reasoning"] = "\n".join(lines)
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
                    f"Adjacency list: {adj_str}\n"
                    f"Partial coloring: {test_assignment}"
                )
                prompt = build_prompt_c(
                    adj_str, assignment, facts, target_node, correct_color, max_colors)
                llm_trace = call_llm(prompt, max_tokens=600)
                if llm_trace:
                    entry["reasoning"] = llm_trace
                    entry["llm_generated"] = True
                    llm_success += 1
                else:
                    neighbors = sorted(G.neighbors(target_node))
                    neighbor_colors = {nb: assignment[nb] for nb in neighbors if nb in assignment}
                    forbidden = sorted(set(neighbor_colors.values()))
                    available = [c for c in range(max_colors) if c not in forbidden]
                    entry["reasoning"] = (
                        f"Node {target_node} neighbors: {neighbors}.\n"
                        f"Neighbor colors: {neighbor_colors}.\n"
                        f"Forbidden: {forbidden}. Available: {available}.\n"
                        f"Assign color {correct_color}."
                    )
                    llm_fallback += 1
                entry["solution"] = f"Node {target_node} → color {correct_color}."

            # ---- D: kromatikus szám ----
            elif task_type == 'task_d_chromatic':
                entry["instruction"] = (
                    f"Determine the chromatic number χ(G) of this graph — "
                    f"the minimum number of colors needed so that no two adjacent "
                    f"nodes share the same color.\n"
                    f"Adjacency list: {adj_str}"
                )
                prompt = build_prompt_d(adj_str, facts, k_min)
                llm_trace = call_llm(prompt, max_tokens=750)
                if llm_trace:
                    entry["reasoning"] = llm_trace
                    entry["llm_generated"] = True
                    llm_success += 1
                else:
                    # Sablon fallback D-hez
                    lines = [
                        f"Lower bound: clique of size {facts['clique_number']} → χ(G)≥{facts['clique_number']}."
                    ]
                    if facts.get('odd_cycle_example'):
                        lines.append(f"Odd cycle {facts['odd_cycle_example']} → χ(G)≥3.")
                    lines.append(f"Upper bound (Brooks): χ(G)≤{facts['max_degree']}.")
                    lines.append(f"Conclusion: χ(G) = {k_min}.")
                    entry["reasoning"] = "\n".join(lines)
                    llm_fallback += 1
                entry["solution"] = str(k_min)

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            tasks_completed += 1

            if tasks_completed % 50 == 0:
                elapsed = time.time() - start_time
                rate = (tasks_completed - already_done) / elapsed
                eta_min = (num_tasks - tasks_completed) / rate / 60 if rate > 0 else 0
                llm_pct = llm_success / max(1, llm_success + llm_fallback) * 100
                print(
                    f"[{tasks_completed}/{num_tasks}] "
                    f"{rate:.1f} feladat/s | ETA {eta_min:.1f} perc | "
                    f"LLM: {llm_success} ({llm_pct:.0f}%) | "
                    f"Fallback: {llm_fallback} | "
                    f"Stratégiák: {strategy_counts}"
                )
                print_provider_status()

    total = get_existing_count(output_file)
    llm_pct = llm_success / max(1, llm_success + llm_fallback) * 100
    print(f"\n--- Kész. Összesen mentve: {total} ---")
    print(f"Elapsed: {time.time() - start_time:.1f}s")
    print(f"LLM trace-ek: {llm_success} ({llm_pct:.0f}%)")
    print(f"Sablon fallback: {llm_fallback}")
    print(f"Stratégia eloszlás: {strategy_counts}")


if __name__ == "__main__":
    # API kulcsok beállítása:
    # export GROQ_API_KEY="gsk_..."
    # export HF_API_KEY="hf_..."
    # export MISTRAL_API_KEY="..."
    run_pipeline(num_tasks=100_000, output_file="llm_dataset.jsonl")