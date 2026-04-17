import networkx as nx
import random
import z3


# ---------------------------------------------------------------------------
# Gráfméret korlátok  —  RAM-barát tartományok task-típusonként
# ---------------------------------------------------------------------------

GRAPH_SIZE_RANGES = {
    "small":  (5, 12),   # task_d chromatic (Z3 exact), task_c missing color
    "medium": (8, 18),   # task_a coloring, task_b validation
    "large":  (12, 25),  # ritka, csak ha explicit kérik
}


class GraphColoringEngine:
    def __init__(self, min_nodes=5, max_nodes=20, graph_size_hint="medium"):
        """
        graph_size_hint: "small" | "medium" | "large"
        Ha min_nodes/max_nodes explicit meg van adva, az felülírja a hint-et.
        """
        lo, hi = GRAPH_SIZE_RANGES.get(graph_size_hint, (min_nodes, max_nodes))
        # Explicit paraméter felülírja a hint-et
        actual_min = min_nodes if min_nodes != 5 else lo
        actual_max = max_nodes if max_nodes != 20 else hi
        self.G, self.g_type = self._generate_diverse_graph(actual_min, actual_max)
        self.num_nodes = self.G.number_of_nodes()

    # ------------------------------------------------------------------
    # Gráf generálás
    # ------------------------------------------------------------------

    def _generate_diverse_graph(self, min_nodes, max_nodes):
        g_type = random.choices(
            ['random', 'bipartite', 'tree', 'dense', 'near_clique'],
            weights=[0.25, 0.15, 0.20, 0.25, 0.15]   # near_clique súlya csökkent
        )[0]
        n = random.randint(min_nodes, max_nodes)

        # Near_clique-nál explicit méretkorlát: n >= 15 esetén Q*N él → RAM-robbanás
        if g_type == 'near_clique' and n > 15:
            n = random.randint(min_nodes, min(15, max_nodes))

        if g_type == 'tree':
            G = nx.random_labeled_tree(n)
        elif g_type == 'bipartite':
            n1 = n // 2
            G = nx.bipartite.random_graph(n1, n - n1, p=random.uniform(0.3, 0.7))
        elif g_type == 'dense':
            # Dense-nél is korlátozzuk a sűrűséget nagy n-nél
            max_p = 0.6 if n > 15 else 0.85
            G = nx.erdos_renyi_graph(n, p=random.uniform(0.45, max_p))
        elif g_type == 'near_clique':
            G = nx.complete_graph(n)
            edges = list(G.edges())
            remove_count = int(len(edges) * random.uniform(0.10, 0.30))
            for e in random.sample(edges, remove_count):
                G.remove_edge(*e)
        else:  # random
            G = nx.erdos_renyi_graph(n, p=random.uniform(0.15, 0.45))

        # Összefüggővé tesszük
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps) - 1):
                G.add_edge(
                    random.choice(list(comps[i])),
                    random.choice(list(comps[i + 1]))
                )

        # Node-ok átszámozása random sorrendben (de 0-tól n-1-ig)
        mapping = dict(zip(G.nodes(), random.sample(range(n), n)))
        G = nx.relabel_nodes(G, mapping)
        return G, g_type

    # ------------------------------------------------------------------
    # Segédfüggvények
    # ------------------------------------------------------------------

    def get_difficulty(self):
        density = nx.density(self.G)
        n = self.G.number_of_nodes()
        if self.g_type == 'tree' or density < 0.15:
            return "easy"
        if self.g_type == 'near_clique' or density >= 0.5 or n >= 20:
            return "hard"
        return "medium"

    def get_adjacency_list_string(self):
        """Teljes adjacency lista — NEM csonkított. Mindig az instruction mezőbe kerül."""
        parts = []
        for node in sorted(self.G.nodes()):
            neighbors = sorted(self.G.neighbors(node))
            parts.append(f"{node}:[{','.join(map(str, neighbors))}]")
        return " ".join(parts)

    def get_adjacency_list_string_truncated(self, max_chars: int = 800):
        """
        Csak a LLM-promptba szánt csonkított változat.
        Sosem kerül az instruction mezőbe.
        """
        full = self.get_adjacency_list_string()
        if len(full) <= max_chars:
            return full
        truncated = full[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars // 2:
            truncated = truncated[:last_space]
        return truncated + " ...[truncated for prompt]"

    def get_graph_facts(self):
        """Strukturális tények — minden reasoning-építőhöz használjuk."""
        G = self.G
        facts = {}
        facts['n'] = G.number_of_nodes()
        facts['m'] = G.number_of_edges()
        facts['density'] = round(nx.density(G), 3)
        facts['max_degree'] = max(dict(G.degree()).values())
        facts['min_degree'] = min(dict(G.degree()).values())
        facts['is_bipartite'] = nx.is_bipartite(G)
        facts['is_tree'] = nx.is_tree(G)

        try:
            cliques = list(nx.find_cliques(G))
            facts['clique_number'] = max(len(c) for c in cliques)
            facts['largest_clique'] = sorted(max(cliques, key=len))
        except Exception:
            facts['clique_number'] = 1
            facts['largest_clique'] = []

        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        facts['top_degree_nodes'] = [node for node, _ in degrees[:5]]
        facts['degree_map'] = dict(G.degree())

        if not facts['is_bipartite']:
            try:
                facts['odd_cycle_example'] = list(nx.find_odd_cycle(G))
            except Exception:
                facts['odd_cycle_example'] = []
        else:
            facts['odd_cycle_example'] = []

        return facts

    # ------------------------------------------------------------------
    # Z3 solver  —  dinamikus timeout + méret-alapú fallback
    # ------------------------------------------------------------------

    def _z3_timeout_ms(self, n_nodes: int, base_ms: int) -> int:
        """
        Kisebb gráfon pontosabb megoldás, nagyobbon gyorsabb fallback.
        Ajánlott base_ms=4000.
        """
        if n_nodes <= 10:
            return base_ms
        elif n_nodes <= 15:
            return max(2000, base_ms // 2)
        elif n_nodes <= 20:
            return max(1000, base_ms // 3)
        else:
            return max(500, base_ms // 6)

    def get_chromatic_number_z3(self, max_search=8, max_mem_mb=4096, timeout_ms=4000):
        """
        Returns (k_min, is_exact).
        is_exact=False → greedy fallback, csak task_a/b/c-nél használható.
        task_d-hez KÖTELEZŐ is_exact=True!
        """
        n = self.G.number_of_nodes()
        effective_timeout = self._z3_timeout_ms(n, timeout_ms)

        # Kis gráfnál teljes keresés, nagyobbnál korlátozzuk
        if n > 20:
            max_search = min(max_search, 4)

        for k in range(1, max_search + 1):
            solver = z3.Solver()
            solver.set("max_memory", max_mem_mb)
            solver.set("timeout", effective_timeout)
            colors = {node: z3.Int(f"c_{node}") for node in self.G.nodes()}
            for node in self.G.nodes():
                solver.add(colors[node] >= 0, colors[node] < k)
            for u, v in self.G.edges():
                solver.add(colors[u] != colors[v])
            result = solver.check()
            if result == z3.sat:
                return k, True
            elif result == z3.unknown:
                break

        # Greedy fallback — is_exact=False jelzi, hogy csak közelítő
        greedy_coloring = nx.coloring.greedy_color(self.G, strategy='largest_first')
        greedy_k = max(greedy_coloring.values()) + 1
        print(f"  [!] Z3 timeout ({n} nodes) — greedy fallback: χ≤{greedy_k}.")
        return greedy_k, False

    def verify_coloring_z3(self, assignment, max_colors):
        """Ellenőrzi, hogy az assignment valóban érvényes-e (Z3-mal)."""
        solver = z3.Solver()
        colors = {node: z3.Int(f"c_{node}") for node in self.G.nodes()}
        for node in self.G.nodes():
            solver.add(colors[node] >= 0, colors[node] < max_colors)
            solver.add(colors[node] == assignment[node])
        for u, v in self.G.edges():
            solver.add(colors[u] != colors[v])
        return solver.check() == z3.sat

    # ------------------------------------------------------------------
    # STRATÉGIA 1 — MRV Backtracking
    # ------------------------------------------------------------------

    def solve_backtracking(self, max_colors, facts):
        G = self.G
        assignment = {}
        trace = []

        trace.append(
            f"Strategy: Backtracking with MRV (Most Restricted Variable) heuristic.\n"
            f"Nodes sorted by descending degree: hardest nodes first.\n"
            f"Top nodes by degree: {facts['top_degree_nodes']}."
        )

        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        backtrack_events = []
        MAX_RICH_BACKTRACKS = 8  # csökkentve a memória kímélése érdekében

        def _find_blocked_node(node, color, current_assignment):
            for nb in G.neighbors(node):
                if nb in current_assignment:
                    continue
                nb_forbidden = {current_assignment[n2]
                                for n2 in G.neighbors(nb)
                                if n2 in current_assignment}
                nb_forbidden.add(color)
                if len(nb_forbidden) >= max_colors:
                    return nb, sorted(nb_forbidden)
            return None

        def backtrack(idx):
            if idx == len(sorted_nodes):
                return True
            node = sorted_nodes[idx]
            forbidden = {assignment[nb]
                         for nb in G.neighbors(node)
                         if nb in assignment}

            for c in range(max_colors):
                if c not in forbidden:
                    assignment[node] = c
                    blocked = None
                    if len(backtrack_events) < MAX_RICH_BACKTRACKS:
                        blocked = _find_blocked_node(node, c, assignment)

                    if backtrack(idx + 1):
                        return True

                    del assignment[node]
                    if len(backtrack_events) < MAX_RICH_BACKTRACKS:
                        if blocked:
                            blocked_node, blocked_forbidden = blocked
                            backtrack_events.append(
                                f"Assigned color {c} to node {node} "
                                f"(degree {G.degree(node)}), but this blocked node "
                                f"{blocked_node} — neighbors already use colors "
                                f"{blocked_forbidden}, no color left from {max_colors}. "
                                f"Backtrack: try different color for node {node}."
                            )
                        else:
                            backtrack_events.append(
                                f"Color {c} on node {node} caused conflict deeper in tree. "
                                f"Backtracking."
                            )
            return False

        success = backtrack(0)

        if backtrack_events:
            trace.append("\nSearch trace (backtrack events):")
            for ev in backtrack_events:
                trace.append(f"  • {ev}")

        if success:
            color_groups = {}
            for node, color in assignment.items():
                color_groups.setdefault(color, []).append(node)
            group_str = "; ".join(
                f"color {c} → {sorted(ns)}"
                for c, ns in sorted(color_groups.items())
            )
            trace.append(
                f"\nResult: valid coloring found using {len(color_groups)} color(s).\n"
                f"Color groups: {group_str}.\n"
                f"Backtracks logged: {len(backtrack_events)}."
            )
        else:
            trace.append(
                f"\nResult: no valid {max_colors}-coloring found."
            )

        return success, assignment, "\n".join(trace)

    # ------------------------------------------------------------------
    # STRATÉGIA 2 — Welsh-Powell (greedy, fokszám szerint rendezve)
    # ------------------------------------------------------------------

    def solve_welsh_powell(self, max_colors, facts):
        G = self.G
        assignment = {}
        trace = []

        trace.append(
            f"Strategy: Welsh-Powell greedy coloring.\n"
            f"Nodes sorted by descending degree, colored in that order.\n"
            f"Each node gets the smallest color unused by its already-colored neighbors.\n"
            f"Sorted order (first 8): "
            + ", ".join(
                f"node {node}(d={G.degree(node)})"
                for node in sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:8]
            ) + ("…" if facts['n'] > 8 else "")
        )

        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        step_log = []

        for node in sorted_nodes:
            neighbor_colors = {assignment[nb]
                               for nb in G.neighbors(node)
                               if nb in assignment}
            color = 0
            while color in neighbor_colors:
                color += 1
            assignment[node] = color

            if len(step_log) < 15:
                step_log.append(
                    f"Node {node} (degree {G.degree(node)}): "
                    f"neighbors use {sorted(neighbor_colors) if neighbor_colors else '∅'}. "
                    f"Assign color {color}."
                )
            elif len(step_log) == 15:
                step_log.append(f"  [remaining {facts['n'] - 15} nodes abbreviated]")
                step_log.append(f"  node {node} → color {color}")
            else:
                step_log.append(f"  node {node} → color {color}")

        trace.append("\nStep-by-step:\n" + "\n".join(step_log))

        color_groups = {}
        for node, color in assignment.items():
            color_groups.setdefault(color, []).append(node)
        group_str = "; ".join(
            f"color {c} → {sorted(ns)}"
            for c, ns in sorted(color_groups.items())
        )
        trace.append(
            f"\nResult: {len(set(assignment.values()))} color(s) used.\n"
            f"Color groups: {group_str}."
        )

        return True, assignment, "\n".join(trace)

    # ------------------------------------------------------------------
    # STRATÉGIA 3 — DSATUR (dinamikus saturáció)
    # ------------------------------------------------------------------

    def solve_dsatur(self, max_colors, facts):
        G = self.G
        assignment = {}
        saturation = {node: 0 for node in G.nodes()}
        trace = []

        trace.append(
            f"Strategy: DSATUR (Dynamic Saturation) coloring.\n"
            f"At each step: pick the uncolored node with the highest saturation\n"
            f"(= number of distinct colors among its colored neighbors).\n"
            f"Ties broken by degree (higher first)."
        )

        uncolored = set(G.nodes())
        step_log = []

        while uncolored:
            node = max(
                uncolored,
                key=lambda n: (saturation[n], G.degree(n))
            )
            neighbor_colors = {assignment[nb]
                               for nb in G.neighbors(node)
                               if nb in assignment}
            color = 0
            while color in neighbor_colors:
                color += 1
            assignment[node] = color
            uncolored.remove(node)

            for nb in G.neighbors(node):
                if nb in uncolored:
                    nb_colors = {assignment[n2]
                                 for n2 in G.neighbors(nb)
                                 if n2 in assignment}
                    saturation[nb] = len(nb_colors)

            if len(step_log) < 15:
                step_log.append(
                    f"Node {node} (sat={saturation[node]}, deg={G.degree(node)}): "
                    f"neighbor colors={sorted(neighbor_colors) if neighbor_colors else '∅'}. "
                    f"Assign color {color}."
                )
            elif len(step_log) == 15:
                step_log.append(f"  [remaining {len(uncolored) + 1} nodes abbreviated]")
                step_log.append(f"  node {node} (sat={saturation[node]}) → color {color}")
            else:
                step_log.append(f"  node {node} (sat={saturation[node]}) → color {color}")

        trace.append("\nDSATUR log:\n" + "\n".join(step_log))

        color_groups = {}
        for node, color in assignment.items():
            color_groups.setdefault(color, []).append(node)
        group_str = "; ".join(
            f"color {c} → {sorted(ns)}"
            for c, ns in sorted(color_groups.items())
        )
        trace.append(
            f"\nResult: {len(set(assignment.values()))} color(s) used.\n"
            f"Color groups: {group_str}.\n"
            f"DSATUR achieves near-optimal results by always coloring the most constrained node."
        )

        return True, assignment, "\n".join(trace)

    # ------------------------------------------------------------------
    # Belépési pont — véletlenszerű stratégia
    # ------------------------------------------------------------------

    def solve_with_trace(self, max_colors):
        facts = self.get_graph_facts()
        strategy = random.choices(
            ['backtracking', 'welsh_powell', 'dsatur'],
            weights=[0.40, 0.30, 0.30]
        )[0]

        if strategy == 'backtracking':
            success, assignment, trace = self.solve_backtracking(max_colors, facts)
        elif strategy == 'welsh_powell':
            success, assignment, trace = self.solve_welsh_powell(max_colors, facts)
        else:
            success, assignment, trace = self.solve_dsatur(max_colors, facts)

        # Greedy módszerek nem garantálják a max_colors betartását
        if success and assignment:
            colors_used = len(set(assignment.values()))
            if colors_used > max_colors:
                # Fallback backtrackingre
                strategy = 'backtracking'
                success, assignment, trace = self.solve_backtracking(max_colors, facts)

        return success, assignment, trace, strategy