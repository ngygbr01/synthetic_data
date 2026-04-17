import networkx as nx
import random
import z3


class GraphColoringEngine:
    def __init__(self, min_nodes=10, max_nodes=65):
        self.G, self.g_type = self._generate_diverse_graph(min_nodes, max_nodes)
        self.num_nodes = self.G.number_of_nodes()

    # ------------------------------------------------------------------
    # Graph generation
    # ------------------------------------------------------------------

    def _generate_diverse_graph(self, min_nodes, max_nodes):
        g_type = random.choices(
            ['random', 'bipartite', 'tree', 'dense', 'near_clique'],
            weights=[0.25, 0.15, 0.15, 0.25, 0.20]
        )[0]
        n = random.randint(min_nodes, max_nodes)

        if g_type == 'tree':
            G = nx.random_labeled_tree(n)
        elif g_type == 'bipartite':
            n1 = n // 2
            G = nx.bipartite.random_graph(n1, n - n1, p=random.uniform(0.3, 0.7))
        elif g_type == 'dense':
            G = nx.erdos_renyi_graph(n, p=random.uniform(0.55, 0.85))
        elif g_type == 'near_clique':
            G = nx.complete_graph(n)
            edges = list(G.edges())
            remove_count = int(len(edges) * random.uniform(0.05, 0.25))
            for e in random.sample(edges, remove_count):
                G.remove_edge(*e)
        else:
            G = nx.erdos_renyi_graph(n, p=random.uniform(0.15, 0.45))

        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps) - 1):
                G.add_edge(
                    random.choice(list(comps[i])),
                    random.choice(list(comps[i + 1]))
                )

        mapping = dict(zip(G.nodes(), random.sample(list(range(n)), n)))
        G = nx.relabel_nodes(G, mapping)
        return G, g_type

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_difficulty(self):
        density = nx.density(self.G)
        if self.g_type == 'tree' or density < 0.15:
            return "easy"
        if self.g_type == 'near_clique' or density >= 0.5:
            return "hard"
        return "medium"

    def get_adjacency_list_string(self):
        parts = []
        for n in sorted(self.G.nodes()):
            neighbors = sorted(self.G.neighbors(n))
            parts.append(f"{n}:[{','.join(map(str, neighbors))}]")
        return " ".join(parts)

    def get_graph_facts(self):
        """Structural facts used across all reasoning builders."""
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
        facts['top_degree_nodes'] = [n for n, _ in degrees[:5]]
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
    # Z3 solver
    # ------------------------------------------------------------------

    def get_chromatic_number_z3(self, max_search=8, max_mem_mb=12000, timeout_ms=6000):
        for k in range(1, max_search + 1):
            solver = z3.Solver()
            solver.set("max_memory", max_mem_mb)
            solver.set("timeout", timeout_ms)
            colors = {n: z3.Int(f"c_{n}") for n in self.G.nodes()}
            for n in self.G.nodes():
                solver.add(colors[n] >= 0, colors[n] < k)
            for u, v in self.G.edges():
                solver.add(colors[u] != colors[v])
            result = solver.check()
            if result == z3.sat:
                return k, True
            elif result == z3.unknown:
                break

        greedy_coloring = nx.coloring.greedy_color(self.G, strategy='largest_first')
        greedy_k = max(greedy_coloring.values()) + 1
        print(f" [!] Z3 timeout — greedy fallback: {greedy_k} colors.")
        return greedy_k, False

    def verify_coloring_z3(self, assignment, max_colors):
        solver = z3.Solver()
        colors = {n: z3.Int(f"c_{n}") for n in self.G.nodes()}
        for n in self.G.nodes():
            solver.add(colors[n] >= 0, colors[n] < max_colors)
            solver.add(colors[n] == assignment[n])
        for u, v in self.G.edges():
            solver.add(colors[u] != colors[v])
        return solver.check() == z3.sat

    # ------------------------------------------------------------------
    # STRATEGY 1 — Backtracking with deep conflict explanation
    # ------------------------------------------------------------------

    def solve_backtracking(self, max_colors, facts):
        """
        MRV backtracking. Logs WHY a backtrack happens:
        which downstream node gets blocked and why.
        """
        G = self.G
        assignment = {}
        trace = []

        trace.append(
            f"Strategy: Backtracking with MRV (Most Restricted Variable) heuristic.\n"
            f"Nodes sorted by descending degree so the hardest-to-color nodes are "
            f"placed first, reducing wasted work.\n"
            f"Top 5 nodes by degree: {facts['top_degree_nodes']}."
        )

        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        backtrack_events = []   # collect rich backtrack explanations
        MAX_RICH_BACKTRACKS = 12

        def _find_blocked_node(node, color, current_assignment):
            """
            After tentatively placing `color` on `node`, look ahead one step:
            find a neighbor of `node`'s downstream neighbors that would be
            completely blocked. Returns (blocked_node, its_forbidden_set) or None.
            """
            # Check immediate neighbors of `node` first
            for nb in G.neighbors(node):
                if nb in current_assignment:
                    continue
                nb_forbidden = {current_assignment[n2]
                                for n2 in G.neighbors(nb)
                                if n2 in current_assignment}
                nb_forbidden.add(color)   # tentative color of `node`
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

                    # --- Look-ahead: explain WHY this color is risky ---
                    if len(backtrack_events) < MAX_RICH_BACKTRACKS:
                        blocked = _find_blocked_node(node, c, assignment)
                    else:
                        blocked = None

                    if backtrack(idx + 1):
                        return True

                    # Backtrack — generate rich explanation
                    del assignment[node]
                    if len(backtrack_events) < MAX_RICH_BACKTRACKS:
                        if blocked:
                            blocked_node, blocked_forbidden = blocked
                            backtrack_events.append(
                                f"Assigned color {c} to node {node} "
                                f"(degree {G.degree(node)}), but this forced node "
                                f"{blocked_node} (degree {G.degree(blocked_node)}) "
                                f"into a dead end — its neighbors already use colors "
                                f"{blocked_forbidden}, leaving no valid color out of "
                                f"{max_colors} available. "
                                f"Backtracking: try a different color for node {node}."
                            )
                        else:
                            backtrack_events.append(
                                f"Assigned color {c} to node {node} "
                                f"(degree {G.degree(node)}), but a conflict emerged "
                                f"deeper in the search tree. "
                                f"Backtracking to node {node} and trying next color."
                            )
            return False

        success = backtrack(0)

        # Append backtrack events after the preamble
        if backtrack_events:
            trace.append("\nSearch trace (key backtrack events):")
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
            total_bt = len(backtrack_events)
            trace.append(
                f"\nResult: valid coloring found using {len(color_groups)} color(s).\n"
                f"Color groups: {group_str}.\n"
                f"Total backtracks logged: {total_bt}."
            )
        else:
            trace.append(
                f"\nResult: no valid {max_colors}-coloring exists for this graph."
            )

        return success, assignment, "\n".join(trace)

    # ------------------------------------------------------------------
    # STRATEGY 2 — Welsh-Powell (greedy, degree-ordered)
    # ------------------------------------------------------------------

    def solve_welsh_powell(self, max_colors, facts):
        """
        Welsh-Powell: sort nodes by degree descending, assign smallest
        available color greedily. No backtracking. Explains each choice.
        """
        G = self.G
        assignment = {}
        trace = []

        trace.append(
            f"Strategy: Welsh-Powell greedy coloring.\n"
            f"Nodes are sorted once by descending degree and colored in that order. "
            f"Each node receives the smallest color not used by any of its "
            f"already-colored neighbors. No backtracking — greedy and fast, "
            f"but may use more colors than optimal.\n"
            f"Sorted order (degree): "
            + ", ".join(
                f"node {n}(d={G.degree(n)})"
                for n in sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:8]
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

            # Log first 20 steps in detail; then all remaining nodes compactly
            if len(step_log) < 20:
                step_log.append(
                    f"Node {node} (degree {G.degree(node)}): "
                    f"neighbors use colors {sorted(neighbor_colors) if neighbor_colors else '∅'}. "
                    f"Assign color {color}."
                )
            elif len(step_log) == 20:
                step_log.append(f"[Remaining {facts['n'] - 20} nodes — compact log:]")
                step_log.append(f"  node {node} → color {color}")
            else:
                step_log.append(f"  node {node} → color {color}")

        trace.append("\nStep-by-step (first assignments):\n" + "\n".join(step_log))

        colors_used = set(assignment.values())
        color_groups = {}
        for node, color in assignment.items():
            color_groups.setdefault(color, []).append(node)
        group_str = "; ".join(
            f"color {c} → {sorted(ns)}"
            for c, ns in sorted(color_groups.items())
        )
        trace.append(
            f"\nResult: greedy coloring complete using {len(colors_used)} color(s).\n"
            f"Color groups: {group_str}."
        )

        return True, assignment, "\n".join(trace)

    # ------------------------------------------------------------------
    # STRATEGY 3 — DSATUR (dynamic saturation)
    # ------------------------------------------------------------------

    def solve_dsatur(self, max_colors, facts):
        """
        DSATUR: at each step pick the uncolored node with the highest
        saturation (number of distinct colors among its colored neighbors).
        Ties broken by degree. Explains saturation at each key step.
        """
        G = self.G
        assignment = {}
        saturation = {n: 0 for n in G.nodes()}
        trace = []

        trace.append(
            f"Strategy: DSATUR (Dynamic Saturation) coloring.\n"
            f"At each step, choose the uncolored node with the highest "
            f"saturation — the number of DISTINCT colors already used by "
            f"its neighbors. Highly saturated nodes are the most constrained "
            f"and must be colored before they run out of options.\n"
            f"Ties in saturation are broken by degree (higher degree first)."
        )

        uncolored = set(G.nodes())
        step_log = []

        while uncolored:
            # Pick most saturated node (tie-break: degree)
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

            # Update saturation of uncolored neighbors
            for nb in G.neighbors(node):
                if nb in uncolored:
                    nb_colors = {assignment[n2]
                                 for n2 in G.neighbors(nb)
                                 if n2 in assignment}
                    saturation[nb] = len(nb_colors)

            if len(step_log) < 20:
                step_log.append(
                    f"Node {node} selected (saturation={saturation[node]}, "
                    f"degree={G.degree(node)}): "
                    f"neighbor colors={sorted(neighbor_colors) if neighbor_colors else '∅'}. "
                    f"Assign color {color}."
                )
            elif len(step_log) == 20:
                step_log.append(f"[Remaining {len(uncolored) + 1} nodes — compact log:]")
                step_log.append(f"  node {node} (sat={saturation[node]}) → color {color}")
            else:
                step_log.append(f"  node {node} (sat={saturation[node]}) → color {color}")

        trace.append("\nDSATUR step log:\n" + "\n".join(step_log))

        colors_used = set(assignment.values())
        color_groups = {}
        for node, color in assignment.items():
            color_groups.setdefault(color, []).append(node)
        group_str = "; ".join(
            f"color {c} → {sorted(ns)}"
            for c, ns in sorted(color_groups.items())
        )
        trace.append(
            f"\nResult: DSATUR coloring complete using {len(colors_used)} color(s).\n"
            f"Color groups: {group_str}.\n"
            f"DSATUR typically achieves near-optimal results because it always "
            f"colors the most constrained node next."
        )

        return True, assignment, "\n".join(trace)

    # ------------------------------------------------------------------
    # Public entry point — randomly picks a strategy
    # ------------------------------------------------------------------

    def solve_with_trace(self, max_colors):
        """
        Randomly selects one of three strategies.
        Returns (success, assignment, trace_text, strategy_name).
        Weights: backtracking 40%, welsh_powell 30%, dsatur 30%.
        """
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

        # Welsh-Powell and DSATUR always succeed (greedy), but may exceed max_colors.
        # If they used more colors than allowed, fall back to backtracking.
        if success and assignment:
            colors_used = len(set(assignment.values()))
            if colors_used > max_colors:
                strategy = 'backtracking'
                success, assignment, trace = self.solve_backtracking(max_colors, facts)

        return success, assignment, trace, strategy