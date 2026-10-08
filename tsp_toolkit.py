"""
tsp_toolkit.py - Educational TSP toolkit inspired by the methods in
"The Traveling Salesman Problem: A Computational Study" (Applegate, Bixby,
Chvatal, Cook). This is NOT Concorde; it is a small teaching implementation.

Components
  1. Distance matrix helpers
  2. Tour construction:   nearest neighbour
  3. Local search:        2-opt, Or-opt, iterated local search (double-bridge kicks,
                          the idea behind chained Lin-Kernighan)
  4. Lower bound:         Held-Karp 1-tree bound via subgradient optimisation
  5. Lower bound (LP):    subtour-elimination cutting planes (scipy linprog)
  6. Exact solver:        Bellman-Held-Karp dynamic programming (n <= ~15)
"""
import math
import random
from itertools import combinations

# --------------------------------------------------------------------------
# 1. Helpers
# --------------------------------------------------------------------------
def random_instance(n, seed=0, size=100.0):
    rng = random.Random(seed)
    return [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]


def distance_matrix(pts):
    n = len(pts)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = math.dist(pts[i], pts[j])
    return d


def tour_length(tour, d):
    return sum(d[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


# --------------------------------------------------------------------------
# 2. Construction heuristic
# --------------------------------------------------------------------------
def nearest_neighbour(d, start=0):
    n = len(d)
    unvisited = set(range(n)) - {start}
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: d[last][j])
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


# --------------------------------------------------------------------------
# 3. Local search
# --------------------------------------------------------------------------
def two_opt(tour, d):
    """Repeatedly reverse segments while that shortens the tour."""
    n = len(tour)
    tour = tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, e = tour[j], tour[(j + 1) % n]
                delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    return tour


def or_opt(tour, d, max_seg=3):
    """Move segments of 1..max_seg cities to a better position."""
    n = len(tour)
    tour = tour[:]
    improved = True
    while improved:
        improved = False
        for seg_len in range(1, max_seg + 1):
            for i in range(n):
                if seg_len >= n - 2:
                    continue
                idx = [(i + k) % n for k in range(seg_len)]
                seg = [tour[k] for k in idx]
                prev = tour[(i - 1) % n]
                nxt = tour[(i + seg_len) % n]
                remove_gain = d[prev][seg[0]] + d[seg[-1]][nxt] - d[prev][nxt]
                rest = [tour[(i + seg_len + k) % n] for k in range(n - seg_len)]
                best_delta, best_pos, best_rev = -1e-10, None, False
                for p in range(len(rest) - 1):
                    u, v = rest[p], rest[p + 1]
                    base = d[u][v]
                    fwd = d[u][seg[0]] + d[seg[-1]][v] - base
                    rev = d[u][seg[-1]] + d[seg[0]][v] - base
                    if fwd - remove_gain < best_delta:
                        best_delta, best_pos, best_rev = fwd - remove_gain, p, False
                    if rev - remove_gain < best_delta:
                        best_delta, best_pos, best_rev = rev - remove_gain, p, True
                if best_pos is not None:
                    s = seg[::-1] if best_rev else seg
                    tour = rest[:best_pos + 1] + s + rest[best_pos + 1:]
                    improved = True
                    break
            if improved:
                break
    return tour


def local_search(tour, d):
    while True:
        before = tour_length(tour, d)
        tour = two_opt(tour, d)
        tour = or_opt(tour, d)
        if tour_length(tour, d) >= before - 1e-10:
            return tour


def double_bridge(tour, rng):
    n = len(tour)
    a, b, c = sorted(rng.sample(range(1, n), 3))
    return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]


def iterated_local_search(d, iterations=200, seed=0, start_tour=None):
    """Chained-local-search scheme: kick (double bridge), re-optimise, keep if better."""
    rng = random.Random(seed)
    n = len(d)
    best = local_search(start_tour or nearest_neighbour(d), d)
    best_len = tour_length(best, d)
    if n < 8:
        return best, best_len
    for _ in range(iterations):
        cand = local_search(double_bridge(best, rng), d)
        cand_len = tour_length(cand, d)
        if cand_len < best_len - 1e-10:
            best, best_len = cand, cand_len
    return best, best_len


# --------------------------------------------------------------------------
# 4. Held-Karp 1-tree lower bound (Lagrangian / subgradient)
# --------------------------------------------------------------------------
def _one_tree(w, root=0):
    """Min spanning tree on nodes != root (Prim) + two cheapest edges at root."""
    n = len(w)
    nodes = [v for v in range(n) if v != root]
    in_tree = {nodes[0]}
    key = {v: w[nodes[0]][v] for v in nodes[1:]}
    parent = {v: nodes[0] for v in nodes[1:]}
    edges, total = [], 0.0
    while key:
        v = min(key, key=key.get)
        total += key.pop(v)
        edges.append((parent[v], v))
        in_tree.add(v)
        for u in key:
            if w[v][u] < key[u]:
                key[u], parent[u] = w[v][u], v
    r = sorted(nodes, key=lambda v: w[root][v])[:2]
    for v in r:
        total += w[root][v]
        edges.append((root, v))
    return total, edges


def held_karp_bound(d, iters=300, upper_bound=None):
    """Return a lower bound on the optimal tour length (Held-Karp 1-tree bound)."""
    n = len(d)
    pi = [0.0] * n
    best = -float("inf")
    if upper_bound is None:
        upper_bound = tour_length(nearest_neighbour(d), d)
    lam, stall = 2.0, 0
    for _ in range(iters):
        w = [[d[i][j] + pi[i] + pi[j] for j in range(n)] for i in range(n)]
        cost, edges = _one_tree(w)
        lb = cost - 2 * sum(pi)
        if lb > best + 1e-9:
            best, stall = lb, 0
        else:
            stall += 1
            if stall >= 10:
                lam, stall = lam / 2, 0
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        g = [x - 2 for x in deg]
        norm = sum(x * x for x in g)
        if norm == 0 or lam < 1e-6:
            break
        step = lam * (upper_bound - lb) / norm
        pi = [pi[i] + step * g[i] for i in range(n)]
    return best


# --------------------------------------------------------------------------
# 5. Cutting-plane LP bound with subtour-elimination constraints
# --------------------------------------------------------------------------
def lp_subtour_bound(d, max_rounds=100):
    """
    Solve  min sum d_e x_e  s.t. degree(v)=2, 0<=x<=1, plus subtour cuts
    x(delta(S)) >= 2 added lazily. Separation here uses connected components of the
    support graph (fast, but weaker than exact min-cut separation used in Concorde).
    """
    import numpy as np
    from scipy.optimize import linprog
    n = len(d)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx = {e: k for k, e in enumerate(edges)}
    c = np.array([d[i][j] for i, j in edges])
    A_eq = np.zeros((n, len(edges)))
    for (i, j), k in idx.items():
        A_eq[i, k] = A_eq[j, k] = 1
    b_eq = np.full(n, 2.0)
    A_ub, b_ub = [], []
    val = None
    for _ in range(max_rounds):
        res = linprog(c, A_ub=np.array(A_ub) if A_ub else None,
                      b_ub=np.array(b_ub) if b_ub else None,
                      A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
        val, x = res.fun, res.x
        # components of support graph
        adj = {v: [] for v in range(n)}
        for (i, j), k in idx.items():
            if x[k] > 1e-6:
                adj[i].append(j)
                adj[j].append(i)
        seen, comps = set(), []
        for s in range(n):
            if s in seen:
                continue
            comp, stack = [], [s]
            seen.add(s)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
        if len(comps) == 1:
            break                      # no violated subtour cut found
        for comp in comps:
            S = set(comp)
            row = np.zeros(len(edges))
            for (i, j), k in idx.items():
                if (i in S) != (j in S):
                    row[k] = -1.0      # -x(delta(S)) <= -2
            A_ub.append(row)
            b_ub.append(-2.0)
    return val


# --------------------------------------------------------------------------
# 6. Exact solver: Bellman-Held-Karp DP  (O(n^2 2^n); keep n <= 15)
# --------------------------------------------------------------------------
def held_karp_exact(d):
    n = len(d)
    if n > 16:
        raise ValueError("DP is exponential; use n <= 16")
    INF = float("inf")
    dp = {(1 << 0, 0): (0.0, None)}
    for size in range(1, n):
        for subset in combinations(range(1, n), size):
            mask = 1
            for s in subset:
                mask |= 1 << s
            for j in subset:
                prev_mask = mask ^ (1 << j)
                best = (INF, None)
                for k in ([0] if prev_mask == 1 else
                          [x for x in subset if x != j]):
                    cost = dp[(prev_mask, k)][0] + d[k][j]
                    if cost < best[0]:
                        best = (cost, k)
                dp[(mask, j)] = best
    full = (1 << n) - 1
    cost, last = min((dp[(full, j)][0] + d[j][0], j) for j in range(1, n))
    tour, mask = [], full
    while last is not None:
        tour.append(last)
        mask, last = mask ^ (1 << last), dp[(mask, last)][1]
    tour.append(0)
    return tour[::-1], cost


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Small instance: compare everything against the exact optimum ---
    pts = random_instance(12, seed=1)
    d = distance_matrix(pts)
    nn = nearest_neighbour(d)
    ils, ils_len = iterated_local_search(d, iterations=100)
    opt_tour, opt_len = held_karp_exact(d)
    print("n = 12")
    print(f"  Nearest neighbour : {tour_length(nn, d):8.2f}")
    print(f"  Iterated local srch: {ils_len:8.2f}")
    print(f"  Exact (DP)        : {opt_len:8.2f}")
    print(f"  1-tree lower bound: {held_karp_bound(d, upper_bound=opt_len):8.2f}")
    print(f"  LP subtour bound  : {lp_subtour_bound(d):8.2f}")

    # --- Larger instance: heuristic tour + provable lower bound -> optimality gap ---
    n = 60
    pts = random_instance(n, seed=7)
    d = distance_matrix(pts)
    tour, length = iterated_local_search(d, iterations=150, seed=3)
    lb = held_karp_bound(d, upper_bound=length)
    print(f"\nn = {n}")
    print(f"  Best tour found   : {length:8.2f}")
    print(f"  Lower bound (1-tr): {lb:8.2f}")
    print(f"  Guaranteed gap    : {100 * (length - lb) / lb:5.2f}%")
