# Tiny debug helpers (pretty printers + toggle)
DEBUG = True
def hr(title=None):
    print("\n" + ("─"*60 if title is None else f"── {title} " + "─"*60))

def show_set(name, S):
    print(f"{name}: {{{', '.join(map(str, sorted(S)))}}}")

def show_list(name, L):
    print(f"{name}: [{', '.join(map(str, L))}]")

def show_dict(name, D):
    items = ', '.join([f"{k}:{v}" for k,v in sorted(D.items())])
    print(f"{name}: {{{items}}}")

# We'll keep both an adjacency list (dict of sets) and an adjacency matrix (list of lists).

def build_adj_undirected(nodes, edges):
    idx = {n:i for i, n in enumerate(nodes)}
    N = len(nodes)

    # adjacency list as dict: node -> set(neighbors)
    adj = {n:set() for n in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # adjacency matrix A[i][j] in {0,1}
    A = [[0]*N for _ in range(N)]
    for u, v in edges:
        iu, iv = idx[u], idx[v]
        A[iu][iv] = 1
        A[iv][iu] = 1  # undirected

    return adj, A, idx

# Tip: on large graphs this helps you "see" sparsity and structure.
import matplotlib.pyplot as plt
import numpy as np

def plot_adjacency_matrix(A, title="Adjacency matrix"):
    M = np.array(A, dtype=int)
    plt.figure(figsize=(5,5))
    plt.imshow(M, interpolation="nearest")
    plt.title(title)
    plt.xlabel("j (column)")
    plt.ylabel("i (row)")
    plt.colorbar(label="A[i,j]")
    plt.show()

# Degree and distributions

def degree_of(adj, node):
    return len(adj[node])

def degree_sequence(adj, nodes):
    return [len(adj[n]) for n in nodes]

def degree_distribution(adj, nodes):
    from collections import Counter
    seq = degree_sequence(adj, nodes)
    counts = Counter(seq)  # k -> count
    N = len(nodes)
    # probability mass function: P(k) = N_k / N
    return {k: c / N for k, c in sorted(counts.items())}

def plot_degree_distribution(adj, nodes):
    # compute counts
    from collections import Counter
    degs = [len(adj[n]) for n in nodes]
    C = Counter(degs)
    ks = sorted(C.keys())
    Ns = [C[k] for k in ks]
    N = len(nodes)
    Ps = [c/N for c in Ns]

    hr("Degree distribution (numeric)")
    for k, p in zip(ks, Ps):
        print(f"k={k:2d}  P(k)={p:.4f}  count={int(p*N)}")

    # Linear “PMF” bar plot
    plt.figure(figsize=(5,3))
    plt.bar([str(k) for k in ks], Ps)
    plt.title("Degree distribution (linear)")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.tight_layout()
    plt.show()

# Density (undirected, no self-loops) = 2L / (N*(N-1))
# Sparsity = 1 - Density

def density_undirected(N, L):
    if N <= 1:
        return 0.0
    return (2.0 * L) / (N * (N - 1))

def sparsity_undirected(N, L):
    return 1.0 - density_undirected(N, L)

# BFS for shortest paths on unweighted graphs
from collections import deque

def bfs_trace(adj, source, target=None):
    """
    Standard BFS that records a 'snapshot' at each step so we can replay it.
    If target is given, stops when reached. Returns (distance_map, parent, snapshots).
    """
    visited = {source}
    parent = {source: None}
    dist = {source: 0}
    q = deque([source])

    # each snapshot is a dict we can pretty-print later
    snapshots = [{
        "event": "init",
        "queue": list(q),
        "visited": set(visited),
        "parent": dict(parent),
        "dist": dict(dist),
        "frontier_node": None,
    }]

    while q:
        u = q.popleft()
        # expand u
        snapshots.append({
            "event": "pop",
            "queue": list(q),
            "visited": set(visited),
            "parent": dict(parent),
            "dist": dict(dist),
            "frontier_node": u,
        })
        for v in sorted(adj[u]):  # sort for determinism
            if v not in visited:
                visited.add(v)
                parent[v] = u
                dist[v] = dist[u] + 1
                q.append(v)
                snapshots.append({
                    "event": f"discover {v} via {u}",
                    "queue": list(q),
                    "visited": set(visited),
                    "parent": dict(parent),
                    "dist": dict(dist),
                    "frontier_node": u,
                })
                if target is not None and v == target:
                    snapshots.append({
                        "event": "target reached",
                        "queue": list(q),
                        "visited": set(visited),
                        "parent": dict(parent),
                        "dist": dict(dist),
                        "frontier_node": u,
                    })
                    return dist, parent, snapshots
    return dist, parent, snapshots

def pretty_play_bfs(snapshots):
    for i, s in enumerate(snapshots):
        hr(f"Step {i}: {s['event']}")
        show_list("queue", s["queue"])
        show_set("visited", s["visited"])
        show_dict("parent", s["parent"])
        show_dict("dist", s["dist"])

# reconstruct the path 0 -> 33 from parent map
def reconstruct_path(parent, s, t):
    if t not in parent: 
        return None
    path = [t]
    while path[-1] is not None:
        path.append(parent[path[-1]])
    path.pop()            # drop the trailing None
    path.reverse()
    return path if path[0]==s else None

# Diameter via BFS from all nodes
def diameter_with_progress(adj, nodes):
    diam = 0
    pair = None
    for i, s in enumerate(sorted(nodes)):
        # BFS from s to get distances
        from collections import deque
        visited = {s}
        dist = {s:0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    dist[v] = dist[u] + 1
                    q.append(v)
        # current eccentricity
        ecc = max(dist.values())
        if ecc > diam:
            diam = ecc
            # choose any farthest t
            t = max(dist, key=lambda x: dist[x])
            pair = (s, t)
        if DEBUG and i % 5 == 0:
            print(f"progress: source={s:2d}  ecc={ecc:2d}  current_diam={diam:2d}  far_pair={pair}")
    return diam, pair

# Connected components with tracing

def connected_components_trace(adj, nodes):
    seen = set()
    comps = []
    steps = []
    for s in sorted(nodes):
        if s in seen:
            continue
        comp = []
        stack = [s]
        seen.add(s)
        steps.append(("start_component", s, list(stack), set(seen)))
        while stack:
            u = stack.pop()
            comp.append(u)
            steps.append(("visit", u, list(stack), set(seen)))
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
                    steps.append(("discover", (u, v), list(stack), set(seen)))
        comps.append(sorted(comp))
        steps.append(("finish_component", tuple(comp), list(stack), set(seen)))
    return comps, steps

def play_components_steps(steps):
    for i, (evt, payload, stack, seen) in enumerate(steps):
        hr(f"CC Step {i}: {evt}")
        print("payload:", payload)
        show_list("stack", stack)
        show_set("seen", seen)

# Clustering coefficient of a node

def explain_local_clustering(adj, node):
    neigh = sorted(adj[node])
    k = len(neigh)
    hr(f"Node {node} — neighbors and pairs")
    show_list("neighbors", neigh)
    if k < 2:
        print("Degree < 2 ⇒ clustering = 0.0")
        return 0.0
    # enumerate neighbor pairs and test whether they are connected
    links = 0
    pairs = 0
    for i in range(len(neigh)):
        for j in range(i+1, len(neigh)):
            u, v = neigh[i], neigh[j]
            connected = (v in adj[u])
            pairs += 1
            links += 1 if connected else 0
            print(f"pair ({u},{v})  connected={connected}")
    Ci = (2.0 * links) / (k*(k-1))
    hr(f"Summary")
    print(f"pairs among neighbors = {pairs}")
    print(f"edges among neighbors = {links}")
    print(f"C_{node} = (2*{links}) / ({k}*({k}-1)) = {Ci:.6f}")
    return Ci