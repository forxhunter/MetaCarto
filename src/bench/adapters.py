"""Run competing layout algorithms on the same graph MetaCarto draws.

Bioinformatics requires that "new methods MUST be compared to existing
state-of-the-art methods, using real biological data". The two closest tools
are MetDraw (Bioinformatics 2014), which lays out with **Graphviz**, and
MetExploreViz (Bioinformatics 2018), which uses a **D3 force-directed**
layout. Neither published a quantitative layout-quality evaluation, so there
is no number to cite -- they have to be re-run.

Fairness, and where it is deliberately tilted against us
--------------------------------------------------------
Every method here is given:

  * the *same* compound graph, built by `compound.build_compound_graph` and
    oriented by `direction.orient_compound_graph`;
  * the *same* renderer, `render.build_escher_map`, with no route hints.

so the only thing that varies is where the nodes go. Two consequences worth
stating plainly in the paper:

1. **The baselines are handed primary-compound reduction, which is our
   contribution.** Run on the raw bipartite graph -- what MetDraw and
   MetExploreViz actually receive -- they would do considerably worse. Ablation
   1 measures that separately. Giving it to them here is the conservative
   choice.

2. **MetaCarto is measured without its own edge routing.** Passing no `routes`
   means its multi-layer edges are not followed along their dummy chains, so
   the maps scored here are worse than the ones it ships. That also understates
   our result, which is the direction an evaluation should err in.

Scale normalisation
-------------------
Graphviz reports points, `spring_layout` reports roughly [-1, 1], and MetaCarto
works in Escher units. Most metrics are scale-invariant, but `axis_aligned`
uses an absolute 1.0-unit tolerance and `min_separation_ratio` divides by a
pitch, so every layout is scaled to put its *median nearest-neighbour distance*
at `TARGET_PITCH`. That makes a node in one drawing the same size as a node in
another, which is the only way those two metrics mean anything across methods.
"""

import math
import os
import shutil
import subprocess
import tempfile

TARGET_PITCH = 180.0


def available():
    """Which baselines can run here, and why the others cannot."""
    status = {}
    try:
        import networkx  # noqa: F401
        status["spring"] = True
        status["kamada_kawai"] = True
    except ImportError:                                   # pragma: no cover
        status["spring"] = status["kamada_kawai"] = False
    has_binary = shutil.which("dot") is not None
    try:
        import pygraphviz  # noqa: F401
        has_py = True
    except ImportError:
        has_py = False
    for engine in ("dot", "neato", "fdp"):
        status[engine] = has_binary or has_py
    return status


def _normalise(pos):
    """Scale a layout so its median nearest-neighbour distance is TARGET_PITCH."""
    keys = list(pos)
    if len(keys) < 2:
        return {k: (0.0, 0.0) for k in keys}
    points = [pos[k] for k in keys]

    # Nearest neighbour by a uniform grid, so this stays linear on big maps.
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    cell = span / max(1.0, math.sqrt(len(points)))
    buckets = {}
    for i, (x, y) in enumerate(points):
        buckets.setdefault((int(x // cell), int(y // cell)), []).append(i)

    nearest = []
    for (cx, cy), members in buckets.items():
        near = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for j in buckets.get((cx + dx, cy + dy), ())]
        for i in members:
            best = math.inf
            for j in near:
                if i == j:
                    continue
                d = math.hypot(points[i][0] - points[j][0],
                               points[i][1] - points[j][1])
                if 0.0 < d < best:
                    best = d
            if best < math.inf:
                nearest.append(best)
    if not nearest:
        return {k: (pos[k][0], pos[k][1]) for k in keys}
    nearest.sort()
    median = nearest[len(nearest) // 2] or 1.0
    factor = TARGET_PITCH / median
    return {k: (pos[k][0] * factor, pos[k][1] * factor) for k in keys}


def _graphviz(graph, engine):
    """Node positions from a Graphviz engine. This is MetDraw's mechanism."""
    try:
        import pygraphviz as pgv
        agraph = pgv.AGraph(directed=True)
        for node in graph.nodes:
            agraph.add_node(str(node))
        for u, v in graph.edges:
            agraph.add_edge(str(u), str(v))
        agraph.layout(prog=engine)
        out = {}
        for node in graph.nodes:
            raw = agraph.get_node(str(node)).attr["pos"]
            if not raw:
                continue
            x, y = raw.split(",")[:2]
            out[node] = (float(x), float(y))
        return out
    except ImportError:
        pass

    if shutil.which(engine) is None and shutil.which("dot") is None:
        raise RuntimeError("graphviz not installed: no %s binary, no pygraphviz"
                           % engine)

    index = {node: "n%d" % i for i, node in enumerate(graph.nodes)}
    lines = ["digraph G {", "  node [shape=circle,width=0.3,label=\"\"];"]
    for node in graph.nodes:
        lines.append("  %s;" % index[node])
    for u, v in graph.edges:
        lines.append("  %s -> %s;" % (index[u], index[v]))
    lines.append("}")
    source = "\n".join(lines)

    handle, path = tempfile.mkstemp(suffix=".gv")
    os.close(handle)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        raw = subprocess.check_output([engine, "-Tplain", path],
                                      stderr=subprocess.DEVNULL).decode()
    finally:
        os.unlink(path)

    back = {v: k for k, v in index.items()}
    out = {}
    for line in raw.splitlines():
        parts = line.split()
        # plain format: "node <name> <x> <y> <w> <h> <label> ..."
        if len(parts) >= 4 and parts[0] == "node" and parts[1] in back:
            out[back[parts[1]]] = (float(parts[2]), float(parts[3]))
    return out


def place(graph, method, seed=0):
    """Node positions for `graph` under one layout method.

    Returns a dict node -> (x, y), normalised to TARGET_PITCH.
    """
    if method in ("dot", "neato", "fdp"):
        return _normalise(_graphviz(graph, method))

    import networkx as nx
    undirected = nx.Graph(graph)
    if method == "spring":
        # What MetExploreViz does: a D3 force simulation. seed fixed so the
        # baseline is reproducible, which the real tool is not.
        raw = nx.spring_layout(undirected, seed=seed)
    elif method == "kamada_kawai":
        raw = nx.kamada_kawai_layout(undirected)
    else:
        raise ValueError("unknown method %r" % method)
    return _normalise({k: (float(v[0]), float(v[1])) for k, v in raw.items()})
