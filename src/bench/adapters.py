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
    try:
        import pygraphviz  # noqa: F401
        has_py = True
    except ImportError:
        has_py = False
    # Checked per engine. Keying all three off `which("dot")` reported neato as
    # available on a machine that has only dot, and compare.py then produced a
    # FileNotFoundError traceback per cluster instead of skipping it.
    for engine in ("dot", "neato", "fdp"):
        status[engine] = has_py or shutil.which(engine) is not None
    return status


def normalise(pos):
    """Scale a layout so its median nearest-neighbour distance is TARGET_PITCH."""
    keys = list(pos)
    if len(keys) < 2:
        return {k: (float(pos[k][0]), float(pos[k][1])) for k in keys}
    points = [pos[k] for k in keys]

    # Exact nearest neighbour. The previous grid search only looked in the
    # 3x3 cell window, so a point whose nearest neighbour sat two cells away
    # got a too-large distance, and a point with an empty window was dropped
    # from the median altogether -- which biased precisely the isolated points
    # that a hairball-shaped layout has most of.
    nearest = []
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        nearest = [float(d[1]) for d in distances if d[1] > 0.0]
    except ImportError:                                   # pragma: no cover
        for i, a in enumerate(points):
            best = math.inf
            for j, b in enumerate(points):
                if i == j:
                    continue
                d = math.hypot(a[0] - b[0], a[1] - b[1])
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
        # The same node footprint the subprocess branch writes. Without this
        # the branches lay out different graphs: bare pygraphviz nodes inherit
        # Graphviz's 0.75x0.5in default and the id as a label, and node size
        # drives dot and fdp placement. Measured on one 17-node graph, the dot
        # aspect ratio came out 7.89 through pygraphviz and 1.62 through the
        # binary -- a reported metric depending on which happened to be
        # installed.
        agraph.node_attr.update(shape="circle", width="0.3", height="0.3",
                                label="", fixedsize="true")
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

    if shutil.which(engine) is None:
        raise RuntimeError("graphviz not installed: no %s binary, no pygraphviz"
                           % engine)

    index = {node: "n%d" % i for i, node in enumerate(graph.nodes)}
    lines = ["digraph G {",
             "  node [shape=circle,width=0.3,height=0.3,label=\"\","
             "fixedsize=true];"]
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
        return normalise(_graphviz(graph, method))

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
    return normalise({k: (float(v[0]), float(v[1])) for k, v in raw.items()})
