"""Ring motifs (layout_algorithm.md S2/S4).

A layered drawing has to break every cycle, so on its own it renders the TCA
cycle as a straight chain plus one long edge doubling back. Every curated map
instead draws it as a ring, and readers recognise the shape before they read
any label. So cycles are detected first, contracted to a single super-node for
the layering pass, and expanded afterwards onto a circle whose rotation is
chosen to face its entry and exit edges.
"""

import math

import networkx as nx

MIN_RING = 4          # triangles in metabolism are usually futile cycles
MAX_RING = 14
_CYCLE_ENUMERATION_CAP = 20000


def find_rings(D, min_len=MIN_RING, max_len=MAX_RING):
    """Node-disjoint directed cycles, longest first.

    Chordless cycles are preferred: a cycle with a chord is two smaller cycles
    sharing an arc, and drawing the outer one as a circle would hide the chord
    across the middle.
    """
    cycles = []
    try:
        for i, cycle in enumerate(nx.simple_cycles(D, length_bound=max_len)):
            if i >= _CYCLE_ENUMERATION_CAP:
                break
            if len(cycle) >= min_len:
                cycles.append(cycle)
    except TypeError:                         # networkx < 3.1
        for cycle in nx.simple_cycles(D):
            if min_len <= len(cycle) <= max_len:
                cycles.append(cycle)

    def chordless(cycle):
        members = set(cycle)
        n = len(cycle)
        adjacent = {
            frozenset((cycle[i], cycle[(i + 1) % n])) for i in range(n)
        }
        for u in cycle:
            for v in members:
                if u == v or frozenset((u, v)) in adjacent:
                    continue
                if D.has_edge(u, v):
                    return False
        return True

    cycles.sort(key=lambda c: (not chordless(c), -len(c)))

    chosen, used = [], set()
    for cycle in cycles:
        if used.isdisjoint(cycle):
            chosen.append(cycle)
            used.update(cycle)
    return chosen


def ring_radius(size, pitch):
    """Radius that gives consecutive members roughly `pitch` of arc between."""
    return max(pitch, size * pitch / (2.0 * math.pi))


def contract_rings(D, rings, pitch):
    """Return (contracted graph, ring records). Ring records carry the layout
    size of each super-node so the layering can reserve space for it."""
    ring_of = {}
    records = {}
    for i, cycle in enumerate(rings):
        super_id = f"__ring__{i}"
        radius = ring_radius(len(cycle), pitch)
        records[super_id] = {
            "members": list(cycle),
            "radius": radius,
            "size": 2.0 * radius,
        }
        for node in cycle:
            ring_of[node] = super_id

    C = nx.DiGraph()
    C.add_nodes_from(n for n in D.nodes if n not in ring_of)
    C.add_nodes_from(records)

    for u, v, data in D.edges(data=True):
        a = ring_of.get(u, u)
        b = ring_of.get(v, v)
        if a == b:
            continue                      # internal ring arc (or a chord)
        if C.has_edge(a, b):
            C.edges[a, b]["rxns"].extend(data["rxns"])
        else:
            C.add_edge(a, b, rxns=list(data["rxns"]))
    return C, records, ring_of


def expand_rings(pos, records, ring_of, D):
    """Place ring members on their circle, rotated to face their outside world.

    The rotation (and winding direction) is chosen by brute force over the
    2n possibilities -- n is at most 14 -- minimising the distance from each
    member to the neighbours it connects to outside the ring. That is what
    makes the ring's entry arc point at the pathway feeding it.
    """
    placed = dict(pos)

    for super_id, record in records.items():
        if super_id not in pos:
            continue
        cx, cy = pos[super_id]
        members = record["members"]
        n = len(members)
        radius = record["radius"]

        external = {}
        for m in members:
            targets = []
            for nb in list(D.predecessors(m)) + list(D.successors(m)):
                if ring_of.get(nb) == super_id:
                    continue
                anchor = ring_of.get(nb, nb)
                if anchor in pos:
                    targets.append(pos[anchor])
            external[m] = targets

        def positions(offset, winding):
            out = {}
            for i, m in enumerate(members):
                angle = 2.0 * math.pi * ((winding * i + offset) % n) / n - math.pi / 2.0
                out[m] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            return out

        def cost(candidate):
            total = 0.0
            for m, targets in external.items():
                mx, my = candidate[m]
                for tx, ty in targets:
                    total += (mx - tx) ** 2 + (my - ty) ** 2
            return total

        best = min(
            (positions(offset, winding) for offset in range(n) for winding in (1, -1)),
            key=cost,
        )
        placed.pop(super_id, None)
        placed.update(best)

    return placed
