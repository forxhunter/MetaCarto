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

# Cycles every curated map draws as a ring, whatever the flux does.
#
# Finding rings among *directed* cycles of the oriented graph is not enough for
# these. pFBA orients each reaction by the flux it carries under the model's own
# objective, and Recon3D's runs the TCA cycle partly reductively (ACONTm,
# ICDHym and SUCOAS carry negative flux), so citrate <- isocitrate breaks the
# directed cycle and the TCA cycle came out as a straight column. MitoMammal
# did the same. A reader recognises the TCA cycle by its shape before reading a
# label, and a reductive flux is something the data overlay shows, not
# something the drawing should hide.
#
# Each cycle is its anchor compounds in textbook order. Consecutive anchors may
# be joined through up to `_MAX_GAP - 1` other compounds (cis-aconitate between
# citrate and isocitrate, succinyl-CoA between 2-oxoglutarate and succinate),
# and through transport steps, which is how the urea cycle crosses the
# mitochondrial membrane. A compound matches an anchor by its BiGG species, its
# KEGG id, or its name, so the cycle is found in Human-GEM and Yeast-GEM too.
CANONICAL_CYCLES = {
    "citric acid cycle": (
        ("oaa", "C00036", ("oxaloacetate", "oxalacetate")),
        ("cit", "C00158", ("citrate", "citric acid")),
        ("icit", "C00311", ("isocitrate", "d-threo-isocitrate", "isocitric acid")),
        ("akg", "C00026", ("2-oxoglutarate", "alpha-ketoglutarate", "2-oxoglutaric acid",
                           "akg", "oxoglutarate")),
        # An anchor, not a gap: every 2-oxoglutarate-dependent dioxygenase
        # turns akg into succinate, and with succinyl-CoA optional Recon3D's
        # ring closed through a prolyl hydroxylase instead of the TCA cycle.
        ("succoa", "C00091", ("succinyl-coa", "succinyl coenzyme a")),
        ("succ", "C00042", ("succinate", "succinic acid")),
        ("fum", "C00122", ("fumarate", "fumaric acid")),
        ("mal__L", "C00149", ("l-malate", "malate", "(s)-malate")),
    ),
    "urea cycle": (
        ("orn", "C00077", ("ornithine", "l-ornithine")),
        ("citr__L", "C00327", ("citrulline", "l-citrulline")),
        ("argsuc", "C03406", ("argininosuccinate", "n(omega)-(l-arginino)succinate",
                              "l-argininosuccinate")),
        ("arg__L", "C00062", ("arginine", "l-arginine")),
    ),
    "methionine cycle": (
        ("met__L", "C00073", ("methionine", "l-methionine")),
        ("amet", "C00019", ("s-adenosyl-l-methionine", "s-adenosylmethionine")),
        ("ahcys", "C00021", ("s-adenosyl-l-homocysteine", "s-adenosylhomocysteine")),
        ("hcys__L", "C00155", ("homocysteine", "l-homocysteine")),
    ),
}
_MAX_GAP = 3


def _currency_keys():
    from .compound import NEVER_PRIMARY, SUPPRESSED
    return NEVER_PRIMARY | SUPPRESSED


def _anchor_matches(node, anchor):
    from . import identity

    species, kegg, names = anchor
    return (identity.species(node) == species
            or kegg in identity.kegg(node)
            or identity.name(node) in names)


def _shortest_path(U, start, targets, forbidden, max_len):
    """Shortest path from `start` to any of `targets` avoiding `forbidden`.

    Breadth-first over sorted neighbours, so ties resolve the same way every
    run. Among paths of equal length the one ending in `start`'s compartment
    wins: the TCA cycle in the mitochondrion should not close through a
    cytosolic copy of malate when a mitochondrial one is reachable.
    """
    from . import identity

    home = identity.compartment(start)
    frontier, seen = [[start]], {start}
    for _ in range(max_len):
        reached, extended = [], []
        for path in frontier:
            for nb in sorted(U.neighbors(path[-1]), key=str):
                if nb in targets:
                    reached.append(path + [nb])
                elif nb not in seen and nb not in forbidden:
                    seen.add(nb)
                    extended.append(path + [nb])
        if reached:
            reached.sort(key=lambda p: (identity.compartment(p[-1]) != home, str(p[-1])))
            return reached[0]
        frontier = extended
    return None


def _can_run(cgraph, rid, x, y):
    """True if reaction `rid` may carry flux from compound x to compound y."""
    rec = cgraph.reactions.get(rid)
    if rec is None:
        return False
    if rec.reversible:
        return True
    st = rec.stoichiometry
    return st.get(x, 0) < 0 < st.get(y, 0)


def canonical_cycles(cgraph):
    """Find each canonical cycle present in the graph and claim its arcs.

    Returns [(name, [node, ...])], the node list in the cycle's textbook
    direction.

    An arc between two consecutive anchors may be carried by a reaction whose
    main pair is something else. Citrate synthase is the case that matters:
    by formula, acetyl-CoA -> citrate shares more carbon than oxaloacetate ->
    citrate, so the TCA cycle had no oxaloacetate -> citrate edge and could
    never close. Every curated map draws citrate synthase on the ring with
    acetyl-CoA joining it, so on a canonical cycle the ring arc becomes the
    reaction's main pair and its old partner a side branch.

    A cycle is returned only if every arc has a reaction that can run that way
    -- an irreversible step written against the cycle means the model does not
    contain it as a cycle, and drawing a ring would assert something false.
    """
    from . import identity

    D = cgraph.D
    found, used = [], set()
    for name, anchors in CANONICAL_CYCLES.items():
        candidates = [sorted((n for n in D.nodes if _anchor_matches(n, a)), key=str)
                      for a in anchors]
        if any(not c for c in candidates):
            continue
        k = len(anchors)

        # Adjacency: the drawn edges, plus a step from each anchor to every
        # non-currency compound on the other side of a reaction it takes part
        # in. That covers citrate synthase (oxaloacetate -> citrate, whatever
        # its main pair), and a step split over a carrier: yeast's 2-oxoglutarate
        # dehydrogenase runs akg -> S-succinyldihydrolipoamide -> succinyl-CoA,
        # and by formula both halves pair lipoamide with lipoamide, so akg was
        # joined to nothing on the way to succinyl-CoA.
        A = nx.Graph(D.to_undirected(as_view=True))
        links = {}

        def link(x, y, rid):
            links.setdefault(frozenset((x, y)), set()).add(rid)
            A.add_edge(x, y)

        anchored = set().union(*map(set, candidates))
        currency = _currency_keys()
        for rid, rec in cgraph.reactions.items():
            if rec.is_boundary or rec.main_sub is None:
                continue
            st = rec.stoichiometry
            for x in anchored.intersection(st):
                for y, c in st.items():
                    if c * st[x] < 0 and y in D and identity.canonical(y) not in currency:
                        link(x, y, rid)

        def arc_reactions(x, y):
            rxns = set(links.get(frozenset((x, y)), ()))
            for e in ((x, y), (y, x)):
                if D.has_edge(*e):
                    rxns.update(D.edges[e]["rxns"])
            return rxns

        options = []
        for start in candidates[0]:
            if start in used:
                continue
            path, ok = [start], True
            for i in range(1, k + 1):
                targets = {start} if i == k else set(candidates[i]) - used
                # Other anchors are off limits between two anchors, so a step
                # is never skipped by a shortcut through a later compound.
                # Copies of the two anchors being joined are allowed: that is
                # citrulline crossing from the mitochondrion to the cytosol.
                others = set().union(*(set(candidates[j]) for j in range(k)
                                       if j not in (i - 1, i % k)))
                forbidden = (others | set(path) | used) - targets
                leg = _shortest_path(A, path[-1], targets, forbidden, _MAX_GAP)
                if leg is None:
                    ok = False
                    break
                path.extend(leg[1:])
            cycle = path[:-1]
            if not ok or not MIN_RING <= len(cycle) <= MAX_RING or len(set(cycle)) != len(cycle):
                continue
            arcs = list(zip(cycle, cycle[1:] + cycle[:1]))
            if not all(any(_can_run(cgraph, r, x, y) for r in arc_reactions(x, y))
                       for x, y in arcs):
                continue
            compartments = len({identity.compartment(n) for n in cycle})
            options.append((compartments, len(cycle), str(start), cycle))
        # Several compartments can each hold a copy; draw every disjoint one,
        # the most self-contained first.
        for *_, cycle in sorted(options, key=lambda o: o[:3]):
            if used.isdisjoint(cycle):
                for x, y in zip(cycle, cycle[1:] + cycle[:1]):
                    if not (D.has_edge(x, y) or D.has_edge(y, x)):
                        for rid in sorted(links.get(frozenset((x, y)), ())):
                            if _can_run(cgraph, rid, x, y):
                                _repair(cgraph, rid, x, y)
                found.append((name, cycle))
                used.update(cycle)
    return found


def _repair(cgraph, rid, x, y):
    """Make x -> y (in stored orientation) the main pair of reaction `rid`."""
    D, rec = cgraph.D, cgraph.reactions[rid]
    old = (rec.main_sub, rec.main_prod)
    if D.has_edge(*old):
        rxns = D.edges[old]["rxns"]
        if rid in rxns:
            rxns.remove(rid)
        if not rxns:
            D.remove_edge(*old)
    st = rec.stoichiometry
    sub, prod = (x, y) if st.get(x, 0) < 0 else (y, x)
    rec.main_sub, rec.main_prod = sub, prod
    rec.consumed = [m for m, c in st.items() if c < 0 and m != sub]
    rec.produced = [m for m, c in st.items() if c > 0 and m != prod]
    if D.has_edge(sub, prod):
        D.edges[sub, prod]["rxns"].append(rid)
    else:
        D.add_edge(sub, prod, rxns=[rid])
    for node in old:
        if node in D and D.degree(node) == 0 and node not in cgraph.roles:
            D.remove_node(node)


def orient_cycle(cgraph, cycle, flipped):
    """Turn every reaction that can run around `cycle` to run that way.

    `flipped` is the set of reaction ids drawn against their stored main pair;
    it is updated in place. Reactions that cannot run that way (an
    irreversible step written against the cycle) stay where they are and are
    drawn alongside as the opposed member of the pair.
    """
    D = cgraph.D
    for x, y in zip(cycle, cycle[1:] + cycle[:1]):
        if D.has_edge(y, x):
            data = D.edges[y, x]
            movable = [r for r in data["rxns"] if _can_run(cgraph, r, x, y)]
            if movable:
                data["rxns"] = [r for r in data["rxns"] if r not in movable]
                if D.has_edge(x, y):
                    D.edges[x, y]["rxns"].extend(movable)
                else:
                    D.add_edge(x, y, rxns=list(movable))
                if not data["rxns"]:
                    D.remove_edge(y, x)
        if not D.has_edge(x, y):
            continue
        for r in D.edges[x, y]["rxns"]:
            # Decided from the stored pair, not toggled: an edge only some of
            # whose reactions flux flipped stays put, so the edge a reaction
            # sits on does not always say which way it is drawn.
            if not _can_run(cgraph, r, x, y):
                continue
            if cgraph.reactions[r].main_sub == x:
                flipped.discard(r)
            else:
                flipped.add(r)


def find_rings(D, min_len=MIN_RING, max_len=MAX_RING, preferred=()):
    """Node-disjoint directed cycles, longest first.

    `preferred` cycles (the canonical ones, already oriented) are taken first,
    whatever their chords: the TCA cycle with the glyoxylate shunt across it is
    still the TCA cycle, and the shorter shunt ring it would otherwise lose to
    leaves 2-oxoglutarate and succinyl-CoA hanging off on a long detour.

    Among the rest, chordless cycles are preferred: a cycle with a chord is two
    smaller cycles sharing an arc, and drawing the outer one as a circle would
    hide the chord across the middle.
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
    for cycle in preferred:
        n = len(cycle)
        if (used.isdisjoint(cycle) and all(D.has_edge(cycle[i], cycle[(i + 1) % n])
                                           for i in range(n))):
            chosen.append(list(cycle))
            used.update(cycle)
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
