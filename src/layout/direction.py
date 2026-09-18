"""Reaction orientation (layout_algorithm.md S1, "edge direction").

BiGG writes reversible reactions in whichever direction the reconstruction
happened to pick, so PGM appears as 2pg -> 3pg and PGK as 3pg -> 13dpg, i.e.
glycolysis is stored partly backwards. Drawn as stored, the pathway fragments
into three short chains instead of one spine. Orientation therefore has to be
decided, not read off the model.

Two sources, in priority order:

  1. parsimonious FBA flux sign -- the direction the reaction actually carries
     flux in under the model's own medium and objective. This is the
     biologically correct answer and it is cheap (one LP per model, cached).
  2. a linear arrangement of the compound graph, used for reactions that carry
     no flux. Every reversible reaction is offered to the feedback-arc-set
     solver in *both* directions and the arrangement picks one, which maximises
     the length of the chains the layered drawing can then straighten.
"""

FLUX_TOLERANCE = 1e-9

_flux_cache = {}


def flux_directions(model, verbose=False):
    """{reaction id: +1 | -1} from pFBA. Empty dict if the LP does not solve."""
    key = id(model)
    if key in _flux_cache:
        return _flux_cache[key]

    directions = {}
    try:
        from cobra.flux_analysis import pfba

        solution = pfba(model)
        for rid, flux in solution.fluxes.items():
            if abs(flux) > FLUX_TOLERANCE:
                directions[rid] = 1 if flux > 0 else -1
        if verbose:
            print(f"    pFBA: {len(directions)}/{len(model.reactions)} reactions carry flux")
    except Exception as exc:                                  # infeasible, no solver, ...
        if verbose:
            print(f"    pFBA unavailable ({exc}); using topological orientation only")

    _flux_cache[key] = directions
    return directions


def orient_compound_graph(cgraph, model=None, use_fba=True, verbose=False):
    """Flip edges of `cgraph.D` in place so the drawing reads with the biology.

    Returns the set of reaction ids whose stored direction was reversed, so the
    renderer can draw their arrowheads correctly.
    """
    import networkx as nx

    from .sugiyama import greedy_feedback_arc_set

    D = cgraph.D
    directions = flux_directions(model, verbose=verbose) if (use_fba and model is not None) else {}

    # --- pass 1: flux sign ---
    flipped_reactions = set()
    decided = set()
    for u, v, data in list(D.edges(data=True)):
        signs = {directions.get(r, 0) for r in data["rxns"]}
        signs.discard(0)
        if len(signs) != 1:
            continue
        decided.add((u, v))
        if signs.pop() < 0:
            flipped_reactions.update(data["rxns"])

    # --- pass 2: linear arrangement for everything flux left undecided ---
    # Offer each undecided reversible edge in both directions; the arrangement
    # keeps whichever agrees with the global ordering.
    probe = nx.DiGraph()
    probe.add_nodes_from(D.nodes)
    cost = {}
    for u, v, data in D.edges(data=True):
        reversible = any(cgraph.reactions[r].reversible for r in data["rxns"])
        stored = (v, u) if any(r in flipped_reactions for r in data["rxns"]) else (u, v)
        probe.add_edge(*stored)
        cost[stored] = 6.0 if (u, v) in decided else (1.0 if reversible else 3.0)
        if reversible and (u, v) not in decided:
            back = stored[::-1]
            if not probe.has_edge(*back):
                probe.add_edge(*back)
                cost[back] = 1.0

    reverse = greedy_feedback_arc_set(probe, lambda a, b: cost.get((a, b), 1.0))

    for u, v, data in list(D.edges(data=True)):
        if (u, v) in decided:
            continue
        stored = (v, u) if any(r in flipped_reactions for r in data["rxns"]) else (u, v)
        back = stored[::-1]
        # Reversed by the arrangement, and the opposite direction survived.
        if stored in reverse and back not in reverse:
            flipped_reactions.update(data["rxns"])

    # --- apply ---
    for u, v, data in list(D.edges(data=True)):
        if all(r in flipped_reactions for r in data["rxns"]):
            D.remove_edge(u, v)
            if D.has_edge(v, u):
                D.edges[v, u]["rxns"].extend(data["rxns"])
            else:
                D.add_edge(v, u, **data)

    flipped_reactions |= _collapse_antiparallel(D, directions)

    for rid in flipped_reactions:
        rec = cgraph.reactions.get(rid)
        if rec is None:
            continue
        rec.main_sub, rec.main_prod = rec.main_prod, rec.main_sub
        rec.consumed, rec.produced = rec.produced, rec.consumed

    return flipped_reactions


def _collapse_antiparallel(D, directions):
    """Merge futile-cycle pairs (PFK/FBP, PYK/PPS) onto a single axis.

    A kinase and its phosphatase are stored as two opposed edges between the
    same metabolites. Left alone they are a 2-cycle, which forces the layering
    to break one arbitrarily and pushes fructose-1,6-bisphosphate above
    glucose-6-phosphate. Curated maps draw them as two parallel arrows between
    the same pair of nodes, so collapse them to one edge here and let the
    renderer lay the opposed reaction alongside it.
    """
    flipped = set()
    for u, v in list(D.edges):
        if u >= v or not D.has_edge(v, u):
            continue

        forward = D.edges[u, v]["rxns"]
        backward = D.edges[v, u]["rxns"]

        def rank(rxns):
            return (sum(1 for r in rxns if r in directions), len(rxns))

        if rank(backward) > rank(forward):
            keep, drop, keep_key, drop_key = backward, forward, (v, u), (u, v)
        else:
            keep, drop, keep_key, drop_key = forward, backward, (u, v), (v, u)

        flipped.update(drop)
        D.edges[keep_key]["rxns"] = list(keep) + list(drop)
        D.edges[keep_key]["opposed"] = list(drop)
        D.remove_edge(*drop_key)
    return flipped
