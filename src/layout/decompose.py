"""Functional decomposition into drawable clusters.

`constraints.md` caps a cluster at 60 reactions and requires at least 6, and
those limits exist for a reason the layered pass makes concrete: a single
drawing of 2360 reactions is correct, orthogonal, and unreadable. Many BiGG
models (iAF1260 among them) carry no `subsystem` annotation at all, so the
split has to be inferred.

Order of preference matches `constraints.md` S1: declared subsystem, then
KEGG/EC annotation, then structural community detection.
"""

import networkx as nx

MAX_CLUSTER = 60
MIN_CLUSTER = 6


def declared_subsystems(reactions):
    groups = {}
    for reaction in reactions:
        key = (getattr(reaction, "subsystem", "") or "").strip() or "Uncategorized"
        groups.setdefault(key, []).append(reaction)
    return groups


def _reaction_graph(reactions, cofactor_score, cutoff=0.5):
    """Reaction-metabolite graph with currency removed.

    Currency metabolites connect everything to everything; leaving them in
    makes community detection return one community, which is the failure the
    whole decomposition exists to avoid.
    """
    graph = nx.Graph()
    ids = {r.id for r in reactions}
    for reaction in reactions:
        graph.add_node(reaction.id)
        for metabolite in reaction.metabolites:
            if cofactor_score.get(metabolite.id, 0.0) >= cutoff:
                continue
            graph.add_node(metabolite.id)
            graph.add_edge(reaction.id, metabolite.id)
    return graph, ids


def split_cluster(reactions, cofactor_score, max_size=MAX_CLUSTER):
    """Recursively split a reaction set until every piece fits `max_size`."""
    if len(reactions) <= max_size:
        return [list(reactions)]

    graph, reaction_ids = _reaction_graph(reactions, cofactor_score)
    by_id = {r.id: r for r in reactions}

    try:
        communities = nx.community.greedy_modularity_communities(graph)
    except Exception:
        communities = []

    pieces = []
    for community in communities:
        members = [by_id[n] for n in community if n in reaction_ids]
        if members:
            pieces.append(members)

    if len(pieces) < 2:
        # Community detection could not separate it; chop deterministically so
        # the caller still gets drawable pieces rather than one unreadable one.
        ordered = sorted(reactions, key=lambda r: r.id)
        step = max_size
        return [ordered[i:i + step] for i in range(0, len(ordered), step)]

    out = []
    for piece in pieces:
        if len(piece) > max_size:
            out.extend(split_cluster(piece, cofactor_score, max_size))
        else:
            out.append(piece)
    return out


def clusters(model, cofactor_score, max_size=MAX_CLUSTER, min_size=MIN_CLUSTER):
    """{cluster name: [reactions]} obeying the constraints.md size limits."""
    groups = declared_subsystems(model.reactions)

    if len(groups) <= 1:
        only = next(iter(groups.values())) if groups else []
        pieces = split_cluster(only, cofactor_score, max_size)
        return {f"Cluster_{i}": piece for i, piece in enumerate(pieces)}

    out, leftovers = {}, []
    for name, reactions in groups.items():
        if len(reactions) < min_size:
            leftovers.extend(reactions)
            continue
        pieces = split_cluster(reactions, cofactor_score, max_size)
        if len(pieces) == 1:
            out[name] = pieces[0]
        else:
            for i, piece in enumerate(pieces):
                out[f"{name}_{i}"] = piece

    if leftovers:
        for i, piece in enumerate(split_cluster(leftovers, cofactor_score, max_size)):
            out[f"Uncategorized_{i}" if i else "Uncategorized"] = piece
    return out
