"""Functional decomposition into drawable clusters.

A cluster is a unit a reader is meant to recognise, so it has to mean something
biologically and it has to be big enough to be worth a map of its own. Two
failure modes to avoid, and both were present:

  * Structural community detection on an unannotated model returns hundreds of
    communities, most of them singletons -- iAF1260 came out as 371 clusters of
    which 217 held one reaction. A one-reaction "pathway" is not a pathway.
  * Topology alone is not biology. Reactions that share a metabolite are not
    thereby part of the same pathway, and a curator would not group them that
    way.

So annotation is used wherever it exists and community detection is the last
resort, not the first: declared subsystem, then KEGG pathway via the reaction's
own `kegg.reaction` / `ec-code` annotation, then structure. Size limits from
`constraints.md` are then enforced over *all* paths, and an undersized cluster
is merged into the neighbour it shares the most chemistry with rather than
being swept into an "Uncategorized" bucket.
"""

import json
import os

import networkx as nx

MAX_CLUSTER = 60
MIN_CLUSTER = 6          # constraints.md

KEGG_MAPPING_FILE = os.path.join("data", "kegg", "kegg_mapping.json")
_kegg_mapping = None


def load_kegg_mapping(path=KEGG_MAPPING_FILE):
    """{KEGG reaction id or EC number: [pathway names]}. Empty if unavailable."""
    global _kegg_mapping
    if _kegg_mapping is None:
        try:
            with open(path, encoding="utf-8") as handle:
                _kegg_mapping = json.load(handle)
        except Exception:
            _kegg_mapping = {}
    return _kegg_mapping


def declared_subsystems(reactions):
    groups = {}
    for reaction in reactions:
        key = (getattr(reaction, "subsystem", "") or "").strip() or "Uncategorized"
        groups.setdefault(key, []).append(reaction)
    return groups


def kegg_pathway(reaction, mapping):
    """The reaction's primary KEGG pathway, or None.

    `kegg.reaction` is preferred over `ec-code`: an EC number is an enzyme
    activity and can be shared by reactions in unrelated pathways, so it is a
    weaker signal and only consulted when the reaction id is missing.
    """
    for key in ("kegg.reaction", "ec-code"):
        value = reaction.annotation.get(key)
        if not value:
            continue
        for identifier in (value if isinstance(value, list) else [value]):
            pathways = mapping.get(identifier)
            if pathways:
                return pathways[0]
    return None


# --------------------------------------------------------------------------
# structural fallback
# --------------------------------------------------------------------------

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


def split_cluster(reactions, cofactor_score, max_size=MAX_CLUSTER,
                  min_size=MIN_CLUSTER, _depth=0):
    """Split a reaction set into pieces that each fit `max_size`.

    Community detection answers "what are the communities", not "what are the
    pieces of at most 60 reactions", and on a genome-scale model it returns
    dozens of communities holding two or three reactions each. Returning those
    verbatim is what produced 217 single-reaction clusters, and -- because the
    small ones were then pooled and split again into the same communities --
    what made the size-enforcement loop fail to terminate.

    So the communities are bin-packed afterwards: each is placed in the bin it
    shares the most metabolites with, which keeps chemically related fragments
    together instead of merely balancing counts.
    """
    reactions = list(reactions)
    if len(reactions) <= max_size:
        return [reactions]

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

    if len(pieces) < 2 or _depth > 4:
        # Community detection could not separate it; chop deterministically so
        # the caller still gets drawable pieces rather than one unreadable one.
        ordered = sorted(reactions, key=lambda r: r.id)
        return [ordered[i:i + max_size] for i in range(0, len(ordered), max_size)]

    expanded = []
    for piece in pieces:
        if len(piece) > max_size:
            expanded.extend(split_cluster(piece, cofactor_score, max_size,
                                          min_size, _depth + 1))
        else:
            expanded.append(piece)

    return _bin_pack(expanded, cofactor_score, max_size, min_size)


def _bin_pack(pieces, cofactor_score, max_size, min_size):
    """Combine small pieces into bins of at most `max_size`.

    Placement is by shared chemistry rather than by size alone: two fragments
    of the same pathway belong in the same map, and a bin chosen purely to
    balance counts would put unrelated chemistry together.
    """
    pieces = sorted(pieces, key=len, reverse=True)
    bins, bin_mets = [], []

    for piece in pieces:
        piece_mets = _cluster_metabolites(piece, cofactor_score)
        best, best_shared = None, 0
        for index, existing in enumerate(bins):
            if len(existing) + len(piece) > max_size:
                continue
            # A bin that is already big enough should not absorb more unless
            # the piece would otherwise be stranded.
            shared = len(bin_mets[index] & piece_mets)
            if best is None or shared > best_shared or (
                    shared == best_shared and len(existing) < len(bins[best])):
                best, best_shared = index, shared

        # Only combine pieces that actually share chemistry. Packing unrelated
        # communities together merely to reach the size floor manufactures a
        # 54-reaction "cluster" that is five unrelated pathways -- which is
        # exactly why such clusters cannot be given a name: there is nothing
        # they have in common to name. Undersized leftovers are handled by
        # merge_small, which merges on shared chemistry too.
        if best is None or best_shared == 0:
            bins.append(list(piece))
            bin_mets.append(piece_mets)
        else:
            bins[best].extend(piece)
            bin_mets[best] |= piece_mets

    return bins


# --------------------------------------------------------------------------
# size enforcement
# --------------------------------------------------------------------------

def _cluster_metabolites(reactions, cofactor_score, cutoff=0.5):
    return {m.id for r in reactions for m in r.metabolites
            if cofactor_score.get(m.id, 0.0) < cutoff}


STRUCTURAL_PREFIXES = ("Unannotated", "Other", "Uncategorized", "Cluster_")


def is_biological(name):
    """True when the cluster name came from annotation rather than topology."""
    return not str(name).startswith(STRUCTURAL_PREFIXES)


BOUNDARY_PREFIXES = ("EX_", "DM_", "SK_", "BIOMASS", "ATPM")


def is_boundary_reaction(reaction, cofactor_score, cutoff=0.5):
    """Exchange, demand, sink, maintenance or biomass step.

    Having no substrates or no products is not enough of a test. The
    e_coli_core biomass reaction has sixteen substrates and seven products --
    but every product is currency (ADP, Pi, NADH, CoA, ...), so it has no
    primary product and behaves as a sink. Checking for a *primary* partner
    catches it; checking for any partner does not.
    """
    identifier = reaction.id.upper()
    if identifier.startswith(BOUNDARY_PREFIXES):
        return True

    substrates = [m for m, c in reaction.metabolites.items() if c < 0]
    products = [m for m, c in reaction.metabolites.items() if c > 0]
    if not substrates or not products:
        return True

    def has_primary(metabolites):
        return any(cofactor_score.get(m.id, 0.0) < cutoff for m in metabolites)

    return not has_primary(substrates) or not has_primary(products)


def is_boundary_cluster(reactions, cofactor_score):
    """True when every reaction in the cluster is a boundary step.

    Such clusters are legitimately small and must not be folded into a pathway.
    Biomass in particular touches most of central carbon metabolism, so the
    merge rule -- "join the neighbour you share the most chemistry with" --
    reliably drops a sixteen-substrate fan into the middle of glycolysis and
    ruins the one tile that was easiest to read.
    """
    return bool(reactions) and all(
        is_boundary_reaction(r, cofactor_score) for r in reactions)


def merge_small(groups, cofactor_score, min_size=MIN_CLUSTER, max_size=MAX_CLUSTER,
                boundary=()):
    """Fold undersized clusters into the neighbour they share the most with.

    Merging into a shared "Uncategorized" bucket, as the previous version did,
    puts a two-reaction fragment of purine metabolism next to an unrelated
    transport step and calls the result a pathway. Merging into the most
    chemically connected neighbour instead keeps the fragment with the pathway
    it came from, and the name follows the larger partner.

    Metabolite sets are cached and an inverted index limits each search to the
    clusters that actually share a metabolite. Recomputing both sides for every
    candidate pair, which is the obvious way to write this, is O(reactions) per
    comparison and does not finish on a genome-scale model.
    """
    groups = {name: list(reactions) for name, reactions in groups.items()}
    if len(groups) < 2:
        return groups

    mets = {name: _cluster_metabolites(reactions, cofactor_score)
            for name, reactions in groups.items()}
    owners = {}
    for name, metabolites in mets.items():
        for metabolite in metabolites:
            owners.setdefault(metabolite, set()).add(name)

    def neighbours(name):
        found = set()
        for metabolite in mets[name]:
            found |= owners.get(metabolite, set())
        found.discard(name)
        return found

    def absorb(keep, drop):
        groups[keep] = groups[keep] + groups[drop]
        for metabolite in mets[drop]:
            owners[metabolite].discard(drop)
            owners[metabolite].add(keep)
        mets[keep] = mets[keep] | mets[drop]
        del groups[drop]
        del mets[drop]

    # Boundary and pathway clusters merge within their own kind but never
    # across: keeping biomass out of glycolysis is the point, but a lone
    # exchange reaction still has no business being its own map.
    boundary = set(boundary)
    while True:
        undersized = [n for n, r in groups.items() if len(r) < min_size]
        if not undersized:
            break
        # Smallest first: the hardest to place, and merging it may make a
        # neighbour large enough that a later merge is unnecessary.
        name = min(undersized, key=lambda n: (len(groups[n]), n))

        best, best_score = None, None
        for other in neighbours(name):
            if (name in boundary) != (other in boundary):
                continue
            if len(groups[other]) + len(groups[name]) > max_size:
                continue
            shared = len(mets[name] & mets[other])
            score = (shared, -len(groups[other]))
            if shared > 0 and (best_score is None or score > best_score):
                best, best_score = other, score

        if best is None:
            # Nothing it shares chemistry with and fits into. Pool the
            # remaining orphans together rather than looping forever.
            orphans = [n for n in undersized]
            if len(orphans) < 2:
                break
            pooled = []
            for orphan in orphans:
                pooled.extend(groups.pop(orphan))
                del mets[orphan]
            # Chop, do not re-cluster. Running community detection over the
            # pooled orphans can hand back the very fragments that were just
            # pooled, and the loop never terminates.
            pooled.sort(key=lambda r: r.id)
            chunks = [pooled[i:i + max_size] for i in range(0, len(pooled), max_size)]
            for piece in chunks:
                key = _unique(groups, "Other")
                groups[key] = piece
                mets[key] = _cluster_metabolites(piece, cofactor_score)
                for metabolite in mets[key]:
                    owners.setdefault(metabolite, set()).add(key)
            for metabolite, holders in owners.items():
                holders &= set(groups)
            continue

        # The surviving name should be the one that means something. A
        # four-reaction fragment of purine metabolism absorbed into a
        # structurally derived blob is still purine metabolism, and letting the
        # larger partner always win would throw that away.
        if is_biological(name) != is_biological(best):
            keep, drop = (name, best) if is_biological(name) else (best, name)
        else:
            keep, drop = ((best, name) if len(groups[best]) >= len(groups[name])
                          else (name, best))
        absorb(keep, drop)

    return groups


def _unique(groups, base):
    if base not in groups:
        return base
    index = 2
    while f"{base} {index}" in groups:
        index += 1
    return f"{base} {index}"


# --------------------------------------------------------------------------
# cycle integrity
# --------------------------------------------------------------------------

def close_cycles(groups, model, cofactor_score, max_size=MAX_CLUSTER, verbose=False):
    """Pull a split cycle back into one cluster.

    A cluster boundary must not cut a ring. e_coli_core files succinate
    dehydrogenase under oxidative phosphorylation, so the cluster named
    "Citric Acid Cycle" holds eight reactions with no succinate -> fumarate
    step: the ring exists in the model, but not in the map, and the TCA cycle
    gets drawn as a straight chain under a caption promising a cycle. That is
    the single most recognisable shape in metabolism and the one a reader
    checks first.

    So rings are detected once on the whole-model compound graph, and any ring
    whose reactions are spread across clusters is consolidated into the cluster
    already holding most of it.
    """
    from .compound import build_compound_graph
    from .direction import orient_compound_graph
    from .motifs import find_rings

    whole = build_compound_graph(model, list(model.reactions))
    orient_compound_graph(whole, model=model)
    rings = find_rings(whole.D)
    if not rings:
        return groups

    owner = {}
    for name, reactions in groups.items():
        for reaction in reactions:
            owner[reaction.id] = name

    groups = {name: list(reactions) for name, reactions in groups.items()}
    moved_total = 0

    for cycle in rings:
        members = set(cycle)
        # Reactions whose main pair is an arc of this ring.
        arc_reactions = []
        for u, v, data in whole.D.edges(data=True):
            if u in members and v in members:
                arc_reactions.extend(data["rxns"])

        holders = {}
        for rid in arc_reactions:
            name = owner.get(rid)
            if name is not None:
                holders.setdefault(name, []).append(rid)
        if len(holders) < 2:
            continue

        target = max(holders, key=lambda n: (len(holders[n]), -len(groups[n])))
        incoming = [rid for name, rids in holders.items() if name != target
                    for rid in rids]
        if len(groups[target]) + len(incoming) > max_size:
            continue

        by_id = {r.id: r for r in model.reactions}
        for rid in incoming:
            source = owner[rid]
            groups[source] = [r for r in groups[source] if r.id != rid]
            groups[target].append(by_id[rid])
            owner[rid] = target
            moved_total += 1
        if verbose:
            print(f"    closed a {len(cycle)}-member ring into '{target}' "
                  f"({len(incoming)} reactions moved)")

    return {name: reactions for name, reactions in groups.items() if reactions}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def clusters(model, cofactor_score, max_size=MAX_CLUSTER, min_size=MIN_CLUSTER,
             kegg_mapping=None, close_rings=True, name_structural=True,
             verbose=False):
    """{cluster name: [reactions]}, biological where the model allows it."""
    mapping = load_kegg_mapping() if kegg_mapping is None else kegg_mapping

    declared = declared_subsystems(model.reactions)
    real = {n: r for n, r in declared.items() if n != "Uncategorized"}
    leftovers = list(declared.get("Uncategorized", []))

    groups = dict(real)
    source = "subsystem" if real else None

    # KEGG pathway for whatever the model did not annotate itself.
    if leftovers and mapping:
        assigned, still_left = {}, []
        for reaction in leftovers:
            pathway = kegg_pathway(reaction, mapping)
            if pathway:
                assigned.setdefault(pathway, []).append(reaction)
            else:
                still_left.append(reaction)
        if assigned:
            for name, reactions in assigned.items():
                groups[_unique(groups, name) if name in groups else name] = reactions
            leftovers = still_left
            source = source or "KEGG"
            if verbose:
                covered = sum(len(r) for r in assigned.values())
                print(f"    KEGG pathways: {len(assigned)} clusters, {covered} reactions")

    # Structure only for what annotation could not reach.
    if leftovers:
        for piece in split_cluster(leftovers, cofactor_score, max_size):
            groups[_unique(groups, "Unannotated")] = piece
        source = source or "structure"
        if verbose:
            print(f"    structural fallback: {len(leftovers)} reactions")

    # A boundary step inside an otherwise ordinary subsystem still has to come
    # out: e_coli_core files biomass under "Biomass and maintenance functions",
    # but many reconstructions leave it in whatever subsystem it fell into.
    pulled = []
    for name in list(groups):
        keep = [r for r in groups[name] if not is_boundary_reaction(r, cofactor_score)]
        moved = [r for r in groups[name] if is_boundary_reaction(r, cofactor_score)]
        if moved and keep:
            groups[name] = keep
            pulled.extend(moved)
        elif not keep:
            pass                        # already an all-boundary cluster
    if pulled:
        groups[_unique(groups, "Biomass and exchange")] = pulled

    # Size limits apply to every path, not just the annotated one.
    sized = {}
    for name, reactions in groups.items():
        pieces = split_cluster(reactions, cofactor_score, max_size)
        if len(pieces) == 1:
            sized[name] = pieces[0]
        else:
            for index, piece in enumerate(pieces):
                sized[f"{name} ({index + 1})"] = piece

    boundary = {name for name, reactions in sized.items()
                if is_boundary_cluster(reactions, cofactor_score)}
    merged = merge_small(sized, cofactor_score, min_size=min_size,
                         max_size=max_size, boundary=boundary)

    # Name whatever structure produced before anything else looks at the names.
    # A caption reading "Cluster_8" tells a reader nothing, and the reactions in
    # that cluster already say what they are.
    if name_structural:
        from .naming import name_clusters
        merged = name_clusters(merged, model)

    # Ring consolidation has to see the final cluster boundaries, so it runs
    # last -- but taking reactions out of a donor can leave the donor under the
    # floor, so the floor is re-applied afterwards. Merging only ever grows a
    # cluster, so it cannot re-cut a ring that was just closed.
    if close_rings:
        merged = close_cycles(merged, model, cofactor_score, max_size=max_size,
                              verbose=verbose)
        boundary = {name for name, reactions in merged.items()
                    if is_boundary_cluster(reactions, cofactor_score)}
        merged = merge_small(merged, cofactor_score, min_size=min_size,
                             max_size=max_size, boundary=boundary)

    # After merging, boundary clusters no longer need one name per chunk.
    renamed, index = {}, 0
    for name, reactions in merged.items():
        if is_boundary_cluster(reactions, cofactor_score) and len(merged) > 1:
            index += 1
            label = "Exchange and biomass" if index == 1 else f"Exchange and biomass {index}"
            renamed[label] = reactions
        else:
            renamed[name] = reactions
    return renamed
