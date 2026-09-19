"""Primary-compound reduction (layout_algorithm.md S1).

Collapses the bipartite reaction/metabolite graph into a sparse *directed*
graph over primary metabolites, one edge per reaction, plus a table of side
(cofactor) metabolites per reaction. This is the representation curated KEGG
and Escher maps are drawn in, and it is what makes an orthogonal drawing
possible at all: in the bipartite graph every reaction node has degree 4-8.
"""

import math
from dataclasses import dataclass, field

import networkx as nx

from .formula import parse_formula, moiety_score

# Prior, not the rule. Metabolites outside this list are still detected as
# cofactors by connectivity; the list exists so that a small subsystem, where
# nothing looks high-degree locally, still gets ATP/NAD(P)H right.
CURATED_COFACTORS = {
    "h", "h2o", "o2", "co2", "pi", "ppi", "pppi", "nh4", "nh3", "so4", "so3",
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "utp", "udp", "ump", "ctp",
    "cdp", "cmp", "itp", "idp", "imp", "datp", "dadp", "damp", "dgtp", "dctp",
    "dttp", "nad", "nadh", "nadp", "nadph", "fad", "fadh2", "fmn", "fmnh2",
    "coa", "q8", "q8h2", "mqn8", "mql8", "q6", "q6h2", "thf", "mlthf",
    "methf", "5mthf", "10fthf", "trdox", "trdrd", "gthox", "gthrd",
    "na1", "k", "cl", "ca2", "mg2", "fe2", "fe3", "mn2", "zn2", "cu2",
    "cobalt2", "hco3", "h2o2", "no", "no2", "no3", "n2", "h2", "h2s",
}

# Never drawn as nodes. Curated maps omit protons and water entirely -- t1
# draws zero of each -- because they participate in most reactions and carry no
# information when they do. They accounted for roughly 30% of every disc in the
# emitted figures.
SUPPRESSED = {"h", "h2o"}

# Never drawn as a primary (large, coloured) node, whatever the main-pair
# search decided, unless a map has no other chemistry at all. A big orange ATP
# reads as a pathway intermediate.
NEVER_PRIMARY = {
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "utp", "udp", "ump", "ctp",
    "cdp", "cmp", "pi", "ppi", "pppi", "nad", "nadh", "nadp", "nadph",
    "fad", "fadh2", "fmn", "fmnh2", "coa", "co2", "o2", "h2o2", "nh4",
    "gthrd", "gthox", "q8", "q8h2", "mqn8", "mql8", "trdox", "trdrd",
    "h", "h2o",
}

# Multiplicative, not additive: a cofactor pair such as ATP -> ADP shares a
# large absolute skeleton, so an additive penalty would need retuning per
# model. Scaling the score instead is unit-free and holds across models.
_COFACTOR_DAMPING = 0.88


@dataclass
class ReactionRecord:
    rid: str
    name: str
    reversible: bool
    stoichiometry: dict
    main_sub: str = None
    main_prod: str = None
    consumed: list = field(default_factory=list)   # side metabolites, substrate side
    produced: list = field(default_factory=list)   # side metabolites, product side
    is_boundary: bool = False
    genes: str = ""


@dataclass
class CompoundGraph:
    D: nx.DiGraph                 # primary metabolites; edge attr 'rxns' = [reaction id]
    reactions: dict               # reaction id -> ReactionRecord
    metabolites: dict             # metabolite id -> {name, formula, compartment, degree, virtual}
    roles: dict                   # metabolite id -> 'source' | 'sink'
    cofactor_score: dict          # metabolite id -> [0, 1]


def strip_compartment(met_id):
    """'atp_c' -> 'atp'. BiGG compartment suffixes are 1-2 characters."""
    for n in (2, 3):
        if len(met_id) > n and met_id[-n] == "_":
            return met_id[:-n]
    return met_id


def compute_cofactor_scores(model):
    """Score in [0, 1]: how cofactor-like each metabolite is.

    Connectivity is measured on the *whole* model, not on the subsystem being
    drawn -- a metabolite in 200 reactions is currency wherever it appears,
    even if the current subsystem happens to use it twice.
    """
    degrees = {m.id: len(m.reactions) for m in model.metabolites}
    if not degrees:
        return {}

    ordered = sorted(degrees.values())
    n = len(ordered)
    # Connectivity contributes nothing below the 85th percentile and ramps to
    # 1.0 at the 98th.
    lo = ordered[max(0, min(n - 1, int(0.85 * n)))]
    hi = ordered[max(0, min(n - 1, int(0.98 * n)))]

    scores = {}
    for met_id, deg in degrees.items():
        if hi > lo:
            by_degree = (deg - lo) / (hi - lo)
        else:
            by_degree = 1.0 if deg >= hi else 0.0
        by_degree = max(0.0, min(1.0, by_degree))
        by_prior = 1.0 if strip_compartment(met_id) in CURATED_COFACTORS else 0.0
        scores[met_id] = max(by_degree, by_prior)
    return scores


_COFACTOR_CUTOFF = 0.5


def _candidate_pairs(rxn, formulas, cofactor_score, degrees):
    """Rank candidate (substrate, product) main pairs for one reaction.

    Cofactor-ness is applied as a *tier*, not only as a score penalty. Carrier
    chemistry is the reason: CoA -> acetyl-CoA shares 21 carbons and scores far
    higher than pyruvate -> acetyl-CoA, so any purely additive or multiplicative
    penalty has to be tuned until it happens to invert that one comparison.
    Tiering states the actual rule instead -- a curated/high-degree cofactor is
    never a backbone member while a non-cofactor alternative exists on its side
    -- and degrades gracefully: when *every* participant is currency (H2O_e ->
    H2O_c transport, ion exchange, NAD biosynthesis) the whole reaction sits in
    one tier and raw chemistry decides.

    Returns [(substrate, product, score)] within the best available tier,
    best first.
    """
    subs = [m.id for m, c in rxn.metabolites.items() if c < 0]
    prods = [m.id for m, c in rxn.metabolites.items() if c > 0]
    if not subs or not prods:
        return []

    candidates = []
    for s in subs:
        for p in prods:
            if s == p:
                continue
            cof_s = cofactor_score.get(s, 0.0)
            cof_p = cofactor_score.get(p, 0.0)
            # The curated list outranks the connectivity score.
            #
            # `compute_cofactor_scores` is a connectivity percentile, and on a
            # genome-scale model it saturates: in iJO1366 pyruvate and
            # acetyl-CoA both score 1.000, exactly like NAD. Every participant
            # of pyruvate dehydrogenase therefore landed in the same tier, the
            # tier carried no signal, the damping cancelled, and the ranking
            # fell back to raw shared chemistry -- which picks NAD -> NADH,
            # because they share an entire ADP-ribose skeleton. PDH is the
            # example this module's own docstring uses to explain why tiering
            # exists, and it was drawing the cofactor pair as the backbone.
            # Acetyl-CoA synthetase was drawing CoA -> acetyl-CoA for the same
            # reason.
            #
            # NEVER_PRIMARY was defined here for exactly this and then consulted
            # only by the renderer. Weighted above the score tier it degrades
            # the same graceful way: when every participant is on the list
            # (water transport, ion exchange) all pairs tie and raw chemistry
            # still decides.
            # Only the curated list tiers. The connectivity percentile stays
            # as the damping term below, where a continuous score belongs.
            #
            # Tiering on it as well was measured and dropped: against the
            # substrate/product pairs KEGG's curators drew, across four models,
            # it cost 0.9, 0.4 and 0.7 points of agreement and gained nothing
            # on the fourth. It saturates -- pyruvate and acetyl-CoA score
            # 1.000 in iJO1366, the same as NAD -- so as a *threshold* it fires
            # on ordinary metabolites and adds noise, while as a weight it
            # still orders candidates usefully.
            never = ((strip_compartment(s) in NEVER_PRIMARY)
                     + (strip_compartment(p) in NEVER_PRIMARY))
            tier = never

            value = moiety_score(formulas.get(s, {}), formulas.get(p, {}))
            # Soft damping still orders candidates *within* a tier.
            value *= (1.0 - _COFACTOR_DAMPING * cof_s) * (1.0 - _COFACTOR_DAMPING * cof_p)
            # Tie-breakers: prefer the less connected pair, and a pair that
            # stays in one compartment -- a transport step should read as one
            # arrow, not as a compartment change bolted onto a chemical change.
            # Both are multiplicative and bounded. Subtracting an absolute
            # constant instead lets connectivity swamp the score outright on a
            # genome-scale model, where H2O sits in thousands of reactions:
            # catalase and CO2 transport then score negative, get filtered, and
            # the reaction silently vanishes from the map.
            connectivity = min(degrees.get(s, 0) + degrees.get(p, 0), 2000)
            value *= (1.0 - 0.1 * connectivity / 2000.0)
            if s[-2:] == p[-2:]:
                value *= 1.02
            if value > 0.0:
                candidates.append((tier, -value, s, p))

    if not candidates:
        return []

    candidates.sort()
    best_tier = candidates[0][0]
    return [(s, p, -neg) for tier, neg, s, p in candidates if tier == best_tier]


_ALTERNATE_TOLERANCE = 0.70


def _resolve_main_pairs(pending, passes=3):
    """Commit one main pair per reaction, preferring globally coherent choices.

    Chemistry alone often leaves two near-tied candidates -- citrate synthase
    scores acetyl-CoA -> citrate and oxaloacetate -> citrate within 2% of each
    other. Chemistry cannot break that tie, but topology can: only one of them
    closes the TCA ring. So candidates within `_ALTERNATE_TOLERANCE` of the
    best are re-scored by what they do to the backbone -- close a cycle, or
    join two disconnected pieces -- which is the same criterion a human curator
    applies when deciding which arrow is "the" pathway.
    """
    for rec, candidates in pending.values():
        rec.main_sub, rec.main_prod = candidates[0][0], candidates[0][1]

    flexible = []
    for rec, candidates in pending.values():
        best = candidates[0][2]
        alternates = [c for c in candidates if c[2] >= _ALTERNATE_TOLERANCE * best]
        if len(alternates) > 1:
            flexible.append((rec, alternates, best))
    if not flexible:
        return

    for _ in range(passes):
        changed = False
        edge_use = {}
        for rec, _candidates in pending.values():
            key = (rec.main_sub, rec.main_prod)
            edge_use[key] = edge_use.get(key, 0) + 1

        for rec, alternates, best in flexible:
            current = (rec.main_sub, rec.main_prod)
            others = nx.DiGraph()
            for key, count in edge_use.items():
                if key == current and count == 1:
                    continue
                others.add_edge(*key)
            components = {
                node: i
                for i, comp in enumerate(nx.weakly_connected_components(others))
                for node in comp
            }

            def value(option):
                s, p, score = option
                v = score / best
                if others.has_node(p) and others.has_node(s) and nx.has_path(others, p, s):
                    v += 1.0          # closes a cycle
                elif components.get(s, -1) != components.get(p, -2):
                    v += 0.8          # joins two disconnected pieces
                # Continue an existing chain rather than dangle a new stub off
                # it: prefer a substrate something already produces and a
                # product something already consumes.
                if others.has_node(s) and others.in_degree(s) > 0:
                    v += 0.3
                if others.has_node(p) and others.out_degree(p) > 0:
                    v += 0.3
                return v

            choice = max(alternates, key=value)
            if (choice[0], choice[1]) != current:
                edge_use[current] -= 1
                if edge_use[current] == 0:
                    del edge_use[current]
                new_key = (choice[0], choice[1])
                edge_use[new_key] = edge_use.get(new_key, 0) + 1
                rec.main_sub, rec.main_prod = choice[0], choice[1]
                changed = True

        if not changed:
            break


def _add_edge(D, u, v, rid):
    if D.has_edge(u, v):
        D.edges[u, v]["rxns"].append(rid)
    else:
        D.add_edge(u, v, rxns=[rid])


def build_compound_graph(model, reactions):
    """Reduce `reactions` (a list of cobra Reactions) to a CompoundGraph."""
    cofactor_score = compute_cofactor_scores(model)
    degrees = {m.id: len(m.reactions) for m in model.metabolites}
    formulas = {m.id: parse_formula(m.formula) for m in model.metabolites}

    D = nx.DiGraph()
    records, met_info, roles, pending = {}, {}, {}, {}

    def touch(met_id, name=None, virtual=False):
        if met_id not in met_info:
            met = model.metabolites.get_by_id(met_id) if met_id in model.metabolites else None
            met_info[met_id] = {
                "name": (met.name if met is not None else name) or met_id,
                "formula": met.formula if met is not None else "",
                "compartment": met.compartment if met is not None else "",
                "degree": degrees.get(met_id, 0),
                "virtual": virtual or met is None,
            }
        D.add_node(met_id)

    for rxn in reactions:
        stoich = {m.id: c for m, c in rxn.metabolites.items()}
        rec = ReactionRecord(
            rid=rxn.id,
            name=rxn.name or rxn.id,
            reversible=(rxn.lower_bound < 0 < rxn.upper_bound),
            stoichiometry=stoich,
            genes=getattr(rxn, "gene_reaction_rule", "") or "",
        )
        subs = [m for m, c in stoich.items() if c < 0]
        prods = [m for m, c in stoich.items() if c > 0]

        if not subs and not prods:
            continue

        if not subs or not prods:
            # Boundary reaction: exchange, demand, sink, or biomass.
            rec.is_boundary = True
            present = subs or prods
            for met_id in present:
                touch(met_id)

            if subs:
                role = "source" if rxn.lower_bound < 0 else "sink"
            else:
                role = "source"

            if len(present) == 1:
                # Ordinary exchange: no extra node, just a boundary role that
                # pins the metabolite to the top or bottom of the drawing.
                met_id = present[0]
                if role == "source":
                    rec.main_prod = met_id
                else:
                    rec.main_sub = met_id
                roles[met_id] = role
            else:
                # Biomass / multi-substrate demand: draw as a fan into a named
                # pool node, the way curated maps draw "Biomass".
                pool = f"{rxn.id}__pool"
                touch(pool, name=rxn.name or rxn.id, virtual=True)
                roles[pool] = "sink" if subs else "source"
                for met_id in present:
                    if subs:
                        _add_edge(D, met_id, pool, rxn.id)
                    else:
                        _add_edge(D, pool, met_id, rxn.id)
                rec.main_sub = present[0] if subs else pool
                rec.main_prod = pool if subs else present[0]
            records[rxn.id] = rec
            continue

        candidates = _candidate_pairs(rxn, formulas, cofactor_score, degrees)
        if not candidates:
            continue
        pending[rxn.id] = (rec, candidates)
        records[rxn.id] = rec

    _resolve_main_pairs(pending)

    for rid, (rec, _) in pending.items():
        subs = [m for m, c in rec.stoichiometry.items() if c < 0]
        prods = [m for m, c in rec.stoichiometry.items() if c > 0]
        rec.consumed = [m for m in subs if m != rec.main_sub]
        rec.produced = [m for m in prods if m != rec.main_prod]

        touch(rec.main_sub)
        touch(rec.main_prod)
        for met_id in rec.consumed + rec.produced:
            touch(met_id)
        _add_edge(D, rec.main_sub, rec.main_prod, rid)

    # Metabolites that ended up with no edge are pure cofactors of this
    # subsystem. They are dropped from the layout graph but still drawn, as
    # per-reaction side nodes.
    #
    # Boundary metabolites are the exception: an exchange reaction contributes
    # no edge because it has only one participant, so dropping its metabolite
    # deletes the reaction from the map. A cluster of nothing but exchanges --
    # which is exactly what `Extracellular exchange` is -- would then vanish
    # entirely. Curated maps draw these as named nodes on the border.
    for met_id in [n for n in D.nodes if D.degree(n) == 0 and n not in roles]:
        D.remove_node(met_id)

    return CompoundGraph(
        D=D,
        reactions=records,
        metabolites=met_info,
        roles=roles,
        cofactor_score=cofactor_score,
    )
