"""Tests for bugs that actually happened.

Every case here shipped at some point. They are written against the specific
failure, not against a general notion of correctness, because a general test
would not have caught any of them.
"""

import json
import re
import os

import pytest


# --------------------------------------------------------------------------
# Main-pair selection. Shipped wrong: PDH drew nad_c -> nadh_c and ACS drew
# coa_c -> accoa_c, because NEVER_PRIMARY was defined in compound.py and only
# ever read by render.py, while the tier came from a connectivity percentile
# that saturates -- pyruvate and acetyl-CoA both score 1.000, same as NAD.
# --------------------------------------------------------------------------

def _main_pairs(model, reactions):
    from src.layout.compound import build_compound_graph
    cgraph = build_compound_graph(model, reactions)
    out = {}
    for u, v, data in cgraph.D.edges(data=True):
        for rid in data.get("rxns", ()):
            out.setdefault(rid, (u, v))
    return out


def test_pdh_draws_the_carbon_backbone_not_the_cofactor_pair(core_model):
    pairs = _main_pairs(core_model, list(core_model.reactions))
    assert pairs.get("PDH") == ("pyr_c", "accoa_c"), (
        "pyruvate dehydrogenase must be drawn pyruvate -> acetyl-CoA. It "
        "shipped as nad_c -> nadh_c."
    )


def test_carrier_loses_even_when_every_cofactor_score_saturates(core_model):
    """The actual invariant behind the PDH bug, testable on a small model.

    `compute_cofactor_scores` is a connectivity percentile, and it only
    saturates at genome scale: in iJO1366 pyruvate, acetyl-CoA and NAD all
    score 1.000, while in e_coli_core pyruvate scores 0.222 and acetyl-CoA
    0.000. So the bug cannot be reproduced on the small model at all -- a
    suite that only drew e_coli_core passed with the fix reverted, which is
    how this test came to exist.

    Feeding saturated scores directly reproduces the genome-scale condition
    without loading a genome-scale model: every participant at the top
    percentile, so the score tier carries no information and only the curated
    list can separate the carrier pair from the carbon backbone.
    """
    from src.layout.compound import _candidate_pairs
    from src.layout.formula import parse_formula

    rxn = core_model.reactions.get_by_id("PDH")
    formulas = {m.id: parse_formula(m.formula) for m in core_model.metabolites}
    saturated = {m.id: 1.0 for m in rxn.metabolites}
    degrees = {m.id: 100 for m in rxn.metabolites}

    ranked = _candidate_pairs(rxn, formulas, saturated, degrees)
    assert ranked, "no candidate pairs for PDH"
    substrate, product, _score = ranked[0]
    assert (substrate, product) == ("pyr_c", "accoa_c"), (
        "with every cofactor score saturated the carrier pair nad_c -> nadh_c "
        "wins on raw shared chemistry unless NEVER_PRIMARY tiers above it; "
        "got %s -> %s" % (substrate, product))


def test_citrate_synthase_draws_a_pair_kegg_also_draws(core_model):
    """Citrate synthase must produce citrate from one of its real substrates.

    layout_algorithm.md says the backbone-effect re-scoring resolves the
    accoa/oaa near-tie in favour of oaa. That is model-dependent and not what
    happens here: iJO1366 gives oaa_c -> cit_c, e_coli_core gives
    accoa_c -> cit_c. Neither is wrong -- KEGG's own drawing of R00351
    contains both pairs -- but the documentation overstates how determined
    that choice is, so this asserts what is actually required.
    """
    pairs = _main_pairs(core_model, list(core_model.reactions))
    assert pairs.get("CS") in {("oaa_c", "cit_c"), ("accoa_c", "cit_c")}


def test_carriers_are_only_a_backbone_when_nothing_else_is_available(core_model):
    """A carrier pair is allowed only when every participant is a carrier.

    That is the documented fallback -- H2O_e -> H2O_c transport, ion exchange,
    ATPM -- and it is correct. What must never happen is choosing a carrier
    pair while a non-carrier substrate or product was available, which is the
    PDH failure.
    """
    from src.layout.compound import NEVER_PRIMARY, strip_compartment
    pairs = _main_pairs(core_model, list(core_model.reactions))
    by_id = {r.id: r for r in core_model.reactions}

    avoidable = []
    for rid, (a, b) in pairs.items():
        if strip_compartment(a) not in NEVER_PRIMARY:
            continue
        if strip_compartment(b) not in NEVER_PRIMARY:
            continue
        rxn = by_id.get(rid)
        if rxn is None:
            continue
        others = [m.id for m in rxn.metabolites
                  if strip_compartment(m.id) not in NEVER_PRIMARY]
        if others:
            avoidable.append((rid, a, b, others[:3]))
    assert not avoidable, (
        "carrier backbone chosen while non-carriers were available: %s"
        % avoidable[:5])


def test_the_tiering_ablation_actually_disables_tiering(core_model):
    """An ablation that measures nothing must not report that as "no effect".

    `no_cofactor_tiering` patched `_COFACTOR_CUTOFF`, which stopped being read
    when the tier moved to NEVER_PRIMARY membership. The patch was therefore a
    no-op, and the ablation reported agreement with KEGG identical to the full
    method -- to the reaction, on iJO1366 and iYO844 -- which reads as "the
    carrier constraint is worth nothing". Corrected, it is worth 3.9-6.3
    points. Every variant has to be shown to bite before its number means
    anything.
    """
    from src.bench import ablate
    from src.layout.compound import _candidate_pairs
    from src.layout.formula import parse_formula

    rxn = core_model.reactions.get_by_id("PDH")
    formulas = {m.id: parse_formula(m.formula) for m in core_model.metabolites}
    saturated = {m.id: 1.0 for m in rxn.metabolites}
    degrees = {m.id: 100 for m in rxn.metabolites}

    def chosen():
        s, p, _ = _candidate_pairs(rxn, formulas, saturated, degrees)[0]
        return s, p

    assert chosen() == ("pyr_c", "accoa_c")
    with ablate.no_cofactor_tiering():
        assert chosen() == ("nad_c", "nadh_c"), (
            "no_cofactor_tiering left the carrier constraint in force, so the "
            "ablation measures nothing")
    assert chosen() == ("pyr_c", "accoa_c"), "patch was not undone"


# --------------------------------------------------------------------------
# Escher schema. v1 emitted node_type "reaction", null b1/b2 and a hardcoded
# coefficient of 1; Escher will not render any of that.
# --------------------------------------------------------------------------

LEGAL_NODE_TYPES = {"metabolite", "multimarker", "midmarker"}


def test_only_the_three_legal_node_types(core_map):
    kinds = {n["node_type"] for n in core_map[1]["nodes"].values()}
    assert kinds <= LEGAL_NODE_TYPES, "illegal node types: %s" % (
        kinds - LEGAL_NODE_TYPES)


def test_every_segment_endpoint_exists(core_map):
    nodes = core_map[1]["nodes"]
    for rid, reaction in core_map[1]["reactions"].items():
        for sid, seg in reaction["segments"].items():
            assert seg["from_node_id"] in nodes, (rid, sid)
            assert seg["to_node_id"] in nodes, (rid, sid)


def test_stoichiometry_is_signed(core_map):
    # Direction is derived from the sign, so unsigned coefficients silently
    # lose every arrowhead.
    signs = set()
    for reaction in core_map[1]["reactions"].values():
        for met in reaction.get("metabolites", []):
            signs.add(met["coefficient"] < 0)
    assert signs == {True, False}, (
        "expected both consumed and produced metabolites; v1 hardcoded every "
        "coefficient to 1 and lost direction entirely")


def test_reactions_carry_a_midmarker(core_map):
    nodes = core_map[1]["nodes"]
    for rid, reaction in core_map[1]["reactions"].items():
        kinds = set()
        for seg in reaction["segments"].values():
            kinds.add(nodes[seg["from_node_id"]]["node_type"])
            kinds.add(nodes[seg["to_node_id"]]["node_type"])
        assert "midmarker" in kinds, "reaction %s has no midmarker" % rid


# --------------------------------------------------------------------------
# Geometry. Node overlap regressed repeatedly: the Brandes-Koepf balancing
# step takes a per-node median over four runs, which is not a convex
# combination, and the renderer then adds cofactor stubs the layout never saw.
# --------------------------------------------------------------------------

def test_primary_metabolites_do_not_overlap(core_map):
    import math
    nodes = [n for n in core_map[1]["nodes"].values()
             if n["node_type"] == "metabolite" and n.get("node_is_primary", True)]
    worst = None
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            gap = math.hypot(a["x"] - b["x"], a["y"] - b["y"]) - 60.0
            if worst is None or gap < worst[0]:
                worst = (gap, a.get("bigg_id"), b.get("bigg_id"))
    assert worst is None or worst[0] >= 0, "overlapping primaries: %s" % (worst,)


# --------------------------------------------------------------------------
# Determinism. The method's claim is that it is constructive, not annealed:
# the same model must always give the same map.
# --------------------------------------------------------------------------

def test_layout_is_deterministic(core_model, core_clusters):
    from src.layout.engine import layout_reactions
    name = sorted(core_clusters)[0]
    first = layout_reactions(core_model, core_clusters[name], name)
    second = layout_reactions(core_model, core_clusters[name], name)
    assert json.dumps(first.escher_map, sort_keys=True) == \
           json.dumps(second.escher_map, sort_keys=True)


def test_node_numbering_does_not_depend_on_dict_order(core_model, core_clusters):
    """Determinism has to survive leaving the process, not just the call.

    Node ids come from a counter, so they follow the order metabolites are
    added, and that order used to be `pos`'s -- which inherits from the layered
    pass and passes through a set upstream, so it varied with PYTHONHASHSEED.
    Two runs of `layout_v2.py` in separate processes produced maps that were
    geometrically identical and byte-different, while the in-process test above
    passed. Feeding the same positions in a different order reproduces that
    without spawning an interpreter.
    """
    from src.layout.engine import layout_reactions
    from src.layout.render import build_escher_map

    name = sorted(core_clusters)[0]
    result = layout_reactions(core_model, core_clusters[name], name, render=False)
    assert result is not None and len(result.pos) > 2

    forward = build_escher_map(result.cgraph, dict(result.pos), name)
    reversed_pos = dict(reversed(list(result.pos.items())))
    backward = build_escher_map(result.cgraph, reversed_pos, name)

    assert json.dumps(forward, sort_keys=True) == \
           json.dumps(backward, sort_keys=True), (
        "node numbering follows the order positions arrive in, so the emitted "
        "JSON is not reproducible across processes")


# --------------------------------------------------------------------------
# Rings. find_rings runs after orientation, and the acyclic orientation used
# when there is no flux removes every cycle first -- so --no-fba silently
# turned ring drawing off. Pinned so that coupling cannot change unnoticed.
# --------------------------------------------------------------------------

def test_tca_cycle_is_drawn_as_a_circle(core_model, core_clusters):
    import math
    import statistics
    from src.layout.engine import layout_reactions

    for name in sorted(core_clusters):
        result = layout_reactions(core_model, core_clusters[name], name,
                                  use_fba=True, render=False)
        if result is None:
            continue
        # The rings the layout drew. Re-detecting here without the canonical
        # preference measured the glyoxylate-shunt cycle, a chord of the drawn
        # TCA ring, rather than the ring itself.
        for ring in result.rings:
            points = [result.pos[m] for m in ring if m in result.pos]
            if len(points) < 3:
                continue
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            radii = [math.hypot(x - cx, y - cy) for x, y in points]
            cv = statistics.pstdev(radii) / (sum(radii) / len(radii))
            assert cv < 0.10, (
                "cycle in %s drawn with radial CV %.3f, i.e. not round" %
                (name, cv))
            return
    pytest.skip("no cycle detected in e_coli_core; nothing to assert")


@pytest.mark.parametrize("use_fba", [True, False])
def test_tca_ring_is_the_whole_cycle_whatever_the_flux(core_model, use_fba):
    """The drawn ring is the textbook TCA cycle, not the glyoxylate shunt.

    Ring detection used to take directed cycles of the flux-oriented graph.
    Recon3D's pFBA runs aconitase backwards, which broke the cycle and drew
    the TCA cycle as a column; without FBA no ring was drawn at all; and in
    e_coli_core the shorter shunt ring won, leaving 2-oxoglutarate and
    succinyl-CoA on a long detour.
    """
    from src.layout.engine import layout_reactions

    result = layout_reactions(core_model, list(core_model.reactions), "core",
                              use_fba=use_fba, render=False)
    tca = {"oaa_c", "cit_c", "icit_c", "akg_c", "succoa_c", "succ_c", "fum_c", "mal__L_c"}
    assert any(tca <= set(ring) for ring in result.rings), result.rings
    cs = result.cgraph.reactions["CS"]
    assert (cs.main_sub, cs.main_prod) == ("oaa_c", "cit_c"), (
        "citrate synthase is drawn on the ring, with acetyl-CoA joining it")


def test_tca_ring_survives_reductive_flux(core_model):
    """A reversible step running against the cycle does not open the ring."""
    from src.layout import direction
    from src.layout.engine import layout_reactions

    fluxes = dict(direction.flux_directions(core_model))
    fluxes["ACONTa"] = fluxes["ACONTb"] = fluxes["ICDHyr"] = -1
    direction._flux_cache[id(core_model)] = fluxes
    try:
        result = layout_reactions(core_model, list(core_model.reactions), "core",
                                  render=False)
    finally:
        direction._flux_cache.pop(id(core_model), None)
    assert any({"cit_c", "icit_c", "akg_c"} <= set(ring) for ring in result.rings)


def test_proton_transport_is_drawn():
    """H+ -> H+ shares no skeleton, which used to drop the reaction entirely."""
    cobra = pytest.importorskip("cobra")
    from src.layout.engine import layout_reactions

    model = cobra.Model("protons")
    h = {c: cobra.Metabolite("h_" + c, formula="H", charge=1, compartment=c)
         for c in "ecm"}
    for rid, a, b in (("Hct", "e", "c"), ("Hmt", "e", "m")):
        r = cobra.Reaction(rid, lower_bound=-1000, upper_bound=1000)
        r.add_metabolites({h[a]: -1, h[b]: 1})
        model.add_reactions([r])
    result = layout_reactions(model, list(model.reactions), "protons", use_fba=False)
    drawn = {r["bigg_id"] for r in result.escher_map[1]["reactions"].values()}
    assert drawn == {"Hct", "Hmt"}


# --------------------------------------------------------------------------
# Publishing. prepare_repo.py used to overwrite the hand-written README with a
# four-line stub on every sync.
# --------------------------------------------------------------------------

def test_prepare_repo_does_not_generate_a_readme():
    source = open(os.path.join("scripts", "prepare_repo.py"),
                  encoding="utf-8").read()
    assert "README" not in source.split('"""')[2] or "left untouched" in source
    assert 'write("# Escher Maps' not in source


# --------------------------------------------------------------------------
# Naming the published maps. A third of the collection shipped as "Other
# metabolism (N)" and nearly every split map as "<superclass> (N)".
# --------------------------------------------------------------------------

class _Met:
    def __init__(self, mid, name="", formula="C6H12O6"):
        self.id, self.name, self.formula = mid, name or mid, formula

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class _Rxn:
    def __init__(self, rid, consumed, produced, subsystem="", name=""):
        self.id, self.subsystem, self.name = rid, subsystem, name or rid
        self.annotation = {}
        self.metabolites = {_Met(m): -1.0 for m in consumed}
        self.metabolites.update({_Met(m): 1.0 for m in produced})


def test_bigg_models_are_fetched_with_their_subsystems():
    """BiGG's SBML export drops `subsystem`; its JSON export keeps it. Built
    from SBML, every model but e_coli_core looked unannotated."""
    source = open(os.path.join("scripts", "fetch_bigg.py"), encoding="utf-8").read()
    assert 'default="json"' in source


def test_subsystem_strings_are_cleaned_into_captions():
    from src.layout.decompose import clean_subsystem
    assert clean_subsystem("Lipid &amp; Cell Wall Metabolism") == "Lipid & Cell Wall Metabolism"
    assert clean_subsystem("S_Alternate_Carbon_source") == "Alternate Carbon source"
    assert clean_subsystem("&apos;") == ""
    assert clean_subsystem("Fatty acid synthesis: omega-3") == "Fatty acid synthesis - omega-3"


def test_a_kegg_pathway_starting_with_other_is_still_a_pathway():
    """The structural-name test was an unanchored prefix, so KEGG's "Other
    glycan degradation" was treated as a placeholder and renamed."""
    from src.layout.taxonomy import is_structural, superclass
    assert not is_structural("Other glycan degradation")
    assert superclass("Other glycan degradation") == "Glycan metabolism"
    assert superclass("Other carbon fixation pathways") == "Energy metabolism"
    for placeholder in ("Other", "Other 3", "Unannotated 12 (2)", "UNASSIGNED",
                        "Miscellaneous", "Cluster_8"):
        assert is_structural(placeholder), placeholder


def test_superclass_recognises_its_own_page_titles():
    """"Carbohydrate metabolism (2)" matched no keyword, so on the combined map
    every carbohydrate page was regrouped under Other."""
    from src.layout.taxonomy import LABELS, superclass
    for label in LABELS:
        assert superclass(label) == label
        assert superclass(f"{label} (2)") == label
        assert superclass(f"{label}: Glycolysis; Pentose phosphate") == label


def test_kegg_names_follow_brite_not_the_first_keyword_they_contain():
    from src.layout.taxonomy import superclass
    assert superclass("Mannose type O-glycan biosynthesis") == "Glycan metabolism"
    assert superclass("Keratan sulfate degradation") == "Glycan metabolism"
    assert superclass("Heme degradation") == "Cofactor and vitamin metabolism"
    assert superclass("Phosphatidylinositol phosphate metabolism") == "Lipid metabolism"
    assert superclass("Brassinosteroid biosynthesis") == "Terpenoid and polyketide metabolism"
    assert superclass("Tetrahydrobiopterin metabolism") == "Cofactor and vitamin metabolism"


def test_a_piece_qualifier_does_not_reclassify_the_piece():
    """"Extracellular exchange: amino acids and peptides" matched "amino acid"
    before "exchange" and was filed as amino-acid metabolism."""
    from src.layout.taxonomy import superclass
    assert superclass("Extracellular exchange: amino acids and peptides") == \
        "Transport and exchange"


def test_an_uninformative_name_is_classified_by_its_reactions():
    from src.layout.taxonomy import OTHER, UNASSIGNED, classify
    reactions = [_Rxn(f"R{i}", [f"a{i}"], [f"b{i}"], subsystem="Purine metabolism")
                 for i in range(8)]
    assert classify("Miscellaneous", reactions) == "Nucleotide metabolism"
    assert classify("Hydratase", reactions) == "Nucleotide metabolism"
    # No evidence at all: say so rather than guess.
    blank = [_Rxn(f"R{i}", [f"a{i}"], [f"b{i}"]) for i in range(8)]
    assert classify("Hydratase", blank) == OTHER
    assert classify("Miscellaneous", blank) == UNASSIGNED


def test_one_unplaceable_cluster_does_not_drag_the_others_into_a_pool():
    """merge_small pooled *every* undersized cluster the moment the smallest
    had no partner, then chopped the pool alphabetically by reaction id: a
    third of the annotated corpus was drawn as runs like 10FTHF5GLUtm ..
    CRVNCtr, spanning twenty subsystems."""
    from src.layout.decompose import merge_small
    pathway = [_Rxn(f"P{i}", [f"m{i}"], [f"m{i + 1}"], "Purine metabolism")
               for i in range(10)]
    fragment = [_Rxn("F1", ["m3"], ["x1"], "Purine metabolism"),
                _Rxn("F2", ["x1"], ["x2"], "Purine metabolism")]
    lone_a = [_Rxn("A1", ["q1"], ["q2"], "Thiamine metabolism")]
    lone_b = [_Rxn("B1", ["r1"], ["r2"], "Riboflavin metabolism")]
    groups = {"Purine metabolism": pathway, "Purine metabolism (2)": fragment,
              "Thiamine metabolism": lone_a, "Riboflavin metabolism": lone_b}
    merged = merge_small(groups, {}, min_size=6, max_size=60)
    home = next(name for name, rs in merged.items() if any(r.id == "F1" for r in rs))
    ids = {r.id for r in merged[home]}
    assert "P0" in ids, (
        "the purine fragment shares chemistry with purine metabolism and must "
        "merge into it")
    # The old pool took the fragment *and* the thiamine and riboflavin steps,
    # and the pool -- sharing the fragment's chemistry -- then merged into
    # purine metabolism, putting vitamins on the purine map.
    assert not ids & {"A1", "B1"}, (
        "steps that share no chemistry with purine metabolism ended up on its map")


def test_page_titles_name_their_pathways_and_are_distinct():
    from src.layout.naming import title_pages
    titles = title_pages("Lipid metabolism", [
        [("Glycerophospholipid Metabolism: cardiolipin", 60),
         ("Membrane Lipid Metabolism", 40)],
        [("Glycerophospholipid Metabolism: phosphatidylserine", 100)],
        [("Inorganic Ion Transport and Metabolism", 30)],
    ])
    assert len(set(titles)) == len(titles)
    assert not any(re.search(r"\(\d+\)$", t) for t in titles), titles
    assert "cardiolipin" in titles[0] and "Membrane Lipid" in titles[0]
    assert "phosphatidylserine" in titles[1]
    # "Metabolism" is dropped as redundant, but not where it is half the name.
    assert titles[2].endswith("Inorganic Ion Transport and Metabolism")


def test_split_superclasses_get_titles_not_numbers(core_model, core_clusters):
    import layout_v2
    pages = layout_v2.merge_by_function(core_clusters, max_size=15)
    assert len(pages) > len({n.split(":")[0] for n in pages}), \
        "max_size=15 should split at least one superclass into pages"
    for name in pages:
        assert not re.search(r" \(\d+\)$", name), name
        assert not name.startswith("Other metabolism"), name


def test_map_file_names_stay_short_when_titles_are_long():
    """Page titles became sentences and were used whole as file names, which
    pushed paths past Windows' 260-character limit: maps failed to save with
    "[Errno 2] No such file or directory"."""
    import layout_v2
    long_a = ("Amino acid metabolism: Arginine and Proline; Tyrosine, Tryptophan, "
              "and Phenylalanine; tRNA Charging; +1 more")
    long_b = long_a.replace("+1 more", "+2 more")
    a, b = layout_v2.file_stem(long_a), layout_v2.file_stem(long_b)
    assert len(a) <= layout_v2.MAX_STEM and len(b) <= layout_v2.MAX_STEM
    assert a != b, "truncation must not make two titles share a file"
    assert layout_v2.file_stem("Glycolysis/Gluconeogenesis") == "Glycolysis_Gluconeogenesis"


# --------------------------------------------------------------------------
# The species canvas. Its claims are that pathways never overlap -- by
# construction, not by luck -- and that it is denser than tiling rectangles.
# --------------------------------------------------------------------------

def test_packer_positions_match_a_brute_force_check():
    """Feasibility and contact come from FFT correlation over a window; one
    off-by-one in the padding that lines the kernels up would put shapes on
    top of each other while every metric still looked fine."""
    import numpy as np
    from src.layout.canvas import Packer, _dilate
    rng = np.random.default_rng(1)
    for _ in range(40):
        packer = Packer(gap=int(rng.integers(1, 3)), outer_gap=int(rng.integers(3, 5)))
        packer.done = rng.random((40, 44)) < 0.06
        packer.group = rng.random((40, 44)) < 0.06
        packer.fixed = np.zeros_like(packer.done)
        packer.box = (8, 8, 32, 36)
        mask = rng.random((int(rng.integers(2, 6)), int(rng.integers(2, 6)))) < 0.7
        mask[0, 0] = True
        feasible, _, rows, cols, _ = packer.candidates(mask, window=(10, 25, 10, 30))

        def clear(grid, d, r, c):
            kernel = _dilate(mask, d)
            r0, c0 = r - d, c - d
            for i, j in zip(*np.nonzero(kernel)):
                y, x = r0 + i, c0 + j
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x]:
                    return False
            return True

        g_out = max(packer.outer_gap, packer.gap)
        for i in range(feasible.shape[0]):
            for j in range(feasible.shape[1]):
                r, c = int(rows[i, 0]), int(cols[0, j])
                expected = clear(packer.group, packer.gap, r, c) and clear(packer.done, g_out, r, c)
                assert bool(feasible[i, j]) == expected, (r, c)


def test_enclosed_holes_are_never_offered_to_a_neighbour():
    """A TCA ring encloses empty space; nothing else may be packed inside it."""
    import numpy as np
    from src.layout.canvas import _fill_holes
    ring = np.zeros((7, 7), dtype=bool)
    ring[1, 1:6] = ring[5, 1:6] = ring[1:6, 1] = ring[1:6, 5] = True
    filled = _fill_holes(ring)
    assert filled[2:5, 2:5].all()
    assert not filled[0].any() and not filled[:, 0].any()


def _core_canvas(core_model, core_clusters):
    from src.layout import taxonomy
    from src.layout.canvas import compose_canvas
    from src.layout.compose import build_meta_graph
    from src.layout.compound import compute_cofactor_scores
    from src.layout.engine import layout_reactions
    tiles = []
    for name, reactions in sorted(core_clusters.items()):
        result = layout_reactions(core_model, reactions, name)
        if result is not None:
            tiles.append((name, result.escher_map))
    labels = {n: taxonomy.classify(n, core_clusters[n]) for n, _ in tiles}
    meta = build_meta_graph(core_clusters, compute_cofactor_scores(core_model))
    return tiles, compose_canvas(tiles, labels, meta, "e_coli_core")


def test_species_canvas_draws_everything_and_overlaps_nothing(core_model, core_clusters):
    from src.layout.canvas import cross_overlaps
    tiles, canvas = _core_canvas(core_model, core_clusters)
    body = canvas[1]
    assert len(body["nodes"]) == sum(len(t[1]["nodes"]) for _, t in tiles)
    assert len(body["reactions"]) == sum(len(t[1]["reactions"]) for _, t in tiles)
    assert cross_overlaps(canvas) == 0


def test_species_canvas_is_denser_than_tiling_rectangles(core_model, core_clusters):
    """The point of planning on real shapes: the rectangle-packed poster was
    more than half empty."""
    from src.layout import metrics
    from src.layout.compose import build_meta_graph, compose
    from src.layout.compound import compute_cofactor_scores
    tiles, canvas = _core_canvas(core_model, core_clusters)
    meta = build_meta_graph(core_clusters, compute_cofactor_scores(core_model))
    tiled = compose(tiles, meta, "e_coli_core")
    assert (metrics.blank_space(canvas)["blank_share"]
            < 0.75 * metrics.blank_space(tiled)["blank_share"])


# --------------------------------------------------------------------------
# Reactions drawn on top of each other, and text drawn on top of either.
# --------------------------------------------------------------------------

def _core_maps(core_model, core_clusters):
    from src.layout.engine import layout_reactions
    maps = []
    for name, reactions in sorted(core_clusters.items()):
        result = layout_reactions(core_model, reactions, name)
        if result is not None:
            maps.append(result.escher_map)
    return maps


def _collinear_pairs(escher_map, min_shared=12.0):
    """(reaction, reaction) pairs whose straight segments share a stretch of line."""
    body = escher_map[1]
    nodes = body["nodes"]
    lines = {}
    for key, reaction in body["reactions"].items():
        for segment in reaction["segments"].values():
            if segment.get("b1") or segment.get("b2"):
                continue
            a, b = nodes[segment["from_node_id"]], nodes[segment["to_node_id"]]
            if abs(a["y"] - b["y"]) < 1 and abs(a["x"] - b["x"]) >= 1:
                lines.setdefault(("h", round(a["y"] / 4)), []).append(
                    (min(a["x"], b["x"]), max(a["x"], b["x"]), key))
            elif abs(a["x"] - b["x"]) < 1 and abs(a["y"] - b["y"]) >= 1:
                lines.setdefault(("v", round(a["x"] / 4)), []).append(
                    (min(a["y"], b["y"]), max(a["y"], b["y"]), key))
    pairs = set()
    for spans in lines.values():
        spans.sort()
        for i, (lo, hi, key) in enumerate(spans):
            for lo2, hi2, key2 in spans[i + 1:]:
                if lo2 >= hi:
                    break
                if key != key2 and min(hi, hi2) - max(lo, lo2) > min_shared:
                    pairs.add(tuple(sorted((key, key2))))
    return pairs


def _metabolites_of(escher_map):
    body = escher_map[1]
    out = {}
    for key, reaction in body["reactions"].items():
        for segment in reaction["segments"].values():
            for end in (segment["from_node_id"], segment["to_node_id"]):
                node = body["nodes"][end]
                if node["node_type"] == "metabolite":
                    out.setdefault(key, set()).add(end)
    return out


def test_reactions_joining_one_pair_are_not_drawn_as_one_line(core_model, core_clusters):
    """NADH16 and CYTBD both reduce ubiquinone. Drawn on one shared axis they
    read Q8 -> NADH16 -> CYTBD -> Q8H2, a two-step chain through an
    intermediate that does not exist."""
    for escher_map in _core_maps(core_model, core_clusters):
        nodes = escher_map[1]["nodes"]
        primaries = {k: frozenset(m for m in ms if nodes[m].get("node_is_primary", True))
                     for k, ms in _metabolites_of(escher_map).items()}
        for a, b in _collinear_pairs(escher_map):
            assert not (primaries.get(a) and primaries.get(a) == primaries.get(b)), (a, b)


def test_unrelated_reactions_are_not_drawn_on_one_line(core_model, core_clusters):
    for escher_map in _core_maps(core_model, core_clusters):
        touches = _metabolites_of(escher_map)
        for a, b in _collinear_pairs(escher_map):
            assert touches.get(a, set()) & touches.get(b, set()), (
                f"{a} and {b} share no metabolite but are drawn on one line")


def test_no_text_on_a_node_an_edge_or_other_text(core_model, core_clusters):
    """Label-on-edge used to be measured against straight segments only, so
    809 labels on iML1515 sitting on ring arcs and cofactor curves were never
    counted; and labels that could not fit were hidden, which Escher ignores."""
    from src.layout import metrics
    for escher_map in _core_maps(core_model, core_clusters):
        values = metrics.score(escher_map)
        assert values["label_overlaps"] == 0
        assert values["label_on_node"] == 0
        assert values["label_on_edge"] == 0
        assert not any(n.get("label_hidden") for n in escher_map[1]["nodes"].values())


def test_nudging_gives_unrelated_reactions_their_own_track():
    """Two reactions with no metabolite in common, routed down one channel,
    must end up on two lines -- and the legs at each end must stay straight."""
    from src.layout import render
    builder = render.EscherBuilder()
    a = builder.add_metabolite("a_c", "a", 0.0, -200.0)
    b = builder.add_metabolite("b_c", "b", 400.0, 200.0)
    c = builder.add_metabolite("c_c", "c", 100.0, -300.0)
    d = builder.add_metabolite("d_c", "d", 500.0, 300.0)
    m1, m2 = builder.add_marker("multimarker", 0.0, 0.0), builder.add_marker("multimarker", 400.0, 0.0)
    n1, n2 = builder.add_marker("multimarker", 100.0, 0.0), builder.add_marker("multimarker", 500.0, 0.0)

    def chain(*ids):
        return {f"s{i}": {"from_node_id": p, "to_node_id": q, "b1": None, "b2": None}
                for i, (p, q) in enumerate(zip(ids, ids[1:]))}

    builder.reactions["R1"] = {"segments": chain(a, m1, m2, b)}
    builder.reactions["R2"] = {"segments": chain(c, n1, n2, d)}
    render._nudge_overlaps(builder)
    nodes = builder.nodes
    assert abs(nodes[m1]["y"] - nodes[n1]["y"]) >= render.NUDGE * 0.9
    assert nodes[m1]["y"] == nodes[m2]["y"] and nodes[n1]["y"] == nodes[n2]["y"]
    for reaction in builder.reactions.values():
        for segment in reaction["segments"].values():
            p, q = nodes[segment["from_node_id"]], nodes[segment["to_node_id"]]
            if nodes[segment["from_node_id"]]["node_type"] == "metabolite" or \
               nodes[segment["to_node_id"]]["node_type"] == "metabolite":
                continue          # the short stub into a fixed metabolite may slant
            assert abs(p["x"] - q["x"]) < 1e-6 or abs(p["y"] - q["y"]) < 1e-6


# --------------------------------------------------------------------------
# Generality. The pipeline was written against BiGG and read compounds and
# currency off the id: strip `_c` and look the rest up in a list. Human-GEM
# spells ATP `MAM01371c` and Yeast-GEM numbers it `s_0434` per compartment,
# so neither model had a single cofactor recognised -- ATP and NADH competed
# for the backbone and every Yeast-GEM reaction looked like a transport step.
# --------------------------------------------------------------------------

_COMPARTMENT_WORDS = {"c": "cytoplasm", "e": "extracellular", "p": "periplasm"}


def _respelled(core_model, style):
    """e_coli_core with every id rewritten the way another model family writes it.

    'human': MAM00001c / MAR00001, annotations kept (Human-GEM).
    'yeast': s_0001 / r_0001, compartment only in the name, no annotations
             at all, so currency has to be recognised from names (Yeast-GEM).
    """
    model = core_model.copy()
    back = {}
    for n, met in enumerate(sorted(model.metabolites, key=lambda m: m.id)):
        old = met.id
        if style == "human":
            met.id = f"MAM{n:05d}{met.compartment}"
        else:
            met.id = f"s_{n:04d}"
            met.annotation = {}
            met.name = f"{met.name} [{_COMPARTMENT_WORDS.get(met.compartment, met.compartment)}]"
        back[met.id] = old
    reactions = {}
    for n, rxn in enumerate(sorted(model.reactions, key=lambda r: r.id)):
        old = rxn.id
        rxn.id = f"MAR{n:05d}" if style == "human" else f"r_{n:04d}"
        if style == "yeast":
            rxn.annotation = {}
        reactions[rxn.id] = old
    model.repair()
    return model, back, reactions


@pytest.mark.parametrize("style", ["human", "yeast"])
def test_main_pairs_do_not_depend_on_how_ids_are_spelled(core_model, style):
    expected = _main_pairs(core_model, list(core_model.reactions))
    model, back, reactions = _respelled(core_model, style)
    got = _main_pairs(model, list(model.reactions))
    translated = {reactions[rid]: (back.get(u, u), back.get(v, v)) for rid, (u, v) in got.items()}
    differ = sorted(rid for rid in expected if translated.get(rid) != expected[rid])
    assert not differ, (
        f"{len(differ)} reactions change their main pair when ids are spelled "
        f"the {style} way, e.g. {[(r, expected[r], translated.get(r)) for r in differ[:3]]}"
    )


def test_currency_is_recognised_without_bigg_ids(core_model):
    from src.layout import identity
    for style in ("human", "yeast"):
        model, back, _ = _respelled(core_model, style)
        identity.register(model)
        for met in model.metabolites:
            base = back[met.id].rsplit("_", 1)[0]
            if base in ("atp", "adp", "nad", "nadh", "nadph", "h2o", "h", "co2", "pi", "coa"):
                assert identity.currency(met) == base, (style, back[met.id], met.name)
    identity.register(core_model)


def test_opaque_ids_are_labelled_by_name(core_model, core_clusters):
    from src.layout.engine import layout_reactions
    model, back, reactions = _respelled(core_model, "human")
    forward = {old: new for new, old in reactions.items()}
    name = "Citric Acid Cycle" if "Citric Acid Cycle" in core_clusters else next(iter(core_clusters))
    subset = [model.reactions.get_by_id(forward[r.id]) for r in core_clusters[name]]
    escher_map = layout_reactions(model, subset, name).escher_map
    labels = [n["label_text"] for n in escher_map[1]["nodes"].values()
              if n["node_type"] == "metabolite"]
    assert labels and not [t for t in labels if re.match(r"MAM\d", t)], labels[:10]


# --------------------------------------------------------------------------
# DIY maps. A reader's own cut of a model is held to the published maps'
# rules; a two-pathway map once came out half blank because its title, set at
# full size, was twice as wide as the drawing under it.
# --------------------------------------------------------------------------

def test_diy_map_meets_the_published_criteria(core_model, tmp_path):
    import diy_map
    code = diy_map.main(["--model", "e_coli_core", "--pathway", "Glycolysis",
                         "--pathway", "Pentose*", "--out", str(tmp_path),
                         "--strict", "--quiet", "--name", "Glycolysis and PPP"])
    assert code == 0
    folder = tmp_path / "e_coli_core"
    escher_map = json.loads((folder / "Glycolysis_and_PPP.json").read_text(encoding="utf-8"))
    assert (folder / "Glycolysis_and_PPP.svg").exists()
    chosen = json.loads((folder / "Glycolysis_and_PPP.selection.json").read_text(encoding="utf-8"))
    drawn = {r["bigg_id"] for r in escher_map[1]["reactions"].values()}
    assert set(chosen["reactions"]) == drawn
    assert {"PGI", "PFK", "G6PDH2r", "TKT1"} <= drawn

    from src.layout import metrics
    assert metrics.blank_space(escher_map)["blank_share"] < 0.25, "title widened the canvas"


def test_connect_joins_two_islands(core_model):
    import argparse
    import diy_map
    catalogue = diy_map.Catalogue(core_model)
    args = argparse.Namespace(pathway=None, superclass=None, reaction=["PGI,CS"],
                              from_file=None, metabolite=None, radius=1, search=None,
                              exclude=None, no_boundary=False, connect=4)
    chosen, _ = diy_map.select(catalogue, args)
    ids = {r.id for r in chosen}
    graph = diy_map._reaction_graph(catalogue, chosen)
    import networkx as nx
    assert {"PGI", "CS"} <= ids and nx.is_connected(graph), sorted(ids)


def test_canvas_records_which_pathway_each_reaction_belongs_to(core_model, core_clusters):
    """The viewer selects a whole pathway or region from the map header.

    Without it, the only record of membership was the node-id prefix, which
    says nothing about reactions and nothing about regions.
    """
    _, canvas = _core_canvas(core_model, core_clusters)
    header, body = canvas
    listed = [key for entry in header["pathways"] for key in entry["reactions"]]
    assert sorted(listed) == sorted(body["reactions"]), "every reaction in exactly one pathway"
    for entry in header["pathways"]:
        assert entry["caption"] in body["text_labels"]
        assert entry["region"]
    for label_id in header["regions"].values():
        assert label_id in body["text_labels"]


def test_publishing_v2_leaves_v1_in_place(tmp_path):
    """v1 stays published at the root while v2 is published under v2/."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("prepare_repo", os.path.join("scripts", "prepare_repo.py"))
    prepare_repo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_repo)
    spec = importlib.util.spec_from_file_location("build_map_index", os.path.join("scripts", "build_map_index.py"))
    build_map_index = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_map_index)

    repo = tmp_path / "repo"
    (repo / "e_coli_core").mkdir(parents=True)
    v1_map = repo / "e_coli_core" / "Old_page.json"
    v1_map.write_text(json.dumps([{"map_name": "Old page"}, {"reactions": {"A": {}}, "nodes": {}}]))
    source = tmp_path / "v2_maps" / "e_coli_core"
    source.mkdir(parents=True)
    (source / "New_page.json").write_text(json.dumps([{"map_name": "New page"}, {"reactions": {"B": {}}, "nodes": {}}]))
    (source / "e_coli_core_Canvas.json").write_text(json.dumps([{"map_name": "Canvas"}, {"reactions": {"B": {}, "C": {}}, "nodes": {}}]))

    prepare_repo.main(["--source", str(tmp_path / "v2_maps"), "--repo", str(repo / "v2")])
    build_map_index.main(["--root", str(repo)])
    build_map_index.main(["--root", str(repo / "v2")])

    assert v1_map.exists(), "a v2 sync removed a v1 map"
    v1_index = json.loads((repo / "map_index.json").read_text())
    assert [m["id"] for m in v1_index["models"]] == ["e_coli_core"], "v2/ listed as a model"
    v2_maps = json.loads((repo / "v2" / "e_coli_core" / "model_index.json").read_text())["maps"]
    assert v2_maps[0]["canvas"] and v2_maps[0]["file"] == "e_coli_core_Canvas.json"
    assert json.loads((repo / "e_coli_core" / "model_index.json").read_text())["maps"][0]["file"] == "Old_page.json"


def test_a_small_model_canvas_is_not_mostly_white(core_model, core_clusters):
    """e_coli_core's canvas once left a quarter of the page empty.

    Greedy growth from central carbon makes a round cluster, and on a
    rectangular page with a handful of large pathways the corners stay empty
    -- the transport region sat far below the title with nothing above it.
    Small models are now packed several ways, including into a frame sized
    up front, and the tightest packing that keeps the regions together wins.
    """
    from src.layout import metrics, taxonomy
    from src.layout.canvas import organisation
    from src.layout.compose import build_meta_graph
    from src.layout.compound import compute_cofactor_scores
    tiles, canvas = _core_canvas(core_model, core_clusters)
    blank = metrics.blank_space(canvas)
    assert blank["blank_share"] < 0.22, blank
    assert blank["largest_blank_rect_share"] < 0.06, blank
    labels = {n: taxonomy.classify(n, core_clusters[n]) for n, _ in tiles}
    meta = build_meta_graph(core_clusters, compute_cofactor_scores(core_model))
    order = organisation(canvas, labels, meta, [n for n, _ in tiles])
    # Three regions -- carbohydrate, energy, transport -- and the energy
    # region is one six-reaction pathway, whose every neighbour is another
    # region's: cohesion cannot approach 1 here however well it is drawn.
    assert order["region_cohesion"] > 0.65, order


def test_energy_metabolism_is_not_filed_as_an_exchange():
    """NADH dehydrogenase, cytochrome oxidase and ATP synthase are pure
    currency on both sides, and the boundary test -- no primary compound on
    *either* side -- filed all of oxidative phosphorylation under "Biomass and
    exchange". A boundary step has currency on exactly one side."""
    import cobra
    from src.layout.decompose import is_boundary_reaction
    model = cobra.io.load_json_model(os.path.join("data", "bigg", "models", "e_coli_core.json"))
    for rid in ("NADH16", "CYTBD", "ATPS4r", "THD2", "PDH", "O2t"):
        assert not is_boundary_reaction(model.reactions.get_by_id(rid), {}), rid
    for rid in ("BIOMASS_Ecoli_core_w_GAM", "ATPM", "EX_glc__D_e"):
        assert is_boundary_reaction(model.reactions.get_by_id(rid), {}), rid


def test_a_small_pathway_merges_into_metabolism_not_transport(core_model):
    """e_coli_core's glutamate metabolism shares glutamate with its
    transporters more than with anything else, merged into them, and was
    drawn and captioned as "Transport, Extracellular"."""
    from src.layout.compound import compute_cofactor_scores
    from src.layout.decompose import clusters
    groups = clusters(core_model, compute_cofactor_scores(core_model))
    home = {r.id: name for name, rs in groups.items() for r in rs}
    for rid in ("GLUDy", "GLNS", "ME1", "PPC"):
        assert "Transport" not in home[rid] and "exchange" not in home[rid], (rid, home[rid])
    assert "Oxidative Phosphorylation" in groups


def test_a_ring_middle_is_not_counted_as_blank():
    """The inside of a drawn ring is the ring, not a hole in the page."""
    import math
    from src.layout import metrics

    def ring_map(tagged):
        nodes = {}
        for i in range(12):
            a = 2 * math.pi * i / 12
            x, y = 2000 * math.cos(a), 2000 * math.sin(a)
            nodes[str(i)] = {"node_type": "metabolite", "x": x, "y": y,
                             "label_x": x + 40, "label_y": y, "bigg_id": f"m{i}",
                             "name": f"m{i}", **({"ring": "ring0"} if tagged else {})}
        return [{}, {"nodes": nodes, "reactions": {}, "text_labels": {}}]

    plain = metrics.blank_space(ring_map(False))["blank_share"]
    ring = metrics.blank_space(ring_map(True))["blank_share"]
    assert plain > 0.3 and ring < plain / 2, (plain, ring)


# --------------------------------------------------------------------------
# The TCA cycle, end to end. Shipped wrong three ways: MitoMammal's fumarase
# and malate dehydrogenase were filed under the malate-aspartate shuttle,
# because ring closure met the shuttle's ring before the TCA cycle; its
# succinate/malate carrier was paired malate -> succinate on the whole model,
# which is a chord of the ring, so ring closure pulled a transporter into the
# TCA map; and antiporters across the corpus (yeast's citrate/isocitrate
# carrier, the ornithine/citrulline carrier, the ADP/ATP translocase) were
# drawn as conversions between the two compounds they swap.
# --------------------------------------------------------------------------

def test_a_carrier_is_drawn_as_the_compound_it_carries(core_model):
    """An antiporter converts nothing, however much skeleton the two swapped
    compounds share and whatever cycle pairing them would close."""
    import cobra
    from src.layout import identity
    from src.layout.compound import build_compound_graph

    model = core_model.copy()
    carrier = cobra.Reaction("SUCMALt_test", lower_bound=-1000, upper_bound=1000)
    m = model.metabolites
    carrier.add_metabolites({m.succ_c: -1, m.mal__L_e: -1, m.succ_e: 1, m.mal__L_c: 1})
    model.add_reactions([carrier])
    identity.register(model)
    rec = build_compound_graph(model, list(model.reactions)).reactions["SUCMALt_test"]
    assert identity.species(rec.main_sub) == identity.species(rec.main_prod), (
        rec.main_sub, rec.main_prod)


def _tca_report(path, **kwargs):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "check_tca", os.path.join("scripts", "check_tca.py"))
    check_tca = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_tca)
    report = check_tca.check_model(path, **kwargs)
    failures = [f"{name}: {detail}" for name, ok, detail in report.rows if ok is False]
    checked = {name for name, ok, _ in report.rows if ok is not None}
    return failures, checked


def test_tca_cycle_is_drawn_whole_and_correct(core_model):
    """Ring in textbook order, one node per member, round, every step on its
    own arc, no false chord, one cluster, clean text: scripts/check_tca.py."""
    failures, checked = _tca_report(os.path.join("data", "bigg", "models", "e_coli_core.json"))
    assert {"cluster", "ring", "closed", "round", "arcs", "chords", "middle", "clean"} <= checked
    assert not failures, failures


MITOMAMMAL = os.path.join("data", "other_models", "MitoMammal.json")


@pytest.mark.skipif(not os.path.exists(MITOMAMMAL),
                    reason="MitoMammal.json not present (data/other_models/)")
def test_mitomammal_tca_cycle_is_one_map():
    """FUMm and MDHm on the TCA map, with complex II closing the ring, and
    no transporter pulled in as a chord."""
    failures, checked = _tca_report(MITOMAMMAL)
    assert {"cluster", "ring", "closed", "round", "arcs", "chords", "middle"} <= checked
    assert not failures, failures
