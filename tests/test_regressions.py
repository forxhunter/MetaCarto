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
    from src.layout.motifs import find_rings

    for name in sorted(core_clusters):
        result = layout_reactions(core_model, core_clusters[name], name,
                                  use_fba=True, render=False)
        if result is None:
            continue
        for ring in find_rings(result.cgraph.D):
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
