"""Shared fixtures.

e_coli_core is the fixture for everything that does not specifically need a
genome-scale model: 95 reactions, loads in well under a second, and contains
the cases these tests care about -- glycolysis as a backbone, the TCA cycle as
a real cycle, and PDH, the reaction whose main pair this project has got wrong
twice.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join("data", "bigg", "models")


@pytest.fixture(scope="session")
def core_model():
    cobra = pytest.importorskip("cobra")
    path = os.path.join(MODEL_DIR, "e_coli_core.json")
    if not os.path.exists(path):
        pytest.skip("e_coli_core.json not present; run scripts/fetch_bigg.py")
    return cobra.io.load_json_model(path)


@pytest.fixture(scope="session")
def core_clusters(core_model):
    from src.layout.compound import compute_cofactor_scores
    from src.layout.decompose import clusters
    return clusters(core_model, compute_cofactor_scores(core_model))


@pytest.fixture(scope="session")
def core_map(core_model, core_clusters):
    """One rendered map, for the schema and geometry tests.

    The *largest* cluster, not the first alphabetically: that one is
    "Acetaldehyde exchange", which is all EX_ reactions, and an exchange
    reaction has only one side -- so the map contains no produced metabolite
    and a test for signed stoichiometry fails on a map that is perfectly
    correct.
    """
    from src.layout.engine import layout_reactions

    def two_sided(reactions):
        return sum(1 for r in reactions
                   if any(c < 0 for c in r.metabolites.values())
                   and any(c > 0 for c in r.metabolites.values()))

    # Most reactions that actually convert something. Picking by raw size gives
    # "Acetaldehyde exchange" -- 20 reactions, none of them two-sided, because
    # an exchange reaction has one side. A map of those is perfectly correct
    # and contains no produced metabolite at all.
    name = max(core_clusters, key=lambda k: two_sided(core_clusters[k]))
    result = layout_reactions(core_model, core_clusters[name], name)
    assert result is not None, "e_coli_core's first cluster produced no layout"
    return result.escher_map
