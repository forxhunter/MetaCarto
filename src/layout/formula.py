"""Chemical formula parsing, used to score conserved-moiety transfer between
a reaction's substrates and products (layout_algorithm.md S1)."""

import re

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

# Weights express "how much of a recognisable molecular skeleton is carried
# across this substrate/product pair". Carbon dominates: in curated maps the
# main pair of a reaction is essentially always the carbon-carrying pair.
_ELEMENT_WEIGHT = {"C": 1.0, "N": 0.3, "P": 0.12, "S": 0.12}


def parse_formula(formula):
    """'C6H11O9P' -> {'C': 6, 'H': 11, 'O': 9, 'P': 1}.

    Returns {} for missing/unparseable formulas. BiGG uses bare element strings
    with no brackets; 'R' groups and charge suffixes are tolerated and ignored.
    """
    if not formula:
        return {}
    counts = {}
    for element, digits in _TOKEN.findall(formula):
        counts[element] = counts.get(element, 0) + (int(digits) if digits else 1)
    return counts


def _weighted_size(counts):
    size = sum(_ELEMENT_WEIGHT.get(el, 0.0) * n for el, n in counts.items())
    if size == 0.0:
        size = 0.15 * sum(n for el, n in counts.items() if el != "H")
    return size


def moiety_score(counts_a, counts_b):
    """How much molecular skeleton a substrate/product pair plausibly shares.

    A formula-only stand-in for KEGG RPAIR 'main' classification. Two terms:

      * absolute overlap  -- a 6-carbon transfer is more backbone-like than a
        1-carbon one, so CO2 release never outranks the real product;
      * proportional similarity -- ATP -> ADP shares 10 carbons but so does
        ATP -> anything large, and only the proportional term separates a real
        skeleton rearrangement from an incidental carbon count match.

    Carbon-free chemistry (sulfur, phosphate, inorganic transport) falls back
    to generic heavy-atom overlap so those reactions still get a main pair.
    """
    if not counts_a or not counts_b:
        return 0.0

    overlap = 0.0
    for element, weight in _ELEMENT_WEIGHT.items():
        overlap += weight * min(counts_a.get(element, 0), counts_b.get(element, 0))

    if overlap == 0.0:
        shared_heavy = sum(
            min(counts_a.get(el, 0), counts_b.get(el, 0))
            for el in set(counts_a) | set(counts_b)
            if el != "H"
        )
        overlap = 0.15 * shared_heavy

    if overlap == 0.0:
        return 0.0

    similarity = overlap / max(_weighted_size(counts_a), _weighted_size(counts_b), 1e-9)
    return overlap * (0.4 + 0.6 * min(similarity, 1.0))


def carbon_count(counts):
    return counts.get("C", 0)
