"""Metabolic superclasses, used to organise the whole-model map.

A whole-model map that merely packs pathway tiles in flow order is still a pile
of tiles: a reader looking for lipid metabolism has to read every caption. Every
curated global map -- KEGG's (`templates/t4`), the metro map (`t2`) -- instead
groups pathways into a handful of regions and lets the reader navigate by
region first.

The grouping follows KEGG's BRITE top-level metabolism categories, matched by
keyword against the pathway name. Keyword matching rather than a downloaded
hierarchy because the cluster names come from three different sources (the
model's own `subsystem` string, a KEGG pathway name, or a structural label) and
only the text is common to all of them.

A name is not always enough. "Miscellaneous", "Unassigned" and a caption
derived from free text say nothing about the superclass, and filing all of
them under "Other metabolism" is what made a third of the published maps
"Other metabolism (N)". `classify` therefore falls back to the reactions
themselves: each one votes with its own subsystem, its KEGG pathways, or --
for a transport or boundary step -- its shape.
"""

import re

from . import identity

# Ordered: this is also the order regions are laid out in, and it follows how
# metabolism is normally taught and drawn -- central carbon first, then what
# feeds off it, with degradation and transport at the edges. Order also breaks
# ties: the first superclass with a matching keyword wins, so "nucleotide
# sugar" is carbohydrate before "nucleotide" can make it nucleotide.
#
# Keywords match at the start of a word, so "tca" cannot fire inside another
# word and "nucleotide" still matches "Nucleotides".
SUPERCLASSES = [
    ("Carbohydrate metabolism", (
        "carbohydrate", "glycolysis", "gluconeogen", "citrate cycle",
        "citric acid", "tca", "central metabolism", "pentose", "glucuronate",
        "fructose", "mannose", "galactose", "starch", "sucrose",
        "amino sugar", "aminosugar", "nucleotide sugar", "pyruvate metab",
        "glyoxylate", "dicarboxylate", "propanoate", "butanoate", "ascorbate",
        "aldarate", "inositol phosphate", "inositol metab", "c5-branched",
        "anaplerotic", "alternate carbon", "methylglyoxal", "propanediol",
        "sulfoquinovose",
    )),
    ("Energy metabolism", (
        "energy metab", "oxidative phosphoryl", "photosynthesis",
        "carbon fixation", "methane", "methanogenesis", "nitrogen metab",
        "nitrogen cycle", "nitrite metab", "sulfur metab", "sulfur cycle",
        "sulfite metab", "respiration", "electron transport",
        "oxidoreduction of electron",
    )),
    ("Amino acid metabolism", (
        "alanine", "aspartate", "glutamate", "glutamine", "glycine", "serine",
        "threonine", "cysteine", "methionine", "valine", "leucine",
        "isoleucine", "lysine", "arginine", "proline", "histidine", "tyrosine",
        "phenylalanine", "tryptophan", "beta-alanine", "taurine",
        "phosphonate", "selenocompound", "cyanoamino", "d-amino",
        "d-glutamine", "d-arginine", "glutathione", "urea cycle", "amino acid",
        "hydroxyphenylacetate", "hippurate", "stickland", "peptide metab",
        "selenoamino",
        # KEGG files aminoacyl-tRNA biosynthesis under Translation, outside
        # metabolism altogether. It is drawn here because it is where its
        # substrates are, and a "Translation" region of one map would be
        # stranger than tRNA charging beside amino-acid synthesis.
        "trna charging", "aminoacyl-trna", "translation", "protein production",
        "protein formation",
    )),
    ("Nucleotide metabolism", (
        "purine", "pyrimidine", "nucleotide", "nucleoside", "ribonucleoside",
        "nucleotidase", "imp biosynth",
    )),
    ("Lipid metabolism", (
        "lipid", "fatty acid", "glycerolipid", "glycerophospholipid",
        "phospholipid", "phosphoglycerolipid", "galactoglycerolipid",
        "galactolipid", "phosphatidyl", "phospholipase", "membrane metab",
        "sterol", "polyhydroxy", "phas metab",
        "sphingolipid", "ether lipid", "arachidonic", "linoleic", "linolenic",
        "linoleate", "steroid", "cholesterol", "squalene", "androgen",
        "estrogen", "bile acid", "cutin", "suberine", "wax", "eicosanoid",
        "leukotriene", "prostaglandin", "thromboxane", "triacylglycerol",
        "diacylglycerol", "dag metab", "carnitine shuttle", "r group synth",
        "membrane lipid", "beta-oxidation", "ketone body", "mycolic",
    )),
    ("Cofactor and vitamin metabolism", (
        "cofactor", "coenzyme", "prosthetic", "porphyrin", "heme",
        "chlorophyll", "thiamine", "riboflavin", "vitamin", "nicotinate",
        "nicotinamide", "nad metab", "pantothenate", "coa biosynth",
        "coa synth", "coa catab", "biotin", "lipoic", "lipoate", "folate",
        "one carbon pool", "pterin", "biopterin", "tetrahydrobiopterin",
        "methanopterin", "tetrahydromethanopterin", "tetrahydramethanopterin",
        "retinol",
        "ubiquinone", "terpenoid-quinone", "cytochrome metab", "queuosine",
    )),
    ("Glycan metabolism", (
        "glycan", "peptidoglycan", "murein", "cell envelope", "cell wall",
        "lps", "oligosaccharide",
        "o-antigen", "glycosaminoglycan", "chondroitin", "keratan", "heparan",
        "dermatan", "hyaluronan", "mucin", "blood group",
        "glycosylphosphatidyl", "teichoic", "arabinogalactan",
        "lipoarabinomannan",
    )),
    ("Terpenoid and polyketide metabolism", (
        "terpenoid", "polyketide", "carotenoid", "zeatin", "limonene",
        "brassinosteroid", "insect hormone", "geraniol", "macrolide",
        "ansamycin", "enediyne", "nonribosomal",
    )),
    ("Secondary metabolite biosynthesis", (
        "secondary metabolite", "alkaloid", "flavonoid", "flavone",
        "phenylpropanoid", "stilbenoid", "stilbene", "coumarin", "lignin",
        "betalain", "glucosinolate", "anthocyanin", "isoflavonoid",
        "novobiocin", "acarbose", "streptomycin", "penicillin", "cephalosporin",
        "clavulanic", "puromycin", "staurosporine", "tetracycline",
        "aminoglycoside", "vancomycin", "monobactam", "carbapenem", "phenazine",
        "prodigiosin", "neomycin", "siderophore", "aflatoxin", "benzoxazinoid",
        "caffeine", "antibiotic",
    )),
    ("Xenobiotic degradation", (
        "xenobiotic", "drug metabolism", "cyp metab", "degradation",
        "aromatic", "benzoate", "naphthalene", "toluene", "dioxin",
        "chloroalkane", "chlorocyclohexane", "styrene", "atrazine",
        "caprolactam", "fluorobenzoate", "aminobenzoate", "nitrotoluene",
        "ethylbenzene", "furfural", "polycyclic", "bisphenol", "xylene",
    )),
    ("Transport and exchange", (
        "transport", "tranpsort", "exchange", "extracellular", "uptake",
        "secretion", "demand", "sink", "biomass", "diffusion", "channel",
        "facilitator", "porin",
    )),
]

# Names whose first general keyword files them against KEGG BRITE. Checked
# before SUPERCLASSES. "Mannose type O-glycan biosynthesis" is a glycan, not
# mannose metabolism; "Keratan sulfate degradation" is a glycan, not a
# xenobiotic; "Phosphatidylinositol phosphate metabolism" is a lipid, not
# inositol phosphate metabolism.
PRIORITY = (
    ("phosphatidylinositol", "Lipid metabolism"),
    ("glycosphingolipid", "Glycan metabolism"),
    ("lipopolysacch", "Glycan metabolism"),
    ("o-glycan", "Glycan metabolism"),
    ("n-glycan", "Glycan metabolism"),
    ("sulfate degradation", "Glycan metabolism"),
    ("glycan degradation", "Glycan metabolism"),
    ("heme degradation", "Cofactor and vitamin metabolism"),
    ("brassinosteroid", "Terpenoid and polyketide metabolism"),
    ("limonene", "Terpenoid and polyketide metabolism"),
    ("pinene", "Terpenoid and polyketide metabolism"),
    ("steroid degradation", "Xenobiotic degradation"),
    ("degradation of flavonoids", "Secondary metabolite biosynthesis"),
    ("drug metabolism", "Xenobiotic degradation"),
    ("cytochrome p450", "Xenobiotic degradation"),
    ("aromatic amino", "Amino acid metabolism"),
)

OTHER = "Other metabolism"
UNASSIGNED = "Unassigned clusters"
LABELS = tuple(label for label, _ in SUPERCLASSES) + (OTHER, UNASSIGNED)

# Placeholder captions: what decompose hands out when it has nothing better,
# and the catch-alls reconstructions use for the same purpose. Anchored, so a
# real pathway that happens to begin with the word -- KEGG's "Other glycan
# degradation", "Other carbon fixation pathways" -- is still a pathway.
_STRUCTURAL = re.compile(
    r"^(?:unannotated|others?|uncategorized|unassigned|miscellaneous)"
    r"(?: \d+)?(?: \(\d+\))?$|^cluster_", re.IGNORECASE)


def is_structural(name):
    """True when a cluster name is a placeholder rather than biology."""
    return bool(_STRUCTURAL.match(str(name).strip()))


def _pattern(keywords):
    return re.compile(r"(?<![a-z0-9])(?:%s)" % "|".join(
        re.escape(k) for k in sorted(keywords, key=len, reverse=True)))


_PRIORITY = [(_pattern([k]), label) for k, label in PRIORITY]
_MATCHERS = [(_pattern(keywords), label) for label, keywords in SUPERCLASSES]


def _own_label(name):
    """The superclass a name already is, e.g. a page "Lipid metabolism (2)"."""
    lowered = name.lower()
    for label in LABELS:
        low = label.lower()
        if lowered == low or lowered.startswith((low + " (", low + ":")):
            return label
    return None


def superclass(cluster_name):
    """Which metabolic region a cluster belongs to, judged by its name alone.

    Structurally derived clusters have no biological name to match on, so they
    are kept in their own region rather than being guessed into one -- a wrong
    region is worse than an honest "unassigned". `classify` can do better when
    the reactions are available.
    """
    name = str(cluster_name).strip()
    own = _own_label(name)
    if own:
        return own
    # "Extracellular exchange: amino acids" is an exchange cluster that carries
    # amino acids. The qualifier says what is in a piece, not what it is.
    name = name.partition(": ")[0]
    if is_structural(name):
        return UNASSIGNED

    lowered = name.lower().replace("_", " ")
    for pattern, label in _PRIORITY:
        if pattern.search(lowered):
            return label
    for pattern, label in _MATCHERS:
        if pattern.search(lowered):
            return label
    return OTHER


def _is_boundary(reaction):
    """Exchange, demand or sink: one side only."""
    coefficients = list(reaction.metabolites.values())
    return bool(coefficients) and (all(c < 0 for c in coefficients)
                                   or all(c > 0 for c in coefficients))


def _is_transport(reaction):
    """Moves a compound between compartments: same base id on both sides."""
    sides = {}
    for metabolite, coefficient in reaction.metabolites.items():
        base = identity.species(metabolite)
        sides.setdefault(base, set()).add(coefficient > 0)
    return any(len(s) == 2 for s in sides.values())


def _kegg_ids(reaction):
    annotation = getattr(reaction, "annotation", None) or {}
    for key in ("kegg.reaction", "ec-code"):
        value = annotation.get(key)
        if value:
            yield key, (value if isinstance(value, list) else [value])


def reaction_votes(reaction, kegg_mapping=None):
    """{superclass: weight} for one reaction, from the best evidence it has.

    Evidence is taken in order of how specific it is and the first that
    classifies wins: the reaction's own subsystem, then its KEGG pathways
    (each pathway an equal share), then its shape. A reaction with none of
    these abstains rather than voting "Other".
    """
    subsystem = str(getattr(reaction, "subsystem", "") or "").strip()
    if subsystem:
        label = superclass(subsystem)
        if label not in (OTHER, UNASSIGNED):
            return {label: 1.0}

    if kegg_mapping:
        for _key, identifiers in _kegg_ids(reaction):
            labels = []
            for identifier in identifiers:
                for pathway in kegg_mapping.get(identifier, ()):
                    label = superclass(pathway)
                    if label not in (OTHER, UNASSIGNED):
                        labels.append(label)
            if labels:
                share = 1.0 / len(labels)
                votes = {}
                for label in labels:
                    votes[label] = votes.get(label, 0.0) + share
                return votes

    if _is_boundary(reaction) or _is_transport(reaction):
        return {"Transport and exchange": 1.0}
    return {}


# How much of a cluster has to agree before its reactions overrule an
# uninformative name. A plurality of the votes cast, and enough of the cluster
# voting that the plurality is not three reactions out of sixty.
MIN_SHARE = 0.4
MIN_COVERAGE = 0.25


def classify(cluster_name, reactions=(), kegg_mapping=None):
    """Superclass of a cluster, by its name when that says, else its content."""
    label = superclass(cluster_name)
    if label not in (OTHER, UNASSIGNED) or not reactions:
        return label

    totals, voted = {}, 0
    for reaction in reactions:
        votes = reaction_votes(reaction, kegg_mapping)
        if votes:
            voted += 1
        for vote_label, weight in votes.items():
            totals[vote_label] = totals.get(vote_label, 0.0) + weight
    if not totals or voted < MIN_COVERAGE * len(reactions):
        return label

    # Ties go to canonical order, so the result does not depend on dict order.
    best = max(totals, key=lambda l: (totals[l], -order_index(l)))
    if totals[best] < MIN_SHARE * voted:
        return label
    return best


# What a transport page carries, by the superclass its cargo belongs to.
INORGANIC = "Inorganic ions"
CARGO_NOUNS = {
    "Carbohydrate metabolism": "sugars and organic acids",
    "Energy metabolism": "respiratory substrates",
    "Amino acid metabolism": "amino acids and peptides",
    "Nucleotide metabolism": "nucleotides and nucleosides",
    "Lipid metabolism": "lipids",
    "Cofactor and vitamin metabolism": "cofactors and vitamins",
    "Glycan metabolism": "glycans and cell-wall precursors",
    "Terpenoid and polyketide metabolism": "terpenoids and polyketides",
    "Secondary metabolite biosynthesis": "secondary metabolites",
    "Xenobiotic degradation": "drugs and xenobiotics",
    INORGANIC: "inorganic ions",
}

_ORGANIC = re.compile(r"C(?![a-z])")      # carbon, not Cl, Ca, Co, Cu ...
COMPOUND_MAJORITY = 0.5


def _base(metabolite_id):
    return identity.species(metabolite_id)


def compound_classes(reactions, kegg_mapping=None):
    """{compound base id: superclass}, read off where each compound is used.

    A transport or exchange step votes "transport" and nothing else, which is
    true and useless for telling one from the next: what distinguishes them is
    the cargo. The cargo's own class is in the rest of the model -- glutamine
    is an amino acid because the reactions that make and use it are amino-acid
    metabolism -- so this tallies, per compound, the votes of every
    non-transport reaction it takes part in. A compound with no carbon is an
    inorganic ion whatever it takes part in.

    Also returns each compound's degree, which `cargo` uses to tell the cargo
    from the proton or ATP that moves it.
    """
    tallies, degree, inorganic = {}, {}, set()
    for reaction in reactions:
        bases = set()
        for metabolite in reaction.metabolites:
            base = _base(metabolite.id)
            bases.add(base)
            formula = getattr(metabolite, "formula", "") or ""
            if formula and not _ORGANIC.search(formula):
                inorganic.add(base)
        for base in bases:
            degree[base] = degree.get(base, 0) + 1
        if _is_boundary(reaction) or _is_transport(reaction):
            continue
        votes = reaction_votes(reaction, kegg_mapping)
        for base in bases:
            tally = tallies.setdefault(base, {})
            for label, weight in votes.items():
                tally[label] = tally.get(label, 0.0) + weight

    classes = {base: INORGANIC for base in inorganic}
    for base, tally in tallies.items():
        if base in classes or not tally:
            continue
        best = max(tally, key=lambda l: (tally[l], -order_index(l)))
        # A hub such as succinate takes part in amino-acid, carbohydrate and
        # energy metabolism alike, and a plurality among those calls it an
        # amino acid. A compound its reactions do not mostly agree on is left
        # unclassified rather than captioned wrongly.
        if tally[best] >= COMPOUND_MAJORITY * sum(tally.values()):
            classes[base] = best
    return classes, degree


def cargo(reaction, classes, degree):
    """(class, compound) a transport or exchange step carries, else None.

    The cargo is the moved compound with the fewest reactions: a proton
    symport moves both glutamine and a proton, and the proton is in hundreds.
    """
    if _is_boundary(reaction):
        moved = {_base(m.id) for m in reaction.metabolites}
    elif _is_transport(reaction):
        sides = {}
        for metabolite, coefficient in reaction.metabolites.items():
            sides.setdefault(_base(metabolite.id), set()).add(coefficient > 0)
        moved = {base for base, s in sides.items() if len(s) == 2}
    else:
        return None
    if not moved:
        return None
    compound = min(moved, key=lambda b: (degree.get(b, 0), b))
    return classes.get(compound), compound


def theme(reaction, classes, degree, kegg_mapping=None):
    """The superclass a single reaction is *about*, for ordering and naming.

    For a metabolic step that is its own vote; for transport and exchange it
    is the class of what is carried.
    """
    carried = cargo(reaction, classes, degree)
    if carried is not None:
        return carried[0] or ""
    votes = reaction_votes(reaction, kegg_mapping)
    if not votes:
        return ""
    return max(votes, key=lambda l: (votes[l], -order_index(l)))


def order_index(label):
    """Sort key placing regions in the canonical order above."""
    for index, (name, _) in enumerate(SUPERCLASSES):
        if name == label:
            return index
    return len(SUPERCLASSES) + (0 if label == OTHER else 1)


def group(cluster_names, labels=None):
    """{superclass: [cluster names]}, regions in canonical order.

    `labels` is an optional {name: superclass} from `classify`, for callers
    that had the reactions; anything missing from it is judged by name.
    """
    labels = labels or {}
    regions = {}
    for name in cluster_names:
        label = labels.get(name) or superclass(name)
        regions.setdefault(label, []).append(name)
    return dict(sorted(regions.items(), key=lambda item: order_index(item[0])))
