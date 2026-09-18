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
"""

# Ordered: this is also the order regions are laid out in, and it follows how
# metabolism is normally taught and drawn -- central carbon first, then what
# feeds off it, with degradation and transport at the edges.
SUPERCLASSES = [
    ("Carbohydrate metabolism", (
        "glycolysis", "gluconeogen", "citrate cycle", "citric acid", "tca",
        "pentose", "glucuronate", "fructose", "mannose", "galactose", "starch",
        "sucrose", "amino sugar", "nucleotide sugar", "pyruvate metab",
        "glyoxylate", "dicarboxylate", "propanoate", "butanoate", "ascorbate",
        "aldarate", "inositol phosphate", "c5-branched", "anaplerotic",
    )),
    ("Energy metabolism", (
        "oxidative phosphoryl", "photosynthesis", "carbon fixation", "methane",
        "nitrogen metab", "sulfur metab", "respiration", "electron transport",
    )),
    ("Amino acid metabolism", (
        "alanine", "aspartate", "glutamate", "glutamine", "glycine", "serine",
        "threonine", "cysteine", "methionine", "valine", "leucine",
        "isoleucine", "lysine", "arginine", "proline", "histidine", "tyrosine",
        "phenylalanine", "tryptophan", "beta-alanine", "taurine",
        "phosphonate", "selenocompound", "cyanoamino", "d-amino",
        "d-glutamine", "d-arginine", "glutathione", "urea cycle", "amino acid",
    )),
    ("Nucleotide metabolism", (
        "purine", "pyrimidine", "nucleotide metab",
    )),
    ("Lipid metabolism", (
        "fatty acid", "glycerolipid", "glycerophospholipid", "sphingolipid",
        "ether lipid", "arachidonic", "linoleic", "linolenic", "steroid",
        "bile acid", "cutin", "suberine", "wax", "lipid metab", "lipopolysacch",
        "membrane lipid", "beta-oxidation", "ketone body",
    )),
    ("Cofactor and vitamin metabolism", (
        "porphyrin", "chlorophyll", "thiamine", "riboflavin", "vitamin",
        "nicotinate", "nicotinamide", "pantothenate", "coa biosynth", "biotin",
        "lipoic", "folate", "one carbon pool", "retinol", "ubiquinone",
        "terpenoid-quinone", "cofactor",
    )),
    ("Glycan metabolism", (
        "glycan", "peptidoglycan", "o-antigen", "glycosaminoglycan", "mucin",
        "glycosylphosphatidyl", "teichoic",
    )),
    ("Terpenoid and polyketide metabolism", (
        "terpenoid", "polyketide", "carotenoid", "zeatin", "limonene",
        "brassinosteroid", "insect hormone", "geraniol",
    )),
    ("Secondary metabolite biosynthesis", (
        "alkaloid", "flavonoid", "flavone", "phenylpropanoid", "stilbenoid",
        "betalain", "glucosinolate", "anthocyanin", "isoflavonoid",
        "novobiocin", "acarbose", "streptomycin", "penicillin", "cephalosporin",
        "clavulanic", "puromycin", "staurosporine", "tetracycline",
        "aminoglycoside", "vancomycin", "monobactam", "carbapenem", "phenazine",
        "prodigiosin", "neomycin", "secondary metabolite",
    )),
    ("Xenobiotic degradation", (
        "degradation", "xenobiotic", "drug metabolism", "benzoate",
        "naphthalene", "toluene", "dioxin", "chloroalkane", "chlorocyclohexane",
        "styrene", "atrazine", "caprolactam", "fluorobenzoate", "aminobenzoate",
        "nitrotoluene", "ethylbenzene", "furfural", "polycyclic", "bisphenol",
    )),
    ("Transport and exchange", (
        "transport", "exchange", "extracellular", "uptake", "secretion",
        "demand", "sink", "biomass",
    )),
]

OTHER = "Other metabolism"
UNASSIGNED = "Unassigned clusters"

STRUCTURAL_PREFIXES = ("Unannotated", "Other", "Uncategorized", "Cluster_")


def superclass(cluster_name):
    """Which metabolic region a cluster belongs to.

    Structurally derived clusters have no biological name to match on, so they
    are kept in their own region rather than being guessed into one -- a wrong
    region is worse than an honest "unassigned".
    """
    name = str(cluster_name).strip()
    if name.startswith(STRUCTURAL_PREFIXES):
        return UNASSIGNED

    lowered = name.lower()
    for label, keywords in SUPERCLASSES:
        for keyword in keywords:
            if keyword in lowered:
                return label
    return OTHER


def order_index(label):
    """Sort key placing regions in the canonical order above."""
    for index, (name, _) in enumerate(SUPERCLASSES):
        if name == label:
            return index
    return len(SUPERCLASSES) + (0 if label == OTHER else 1)


def group(cluster_names):
    """{superclass: [cluster names]}, regions in canonical order."""
    regions = {}
    for name in cluster_names:
        regions.setdefault(superclass(name), []).append(name)
    return dict(sorted(regions.items(), key=lambda item: order_index(item[0])))
