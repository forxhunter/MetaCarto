"""What a metabolite is, independent of how its model spells the id.

The pipeline was written against BiGG, where `atp_c` is ATP in the cytosol and
stripping `_c` gives the compound. Other reconstructions spell the same thing
`MAM01371c` (Human-GEM), `s_0434` (Yeast-GEM, numbered per compartment),
`cpd00002_c0` (ModelSEED) or `atp[c]` (COBRA Toolbox). Read with BiGG rules,
Yeast-GEM collapses every metabolite to `s` and draws every reaction as a
transport step; Human-GEM never sees a transport step at all, and draws water
and protons on every reaction because `MAM02040c` is not on a list of BiGG ids.

So two questions are answered here once, from the chemistry and the model's
own metadata, and every module asks them here:

  species(id)   the same compound in another compartment has the same key;
  currency(id)  'atp', 'h2o', 'nad', ... for currency metabolites, else None.

`register(model)` reads a model; until then (or for an id it did not see) the
answers fall back to the BiGG rules, so BiGG behaviour is unchanged.
"""

import re

# Currency metabolites by canonical key: KEGG compound, ModelSEED compound,
# and names as reconstructions write them. Keys are BiGG base ids, which is
# what the rest of the pipeline's lists are written in.
_CURRENCY = {
    "h": (("C00080",), ("cpd00067",), ("h+", "proton", "hydrogen ion", "h(+)")),
    "h2o": (("C00001",), ("cpd00001",), ("h2o", "water")),
    "o2": (("C00007",), ("cpd00007",), ("o2", "oxygen", "dioxygen")),
    "co2": (("C00011",), ("cpd00011",), ("co2", "carbon dioxide")),
    "pi": (("C00009",), ("cpd00009",), ("phosphate", "orthophosphate", "pi", "inorganic phosphate")),
    "ppi": (("C00013",), ("cpd00012",), ("diphosphate", "pyrophosphate", "ppi", "inorganic diphosphate")),
    "pppi": (("C00536",), ("cpd00421",), ("triphosphate", "inorganic triphosphate")),
    "nh4": (("C01342", "C00014"), ("cpd00013",), ("ammonium", "ammonia", "nh4+", "nh3")),
    "so4": (("C00059",), ("cpd00048",), ("sulfate",)),
    "so3": (("C00094",), ("cpd00081",), ("sulfite",)),
    "atp": (("C00002",), ("cpd00002",), ("atp", "adenosine triphosphate", "adenosine 5'-triphosphate")),
    "adp": (("C00008",), ("cpd00008",), ("adp", "adenosine diphosphate", "adenosine 5'-diphosphate")),
    "amp": (("C00020",), ("cpd00018",), ("amp", "adenosine monophosphate", "adenosine 5'-monophosphate")),
    "gtp": (("C00044",), ("cpd00038",), ("gtp", "guanosine triphosphate")),
    "gdp": (("C00035",), ("cpd00031",), ("gdp", "guanosine diphosphate")),
    "gmp": (("C00144",), ("cpd00126",), ("gmp", "guanosine monophosphate")),
    "utp": (("C00075",), ("cpd00062",), ("utp", "uridine triphosphate")),
    "udp": (("C00015",), ("cpd00014",), ("udp", "uridine diphosphate")),
    "ump": (("C00105",), ("cpd00091",), ("ump", "uridine monophosphate")),
    "ctp": (("C00063",), ("cpd00052",), ("ctp", "cytidine triphosphate")),
    "cdp": (("C00112",), ("cpd00096",), ("cdp", "cytidine diphosphate")),
    "cmp": (("C00055",), ("cpd00046",), ("cmp", "cytidine monophosphate")),
    "nad": (("C00003",), ("cpd00003",), ("nad", "nad+", "nicotinamide adenine dinucleotide")),
    "nadh": (("C00004",), ("cpd00004",), ("nadh", "nicotinamide adenine dinucleotide - reduced")),
    "nadp": (("C00006",), ("cpd00006",), ("nadp", "nadp+", "nicotinamide adenine dinucleotide phosphate")),
    "nadph": (("C00005",), ("cpd00005",), ("nadph", "nicotinamide adenine dinucleotide phosphate - reduced")),
    "fad": (("C00016",), ("cpd00015",), ("fad", "flavin adenine dinucleotide", "flavin adenine dinucleotide oxidized")),
    "fadh2": (("C01352",), ("cpd00982",), ("fadh2", "flavin adenine dinucleotide reduced")),
    "fmn": (("C00061",), ("cpd00050",), ("fmn", "flavin mononucleotide")),
    "fmnh2": (("C01847",), (), ("fmnh2", "reduced fmn")),
    "coa": (("C00010",), ("cpd00010",), ("coa", "coenzyme a")),
    "thf": (("C00101",), ("cpd00087", "cpd29254"), ("thf", "tetrahydrofolate", "5,6,7,8-tetrahydrofolate")),
    "gthrd": (("C00051",), ("cpd00042",), ("glutathione", "reduced glutathione", "gsh")),
    "gthox": (("C00127",), ("cpd00111",), ("oxidized glutathione", "glutathione disulfide", "gssg")),
    "h2o2": (("C00027",), ("cpd00025",), ("hydrogen peroxide", "h2o2")),
    "hco3": (("C00288",), ("cpd00242",), ("bicarbonate", "hydrogen carbonate", "hco3-")),
    "na1": (("C01330",), ("cpd00971",), ("sodium", "na+")),
    "k": (("C00238",), ("cpd00205",), ("potassium", "k+")),
    "cl": (("C00698",), ("cpd00099",), ("chloride", "cl-")),
    "ca2": (("C00076",), ("cpd00063",), ("calcium", "ca2+")),
    "mg2": (("C00305",), ("cpd00254",), ("magnesium", "mg2+")),
    "fe2": (("C14818",), ("cpd10515",), ("fe2+", "iron(2+)", "ferrous")),
    "fe3": (("C14819",), ("cpd10516",), ("fe3+", "iron(3+)", "iron (fe3+)", "ferric")),
    "zn2": (("C00038",), ("cpd00034",), ("zinc", "zn2+")),
    "mn2": (("C00034", "C19610"), ("cpd00030", "cpd20863"), ("manganese", "mn2+")),
    "cu2": (("C00070",), ("cpd00058",), ("copper", "cu2+")),
    "cobalt2": (("C00175",), ("cpd00149",), ("cobalt", "co2+")),
    "itp": (("C00081",), ("cpd00068",), ("itp", "inosine triphosphate")),
    "idp": (("C00104",), ("cpd00090",), ("idp", "inosine diphosphate")),
    "imp": (("C00130",), ("cpd00114",), ("imp", "inosine monophosphate")),
    "datp": (("C00131",), ("cpd00115",), ("datp",)),
    "dadp": (("C00206",), ("cpd00177",), ("dadp",)),
    "damp": (("C00360",), ("cpd00294",), ("damp",)),
    "dgtp": (("C00286",), ("cpd00241",), ("dgtp",)),
    "dctp": (("C00458",), ("cpd00356",), ("dctp",)),
    "dttp": (("C00459",), ("cpd00357",), ("dttp",)),
    "q8": (("C17569",), ("cpd15560",), ("ubiquinone-8", "ubiquinone 8")),
    "q8h2": ((), ("cpd15561", "cpd29608"), ("ubiquinol-8", "ubiquinol 8")),
    "q6": (("C17568",), ("cpd15290",), ("ubiquinone-6", "ubiquinone 6")),
    "q6h2": ((), ("cpd15291",), ("ubiquinol-6", "ubiquinol 6")),
    "mqn8": ((), ("cpd15500",), ("menaquinone-8", "menaquinone 8")),
    "mql8": ((), ("cpd15499",), ("menaquinol-8", "menaquinol 8")),
    "trdox": (("C00343",), ("cpd11420", "cpd27735", "cpd29682"), ("oxidized thioredoxin", "thioredoxin disulfide")),
    "trdrd": (("C00342",), ("cpd11421", "cpd28060"), ("reduced thioredoxin", "thioredoxin")),
    "mlthf": (("C00143",), (), ("5,10-methylenetetrahydrofolate",)),
    "methf": (("C00445",), ("cpd00347",), ("5,10-methenyltetrahydrofolate",)),
    "5mthf": (("C00440",), ("cpd00345",), ("5-methyltetrahydrofolate",)),
    "10fthf": (("C00234",), ("cpd00201",), ("10-formyltetrahydrofolate",)),
    "h2": (("C00282",), ("cpd11640",), ("h2", "dihydrogen")),
    "h2s": (("C00283",), ("cpd00239", "cpd24697"), ("hydrogen sulfide",)),
    "n2": (("C00697",), ("cpd00528",), ("dinitrogen", "n2")),
    "no": (("C00533",), (), ("nitric oxide",)),
    "no2": (("C00088",), ("cpd00075",), ("nitrite",)),
    "no3": (("C00244",), ("cpd00209",), ("nitrate",)),
}

# Formulas that identify a currency compound on their own, with the charge it
# carries when the model records one. The charge is what keeps superoxide
# (O2, charge -1) from being read as oxygen. Read last: an annotation or a
# name is better evidence than a formula.
_FORMULA = {"H2O": ("h2o", 0), "H": ("h", 1), "O2": ("o2", 0), "CO2": ("co2", 0),
            "H2O2": ("h2o2", 0), "Na": ("na1", 1), "K": ("k", 1), "Cl": ("cl", -1),
            "Ca": ("ca2", 2), "Mg": ("mg2", 2), "Zn": ("zn2", 2), "Mn": ("mn2", 2),
            "Fe": (None, None)}

_XREFS = ("kegg.compound", "chebi", "seed.compound", "metanetx.chemical",
          "hmdb", "biocyc", "inchi_key")
_EXPECTED = {key: (formula, charge) for formula, (key, charge) in _FORMULA.items() if key}

_BY_KEGG, _BY_SEED, _BY_NAME = {}, {}, {}
for _key, (_kegg, _seed, _names) in _CURRENCY.items():
    for _k in _kegg:
        _BY_KEGG[_k] = _key
    for _s in _seed:
        _BY_SEED[_s] = _key
    for _n in _names:
        _BY_NAME[_n] = _key

_species, _currency, _compartment = {}, {}, {}
_registered = [None]


def _bigg_strip(met_id):
    """'atp_c' -> 'atp', the BiGG rule: a one- or two-character suffix."""
    for n in (2, 3):
        if len(met_id) > n and met_id[-n] == "_":
            return met_id[:-n]
    return met_id


def _strip(met_id, compartment):
    """The id without its compartment, in any of the spellings models use.

    None when the id does not carry its compartment at all (Yeast-GEM's
    `s_0434`), so the caller knows the id cannot name the compound.
    """
    if compartment:
        for suffix in ("_" + compartment, "[" + compartment + "]", "(" + compartment + ")"):
            if met_id.endswith(suffix) and len(met_id) > len(suffix):
                return met_id[:-len(suffix)]
        # A bare suffix only after a number, as in Human-GEM's MAM01371c:
        # `glucose` in compartment `e` is not `glucos` outside it.
        if (met_id.endswith(compartment) and len(met_id) > len(compartment)
                and met_id[-len(compartment) - 1].isdigit()):
            return met_id[:-len(compartment)]
    stripped = _bigg_strip(met_id)
    return stripped if stripped != met_id else None


def _clean_name(name, compartment="", formula=""):
    """Lower-case name without what some models append to it.

    'ATP [cytoplasm]' (Yeast-GEM), 'ATP_c0' (ModelSEED), 'ATP C10H12N5O13P3'
    (BiGG) and 'ATP' all give 'atp'.
    """
    name = str(name or "").strip()
    name = re.sub(r"\s*\[[^\]]*\]$", "", name)
    if compartment and name.endswith("_" + compartment):
        name = name[:-len(compartment) - 1]
    if formula and name.endswith(" " + formula):
        name = name[:-len(formula) - 1]
    # The formula as written in the name need not be the charged one stored.
    name = re.sub(r"\s+(?=\S*\d)(?:[A-Z][a-z]?\d*){2,}$", "", name)
    return name.strip().lower()


def _annotation(met, key):
    value = (getattr(met, "annotation", None) or {}).get(key)
    if not value:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(v).split(":")[-1] for v in values]


def _consistent(met, key):
    """False when the metabolite's own formula or charge contradicts `key`.

    Annotations are wrong often enough to matter: BiGG annotates superoxide
    with oxygen's KEGG id and hydroxide with water's, and believing them hid
    superoxide dismutase's substrate as if it were O2.
    """
    expected = _EXPECTED.get(key)
    if expected is None:
        return True
    formula = (getattr(met, "formula", "") or "").strip()
    charge = getattr(met, "charge", None)
    if formula and formula != expected[0]:
        return False
    return charge is None or charge == expected[1]


def _currency_of(met, base, name):
    if base and base in _CURRENCY:
        return base
    for kegg in _annotation(met, "kegg.compound"):
        if kegg in _BY_KEGG and _consistent(met, _BY_KEGG[kegg]):
            return _BY_KEGG[kegg]
    for seed in _annotation(met, "seed.compound") + [base or ""]:
        if seed in _BY_SEED and _consistent(met, _BY_SEED[seed]):
            return _BY_SEED[seed]
    if name in _BY_NAME and _consistent(met, _BY_NAME[name]):
        return _BY_NAME[name]
    # A formula is the weakest evidence: superoxide is O2 too. Use it only
    # for a metabolite with no database cross-reference saying what it is.
    if any(_annotation(met, key) for key in _XREFS):
        return None
    formula = (getattr(met, "formula", "") or "").strip()
    if formula == "Fe":
        return {2: "fe2", 3: "fe3"}.get(getattr(met, "charge", None))
    if formula in _FORMULA and _consistent(met, _FORMULA[formula][0]):
        return _FORMULA[formula][0]
    return None


def register(model):
    """Read every metabolite of `model`; replaces whatever was registered.

    Cheap to call again for the model already registered, so every entry
    point that receives a model can call it.
    """
    stamp = (id(model), hash(tuple(m.id for m in model.metabolites)))
    if _registered[0] == stamp:
        return
    _registered[0] = stamp
    _species.clear()
    _currency.clear()
    _compartment.clear()
    for met in model.metabolites:
        compartment = getattr(met, "compartment", "") or ""
        base = _strip(met.id, compartment)
        name = _clean_name(getattr(met, "name", ""), compartment, getattr(met, "formula", "") or "")
        if base is not None:
            key = base
        else:
            # Ids numbered per compartment (Yeast-GEM): the id says nothing,
            # the name and formula say which compound this is.
            key = "name:" + name + "|" + (getattr(met, "formula", "") or "")
        _species[met.id] = key
        _currency[met.id] = _currency_of(met, base, name)
        _compartment[met.id] = compartment


def species(met):
    """Key shared by one compound in every compartment."""
    met_id = getattr(met, "id", met)
    if met_id in _species:
        return _species[met_id]
    return _bigg_strip(met_id)


def currency(met):
    """Canonical currency key ('atp', 'h2o', ...) or None."""
    met_id = getattr(met, "id", met)
    if met_id in _currency:
        return _currency[met_id]
    base = _bigg_strip(met_id)
    return base if base in _CURRENCY else None


def canonical(met):
    """Currency key if the metabolite is currency, else its species key.

    The vocabulary the pipeline's curated lists are written in: BiGG base ids.
    """
    return currency(met) or species(met)


def compartment(met):
    met_id = getattr(met, "id", met)
    if met_id in _compartment:
        return _compartment[met_id]
    base = _bigg_strip(met_id)
    return met_id[len(base) + 1:] if base != met_id else ""


def opaque(identifier):
    """True for ids that mean nothing to a reader: MAM01371, s_0434, cpd00002.

    A short prefix and a long serial number. BiGG's lipid ids (`dag181`,
    `pe160`) carry a chain length, not a serial, and stay readable.
    """
    return bool(re.fullmatch(r"[A-Za-z]{1,4}_?\d{4,}", str(identifier)))
