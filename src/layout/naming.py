"""Name a structurally derived cluster from what its reactions say they are.

A map captioned `Cluster_8` tells a reader nothing. But the reactions in that
cluster already carry names -- "Very-Long-Chain 3-Oxoacyl Coenzyme A Synthase",
"Very-Long-Chain (3R)-3-Hydroxyacyl Coenzyme A Dehydratase", "Long-Chain-Fatty-
Acid Coenzyme A Ligase" -- and a reader looking at those would call the cluster
very-long-chain fatty-acid elongation. No external annotation is needed; the
information is in the file.

Plain majority vote over words does not work, because the most frequent words
in any set of reaction names are the generic ones: "coenzyme", "dehydrogenase",
"transport". What distinguishes a cluster is the vocabulary that is common
*here* and rare *elsewhere in this model*, which is what TF-IDF measures.
"""

import math
import re
from collections import Counter

from .taxonomy import is_structural

# Words that describe enzymology or chemistry in general rather than this
# pathway in particular. Kept small: the TF-IDF weighting already suppresses
# most generic vocabulary, and an over-long stoplist starts deleting real
# pathway names ("fatty", "acid").
_STOPWORDS = {
    "reaction", "reactions", "and", "the", "of", "for", "with", "from", "into",
    "via", "type", "form", "other", "unknown", "unnamed", "misc", "general",
    "reversible", "irreversible", "spontaneous", "isomer", "isoform",
    "cytosol", "cytosolic", "mitochondrial", "mitochondria", "peroxisomal",
    "peroxisome", "lysosomal", "lysosome", "nuclear", "nucleus", "golgi",
    "extracellular", "intracellular", "reticulum", "endoplasmic", "membrane",
    "periplasm", "periplasmic", "n-c", "c-n",
    "coa", "atp", "adp", "nad", "nadh", "nadp", "nadph", "water", "proton",
    # Carrier and enzyme-class vocabulary: present in thousands of names and
    # never the thing that distinguishes one pathway from another.
    "coenzyme", "carrier", "apparatus", "production", "formation", "synthesis",
    "degradation", "metabolism", "utilization", "conversion", "exchange",
    "diffusion", "facilitated", "sodium", "potassium", "proton-coupled",
    "reaction-", "unknown-", "putative",
    # Identifier and provenance noise. Recon3D leaves thousands of reactions
    # named after their own id ("HMR 0203") or after a source database, and
    # those tokens are both frequent and perfectly uninformative -- left in,
    # "hmr" wins the vote and the cluster gets captioned "Hmr 4".
    "hmr", "tcdb", "recon", "homo", "sapiens", "hepatocytes", "utilized",
    "rbc", "ec-code", "kegg",
}

# A name that is only one of these is grammatically a name and biologically
# nothing: every cluster contains a dehydrogenase. Rejected as a final label
# even when it wins on frequency, so the caller falls back rather than printing
# a caption that does not distinguish this map from thirty others.
_UNINFORMATIVE = {
    "acid", "acids", "transport", "exchange", "dehydrogenase", "reductase",
    "kinase", "oxidase", "synthase", "synthetase", "ligase", "hydrolase",
    "transferase", "isomerase", "phosphatase", "ester", "esters",
    "beta", "alpha", "gamma", "delta", "chain", "long", "short", "medium",
    "protein", "complex", "subunit", "family", "group", "class",
}

# Longest caption worth printing. Past this a title stops being read and starts
# being a paragraph; the first phrase alone is the informative part.
_MAX_LABEL = 46

_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def _tokens(text):
    for raw in _TOKEN.findall(str(text or "")):
        token = raw.lower().strip("-")
        if len(token) < 3 or token in _STOPWORDS:
            continue
        yield token


def document_frequency(reactions):
    """How many reactions in the whole model use each word."""
    frequency = {}
    for reaction in reactions:
        for token in set(_tokens(reaction.name)):
            frequency[token] = frequency.get(token, 0) + 1
    return frequency


def _phrases(text, max_length=3):
    """Word n-grams from one reaction name, longest first."""
    words = list(_tokens(text))
    for length in range(min(max_length, len(words)), 0, -1):
        for start in range(len(words) - length + 1):
            yield " ".join(words[start:start + length])


def phrase_frequency(reactions, max_length=3):
    """How many reactions in the whole model contain each phrase."""
    frequency = {}
    for reaction in reactions:
        for phrase in set(_phrases(reaction.name, max_length)):
            frequency[phrase] = frequency.get(phrase, 0) + 1
    return frequency


def describe(reactions, frequency, total, max_length=3, exclude=()):
    """A short name for a cluster, or None if its names say nothing.

    Scores phrases, not words. The vocabulary that identifies a pathway is
    almost always a phrase -- "very-long-chain", "beta oxidation", "bile acid"
    -- and scoring single words instead returns the carrier that every member
    happens to mention ("coenzyme"), which names nothing.
    """
    if not reactions:
        return None

    exclude = set(exclude)
    counts = {}
    for reaction in reactions:
        for phrase in set(_phrases(reaction.name, max_length)):
            if phrase in exclude:
                continue
            counts[phrase] = counts.get(phrase, 0) + 1

    def rank(min_share, min_count):
        out = []
        for phrase, count in counts.items():
            share = count / len(reactions)
            if share < min_share or count < min_count:
                continue
            idf = math.log((total + 1) / (frequency.get(phrase, 0) + 1))
            words = phrase.count(" ") + 1
            # A longer phrase that is still characteristic is more informative
            # than a single word with the same coverage.
            out.append((share * idf * (1.0 + 0.6 * (words - 1)), phrase))
        out.sort(reverse=True)
        return out

    # Strict pass: a phrase carried by most of the cluster. When it finds
    # something the name is trustworthy enough to print unqualified.
    scored = rank(0.3, 2)
    floor = 0.25

    if not scored or scored[0][0] < floor:
        # Relaxed pass. A cluster of mixed chemistry has no phrase in a third of
        # its members, but it usually still has one shared by a sixth -- and a
        # partial descriptor ("Bile acid", "Acylcarnitine") beats the ordinal
        # caption that is the only alternative. The bar is lowered, not removed:
        # below it the vocabulary really is generic and None is the honest answer.
        scored = rank(0.15, 2)
        floor = 0.12

    if not scored:
        return None

    best_score, best = scored[0]
    if best_score < floor:
        return None

    # A single generic word names nothing, so prefer the next candidate that
    # says something. But only *prefer* it: rejecting outright returns None and
    # the caller then prints "Unannotated 35", which is strictly worse than a
    # vague-but-true "Transport". Demote, never discard.
    if best in _UNINFORMATIVE:
        for score, phrase in scored[1:]:
            if phrase not in _UNINFORMATIVE:
                best_score, best = score, phrase
                break

    # A second, non-overlapping phrase adds specificity when there is one.
    parts = [best]
    for score, phrase in scored[1:]:
        if score < best_score * 0.55:
            break
        if phrase in _UNINFORMATIVE:
            continue
        if any(word in set(best.split()) for word in phrase.split()):
            continue
        parts.append(phrase)
        break

    label = " ".join(part.capitalize() for part in parts)
    if len(label) > _MAX_LABEL:
        label = parts[0].capitalize()
    return label[:_MAX_LABEL].rstrip(" -,")


def transported_species(reactions, limit=2):
    """Names of the compounds a transport cluster actually moves.

    A transport reaction is the one case where the reaction names are reliably
    useless -- thousands of them are called nothing but "transport", "facilated
    transport" or a carrier-family code, so the phrase vote returns "Transport"
    and the caller falls back to "Transport 31". What distinguishes one
    transport cluster from another is not the verb, it is the cargo, and the
    cargo is named in the *metabolite* table.

    A metabolite is being transported when the same compound appears on both
    sides of the reaction in different compartments, which is exactly the
    signature `compound.py` uses to recognise a transport step.
    """
    cargo = {}
    for reaction in reactions:
        by_base = {}
        for metabolite, coefficient in reaction.metabolites.items():
            base = metabolite.id.rsplit("_", 1)[0]
            by_base.setdefault(base, []).append(coefficient)
        for base, coefficients in by_base.items():
            # Moved, not consumed: present as both substrate and product.
            if not (any(c < 0 for c in coefficients) and any(c > 0 for c in coefficients)):
                continue
            for metabolite in reaction.metabolites:
                if metabolite.id.rsplit("_", 1)[0] != base:
                    continue
                name = (getattr(metabolite, "name", "") or "").strip()
                if not name:
                    continue
                tokens = [t for t in _tokens(name)]
                if tokens:
                    cargo[base] = " ".join(tokens[:3])
                break
    if not cargo:
        return []
    counts = {}
    for label in cargo.values():
        counts[label] = counts.get(label, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [label for label, _ in ranked[:limit]]


def cargo_species(reactions, limit=2):
    """What a boundary cluster handles, for naming exchange and sink maps.

    An exchange reaction has one side, so `transported_species` finds nothing:
    nothing is "moved" in the both-sides sense. But `EX_chsterol_e` is still
    obviously about cholesterol, and the metabolite table says so. Falls back to
    the compounds the cluster touches most often.
    """
    moved = transported_species(reactions, limit)
    if moved:
        return moved
    counts = {}
    for reaction in reactions:
        for metabolite in reaction.metabolites:
            name = (getattr(metabolite, "name", "") or "").strip()
            if not name:
                continue
            tokens = [t for t in _tokens(name)][:3]
            if tokens:
                label = " ".join(tokens)
                counts[label] = counts.get(label, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [label for label, _ in ranked[:limit]]


def name_clusters(groups, model):
    """Replace structural cluster names with ones derived from reaction names.

    Clusters that already carry a biological name -- from the model's own
    subsystem field or from a KEGG pathway -- are left alone; those names come
    from curation and beat anything inferred from free text.
    """
    reactions = list(model.reactions)
    frequency = phrase_frequency(reactions)
    total = len(reactions)

    renamed, used = {}, set()
    for name, members in groups.items():
        if not is_structural(name):
            renamed[name] = members
            used.add(name)
            continue

        # Note: retrying with the winning phrase excluded, to turn "Transport
        # 31..42" into distinct names, makes things worse rather than better --
        # it exhausts the shared vocabulary and the cluster ends up as "Other".
        # The numbering is a symptom of the bins being arbitrary mixtures, and
        # has to be fixed in the decomposition (group transport by what is
        # transported), not here.
        label = describe(members, frequency, total)

        # "Transport" names nothing on its own, and numbering it names less.
        # When the verb is all the reaction names gave us, ask what is being
        # carried instead.
        if not label or label.strip().lower() in _UNINFORMATIVE:
            cargo = transported_species(members)
            if cargo:
                label = "%s transport" % cargo[0].capitalize()

        if not label:
            renamed[name] = members
            used.add(name)
            continue

        candidate, index = label, 2
        while candidate in used:
            candidate = f"{label} {index}"
            index += 1
        renamed[candidate] = members
        used.add(candidate)

    return renamed


# --------------------------------------------------------------------------
# telling the pieces of one subsystem apart
# --------------------------------------------------------------------------

# Past this a compound is currency, whatever the cofactor list says: a
# qualifier naming ATP distinguishes nothing.
_CURRENCY_DEGREE = 40

# Enzyme-class and carrier vocabulary. Fine inside a phrase, but a qualifier
# made only of these -- "Synthase", "Acyltransferase Phosphate" -- tells one
# piece from the next no better than an ordinal does.
_GENERIC = _UNINFORMATIVE | {
    "transporter", "transfer", "acyltransferase", "phospholipase",
    "lysophospholipase", "phosphate", "phosphates", "hydratase", "mutase",
    "lyase", "oxidoreductase", "transaminase", "aminotransferase",
    "deaminase", "decarboxylase", "carboxylase", "dehydratase", "epimerase",
    "racemase", "esterase", "lipase", "peptidase", "nucleosidase",
    "phosphorylase", "phosphoribosyltransferase", "glycosyltransferase",
    "monooxygenase", "dioxygenase", "hydroxylase", "thioesterase", "enzyme",
    "system", "periplasm", "sink", "demand", "export", "import", "uptake",
}

# Longest compound name worth printing as a qualifier.
_MAX_COMPOUND = 28


def _informative(phrase):
    return bool(phrase) and any(w not in _GENERIC for w in phrase.lower().split())


def _short_compound(name):
    """A compound name short enough to caption with, or None.

    Keeps the name up to its first bracket, slash or colon -- "Propionate
    (n-C3:0)" is propionate, "Triacylglycerol/16:0/16:1" a triacylglycerol --
    and rejects what is still a paragraph, like
    "CDP-1,2-dioctadec-11-enoylglycerol".
    """
    name = re.split(r"[(\[/:]", str(name or ""), maxsplit=1)[0].strip(" ,;-")
    if not name or len(name) > _MAX_COMPOUND or not re.search(r"[A-Za-z]{3}", name):
        return None
    return name


def _themes(piece, classes, degree, kegg_mapping):
    """Counter of what each reaction in a piece is about."""
    from .taxonomy import theme
    return Counter(t for t in (theme(r, classes, degree, kegg_mapping) for r in piece) if t)


def _theme_qualifier(themes, size):
    """"amino acids", or "amino acids / lipids" for a piece that is two things."""
    from .taxonomy import CARGO_NOUNS
    ranked = [(n, label) for label, n in themes.most_common() if label in CARGO_NOUNS]
    if not ranked:
        return None
    if ranked[0][0] >= 0.5 * size:
        return CARGO_NOUNS[ranked[0][1]]
    if len(ranked) > 1 and ranked[0][0] + ranked[1][0] >= 0.6 * size:
        return "%s / %s" % (CARGO_NOUNS[ranked[0][1]], CARGO_NOUNS[ranked[1][1]])
    return None


def _compound_qualifier(piece, family, degree, avoid=()):
    """The compound most particular to this piece among its siblings.

    A piece of "Alternate Carbon Metabolism" that phrase-voting cannot name is
    still recognisably the galactarate piece: galactarate is in most of its
    reactions and none of its siblings'.
    """
    in_piece, in_family, names = {}, {}, {}
    for reactions, counts in ((piece, in_piece), (family, in_family)):
        for reaction in reactions:
            for metabolite in reaction.metabolites:
                base = metabolite.id.rsplit("_", 1)[0]
                counts[base] = counts.get(base, 0) + 1
                short = _short_compound(getattr(metabolite, "name", ""))
                if short:
                    names.setdefault(base, short)
    avoid = {a.lower() for a in avoid}
    candidates = [(count / in_family[base], count, base)
                  for base, count in in_piece.items()
                  if base in names and names[base].lower() not in avoid
                  and degree.get(base, 0) <= _CURRENCY_DEGREE]
    if not candidates:
        return None
    return names[max(candidates)[2]]


def _landmark(piece, dominant, classes, degree, kegg_mapping):
    """The best-known compound of a piece's dominant class.

    Two pieces of extracellular amino-acid exchange differ in which compounds
    they carry, and each exchange step carries one, so "the compound most
    particular to the piece" is a forty-way tie broken arbitrarily -- which is
    how a lipid page came to be captioned with xanthosine. The compound a
    reader is likeliest to recognise is the best-connected one, as a map sheet
    is named after its largest town.
    """
    from .taxonomy import cargo, theme
    names, bases = {}, set()
    for reaction in piece:
        if theme(reaction, classes, degree, kegg_mapping) != dominant:
            continue
        carried = cargo(reaction, classes, degree)
        for metabolite in reaction.metabolites:
            base = metabolite.id.rsplit("_", 1)[0]
            short = _short_compound(getattr(metabolite, "name", ""))
            if short:
                names.setdefault(base, short)
            if (carried and base == carried[1]) or (
                    not carried and classes.get(base) == dominant):
                bases.add(base)
    candidates = [(degree.get(b, 0), names[b]) for b in sorted(bases)
                  if b in names and degree.get(b, 0) <= _CURRENCY_DEGREE]
    if not candidates:
        return None
    # Best connected; alphabetical among equals, so the choice is stable.
    return min(candidates, key=lambda c: (-c[0], c[1]))[1]


def qualify(pieces, family_name, classes, degree, kegg_mapping=None):
    """One short qualifier per piece of a split subsystem, all distinct.

    Splitting "Transport, Inner Membrane" into pieces that fit a page is
    unavoidable; captioning them "(1)" to "(149)" is not. In order of
    preference, a piece is told from its siblings by:

      * what it is about, when the siblings differ in that -- the pieces of a
        transport subsystem carry amino acids, sugars, inorganic ions;
      * the vocabulary common in it and rare in its siblings, which is
        `describe` with the family, not the whole model, as the reference
        corpus, and the family's own words excluded;
      * the compound most particular to it.

    Pieces that still share a qualifier get a landmark compound after a comma,
    "amino acids and peptides, L-Glutamate", and only then an ordinal.

    `classes` and `degree` come from `taxonomy.compound_classes`.
    """
    family = [r for piece in pieces for r in piece]
    frequency = phrase_frequency(family)
    exclude = set(_phrases(family_name))

    profiles = [_themes(piece, classes, degree, kegg_mapping) for piece in pieces]
    by_theme = len({p.most_common(1)[0][0] for p in profiles if p}) > 1

    labels, kinds = [], []
    for piece, profile in zip(pieces, profiles):
        label, kind = None, None
        if by_theme:
            label, kind = _theme_qualifier(profile, len(piece)), "theme"
        if not label:
            phrase = describe(piece, frequency, len(family), exclude=exclude)
            label, kind = (phrase if _informative(phrase) else None), "phrase"
        if not label:
            label, kind = _compound_qualifier(piece, family, degree), "compound"
        labels.append(label or "")
        kinds.append(kind)

    counts = Counter(labels)
    refined = []
    for piece, profile, label, kind in zip(pieces, profiles, labels, kinds):
        if label and counts[label] > 1:
            landmark = None
            if kind == "theme":
                landmark = _landmark(piece, profile.most_common(1)[0][0],
                                     classes, degree, kegg_mapping)
                if not landmark:
                    # Pieces of all-currency steps have no compound worth
                    # naming, but their reaction names still differ.
                    phrase = describe(piece, frequency, len(family), exclude=exclude)
                    landmark = phrase if _informative(phrase) else None
            if not landmark:
                landmark = _compound_qualifier(piece, family, degree, avoid=[label])
            if landmark:
                label = f"{label}, {landmark}"
        refined.append(label)

    counts = Counter(refined)
    index, result = {}, []
    for label in refined:
        if not label or counts[label] > 1:
            index[label] = index.get(label, 0) + 1
            label = f"{label} {index[label]}".strip()
        result.append(label)
    return result


# --------------------------------------------------------------------------
# titling the pages of one superclass
# --------------------------------------------------------------------------

# Longest page title worth printing before the rest becomes "+N more", and
# longest run of qualifiers inside one pathway's parentheses.
_MAX_TITLE = 80
_MAX_QUALIFIERS = 48


def split_piece(name):
    """("Transport, Inner Membrane", "amino acids") from a piece's caption."""
    family, _, qualifier = str(name).partition(": ")
    return re.sub(r" \(\d+\)$", "", family), qualifier


def _page_text(families, on_pages, brief):
    parts = []
    for family, (_, qualifiers) in sorted(
            families.items(), key=lambda kv: (-kv[1][0], kv[0])):
        # The page is already captioned "... metabolism"; saying it again for
        # every pathway on it only pushes the next one off the end. Not for
        # "Inorganic Ion Transport and Metabolism", where it is half the name.
        short = re.sub(r"(?<!\band)\s+metabolism$", "", family, flags=re.IGNORECASE)
        if on_pages[family] > 1 and qualifiers:
            shown = qualifiers
            if brief:
                # "lipids, Estrone 3-sulfate; lipids 2" -> "lipids"
                shown = list(dict.fromkeys(
                    re.sub(r" \d+$", "", q.split(", ")[0]) for q in qualifiers))
            text, used = shown[0], 1
            for extra in shown[1:2]:
                if len(text) + 2 + len(extra) <= _MAX_QUALIFIERS:
                    text, used = f"{text}; {extra}", used + 1
            if len(shown) > used:
                text += f"; +{len(shown) - used}"
            parts.append(f"{short} ({text})")
        else:
            parts.append(short)
    text, used = parts[0], 1
    for part in parts[1:]:
        if len(text) + 2 + len(part) > _MAX_TITLE:
            break
        text, used = f"{text}; {part}", used + 1
    if used < len(parts):
        text += f"; +{len(parts) - used} more"
    return text


def title_pages(label, pages):
    """Distinct titles for the pages one superclass is split into.

    `pages` is a list of pages, each a list of (cluster name, reaction count).
    A title names the pathways on the page, largest first: "Lipid metabolism:
    Glycerophospholipid; Membrane Lipid; +2 more". A pathway that continues
    onto other pages says which part of it this is -- "Transport, extracellular
    (lipids)" -- with landmarks only where that alone would leave two pages
    with one title. "(1)" to "(10)" said only that there were ten.
    """
    on_pages = Counter()
    contents = []
    for page in pages:
        families = {}
        for name, size in page:
            family, qualifier = split_piece(name)
            entry = families.setdefault(family, [0, []])
            entry[0] += size
            if qualifier:
                entry[1].append(qualifier)
        contents.append(families)
        on_pages.update(families.keys())

    titles = [f"{label}: {_page_text(f, on_pages, True)}" for f in contents]
    counts = Counter(titles)
    titles = [f"{label}: {_page_text(f, on_pages, False)}" if counts[t] > 1 else t
              for t, f in zip(titles, contents)]

    counts, index, out = Counter(titles), {}, []
    for title in titles:
        if counts[title] > 1:
            index[title] = index.get(title, 0) + 1
            title = f"{title} ({index[title]})"
        out.append(title)
    return out
