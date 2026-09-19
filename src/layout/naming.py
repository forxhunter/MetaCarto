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


STRUCTURAL_PREFIXES = ("Unannotated", "Other", "Uncategorized", "Cluster_")


def name_clusters(groups, model, prefixes=STRUCTURAL_PREFIXES):
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
        if not str(name).startswith(prefixes):
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
