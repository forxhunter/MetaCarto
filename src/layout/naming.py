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
}

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

    scored = []
    for phrase, count in counts.items():
        share = count / len(reactions)
        if share < 0.3 or count < 2:
            continue
        idf = math.log((total + 1) / (frequency.get(phrase, 0) + 1))
        words = phrase.count(" ") + 1
        # A longer phrase that is still characteristic is more informative than
        # a single word with the same coverage.
        scored.append((share * idf * (1.0 + 0.6 * (words - 1)), phrase))

    if not scored:
        return None
    scored.sort(reverse=True)

    best_score, best = scored[0]
    if best_score < 0.25:
        return None

    # A second, non-overlapping phrase adds specificity when there is one.
    parts = [best]
    for score, phrase in scored[1:]:
        if score < best_score * 0.55:
            break
        if any(word in set(best.split()) for word in phrase.split()):
            continue
        parts.append(phrase)
        break

    return " ".join(part.capitalize() for part in parts)


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
