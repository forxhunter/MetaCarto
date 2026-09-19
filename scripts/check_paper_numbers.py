"""Check every numeric claim in the manuscript against benchmarks/results/.

The tables are generated (`make_tables.py`), so they cannot drift. The prose
is written by hand and can: a rerun that moves a median leaves the sentence
quoting the old one, and nothing catches it. This reads the authoritative
value out of the result files, formats it the way the manuscript writes it,
and fails if that string is not in the text.

It is deliberately a string check rather than a parse. Extracting numbers from
prose is guesswork -- "0.615" could be a threshold, a median or a p-value --
whereas asserting that the number the data gives is present is exact, and the
failure message says which file the truth came from.

    python scripts/check_paper_numbers.py
    python scripts/check_paper_numbers.py --verbose
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join("benchmarks", "results")
PAPER = os.path.join(
    "bioinfomatics", "latex_template", "oxford-bioinformatics-template-master",
    "oxford-bioinformatics-template-master")


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def manuscript():
    """Everything the reader sees: prose, and the generated tables it inputs.

    The tables were excluded at first, on the reasoning that `make_tables.py
    --check` already pins them to the data. That made this report a false
    negative for any figure the paper states only in a table -- the
    corpus-wide agreement range, for one -- because it was looking for the
    number in the wrong file.
    """
    text = []
    sources = (sorted(glob.glob(os.path.join(PAPER, "paper", "*.tex")))
               + sorted(glob.glob(os.path.join(PAPER, "tables", "*.tex")))
               + [os.path.join(PAPER, "supplementary.tex")])
    for path in sources:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            text.append(handle.read())
    joined = "\n".join(text)
    # Collapse LaTeX's thin-space digit grouping and line breaks so that
    # "240\,398" and a number split across a line both match plainly.
    joined = joined.replace("\\,", "").replace("\\%", "%")
    return re.sub(r"\s+", " ", joined)


def claims():
    """(name, expected string, source file) for everything worth pinning."""
    out = []

    corpus = load("corpus.json")
    if corpus:
        m = corpus["metrics"]
        out += [
            ("corpus maps", str(corpus["maps"]), "corpus.json"),
            ("corpus models", str(corpus["models"]), "corpus.json"),
            ("corpus reactions", "%d" % corpus["reactions"], "corpus.json"),
            ("median axis-aligned", "%.3f" % m["axis_aligned"]["median"],
             "corpus.json"),
            ("axis-aligned in band", "%.1f" % (100 * m["axis_aligned"]["in_band"]),
             "corpus.json"),
            ("median crossings", "%.3f" % m["crossings_per_edge"]["median"],
             "corpus.json"),
            ("median label pt", "%.1f" % m["label_pt"]["median"], "corpus.json"),
        ]

    everything = load("correctness_all.json")
    if everything and everything.get("summary", {}).get("full"):
        s = everything["summary"]
        full = s["full"]
        out.append(("KEGG agreement range",
                    "%.1f--%.1f" % (100 * full["min_rate"], 100 * full["max_rate"]),
                    "correctness_all.json"))
        out.append(("models scored", str(s["models_scored"]),
                    "correctness_all.json"))
    else:
        # Fall back to the four single-model files the paper was first written
        # against, so this still checks something before the corpus run lands.
        rates = []
        for path in glob.glob(os.path.join(RESULTS, "correctness_i*.json")):
            with open(path, encoding="utf-8") as handle:
                rates.append(json.load(handle)["variants"]["full"]["rate"])
        if rates:
            out.append(("KEGG agreement range",
                        "%.1f--%.1f" % (100 * min(rates), 100 * max(rates)),
                        "correctness_*.json"))

    summary = load("baseline_summary.json")
    if summary:
        counts = summary["counts"]
        out += [
            ("baseline better", str(counts["better"]), "baseline_summary.json"),
            ("baseline worse", str(counts["worse"]), "baseline_summary.json"),
            ("baseline n.s.", str(counts["not_significant"]),
             "baseline_summary.json"),
            ("baseline maps", str(summary["metrics"]["axis_aligned"]["dot"]["pairs"]),
             "baseline_summary.json"),
        ]
        pool = []
        for path in glob.glob(os.path.join(RESULTS, "baselines_*.json")):
            with open(path, encoding="utf-8") as handle:
                for row in json.load(handle)["methods"]["metacarto"]["per_map"]:
                    pool.append(row["axis_aligned"])
        if pool:
            out.append(("benchmark axis-aligned median",
                        "%.3f" % statistics.median(pool), "baselines_*.json"))

    scaling = load("scaling.json")
    if scaling and "Recon3D" in scaling["models"]:
        r = scaling["models"]["Recon3D"]
        total = r["load_s"] + r["decompose_s"] + r["draw_total_s"]
        out += [
            ("Recon3D total seconds", "%d" % round(total), "scaling.json"),
            ("Recon3D decompose", "%.1f" % r["decompose_s"], "scaling.json"),
            ("Recon3D draw", "%.1f" % r["draw_total_s"], "scaling.json"),
            ("Recon3D maps", str(r["maps"]), "scaling.json"),
        ]

    calib = load("kegg_calibration.json")
    if calib:
        out.append(("curated KEGG drawings", str(calib["kegg_pathways_scored"]),
                    "kegg_calibration.json"))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    text = manuscript()
    bad = 0
    for name, expected, source in claims():
        present = expected in text
        if not present:
            bad += 1
        if args.verbose or not present:
            print("%-32s %-16s %-26s %s"
                  % (name, expected, source, "ok" if present else "NOT IN PAPER"))
    if bad:
        print("\n%d claim(s) in benchmarks/results/ do not appear in the text" % bad)
        return 1
    print("every checked number in benchmarks/results/ appears in the manuscript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
