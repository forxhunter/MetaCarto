"""Generate the manuscript's tables from benchmarks/results/.

The tables were typed by hand from readings of the result files, which is the
arrangement that lets a number in the paper drift away from the number in the
data. These are emitted instead, as bare `tabular` environments that the paper
wraps in its own caption and float, so the numbers cannot disagree with the
run that produced them.

    python scripts/make_tables.py          # writes tables/*.tex next to main.tex
    python scripts/make_tables.py --check  # non-zero if a table is out of date
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join("benchmarks", "results")
PAPER = os.path.join(
    "bioinfomatics", "latex_template", "oxford-bioinformatics-template-master",
    "oxford-bioinformatics-template-master")
TABLES = os.path.join(PAPER, "tables")

# Reported in this order throughout, largest curated intersection first.
MODELS = ("iJO1366", "iMM904", "iYO844", "iAF692")

BASELINES = (("dot", r"\textsc{dot}"), ("neato", r"\textsc{neato}"),
             ("fdp", r"\textsc{fdp}"), ("spring", "spring"),
             ("kamada_kawai", "KK"))

METRIC_LABELS = (
    ("crossings_per_graph_edge", "Crossings/edge"),
    ("axis_aligned", "Axis-aligned"),
    ("hairball_index", "Local density"),
    ("aspect_ratio", "Aspect ratio"),
    ("occupancy", "Occupancy"),
    ("min_separation_ratio", "Separation"),
)


# Returned by a builder whose inputs are not ready yet -- the corpus
# correctness run writes its summary only when it finishes, and a partial
# table is worse than no table.
SKIPPED = object()


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(os.path.join(RESULTS, name), encoding="utf-8") as handle:
        return json.load(handle)


def rows(*lines):
    return "\n".join(lines) + "\n"


def correctness_table():
    body = []
    for model in MODELS:
        d = load("correctness_%s.json" % model)
        v = d["variants"]
        full = v["full"]
        cells = [model, "%d" % full["considered"], "%.1f" % (100 * full["rate"])]
        for variant in ("no_cofactor_tiering", "no_cofactor_handling"):
            cells.append("%.1f" % (100 * v[variant]["rate"])
                         if variant in v else "--")
        body.append(" & ".join(cells) + r" \\")
    return rows(
        r"\begin{tabular}{@{}lrrrr@{}}\toprule",
        r"Model & Scored & Full & $-$tier & $-$cof. \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


def baselines_table():
    summary = load("baseline_summary.json")
    pooled = pooled_medians()
    body = []
    for key, label in METRIC_LABELS:
        cells = [label, "%.3f" % pooled[key]]
        stats = summary["metrics"][key]
        for name, _pretty in BASELINES:
            s = stats[name]
            if s["verdict"] == "n.s.":
                cells.append(r"\textit{n.s.}")
            else:
                r = s["rank_biserial"]
                cells.append("$%s$%.2f" % ("+" if r >= 0 else "-", abs(r)))
        body.append(" & ".join(cells) + r" \\")
    header = " & ".join(["Metric", r"\mytool"] + [p for _n, p in BASELINES])
    return rows(
        r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


def pooled_medians():
    """MetaCarto's median per metric over every map in the baseline corpus."""
    import statistics
    pool = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "baselines_*.json"))):
        with open(path, encoding="utf-8") as handle:
            d = json.load(handle)
        for row in d["methods"]["metacarto"]["per_map"]:
            for key, _label in METRIC_LABELS:
                if key in row:
                    pool.setdefault(key, []).append(row[key])
    return {k: statistics.median(v) for k, v in pool.items()}


def metdraw_table():
    body = []
    for model in MODELS:
        d = load("correctness_%s.json" % model)
        rate = 100 * d["variants"]["full"]["rate"]
        body.append(r"%s & \mytool & %.1f & %.1f & 1.00 \\" % (model, rate, rate))
        md = d.get("metdraw")
        if not md:
            continue
        rec = [100 * md[k]["recall"] for k in ("p80", "p90", "p95")]
        pre = [100 * md[k]["precision"] for k in ("p80", "p90", "p95")]
        epr = [md[k]["edges_per_reaction"] for k in ("p80", "p90", "p95")]
        body.append(
            r"        & clone p80--p95 & %.1f--%.1f & %.1f--%.1f & %.2f--%.2f \\"
            % (min(rec), max(rec), min(pre), max(pre), min(epr), max(epr)))
    return rows(
        r"\begin{tabular}{@{}llrrr@{}}\toprule",
        r"Model & Method & Recall & Precision & Edges/rxn \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


def scaling_table():
    d = load("scaling.json")["models"]
    pretty = {"e_coli_core": r"\textit{E.~coli} core"}
    body = []
    for name, v in d.items():
        body.append(r"%s & %s & %d & %.1f & %.1f & %.1f \\"
                    % (pretty.get(name, name),
                       "10\\,600" if v["reactions"] > 9999 else v["reactions"],
                       v["maps"], v["load_s"], v["decompose_s"],
                       v["draw_total_s"]))
    return rows(
        r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
        r"Model & Reactions & Maps & Load & Decompose & Draw \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


def ablations_table():
    """Every ablation cell, for the supplement."""
    cells = {}
    models = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "ablations_*.json"))):
        with open(path, encoding="utf-8") as handle:
            d = json.load(handle)
        models.append(d["model"])
        for c in d["cells"]:
            cells[(c["variant"], c["metric"], d["model"])] = c
    variants = sorted({k[0] for k in cells})
    width = 1 + len(models)          # label column plus one per model
    body = []
    for variant in variants:
        body.append(r"\multicolumn{%d}{@{}l}{\texttt{%s}} \\"
                    % (width, variant.replace("_", r"\_")))
        for key, label in METRIC_LABELS:
            if not any((variant, key, m) in cells for m in models):
                continue
            line = ["~~" + label]
            for model in models:
                c = cells.get((variant, key, model))
                if c is None:
                    line.append("--")
                    continue
                p = c["p_holm"]
                mark = r"$^{*}$" if (p is not None and p < 0.05) else ""
                line.append("%.3f%s" % (c["ablated"], mark))
            body.append(" & ".join(line) + r" \\")
    header = " & ".join(["Variant / metric"] + models)
    return rows(
        r"\begin{tabular}{@{}l" + "r" * len(models) + r"@{}}\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


def per_model_table():
    """Median per metric, per method, per model -- the detail behind Table 2."""
    order = ["metacarto"] + [n for n, _p in BASELINES]
    pretty = dict(BASELINES)
    pretty["metacarto"] = r"\mytool"
    body = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "baselines_*.json"))):
        with open(path, encoding="utf-8") as handle:
            d = json.load(handle)
        body.append(r"\multicolumn{7}{@{}l}{%s, %d maps} \\"
                    % (d["model"], d["clusters_drawn"]))
        for method in order:
            stats = d["methods"].get(method, {}).get("metrics", {})
            line = ["~~" + pretty[method]]
            for key, _label in METRIC_LABELS:
                s = stats.get(key)
                line.append("%.3f" % s["median"] if s else "--")
            body.append(" & ".join(line) + r" \\")
    header = " & ".join(["Method"] + [l for _k, l in METRIC_LABELS])
    return rows(
        r"\begin{tabular}{@{}l" + "r" * len(METRIC_LABELS) + r"@{}}\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


CORPUS_LABELS = (
    ("axis_aligned", "Axis-aligned fraction"),
    ("occupancy", "Occupancy"),
    ("aspect_ratio", "Aspect ratio"),
    ("label_overlaps", "Label--label overlaps"),
    ("label_on_edge", "Label--edge overlaps"),
    ("label_on_node", "Label--node overlaps"),
    ("crossings_per_edge", "Crossings per edge"),
    ("hairball_index", "Local density"),
    ("min_separation_ratio", "Node separation"),
    ("longest_run_ratio", "Longest straight run"),
    ("label_pt", "Label size at 180\\,mm (pt)"),
)


def corpus_table():
    """Every metric over every published map, ordered by how well it does.

    Ordered worst-last rather than grouped, because the point of reporting all
    eleven is that the four at the bottom are visible: a table that stops after
    the ones that pass reads as a pass.
    """
    d = load("corpus.json")
    m = d["metrics"]
    body = []
    for key, label in CORPUS_LABELS:
        s = m[key]
        share = s.get("in_band")
        body.append("%s & %.3f & %.3f & %.3f & %s \\\\"
                    % (label, s["p10"], s["median"], s["p90"],
                       "%.1f" % (100 * share) if share is not None else "--"))
    return rows(
        r"\begin{tabular}{@{}lrrrr@{}}\toprule",
        r"Metric & 10th & Median & 90th & In band (\%) \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


def correctness_corpus_table():
    """KEGG agreement over every model that can be scored, not four of them.

    Pooled and median are both given because they answer different questions:
    pooled counts every scorable reaction once, so the large well-annotated
    models dominate it, while the median weights every model equally. A model
    contributing a handful of scorable reactions is excluded rather than
    averaged in, and the count of those is in the caption.
    """
    d = load("correctness_all.json") or {}
    # The corpus run writes per_model incrementally and the summary
    # only at the end, so an in-progress file has no summary yet.
    s = d.get("summary") or {}
    if not s.get("full"):
        return SKIPPED
    rows_out = []
    labels = (("full", "\\mytool"),
              ("no_cofactor_tiering", "~~$-$ carrier tier"),
              ("no_cofactor_handling", "~~$-$ cofactor handling"))
    for key, label in labels:
        v = s.get(key)
        if not v:
            continue
        rows_out.append("%s & %.1f & %.1f & %.1f--%.1f \\\\"
                        % (label, 100 * v["pooled_rate"], 100 * v["median_rate"],
                           100 * v["min_rate"], 100 * v["max_rate"]))
    return rows(
        r"\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Variant & Pooled & Median & Range \\",
        r"\midrule",
        *rows_out,
        r"\botrule",
        r"\end{tabular}")


def metdraw_corpus_table():
    """Connectivity cloning over the same corpus, at three thresholds.

    Separate from the agreement table because the quantities are different:
    \\mytool draws one edge so its precision and recall are the same number,
    while cloning trades one against the other and the trade is the point.
    """
    d = load("correctness_all.json") or {}
    # The corpus run writes per_model incrementally and the summary
    # only at the end, so an in-progress file has no summary yet.
    s = d.get("summary") or {}
    if not s.get("full"):
        return SKIPPED
    full = s["full"]
    body = [r"\mytool & %.1f & %.1f & 1.00 \\"
            % (100 * full["median_rate"], 100 * full["median_rate"])]
    for key in ("p80", "p90", "p95"):
        v = s.get("metdraw", {}).get(key)
        if not v:
            continue
        body.append("~~clone %s & %.1f & %.1f & %.2f \\\\"
                    % (key, 100 * v["recall_median"],
                       100 * v["precision_median"], v["edges_median"]))
    return rows(
        r"\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Method & Recall & Precision & Edges/rxn \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


def control_table():
    """What chance scores on the agreement test, next to what MetaCarto scores.

    Agreement is membership in a reference set, so the rate means nothing
    without the size of that set and the size of the space the choice was made
    from. Both are here, and so is the rate a uniform random choice would get.
    """
    d = load("correctness_control.json")["models"]
    sens = (load("correctness_sensitivity.json") or {}).get("models", {})
    strict_variants = ("intersection", "majority", "one_to_one")
    body = []
    for model in MODELS:
        s = d.get(model)
        if not s:
            continue
        # The strict column is the worst of the stricter reference definitions
        # for this model, so the table states the floor rather than the one
        # that happens to look best.
        rates = [sens.get(model, {}).get(v, {}).get("rate")
                 for v in strict_variants]
        rates = [r for r in rates if r is not None]
        strict = "%.1f" % (100 * min(rates)) if rates else "--"
        body.append(
            "%s & %d & %d & %.1f & %.1f & %s \\\\"
            % (model, s["reactions"], s["candidates_median"],
               100 * s["chance_mean"], 100 * s["agreement"], strict))
    return rows(
        r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
        r"Model & Scored & Cand. & Random & \mytool & Strict ref. \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


def thresholds_table():
    """The acceptance band per metric, and the curated band where one exists.

    Read straight out of `metrics.TARGETS` and `metrics.CURATED_REFERENCE` so
    the paper cannot state a threshold the code does not enforce. The third
    column is empty for the metrics that do not transfer to KEGG drawings.
    """
    from src.layout import metrics

    def num(v):
        # Three decimals throughout: the curated crossings p90 is 0.115 and
        # the gate is 0.05, and rounding either to two makes that comparison
        # unreadable. Counts stay integers.
        return "%d" % v if float(v).is_integer() and abs(v) < 1 else "%.3f" % v

    def band(pair):
        if pair is None:
            return "--"
        low, high = pair
        if low is not None and high is not None:
            return "%s--%s" % (num(low), num(high))
        if low is not None:
            return "$\\geq$ %s" % num(low)
        return "$\\leq$ %s" % num(high)

    body = []
    for key, label in CORPUS_LABELS:
        if key == "label_pt":
            body.append("%s & $\\geq$ 5 & -- \\\\" % label)
            continue
        body.append("%s & %s & %s \\\\"
                    % (label, band(metrics.TARGETS.get(key)),
                       band(metrics.CURATED_REFERENCE.get(key))))
    return rows(
        r"\begin{tabular}{@{}lrr@{}}\toprule",
        r"Metric & Acceptance band & Curated 10th--90th \\",
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular}")


BUILDERS = {
    "thresholds": thresholds_table,
    "control": control_table,
    "corpus": corpus_table,
    "correctness": correctness_table,
    "correctness_corpus": correctness_corpus_table,
    "metdraw_corpus": metdraw_corpus_table,
    "baselines": baselines_table,
    "metdraw": metdraw_table,
    "scaling": scaling_table,
    "ablations_full": ablations_table,
    "per_model": per_model_table,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=TABLES)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any table differs from disk")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    stale = []
    for name, build in sorted(BUILDERS.items()):
        text = build()
        if text is SKIPPED:
            print("%-20s skipped (inputs not ready)" % name)
            continue
        path = os.path.join(args.out, name + ".tex")
        old = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                old = handle.read()
        if args.check:
            if old != text:
                stale.append(name)
            continue
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("%-16s %s" % (name, "unchanged" if old == text else "written"))

    if args.check:
        if stale:
            print("out of date: %s" % ", ".join(stale))
            return 1
        print("all tables match benchmarks/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
