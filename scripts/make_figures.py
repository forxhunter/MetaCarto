"""Regenerate every manuscript figure from the published maps.

A figure that cannot be regenerated is a figure whose caption cannot be
checked, so each one is built here from a map in `data/bigg/` -- the same file
the released collection serves -- rather than exported by hand from a viewer.

Output is vector PDF at the placed print width, so the label sizes quoted in
the captions are the sizes on the page. PNG is written alongside for drafts
and for the repository README.

    python scripts/make_figures.py                # all figures
    python scripts/make_figures.py --only fig1
    python scripts/make_figures.py --list         # what each figure shows
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layout import metrics, pdfout, preview, svgout
from src.layout.render import (ESCHER_DEFAULT_FONT_BASE,
                               METABOLITE_FONT_FACTOR, REACTION_FONT_FACTOR)

# Measured from bioinfo.cls rather than assumed: the text block is 488.295pt
# wide (172.3 mm), a column is 235.15pt (82.9 mm) and the text height is 681pt
# (240.2 mm). A float also has to carry its caption, so a full-page figure
# gets about 220 mm of height, not the whole text block.
COLUMN_MM = 82.9
FULL_MM = 172.3
MAX_HEIGHT_MM = 220.0

# Figures are written where LaTeX reads them, so there is one copy rather than
# a source copy and a stale one next to main.tex.
PAPER_FIGURES = os.path.join(
    "bioinfomatics", "latex_template", "oxford-bioinformatics-template-master",
    "oxford-bioinformatics-template-master", "figures")


# fig2 is a screenshot of the map library in the browser viewer, so it is
# captured by hand rather than built here. It is placed at the full text width
# (172.3 mm), which sets two requirements the capture has to meet:
#
#   legibility  the interface type must survive the reduction. At 83 mm it
#               sets at about 4 pt, which is why the figure is full width.
#   resolution  >= 300 dpi at 172.3 mm needs >= 2035 px across; a 762 px
#               capture is 117 dpi there and prints soft.
#
# A capture at 2x device pixel ratio, or a browser window near 2600 px wide,
# satisfies both.
FIGURE2_MIN_PX = 2035

FIGURES = {
    "fig1": {
        "model": "Recon3D",
        "map": "Terpenoid_and_polyketide_metabolism",
        # Placed at 166 mm so that figure plus caption fit one page; this
        # must match the \includegraphics width in implementation.tex.
        "width_mm": 166.0,
        "what": (
            "The mevalonate pathway of Recon3D (10,600 reactions), drawn "
            "automatically. Peroxisomal (left) and cytosolic (right) isoforms "
            "are placed as two parallel backbones without being told they are "
            "the same pathway; cofactors leave each reaction as side stubs, "
            "with ATP/ADP banked on one side."),
    },
}


def agreement_chart(out_dir, tag="fig3", width_mm=COLUMN_MM):
    """Per-model KEGG agreement for the full method and its two ablations.

    An empirical cumulative distribution rather than a bar chart: there are 104
    models, the question is how tightly they cluster and how long the low tail
    is, and a mean with an error bar answers neither. Reading across at 0.5
    gives the median; the horizontal gap between curves is the effect of the
    ablation at every quantile, not only at the centre.
    """
    from src.layout.chart import Chart, ecdf

    with open(os.path.join("benchmarks", "results",
                           "correctness_all.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    floor = data["summary"]["min_scorable"]
    models = [r for r in data["per_model"].values()
              if r.get("scored", 0) >= floor]

    series = (
        ("full", "MetaCarto", "#1b3a5c", None),
        ("no_cofactor_tiering", "without carrier tier", "#c0561f", (2.4, 1.6)),
        ("no_cofactor_handling", "without cofactor handling", "#8a8f96",
         (0.9, 1.6)),
    )

    chart = Chart(width_mm, 0.74 * width_mm)
    chart.xlim = (35.0, 100.0)
    chart.ylim = (0.0, 1.0)
    chart.axes(xticks=(40, 55, 70, 85, 100), yticks=(0, 0.25, 0.5, 0.75, 1.0),
               xlabel="agreement with curated reference (%)",
               ylabel="fraction of models",
               yfmt="%.2f")

    for key, _label, colour, dash in series:
        values = [100.0 * r["variants"][key]["rate"] for r in models]
        chart.polyline(ecdf(values), colour, 1.2, dash)
        ordered = sorted(values)
        chart.dot(ordered[len(ordered) // 2], 0.5, colour)

    chart.legend([(label, colour, dash) for _k, label, colour, dash in series],
                 chart.ml + 6.0, chart.mb + chart.plot_h - 6.0)

    stem = os.path.join(out_dir, tag)
    chart.save(stem + ".pdf")

    full = sorted(100.0 * r["variants"]["full"]["rate"] for r in models)
    print("%s  KEGG agreement across %d models" % (tag, len(models)))
    print("    median %.1f%%, quartiles %.1f--%.1f%%, lowest %.1f%%"
          % (full[len(full) // 2], full[len(full) // 4],
             full[3 * len(full) // 4], full[0]))
    print("    placed %.0f mm wide" % width_mm)
    return stem


def load(model, name, root=os.path.join("data", "bigg")):
    path = os.path.join(root, model, name + ".json")
    if not os.path.exists(path):
        raise SystemExit("no such map: %s" % path)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def describe(doc, width_mm):
    """Placed size and label sizes, so a caption can state them."""
    body = doc[1]
    box = pdfout.content_box(body)
    scale = (width_mm * pdfout.PT_PER_MM) / box["width"]
    height_mm = box["height"] * width_mm / box["width"]
    if height_mm > MAX_HEIGHT_MM:
        scale *= MAX_HEIGHT_MM / height_mm
        width_mm = width_mm * MAX_HEIGHT_MM / height_mm
        height_mm = MAX_HEIGHT_MM
    base = ESCHER_DEFAULT_FONT_BASE * scale
    return {
        "reactions": len(body["reactions"]),
        "metabolites": sum(1 for n in body["nodes"].values()
                           if n["node_type"] == "metabolite"),
        "width_mm": width_mm,
        "height_mm": height_mm,
        "metabolite_pt": base * METABOLITE_FONT_FACTOR,
        "reaction_pt": base * REACTION_FONT_FACTOR,
    }


def build(tag, spec, out_dir):
    doc = load(spec["model"], spec["map"])
    stem = os.path.join(out_dir, tag)
    pdfout.render(doc, stem + ".pdf", width_mm=spec["width_mm"],
                  max_height_mm=MAX_HEIGHT_MM)
    svgout.render(doc, stem + ".svg")
    preview.render(doc, stem + ".png", max_pixels=2126)

    facts = describe(doc, spec["width_mm"])
    scored = metrics.score(doc)
    print("%s  %s / %s" % (tag, spec["model"], spec["map"]))
    print("    %d reactions, %d metabolites, placed %.0f x %.0f mm"
          % (facts["reactions"], facts["metabolites"],
             facts["width_mm"], facts["height_mm"]))
    print("    labels: %.1f pt reaction, %.1f pt metabolite"
          % (facts["reaction_pt"], facts["metabolite_pt"]))
    print("    crossings/edge %.3f, axis-aligned %.3f, label collisions %d/%d/%d"
          % (scored["crossings_per_edge"], scored["axis_aligned"],
             scored["label_overlaps"], scored["label_on_node"],
             scored["label_on_edge"]))
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=PAPER_FIGURES)
    parser.add_argument("--only", help="build just this figure, e.g. fig1")
    parser.add_argument("--list", action="store_true",
                        help="print what each figure shows and exit")
    args = parser.parse_args(argv)

    if args.list:
        for tag, spec in sorted(FIGURES.items()):
            print("%s  %s / %s\n    %s\n"
                  % (tag, spec["model"], spec["map"], spec["what"]))
        return 0

    os.makedirs(args.out, exist_ok=True)
    wanted = [args.only] if args.only else sorted(FIGURES)
    for tag in wanted:
        if tag not in FIGURES:
            raise SystemExit("unknown figure %r; have %s"
                             % (tag, ", ".join(sorted(FIGURES))))
        build(tag, FIGURES[tag], args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
