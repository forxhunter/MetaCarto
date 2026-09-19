"""Render the published map corpus to SVG (and optionally PDF/PNG).

`layout_v2.py --preview` writes these alongside the JSON, but only for the run
that produced them. Re-running the whole pipeline to add a format means
recomputing every layout for output that is a pure function of the JSON
already on disk, so this reads the maps back and renders them instead. On the
full corpus that is minutes rather than an hour.

SVG is the format that ships with the JSON: it is vector, it is a third the
size of the PNG, and a browser opens it without a viewer. PDF and PNG are
available but off by default -- PNG in particular is the largest of the three
and adds most of a gigabyte to the collection.

    python scripts/render_corpus.py                      # SVG for every map
    python scripts/render_corpus.py --model Recon3D
    python scripts/render_corpus.py --formats svg,pdf
    python scripts/render_corpus.py --force              # ignore timestamps
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layout import pdfout, preview, svgout

SKIP_DIRS = {"models", "test_model"}

# Placed width for the vector formats, in millimetres. 180 is the full text
# width of a journal page, which is what makes the PDF usable as a figure
# without rescaling.
WIDTH_MM = 180.0


def renderers(names):
    out = []
    for name in names:
        if name == "svg":
            out.append(("svg", lambda doc, path: svgout.render(doc, path)))
        elif name == "pdf":
            out.append(("pdf", lambda doc, path: pdfout.render(
                doc, path, width_mm=WIDTH_MM)))
        elif name == "png":
            out.append(("png", lambda doc, path: preview.render(
                doc, path, max_pixels=2126)))
        else:
            raise SystemExit("unknown format %r (svg, pdf, png)" % name)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join("data", "bigg"))
    parser.add_argument("--model", help="one model directory, not all of them")
    parser.add_argument("--formats", default="svg")
    parser.add_argument("--force", action="store_true",
                        help="re-render even when the output is up to date")
    args = parser.parse_args(argv)

    jobs = renderers([f.strip() for f in args.formats.split(",") if f.strip()])
    pattern = os.path.join(args.root, args.model or "*", "*.json")

    written = skipped = failed = 0
    bytes_out = 0
    largest = (0, "")
    started = time.time()
    for source in sorted(glob.glob(pattern)):
        model = os.path.basename(os.path.dirname(source))
        if model in SKIP_DIRS:
            continue
        stem = source[:-len(".json")]
        wanted = [(ext, fn) for ext, fn in jobs
                  if args.force or not _fresh(stem + "." + ext, source)]
        if not wanted:
            skipped += 1
            continue
        try:
            with open(source, encoding="utf-8") as handle:
                doc = json.load(handle)
        except Exception as exc:                       # noqa: BLE001
            print("FAILED to read %s: %s" % (source, exc))
            failed += 1
            continue
        if not isinstance(doc, list) or len(doc) < 2:
            continue
        for ext, render in wanted:
            target = stem + "." + ext
            try:
                render(doc, target)
            except Exception as exc:                   # noqa: BLE001
                print("FAILED %s: %s" % (target, exc))
                failed += 1
                continue
            size = os.path.getsize(target)
            bytes_out += size
            if size > largest[0]:
                largest = (size, target)
            written += 1

    elapsed = time.time() - started
    print("wrote %d files (%.0f MB) in %.0f s; %d maps already current, %d failed"
          % (written, bytes_out / 1e6, elapsed, skipped, failed))
    if largest[1]:
        print("largest: %s at %.1f MB" % (largest[1], largest[0] / 1e6))
    # GitHub rejects a single file over 100 MB outright and warns past 50.
    if largest[0] > 50e6:
        print("WARNING: %s exceeds GitHub's 50 MB advisory limit" % largest[1])
    return 1 if failed else 0


def _fresh(target, source):
    return (os.path.exists(target)
            and os.path.getmtime(target) >= os.path.getmtime(source))


if __name__ == "__main__":
    raise SystemExit(main())
