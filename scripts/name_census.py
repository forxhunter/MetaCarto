"""How the release would name its maps, without drawing any of them.

Runs the release's decomposition and `--group-function` merge over every model
and counts what a reader of the collection would see: maps filed under
"Other metabolism", the share of reactions on them, and titles that say only
"(N)". Layout is skipped, so the whole corpus takes minutes rather than an
hour; the names are a pure function of the decomposition.

    python scripts/name_census.py                 # summary
    python scripts/name_census.py --pages out.tsv # every title, per model
"""

import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join("data", "bigg", "models")


def model_paths():
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*.json"))
                   + glob.glob(os.path.join(MODEL_DIR, "*.xml")))
    seen, unique = set(), []
    for path in paths:                       # JSON first: it has subsystems
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in seen:
            seen.add(stem)
            unique.append(path)
    return unique


def census(args):
    path, max_cluster = args
    os.chdir(ROOT)
    import layout_v2
    from src.layout.compound import compute_cofactor_scores
    from src.layout.decompose import clusters

    model = layout_v2.load_model(path)
    groups = clusters(model, compute_cofactor_scores(model), max_size=max_cluster)
    pages = layout_v2.merge_by_function(groups, max_size=max_cluster)
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem, [(name, len(reactions)) for name, reactions in pages.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-cluster", type=int, default=120,
                        help="as the release is built (default 120)")
    parser.add_argument("--pages", help="write every title to this TSV")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    jobs = [(path, args.max_cluster) for path in model_paths()]
    with Pool(args.jobs) as pool:
        results = sorted(pool.map(census, jobs))

    maps = other = numbered = reactions = on_other = 0
    for _, pages in results:
        for name, size in pages:
            maps += 1
            reactions += size
            if name.startswith("Other metabolism"):
                other += 1
                on_other += size
            if re.search(r" \(\d+\)$", name):
                numbered += 1
    print(f"{len(results)} models, {maps} maps, {reactions} reactions")
    print(f"  filed under Other metabolism: {other} maps, "
          f"{on_other} reactions ({100.0 * on_other / max(1, reactions):.1f}%)")
    print(f"  titles that are only a number: {numbered}")

    if args.pages:
        with open(args.pages, "w", encoding="utf-8") as out:
            for stem, pages in results:
                for name, size in pages:
                    out.write(f"{stem}\t{size}\t{name}\n")
        print(f"wrote {args.pages}")


if __name__ == "__main__":
    main()
