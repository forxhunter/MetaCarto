"""Sync generated maps into the publishable `escher_maps_BiGG` mirror.

Reads `data/bigg/{model_id}/*.json` -- where `layout_v2.py --all --out data/bigg`
writes -- and mirrors each model directory into the repo.

Two things this deliberately does NOT do:

* It does not write README.md. An earlier version did, clobbering the
  hand-written licence-and-citation README with a four-line stub every time
  anyone synced. The README is maintained by hand and is not generated.
* It does not leave stale maps behind. A model re-clustered under a new
  pipeline produces a different set of filenames, and a plain copy leaves the
  old ones sitting in the repo forever -- RECON1 was still publishing
  `RECON1_Uncategorized_0..11.json` from a pipeline that no longer exists.
  Each model directory is emptied of `.json` first, so what is published is
  exactly what was generated.
"""

import argparse
import os
import shutil

SOURCE_ROOT = os.path.join("data", "bigg")
REPO_NAME = "escher_maps_BiGG"
SKIP = {"models", "test_model"}

# What gets published alongside the JSON. SVG is here because the JSON is only
# readable through a viewer, while an SVG opens in any browser and stays sharp
# at any zoom -- the map collection is meant to be looked at, not only loaded.
# PNG is deliberately absent: it is three times the size of the SVG and adds
# most of a gigabyte for a worse picture. `scripts/render_corpus.py` writes
# these next to the JSON; this only copies what is there.
PUBLISHED = (".json", ".svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE_ROOT)
    parser.add_argument("--repo", default=REPO_NAME)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without touching files")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo)
    if not os.path.isdir(repo_root):
        os.makedirs(repo_root)

    copied = removed = models = 0
    by_ext = {ext: 0 for ext in PUBLISHED}
    for item in sorted(os.listdir(args.source)):
        source_dir = os.path.join(args.source, item)
        if item in SKIP or not os.path.isdir(source_dir):
            continue

        maps = sorted(f for f in os.listdir(source_dir)
                      if f.endswith(PUBLISHED))
        if not maps:
            continue

        dest_dir = os.path.join(repo_root, item)
        if os.path.isdir(dest_dir):
            for stale in os.listdir(dest_dir):
                if stale.endswith(PUBLISHED) and stale not in maps:
                    if not args.dry_run:
                        os.remove(os.path.join(dest_dir, stale))
                    removed += 1
        elif not args.dry_run:
            os.makedirs(dest_dir)

        for name in maps:
            if not args.dry_run:
                shutil.copy2(os.path.join(source_dir, name),
                             os.path.join(dest_dir, name))
            copied += 1
            by_ext[os.path.splitext(name)[1]] += 1
        models += 1

    verb = "would publish" if args.dry_run else "published"
    # Counted per extension, because "maps" and "files" stopped being the same
    # number once SVG joined the JSON and a combined count reads as twice the
    # corpus.
    detail = ", ".join("%d %s" % (n, ext.lstrip("."))
                       for ext, n in sorted(by_ext.items()) if n)
    print(f"{verb} {detail} across {models} models into {repo_root}")
    if removed:
        print(f"{'would remove' if args.dry_run else 'removed'} {removed} stale files")
    print("README.md left untouched (maintained by hand)")


if __name__ == "__main__":
    main()
