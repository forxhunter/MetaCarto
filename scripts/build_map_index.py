"""Build the map library index consumed by the Escher map picker.

The Escher app cannot list a repository's contents on its own, and bundling a
few thousand map files into the app would make every deploy enormous. So the
maps stay in `escher_maps_BiGG` and this writes indexes next to them, which the
app fetches at runtime. Regenerating the maps and re-running this is enough to
update the picker -- the app does not have to be rebuilt.

The index is two-level on purpose. A flat index of every map across 109 models
is ~9 MB, which is not something to download before the picker can open. So
`map_index.json` at the root lists only the models (a few KB) and each model
gets its own `<model>/model_index.json`, fetched when that model is selected.

    python scripts/build_map_index.py                       # default paths
    python scripts/build_map_index.py --root escher_maps_BiGG
"""

import argparse
import json
import os
import sys
from datetime import date

DEFAULT_ROOT = "escher_maps_BiGG"
# Empty means "resolve map paths relative to wherever this index was fetched
# from". That keeps one index file working unchanged whether it is served from
# raw.githubusercontent, a local http.server during development, or a mirror.
DEFAULT_BASE_URL = ""
INDEX_NAME = "map_index.json"
MODEL_INDEX_NAME = "model_index.json"


def describe(path):
    """(reaction count, node count, map name) for one Escher map, or None."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        header, body = data[0], data[1]
    except Exception:
        return None
    if not isinstance(body, dict) or "reactions" not in body:
        return None
    return (
        len(body.get("reactions", {})),
        len(body.get("nodes", {})),
        header.get("map_name") or os.path.splitext(os.path.basename(path))[0],
    )


def build(root, base_url):
    models = []
    for model_id in sorted(os.listdir(root)):
        model_dir = os.path.join(root, model_id)
        if not os.path.isdir(model_dir) or model_id.startswith("."):
            continue

        maps = []
        for name in sorted(os.listdir(model_dir)):
            if not name.endswith(".json") or name in (INDEX_NAME, MODEL_INDEX_NAME):
                continue
            described = describe(os.path.join(model_dir, name))
            if described is None:
                continue
            reactions, nodes, map_name = described
            maps.append({
                "name": map_name,
                "file": name,
                "path": f"{model_id}/{name}",
                "reactions": reactions,
                "nodes": nodes,
                # The whole-model map is what a user most often wants first.
                "combined": name.endswith("_Combined.json"),
            })

        if not maps:
            continue

        maps.sort(key=lambda m: (not m["combined"], -m["reactions"], m["name"]))
        with open(os.path.join(model_dir, MODEL_INDEX_NAME), "w", encoding="utf-8") as handle:
            json.dump({"schema": 1, "id": model_id, "maps": maps}, handle, indent=1)

        models.append({
            "id": model_id,
            "index": f"{model_id}/{MODEL_INDEX_NAME}",
            "map_count": len(maps),
            "reactions": sum(m["reactions"] for m in maps),
        })

    return {
        "schema": 1,
        "generated": date.today().isoformat(),
        "base_url": base_url,
        "models": models,
        "map_count": sum(m["map_count"] for m in models),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="directory of {model}/{map}.json (default: %(default)s)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="absolute URL to resolve map paths against; empty (the default) resolves them relative to the index itself")
    parser.add_argument("--out", default=None,
                        help="output file (default: <root>/map_index.json)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.root):
        parser.error(f"not a directory: {args.root}")

    index = build(args.root, args.base_url)
    out = args.out or os.path.join(args.root, INDEX_NAME)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=1)

    size_kb = os.path.getsize(out) / 1024.0
    print(f"{out}: {len(index['models'])} models, {index['map_count']} maps, {size_kb:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
