"""Download every BiGG model into data/bigg/models/.

JSON by default, because it is the only BiGG export that carries the curated
`subsystem` of each reaction. The SBML export drops it: e_coli_core.json has a
subsystem on 95 of 95 reactions and e_coli_core.xml on none, and the same
holds for iML1515 (2712 of 2712), iAF1260 and Recon3D. Built from SBML, every
model but e_coli_core looked unannotated, so the whole collection fell through
to structural clusters captioned from free text -- "Periplasm 6", "N-c" --
which the taxonomy could only file under "Other metabolism".

`--format xml` still fetches SBML for the benchmarks that read it. The layout
driver prefers .json when both exist.
"""

import argparse
import json
import os
import time

import requests

BASE_URL = "https://bigg.ucsd.edu/api/v2"
STATIC_URL = "https://bigg.ucsd.edu/static/models"
OUTPUT_DIR = "data/bigg/models"


def _valid(content, fmt):
    # A redirect or error page arrives as HTML with status 200 often enough
    # that a successful GET is not evidence of a model.
    if fmt == "json":
        try:
            return bool(json.loads(content).get("reactions"))
        except ValueError:
            return False
    return content.lstrip().startswith(b"<?xml") and b"<sbml" in content[:4096]


def fetch_all_models(fmt="json"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching model list from BiGG...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        response.raise_for_status()
        models = response.json()["results"]
    except Exception as e:
        print(f"Failed to fetch model list: {e}")
        return

    print(f"Found {len(models)} models. Starting download...")
    failed = []
    for i, m in enumerate(models):
        bigg_id = m["bigg_id"]
        file_path = os.path.join(OUTPUT_DIR, f"{bigg_id}.{fmt}")
        prefix = f"[{i+1}/{len(models)}] {bigg_id}"

        if os.path.exists(file_path):
            print(f"{prefix} already exists. Skipping.")
            continue

        print(f"{prefix} downloading...", end=" ", flush=True)
        try:
            res = requests.get(f"{STATIC_URL}/{bigg_id}.{fmt}")
            if res.status_code == 200 and _valid(res.content, fmt):
                with open(file_path, "wb") as f:
                    f.write(res.content)
                print("Done.")
            else:
                print(f"Failed (status {res.status_code}, not a {fmt} model)")
                failed.append(bigg_id)
        except Exception as e:
            print(f"Error: {e}")
            failed.append(bigg_id)

        time.sleep(0.5)            # be nice to the server

    if failed:
        print(f"FAILED: {len(failed)} model(s): {' '.join(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--format", choices=("json", "xml"), default="json")
    fetch_all_models(parser.parse_args().format)
