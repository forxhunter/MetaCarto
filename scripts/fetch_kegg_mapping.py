
import requests
import json
import os

# KEGG API Endpoint
# Link Reaction (rn) to Pathway (path)
URL = "http://rest.kegg.jp/link/pathway/rn"
OUTPUT_FILE = "data/kegg/kegg_mapping.json"
OUTPUT_DIR = "data/kegg"

def fetch_mapping():
    print(f"Fetching mapping from {URL}...")
    try:
        response = requests.get(URL)
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            return
            
        # Parse text response
        # Format: "rn:R00001\tpath:map00010"
        mapping = {}
        lines = response.text.splitlines()
        
        print(f"Parsing {len(lines)} entries...")
        
        for line in lines:
            if not line.strip(): continue
            parts = line.split('\t')
            if len(parts) < 2: continue
            
            rn_id = parts[0].replace('rn:', '')
            path_id = parts[1].replace('path:', '')
            
            # Filter generic "Metabolic pathways" (map01100) and "Biosynthesis of secondary metabolites" (map01110)
            # as they are too broad.
            if path_id in ['map01100', 'map01110', 'map01120', 'map01200', 'map01210', 'map01212', 'map01230', 'map01220']:
                continue
                
            if rn_id not in mapping:
                mapping[rn_id] = []
            
            if path_id not in mapping[rn_id]:
                mapping[rn_id].append(path_id)
                
        # 2. Fetch EC -> Pathway Mapping
        print("Fetching EC mapping from http://rest.kegg.jp/link/pathway/ec...")
        resp_ec = requests.get("http://rest.kegg.jp/link/pathway/ec")
        if resp_ec.status_code == 200:
             lines = resp_ec.text.splitlines()
             print(f"Parsing {len(lines)} EC entries...")
             for line in lines:
                 if not line.strip(): continue
                 parts = line.split('\t')
                 if len(parts) < 2: continue
                 
                 # Format: "ec:1.1.1.1\tpath:map00010"
                 ec_id = parts[0].replace('ec:', '')
                 path_id = parts[1].replace('path:', '')
                 
                 # Filter generic maps
                 if path_id in ['map01100', 'map01110', 'map01120', 'map01200', 'map01210', 'map01212', 'map01230', 'map01220']:
                     continue
                 
                 if ec_id not in mapping:
                     mapping[ec_id] = []
                 if path_id not in mapping[ec_id]:
                     mapping[ec_id].append(path_id)
        
        print(f"Total mapped IDs (Reacton + EC): {len(mapping)}")
        
        # We also need Pathway ID -> Name mapping to be useful
        # Fetch Pathway Names
        print("Fetching pathway names...")
        path_url = "http://rest.kegg.jp/list/pathway"
        resp_path = requests.get(path_url)
        
        path_names = {}
        if resp_path.status_code == 200:
             for line in resp_path.text.splitlines():
                 if not line.strip(): continue
                 parts = line.split('\t')
                 if len(parts) < 2: continue
                 pid = parts[0].replace('path:', '')
                 pname = parts[1]
                 path_names[pid] = pname
                 
        # Combine into final JSON: { "R00001": ["Glycolysis / Gluconeogenesis", "Citrate cycle..."] }
        final_mapping = {}
        unique_subsystems = set()
        
        for rn, pids in mapping.items():
            readable_paths = []
            for pid in pids:
                if pid in path_names:
                    # Clean up name "Glycolysis / Gluconeogenesis - Homo sapiens (human)" -> "Glycolysis / Gluconeogenesis"
                    # Actually KEGG pathway names often don't have organism if fetched via 'map' (generic)
                    name = path_names[pid]
                    readable_paths.append(name)
                    unique_subsystems.add(name)
            
            if readable_paths:
                final_mapping[rn] = readable_paths

        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(final_mapping, f, indent=2)
            
        print(f"Saved mapping for {len(final_mapping)} reactions to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_mapping()
