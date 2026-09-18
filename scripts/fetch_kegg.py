import os
import argparse
from bioservices import KEGG

def download_kegg_pathway(pathway_id, output_dir):
    """Downloads a generic KGML file for a given pathway ID."""
    s = KEGG()
    print(f"Fetching {pathway_id}...")
    xml_data = s.get(pathway_id, "kgml")
    
    if xml_data == 404 or not xml_data:
        print(f"Error: Pathway {pathway_id} not found.")
        return

    output_path = os.path.join(output_dir, f"{pathway_id}.xml")
    with open(output_path, "w") as f:
        f.write(xml_data)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download KEGG KGML files.")
    parser.add_argument("--ids", nargs="+", help="List of pathway IDs (e.g., map00010)")
    parser.add_argument("--orgs", nargs="+", help="List of organism codes (e.g., 'eco hsa sce') to download ALL pathways for.")
    parser.add_argument("--out", help="Output directory", default="data/kegg")
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    ids_to_fetch = []
    if args.ids:
        ids_to_fetch.extend(args.ids)
        
    if args.orgs:
        s = KEGG()
        for org in args.orgs:
            print(f"Fetching pathway list for {org}...")
            # KEGG API returns string like "path:eco00010\tGlycolysis..."
            pathways = s.list("pathway", org)
            if hasattr(pathways, 'splitlines'): # Check if valid response
                for line in pathways.splitlines():
                    if line.strip():
                        # extract 'eco00010' from 'path:eco00010'
                        pid = line.split('\t')[0].replace('path:', '')
                        ids_to_fetch.append(pid)
            else:
                print(f"Failed to list pathways for {org}.")
            
    print(f"Found {len(ids_to_fetch)} pathways to download.")
    for pid in ids_to_fetch:
        download_kegg_pathway(pid, args.out)
