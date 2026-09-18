import requests
import os
import json
import time

BASE_URL = "http://bigg.ucsd.edu/api/v2"
OUTPUT_DIR = "data/bigg/models"

def fetch_all_models():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Get List of Models
    print("Fetching model list from BiGG...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        response.raise_for_status()
        data = response.json()
        models = data['results']
    except Exception as e:
        print(f"Failed to fetch model list: {e}")
        return

    print(f"Found {len(models)} models. Starting download...")
    
    # 2. Download loop
    for i, m in enumerate(models):
        bigg_id = m['bigg_id']
        file_path = os.path.join(OUTPUT_DIR, f"{bigg_id}.xml")
        
        if os.path.exists(file_path):
            print(f"[{i+1}/{len(models)}] {bigg_id} already exists. Skipping.")
            continue
            
        print(f"[{i+1}/{len(models)}] Downloading {bigg_id}...", end=" ")
        
        # Construct download URL (standard BiGG pattern)
        # http://bigg.ucsd.edu/static/models/e_coli_core.xml
        download_url = f"http://bigg.ucsd.edu/static/models/{bigg_id}.xml"
        
        try:
            res = requests.get(download_url)
            if res.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(res.content)
                print("Done.")
            else:
                print(f"Failed (Status {res.status_code})")
        except Exception as e:
            print(f"Error: {e}")
            
        # Be nice to the server
        time.sleep(0.5)

if __name__ == "__main__":
    fetch_all_models()
