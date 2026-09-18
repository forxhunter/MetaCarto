import sys
import os

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"[OK] {module_name}")
    except ImportError as e:
        print(f"[FAIL] {module_name}: {e}")

print("Verifying Python Environment...")
check_import("cobra")
check_import("networkx")
check_import("torch")
check_import("torch_geometric")
check_import("rdkit")
check_import("bioservices")

print("\nVerifying Source Code Imports...")
try:
    from src import parsing, fba, chemistry
    print("[OK] src modules imported successfully")
except Exception as e:
    print(f"[FAIL] src modules: {e}")

print("\nSetup verification complete.")
