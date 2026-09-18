import cobra
import glob
import os

files = glob.glob("data/bigg/models/e_coli_core.xml")
if not files:
    print("e_coli_core.xml not found")
else:
    model = cobra.io.read_sbml_model(files[0])
    print(f"Model: {model.id}")
    print("First 5 reactions subsystems:")
    for i, rxn in enumerate(model.reactions[:20]):
        print(f"{rxn.id}: '{rxn.subsystem}'")
    
    if hasattr(model, 'groups'):
        print(f"Groups: {len(model.groups)}")
        for g in model.groups[:5]:
            print(f"Group {g.name} ({g.kind}): {len(g.members)} members")
