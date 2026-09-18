"""Biologically-structured metabolic map layout engine (v2).

See layout_algorithm.md for the design. Pipeline:

    cobra model + reaction subset
      -> compound.build_compound_graph()   primary-compound reduction (S1)
      -> motifs.find_rings()               ring motifs, contracted to super-nodes (S2)
      -> sugiyama.layout()                 layering + crossing min + Brandes-Koepf (S3)
      -> motifs.expand_rings()             ring rotation + member placement (S4)
      -> render.build_escher_map()         midmarkers, cofactor stubs, routing (S5/S7)
      -> metrics.score()                   acceptance metrics (S9)
"""
