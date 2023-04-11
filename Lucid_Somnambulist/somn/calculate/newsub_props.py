import molli as ml

"""
This needs to be a driver for molli workflow:

1. Take collection of 3D structures input
2. Preoptimize geometry with gfnff
3. Conformer search with standard params
4. Reject any failed searches and give informative outputs (which ones failed)
5. Provide alternative routes to getting geometries. Perhaps gfnff instead of gfn2-xtb, for example.
6. Minimize conformer geometries. 
7. Collate output structures and save them to a buffer repository in somn.data. That should be used to update models.
8. Run the atom properties calculation in xtb, and collate the outputs. Incorporate those into the atomprops json/dict in somn.data.

When this fails, the tool fails. It will probably take significant development. 
"""
