# This workflow adds structures to the feature database (without requiring experimental data).
# This may be useful for analyzing component space or performing unsupervised learning tasks.
# This workflow also can be used by predict to generate features for new structures.

from sys import argv
import molli as ml
import argparse
from pathlib import Path
from somn.build import parsing

# temp_work = r"C:\Users\rineharn\workspace/"

temp_work = r"/mnt/c/Users/rineharn/workspace/linux/"


if __name__ == "__main__":

    ### Basic checks on the input - start with the more difficult cdxml input, then go to smiles later
    ### Use molli for cdxml parse - it is better than openbabel. Use openbabel for smiles parsing and adding hydrogens (to both)
    assert len(argv) > 1
    parse = parsing.InputParser(serialize=True, path_to_write=temp_work)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fmt", nargs=2, help="Format for the input - cdxml or (eventually) smiles"
    )
    parser.add_argument(
        "r", help="The type of input - nucleophile (nuc) or electrophile (el)"
    )
    args = parser.parse_args()
    if args.fmt[0] == "smi":
        # raise Exception(
        #     "SMILES parsing not yet supported; please use fpath option for CDXML"
        # )
        try:
            collection = parse.get_mol_from_smiles(args.fmt[1])
        except:
            raise Warning(
                "Something went wrong with openbabel smiles parsing - check input/output"
            )
        prep, err = parse.preopt_geom(collection)

    elif args.fmt[0] == "cdxml":
        try:
            p = Path(args.fmt[1])
            assert p.exists()
        except:
            raise Exception(
                "Please pass a valid path for a cdxml file - parsing failed"
            )
        collection = parse.get_mol_from_graph(args.fmt[1])
        prep, err = parse.prep_collection(collection)
    else:
        raise Exception("Please pass cdxml as input (eventually smi will be accepted)")

    ###     Parsing molecular drawings into mol objects using firstgen molli. Then, hydrogens are added.
    ###     These preoptimized structures will then be optimized using xtb, which generally returns reasonable
    ###     geometries.
