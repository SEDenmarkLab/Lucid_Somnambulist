# This workflow adds structures to the feature database (without requiring experimental data).
# This may be useful for analyzing component space or performing unsupervised learning tasks.
# This workflow also can be used by predict to generate features for new structures.

from sys import argv
import molli as ml
import argparse
from pathlib import Path
from somn.build import parsing
from somn.workflows import STRUC_

# temp_work = r"C:\Users\rineharn\workspace/"
# temp_work = r"/mnt/c/Users/rineharn/workspace/linux/"

if __name__ == "__main__":
    ### Basic checks on the input - start with the more difficult cdxml input, then go to smiles later
    ### Use molli for cdxml parse - it is better than openbabel. Use openbabel for smiles parsing and adding hydrogens (to both)
    assert len(argv) > 1
    parser = argparse.ArgumentParser(
        usage="Specify format (smi or cdxml), then a smiles string/file with smiles or cdxml file, and finally indicate 'el' or 'nuc' for electrophile or nucleophile. Optionally, serialize output structures with '-ser' - must pass some input as an argument after, standard use is 'y'"
    )
    parser.add_argument(
        "fmt",
        nargs=2,
        help="Format for the input - cdxml or smi, followed by file or smiles string",
    )
    parser.add_argument(
        "r", help="The type of input - nucleophile (nuc) or electrophile (el)"
    )
    parser.add_argument(
        "-ser",
        help="Optional serialize argument, pass -ser and the path to save to",
    )
    args = parser.parse_args()
    if (
        args.ser
    ):  # Serialization during parsing to check for errors - this is important for users to troubleshoot
        assert Path(args.ser[1]).exists()
        parse = parsing.InputParser(serialize=True, path_to_write=args.ser[1])
    else:
        parse = parsing.InputParser(serialize=False)
    if args.fmt[0] == "smi":
        # raise Exception(
        #     "SMILES parsing not yet supported; please use fpath option for CDXML"
        # )
        if Path(args.fmt[1]).exists():
            raise Exception(
                "Looks like a file path was passed as a SMILES - check inputs. If multiple smiles are being input, use 'multsmi' for format"
            )
        try:
            collection = parse.get_mol_from_smiles(args.fmt[1])
        except:
            raise Warning(
                "Something went wrong with openbabel smiles parsing - check input/output"
            )
        prep, err = parse.preopt_geom(collection)
    elif args.fmt[0] == "multsmi":
        print("got to multsmi")
        collection = parse.scrape_smiles_csv(args.fmt[1])
        prep, err = parse.prep_collection(collection, update=20)
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
        raise Exception(
            "Please pass cdxml or smiles as input. See help for instructions."
        )

    collection.to_zip(parse.path_to_write + "input_struc_preopt_col.zip")
    # =====================
    # Need to build up database class FIRST, then end workflow with plugging into database
    # =====================

    ### Need to save collection to a buffer, then calculate atom props to add to the JSON. Buffer for new compounds until new models are trained.
