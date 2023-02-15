# This workflow adds structures to the feature database (without requiring experimental data).
# This may be useful for analyzing component space or performing unsupervised learning tasks.
# This workflow also can be used by predict to generate features for new structures.

from sys import argv
import molli as ml
import argparse
from pathlib import Path


def get_mol_from_graph(user_input):
    """
    Take user input stream (from commandline) as a file path to a cdxml and parse it to a molli molecule object

    This is one option for input, others will need to be made for SMILES, etc
    """
    col = ml.split_cdxml(user_input, enum=False, fmt="{idx}")
    assert isinstance(col, ml.Collection)
    return col


def get_mol_from_smiles(user_input):
    """
    Take user input of smiles string and convert it to a molli molecule object

    NOTE: If possible, avoid using RDKit to avoid a large dependency

    ***development to come***

    """
    ...


if __name__ == "__main__":
    assert len(argv) > 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdxml")
    parser.add_argument("--smi")
    args = parser.parse_args()
    if args.smi != None:
        raise Exception(
            "SMILES parsing not yet supported; please use fpath option for CDXML"
        )
    # if not isinstance(args.cdxml, str):
    #     raise Exception("Please pass a filepath for a cdxml file")
    try:
        p = Path(args.cdxml)
        assert p.exists()
    except:
        raise Exception("Please pass a valid path for a cdxml file - parsing failed")
    collection = get_mol_from_graph(p)
