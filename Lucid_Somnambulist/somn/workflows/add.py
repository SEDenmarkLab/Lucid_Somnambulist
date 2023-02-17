# This workflow adds structures to the feature database (without requiring experimental data).
# This may be useful for analyzing component space or performing unsupervised learning tasks.
# This workflow also can be used by predict to generate features for new structures.

from sys import argv
import molli as ml
import argparse
from pathlib import Path
from datetime import date
from openbabel import openbabel as ob

# temp_work = r"C:\Users\rineharn\workspace/"
temp_work = r"/mnt/c/Users/rineharn/workspace/linux/"


def get_mol_from_graph(user_input):
    """
    Take user input stream (from commandline) as a file path to a cdxml and parse it to a molli molecule object

    This is one option for input, others will need to be made for SMILES, etc
    """
    if user_input.split(".")[1] != "cdxml":
        raise Exception("cdxml path not specified - wrong extension, but valid path")
    col = ml.parsing.split_cdxml(user_input, enum=True, fmt="pr{idx}")
    assert isinstance(col, ml.Collection)
    return col


def get_mol_from_smiles(user_input):
    """
    Take user input of smiles string and convert it to a molli molecule object

    NOTE: If possible, avoid using RDKit to avoid a large dependency

    ***development to come***

    """
    ...


def add_hydrogens(col: ml.Collection):
    """
    Openbabel can at least do this one thing right - add hydrogens.

    Note: any explicit hydrogens in a parsed structure will cause this to fail...

    """

    output = []
    for mol in col:
        obmol = ob.OBMol()
        obconv = ob.OBConversion()
        obconv.SetInAndOutFormats("mol2", "mol2")
        obconv.ReadString(obmol, mol.to_mol2())
        obmol.AddHydrogens()
        newmol = ml.Molecule.from_mol2(obconv.WriteString(obmol), mol.name)
        output.append(newmol)
    return ml.Collection(f"{col.name}_hadd", output)


def preopt_geom(col: ml.Collection):
    xtb = ml.XTBDriver("preopt", scratch_dir=temp_work + "scratch/", nprocs=1)
    opt = ml.Concurrent(col, backup_dir=temp_work + "scratch/", update=2)(xtb.optimize)(
        method="gfnff"
    )
    col_opt = ml.Collection(
        name="opt-test", molecules=[f for f in opt if isinstance(f, ml.Molecule)]
    )
    print(opt)
    return col_opt


if __name__ == "__main__":
    assert len(argv) > 1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fmt", nargs=2, help="Format for the input - cdxml or (eventually) smiles"
    )
    # parser.add_argument("--cdxml")
    # parser.add_argument("--smi")
    parser.add_argument(
        "r", help="The type of input - nucleophile (nuc) or electrophile (el)"
    )
    args = parser.parse_args()
    if args.fmt[0] == "smi":
        raise Exception(
            "SMILES parsing not yet supported; please use fpath option for CDXML"
        )
    # if not isinstance(args.cdxml, str):
    #     raise Exception("Please pass a filepath for a cdxml file")
    elif args.fmt[0] == "cdxml":
        try:
            p = Path(args.fmt[1])
            assert p.exists()
        except:
            raise Exception(
                "Please pass a valid path for a cdxml file - parsing failed"
            )
    else:
        raise Exception("Please pass cdxml as input (eventually smi will be accepted)")
    ###
    ###     Parsing molecular drawings into mol objects using firstgen molli. Then, hydrogens are added.
    ###     These preoptimized structures will then be optimized using xtb, which generally returns reasonable
    ###     geometries.

    collection = get_mol_from_graph(
        args.fmt[1]
    )  # Temp name generated; this will be corrected later, when they are databased, and the names will be set
    collection.to_zip(temp_work + "cdxml_parse.zip")
    for mol in collection:
        with open(temp_work + "{mol.name}.mol2", "w") as g:
            g.write(mol.to_mol2())
    col_h = add_hydrogens(collection)
    ID_ = date.today()
    col_h.to_zip(temp_work + f"new_{args.r}_input_{ID_}.zip")
    for mol in col_h:
        with open(temp_work + "{mol.name}_hadd.mol2", "w") as g:
            g.write(mol.to_mol2())
    preopt = preopt_geom(col_h)
    for mol in preopt:
        with open(temp_work + "{mol.name}_preopt.mol2", "w") as g:
            g.write(mol.to_mol2())
