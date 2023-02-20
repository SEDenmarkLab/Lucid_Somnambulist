import molli as ml
from os import makedirs
from datetime import date
from openbabel import openbabel as ob
from openbabel import pybel
from somn._core._core import check_parsed_mols
import warnings


class InputParser:

    """
    Input parsing class with built-in warnings and error serialization features
    """

    def __init__(self, serialize=False, path_to_write="somn_failed_to_parse/"):
        self.ser = serialize
        self.path_to_write = path_to_write

    def serialize(self, mols_to_write: list, specific_msg=""):
        makedirs(self.path_to_write, exist_ok=True)
        for mol in mols_to_write:
            assert isinstance(mol, ml.Molecule)
            with open(self.path_to_write + f"{mol.name}_{specific_msg}.mol2", "w") as g:
                g.write(mol.to_mol2())

    def get_mol_from_graph(self, user_input):
        """
        Take user input stream (from commandline) as a file path to a cdxml and parse it to a molli molecule object

        Later, this should be implemented for a GUI-based input.

        For now, the names are enumerated with a prefix "pr" for easy detection. Later, these can be  \
        changed to final names (or ELN numbers)
        """
        if user_input.split(".")[1] != "cdxml":
            raise Exception(
                "cdxml path not specified - wrong extension, but valid path"
            )
        col = ml.parsing.split_cdxml(user_input, enum=True, fmt="pr{idx}")
        assert isinstance(col, ml.Collection)
        return col

    def get_mol_from_smiles(self, user_input):
        """
        Take user input of smiles string and convert it to a molli molecule object

        """
        obmol = ob.OBMol()
        obconv = ob.OBConversion()
        obconv.SetInAndOutFormats("smi", "mol2")
        obconv.AddOption("h", ob.OBConversion.GENOPTIONS)
        try:
            obconv.ReadString(obmol, user_input + "    pr" + str(date.today()))
        except:
            raise Exception("Failed to parse smiles string; check format for errors")
        gen3d = ob.OBOp.FindType("gen3D")
        gen3d.Do(
            obmol, "--fastest"
        )  # Documentation is lacking for pybel/python, found github issue from 2020 with this
        newmol = ml.Molecule.from_mol2(
            obconv.WriteString(obmol), name="pr" + str(date.today())
        )
        if self.ser == True:
            self.serialize([newmol], specific_msg="smiles_preopt")
        col = ml.Collection(name="from_smi_hadd", molecules=[newmol])
        return col

    def add_hydrogens(self, col: ml.Collection, specific_msg=""):
        """
        Openbabel can at least do this one thing right - add hydrogens.

        Note: any explicit hydrogens in a parsed structure will cause this to fail...

        """
        output = []
        for mol_ in col:
            obmol = ob.OBMol()
            obconv = ob.OBConversion()
            obconv.SetInAndOutFormats("mol2", "mol2")
            obconv.ReadString(obmol, mol_.to_mol2())
            obmol.AddHydrogens()
            newmol = ml.Molecule.from_mol2(obconv.WriteString(obmol), mol_.name)
            output.append(newmol)
        mols, errs = check_parsed_mols(output, col)
        if len(errs) > 0:
            warnings.warn(
                message="It looks like adding hydrogens to at least one input structure failed... \
            try removing ALL explicit hydrogens from input."
            )
        if self.ser == True:
            self.serialize(errs, specific_msg="addH_err")
            self.serialize(mols, specific_msg="addH_suc")
        return ml.Collection(f"{col.name}_hadd", mols), errs

    def preopt_geom(self, col: ml.Collection):
        xtb = ml.XTBDriver(
            "preopt", scratch_dir=self.path_to_write + "scratch/", nprocs=1
        )
        opt = ml.Concurrent(col, backup_dir=self.path_to_write + "scratch/", update=2)(
            xtb.optimize
        )(method="gfn2")
        mols, errs = check_parsed_mols(opt, col)
        if self.ser == True:
            self.serialize(errs, specific_msg="preopt_err")
            self.serialize(mols, specific_msg="preopt_suc")
        return ml.Collection(f"{col.name}_preopt", mols), errs

    def prep_collection(self, col: ml.Collection):
        """
        Several consistent steps to prep incoming geometries
        """
        col_h, errs_h = self.add_hydrogens(col)
        # ID_ = date.today()
        preopt, errs_pre = self.preopt_geom(col_h)
        return preopt, errs_h.extend(errs_pre)
