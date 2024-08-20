import molli as ml
from os import makedirs
from datetime import date
import somn
import warnings

# from pathlib import Path
# from os.path import splitext
import pandas as pd

# import numpy as np
# from collections import namedtuple
import random
import string


class InputParser:
    """
    Input parsing class with built-in warnings and error serialization features
    """

    def __init__(self, serialize=False, path_to_write="somn_failed_to_parse/"):
        self.ser = serialize
        self.path_to_write = path_to_write
        import somn.data

        somn.data.load_all_desc()
        from somn.data import ACOL, BCOL

        names_known = [f.name for f in ACOL.molecules]
        names_known.extend([k.name for k in BCOL.molecules])
        self.names_known_tot = names_known

    def serialize(self, mols_to_write: list, specific_msg=""):
        makedirs(self.path_to_write + "/mol_buffer", exist_ok=True)
        for mol in mols_to_write:
            assert isinstance(mol, ml.Molecule)
            with open(
                self.path_to_write + f"/{mol.name}_{specific_msg}.mol2", "w"
            ) as g:
                g.write(mol.to_mol2())

    def get_mol_from_graph(self, user_input):
        """
        Take user input stream (from commandline) as a file path to a cdxml and parse it to a molli molecule object

        Later, this should be implemented for a GUI-based input.

        For now, the names are enumerated with a prefix "pr" for easy detection. Later, these can be
        changed to final names (or ELN numbers)
        """
        if user_input.split(".")[1] != "cdxml":
            raise Exception(
                "cdxml path not specified - wrong extension, but valid path"
            )
        col = ml.parsing.split_cdxml(user_input, enum=True, fmt="pr{idx}")
        assert isinstance(col, ml.Collection)
        return col

    def get_smi_from_mols(self, col: ml.Collection):
        """
        Get smiles when inputting structures using cdxml
        """
        from openbabel import openbabel as ob

        output = {}
        for mol in col.molecules:
            obmol = ob.OBMol()
            obconv = ob.OBConversion()
            obconv.SetInAndOutFormats("mol2", "smi")
            # obconv.AddOption("h", ob.OBConversion.GENOPTIONS)
            try:
                obconv.ReadString(obmol, mol.to_mol2())
            except Exception:
                raise Exception("Failed to parse mol2 string, cannot fetch smiles")
            # gen3d = ob.OBOp.FindType("gen3D")
            # gen3d.Do(
            # obmol, "--fastest"
            # )  # Documentation is lacking for pybel/python, found github issue from 2020 with this
            smi = obconv.WriteString(obmol).split("\t")[
                0
            ]  # Output uses a tab between name and smiles ... why, obabel?
            output[mol.name] = smi
        return output

    def get_mol_from_smiles(self, user_input, recursive_mode=False, names="none"):
        """
        Take user input of smiles string and convert it to a molli molecule object

        Recursive mode (default off) will take a list input and return a collection.
        Default mode will accept a string input and return a collection with one molecule in it.
        """
        from openbabel import openbabel as ob

        flag_ = "".join(
            random.SystemRandom().choice(string.digits + string.ascii_lowercase)
            for _ in range(3)
        )
        if recursive_mode is False:
            ## If ob can convert, this will work. 
            try:
                obmol = ob.OBMol()
                obconv = ob.OBConversion()
                obconv.SetInAndOutFormats("smi", "mol2")
                obconv.AddOption("h", ob.OBConversion.GENOPTIONS)
                obconv.ReadString(
                    obmol, f"{user_input}    pr{str(date.today())}_{flag_}"
                )
                gen3d = ob.OBOp.FindType("gen3D")
                gen3d.Do(
                    obmol, "--fastest"
                )  # Documentation is lacking for pybel/python, found github issue from 2020 with this
            except Exception: #Sometimes, ob can't convert into 3D directly - adding 2D step in the middle. 
                obmol = ob.OBMol()
                obconv = ob.OBConversion()
                obconv.SetInAndOutFormats("smi", "mol2")
                obconv.AddOption("h", ob.OBConversion.GENOPTIONS)
                obconv.ReadString(obmol, f"{user_input}")
                gen3d = ob.OBOp.FindType("gen3D")
                gen3d.Do(
                    obmol, "--fastest"
                )
            newmol = ml.Molecule.from_mol2(
                obconv.WriteString(obmol), name=f"pr{str(date.today())}_{flag_}"
            )
            if self.ser is True:
                self.serialize([newmol], specific_msg="smiles_preopt")
            col = ml.Collection(name="from_smi_hadd", molecules=[newmol])
            smi_d = {newmol.name: user_input}
            return col, smi_d
        elif recursive_mode is True:
            assert type(user_input) is list
            if type(names) is list and len(names) != len(user_input):
                warnings.warn(
                    "Warning: list of names passed were not the same length as input SMILES; failed to infer naming scheme, so enumerating structures."
                )
                names = "none"
            elif any(item in names for item in self.names_known_tot):
                warnings.warn(
                    "Warning: could overwrite structure names; an already-used name was included. Switching to default naming."
                )
                names = "none"
            mols_out = []
            smi_d = {}  # keep track of smiles for saving later
            gen3d = ob.OBOp.FindType("gen3D")
            obconv = ob.OBConversion()
            obconv.SetInAndOutFormats("smi", "mol2")
            obconv.AddOption("h", ob.OBConversion.GENOPTIONS)
            for i, smiles_ in enumerate(user_input):
                obmol = ob.OBMol()
                try:
                    obconv.ReadString(
                        obmol, smiles_ + f"    pr{str(date.today())}_{i+1}"
                    )
                except Exception:
                    Exception(
                        "Failed to parse smiles string; check format for errors"
                    )
                gen3d.Do(
                    obmol, "--best"
                )  # Documentation is lacking for pybel/python, found github issue from 2020 with this
                if type(names) is list:
                    newmol = ml.Molecule.from_mol2(
                        obconv.WriteString(obmol), name=f"{names[i]}"
                    )
                elif names == "none":
                    newmol = ml.Molecule.from_mol2(
                        obconv.WriteString(obmol),
                        name=f"pr{str(date.today())}_{flag_}_{i+1}",
                    )
                mols_out.append(newmol)
                del obmol
                smi_d[newmol.name] = (
                    smiles_  # Save smiles in dict keyed by molecule name
                )

            if self.ser is True:
                self.serialize(mols_out, specific_msg="smiles_preopt")
            col = ml.Collection(name="from_smi_hadd", molecules=mols_out)
            return col, smi_d

    def add_hydrogens(self, col: ml.Collection, specific_msg=""):
        """
        Openbabel can at least do this one thing right - add hydrogens.

        Note: any explicit hydrogens in a parsed structure will cause this to fail...

        """
        from openbabel import openbabel as ob

        output = []
        for mol_ in col:
            obmol = ob.OBMol()
            obconv = ob.OBConversion()
            obconv.SetInAndOutFormats("mol2", "mol2")
            obconv.ReadString(obmol, mol_.to_mol2())
            obmol.AddHydrogens()
            newmol = ml.Molecule.from_mol2(obconv.WriteString(obmol), mol_.name)
            output.append(newmol)
        mols, errs = somn.util.aux_func.check_parsed_mols(output, col)
        if len(errs) > 0:
            warnings.warn(
                message="It looks like adding hydrogens to at least one input structure failed...try removing ALL explicit hydrogens from input."
            )
        if self.ser is True:
            self.serialize(errs, specific_msg="addH_err")
            self.serialize(mols, specific_msg="addH_suc")
        return ml.Collection(f"{col.name}_hadd", mols), errs

    def preopt_geom(self, col: ml.Collection, update=None):
        xtb = ml.XTBDriver(
            "preopt", scratch_dir=self.path_to_write + "/scratch/", nprocs=1
        )
        if update is None:
            opt = ml.Concurrent(
                col,
                backup_dir=self.path_to_write + "/scratch/",
                update=20,
                concurrent=16,
            )(xtb.optimize)(method="gfn2")
        elif type(update) is int:
            opt = ml.Concurrent(
                col,
                backup_dir=self.path_to_write + "/scratch/",
                update=update,
                concurrent=16,
            )(xtb.optimize)(method="gfn2")
        else:
            Exception(
                "Optional update argument passed for input structure preoptimization, but the value passed was not an integer. Either do not use this optional feature and accept 2 second update cycles, or input a valid integer value of seconds."
            )
            opt = ml.Concurrent(
                col,
                backup_dir=self.path_to_write + "/scratch/",
                update=30,
                concurrent=16,
            )(xtb.optimize)(method="gfn2")
        mols, errs = somn.util.aux_func.check_parsed_mols(opt, col)
        if self.ser is True:
            self.serialize(errs, specific_msg="preopt_err")
            self.serialize(mols, specific_msg="preopt_suc")
        return ml.Collection(f"{col.name}_preopt", mols), errs

    def prep_collection(self, col: ml.Collection, update=None, has_hs=True):
        """
        Several consistent steps to prep incoming geometries
        """

        if has_hs is False:
            col_h, errs_h = self.add_hydrogens(col)
        elif has_hs is True:
            col_h, errs_h = col, []
        if update is None:
            preopt, errs_pre = self.preopt_geom(col_h)
        else:
            preopt, errs_pre = self.preopt_geom(col_h, update=update)
        return preopt, errs_h.extend(errs_pre)

    def scrape_requests_csv(self, fpath):
        """
        Scrapes smiles strings out of a csv file

        Requestor ID or uuid should be column 0, Nuc SMILES should be column 1, El SMILES should be column 2, optional name for nuc column 3, optional name for el column 4
        """
        df = pd.read_csv(fpath, header=0, index_col=None)
        assert isinstance(df, pd.DataFrame)
        if df.columns.to_list() == [
            "user",
            "nuc",
            "el",
            "nuc_name",
            "el_name",
            "nuc_idx",
            "el_idx",
        ]:
            nucs, smiles_d = self.get_mol_from_smiles(
                df.iloc[:, 1].to_list(),
                recursive_mode=True,
                names=df.iloc[:, 3].to_list(),
            )
            elecs, smiles_d_ = self.get_mol_from_smiles(
                df.iloc[:, 2].to_list(),
                recursive_mode=True,
                names=df.iloc[:, 4].to_list(),
            )
            roles = ["nuc" for f in nucs] + ["el" for k in elecs]
            nucs.extend(elecs)
            smiles_d.update(smiles_d_)
            raw_indicies = df.iloc[:, 5].to_list() + df.iloc[:, 6].to_list()
            if all(f == "-" for f in raw_indicies):
                return nucs, smiles_d, roles, None
            nuc_indicies, el_indicies = {}, {}
            for m, r, idx in zip(nucs, roles, raw_indicies):
                if r == "el":
                    el_indicies[m.name] = idx
                elif r == "nuc":
                    nuc_indicies[m.name] = idx
            indicies = (nuc_indicies, el_indicies)
            return nucs, smiles_d, roles, indicies
        else:
            raise Exception(
                "Input requests .csv file not formatted correctly - need 3 or 5 columns, see function scrape_smiles_csv in InputParser"
            )

    def deprec_scrape_requests_csv(self, fpath):
        """
        Scrapes smiles strings out of a csv file

        Requestor ID or uuid should be column 0, Nuc SMILES should be column 1, El SMILES should be column 2, optional name for nuc column 3, optional name for el column 4
        """
        df = pd.read_csv(fpath, header=0, index_col=None)
        assert isinstance(df, pd.DataFrame)
        if len(df.columns) == 3:
            nucs, smiles_d = self.get_mol_from_smiles(
                df.iloc[:, 1].to_list(), recursive_mode=True
            )
            elecs, smiles_d_ = self.get_mol_from_smiles(
                df.iloc[:, 2].to_list(), recursive_mode=True
            )
            roles = ["nuc" for f in nucs] + ["el" for k in elecs]
            nucs.extend(elecs)
            smiles_d.update(smiles_d_)
            return nucs, smiles_d, roles
        elif len(df.columns) == 5:
            nucs, smiles_d = self.get_mol_from_smiles(
                df.iloc[:, 1].to_list(),
                recursive_mode=True,
                names=df.iloc[:, 3].to_list(),
            )
            elecs, smiles_d_ = self.get_mol_from_smiles(
                df.iloc[:, 2].to_list(),
                recursive_mode=True,
                names=df.iloc[:, 4].to_list(),
            )
            roles = ["nuc" for f in nucs] + ["el" for k in elecs]
            nucs.extend(elecs)
            smiles_d.update(smiles_d_)
            return nucs, smiles_d, roles
        else:
            raise Exception(
                "Input requests .csv file not formatted correctly - need 3 or 5 columns, see function scrape_smiles_csv in InputParser"
            )

    def scrape_biovia_smi_file(self, fpath):
        """
        Scrapes a file with multiple smiles inputs. Cannot accept names or reactant roles for individual structures.
        """
        with open(fpath, "r") as k:
            ftext = k.read()
        smi = ftext.strip().split(".")
        return smi


def cleanup_handles(data_df: pd.DataFrame):
    """
    Catch-all for fixing weird typos in data entry for data files.
    """
    indices = data_df.index
    strip_indices = pd.Series([f.strip() for f in indices])
    data_df.index = strip_indices
    data_df = data_df[~data_df.index.duplicated(keep="first")]
    return data_df
