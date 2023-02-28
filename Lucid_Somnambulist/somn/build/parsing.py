import molli as ml
from os import makedirs
from datetime import date
import somn
import warnings
from pathlib import Path
from os.path import splitext
import pandas as pd
import numpy as np
from collections import namedtuple


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

    def get_mol_from_smiles(self, user_input, recursive_mode=False, names=None):
        """
        Take user input of smiles string and convert it to a molli molecule object

        """
        from openbabel import openbabel as ob

        if recursive_mode == False:
            obmol = ob.OBMol()
            obconv = ob.OBConversion()
            obconv.SetInAndOutFormats("smi", "mol2")
            obconv.AddOption("h", ob.OBConversion.GENOPTIONS)
            try:
                obconv.ReadString(obmol, user_input + "    pr" + str(date.today()))
            except:
                raise Exception(
                    "Failed to parse smiles string; check format for errors"
                )
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
        elif recursive_mode == True:
            assert type(user_input) == list
            if type(names) == list and len(names) != len(user_input):
                warnings.warn(
                    "Warning: list of names passed were not the same length as input SMILES; failed to infer naming scheme, so enumerating structures."
                )
                names = None
            mols_out = []
            gen3d = ob.OBOp.FindType("gen3D")
            obconv = ob.OBConversion()
            obconv.SetInAndOutFormats("smi", "mol2")
            obconv.AddOption("h", ob.OBConversion.GENOPTIONS)
            for i, smiles_ in enumerate(user_input):
                print(i, smiles_)
                obmol = ob.OBMol()
                try:
                    obconv.ReadString(
                        obmol, smiles_ + f"    pr{str(date.today())}_{i+1}"
                    )
                except:
                    raise Exception(
                        "Failed to parse smiles string; check format for errors"
                    )
                gen3d.Do(
                    obmol, "--best"
                )  # Documentation is lacking for pybel/python, found github issue from 2020 with this
                if names == None:
                    newmol = ml.Molecule.from_mol2(
                        obconv.WriteString(obmol), name=f"pr{str(date.today())}_{i+1}"
                    )
                elif type(names) == list:
                    newmol = ml.Molecule.from_mol2(
                        obconv.WriteString(obmol), name=f"pr{names[i]}"
                    )
                mols_out.append(newmol)
                del obmol

            if self.ser == True:
                self.serialize(mols_out, specific_msg="smiles_preopt")
            col = ml.Collection(name="from_smi_hadd", molecules=mols_out)
            return col

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
        mols, errs = somn.core.check_parsed_mols(output, col)
        if len(errs) > 0:
            warnings.warn(
                message="It looks like adding hydrogens to at least one input structure failed...try removing ALL explicit hydrogens from input."
            )
        if self.ser == True:
            self.serialize(errs, specific_msg="addH_err")
            self.serialize(mols, specific_msg="addH_suc")
        return ml.Collection(f"{col.name}_hadd", mols), errs

    def preopt_geom(self, col: ml.Collection, update=None):
        xtb = ml.XTBDriver(
            "preopt", scratch_dir=self.path_to_write + "scratch/", nprocs=1
        )
        if update == None:
            opt = ml.Concurrent(
                col, backup_dir=self.path_to_write + "scratch/", update=2
            )(xtb.optimize)(method="gfn2")
        elif type(update) == int:
            opt = ml.Concurrent(
                col, backup_dir=self.path_to_write + "scratch/", update=update
            )(xtb.optimize)(method="gfn2")
        else:
            raise Exception(
                "Optional update argument passed for input structure preoptimization, but the value passed was not an integer. Either do not use this optional feature and accept 2 second update cycles, or input a valid integer value of seconds."
            )
        mols, errs = somn.core.check_parsed_mols(opt, col)
        if self.ser == True:
            self.serialize(errs, specific_msg="preopt_err")
            self.serialize(mols, specific_msg="preopt_suc")
        return ml.Collection(f"{col.name}_preopt", mols), errs

    def prep_collection(self, col: ml.Collection, update=None):
        """
        Several consistent steps to prep incoming geometries
        """

        col_h, errs_h = self.add_hydrogens(col)
        # ID_ = date.today()
        if update == None:
            preopt, errs_pre = self.preopt_geom(col_h)
        else:
            preopt, errs_pre = self.preopt_geom(col_h, update=update)
        return preopt, errs_h.extend(errs_pre)

    def scrape_smiles_csv(self, fpath):
        """
        Scrapes smiles strings out of a csv file

        SMILES strings should be column index 0, and optional name should be column index 1
        """
        df = pd.read_csv(fpath, header=0, index_col=0)
        if len(df.columns) == 1:
            collection = self.get_mol_from_smiles(
                df.iloc[:, 0].to_list(), recursive_mode=True
            )
            return collection
        elif len(df.columns) > 1:
            collection = self.get_mol_from_smiles(
                df.iloc[:, 0].to_list(),
                recursive_mode=True,
                names=df.iloc[:, 1].to_list(),
            )
            return collection


class DataHandler:
    """
    Parsing data including I/O functions for regression tasks, as well as labelling for (multi)class modeling

    Designed to be robust and detect duplicate entries, absurd values (e.g. wrong scale), etc.

    When serialized, the outputs should be ready for subsequent steps
    """

    def __init__(self, fpath_or_df, **kwargs):
        """
        CSV/XLSV should have a header AND index column. XLSV will assume first sheet unless kwarg "sheet_name" is passed.
        """
        if isinstance(fpath_or_df, pd.DataFrame):
            self.data = fpath_or_df
        elif isinstance(fpath_or_df, str):
            if Path(fpath_or_df).exists():
                if splitext(fpath_or_df)[1] == ".csv":
                    df = pd.read_csv(fpath_or_df, header=0, index_col=0)
                    self.data = df
                elif splitext(fpath_or_df)[1] == ".xlsx":
                    if "sheet_name" in kwargs:
                        self.sheet_name = kwargs["sheet_name"]
                    else:
                        self.sheet_name = 0
                    df = pd.read_excel(
                        fpath_or_df, header=0, index_col=0, sheet_name=self.sheet_name
                    )
                    self.data = df
                elif splitext(fpath_or_df[1]) == ".feather":
                    df = pd.read_feather(
                        fpath_or_df
                    ).transpose()  # Assumes serialized columns for feather = row indices
                    self.data = df
            else:
                raise Exception(
                    "Cannot parse filepath or buffer passed to DataHandler - check path"
                )
        else:
            raise Exception("Cannot parse input type for DataHandler class.")
        self.handles = self.df.index.to_list()
        if "name" in kwargs.keys():
            self.name = kwargs["name"]
        else:
            self.name = None
        self.cleanup_handles()

    ### These are methods for this class

    def cleanup_handles(self):
        """
        Catch-all for fixing weird typos in data entry for data files.
        """
        indices = self.data.index
        strip_indices = pd.Series([f.strip() for f in indices])
        self.data.index = strip_indices
        # self.data.drop_duplicates(inplace=True) ## This does not work; removes more than it should
        self.data = self.data[~self.data.index.duplicated(keep="first")]

    ### This is for handling IO operations with dataset, partitioning, handle/label

    def zero_nonzero_rand_splits(
        self,
        validation=False,
        n_splits=1,
        fold=7,
        yield_cutoff=1,
    ):
        """
        Split zero/nonzero data, THEN apply random splits function

        Get two output streams for zero and nonzero data to train classification models

        Can set "fuzzy" yield cutoff. This is percent yield that where at or below becomes class zero.

        """
        zero_mask = self.data.to_numpy() < yield_cutoff
        nonzero_data = self.data[~zero_mask]
        zero_data = self.data[zero_mask]
        if validation == False:
            tr_z, te_z = self.random_splits(zero_data, n_splits=n_splits, fold=fold)
            tr_n, te_n = self.random_splits(nonzero_data, n_splits=n_splits, fold=fold)
            tr = pd.concat(tr_z, tr_n, axis=1)
            te = pd.concat(te_z, te_n, axis=1)
            return tr, te
        elif validation == True:
            tr_z, va_z, te_z = self.random_splits(
                zero_data, n_splits=n_splits, fold=fold, validation=validation
            )
            tr_n, va_n, te_n = self.random_splits(
                nonzero_data, n_splits=n_splits, fold=fold, validation=validation
            )
            tr = pd.concat((tr_z, tr_n), axis=1)
            va = pd.concat((va_z, va_n), axis=1)
            te = pd.concat((te_z, te_n), axis=1)
            return tr, va, te
        else:
            raise ValueError(
                "validation parameter for zero/nonzero split function must be Boolean"
            )


def assemble_descriptors_from_handles(handle_input, am_dict, br_dict):
    """
    General utility for assembling ordered descriptors based on input reaction handles and
    calculated amine and bromide rdf descriptor dictionaries. This can be used to automate
    testing hypertuning of rdf calculator hyperparams.


    use sysargv[1] for handle input

    sys.argv[1] should be list of truncated handles:
    amine_bromide,amine_bromide,....

    OR

    pass a list of ALL handles:
    amine_br_cat_solv_base

    This will assemble only descriptors as required by the list of handles, and will
    return the descriptors in the appropriate order

    Can also be all handles from a datafile; whatever.

    This is meant to use am_dict and br_dict as conformer-averaged descriptors.
    This lets the user apply different parameters to descriptor tabulation flexibly.

    """
    if type(handle_input) == str:
        rxn_hndls = [f for f in handle_input.split(",") if f != ""]
        prophetic = True
    elif type(handle_input) == list:
        rxn_hndls = [tuple(f.rsplit("_")) for f in handle_input]
        prophetic = False
    else:
        raise ValueError(
            "Must pass manual string input of handles OR list from dataset"
        )

    # print(handle_input)
    # print(rxn_hndls)
    # outfile_name = date_+'_desc_input'
    directory = "descriptors/"
    basefile = directory + "base_params.csv"
    basedf = pd.read_csv(basefile, header=None, index_col=0).transpose()
    solvfile = directory + "solvent_params.csv"
    solvdf = pd.read_csv(solvfile, header=None, index_col=0).transpose()
    # catfile = directory+'cat_aso_aeif_combined_11_2021.csv' ##Normal ASO/AEIF cats CHANGED TEST
    catfile = (
        directory + "iso_catalyst_embedding.csv"
    )  ##isomap embedded cats CHANGED FOR SIMPLIFICATION
    catdf = pd.read_csv(catfile, header=None, index_col=0).transpose()

    ### Trying to assemble descriptors for labelled examples with specific conditions ###
    if prophetic == False:
        columns = []
        labels = []
        for i, handle in enumerate(rxn_hndls):
            am, br, cat, solv, base = handle
            catdesc = catdf[cat].tolist()
            solvdesc = solvdf[int(solv)].tolist()
            basedesc = basedf[base].tolist()
            amdesc = []
            for key, val in am_dict[am].iteritems():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].iteritems():
                brdesc.extend(val.tolist())
            handlestring = handle_input[i]
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handlestring)
        outdf = pd.DataFrame(columns, index=labels).transpose()
        return outdf

    ### Trying to assemble descriptors for ALL conditions for specific amine/bromide couplings ###
    elif prophetic == True:
        solv_base_cond = ["1_a", "1_b", "1_c", "2_a", "2_b", "2_c", "3_a", "3_b", "3_c"]
        allcats = [str(f + 1) for f in range(21) if f != 14]
        s = "{}_{}_{}"
        exp_handles = []
        for combination in itertools.product(rxn_hndls, allcats, solv_base_cond):
            exp_handles.append(s.format(*combination))
        columns = []
        labels = []
        for handle in exp_handles:
            am, br, cat, solv, base = tuple(handle.split("_"))
            catdesc = catdf[cat].tolist()
            solvdesc = solvdf[int(solv)].tolist()
            basedesc = basedf[base].tolist()
            amdesc = []
            for key, val in am_dict[am].iteritems():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].iteritems():
                brdesc.extend(val.tolist())
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handle)
            # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
        outdf = pd.DataFrame(columns, index=labels).transpose()
        # print(outdf)
        return outdf

    @staticmethod
    def random_splits(df, validation=False, n_splits: int = 1, fold: int = 7):
        """
        Get split handles in tuple.

        Validation boolean decides if output is (train,test) or (train,validate,test)

        Each is a list of handles, train, (val), test

        """
        no_exp = len(df.index)
        rand_arr = np.random.randint(1, high=fold + 1, size=no_exp, dtype=int)
        if validation == False:
            train_mask = (rand_arr > 1).tolist()
            test_mask = (rand_arr == 1).tolist()
            mask_list = [train_mask, test_mask]
        elif validation == True:
            train_mask = (rand_arr > 2).tolist()
            validate_mask = (rand_arr == 2).tolist()
            test_mask = (rand_arr == 1).tolist()
            mask_list = [train_mask, validate_mask, test_mask]
        out = tuple([df.iloc[msk, :] for msk in mask_list])
        return out

    @staticmethod
    def prep_for_binary_classifier(df_in, yield_cutoff: int = 1):
        """
        Prepare data for classifier by getting class labels from continuous yields
        """
        if type(df_in) == tuple:
            out = []
            for df in df_in:
                df = df.where(
                    df > yield_cutoff, other=0, inplace=True
                )  # collapse yields at or below yield cutoff to class zero
                df = df.where(
                    df == 0, other=1, inplace=True
                )  # collapse yields to class one
                out.append(df)
            return tuple(out)
        elif isinstance(df_in, pd.DataFrame):
            df = df.where(
                df > yield_cutoff, other=0, inplace=True
            )  # collapse yields at or below yield cutoff to class zero
            df = df.where(
                df == 0, other=1, inplace=True
            )  # collapse yields to class one
            return df
        else:
            raise Exception(
                "Passed incorrect input to staticmethod of DataHandler to prep data for classification - check input."
            )

    # @classmethod
    # def from_df(self, df: pd.DataFrame):
    #     ...

    @classmethod
    def to_df(self):
        return self.data

    @classmethod
    def to_feather(self, fpath, orient=None):
        """
        Write the data from DataHandler (after processing, etc) to a feather file.
        """
        if orient == None:
            self.data.to_feather(fpath)
        elif orient == "index":
            self.data.transpose().to_feather(fpath)
        elif orient == "column":
            self.data.to_feather(fpath)
        elif orient == "both":
            i, j = self.data.shape
            if i > j:
                self.data.transpose().to_feather(fpath)
                temp_buf = self.data.columns
                path_, ext_ = splitext(fpath)
                temp_buf.to_feather(f"{path_}_cols{ext_}")
            if i < j:
                self.data.transpose().to_feather(fpath)
                temp_buf = self.data.columns  # index of original rotated into cols
                path_, ext_ = splitext(fpath)
                temp_buf.to_feather(
                    f"{path_}_cols{ext_}"
                )  # write original cols separately
