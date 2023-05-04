import molli as ml
import json
from attrs import define, field
from somn.workflows import SCRATCH_, STRUC_, DESC_
from somn.data import ACOL, BCOL, ASMI, BSMI, AMINES, BROMIDES
import pandas as pd


@define
class PropheticInput:
    """
    Object for handling new structure(s). Will take valid input structure from InputParser and add to reactant database(s) and atomproperties files.
    """

    name: str | list = field()
    role: str | list = field()
    smi: str | list = field()
    struc: ml.Molecule | ml.Collection = field()
    state = field(default="")
    known = field(default="")
    conformers = field(default="")
    failures = field(default="")
    roles_d = field(default={})

    # def __attrs_post_init__(self):
    #     self.__setattr__("state", None)
    #     self.__setattr__("known", None)
    #     self.__setattr__("conformers", None)
    #     self.__setattr__("failures", None)
    #     self.__setattr__("roles_d", {})

    def check_input(self):
        """
        This will check input and define operating mode. There should be two modes of use: multimol or single mol. Used by classmethods to build instances.

        This will also check if the molecule is already known, and not run computation on it.
        """
        if isinstance(self.struc, ml.Molecule):
            assert isinstance(self.name, str)
            assert isinstance(self.role, str)
            assert isinstance(self.smi, str)
            self.state = "single"
        elif isinstance(self.struc, ml.Collection):
            assert isinstance(self.name, list)
            assert isinstance(self.role, list)
            assert isinstance(self.smi, list)
            self.state = "multi"
            for k, j in zip(self.name, self.role):
                assert j in ["nuc", "el"]
                self.roles_d[k] = j
        else:
            raise Exception(
                f"Prophetic structure input must be single structure or multiple - check input types"
            )
        ### Check smiles against database
        inv_am = {v: k for k, v in ASMI.items()}
        inv_br = {v: k for k, v in BSMI.items()}
        if (
            self.state == "single"
        ):  # Special case - if it fails, just kill job and give the user the right name.
            if self.role == "nuc" and self.smi in ASMI.values():
                raise Exception(
                    f"Structure is already in database, and computing single structure\nHere is the name of that structure:{inv_am[self.smi]}"
                )
            elif self.role == "el" and self.smi in BSMI.values():
                raise Exception(
                    f"Structure is already in database, and computing single structure\nHere is the name of that structure:{inv_br[self.smi]}"
                )
            else:
                self.known = False
        elif (
            self.state == "multi"
        ):  # Normal case - just don't do computations again on structures we already know about - checking presence in each dictionary.
            self.known = []
            pruned_struc = []
            for smi, role, mol in zip(self.smi, self.role, self.struc.molecules):
                if role == "el" and smi in inv_br.keys():
                    self.known.append(
                        inv_br[smi]
                    )  # Adding name of mol from database because the new mol smiles matches it. These will be skipped
                elif role == "nuc" and smi in inv_am.keys():
                    self.known.append(inv_am[smi])
                else:
                    pruned_struc.append(mol)
            if len(self.known) == 0:  # Makes this easy later.
                self.known = False
            else:
                raise Warning(
                    f"Structures already in database were requested: {self.known}"
                )
            self.struc = ml.Collection(name="pruned_precalc", molecules=pruned_struc)

    def conformer_pipeline(self):
        """
        Calculate conformers
        """
        try:
            self.state
        except NameError:
            raise Exception(
                "Have not defined conformer pipeline mode, check input (or use code as intended)"
            )
        if (
            self.state == "single" and self.known == False
        ):  # Single molecule; must be list to make col
            col = ml.Collection(name="molecule", molecules=[self.struc])
        elif self.state == "multi":
            if self.known == False:
                col = self.struc
            elif isinstance(self.known, list):
                col = ml.Collection(
                    name=self.struc.name,
                    molecules=[
                        f for f in self.struc.molecules if f.name not in self.known
                    ],
                )  ## Making sure that we ignore known structures here - note that they will be called up again using their proper name
        else:
            raise Exception(
                "Defined self.state as something weird for conformer pipeline... try agian."
            )
        # Perform sequential search and screen of conformers.
        crest = ml.CRESTDriver(
            name="confs", scratch_dir=SCRATCH_ + "crest_scratch_1/", nprocs=2
        )
        concur_1 = ml.Concurrent(
            col,
            backup_dir=SCRATCH_ + "crest_search/",
            logfile=SCRATCH_ + "out1.log",
            update=60,
            timeout=10000,
            concurrent=1,
        )
        # print("conformer search beginning")
        output = concur_1(crest.conformer_search)(
            ewin=20, mdlen=5, constr_val_angles=[]
        )
        # print("searched conf\n", output)
        buffer = []
        tracking = {}  # Used to track progress
        for i, k in enumerate(output):
            if isinstance(k, ml.Molecule):
                tracking[k.name] = True
                buffer.append(k)
            else:
                tracking[col.molecules[i].name] = False
        # print(buffer)
        col2 = ml.Collection(
            name="searched", molecules=buffer
        )  # These have undergone a conformer search.
        # print(col2.molecules)
        concur_2 = ml.Concurrent(
            col2,
            backup_dir=SCRATCH_ + "crest_screen/",
            logfile=SCRATCH_ + "out2.log",
            update=30,
            timeout=10000,
            concurrent=1,
        )
        crest = ml.CRESTDriver(
            name="confs", scratch_dir=SCRATCH_ + "crest_scratch_2/", nprocs=2
        )
        # print("conformer screen beginning")
        output2 = concur_2(crest.confomer_screen)(
            method="gfn2", ewin=12
        )  # These get screened to prune out unreasonable structures and reopt.
        buffer2 = []
        # print("screened conf\n", output2)
        assert len(output2) > 0
        for j, k in enumerate(output2):
            if isinstance(k, ml.Molecule):  # By definition, must have succeeded before
                buffer2.append(k)
            else:  # Failed - make sure it is set to False in tracking dictionary
                if (
                    tracking[col2.molecules[j].name] == True
                ):  # If it worked before, but failed here, set it to False.
                    tracking[col2.molecules[j].name] = False
                else:
                    pass  # Already set to False, can skip.
        col3 = ml.Collection(name="screened", molecules=buffer2)
        assert len(buffer2) > 0
        self.conformers = col3
        failures = []
        for key, val in tracking.items():
            if val == False:
                failures.append(col[key])
            elif val == True:
                pass
        if (
            len(failures) > 0
        ):  # If molecules failed, warn the user. Put this in a log or something later. DEV
            raise Warning(
                f"Molecules failed conformer search: {[f.name for f in failures]}"
            )
        self.failures = failures
        if len(self.failures) > 0:
            towrite = ml.Collection(name="failed", molecules=self.failures)
            towrite.to_zip(STRUC_ + "input_mols_failed_conf_step.zip")
        assert len(self.conformers.molecules) > 0
        # These are the core dataset structures; should it be an external volume? DEV
        if self.state == "single":  # Single molecules going in
            assert len(self.conformers.molecules) == 1
            if self.role == "el":
                BCOL.add(self.conformers[0])
                BCOL.to_zip(STRUC_ + "newtotal_bromide.zip")
            elif self.role == "nuc":
                ACOL.add(self.conformers[0])
                ACOL.to_zip(STRUC_ + "newtotal_amine.zip")
        elif (
            self.state == "multi"
        ):  # Many molecules going in; should work for el or nuc
            na = False
            nb = False
            for mol in self.conformers:
                molrole = self.roles_d[mol.name]
                if molrole == "nuc":
                    ACOL.add(mol)
                    if na == False:
                        na = True
                elif molrole == "el":
                    BCOL.add(mol)
                    if nb == False:
                        nb = True
            if na == True:
                ACOL.to_zip(STRUC_ + "newtotal_amine.zip")
            if nb == True:
                BCOL.to_zip(STRUC_ + "newtotal_bromide.zip")
        ## Save things - these are backups
        self.conformers.to_zip(STRUC_ + "newstruc_geoms.zip")
        with open(STRUC_ + "newstruc_roles.json", "w") as k:
            json.dump(self.roles_d, k)

    def atomprop_pipeline(self):
        """
        Calculate atom properties for descriptor calculation, and add to JSON files.
        """
        concur = ml.Concurrent(
            self.conformers,
            backup_dir=SCRATCH_ + "atomprops/",
            logfile=SCRATCH_ + "atomprops.log",
        )
        xtb = ml.XTBDriver(
            name="atomprops", scratch_dir=SCRATCH_ + "xtb_scratch/", nprocs=2
        )
        atomprops = concur(xtb.conformer_atom_props)()
        # print(atomprops[0][0])
        atomprop_out = {}
        failures = []
        for confap, name in zip(atomprops, self.conformers.mol_index):
            if isinstance(confap[0], dict):
                atomprop_out[name] = confap
            else:
                failures.append(name)
        return atomprop_out, failures

    @classmethod
    def from_mol(cls, mol, smi, role):
        """
        Initiate pipeline for new molecule

        molecule, smiles, role
        """
        k = cls(mol.name, role, smi, mol)
        k.check_input()
        return k

    @classmethod
    def from_col(cls, col, smi_list, role_list):
        """
        Initiate pipelien for new molecules

        collection, smiles list, role list
        """
        k = cls([f.name for f in col.molecules], role_list, smi_list, col)
        k.check_input()
        return k
