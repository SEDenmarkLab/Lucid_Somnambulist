import molli as ml


class _entry:
    """
    Abstract class for a database entry.

    Optional name, CAS, smiles can be passed.
    """

    def __init__(self, name=None, CAS=None, smiles=None, features=None):
        self.name = name
        self.cas = CAS
        self.smi = smiles
        self.feat = features


class substrate(_entry):
    def __init__(self, role=None, name=None, CAS=None, smiles=None, features=None):
        if role == None:
            raise Exception("Must pass a role for a substrate entry - nuc or el")
        else:
            self.role = role
        super().__init__(name=None, CAS=None, smiles=None, features=None)


class catalyst(_entry):
    def __init__(self, name=None, CAS=None, smiles=None, features=None):
        super().__init__(name=None, CAS=None, smiles=None, features=None)


class reagent(substrate):
    def __init__(self, role, name=None, CAS=None, smiles=None, features=None):
        super().__init__(role=None, name=None, CAS=None, smiles=None, features=None)


class reaction(_entry):
    """
    Database for storing reaction data.

    -Entry names should be reaction handles.
    -CAS and smiles should be for product
    -features should be fully concatenated for the reaction, so ready for partitioning

    """

    def __init__(self, nuc, el, cat, solv, base, name="", write_dir=""):
        self.nuc = nuc
        self.el = el
        self.cat = cat
        self.solv = solv
        self.base = base
        self.handle = f"{nuc.name}_{el.name}_{cat.name}_{solv.name}_{base.name}"
        super().__init__(name="", write_dir="")
