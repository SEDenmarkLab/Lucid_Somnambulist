import molli as ml
from somn.database import dtypes
import json


class somn_DB:
    def __init__(self, name="", write_dir="", entries=[]):
        self.name = name
        self.loc = write_dir
        self.entries = {}
        for i, ent in enumerate(entries):
            if ent.name in self.entries.keys():
                Warning.warn(
                    f"Duplicate entires found for database {self.name}, with name {ent.name} - possible data loss by overwrite avoided. Check names."
                )
                self.entries[ent.name + str(i + 1)] = ent
            self.entries[ent.name] = ent

    def add_entry():
        ...

    def remove_entry():
        ...

    def add_edge():
        ...

    def remove_edge():
        ...

    def add_attribute():
        ...

    def remove_attribute():
        ...

    def merge(self, db):
        """
        Merge two databases

        Careful - this can lead to overwriting original entries with new ones IF duplicate names are present
        """
        self.entries.update(db.entries)

    def _check_for_duplicates(self, names=[]):
        """
        Internal utility to check for duplicate keys BEFORE updating. This should prevent overwriting
        """
        keys = self.entries.keys()
        dupes = [f for f in keys if f in names]
        if len(dupes) == 0:
            return False
        else:
            return dupes


class nucleophiles(somn_DB):
    """
    Database for storing nucleophiles.

    These entries are "substrate"
    """

    def __init__(self):
        super().__init__(name="", write_dir="", entries=[])


class electrophiles(somn_DB):
    """
    Database for storing electrophiles.

    These entries are "substrate"
    """

    def __init__(self):
        super().__init__(name="", write_dir="", entries=[])


class catalysts(somn_DB):
    """
    Database for catalysts.

    -smiles and CAS should be for the LIGAND, not the complex (many don't have CAS)
    -features should be generated as described for the complexes, and have the metal pruned
    -naming should use *ideally* a common, interpretable, commercial name, but CAS or arbitrary numbers are OK

    """

    def __init__(self):
        super().__init__(name="", write_dir="", entries=[])


class reagents(somn_DB):
    """
    Database for storing information about Solvents or Bases.

    -Entries should be a pairwise combination of solvent/base (important for solubility descriptor)
    -Entries should have CAS and smiles available

    """

    def __init__(self, roles: list):
        super().__init__(name="", write_dir="", entries=[])
        self.solvents = {}
        self.bases = {}
        try:
            for key, val in self.entries.items():
                if val.role == "solvent" or val.role == "solv":
                    self.solvents[key] = val
                elif val.role == "base":
                    self.bases[key] = val
        except:
            raise Exception(
                "Failed to add reagents to reagent database. Check 'role' attribute for reagent entries"
            )
