import molli as ml

# from somn.database import dtypes
import json
import pickle
from pathlib import Path, PurePath

from collections import defaultdict
from dataclasses import dataclass
from attrs import define, field


@define
class Organizer:
    """
    Basic class for organizing and tracking the serialization of data.

    This is "pointed" to by any somn_DB class object, and then when that object is serialized, this organizer
    will have a record of when/where. Then, this object should be used later to gracefully reconstitute evertyhing.
    """

    name: str
    write_path: str
    data: defaultdict

    def add(self, db_name="", db_path="", db_type=None, db_subtype=None):
        """
        Add entries, has support for top-level, singly- and doubly-nested entries.

        For utility - top level entries should be notes about experiment numbers.
        """

        def _check_key(k: str, d: dict):
            if k in d.keys():
                return True
            elif k not in d.keys():
                d[k] = {}
                return False

        if isinstance(db_type, str) and db_subtype == None:
            if _check_key(db_type, self.data):
                self.data[db_type][db_name] = db_path
            else:
                self.data[db_type][db_name] = db_path
        if isinstance(db_type, str) and isinstance(db_subtype, str):
            if _check_key(
                db_type, self.data
            ):  # Makes empty 1st nested dict if it doesn't exist
                if _check_key(
                    db_subtype, self.data[db_type]
                ):  # Should run ONLY because dict is defined in parent if; defines empty dict if necessary
                    self.data[db_type][db_name][db_subtype][db_name] = db_path
                else:
                    self.data[db_type][db_name][db_subtype][db_name] = db_path
            else:
                self.data[db_type][
                    db_subtype
                ] = (
                    {}
                )  # Need to define - not checked in if before, but db_type key is set to empty dict
                self.data[db_type][db_subtype][db_name] = db_path
        elif db_type == None:  # top level - no child levels, so list
            if db_name in self.data.keys():
                self.data[db_name].append(db_path)
            else:
                self.data[db_name] = [db_path]
        else:
            raise Exception("Wrong type passed to organizer.add - need string or None")

    def save(self):
        """
        Serialize hashable organizer.
        """
        try:
            self.data["name"] = self.name
            self.data["write"] = self.write_path
            _s = json.dumps(self.data)
        except:
            raise Exception(
                "Failed to dump json - nonhashable Organizer object probably passed."
            )
        write = self.write_path + self.name
        with open(write + ".json", "w") as k:
            k.write(_s)

    def get_new_name(self, db_type=None, db_subtype=None, name=None):
        """
        Get a new name for next experiment - prevents overwriting
        """
        if db_type == None and db_subtype == None:
            temp = self.data.values()
        elif isinstance(db_type, str) and db_subtype == None:
            try:
                temp = self.data[db_type].values()
            except:
                raise Exception("That dbtype is not in organizer")
        elif isinstance(db_type, str) and isinstance(db_subtype, str):
            try:
                temp = self.data[db_type][db_subtype].values()
            except:
                raise Exception("That dbSUBtype is not in organizer")
        else:
            raise Exception("database type/subtypes must be strings or None")
        if name == None:  # auto detect experiment name - try to infer
            bases = [
                PurePath(f).stem for f in temp
            ]  # values should be paths - this list should be the filenames
            vals = [
                int(f.split("_")[-1]) for f in bases
            ]  # Use number at end of filenames separated by "_", then increment
            new_v = max(vals) + 1
            base_idx = vals.index(max(vals))
            new_name = bases[base_idx].rsplit("_", 1)[0] + f"_{new_v}"
            return new_name

    # Will have redundant entries in dict for name and path, but this makes read/write easier
    @classmethod
    def load(cls, path_to_file):
        with open(path_to_file, "r") as g:
            temp = dict(json.load(g))
            return cls(temp["name"], temp["write"], temp)


@define
class somn_DB:
    name: str
    write_dir: str
    organizer: Organizer
    data: defaultdict

    @classmethod
    def build(
        cls, organizer: Organizer, db_types: tuple, write_dir="", entries=[], name=None
    ):
        """
        Build function for somn_DB objects - pass list of entry objects

        MUST pass tuple of length 2 for types_, even None is acceptable
        """
        _dict = {}
        assert len(db_types) == 2
        db_type, db_subtype = db_types
        if name == None:
            name = organizer.get_new_name(db_type=db_type, db_subtype=db_subtype)
        for i, ent in enumerate(entries):
            _dict[ent.name] = ent
        return cls(name, write_dir, organizer, _dict)

    @property
    def unique_couplings(self):
        return sorted(
            list(set([f.strip().rsplit("_", 3)[0] for f in self.data.keys()]))
        )

    def add_entry(self, k, v):
        self.data[k] = v

    def remove_entry(self, k):
        del self.data[k]

    def merge(self, db):
        """
        Merge two databases

        Careful - this can lead to overwriting original entries with new ones IF duplicate names are present
        """
        if bool(set(self.data.keys()) & set(db.data.keys())) == True:  # Matching keys
            raise Exception("Trying to overwrite entries during merge!")
        self.data.update(db.data)

    def save(self):
        write_path = self.write_path + self.name + ".p"
        if Path.exists(write_path):
            raise UserWarning("May overwrite an existing database - check code")
        with open(write_path, "wb") as g:
            pickle.dump(self, g)

    @classmethod
    def load(read_path="", name=""):
        with open(read_path + name + ".p", "rb") as g:
            db = pickle.load(g)
            return db


@define
class nucleophiles(somn_DB):
    """
    Database for storing nucleophiles.

    These entries are "substrate"
    """

    @classmethod
    def build(
        cls, organizer: Organizer, db_types: tuple, write_dir="", entries=[], name=None
    ):
        """
        Build function for somn_DB objects - pass list of entry objects

        MUST pass tuple of length 2 for types_, even None is acceptable
        """
        _dict = {}
        assert len(db_types) == 2
        db_type, db_subtype = db_types
        if name == None:
            name = organizer.get_new_name(db_type=db_type, db_subtype=db_subtype)
        for i, ent in enumerate(entries):
            _dict[ent.name] = ent
        return cls(name, write_dir, organizer, _dict)


@define
class electrophiles(somn_DB):
    """
    Database for storing electrophiles.

    These entries are "substrate"
    """

    @classmethod
    def build(
        cls, organizer: Organizer, db_types: tuple, write_dir="", entries=[], name=None
    ):
        """
        Build function for somn_DB objects - pass list of entry objects

        MUST pass tuple of length 2 for types_, even None is acceptable
        """
        _dict = {}
        assert len(db_types) == 2
        db_type, db_subtype = db_types
        if name == None:
            name = organizer.get_new_name(db_type=db_type, db_subtype=db_subtype)
        for i, ent in enumerate(entries):
            _dict[ent.name] = ent
        return cls(name, write_dir, organizer, _dict)


@define
class catalysts(somn_DB):
    """
    Database for catalysts.

    -smiles and CAS should be for the LIGAND, not the complex (many don't have CAS)
    -features should be generated as described for the complexes, and have the metal pruned
    -naming should use *ideally* a common, interpretable, commercial name, but CAS or arbitrary numbers are OK

    """

    @classmethod
    def build(
        cls, organizer: Organizer, db_types: tuple, write_dir="", entries=[], name=None
    ):
        """
        Build function for somn_DB objects - pass list of entry objects

        MUST pass tuple of length 2 for types_, even None is acceptable
        """
        _dict = {}
        assert len(db_types) == 2
        db_type, db_subtype = db_types
        if name == None:
            name = organizer.get_new_name(db_type=db_type, db_subtype=db_subtype)
        for i, ent in enumerate(entries):
            _dict[ent.name] = ent
        return cls(name, write_dir, organizer, _dict)


@define
class reagents(somn_DB):
    """
    Database for storing information about Solvents or Bases.

    -Entries should be a pairwise combination of solvent/base (important for solubility descriptor)
    -Entries should have CAS and smiles available

    """

    @classmethod
    def build(
        cls, organizer: Organizer, db_types: tuple, write_dir="", entries=[], name=None
    ):
        """
        Build function for somn_DB objects - pass list of entry objects

        MUST pass tuple of length 2 for types_, even None is acceptable
        """
        _dict = {}
        assert len(db_types) == 2
        db_type, db_subtype = db_types
        if name == None:
            name = organizer.get_new_name(db_type=db_type, db_subtype=db_subtype)
        for i, ent in enumerate(entries):
            _dict[ent.name] = ent
        return cls(name, write_dir, organizer, _dict)
