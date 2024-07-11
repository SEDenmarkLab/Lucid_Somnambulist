from pathlib import Path
import shutil as su
from uuid import uuid1
import json
from json import JSONDecoder
from collections import OrderedDict
from attrs import define, field


class Project(object):
    """
    Experimental singleton object for organizing serialization of data.

    Generates unique ID and creates subfolders that can be used to write things.

    Writes unique IDs to a JSON, and these can be retrieved later, and used to rebuild a functionally
    equivalent instance to a previous singleton instance. This prevents overwriting data.

    Only gets a timestamp if it's been reloaded
    """

    _instance = None

    def __new__(cls, path="./somn_scratch/"):
        if cls._instance is None:
            cls._instance = super(Project, cls).__new__(cls)
            unique = uuid1().hex
            cls.unique = unique
            _path = Path(path) / unique
            cls.path = _path
            cls.partitions = Path(f"{_path}/partitions/")
            cls.scratch = Path(f"{_path}/scratch/")
            cls.descriptors = Path(f"{_path}/descriptors/")
            cls.structures = Path(f"{_path}/structures/")
            cls.output = Path(f"{_path}/outputs/")
            Path(cls.path).mkdir(parents=True, exist_ok=False)
            Path(cls.partitions).mkdir(parents=True, exist_ok=False)
            Path(cls.structures).mkdir(parents=True, exist_ok=False)
            Path(cls.descriptors).mkdir(parents=True, exist_ok=False)
            Path(cls.output).mkdir(parents=True, exist_ok=False)
            Path(cls.scratch).mkdir(parents=True, exist_ok=False)
        return cls._instance

    def save(cls, identifier=None):
        """
        Saves the details of project to a JSON
        """
        from datetime import date

        timestamp = date.today()
        # print(timestamp) ## DEBUG
        output = {
            "path": rf"{cls.path}/",
            "timestamp": rf"{timestamp}",
            "unique": rf"{cls.unique}",
        }
        # print(output) ## DEBUG
        ## Identifier gets added IF it is known (i.e. user specifies a special name). Placeholder for now.
        if type(identifier) == str:
            output["identifier"] = identifier

        pkg = cls.get_json()
        with open(pkg, "r") as g:
            projects = json.load(g, object_pairs_hook=OrderedDict)
            # last_ = max(list(map(int, projects.keys())))
        if cls.unique in projects.keys():
            import warnings

            #             warnings.warn(
            #                 f"The identifier {cls.unique} is already a known project: check prior work with this identifier. \
            # Saving a preexisting project is not necessary and changes the order of projects.JSON, \
            # as well as posing risks for errors. Project has not been saved again."
            #             )
            return None
        projects[cls.unique] = output
        with open(pkg, "w") as k:
            json.dump(projects, k, indent=4)

    @staticmethod
    def get_json():
        """
        Get package JSON path to look up projects
        """
        import pkg_resources

        pkg = pkg_resources.resource_filename("somn.data", "projects.JSON")
        return pkg

    @classmethod
    def reload(cls, how=""):
        """
        IF
        how = "last"
        return most recent entry
        OR
        how = [unique ID]
        return entry for specific identifier
        """

        def __load_entry(cls, entry: dict):
            """
            Used to reload a specific instance (need the pass the unique ID)
            """
            path = entry["path"]
            unique = entry["unique"]
            timestamp = entry["timestamp"]
            cls._instance = super(Project, cls).__new__(cls)
            cls.path = path
            cls.unique = unique
            cls.timestamp = timestamp
            cls.partitions = Path(f"{path}/partitions/")
            cls.scratch = Path(f"{path}/scratch/")
            cls.descriptors = Path(f"{path}/descriptors/")
            cls.structures = Path(f"{path}/structures/")
            cls.output = Path(f"{path}/outputs/")
            if (
                "latest" in entry.keys()
            ):  ## Allows 'latest' alias to be passed in order to specify a specific model set.
                cls.latest = entry["latest"]
            return cls._instance

        pkg = cls.get_json()
        with open(pkg, "r") as g:
            projects = json.load(g, object_pairs_hook=OrderedDict)
        # print(f"KEYS, {projects.keys()}")
        if how == "last":
            last_entry = projects.popitem(last=True)[1]
            # print(last_entry) ## DEBUG
            last_instance = __load_entry(cls, entry=last_entry)
            return last_instance
        elif how in projects.keys():
            entry = projects[how]
            # print(entry)  ## DEBUG
            return __load_entry(cls, entry=entry)
        else:
            raise ValueError(
                f"Did not find {how} in projects.JSON; cannot look up files."
            )
