from pathlib import Path
import shutil as su
from uuid import uuid1


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

    def __getstate__(cls):
        state = cls.__dict__.copy()
        state["path"] = cls.path
        print(state)
        return state

    def __setstate__(cls, state):
        cls.__dict__.update(state)

    def save(cls, path=""):
        """
        Saves the details of project to a JSON
        """
        from datetime import date

        timestamp = date.today()
        print(timestamp)
        output = {
            "path": rf"{cls.path}/",
            "timestamp": rf"{timestamp}",
            "unique": rf"{cls.unique}",
        }
        print(output)
        import pkg_resources
        import json

        pkg = pkg_resources.resource_filename("somn.data", "projects.JSON")
        with open(pkg, "r") as g:
            projects = json.load(g)
            last_ = max(list(map(int, projects.keys())))
            projects[last_ + 1] = output
        with open(pkg, "w") as k:
            json.dump(projects, k)

    @classmethod
    def reload(cls, entry: dict):
        """
        Used to reload a specific instance (need the pass the unique ID)

        Does not make directories, but still should operate as a singleton.
        """
        path = entry["path"]
        unique = entry["unique"]
        timestamp = entry["timestamp"]
        if path == None:
            raise Exception("reloading project requires specific path to be specified")
        else:
            cls._instance = super(Project, cls).__new__(cls)
            cls.path = path
            cls.unique = unique
            cls.timestamp = timestamp
            cls.partitions = Path(f"{path}/partitions/")
            cls.scratch = Path(f"{path}/scratch/")
            cls.descriptors = Path(f"{path}/descriptors/")
            cls.structures = Path(f"{path}/structures/")
            cls.output = Path(f"{path}/outputs/")
        return cls._instance


### TESTING
# def __enter__(self):
#     print("Firing up the working directory tracker")
#     self.path.mkdir(exist_ok=True, parents=True)
#     # Here's the part where you get to make your subfolders
#     self.stuff = self.path / "stuff"
#     self.stuff.mkdir(exist_ok=True)
#     return self

# def __exit__(self, *things):
#     print("Shutting down the party")
#     print("results:", things)
#     if (vipfile := self.stuff / "shit.txt").is_file():
#         with open("output.txt", "wb") as f:
#             f.write(vipfile.read_bytes())

# su.rmtree(self.path)


if __name__ == "__main__":
    # k = Project()
    # k.save()

    import json

    p = json.load(
        open(
            "somn_container/Lucid_Somnambulist/Lucid_Somnambulist/somn/data/projects.JSON",
            "r",
        )
    )
    l = Project.reload(entry=p["1"])
    print(l.unique)
    print(l.path)
    print(l.timestamp)
    print(p["1"])
