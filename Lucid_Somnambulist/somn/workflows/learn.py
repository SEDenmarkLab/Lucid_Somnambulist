from somn.learn.learning import hypermodel_search
from somn.util.project import Project
from sys import argv


if __name__ == "__main__":
    project = Project.reload(how="last")
    hypermodel_search(experiment="testing_search04")
    # experiment = argv[1]
