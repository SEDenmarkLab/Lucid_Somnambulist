from somn.learn.learning import hypermodel_search
from somn.util.project import Project
from sys import argv
from argparse import ArgumentParser


def main(args=None):
    """
    Learn workflow wrapper with argument handling.

    Need to make sure that "last" works

    """
    parser = ArgumentParser(usage=f"Specify [projectID] [new model ID]")
    parser.add_argument(
        "proj",
        help="Project ID (uuid string) where pretrained models are stored (or 'last')",
    )
    parser.add_argument(
        "exp",
        help="Model set id - this is a new, unique identifier that will be used later to fetch these models",
    )
    args = parser.parse_args(args=args)
    # print("DEV", args)
    try:
        project = Project.reload(how=args.proj)
    except:
        raise ValueError("Wrong project ID - could not load")
    try:
        hypermodel_search(experiment=args.exp)
    except:
        raise RuntimeError(
            "Hypermodel search failed - check output, and try restarting if necessary."
        )


if __name__ == "__main__":
    project = Project.reload(how="last")
    hypermodel_search(experiment="full_search")
    # experiment = argv[1]
