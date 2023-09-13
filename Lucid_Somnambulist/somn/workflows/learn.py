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

    # try:
    #     raw_predictions, requests = hypermodel_search(
    #         project=project,
    #         model_experiment=args.mdl,
    #         prediction_experiment=args.exp,
    #     )
    # except:
    #     raise RuntimeError(
    #         "Something went wrong with hypermodel inferences - check model experiment ID, and \
    #                        if prediction experiment label is unique and new."
    #     )
    # stamp = str(date.today())
    # plot_preds(query="all", prediction_experiment=args.exp, requestor=stamp)
    # print(
    #     f"Finished with predictions - please see {project.output}/{args.exp}_rawpredictions.csv \n \
    # Processed predictions are in {project.output}/{args.exp}/{stamp}/ "
    # )


if __name__ == "__main__":
    project = Project.reload(how="last")
    hypermodel_search(experiment="full_search")
    # experiment = argv[1]
