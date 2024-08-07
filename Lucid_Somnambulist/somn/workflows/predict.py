from somn.learn.inference import hypermodel_inference
from argparse import ArgumentParser
from sys import argv
from somn.util.project import Project
from somn.util.visualize import plot_preds
from datetime import date

"""
    parser = argparse.ArgumentParser(
        usage="Specify format (smi or cdxml), then a smiles string/file with smiles or cdxml file, and finally indicate 'el' or 'nuc' for electrophile or nucleophile. Optionally, serialize output structures with '-ser' - must pass some input as an argument after, standard use is 'y'"
    )
    parser.add_argument(
        "fmt",
        nargs=2,
        help="Format for the input - cdxml or smi, followed by file or smiles string",
    )
    parser.add_argument(
        "r", help="The type of input - nucleophile (nuc) or electrophile (el)"
    )
    parser.add_argument(
        "-ser",
        help="Optional serialize argument, pass -ser t",
    )
    args = parser.parse_args(parser_args)
"""


def main(args=None):
    """
    Inference pipeline wrapper with argument handling.

    """
    parser = ArgumentParser(usage=f"Specify [projectID] [modelset ID] [prediction ID]")
    parser.add_argument(
        "proj",
        help="Project ID (uuid string) where pretrained models are stored",
    )
    parser.add_argument(
        "mdl",
        help="Model set id - this is a unique identifier set during model retraining. It will be used to fetch a specific \
            model set.",
    )
    parser.add_argument(
        "exp",
        help="This is a NEW label that will be used to identify this set of predictions.",
    )
    args = parser.parse_args(args=args)
    try:
        project = Project.reload(how=args.proj)
    except:
        raise ValueError("Wrong project ID - could not load")
    try:
        if args.mdl == "latest":
            model_experiment = project.latest
        else:
            model_experiment = args.mdl
        raw_predictions, requests = hypermodel_inference(
            project=project,
            model_experiment=model_experiment,
            prediction_experiment=args.exp,
            all_predictions=False,
            vt=0,  # Default value
        )
    except:
        raise RuntimeError(
            "Something went wrong with hypermodel inferences - check model experiment ID, and \
if prediction experiment label is unique and new."
        )
    stamp = 'couplings'
    try:
        import pathlib
        assert not pathlib.Path(f"{project.output}/{args.exp}/{stamp}/").exists()
    except:
        raise Exception(f"Invalid prediction output path specified - already exists! {project.output}/{args.exp}/{stamp}/")
    plot_preds(query="all", prediction_experiment=args.exp, requestor=stamp)
    print(
        f"Finished with predictions - please see {project.output}/{args.exp}_rawpredictions.csv \n \
Processed predictions are in {project.output}/{args.exp}/{stamp}/ "
    )


def check_input_structures():
    """
    Method to check input structures for multiple reaction sites, and to organize a return to the user for them to \
    specify more detail. 

    Strategy is to check if the "Checked" file output is already created, and then to either terminate the run or \
    continue. If terminating, then the idea is to gracefully succeed when the predict call is run again. 
    """
    from somn.learn.inference import prep_requests

    total_requests, requested_pairs = prep_requests()
    from somn.workflows.add import add_workflow
    from pathlib import Path

    ...


# if __name__ == "__main__":
#     args = argv[1:]
#     if len(args) == 1:
#         raise Exception(
#             "Trying to make predictions, but did not specify input values. Check main() from somn.workflows.predict."
#         )
#     try:
#         assert len(args) == 3
#     except:
#         raise Exception(
#             "Inference workflow received the wrong number of arguments.\n \
#             Specify project ID, then model set ID, then a new, unique \n \
#             identifier for the predictions being made. "
#         )
#     main(args=args)
