##Something
from sys import argv
from argparse import ArgumentParser
import somn

use_msg = """
Welcome to the somn command-line interface. 

[NORMAL USE] To make predictions from pre-trained models, use:
predict [project ID with pre-trained models] [model set ID of specific pre-trained models]  [new prediction ID]


[POWER USER USE] To retrain models, two steps are required: generating a particular set of partitions, then optimizing hyperparameters
partition-wise. These are separated so that custom preprocessing or features can be incorporated at the partition step,
and custom modeling changes can be incorporated or tested at the modeling step. New data for training should be 
incorporated into the dataset_yields.hdf5 file for inclusion in the retraining process.

To create new partitions, use:
partition [project ID, new or old project with no partitions] 

To train a new model set on partitions, use:
learn [project ID with partitions] [new ID for model set] > learn[identifier, e.g. '001'].log 2>&1 & disown 
"""
def _build_from_smiles(args):
    """
    Utility to build .mol2 files from smiles - can be used for debugging/inspecting
    """ 
    from somn.build.parsing import InputParser
    input_smi,out_dir = args.options
    p = InputParser(serialize=True,path_to_write=f"somn_mol_buffer/{out_dir}")
    col,smi = p.get_mol_from_smiles(input_smi,names=[f"{out_dir}".strip("/").strip(r"\\")])
    print(f"""
Check the directory somn_mol_buffer/{out_dir} for output files.
    
    """
    )


def _run_predictions(args):
    """
    parse options for predict CLI

    Wrapper that runs main() from predict workflow

    ProjectID, model set ID, and new identifier for predictions are required arguments

    """
    opts = args.options
    from somn.workflows.predict import main as predict

    predict(args=opts)
    print(
        f"Finished generating predictions for project {opts[0]}, model set {opts[1]}, called {opts[2]}."
    )


def _train_models(args):
    """
    Wrapper that trains new models from CLI arguments.
    """
    opts = args.options
    from somn.workflows.learn import main as learn

    try:
        learn(args=opts)
    except:
        Warning(
            f"Looks like {opts} in the learning workfow led to an error. Check if job trained any partitions, \
and if it did, then try re-starting the job with the same input arguments (known memory leak in Keras backend can cause this) \
Otherwise, check input arguments to ensure that a valid project ID was passed."
        )


def _generate_partitions(args):
    """
    Wrapper that generates partitions

    DEV - checked for "last" and "new" operation.
    """
    opts = args.options
    ## DEV
    # print(f"DEV {opts}")
    ##
    from somn.workflows.partition import main as partition, get_precalc_sub_desc
    from somn.workflows.calculate import main as calc_sub
    from somn.workflows.calculate import preprocess
    from copy import deepcopy
    from somn.util.project import Project

    ## IDENTIFY OR CREATE AND INSTANTIATE PROJECT ##
    if opts[0] == "new":
        assert len(opts) >= 2
        project = Project()
        project.save(identifier=opts[1])  ####DEV####
    else:
        try:
            project = Project.reload(how=opts[0])
        except:
            raise Exception(
                "Must pass valid identifier or 'last' to load project. Can also say 'new' and give an identifier"
            )
    ## CHECK FOR OPTIONAL VALIDATION FLAG - USE AS TEMPLATE TO INTRODUCE MORE OPTIONS LATER ##
    if "val" in opts:
        val_schema = opts[opts.index("val") + 1]
    else:
        val_schema = "random"
    # print("DEV - val_schema: ", val_schema)
    assert val_schema in [
        "to_vi",
        "vi_to",
        "random",
        "vo_to",
        "to_vo",
        "noval_to",
        "to_noval",
    ]
    ## BEGINNING PREP - LOAD STATIC DATA/PREREQUISITES TO DESCRIPTORS
    # (
    #     amines,
    #     bromides,
    #     dataset,
    #     handles,
    #     unique_couplings,
    #     a_prop,
    #     br_prop,
    #     base_desc,
    #     solv_desc,
    #     cat_desc,
    # ) = preprocess.load_data(optional_load="maxdiff_catalyst")

    # Checking project status to make sure sub descriptors are calculated
    sub_desc = get_precalc_sub_desc()
    if sub_desc == False:  # Need to calculate
        real, rand, unique_couplings, dataset = calc_sub(
            project, optional_load="maxdiff_catalyst", substrate_pre=("corr", 0.90)
        )
        sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc = real
    else:  # Already calculated descriptors, just fetching them
        (
            amines,
            bromides,
            dataset,
            handles,
            unique_couplings,
            a_prop,
            br_prop,
            base_desc,
            solv_desc,
            cat_desc,
        ) = preprocess.load_data(optional_load="maxdiff_catalyst")
        sub_am_dict, sub_br_dict, rand = sub_desc
        real = (sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc)
    combos = deepcopy(
        unique_couplings
    )  # This is a guide for building the partitions out - really, any set of combinations can be specified.
    # The amine_bromide individual elements in this list will direct out-of-sample partitioning (if relevant in val_schema).
    # Any specific set of out-of-sample partitions can be designed and introduced here. This *could* be an optional input from
    # the user (e.g. a csv file with a list of items in it).
    import os

    # print(pd.DataFrame(combos).to_string())
    outdir = deepcopy(f"{project.partitions}/")
    os.makedirs(outdir + "real/", exist_ok=True)
    # os.makedirs(outdir + "rand/", exist_ok=True)
    realout = outdir + "real/"
    # randout = outdir + "rand/"
    project.combos = combos
    project.unique_couplings = unique_couplings
    project.dataset = dataset
    partition(
        project,
        val_schema=val_schema,
        vt=0,
        mask_substrates=True,
        rand=rand,
        real=real,
        serialize_rand=False,
    )
    print(
        f"Finished calculating partitions for {project.unique}, stored at {project.partitions}."
    )


def _calculate_descriptors(args):
    """
    Calculate substrate descriptors for an input file.
    """
    from somn.calculate.substrate import calculate_prophetic
    from pathlib import Path
    from somn.util.project import Project
    from somn.workflows.calculate import calculate_substrate_descriptors

    p = Project()  # Local directory tree will be generated
    p.save()
    opts = args.options
    try:
        assert Path(opts[0]).exists()
        requests = opts[0]
        if len(opts) == 3:
            concurrent = int(opts[1])
            nprocs = int(opts[2])
        elif len(opts) == 1:
            concurrent = 2
            nprocs = 2
        else:
            raise Exception(
                "Looks like the improper number of arguments was passed to the calculate operation. \
Please check your arguments: somn calculate [path_to_csv] [optional: concurrent jobs] \
[optional: nprocs per job]"
            )
        calculate_substrate_descriptors(requests, concurrent=concurrent, nprocs=nprocs)
    except:
        raise Exception(
            f"Check input to descriptor calculation request - it seems something went wrong. Some suggestions: \
Check command arguments: Pass a file path to requested molecules, then the number of concurrent \
calculations to run, followed by the number of processors that can be spared for each concurrent job. \
Check the input file: should contain 3 columns, with each row corresponding to a molecule, and containing \
a unique name, a SMILES string, and the type of reactant ('N', 'Br', or 'Cl'). Filepath passed was :\
{opts[0]}."
        )


def _add_(args):
    """
    parse options
    """
    ...


def _visualize_(args):
    """
    parse options
    """
    ...


helpblock = f"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SOMN CLI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the somn CLI, select an operation (e.g. add, calculate, partition, learn, or predict) followed by the
operation-specific arguments. Documentation should describe the required arguments.

Here is a simple summary/guide:

{use_msg}"""


splash = f"""
                                      
                                               
           ____    ___     ___ ___     ___     
          /',__\  / __`\ /' __` __`\ /' _ `\   
         /\__, `\/\ \L\ \/\ \/\ \/\ \/\ \/\ \  
         \/\____/\ \____/\ \_\ \_\ \_\ \_\ \_\ 
          \/___/  \/___/  \/_/\/_/\/_/\/_/\/_/ 
  

         (C) 2023 by N. Ian Rinehart, Ph.D. in the 
         laboratories of Prof. Scott E. Denmark at
         the University of Illinois Urbana-Champaign
        in collaboration with F. Hoffmann-La Roche AG

        
"""


def main():
    # print("DEV - module")
    print(splash)
    parser = ArgumentParser(usage=use_msg)
    parser.add_argument(
        nargs="?",
        choices=[
            "predict",
            "partition",
            "learn",
            "calculate",
            "initialize",
            "add",
            "visualize",
            "help",
            "generate",
        ],
        dest="operation",
        default="help",
    )
    parser.add_argument(
        dest="options",
        nargs="*",
        default=False,
    )
    args = parser.parse_args()
    # print(args)
    if args.operation == "predict":  ## Make predictions
        try:
            _run_predictions(args)
        except:
            raise Warning(
                f"Looks like handling the predict workflow with the arguments: [{args.options}] failed. \
Check that a project ID, model set ID, and a new identifier are present (in that order)."
            )
    elif args.operation == "help":
        print(helpblock)
    elif args.operation == "learn":  ## Train models
        try:
            _train_models(args)
        except:
            Warning(
                f"Looks like handling arguments for model training failed with {args.options}. \
Ensure that a project ID and a new, unique model set ID are being passed (in that order)."
            )
    elif args.operation == "partition":  ## Generate partitions
        try:
            _generate_partitions(args)
        except:
            Warning(
                f"Looks like handling the partition arguments {args.options} led to an error. \
Ensure that project ID is provided, or specify 'new'."
            )
    elif args.operation == "calculate":
        try:
            _calculate_descriptors(args)
        except:
            Warning(
                f"Looks like handling the 'calculate' arguments {args.options} led to an error. \
Ensure that a valid path to reactants is provided."
            )
    elif (
        args.operation == "initialize"
    ):  ## Set up somn_scratch for the first time, test install
        ## PROJECT CLASS LOADED FOR FIRST TIME - WILL MAKE SOMN_SCRATCH DIRECTORY
        from somn.util.project import Project

        # p = Project()
        # p.save(identifier="initialization")
        ## Look for and load projects.JSON & pre-trained models
        if "models" in args.options:
            from pathlib import Path
            import subprocess
            import os
            import json

            os.makedirs("somn_scratch/", exist_ok=True)
        try:
            assert Path("./pretrained-somn.tar.gz").exists()
            assert Path("./projects.JSON").exists()
            ## EXTRACT MODELS INTO SOMN SCRATCH DIRECTORY
            print("\n\nExtracting pre-trained models...\n\n")
            subprocess.run(
                ["tar", "-xzvf", "pretrained-somn.tar.gz", "-C", "somn_scratch/"]
            )
            # print(
            #     "\n\nModels successfully extracted! Now updating package with their location...\n\n"
            # )
            ## LOCATE INSTALL PATH FOR DATA MODULE & UPDATE projects.JSON
            # data_module_path = os.path.dirname(somn.data.__file__)
            # with open(f"./projects.JSON", "r") as k:
            #     upd = json.load(k)
            # with open(f"{data_module_path}/projects.JSON", "r") as g:
            #     proj = json.load(g)
            # proj.update(upd)
            # with open(f"{data_module_path}/projects.JSON", "w") as p:
            #     json.dump(proj, p)
#             print(
#                 "somn package has been installed with pre-trained models. Please look in the somn_scratch directory \
# to find the project '44eb8d94effa11eea46f18c04d0a4970', and look in the 'scratch' subdirectory for an example prediction request input file."
#             )
        except:
            import warnings

            warnings.warn(
                "It looks like no pre-trained models were supplied; skipping initialization step. \
if this is an error, please check documentation and ensure that all files are in \
the somn home directory (at the same level as the somn initialize command is run)"
            )
    elif args.operation == "generate":
        _build_from_smiles(args)
    
    elif args.operation in ["add", "visualize"]:
        raise Exception(
            f"DEV - {args.operation} implementation through CLI is under development"
        )
