import tensorflow as tf
import pandas as pd
from glob import glob
from somn.build.assemble import assemble_descriptors_from_handles
from datetime import date
from somn import Project
from somn.learn.learning import tf_organizer, tfDriver
from somn.build.assemble import (
    load_calculated_substrate_descriptors,
    load_substrate_masks,
)
from somn.calculate.preprocess import load_data
from somn.workflows.partition import (
    check_sub_status,
    fetch_precalc_sub_desc,
    get_precalc_sub_desc,
)


def hypermodel_inference(
    project: Project,
    request_dict: dict,
    model_experiment="",
    prediction_experiment="",
    optional_load="maxdiff_catalyst",
    real=True,
):
    """
    project must contain (1) partitions, and (2) pre-trained hypermodels.
    """
    if real == True:
        organ = tf_organizer(
            f"inference_{model_experiment}",
            partition_dir=f"{project.partitions}/real",
            validation=True,
            inference=True,
        )
    elif real == False:
        try:
            organ = tf_organizer(
                f"inference_{model_experiment}",
                partition_dir=f"{project.partitions}/rand",
                validation=True,
                inference=True,
            )
        except:
            raise Exception(
                "Attempted to make inferences using models trained on random features, but could not find/load those models. Check partitions"
            )
    else:
        raise Exception(
            "Must specify real = True or False, representing whether inferences should be made using 'real' chemical descriptors, or random ones (as a control)"
        )
    drive = tfDriver(organ)
    model_dir = f"{project.output}/{model_experiment}/out/"
    pred_buffer_fp = f"{project.output}prediction_buffer_{model_experiment}_{prediction_experiment}.csv"
    # sub_desc, rand = load_calculated_substrate_descriptors()
    ### Get descriptors so that features for predictions can be assembled
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
    ) = load_data(optional_load=optional_load)
    sub_desc = get_precalc_sub_desc()
    if sub_desc == False:  # Need to calculate
        raise Exception(
            "Tried to load descriptors for inference, but could not locate pre-calcualted descriptors. This could lead to problems with predictions; check input project."
        )
    else:
        real, rand = sub_desc
    masks = load_substrate_masks()
    total_requests = prep_requests()
    assemble_desc_for_inference_mols(
        project=project,
        requests=f"{project.scratch}/all_requests.csv",
        organizer=organ,
        masks=masks,
    )


def prep_requests():
    """
    Get requested predictions
    """
    files = glob(f"{Project().scratch}/*_request.csv")
    assert (
        len(files) > 0
    )  ### DEBUG - If this fails, the user is PROBABLY running this from the wrong "home" directory
    # Quick check for formatting
    df = pd.read_csv(files[0], header=0, index_col=None)
    if len(df.columns) < 2:
        raise Exception(
            "Must pass SMILES and role for each reactant! Request input file (in gproject.scratch)\
                        Shoult have format (col0):SMILES,(col1):role (nuc or el),(col2, optional):mol_name"
        )
    tot = []
    for i, file in enumerate(files):
        if i == 0:
            df = pd.read_csv(files[0], header=0, index_col=None)
            if len(df.columns) < 2:
                raise Exception(
                    "Must pass SMILES and role for each reactant! Request input file (in gproject.scratch)\
                                Shoult have format (col0):SMILES,(col1):role (nuc or el),(col2, optional):mol_name"
                )
        tot.append(df)
    total_requests = pd.concat(tot, axis=0)
    total_requests.to_csv(f"{Project().scratch}/all_requests.csv", header=True)
    req_pairs = []
    for row in total_requests.iterrows():
        data = row[1].values
        pair = f"{data[3]}_{data[4]}"
        req_pairs.append(pair)
    # print(",".join(req_pairs))

    return total_requests, req_pairs


def assemble_desc_for_inference_mols(
    project: Project,
    requests: str,
    organizer: tf_organizer,
    masks: tuple,
    prediction_experiment: str,
    pred_str,
):
    """
    pipeline to generate feature arrays for inference molecules
    """
    ### Get molecular geometries first
    from somn.workflows.add import add_workflow

    args = ["multsmi", requests, "mult", "-ser", "t"]
    try:
        add_workflow(
            project=project,
            prediction_experiment=prediction_experiment,
            parser_args=args,
        )
    except:
        raise Exception(
            "Something went wrong with calculating substrate descriptors for new molecules - check inputs"
        )
    ### Now we're ready to calculate RDF features
    from somn.workflows.firstgen_calc_sub import calculate_prophetic
    import molli as ml
    import json

    prophetic_amine_col = ml.Collection.from_zip(
        f"{project.structures}/{prediction_experiment}/prophetic_amine.zip"
    )
    prophetic_bromide_col = ml.Collection.from_zip(
        f"{project.structures}/{prediction_experiment}/prophetic_bromide.zip"
    )
    p_ap = json.load(
        open(f"{project.structures}/{prediction_experiment}/newmol_ap_buffer.json")
    )

    p_a_desc = calculate_prophetic(
        inc=0.75, geometries=prophetic_amine_col, atomproperties=p_ap, react_type="am"
    )
    p_b_desc = calculate_prophetic(
        inc=0.75, geometries=prophetic_bromide_col, atomproperties=p_ap, react_type="br"
    )
    ### Now we're ready to assemble features


if __name__ == "__main__":
    project = Project.reload(how="cc3d1f3a3d9211eebdbe18c04d0a4970")
    # raise Exception("DEBUG")
    tot, pred_str = prep_requests()

    # raise Exception("DEBUG")

    organ = tf_organizer(
        name="testing", partition_dir=f"{project.partitions}/real", inference=True
    )
    masks = load_substrate_masks()
    assemble_desc_for_inference_mols(
        project,
        f"{project.scratch}/all_requests.csv",
        organ,
        masks,
        prediction_experiment="testing-03",
        pred_str=pred_str,
    )
