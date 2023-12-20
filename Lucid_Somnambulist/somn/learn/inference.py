import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
    # request_dict: dict,
    model_experiment="",
    prediction_experiment="",
    optional_load="maxdiff_catalyst",
    substrate_pre=("corr", 0.90),
    vt=0,
    # real=True,
):
    """
    project must contain (1) partitions, and (2) pre-trained hypermodels.
    """
    # if real == True:
    #     organ = tf_organizer(
    #         f"inference_{model_experiment}",
    #         partition_dir=f"{project.partitions}/real",
    #         validation=True,
    #         inference=True,
    #     )
    # elif real == False:
    #     try:
    #         organ = tf_organizer(
    #             f"inference_{model_experiment}",
    #             partition_dir=f"{project.partitions}/rand",
    #             validation=True,
    #             inference=True,
    #         )
    #     except:
    #         raise Exception(
    #             "Attempted to make inferences using models trained on random features, but could not find/load those models. Check partitions"
    #         )
    # else:
    #     raise Exception(
    #         "Must specify real = True or False, representing whether inferences should be made using 'real' chemical descriptors, or random ones (as a control)"
    #     )
    # drive = tfDriver(organ)
    # model_dir = f"{project.output}/{model_experiment}/out/"
    # pred_buffer_fp = f"{project.output}prediction_buffer_{model_experiment}_{prediction_experiment}.csv"
    # # sub_desc, rand = load_calculated_substrate_descriptors()
    # ### Get descriptors so that features for predictions can be assembled
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
    # ) = load_data(optional_load=optional_load)
    # sub_desc = get_precalc_sub_desc()
    # if sub_desc == False:  # Need to calculate
    #     raise Exception(
    #         "Tried to load descriptors for inference, but could not locate pre-calcualted descriptors. This could lead to problems with predictions; check input project."
    #     )
    # else:
    #     real, rand = sub_desc
    output_buffer = []

    from somn.workflows.calculate import main as calc_sub

    sub_masks = load_substrate_masks()
    total_requests, requested_pairs = prep_requests()
    real, rand = calc_sub(
        project, substrate_pre=substrate_pre, optional_load=optional_load
    )
    pred_str = ",".join(requested_pairs)
    ### Building prophetic feature array with matching substrate and other preprocessing
    prophetic_raw = assemble_desc_for_inference_mols(
        project=project,
        requests=f"{project.scratch}/all_requests.csv",
        sub_masks=sub_masks,
        desc=real,
        prediction_experiment=prediction_experiment,
        pred_str=pred_str,
    )
    prophetic_fp = f"{project.descriptors}/prophetic_{prediction_experiment}.feather"
    try:
        import pathlib

        assert pathlib.Path(prophetic_fp).exists()
    except:
        raise Exception(
            f"Warning, the filepath {prophetic_fp} is not real - something went wrong with \
                        generation of the prophetic feature array. Check project directory for \
                        {project.unique}"
        )
    ### Raw feature arrays are assembled. Now, partition-specific preprocessing is needed.
    from somn.calculate.preprocess import preprocess_prophetic_features

    prophetic_organizer = preprocess_prophetic_features(
        project=project,
        features=prophetic_raw.transpose(),
        prediction_experiment=prediction_experiment,
        vt=vt,
    )

    # model_info = [prophetic_organizer.get_partition_info(m)[0] for m in all_models]
    # prophetic_organizer.models = sorted(
    #     list(glob(f"{project.output}/{model_experiment}/out/*.h5"))
    # )
    all_models = sorted(list(glob(f"{project.output}/{model_experiment}/out/*.h5")))
    prophetic_driver = tfDriver(
        organizer=prophetic_organizer, prophetic_models=all_models
    )
    # prophetic_driver.sort_inference_models(all_models)
    # print(
    #     "DEBUG - inference is",
    #     prophetic_driver.organizer.inference,
    #     prophetic_organizer.inference,
    #     prophetic_driver.models,
    #     len(prophetic_driver.models),
    #     len(prophetic_driver.prophetic),
    # )
    ### Iterate over tuple of models (multiple hyperparameter sets can be handled per partition), and
    ### concatenate predictions for those with the predictions from multiple models from the next partition
    ### (and so on).
    for i, (model_, feat_) in enumerate(
        zip(prophetic_driver.models, prophetic_organizer.prophetic_features)
    ):
        # print(
        #     f"DEVELOPMENT - working on partition {i}\nmodel_ is {model_}\ndriver model is {prophetic_driver.curr_models}"
        # )
        assert model_ == prophetic_driver.curr_models
        assert feat_ == prophetic_driver.curr_prophetic
        from tensorflow import keras
        from keras.models import Model

        models, feat = prophetic_driver.load_prophetic_hypermodels_and_x()
        feat: pd.DataFrame

        if i == 0:
            pred_idx = feat.index.to_list()
            # print("DEV", feat.index)
        else:
            assert (
                feat.index.to_list() == pred_idx
            )  # All index lists should match or something BAD is happening
        for model in models:
            predictions = model.predict(feat.values)
            # print("DEV", predictions.shape)
            output_buffer.append(pd.Series(predictions.ravel(), index=pred_idx))
        prophetic_driver.get_next_part()
    # print("DEVELOPMENT - final output buffer", output_buffer)
    # print("DEVELOPMENT - first output buffer", output_buffer[0], output_buffer[0].shape)
    concat = pd.concat(output_buffer, axis=1)
    # print("DEVELOPMENT - concat ", concat, concat.shape)
    # pred_out = pd.DataFrame(concat, index=pred_idx)
    concat.to_csv(
        f"{project.output}/{prediction_experiment}_rawpredictions.csv", header=True
    )
    # print(concat)
    return concat, total_requests


def prep_requests():
    """
    Get requested predictions after performing some basic checks.

    Returns a DataFrame and a list of requested pairs (i.e. [amine_bromide,])
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
    ### CHECKING USER INPUT NAMES FOR ERROR-INDUCING ISSUES ###
    from somn.data import load_reactant_smiles
    known_amines,known_bromides = load_reactant_smiles()
    for k,h in zip((known_amines,known_bromides),("nuc_name","el_name")):
        name_check = lambda x: x if x not in k.keys() else "pr-"+x #Define explicit check if compound is known
        p = [f.replace("_","-") for f in total_requests[h]] #Explicitly replace all underscores to prevent error later
        fixed = pd.Series(data=list(map(name_check,p)),name=h) #Apply name check
        total_requests[h]=fixed #Replace request data with "fixed" values
    #Overwrite end of compound name with iterable if there are repeats within requests
    req_am = total_requests["nuc_name"]
    req_br = total_requests["el_name"]
    am_check = req_am.duplicated()
    br_check = req_br.duplicated()
    fix_am = []
    fix_br = []
    check=0
    for i,(am,br) in enumerate(zip(req_am,req_br)): 
        checked = False
        if am_check[i] == True:
            fix_am.append(am+f"-{check}")
            checked = True
        if am_check[i] == False: fix_am.append(am)
        if br_check[i] == True:
            fix_br.append(br+f"-{check}")
            checked = True
        if br_check[i] == False: fix_br.append(br)
        if checked == True: check +=1
    # Swap out names with changed repeats
    total_requests["nuc_name"]=fix_am
    total_requests["el_name"]=fix_br
    ### CHANGE END ###
    total_requests.to_csv(f"{Project().scratch}/all_requests.csv", header=True) #These are pre-screened for compatibility
    req_pairs = []
    for row in total_requests.iterrows():
        data = row[1].values
        pair = f"{data[3]}_{data[4]}"
        req_pairs.append(pair)
    # print(",".join(req_pairs))
    # print(total_requests)

    return total_requests, req_pairs


def assemble_desc_for_inference_mols(
    project: Project,
    requests: str,
    desc: tuple,
    sub_masks: tuple,
    prediction_experiment: str,
    pred_str,
):
    """
    pipeline to generate raw feature array for inference molecules

    input "requests" should be a file path to a .csv which will contain many reactions tabulated as:

    UserID,     nucleophile (SMILES)        electrophile (SMILES)       (optional) name for nucleophile     (optional) name for electrophile

    Each row will be considered an individual request. The final output will have all of these in one feature array as a batch of requests.
    """
    ### CHECK IF GEOMS ALREADY CALCULATED (IF FUTURE CALL ON EXISTING PROJECT), THEN CALCULATE, IF NECESSARY
    from somn.workflows.add import add_workflow
    from pathlib import Path

    args = ["multsmi", requests, "mult", "-ser", "t"]
    # try:
    path_to_write=f"{project.structures}/{prediction_experiment}"
    if Path(
    str(path_to_write) + "/newmol_smi_buffer.json").exists(): ## This is generated by add_workflow, and should only exist IF this was previously called
        print(f"Skipping new molecule descriptor calculation, because it looks like this has been done for {project.unique}.\
This may cause an error if new molecules are requested now which were not calculated before - consider making a new project.")
    else:
        add_workflow(
        project=project,
        prediction_experiment=prediction_experiment,
        parser_args=args,
    )
    # except:
    #     raise Exception(
    #         "Something went wrong with calculating substrate descriptors for new molecules - check inputs"
    #     )
    ### Now we're ready to calculate RDF features
    from somn.calculate.substrate import calculate_prophetic
    import molli as ml
    import json

    prophetic_amine_col = ml.Collection.from_zip(
        f"{project.structures}/{prediction_experiment}/prophetic_nucleophile.zip"
    )
    prophetic_bromide_col = ml.Collection.from_zip(
        f"{project.structures}/{prediction_experiment}/prophetic_electrophile.zip"
    )
    p_ap = json.load(
        open(f"{project.structures}/{prediction_experiment}/newmol_ap_buffer.json")
    )

    p_a_desc = calculate_prophetic(
        inc=0.75, geometries=prophetic_amine_col, atomproperties=p_ap, react_type="N"
    )
    p_b_desc = calculate_prophetic(
        inc=0.75, geometries=prophetic_bromide_col, atomproperties=p_ap, react_type="Br"
    )
    ### Now we're ready to assemble features
    am, br, ca, so, ba = desc
    am.update(p_a_desc)
    br.update(p_b_desc)
    upd_desc = (am, br, ca, so, ba)
    # print(upd_desc)
    prophetic_features = assemble_descriptors_from_handles(
        pred_str, desc=upd_desc, sub_mask=sub_masks
    )
    # print("DEBUG", prophetic_features)
    prophetic_features.reset_index(drop=True).to_feather(
        f"{project.descriptors}/prophetic_{prediction_experiment}.feather"
    )
    # from somn.calculate.preprocess import new_mask_random_feature_arrays
    return prophetic_features

