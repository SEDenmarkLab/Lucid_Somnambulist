import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# tf.autograph.set_verbosity(3)
# tf.get_logger().setLevel(logging.ERROR)
# os.environ["KMP_AFFINITY"] = "noverbose"
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
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
from pathlib import Path
import warnings


def write_preds_to_buffer(
    project: Project, raw_predictions, prediction_index, prediction_experiment
):
    """
    dump predictions to output buffer in case job fails partway through
    """
    # pd.Series(predictions.ravel(), index=pred_idx)
    write_path = f"{project.scratch}/{prediction_experiment}_prediction_buffer.csv"
    if Path(write_path).exists():
        buffer = pd.read_csv(write_path, header=None, index_col=0, low_memory=False)
        update = pd.Series(data=raw_predictions.ravel(), index=prediction_index)
        updated_buffer = pd.concat((buffer, update), axis=0)
        updated_buffer.to_csv(write_path, header=None)
    else:
        buffer = pd.Series(data=raw_predictions.ravel(), index=prediction_index)
        buffer.to_csv(write_path, header=None)


def hypermodel_inference(
    project: Project,
    # request_dict: dict,
    model_experiment="",
    prediction_experiment="",
    optional_load="maxdiff_catalyst",
    substrate_pre=("corr", 0.90),
    vt=None,
    all_predictions=False,
):
    """
    project must contain (1) partitions, and (2) pre-trained hypermodels.
    """
    output_buffer = []
    if all_predictions == False:
        total_requests, requested_pairs, reactant_indicies = prep_requests()
        import numpy as np

        nuc_idx_input = np.array(reactant_indicies)[:, 0]
        el_idx_input = np.array(reactant_indicies)[:, 1]
        ref_idx = (nuc_idx_input, el_idx_input)
    elif all_predictions == True:
        requested_pairs = _generate_full_space()
        total_requests = None
    else:
        raise Exception(
            "Function hypermodel_inference received an invalid argument for the all_predictions keyword. This \
should be False under normal circumstances, and True for specific development applications (i.e. getting all possible predictions)"
        )
    from somn.workflows.partition import normal_partition_prep

    _, _, _, real, rand = normal_partition_prep(project=project)
    pred_str = ",".join(requested_pairs)
    sub_masks = load_substrate_masks()
    if all_predictions == False:
        prophetic_raw = assemble_desc_for_inference_mols(
            project=project,
            requests=f"{project.scratch}/all_requests.csv",
            sub_masks=sub_masks,
            desc=real,
            prediction_experiment=prediction_experiment,
            pred_str=pred_str,
            ref_idx=ref_idx,
        )
    elif all_predictions == True:
        prophetic_raw = assemble_descriptors_from_handles(
            pred_str, desc=real, sub_mask=sub_masks
        )
        prophetic_raw.reset_index(drop=True).to_feather(
            f"{project.descriptors}/prophetic_{prediction_experiment}.feather"
        )
    else:
        raise Exception(
            "Function hypermodel_inference received an invalid argument for the all_predictions keyword. This \
should be False under normal circumstances, and True for specific development applications (i.e. getting all possible predictions)"
        )
    prophetic_fp = f"{project.descriptors}/prophetic_{prediction_experiment}.feather"
    try:
        import pathlib

        assert pathlib.Path(prophetic_fp).exists()
    except:
        raise Exception(
            f"The filepath {prophetic_fp} is not real - something went wrong with \
generation of the prophetic feature array. Check project directory for \
{project.unique}"
        )
    from somn.calculate.preprocess import preprocess_prophetic_features

    print("""Assembling features for requested predictions.""")
    prophetic_organizer = preprocess_prophetic_features(
        project=project,
        features=prophetic_raw.transpose(),
        model_experiment=model_experiment,
        prediction_experiment=prediction_experiment,
        vt=vt,
    )
    print("""Features for predictions have been processed...getting predictions now.""")
    import contextlib,os
    @contextlib.contextmanager
    def suppress_print():
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            yield
    
    all_models = sorted(
        list(glob(f"{project.output}/{model_experiment}/out/*.keras"))
    )  ## KERAS vs H5
    prophetic_driver = tfDriver(
        organizer=prophetic_organizer, prophetic_models=all_models
    )
    import gc

    for i, (model_set, feat_set) in enumerate(
        zip(prophetic_driver.models, prophetic_organizer.prophetic_features)
    ):
        assert model_set == prophetic_driver.curr_models
        assert feat_set == prophetic_driver.curr_prophetic
        from tensorflow import keras
        from keras.models import Model

        models, feat = prophetic_driver.load_prophetic_hypermodels_and_x()
        feat: pd.DataFrame
        if i == 0:
            pred_idx = feat.index.to_list()
        else:
            assert (
                feat.index.to_list() == pred_idx
            )
        with suppress_print():
            for model in models:
                predictions = model.predict(feat.values)
                output_buffer.append(pd.Series(predictions.ravel(), index=pred_idx))
                write_preds_to_buffer(project, predictions, pred_idx, prediction_experiment)
            del (
                models,
                feat,
                predictions,
            )
            gc.collect()
        check_done = prophetic_driver.get_next_part()
        tf.keras.backend.clear_session()
        if check_done == 0:
            break
        elif check_done == None:
            pass
        else:
            raise Exception(
                "Error with prophetic driver instance - check that partitions and models in project passed to hypermodel_inference are complete"
            )
    concat = pd.concat(output_buffer, axis=1)
    concat.to_csv(
        f"{project.output}/{prediction_experiment}_rawpredictions.csv", header=True
    )
    return concat, total_requests


def prep_requests():
    """
    Get requested predictions after performing some basic checks.

    Returns a DataFrame and a list of requested pairs (i.e. [amine_bromide,])
    """
    import pathlib
    files = glob(f"{Project().scratch}/*_request.csv")
    pathlib.Path(f"{Project().scratch}/all_requests.csv").unlink(missing_ok=True)
    assert (
        len(files) > 0
    )
    
    df = pd.read_csv(files[0], header=0, index_col=None, dtype=str)
    if len(df.columns) < 2:
        Exception(
            "Must pass SMILES and role for each reactant! Request input file (in gproject.scratch)\
Shoult have format (col0):SMILES,(col1):role (nuc or el),(col2, optional):mol_name"
        )
    del df
    tot = []
    for i, file in enumerate(files):
        df = pd.read_csv(file, header=0, index_col=None, dtype=str)
        tot.append(df)
    total_requests = pd.concat(tot, axis=0)
    pd.options.mode.chained_assignment = None ## Suppress chained assignments - we actually want it here
    ### REMOVING NAME CHECK - SEGMENTING PROPHETIC FEATURES FROM NON-PROPHETIC TO AVOID NAME CONFLICTS ###
    
    # from somn.data import load_reactant_smiles

    # known_amines, known_bromides = load_reactant_smiles()
    # for k, h in zip((known_amines, known_bromides), ("nuc_name", "el_name")):
    #     name_check = lambda x: (
    #         x if x not in k.keys() else "pr-" + x
    #     )  # Define explicit check if compound is known
    #     p = [
    #         f.replace("_", "-") for f in total_requests[h]
    #     ]  # Explicitly replace all underscores to prevent error later
    #     fixed = pd.Series(data=list(map(name_check, p)), name=h)  # Apply name check
    #     total_requests[h] = fixed  # Replace request data with "fixed" values
    # Overwrite end of compound name with iterable if there are repeats within requests
    ### Making sure no "_" values pass - should be redundant with any frontend interface checks ###
    if any("_" in f for f in total_requests["nuc_name"]) or any("_" in f for f in total_requests["el_name"]):
        am_buffer_1 = [f.replace("_", "-") for f in total_requests["nuc_name"]]
        br_buffer_1 = [f.replace("_", "-") for f in total_requests["el_name"]]
        total_requests["nuc_name"] = am_buffer_1
        total_requests["el_name"] = br_buffer_1
        with open(f"{Project().output}/name_contained_underscore.txt",'w') as g:
            g.write(" ")
        # g = pd.DataFrame([pd.Series(am_buffer_1,index=total_requests["nuc_name"].index,name="updated-nuc-names"),
        #                    pd.Series(total_requests["nuc_name"]),
        #                    pd.Series(br_buffer_1,index=total_requests["el_name"].index,name="updated-el-names"),
        #                    pd.Series(total_requests["el_name"])
        #                    ],
        #                    index=total_requests.index,
        #                    )
        # g.to_csv(f"{Project().output}/updated_request_names.csv")
        # del g
    #### "_" check end ####

    # am_check = total_requests["nuc_name"].duplicated(keep=False)
    # br_check = total_requests["el_name"].duplicated(keep=False)
    # duplicated_amines = total_requests["nuc_name"][am_check]
    # duplicated_bromides = total_requests["el_name"][br_check]

    def find_and_enumerate_duplicates(series):
        duplicates = series[series.duplicated(keep=False)]  # Find all duplicated values
        if duplicates.empty:
            return pd.DataFrame()  # Return an empty DataFrame
        # Group by duplicated values and count occurrences
        duplicate_counts = duplicates.value_counts()
        indices = {}
        for value in duplicate_counts.index:
            indices[value] = list(series[series == value].index)
        df = pd.DataFrame({'value': duplicate_counts.index,
                            'count': duplicate_counts.values})
        df['indices'] = df['value'].apply(lambda x: indices[x])
        return df.sort_values(by='count', ascending=False)                                                                                                          

    ### Fixing names in case they are duplicated - this ensures that unique descriptors are calculated ###
    duplic_names = False
    for ser in (total_requests[["nuc_name","nuc"]],total_requests[["el_name","el"]]):
        dupl_n = find_and_enumerate_duplicates(ser.iloc[:,0]) #names unique, don't really care if smiles are repeated
        if dupl_n.empty:
            continue
        else:
            dup_smi = find_and_enumerate_duplicates(ser.iloc[:,1])
            if dup_smi.empty: #SMILES not duplicated with names, but names are. Need to be enumerated.
                duplic_names=True
                vals = dupl_n["value"]
                cnts = dupl_n["count"]
                idxs = dupl_n["indices"]
                for v,c,i in zip(vals,cnts,idxs):
                    r = range(c)
                    for idx,nv in zip(i,[f"{v}-{f}" for f in r]):
                        ser.iloc[:,0][idx] = nv
                total_requests[ser.columns[0]] = ser.iloc[:,0]
                total_requests[ser.columns[1]] = ser.iloc[:,1]
            else:
                if dup_smi["indices"].sort_values().equals(dupl_n["indices"].sort_values()): #names and smiles duplicated together
                    continue
                else: #both names and smiles duplicated, but not together
                    duplic_names = True
                    ### SMILES ###
                    from copy import deepcopy
                    sv = dup_smi["value"]
                    # if isinstance(sv,str):
                    #     tmp = deepcopy(sv)
                    #     sv = pd.Series([sv],name="value")
                    svnt = dup_smi["count"]
                    sidx = dup_smi["indices"]
                    # print(type(sidx))
                    # raise Exception("DEBUG")
                    # if isinstance(sidx,list):
                    #     tmp = deepcopy(sidx)
                    #     sidx = pd.Series([tmp],name="indices")
                    ### NAMES ###
                    idxs = dupl_n["indices"] 
                    for smi,svnt,smi_indices in zip(sv,svnt,sidx):## Complicated case - some repeats good, some repeats bad
                        if smi_indices in idxs.to_list(): #means name and smi are duplicated in matching fashion
                            continue
                        else:
                            name = ser[ser.iloc[:,1] == smi].iloc[:,0][0] #First name given to this smiles
                            for idx,nv in zip(smi_indices,[name for f in range(svnt)]):
                                ser.iloc[:,0][idx] = nv
                    dupl_n_2 = find_and_enumerate_duplicates(ser.iloc[:,0])
                    if dupl_n_2.empty:
                        total_requests[ser.columns[0]] = ser.iloc[:,0]
                        total_requests[ser.columns[1]] = ser.iloc[:,1]
                        continue
                    vals2 = dupl_n_2["value"]
                    cnts2 = dupl_n_2["count"]
                    idxs2 = dupl_n_2["indices"]                           
                    for nm,cn,indexes in zip(vals2,cnts2,idxs2):
                        if indexes in sidx.to_list(): ## We already know this - these are supposed to match
                            continue
                        else: # need to fix names that don't have matching SMILES
                            r = range(cn)
                            for idx,nv in zip(indexes,[f"{nm}-{f}" for f in r]):
                                ser.iloc[:,0][idx] = nv
                    total_requests[ser.columns[0]] = ser.iloc[:,0]
                    total_requests[ser.columns[1]] = ser.iloc[:,1]
            # total_requests[ser.columns[0].name] = ser.iloc[:,0]
            # total_requests[ser.columns[1].name] = ser.iloc[:,1]                                    
    if duplic_names == True:
        with open(f"{Project().output}/duplicate_names_detected.txt",'w') as g:
            g.write(" ")
        total_requests.to_csv(f"{Project().output}/fixed_input_requests.csv",header=True,index=False)
    ### Predict update1 - calculate ALL substrate inputs, and rely on duplicate checks on frontend ###
    total_requests.to_csv(
        f"{Project().scratch}/all_requests.csv", header=True, index=False
    )  # These are pre-screened for compatibility
    req_pairs = []
    indicies = []
    for row in total_requests.iterrows():
        data = row[1].values
        pair = f"{data[3]}_{data[4]}"
        pair_idx = [data[5], data[6]]
        req_pairs.append(pair)
        indicies.append(pair_idx)
    pd.options.mode.chained_assignment = 'warn' ## Resetting to default setting, done with chained assignments
    return total_requests, req_pairs, indicies


def _generate_full_space():
    """
    For testing purposes - create all combinations of each reactant and generate descriptors to make predictions for
    the whole space.

    FOR DEV PURPOSES/TESTING ONLY
    """
    from itertools import product
    import numpy as np
    from somn.data import load_reactant_smiles

    known_amines, known_bromides = load_reactant_smiles()
    combinations = [
        f"{f[0]}_{f[1]}" for f in product(known_amines.keys(), known_bromides.keys())
    ]
    return combinations


def assemble_desc_for_inference_mols(
    project: Project,
    requests: str,
    desc: tuple,
    sub_masks: tuple,
    prediction_experiment: str,
    pred_str,
    allow_skip=True,
    ref_idx=None,
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
    path_to_write = f"{project.structures}/{prediction_experiment}"
    if (
        Path(str(path_to_write) + "/newmol_smi_buffer.json").exists()
        and allow_skip == True
    ):  ## This is generated by add_workflow, and should only exist IF this was previously called
        print(
            f"Skipping new molecule descriptor calculation, because it looks like this has been done for {project.unique}.\
This may cause an error if new molecules are requested now which were not calculated before - consider making a new project."
        )
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
    try:
        prophetic_amine_col = ml.Collection.from_zip(
            f"{project.structures}/{prediction_experiment}/prophetic_nucleophile.zip"
        )
        prophetic_bromide_col = ml.Collection.from_zip(
            f"{project.structures}/{prediction_experiment}/prophetic_electrophile.zip"
        )
    except:
        with open(f"{project.output}/{prediction_experiment}/structure_generation_failed.txt",'w') as g:
            g.write("Could not open one or more of the structures; could not proceed to making predictions.")
        raise Exception("Structure Generation Failed.")

    # p_ap = json.load(
    #     open(f"{project.structures}/{prediction_experiment}/newmol_ap_buffer.json")
    # )
    with open(
        f"{project.structures}/{prediction_experiment}/new_nuc_ap_buffer.json"
    ) as g:
        nuc_ap = json.load(g)
    with open(
        f"{project.structures}/{prediction_experiment}/new_el_ap_buffer.json"
    ) as k:
        el_ap = json.load(k)
    ### Adding atom site selection here ###
    if ref_idx is None:  # Not specified, just autodetect reaction site (old method)
        nuc_inp_idx = el_inp_idx = None
    elif isinstance(
        ref_idx, tuple
    ):  # Passing atom indicies, but we're going to get the atoms, too.
        assert len(ref_idx) == 2
        nuc_inp_idx, el_inp_idx = ref_idx
        nuc_ref = []
        el_ref = []
        for mol, idx in zip(prophetic_amine_col.molecules, nuc_inp_idx):
            if idx == "-":
                nuc_ref.append(idx)
            elif idx.isnumeric():
                assert int(idx) in range(len(mol.atoms))
                nuc_ref.append(mol.atoms[int(idx)])
            else:
                warnings.warn(
                    f"Looks like atom specification was unsuccessful for nucleophile {mol.name}, \
switching to auto detection."
                )
        for mol, idx in zip(prophetic_bromide_col.molecules, el_inp_idx):
            if idx == "-":
                el_ref.append(idx)
            elif idx.isnumeric():
                assert int(idx) in range(len(mol.atoms))
                el_ref.append(mol.atoms[int(idx)])
            else:
                warnings.warn(
                    f"Looks like atom specification was unsuccessful for electrophile {mol.name}, \
switching to auto detection."
                )
    else:
        raise ValueError(
            "Must pass either None or a tuple of reference atom specifications to  \
calculate descriptors for prophetic molecules"
        )
    ### /site selection ###

    p_a_desc = calculate_prophetic(
        inc=0.75,
        geometries=prophetic_amine_col,
        atomproperties=nuc_ap,
        react_type="nuc",
        nuc_el_ref_atoms=nuc_ref,
    )
    p_b_desc = calculate_prophetic(
        inc=0.75,
        geometries=prophetic_bromide_col,
        atomproperties=el_ap,
        react_type="el",
        nuc_el_ref_atoms=el_ref,
    )
    ### Now we're ready to assemble features
    am, br, ca, so, ba = desc
    # am.update(p_a_desc)
    # br.update(p_b_desc)
    upd_desc = (p_a_desc, p_b_desc, ca, so, ba)
    prophetic_features = assemble_descriptors_from_handles(
        pred_str, desc=upd_desc, sub_mask=sub_masks
    )
    prophetic_features.reset_index(drop=True).to_feather(
        f"{project.descriptors}/prophetic_{prediction_experiment}.feather"
    )
    return prophetic_features
