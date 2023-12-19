# import somn
# from copy import deepcopy
import pickle
from somn.calculate.RDF import (
    retrieve_amine_rdf_descriptors,
    retrieve_bromide_rdf_descriptors,
    retrieve_chloride_rdf_descriptors,
)
from somn.calculate.preprocess import new_mask_random_feature_arrays
from somn.build.assemble import (
    # assemble_descriptors_from_handles,
    # assemble_random_descriptors_from_handles,
    make_randomized_features,
    get_labels,
)

# ====================================================================
# Load in data shipped with package for manipulation. Optional import + function call
# ====================================================================
from somn import data
from somn.calculate import preprocess
import pandas as pd
data.load_sub_mols()
data.load_all_desc()
import warnings
import molli as ml
# from somn.util.project import Project

def calculate_substrate_descriptors(fpath):
    """
    CLI function to calculate substrate descriptors for input molecules.
    Should run without using a project. 
    """
    try:
        amines,am_atomprops,bromides,br_atomprops,chlorides,cl_atomprops = calculate_substrate_geoms_and_props(fpath)
        
    except:
        raise Exception("Calculating substrate geometries and atop properties failed.")




def calculate_substrate_geoms_and_props(fpath):
    """

    """
    from somn.calculate.substrate import PropheticInput
    from somn.build.parsing import InputParser
    from pathlib import Path
    import json
    parse = InputParser(serialize=True,path_to_write="./somn_calculate_outputs/")
    # project = Project()
    assert ".csv" == Path(fpath).suffix
    req_am,am_smi,req_br,br_smi,req_cl,cl_smi = scrape_substrate_csv(fpath)
    ### Take care of amines
    amines_pre,smiles_am = parse.get_mol_from_smiles(user_input = am_smi,recursive_mode=True,names=req_am)
    amines,err_0 = parse.preopt_geom(amines_pre,update=60)
    if len(err_0) > 0: raise warnings.warn("Looks like some amines did not generate 3D geometries or preoptimize correctly. Check inputs.")
    amines.to_zip(parse.path_to_write+"amines_preopt_geom.zip")
    amine_input_packet = PropheticInput.from_col(amines,am_smi,roles = ["nuc" for f in am_smi],parser=parse)
    ### Calculating atom properties
    am_atomprops,am_errs = amine_input_packet.atomprop_pipeline()
    if len(am_errs) > 0:
        warnings.warn(
            f"Looks like {len(am_errs)} amines failed at the atomproperty calculation step - this singlepoint calc usually fails because \
            the input structure is not valid. Check that backed up structure in the working directory {parse.path_to_write}"
        )
    with open(f"{parse.path_to_write}/amine_ap_buffer.json", "w") as k:
        json.dump(am_atomprops, k)
    ### Take care of bromides
    bromides_pre,_ = parse.get_mol_from_smiles(user_input = br_smi,recursive_mode=True,names=req_br)
    bromides,err_1 = parse.preopt_geom(bromides_pre,update=60)
    if len(err_1) > 0: raise warnings.warn("Looks like some bromides did not generate 3D geometries or preoptimize correctly. Check inputs.")
    bromides.to_zip(parse.path_to_write+"bromides_preopt_geom.zip")
    bromide_input_packet = PropheticInput.from_col(bromides,br_smi,roles = ["el" for f in br_smi],parser=parse)
    ### Calculating atom properties
    br_atomprops,br_errs = bromide_input_packet.atomprop_pipeline()
    if len(am_errs) > 0:
        warnings.warn(
            f"Looks like {len(br_errs)} bromides failed at the atomproperty calculation step - this singlepoint calc usually fails because \
            the input structure is not valid. Check that backed up structure in the working directory {parse.path_to_write}"
        )
    with open(f"{parse.path_to_write}/bromide_ap_buffer.json", "w") as k:
        json.dump(br_atomprops, k)
    ### Take care of chlorides
    chlorides_pre,_ = parse.get_mol_from_smiles(user_input = cl_smi,recursive_mode=True,names=req_cl)
    chlorides,err_2 = parse.preopt_geom(chlorides_pre,update=60)
    if len(err_2) > 0: raise warnings.warn("Looks like some chlorides did not generate 3D geometries or preoptimize correctly. Check inputs.")
    chlorides.to_zip(parse.path_to_write+"chlorides_preopt_geom.zip")
    chloride_input_packet = PropheticInput.from_col(chlorides,cl_smi,roles = ["el" for f in cl_smi],parser=parse)
    ### Calculating atom properties
    cl_atomprops,cl_errs = chloride_input_packet.atomprop_pipeline()
    if len(am_errs) > 0:
        warnings.warn(
            f"Looks like {len(cl_errs)} chlorides failed at the atomproperty calculation step - this singlepoint calc usually fails because \
            the input structure is not valid. Check that backed up structure in the working directory {parse.path_to_write}"
        )
    with open(f"{parse.path_to_write}/chloride_ap_buffer.json", "w") as k:
        json.dump(cl_atomprops, k)
    return amines,am_atomprops,bromides,br_atomprops,chlorides,cl_atomprops
    

def scrape_substrate_csv(fpath):
    """
    Scrape a csv file with substrates

    First column is SMILES, second column is type ("N", "Br", or "Cl")
    """
    df = pd.read_csv(fpath, header=0, index_col=0)
    try:
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 2
        req_am,req_br,req_cl,am_smi,br_smi,cl_smi = ([] for i in range(6))
        for row in df.iterrows():
            if row[1][1] in ["N","n"]:
                req_am.append(row[0])
                am_smi.append(row[1][0])
            elif row[1][1] in ["Br","br","BR"]:
                req_br.append(row[0])
                br_smi.append(row[1][0])
            elif row[1][1] in ["Cl","cl","CL"]:
                req_cl.append(row[0])
                cl_smi.append(row[1][0])
    except:
        raise Exception("Looks like the input file to request descriptors was not formatted \
                        correctly. Input should be 3 columns: name, SMILES, and 'type' ('N','Br', or 'Cl')")       
    return req_am,am_smi,req_br,br_smi,req_cl,cl_smi
















def main(
    project: Project, inc=0.75, substrate_pre=None, optional_load=None, serialize=True
):
    """
    Run workflow to calculate real and random descriptors for substrates. Saves random features for ALL components,
    but only calculates substrate features. These are keyed feature sets, not assembled arrays.

    Can be called to return real desc (5 member tuple, am,br,cat,solv,base) and random desc (similar tuple)

    returns:
    (sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc), (random versions in same order)
    """
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
    ) = preprocess.load_data(optional_load)

    ### Calculate descriptors for the reactants, and store their 1D vector arrays in a dictionary-like output.
    _inc = inc
    sub_am_dict = retrieve_amine_rdf_descriptors(amines, a_prop, increment=_inc)
    sub_br_dict = retrieve_bromide_rdf_descriptors(bromides, br_prop, increment=_inc)
    ### Preprocess reactant descriptors now, since they are just calculated
    if substrate_pre == None:
        type_, value_ = None, None
    elif isinstance(substrate_pre, tuple):
        from somn.build.assemble import vectorize_substrate_desc
        import pandas as pd

        ### Assemble a feature array with row:instance,column:feature to perform preprocessing
        if (
            len(substrate_pre) == 2
        ):  # This is the first step for correlated features - assembling full arrays
            type_, value_ = substrate_pre
            if type_ == "corr":
                am_desc = {}
                for key in sub_am_dict.keys():
                    am_desc[key] = vectorize_substrate_desc(
                        sub_am_dict, key, feat_mask=None
                    )
                am_label = get_labels(sub_am_dict, "1")
                full_am_df = pd.DataFrame.from_dict(
                    am_desc, orient="index", columns=am_label
                )
                br_desc = {}
                for key in sub_br_dict.keys():
                    br_desc[key] = vectorize_substrate_desc(
                        sub_br_dict, key, feat_mask=None
                    )
                br_label = get_labels(sub_br_dict, "1")
                full_br_df = pd.DataFrame.from_dict(
                    br_desc, orient="index", columns=br_label
                )
                if serialize == True:
                    full_br_df.to_csv(
                        f"{project.descriptors}/bromide_only_features.csv", header=True
                    )
                    full_am_df.to_csv(
                        f"{project.descriptors}/amine_only_features.csv", header=True
                    )
                ### DEV ###
                # print(full_am_df)
                # print(full_am_df.corr())
                # raise Exception("DEBUG")
                # full_am_df.to_csv("testing.csv", header=True)
                # full_am_df.corr().abs().to_csv("correlation.csv", header=True)
                ###

        else:
            raise Exception("Tuple passed to sub preprocessing, but not length 2")
    else:
        raise Exception(
            "Need to pass both arguments for substrate preprocessing in a length 2 tuple"
        )
    if (
        type_ != None and value_ != None
    ):  # Need to process then make matching random features.
        if (
            type_ == "corr"
        ):  # This step will actually compute the correlated features mask
            am_mask = preprocess.corrX_new(
                full_am_df, cut=value_, get_const=True, bool_out=True
            )
            ### DEBUG
            # print("Boolean mask:\n", am_mask)
            # print(am_label)
            br_mask = preprocess.corrX_new(
                full_br_df, cut=value_, get_const=True, bool_out=True
            )
            ### DEBUG
            # print("Boolean mask:\n", br_mask)
            # print(br_label)
            # Saving selected features for inspection later
            pd.Series(br_mask[0], index=br_label).to_csv(
                f"{project.descriptors}/bromide_mask.csv"
            )
            pd.Series(am_mask[0], index=am_label).to_csv(
                f"{project.descriptors}/amine_mask.csv"
            )
            sub_am_proc = full_am_df.loc[:, am_mask[0]]
            assert (sub_am_proc.columns == am_mask[1]).all()
            sub_br_proc = full_br_df.loc[:, br_mask[0]]
            assert (sub_br_proc.columns == br_mask[1]).all()
            sub_am_proc.to_csv(
                f"{project.descriptors}/amine_selected_feat.csv", header=True
            )
            sub_br_proc.to_csv(
                f"{project.descriptors}/bromide_selected_feat.csv", header=True
            )
        else:
            ## Placeholder for alternative other preprocessing methods.
            pass
    rand = make_randomized_features(
        sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc
    )
    # print(rand)
    if serialize == True:
        with open(f"{project.descriptors}/random_am_br_cat_solv_base.p", "wb") as k:
            pickle.dump(rand, k)
        with open(f"{project.descriptors}/real_amine_desc_{_inc}.p", "wb") as g:
            pickle.dump(sub_am_dict, g)
        with open(f"{project.descriptors}/real_bromide_desc_{_inc}.p", "wb") as q:
            pickle.dump(sub_br_dict, q)
    return ((sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc), rand)


# if __name__ == "__main__":
#     # (
#     #     (sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc),
#     #     rand,
#     # ) #Format of output
#     from sys import argv

#     if argv[1] == "new":
#         assert len(argv) >= 3
#         project = Project()
#         project.save(identifier=argv[2])
#     else:
#         try:
#             project = Project.reload(how=argv[1])
#         except:
#             raise Exception(
#                 "Must pass valid identifier or 'last' to load project. Can say 'new' and give an identifier"
#             )

#     desc_out = main(
#         project, substrate_pre=("corr", 0.97), optional_load="experimental_catalyst"
#     )
#     ### DEBUG
#     # import pandas as pd

#     # print(desc_out[0][0]) #Visualize entries
#     ### DEBUG
#     pickle.dump(
#         desc_out, open(f"{project.descriptors}/real_rand_descriptor_buffer.p", "wb")
#     )
