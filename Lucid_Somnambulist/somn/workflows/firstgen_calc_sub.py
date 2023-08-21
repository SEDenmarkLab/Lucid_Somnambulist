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

data.load_sub_mols()
data.load_all_desc()

# from somn.workflows import DESC_
import molli as ml
from somn.util.project import Project

# DEBUG: the global variables exist within the namespace of data, and can be intuitively loaded via:
# data.{global var}
# print(len(data.ACOL.molecules))
# SIMILAR for using universal read/write directories. This is from the workflows.
# from somn.workflows import UNIQUE_
# print(UNIQUE_)

############################ Calculate reactant descriptors #############################


def calculate_prophetic(
    inc=0.75, geometries=ml.Collection, atomproperties=dict, react_type=""
):
    """
    Vanilla substrate descriptor retrieval
    """
    if react_type == "am":
        sub_dict = retrieve_amine_rdf_descriptors(
            geometries, atomproperties, increment=inc
        )
    elif react_type == "br":
        sub_dict = retrieve_bromide_rdf_descriptors(
            geometries, atomproperties, increment=inc
        )
    elif react_type == "cl":
        sub_dict = retrieve_chloride_rdf_descriptors(
            geometries, atomproperties, increment=inc
        )
    return sub_dict


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
            print("Boolean mask:\n", am_mask)
            print(am_label)
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


if __name__ == "__main__":
    # (
    #     (sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc),
    #     rand,
    # ) #Format of output
    from sys import argv

    if argv[1] == "new":
        assert len(argv) >= 3
        project = Project()
        project.save(identifier=argv[2])
    else:
        try:
            project = Project.reload(how=argv[1])
        except:
            raise Exception(
                "Must pass valid identifier or 'last' to load project. Can say 'new' and give an identifier"
            )

    desc_out = main(
        project, substrate_pre=("corr", 0.97), optional_load="experimental_catalyst"
    )
    ### DEBUG
    # import pandas as pd

    # print(desc_out[0][0]) #Visualize entries
    ### DEBUG
    pickle.dump(
        desc_out, open(f"{project.descriptors}/real_rand_descriptor_buffer.p", "wb")
    )
