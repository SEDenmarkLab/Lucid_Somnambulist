from itertools import product
import pandas as pd

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import VarianceThreshold
import numpy as np
from itertools import product

from somn.data import BASEDESC, SOLVDESC, CATDESC


def get_labels(sub_df_dict, sub):
    """
    Get labels for feature array from dict of 2D RDF
    """
    desclabel = []
    for series in (
        sub_df_dict[sub].transpose().itertuples(index=True)
    ):  # Tuples does rows, so need to transpose to go through each rdf sequentially
        desclabel.extend([f"{series.Index}_{i+1}" for i in range(len(series[1:]))])
    return desclabel


def vectorize_substrate_desc(sub_df_dict, sub, feat_mask=None):
    """
    Function to extract 2D RDF features, vectorize them, then apply optional feature selection mask
    (which must be calculated beforehand using optional unsupervised substrate preprocessing or designed)

    Boolean mask sorts out features using list comprehension on zip() tuples of the two.

    """
    ### Substrate name calls up a DF with column - channel, row - slice RDF
    ### Then, each column is stacked in the same order into a list.
    ### If an optional preprocessing was run (or is being tested to assess feature importance),
    ### then the feat_mask is used to mask this 1D vector before returning it.
    subdesc = []
    for series in sub_df_dict[sub].transpose().itertuples(index=False):
        subdesc.extend(list(series))
    if isinstance(feat_mask, pd.Series):
        try:
            assert all([isinstance(f, (np.bool_, bool)) for f in feat_mask])
        except:
            # print(feat_mask, all(feat_mask))
            # print([type(k) for k in feat_mask])
            raise Exception(
                "Feature mask passed for substrate preprocessing is a Series, but appears to not ONLY contain boolean values"
            )
        if len(feat_mask.to_list()) == len(subdesc):
            out = [b for a, b in zip(feat_mask.to_list(), subdesc) if a]
            ### DEBUG
            # print(subdesc)
            ###
            return out
        else:
            raise Exception("length mismatch! cannot apply substrate mask")
    elif isinstance(feat_mask, pd.DataFrame):
        try:
            assert all([isinstance(f, (np.bool_, bool)) for f in feat_mask["0"]])
        except:
            raise Exception(
                "Feature mask passed as a dataframe, and could not interpret which column to use. Check input."
            )
        if len(feat_mask["0"].to_list()) == len(subdesc):
            out = [b for a, b in zip(feat_mask["0"].to_list(), subdesc) if a]
            ### DEBUG
            # print(subdesc)
            ###
            return out
        else:
            raise Exception("length mismatch! cannot apply substrate mask")
    elif isinstance(feat_mask, np.ndarray):  ##NEEDS TESTING
        try:
            assert all([isinstance(f, (np.bool_, bool)) for f in feat_mask])
        except:
            # print(feat_mask, all(feat_mask))
            raise Exception(
                "Feature mask passed for substrate preprocessing does not ONLY contain boolean values"
            )
        if len(feat_mask) == len(subdesc):
            out = [b for a, b in zip(feat_mask, subdesc) if a]
            ### DEBUG
            # print(subdesc)
            ###
            return out
        else:
            raise Exception(
                "A mask for substrate descriptors was passed, but it does not match the length of the raw features."
            )
    elif feat_mask == None:
        return subdesc
    else:
        raise Exception(
            "If a substrate feature mask is passed, it must be an array of boolean values"
        )


# def assemble_random_descriptors_from_handles(
#     handle_input, desc: tuple, substrate_mask=None
# ):
#     """
#     NOTE: Depreciated in favor of passing either real or random descriptor tuples to the same parent function.

#     Input descriptors as: (am_dict, br_dict, catdf, solvdf, basedf)
#     Output is: df with component-wise random features

#     To do this for many compounds, pass every am_br joined with a comma

#     """
#     if type(handle_input) == str:
#         rxn_hndls = [f for f in handle_input.split(",") if f != ""]
#         prophetic = True
#     elif type(handle_input) == list:
#         rxn_hndls = [tuple(f.rsplit("_")) for f in handle_input]
#         prophetic = False
#     else:
#         raise ValueError(
#             "Must pass manual string input of handles OR list from dataset"
#         )
#     if substrate_mask == None:
#         subm = None
#     elif type(substrate_mask) == tuple:
#         assert len(substrate_mask) == 2
#         subm = substrate_mask
#     am_dict_rand, br_dict_rand, cat_rand, solv_rand, base_rand = desc
#     basedf = base_rand.transpose()
#     solvdf = solv_rand.transpose()
#     catdf = (
#         cat_rand.transpose()
#     )  # Confusing - FIX THIS - trying to use it like a dictionary later, but it's clearly still a df. Need to have column-wise lookup
#     br_dict = br_dict_rand
#     am_dict = am_dict_rand
#     ### Trying to assemble descriptors for labelled examples with specific conditions ###
#     if prophetic == False:
#         columns = []
#         labels = []
#         for i, handle in enumerate(rxn_hndls):
#             am, br, cat, solv, base = handle
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             ### CHANGES HERE SUBSTRATE MASKING
#             if subm != None:
#                 assert isinstance(subm, list)
#                 amdesc = vectorize_substrate_desc(am_dict, am, feat_mask=subm[0])
#                 brdesc = vectorize_substrate_desc(br_dict, br, feat_mask=subm[1])
#             elif subm == None:
#                 amdesc = vectorize_substrate_desc(am_dict, am, feat_mask=None)
#                 brdesc = vectorize_substrate_desc(br_dict, br, feat_mask=None)
#             else:
#                 raise Exception(
#                     "Substrate mask was not None or a list - check arguments for assembling descriptors."
#                 )
#             handlestring = handle_input[i]
#             columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
#             labels.append(handlestring)
#         outdf = pd.DataFrame(columns, index=labels).transpose()
#         return outdf
#     ### Trying to assemble descriptors for ALL conditions for specific amine/bromide couplings ###
#     elif prophetic == True:
#         solv_base_cond = ["1_a", "1_b", "1_c", "2_a", "2_b", "2_c", "3_a", "3_b", "3_c"]
#         allcats = [str(f + 1) for f in range(21) if f != 14]
#         s = "{}_{}_{}"
#         exp_handles = []
#         for combination in product(rxn_hndls, allcats, solv_base_cond):
#             exp_handles.append(s.format(*combination))
#         columns = []
#         labels = []
#         for handle in exp_handles:
#             am, br, cat, solv, base = tuple(handle.split("_"))
#             catdesc = catdf[cat].tolist()
#             solvdesc = solvdf[int(solv)].tolist()
#             basedesc = basedf[base].tolist()
#             amdesc = []
#             for key, val in am_dict[am].items():  # This is a pd df
#                 amdesc.extend(val.tolist())
#             brdesc = []
#             for key, val in br_dict[br].items():
#                 brdesc.extend(val.tolist())
#             columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
#             labels.append(handle)
#         outdf = pd.DataFrame(columns, index=labels).transpose()
#         return outdf


def assemble_descriptors_from_handles(handle_input, desc: tuple, sub_mask=None):
    """
    NOTE: Masking for substrates should be a tuple of length 2, with amine, then bromide masks.

    General utility for assembling ordered descriptors based on input reaction handles and
    calculated amine and bromide rdf descriptor dictionaries. This can be used to automate
    testing hypertuning of rdf calculator hyperparams.


    use sysargv[1] for handle input

    sys.argv[1] should be list of truncated handles:
    amine_bromide,amine_bromide,....

    OR

    pass a list of ALL handles:
    amine_br_cat_solv_base

    This will assemble only descriptors as required by the list of handles, and will
    return the descriptors in the appropriate order

    Can also be all handles from a datafile; whatever.

    This is meant to use am_dict and br_dict as conformer-averaged descriptors.
    This lets the user apply different parameters to descriptor tabulation flexibly.



    """
    if type(handle_input) == str:
        rxn_hndls = [f for f in handle_input.split(",") if f != ""]
        prophetic = True
    elif type(handle_input) == list:
        rxn_hndls = [tuple(f.rsplit("_")) for f in handle_input]
        prophetic = False
    else:
        raise ValueError(
            "Must pass manual string input of handles OR list from dataset"
        )
    if sub_mask == None:
        subm = None
    elif type(sub_mask) == tuple:
        assert len(sub_mask) == 2
        # print("DEBUG found substrate mask")
        subm = list(sub_mask)
    am_dict, br_dict, cat_real, solv_real, base_real = desc
    ## These were read in from feather files, and must be transposed
    basedf = base_real.transpose()
    solvdf = solv_real.transpose()
    catdf = (
        cat_real.transpose()
    )  # Confusing - FIX THIS - trying to use it like a dictionary later, but it's clearly still a df. Need to have column-wise lookup
    # br_dict = br_dict_real  # This is just to re-use code, but is confusing.
    # am_dict = am_dict_real

    ### Trying to assemble descriptors for labelled examples with specific conditions ###
    if prophetic == False:
        columns = []
        labels = []
        for i, handle in enumerate(rxn_hndls):
            am, br, cat, solv, base = handle
            catdesc = catdf[cat].tolist()
            solvdesc = solvdf[int(solv)].tolist()
            basedesc = basedf[base].tolist()
            ### CHANGES HERE SUBSTRATE MASKING
            if subm != None:
                assert isinstance(subm, list)
                amdesc = vectorize_substrate_desc(am_dict, am, feat_mask=subm[0])
                brdesc = vectorize_substrate_desc(br_dict, br, feat_mask=subm[1])
                # print(f"DEBUG used sub mask {amdesc}")
            elif subm == None:
                amdesc = vectorize_substrate_desc(am_dict, am, feat_mask=None)
                brdesc = vectorize_substrate_desc(br_dict, br, feat_mask=None)
            else:
                raise Exception(
                    "Substrate mask caused an error - check arguments for assembling descriptors."
                )
            ### END SUB MASK CHANGES
            handlestring = handle_input[i]
            # print(
            #     f"DEBUG desc are {[type(f) for f in [amdesc,brdesc,catdesc,solvdesc,basedesc]]}"
            # )
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handlestring)
        outdf = pd.DataFrame(columns, index=labels).transpose()
        # ### DEVELOPMENT
        # from somn.util.project import Project
        # import uuid

        # outdf.to_csv(
        #     f"{Project().descriptors}/DEBUG-OUTPUTDESC-MASKED{uuid.uuid1().hex}.csv"
        # )
        # ### DEVELOPMENT
        return outdf
    ### Trying to assemble descriptors for ALL conditions for specific amine/bromide couplings ###
    elif prophetic == True:
        solv_base_cond = ["1_a", "1_b", "1_c", "2_a", "2_b", "2_c", "3_a", "3_b", "3_c"]
        allcats = [str(f + 1) for f in range(21) if f != 14]
        s = "{}_{}_{}"
        exp_handles = []
        for combination in product(rxn_hndls, allcats, solv_base_cond):
            exp_handles.append(s.format(*combination))
        columns = []
        labels = []
        for handle in exp_handles:
            am, br, cat, solv, base = tuple(handle.split("_"))
            catdesc = catdf[cat].tolist()
            solvdesc = solvdf[int(solv)].tolist()
            basedesc = basedf[base].tolist()
            if subm != None:
                assert isinstance(subm, list)
                amdesc = vectorize_substrate_desc(am_dict, am, feat_mask=subm[0])
                brdesc = vectorize_substrate_desc(br_dict, br, feat_mask=subm[1])
            elif subm == None:
                amdesc = vectorize_substrate_desc(am_dict, am, feat_mask=None)
                brdesc = vectorize_substrate_desc(br_dict, br, feat_mask=None)
            else:
                raise Exception(
                    "Substrate mask caused an error - check arguments for assembling descriptors."
                )
            ### No masking - depreciated
            # amdesc = []
            # for key, val in am_dict[am].items():  # This is a pd df
            #     amdesc.extend(val.tolist())
            # brdesc = []
            # for key, val in br_dict[br].items():
            #     brdesc.extend(val.tolist())
            ###
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handle)
        outdf = pd.DataFrame(columns, index=labels).transpose()
        return outdf


def randomize_features(feat: np.ndarray):
    """
    Accepts feature array and randomizes values

    NOTE: does not create consistent vector per component - for illustration purposes to compare to "real" random feature control.
    FOR PROPER CONTROL use "make_randomized_features," which generates a library of randomized vectors which are then combinatorially
    applied to assemble a feature array.

    """
    feat_ = feat
    rng = np.random.default_rng()
    feats = rng.random(out=feat)
    return feats


def make_randomized_features(am_dict, br_dict, catdf, solvdf, basedf):
    """
    For running randomized feature control

    Pass dict of dataframes to this to randomize substrate features

    Handles are the dataset partitions (as a tuple...these will be returned with the desired order but randomized)

    output is AMINE, BROMIDE, CATALYST, SOLVENT, BASE
    """
    cat_rand = randomize_features(catdf.to_numpy())
    catdfrand = pd.DataFrame(cat_rand, index=catdf.index, columns=catdf.columns)
    solv_rand = randomize_features(solvdf.to_numpy())
    solvdfrand = pd.DataFrame(solv_rand, index=solvdf.index, columns=solvdf.columns)
    base_rand = randomize_features(basedf.to_numpy())
    basedfrand = pd.DataFrame(base_rand, index=basedf.index, columns=basedf.columns)
    br_dict_rand = {}
    am_dict_rand = {}
    for k, v in am_dict.items():
        rand_f = randomize_features(np.array(v.iloc[:, :9].to_numpy()))
        rand_int = np.random.randint(0, 3, v.iloc[:, 9:].to_numpy().shape)
        concat = np.concatenate((rand_f, rand_int), axis=1)
        am_dict_rand[k] = pd.DataFrame(concat, index=v.index, columns=v.columns)
    for k, v in br_dict.items():
        rand_f = randomize_features(np.array(v.iloc[:, :9].to_numpy()))
        rand_int = np.random.randint(0, 3, v.iloc[:, 9:].to_numpy().shape)
        concat = np.concatenate((rand_f, rand_int), axis=1)
        br_dict_rand[k] = pd.DataFrame(concat, index=v.index, columns=v.columns)
    return am_dict_rand, br_dict_rand, catdfrand, solvdfrand, basedfrand


def load_calculated_substrate_descriptors():
    """
    Load calculated substrate descriptors - skip calculation step to save time
    """
    import pickle
    from glob import glob
    from somn.util.project import Project

    project = Project()
    am = glob(f"{project.descriptors}/real_amine_desc_*.p")
    br = glob(f"{project.descriptors}/real_bromide_desc_*.p")
    try:
        with open(f"{project.descriptors}/random_am_br_cat_solv_base.p", "rb") as k:
            rand = pickle.load(k)
    except:
        raise Exception(
            "Have not calculated descriptors in this session - need to either update data/ or calculate new descriptors in this session."
        )
    with open(am[0], "rb") as g:
        sub_am_dict = pickle.load(g)
    with open(br[0], "rb") as q:
        sub_br_dict = pickle.load(q)
    return ((sub_am_dict, sub_br_dict), rand)


# ### DEV
# if __name__ == "__main__":
#     """
#     Testing/development
#     """
#     from somn.util.project import Project

#     project = Project.reload(how="last")
#     # project.save(identifier="DEV")
#     from somn.workflows.firstgen_calc_sub import main as calc_sub

#     ((sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc), rand) = calc_sub(
#         project=project, substrate_pre=("corr", 0.90)
#     )
#     # print(sub_br_dict["2015"])
#     mask = pd.read_csv(
#         f"{project.descriptors}/bromide_mask.csv", header=0, index_col=0
#     )  # Series with boolean mask
#     # print(mask["0"])
#     # k = vectorize_substrate_desc(sub_br_dict, "2015", feat_mask=None)
#     p = vectorize_substrate_desc(sub_br_dict, "2015", feat_mask=mask["0"].to_numpy())
#     q = vectorize_substrate_desc(sub_br_dict, "2015", feat_mask=mask["0"])
#     m = vectorize_substrate_desc(sub_br_dict, "2015", feat_mask=mask)
#     # print(mask.index)
#     # print("masked: ", len(p))
#     # print("unmasked: ", len(k))
#     # print(type(p))
#     ## Ordering of vector of features matches vector ordering for boolean mask. Hand-checked here.
#     # print([f for f in sub_br_dict["2015"].transpose().itertuples(index=False)])
#     # print(list(zip(p, [f for f in mask.index.to_list() if mask["0"][f]])))
#     # print(sub_br_dict["2015"])
#     # print(mask.to_string())
