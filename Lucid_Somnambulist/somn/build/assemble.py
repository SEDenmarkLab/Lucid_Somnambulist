from itertools import product
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from itertools import product
from somn.workflows import DESC_
from somn.data import BASEDESC, SOLVDESC, CATDESC


def assemble_descriptors_from_handles(handle_input, desc: tuple):
    """
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

    am_dict_real, br_dict_real, cat_real, solv_real, base_real = desc
    basedf = base_real.transpose()
    solvdf = solv_real.transpose()
    catdf = (
        cat_real.transpose()
    )  # Confusing - FIX THIS - trying to use it like a dictionary later, but it's clearly still a df. Need to have column-wise lookup
    br_dict = br_dict_real
    am_dict = am_dict_real

    # print(handle_input)
    # print(rxn_hndls)
    # outfile_name = date_+'_desc_input'
    # directory = DESC_
    # basefile = DESC_ + "base_params.csv"
    # basedf = pd.read_csv(basefile, header=None, index_col=0).transpose()
    # solvfile = directory + "solvent_params.csv"
    # solvdf = pd.read_csv(solvfile, header=None, index_col=0).transpose()
    # # catfile = directory+'cat_aso_aeif_combined_11_2021.csv' ##Normal ASO/AEIF cats CHANGED TEST
    # catfile = (
    #     directory + "iso_catalyst_embedding.csv"
    # )  ##isomap embedded cats CHANGED FOR SIMPLIFICATION
    # catdf = pd.read_csv(catfile, header=None, index_col=0).transpose()

    ### Trying to assemble descriptors for labelled examples with specific conditions ###
    if prophetic == False:
        columns = []
        labels = []
        for i, handle in enumerate(rxn_hndls):
            am, br, cat, solv, base = handle
            catdesc = catdf[cat].tolist()
            solvdesc = solvdf[int(solv)].tolist()
            basedesc = basedf[base].tolist()
            amdesc = []
            for key, val in am_dict[am].items():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].items():
                brdesc.extend(val.tolist())
            handlestring = handle_input[i]
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handlestring)
        outdf = pd.DataFrame(columns, index=labels).transpose()
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
            amdesc = []
            for key, val in am_dict[am].items():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].items():
                brdesc.extend(val.tolist())
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handle)
            # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
        outdf = pd.DataFrame(columns, index=labels).transpose()
        # print(outdf)
        return outdf


def randomize_features(feat=np.array):
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


# def make_randomized_features(am_dict,br_dict,catfile=None,solvfile=None,basefile=None):
#     """
#     For running randomized feature control

#     Pass dict of dataframes to this to randomize substrate features

#     Handles are the dataset partitions (as a tuple...these will be returned with the desired order but randomized)

#     output is AMINE, BROMIDE, CATALYST, SOLVENT, BASE
#     """
#     directory = 'descriptors/'

#     if basefile==None: basefile = directory+'base_params.csv'
#     else: basefile = basefile
#     basedf = pd.read_csv(basefile,header=None,index_col=0).transpose()
#     if solvfile==None: solvfile = directory+'solvent_params.csv'
#     else: solvfile==solvfile
#     solvdf = pd.read_csv(solvfile,header=None,index_col=0).transpose()
#     if catfile==None: catfile = directory+'cat_aso_aeif_combined_11_2021.csv'
#     else: catfile==catfile
#     catdf = pd.read_csv(catfile,header=None,index_col=0).transpose()
#     cat_rand = randomize_features(catdf.to_numpy())
#     catdfrand = pd.DataFrame(cat_rand,index=catdf.index,columns=catdf.columns)
#     solv_rand = randomize_features(solvdf.to_numpy())
#     solvdfrand = pd.DataFrame(solv_rand,index=solvdf.index,columns=solvdf.columns)
#     base_rand = randomize_features(basedf.to_numpy())
#     basedfrand = pd.DataFrame(base_rand,index=basedf.index,columns=basedf.columns)
#     br_dict_rand = {}
#     am_dict_rand = {}
#     for k,v in am_dict.items():
#         rand_f = randomize_features(np.array(v.iloc[:,:9].to_numpy()))
#         rand_int = np.random.randint(0,3,v.iloc[:,9:].to_numpy().shape)
#         concat = np.concatenate((rand_f,rand_int),axis=1)
#         am_dict_rand[k] = pd.DataFrame(concat,index=v.index,columns=v.columns)
#     for k,v in br_dict.items():
#         rand_f = randomize_features(np.array(v.iloc[:,:9].to_numpy()))
#         rand_int = np.random.randint(0,3,v.iloc[:,9:].to_numpy().shape)
#         concat = np.concatenate((rand_f,rand_int),axis=1)
#         br_dict_rand[k] = pd.DataFrame(concat,index=v.index,columns=v.columns)
#     return am_dict_rand,br_dict_rand,catdfrand,solvdfrand,basedfrand


def assemble_random_descriptors_from_handles(handle_input, desc: tuple):
    """
    Input descriptors (real) as: (am_dict, br_dict, catdf, solvdf, basedf)
    Output is: df with component-wise random features

    To do this for all dataset compounds, pass every am_br joined with a comma

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
    # Faster to make random features ONCE, and then keep using it
    # am_dict, br_dict, catdf, solvdf, basedf = desc
    # rand_out = make_randomized_features(am_dict, br_dict, catdf, solvdf, basedf)
    am_dict_rand, br_dict_rand, cat_rand, solv_rand, base_rand = desc
    basedf = base_rand.transpose()
    solvdf = solv_rand.transpose()
    catdf = (
        cat_rand.transpose()
    )  # Confusing - FIX THIS - trying to use it like a dictionary later, but it's clearly still a df. Need to have column-wise lookup
    br_dict = br_dict_rand
    am_dict = am_dict_rand

    ### Trying to assemble descriptors for labelled examples with specific conditions ###
    if prophetic == False:
        columns = []
        labels = []
        for i, handle in enumerate(rxn_hndls):
            am, br, cat, solv, base = handle
            catdesc = catdf[cat].tolist()
            solvdesc = solvdf[int(solv)].tolist()
            basedesc = basedf[base].tolist()
            amdesc = []
            for key, val in am_dict[am].items():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].items():
                brdesc.extend(val.tolist())
            handlestring = handle_input[i]
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handlestring)
        outdf = pd.DataFrame(columns, index=labels).transpose()
        # print(outdf)
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
            amdesc = []
            for key, val in am_dict[am].items():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].items():
                brdesc.extend(val.tolist())
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handle)
            # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
        outdf = pd.DataFrame(columns, index=labels).transpose()
        # print(outdf)
        return outdf


def load_calculated_substrate_descriptors():
    """
    Load calculated substrate descriptors - skip calculation step to save time
    """
    import pickle
    from somn.workflows import DESC_
    from glob import glob

    am = glob(DESC_ + "real_amine_desc_*.p")
    br = glob(DESC_ + "real_bromide_desc_*.p")
    try:
        with open(DESC_ + "random_am_br_cat_solv_base.p", "rb") as k:
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
