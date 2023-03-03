from itertools import product
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np


def assemble_descriptors_from_handles(handle_input, am_dict, br_dict):
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

    # print(handle_input)
    # print(rxn_hndls)
    # outfile_name = date_+'_desc_input'
    directory = "descriptors/"
    basefile = directory + "base_params.csv"
    basedf = pd.read_csv(basefile, header=None, index_col=0).transpose()
    solvfile = directory + "solvent_params.csv"
    solvdf = pd.read_csv(solvfile, header=None, index_col=0).transpose()
    # catfile = directory+'cat_aso_aeif_combined_11_2021.csv' ##Normal ASO/AEIF cats CHANGED TEST
    catfile = (
        directory + "iso_catalyst_embedding.csv"
    )  ##isomap embedded cats CHANGED FOR SIMPLIFICATION
    catdf = pd.read_csv(catfile, header=None, index_col=0).transpose()

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
            for key, val in am_dict[am].iteritems():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].iteritems():
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
            for key, val in am_dict[am].iteritems():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].iteritems():
                brdesc.extend(val.tolist())
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handle)
            # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
        outdf = pd.DataFrame(columns, index=labels).transpose()
        # print(outdf)
        return outdf


def assemble_random_descriptors_from_handles(handle_input, desc: tuple):
    """
    Assemble descriptors from output tuple of make_randomized_features function call

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

    am_dict_rand, br_dict_rand, cat_rand, solv_rand, base_rand = desc
    basedf = base_rand
    solvdf = solv_rand
    catdf = cat_rand
    br_dict = br_dict_rand
    am_dict = am_dict_rand
    # print(catdf)

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
            for key, val in am_dict[am].iteritems():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].iteritems():
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
        for combination in itertools.product(rxn_hndls, allcats, solv_base_cond):
            exp_handles.append(s.format(*combination))
        columns = []
        labels = []
        for handle in exp_handles:
            am, br, cat, solv, base = tuple(handle.split("_"))
            catdesc = catdf[cat].tolist()
            solvdesc = solvdf[int(solv)].tolist()
            basedesc = basedf[base].tolist()
            amdesc = []
            for key, val in am_dict[am].iteritems():  # This is a pd df
                amdesc.extend(val.tolist())
            brdesc = []
            for key, val in br_dict[br].iteritems():
                brdesc.extend(val.tolist())
            columns.append(amdesc + brdesc + catdesc + solvdesc + basedesc)
            labels.append(handle)
            # outdf[handle] = amdesc+brdesc+catdesc+solvdesc+basedesc
        outdf = pd.DataFrame(columns, index=labels).transpose()
        # print(outdf)
        return outdf
