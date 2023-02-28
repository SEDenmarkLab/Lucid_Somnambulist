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


def preprocess_feature_arrays(
    feature_dataframes: (pd.DataFrame), labels: list = None, save_mask=False, _vt=None
):
    """
    NOTE: labels depreciated until further development

    Accepts tuple of dataframes with raw descriptors, then preprocesses them.

    Outputs them as a combined df with labels to retrieve them from labels parameter.
    This ensures equal preprocessing across each feature set.

    Note: pass with COLUMNS as instances and INDICES as features, eg. df[handle]=pd.series([feat1,feat2,feat3...featn])

    Use:
    tuple of dfs: (train_features,validation_features,test_features,prophetic_features)
    optional: list of names: ['train','validate','test','predict']

    returns dfs like this:
    tuple(traindf,validatedf,testdf,predictdf) corresponding to labels

    OR if labels are explicitly passed, then get a df with keys as labels

    Standard use:
    train,val,test,pred = preprocess_feature_arrays((train_pre,val_pre,te_pre_pred_pre))

    TO UNPACK DATAFRAME OUTPUT WHEN LABELS ARE EXPLICIT:
    use dfout[key] to retrieve column-instance/row-feature sub dataframes

    """
    if labels == None:
        labels = [str(f) for f in range(len(feature_dataframes))]
        combined_df = pd.concat(
            feature_dataframes, axis=1, keys=labels
        )  # concatenate instances on columns
        # print(combined_df)
        mask = list(
            combined_df.nunique(axis=1) != 1
        )  # Boolean for rows with more than one unique value
        # print(len(mask))
        filtered_df = combined_df.iloc[
            mask, :
        ]  # Get only indices with more than one unique value
        # print(filtered_df)
        ### IAN CHANGE ADDED VARIANCE THRESHOLD ### - this was probably a mistake and may remove too many features. Scaling first is probably the correct thing to do.
        if type(_vt) == float:
            vt = VarianceThreshold(threshold=_vt)
        elif _vt == "old":
            vt = VarianceThreshold(threshold=0.04)
        elif _vt == None:
            vt = VarianceThreshold(threshold=1e-4)
        # filtered_df_scale = pd.DataFrame(np.transpose(MinMaxScaler().fit_transform(VarianceThreshold(threshold=0.04).fit_transform(filtered_df.transpose().to_numpy()))),columns=filtered_df.columns) ## No variance threshold is better for the new RDFs
        # output = tuple([filtered_df_scale[lbl] for lbl in labels])
        # if save_mask==True: return output,mask,filtered_df.transpose().columns,None
        # elif save_mask==False: return output

        # sc = MinMaxScaler().fit_transform(filtered_df.transpose().to_numpy())
        vt_f = vt.fit_transform(filtered_df.transpose().to_numpy())
        sc = MinMaxScaler().fit_transform(vt_f)
        filtered_df_scale = pd.DataFrame(np.transpose(sc), columns=filtered_df.columns)
        # filtered_df_scale = pd.DataFrame(np.transpose(VarianceThreshold(threshold=0.08).fit_transform(MinMaxScaler().fit_transform(filtered_df.transpose().to_numpy()))),columns=filtered_df.columns)
        # filtered_df_scale = pd.DataFrame(np.transpose(MinMaxScaler().fit_transform(filtered_df.transpose().to_numpy())),columns=filtered_df.columns) ## No variance threshold is better for the new RDFs
        # print(filtered_df_scale)
        output = tuple([filtered_df_scale[lbl] for lbl in labels])
        if save_mask == True:
            return output, mask
        elif save_mask == False:
            return output
    ### Too much development; following is depreciated
    # elif len(labels) != len(feature_dataframes):
    #     raise Exception('Must pass equal number of df labels as feature dfs')
    # else:
    #     combined_df = pd.concat(feature_dataframes,axis=0,keys=labels).transpose() #Gets features to columns
    #     mask = list(combined_df.nunique(axis=0)!=1) # Boolean for columns with more than one unique value
    #     filtered_df = combined_df.iloc[:,mask] # Get only columns with more than one unique value
    #     if save_mask == True:
    #         return filtered_df,mask
    #     elif save_mask == False:
    #         return filtered_df
