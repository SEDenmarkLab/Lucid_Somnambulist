import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import random
import itertools
from copy import deepcopy
from somn.build.parsing import cleanup_handles

from somn import data

data.load_sub_mols()
data.load_all_desc()

from somn.data import (
    ACOL,
    BCOL,
    DATA,
    AMINES,
    BROMIDES,
    CATDESC,
    SOLVDESC,
    BASEDESC,
)


def load_data(optional_load=None):
    """
    Return everything as a copied object from the global variables. Format:
    (amines,bromides,dataset,handles,unique_couplings,a_prop,br_prop,base_desc,solv_desc,cat_desc)

    """
    # amine molecules are stored as ACOL global variable, bromide molecules are stored as BCOL global variable
    ### Define a copy (to protect the global one) for each component needed
    amines = deepcopy(ACOL)
    bromides = deepcopy(BCOL)
    data_raw = deepcopy(DATA)
    a_prop = deepcopy(AMINES)
    br_prop = deepcopy(BROMIDES)
    base_desc = deepcopy(BASEDESC)
    solv_desc = deepcopy(SOLVDESC)
    cat_desc = deepcopy(CATDESC)
    ### Some "safety" check on handles - this will remove duplicates and make sure leading spaces, etc won't cause problems.
    dataset = cleanup_handles(data_raw)
    handles = dataset.index
    unique_couplings = sorted(
        list(set([f.rsplit("_", 3)[0] for f in handles]))
    )  # unique am_br pairs in dataset
    if optional_load == "experimental_catalyst":
        cat_pre = preprocess_grid_maxdiff(cat_desc, threshold=(0.90, 0.89))
        return (
            amines,
            bromides,
            dataset,
            handles,
            unique_couplings,
            a_prop,
            br_prop,
            base_desc,
            solv_desc,
            cat_pre,
        )
    else:
        return (
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
        )


def calcDrop(res):
    # All variables with correlation > cutoff
    all_corr_vars = list(set(res["v1"].tolist() + res["v2"].tolist()))

    # All unique variables in drop column
    poss_drop = list(set(res["drop"].tolist()))

    # Keep any variable not in drop column
    keep = list(set(all_corr_vars).difference(set(poss_drop)))

    # Drop any variables in same row as a keep variable
    p = res[res["v1"].isin(keep) | res["v2"].isin(keep)][["v1", "v2"]]
    q = list(set(p["v1"].tolist() + p["v2"].tolist()))
    drop = list(set(q).difference(set(keep)))

    # Remove drop variables from possible drop
    poss_drop = list(set(poss_drop).difference(set(drop)))

    # subset res dataframe to include possible drop pairs
    m = res[res["v1"].isin(poss_drop) | res["v2"].isin(poss_drop)][["v1", "v2", "drop"]]

    # remove rows that are decided (drop), take set and add to drops
    more_drop = set(list(m[~m["v1"].isin(drop) & ~m["v2"].isin(drop)]["drop"]))
    for item in more_drop:
        drop.append(item)

    return drop


def corrX_new(df, cut=0.9):

    # Get correlation matrix and upper triagle
    corr_mtx = df.corr().abs()
    avg_corr = corr_mtx.mean(axis=1)
    up = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(np.bool))

    dropcols = list()

    res = pd.DataFrame(columns=(["v1", "v2", "v1.target", "v2.target", "corr", "drop"]))

    for row in range(len(up) - 1):
        col_idx = row + 1
        for col in range(col_idx, len(up)):
            if corr_mtx.iloc[row, col] > cut:
                if avg_corr.iloc[row] > avg_corr.iloc[col]:
                    dropcols.append(row)
                    drop = corr_mtx.columns[row]
                else:
                    dropcols.append(col)
                    drop = corr_mtx.columns[col]

                s = pd.Series(
                    [
                        corr_mtx.index[row],
                        up.columns[col],
                        avg_corr[row],
                        avg_corr[col],
                        up.iloc[row, col],
                        drop,
                    ],
                    index=res.columns,
                )

                res = res.append(s, ignore_index=True)

    dropcols_names = calcDrop(res)
    return dropcols_names


def trim_out_of_sample(partition: tuple, reacts: str):
    """
    Pass a string ##_## for amine_bromide that needs to be out of sample. This function will return the partitioned dataframes with those samples
    removed.
    """
    xtr, xval, xte, ytr, yval, yte = [pd.DataFrame(f[0], index=f[1]) for f in partition]
    to_move_tr = get_handles_by_reactants(reacts, ytr.index)
    to_move_va = get_handles_by_reactants(reacts, yval.index)
    # to_move_tot = to_move_tr+to_move_va
    x_trcut = xtr.loc[to_move_tr]
    y_trcut = ytr.loc[to_move_tr]
    xtr.drop(index=to_move_tr, inplace=True)
    ytr.drop(index=to_move_tr, inplace=True)
    x_vacut = xval.loc[to_move_va]
    y_vacut = yval.loc[to_move_va]
    xval.drop(index=to_move_va, inplace=True)
    yval.drop(index=to_move_va, inplace=True)
    xte = pd.concat((xte, x_trcut, x_vacut), axis=0)
    yte = pd.concat((yte, y_trcut, y_vacut), axis=0)
    return xtr, xval, xte, ytr, yval, yte


def get_handles_by_reactants(str_, handles_):
    out = []
    for k in handles_:
        # print(k.rsplit('_',3)[0])
        # print(str_)
        if k.rsplit("_", 3)[0] == str_:
            out.append(k)
    return out


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


def platewise_splits(
    data_df: pd.DataFrame,
    num_coup=5,
    save_mask=True,
    val_int=True,
    val_split=10,
    test_list=None,
):
    """
    Split dataset to withhold specific plates - NOTE: does not withhold based on reactant.
    The num_coup integer indicates the number of am_br reactant combinations to withhold into the
    validate or test split (each)

    Get split handles in tuple for masking features and data later. Validation boolean decides if output is (train,validate,test) or (train,test)

    Val split ignored unless val_int is True

    test_list overrides num_coup, and sets those couplings as the out of sample ones. However, this only does internal validation.

    """
    if val_int == False:
        handles = data_df.index
        reacts = [f.rsplit("_", 3)[0] for f in handles]
        set_ = sorted(list(set(reacts)))  # all amine_bromide combos
        if test_list == None:
            test = random.sample(
                set_, num_coup
            )  # random sampling of PLATES - does not ensure out of sample reactants for reactants used in multiple plates
        elif type(test_list) == list:
            test = test_list
        temp = [f for f in set_ if f not in test]  # temp is train and val
        val = random.sample(temp, num_coup)  # val is sampling of temp (train + val)
        train = [f for f in temp if f not in val]  # train is temp if not in val
        tr_h = [f for f in handles if f.rsplit("_", 3)[0] in train]
        va_h = [f for f in handles if f.rsplit("_", 3)[0] in val]
        te_h = [f for f in handles if f.rsplit("_", 3)[0] in test]
        mask_list = [tr_h, va_h, te_h]
        if save_mask == False:
            out = tuple([data_df.loc[msk, :] for msk in mask_list])
            return out
        if save_mask == True:
            out = tuple([data_df.loc[msk, :] for msk in mask_list] + [val, test])
            return out
    elif (
        val_int == True
    ):  ##This is to keep test plates out of sample, BUT validation and train data from shared plates. This may be necessary on account of the stochasticity of modeling
        handles = data_df.index
        reacts = [f.rsplit("_", 3)[0] for f in handles]
        set_ = sorted(list(set(reacts)))
        if (
            test_list == None
        ):  # This is for randomly sampling a number of couplings ONLY IF TEST_LIST NOT SPECIFIED
            test = random.sample(set_, num_coup)
        elif (
            type(test_list) == list
        ):  # If test_list is specified, then this overrides everything else
            test = test_list
        te_h = [f for f in handles if f.rsplit("_", 3)[0] in test]
        temp = [
            f for f in handles if f.rsplit("_", 3)[0] not in test
        ]  # both train and val will come from here; handles, not am_br
        # print(np.rint(len(temp)/val_split))
        va_h = random.sample(
            temp, int(np.rint(len(temp) / val_split))
        )  # handles sampled randomly from train&val list of handles (temp)
        tr_h = [f for f in temp if f not in va_h]
        print(
            "check :", [f for f in tr_h if f in va_h or f in te_h]
        )  # data leakage test
        mask_list = [tr_h, va_h, te_h]
        if save_mask == False:
            out = tuple([data_df.loc[msk, :] for msk in mask_list])
            return out
        if save_mask == True:
            out = tuple([data_df.loc[msk, :] for msk in mask_list] + [test])
            return out


def outsamp_by_handle(data_df: pd.DataFrame, test_list=[]):
    """
    No validation; just gives train/test using passed handles for test examples.

    """
    handles = data_df.index
    train_list = [f for f in handles if f not in test_list]
    test = data_df.loc[test_list, :]
    train = data_df.loc[train_list, :]
    return train, test


def split_handles_reactants(reacts=[], handle_position: int = 1, handles=[]):
    """
    Partition dataset to withhold specific REACTANTS; flexible to any
    specified position in reaction handle (amine, bromide, catalyst, etc) NOTE: ONE-INDEXED
    """
    str_reacts = [str(f) for f in reacts]
    out_hand = [
        f for f in handles if f.strip().split("_")[handle_position - 1] in str_reacts
    ]  # clean up whitespace, split handles, check specific position for match
    return out_hand


def split_outsamp_reacts(
    dataset_: pd.DataFrame, amines=[], bromides=[], separate=False
):
    """
    Use this to split out of sample reactants in dataset partitions.

    This runs split_out_reactants

    Data should be row = instance

    "separate" boolean triggers optional output with specific handles for out of sample amines OR bromides
    """
    amine_out_hand = split_handles_reactants(
        reacts=amines, handle_position=1, handles=dataset_.index
    )
    # print(amine_out_hand)
    bromide_out_hand = split_handles_reactants(
        reacts=bromides, handle_position=2, handles=dataset_.index
    )
    # print(bromide_out_hand)
    outsamp_handles = sorted(
        list(set(amine_out_hand + bromide_out_hand))
    )  # remove duplicates (from any matches to both reactants) and provide consistent ordering
    if separate == False:
        return outsamp_handles
    elif separate == True:
        am_f = []
        br_f = []
        comb = [
            str(f[0]) + "_" + str(f[1]) for f in itertools.product(amines, bromides)
        ]
        # print(comb)
        both = [f for f in outsamp_handles if f.strip().rsplit("_", 3)[0] in comb]
        not_both = [f for f in outsamp_handles if f not in both]
        for k in amines:
            temp1 = split_handles_reactants(
                reacts=[str(k)], handle_position=1, handles=not_both
            )
            am_f.append(temp1)
        for m in bromides:
            temp2 = split_handles_reactants(
                reacts=[str(m)], handle_position=2, handles=not_both
            )
            br_f.append(temp2)
        return am_f, br_f, both, outsamp_handles


def zero_nonzero_rand_splits(
    self,
    validation=False,
    n_splits=1,
    fold=7,
    yield_cutoff=1,
):
    """
    Split zero/nonzero data, THEN apply random splits function

    Get two output streams for zero and nonzero data to train classification models

    Can set "fuzzy" yield cutoff. This is percent yield that where at or below becomes class zero.

    """
    zero_mask = self.data.to_numpy() < yield_cutoff
    nonzero_data = self.data[~zero_mask]
    zero_data = self.data[zero_mask]
    if validation == False:
        tr_z, te_z = self.random_splits(zero_data, n_splits=n_splits, fold=fold)
        tr_n, te_n = self.random_splits(nonzero_data, n_splits=n_splits, fold=fold)
        tr = pd.concat(tr_z, tr_n, axis=1)
        te = pd.concat(te_z, te_n, axis=1)
        return tr, te
    elif validation == True:
        tr_z, va_z, te_z = self.random_splits(
            zero_data, n_splits=n_splits, fold=fold, validation=validation
        )
        tr_n, va_n, te_n = self.random_splits(
            nonzero_data, n_splits=n_splits, fold=fold, validation=validation
        )
        tr = pd.concat((tr_z, tr_n), axis=1)
        va = pd.concat((va_z, va_n), axis=1)
        te = pd.concat((te_z, te_n), axis=1)
        return tr, va, te
    else:
        raise ValueError(
            "validation parameter for zero/nonzero split function must be Boolean"
        )


def random_splits(df, validation=False, n_splits: int = 1, fold: int = 7):
    """
    Get split handles in tuple.

    Validation boolean decides if output is (train,test) or (train,validate,test)

    Each is a list of handles, train, (val), test

    """
    no_exp = len(df.index)
    rand_arr = np.random.randint(1, high=fold + 1, size=no_exp, dtype=int)
    if validation == False:
        train_mask = (rand_arr > 1).tolist()
        test_mask = (rand_arr == 1).tolist()
        mask_list = [train_mask, test_mask]
    elif validation == True:
        train_mask = (rand_arr > 2).tolist()
        validate_mask = (rand_arr == 2).tolist()
        test_mask = (rand_arr == 1).tolist()
        mask_list = [train_mask, validate_mask, test_mask]
    out = tuple([df.iloc[msk, :] for msk in mask_list])
    return out


def prep_for_binary_classifier(df_in, yield_cutoff: int = 1):
    """
    Prepare data for classifier by getting class labels from continuous yields
    """
    if type(df_in) == tuple:
        out = []
        for df in df_in:
            df = df.where(
                df > yield_cutoff, other=0, inplace=True
            )  # collapse yields at or below yield cutoff to class zero
            df = df.where(
                df == 0, other=1, inplace=True
            )  # collapse yields to class one
            out.append(df)
        return tuple(out)
    elif isinstance(df_in, pd.DataFrame):
        df = df.where(
            df > yield_cutoff, other=0, inplace=True
        )  # collapse yields at or below yield cutoff to class zero
        df = df.where(df == 0, other=1, inplace=True)  # collapse yields to class one
        return df
    else:
        raise Exception(
            "Passed incorrect input to staticmethod of DataHandler to prep data for classification - check input."
        )


def new_mask_random_feature_arrays(
    real_feature_dataframes: (pd.DataFrame),
    rand_feat_dataframes: (pd.DataFrame),
    corr_cut=0.95,
    _vt=None,
):
    """
    Use preprocessing on real features to mask randomized feature arrays, creating an actual randomized feature test which
    has proper component-wise randomization instead of instance-wise randomization, and preserves the actual input shapes
    used for the real features.

    rand out then real out as two tuples

    """
    labels = [str(f) for f in range(len(real_feature_dataframes))]
    combined_df = pd.concat(
        real_feature_dataframes, axis=1, keys=labels
    )  # concatenate instances on columns
    comb_rand = pd.concat(rand_feat_dataframes, axis=1, keys=labels)
    mask = list(
        combined_df.nunique(axis=1) != 1
    )  # Boolean for rows with more than one unique value
    filtered_df = combined_df.iloc[
        mask, :
    ]  # Get only indices with more than one unique value
    filtered_rand = comb_rand.iloc[mask, :]
    if _vt == "old":
        _vt = 0.04  # This preserves an old version of vt, and the next condition ought to still be "if" so that it still runs when this is true
    elif _vt == None:
        _vt = 1e-4
    if (
        type(_vt) == float
    ):  # Found that vt HAS to come first, or else the wrong features are removed.
        vt = VarianceThreshold(threshold=_vt)
        sc = MinMaxScaler()
        vt_real = vt.fit_transform(filtered_df.transpose().to_numpy())
        vt_rand = vt.transform(filtered_rand.transpose().to_numpy())
        sc_vt_real = sc.fit_transform(vt_real)
        sc_vt_rand = sc.transform(vt_rand)
        # sc_df_real = sc.transform(filtered_df.transpose().to_numpy())
        # sc_df_rand = sc.transform(filtered_rand.transpose().to_numpy())
        # vt.fit(sc_df_real)
        vt_df_real = pd.DataFrame(sc_vt_real)
        vt_df_rand = pd.DataFrame(sc_vt_rand)
        ### Below, replace transposed data with noc_[type] dataframes if using correlation cutoff
        processed_rand_feats = pd.DataFrame(
            np.transpose(vt_df_rand.to_numpy()), columns=filtered_df.columns
        )  # Ensures labels stripped; gives transposed arrays (row = feature, column= instance)
        processed_real_feats = pd.DataFrame(
            np.transpose(vt_df_real.to_numpy()), columns=filtered_df.columns
        )
        output_rand = tuple([processed_rand_feats[lbl] for lbl in labels])
        output_real = tuple([processed_real_feats[lbl] for lbl in labels])
        return output_rand, output_real


def get_all_combos(unique_couplings):
    """
    UTILITY

    Returns all possible pairings of amine and bromide which are used in the dataset, regardless of whether those are coupled together.

    Allows COMPLETE testing of every pair of amine and bromide as out of sample.
    """
    am_ = list(set([f.split("_")[0] for f in unique_couplings]))
    br_ = list(set([f.split("_")[1] for f in unique_couplings]))

    combos_ = itertools.product(am_, br_)
    combos = [f[0] + "_" + f[1] for f in combos_]
    return combos


def preprocess_grid_maxdiff(input: pd.DataFrame, threshold=0.80):
    """
    Pipeline function for using gridpoint range to select gridpoints.
    """

    def max_diff(input):
        """
        Return column/row-wise range

        USEWITH pd.Dataframe.apply()
        """
        delta = input.max() - input.min()
        return delta

    def max_diff_sel(df, threshold=0.80):
        """
        Threshold is percentile rank threshold for maximum differences calculated on each column.

        Max-Min for column = max difference (absolute value)
        All zeroes collapse to same percentile rank value, so threshold is NOT a percentage of the quantity of input features - this would
        only be true for an even distribution.
        """
        diff = df.apply(max_diff)
        diff.sort_values(inplace=True, ascending=False)
        ### Get percentile rank - select pct-based slice of features instead of number - like a threshold cutoff
        ranking = diff.rank(pct=True)
        idx = ranking[ranking >= threshold].index.to_list()
        return df[idx]

    def _maxdiff_then_scale(df, threshold=0.80, keyed=False):
        """
        Pass a single threshold to apply this to both aso and aeif

        Pass a length-2 tuple of thresholds to apply them to aso and aeif, respectively

        NOTE: expects raw feature input with aso, then aeif. Exactly half of the columns should be ASO, and the other half AEIF.
        This works on raw-calculated grid descriptors, which is what this is designed for.
        """

        def pull_type_(df):
            labels = df.columns
            aeif_mask = [True if "aeif" in f else False for f in labels]
            aso_mask = [True if "aso" in f else False for f in labels]
            return df[df.columns[aso_mask]], df[df.columns[aeif_mask]]

        def diff_then_scale(df, threshold):
            temp_m = max_diff_sel(df, threshold)
            temp_sc = scale.fit_transform(temp_m)
            cat_sel = pd.DataFrame(temp_sc, index=temp_m.index, columns=temp_m.columns)
            return cat_sel

        scale = MinMaxScaler()
        sli = int(len(df.columns) / 2)
        aso = df.iloc[:, :sli]
        aeif = df.iloc[:, sli:]
        aso = [f"aso_{f+1}" for f in range(sli)]
        aeif = [f"aeif_{f+1}" for f in range(sli)]
        category = aso + aeif
        # print(category)
        temp = deepcopy(df)
        temp.columns = category
        if type(threshold) == tuple:
            assert len(threshold) == 2
            asot, aeift = threshold
            aso_d, aeif_d = pull_type_(temp)
            aso_out = diff_then_scale(aso_d, asot)
            aeif_out = diff_then_scale(aeif_d, aeift)
            if keyed == True:
                out = pd.concat((aso_out, aeif_out), axis=1, keys=["aso", "aeif"])
            else:
                out = pd.concat((aso_out, aeif_out), axis=1)
            return out
        else:
            out = diff_then_scale(temp, threshold)
            return out

    cat_copy = deepcopy(input)
    output = _maxdiff_then_scale(cat_copy, threshold=threshold)
    return output
