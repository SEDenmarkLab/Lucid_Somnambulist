import molli as ml
from somn.workflows.firstgen_calc_sub import main as calc_sub
from somn.calculate import preprocess
from somn.build.assemble import (
    load_calculated_substrate_descriptors,
    assemble_descriptors_from_handles,
    assemble_random_descriptors_from_handles,
)
import os
from somn.workflows import PART_
from copy import deepcopy

# Load in raw calculated descriptors + random descriptors


def main(val_schema=""):
    """
    Validation Schema Argument (val_schema):

    validation or test are abbreviated "v" or "t"
    in-sample and out-of-sample are abbreviate "i" and "o"

    val in, test out (or test out, val in) would be "vi_to" or "to_vi"

    """
    assert bool(set([val_schema]) & set(["to_vi", "vi_to", "random", "vo_to", "to_vo"]))
    if val_schema == "vo_to" or val_schema == "to_vo":
        import random
        for i, val in enumerate(unique_couplings):
            am, br = val.split("_") #The test reactants
            nn_val_sub = random.sample(
                [f for f in combos if f != val and f[0] != am and f[1] != br], 3
            )  # sample three different validation pairs from random combinations that are NOT the test coupling OR either reactant for test
            v_am = []
            v_br = []
            for val_sub in nn_val_sub:  # get list of amines and bromides for validation
                am_, br_ = val_sub.split("_")
                v_am.append(am_)
                v_br.append(br_)
            name_ = (
                str(i + 1)
                + "_te_"
                + f"a{am}_b{br}"
                + "_v_"
                + "_".join(["a" + f for f in v_am])
                + "_"
                + "_".join(["b" + f for f in v_br])
            )
            outsamp_test_handles = preprocess.split_outsamp_reacts(
                dataset, amines=[am], bromides=[br], separate=False
            )
            outsamp_val_handles_contam = preprocess.split_outsamp_reacts(
                dataset, amines=v_am, bromides=v_br, separate=False
            )  # This will sometimes be contaminated by couplings with validation reactants and one of the test reactants
            outsamp_val_handles = [
                f for f in outsamp_val_handles_contam if f not in outsamp_test_handles
            ]  # This explicitly removes ANY match with a test set reaction
            tr_int, te = preprocess.outsamp_by_handle(dataset, outsamp_test_handles)
            tr, va = preprocess.outsamp_by_handle(tr_int, outsamp_val_handles)
            partition_pipeline(name_, tr, va, te)
    else:
        for i, val in enumerate(combos):
            am, br = val.split("_")
            name_ = str(i + 1) + "_" + val + "_" + val_schema + "-schema"
            if val_schema == "random":
                ### RANDOM SPLITS ###
                tr, va, te = preprocess.random_splits(dataset, validation=True, fold=10)
            ### OUT OF SAMPLE TEST, IN SAMPLE VAL ###
            # am_f,br_f,both,outsamp_handles = split_outsamp_reacts(data_df,amines=[44,38,32],bromides=[13],separate=True)
            elif val_schema == "to_vi" or val_schema == "vi_to":
                outsamp_handles = preprocess.split_outsamp_reacts(
                    dataset, amines=[am], bromides=[br], separate=False
                )
                # tr,va,te,valm,testm = preprocess.platewise_splits(dataset,num_coup=5,save_mask=True,val_int=False,val_split=8,test_list=[uni_coup[i]])
                temp, te = preprocess.outsamp_by_handle(dataset, outsamp_handles)
                tr, va = preprocess.random_splits(
                    temp, validation=False, n_splits=1, fold=7
                )
            partition_pipeline(name_, tr, va, te)
            #### DEBUG
            if i == 4:
                break
            # Random features made on a component-basis
            # x_tr = assemble_random_descriptors_from_handles(tr.index.tolist(), rand)
            # x_va = assemble_random_descriptors_from_handles(va.index.tolist(), rand)
            # x_te = assemble_random_descriptors_from_handles(te.index.tolist(), rand)
            # x_tr_real = assemble_descriptors_from_handles(
            #     tr.index.tolist(), sub_am_dict, sub_br_dict
            # )
            # x_va_real = assemble_descriptors_from_handles(
            #     va.index.tolist(), sub_am_dict, sub_br_dict
            # )
            # x_te_real = assemble_descriptors_from_handles(
            #     te.index.tolist(), sub_am_dict, sub_br_dict
            # )
            # (x_tr_, x_va_, x_te_), (
            #     x_tr_re,
            #     x_va_re,
            #     x_te_re,
            # ) = preprocess.new_mask_random_feature_arrays(
            #     (x_tr_real, x_va_real, x_te_real), (x_tr, x_va, x_te), _vt=0.04
            # )
            ##### Eventually, implement masking - want to save these for doing feature importances/visualization later
            # x_tr_.to_feather(randout + name_ + "_xtr.feather")
            # x_va_.to_feather(randout + name_ + "_xva.feather")
            # x_te_.to_feather(randout + name_ + "_xte.feather")
            # x_tr_re.to_feather(realout + name_ + "_xtr.feather")
            # x_va_re.to_feather(realout + name_ + "_xva.feather")
            # x_te_re.to_feather(realout + name_ + "_xte.feather")
            # ############ NOTE: the y-values do not change with random features - so we're just serializing two copies for each set here for convenience. They are small. ####
            # ### "Rand" copies of Y
            # tr.transpose().reset_index(drop=True).to_feather(
            #     randout + name_ + "_ytr.feather"
            # )
            # va.transpose().reset_index(drop=True).to_feather(
            #     randout + name_ + "_yva.feather"
            # )
            # te.transpose().reset_index(drop=True).to_feather(
            #     randout + name_ + "_yte.feather"
            # )
            # ### "Real" copies of Y
            # tr.transpose().reset_index(drop=True).to_feather(
            #     realout + name_ + "_ytr.feather"
            # )
            # va.transpose().reset_index(drop=True).to_feather(
            #     realout + name_ + "_yva.feather"
            # )
            # te.transpose().reset_index(drop=True).to_feather(
            #     realout + name_ + "_yte.feather"
            # )


def partition_pipeline(name_, tr, va, te):
    x_tr = assemble_random_descriptors_from_handles(tr.index.tolist(), rand)
    x_va = assemble_random_descriptors_from_handles(va.index.tolist(), rand)
    x_te = assemble_random_descriptors_from_handles(te.index.tolist(), rand)
    # Real features used to generate masks for random features
    x_tr_real = assemble_descriptors_from_handles(tr.index.tolist(), real)
    x_va_real = assemble_descriptors_from_handles(va.index.tolist(), real)
    x_te_real = assemble_descriptors_from_handles(te.index.tolist(), real)
    (x_tr_, x_va_, x_te_), (
        x_tr_re,
        x_va_re,
        x_te_re,
    ) = preprocess.new_mask_random_feature_arrays(
        (x_tr_real, x_va_real, x_te_real), (x_tr, x_va, x_te), _vt=0.03
    )
    x_tr_.to_feather(randout + name_ + "_rand-feat_xtr.feather")
    x_va_.to_feather(randout + name_ + "_rand-feat_xva.feather")
    x_te_.to_feather(randout + name_ + "_rand-feat_xte.feather")
    x_tr_re.to_feather(realout + name_ + "_real-feat_xtr.feather")
    x_va_re.to_feather(realout + name_ + "_real-feat_xva.feather")
    x_te_re.to_feather(realout + name_ + "_real-feat_xte.feather")
    ############ NOTE: the y-values do not change with random features - so we're just serializing two copies for each set here for convenience. They are small. ####
    ### "Rand" copies of Y
    tr.transpose().reset_index(drop=True).to_feather(
        randout + name_ + "_rand-feat_ytr.feather"
    )
    va.transpose().reset_index(drop=True).to_feather(
        randout + name_ + "_rand-feat_yva.feather"
    )
    te.transpose().reset_index(drop=True).to_feather(
        randout + name_ + "_rand-feat_yte.feather"
    )
    ### "Real" copies of Y
    tr.transpose().reset_index(drop=True).to_feather(
        realout + name_ + "_real-feat_ytr.feather"
    )
    va.transpose().reset_index(drop=True).to_feather(
        realout + name_ + "_real-feat_yva.feather"
    )
    te.transpose().reset_index(drop=True).to_feather(
        realout + name_ + "_real-feat_yte.feather"
    )


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
) = preprocess.load_data(optional_load="experimental_catalyst")
real, rand = calc_sub(optional_load="experimental_catalyst")
# (sub_am_dict, sub_br_dict), rand = load_calculated_substrate_descriptors()
# TESTING - both work.
# print(real[0].keys())
# print(sub_am_dict.keys())

sub_am_dict, sub_br_dict, cat_desc, solv_desc, base_desc = real

# Val have out of sample reactants
combos = preprocess.get_all_combos(unique_couplings)
outdir = deepcopy(PART_)
os.makedirs(outdir + "real/", exist_ok=True)
os.makedirs(outdir + "rand/", exist_ok=True)
realout = outdir + "real/"
randout = outdir + "rand/"

main(val_schema="vo_to")
