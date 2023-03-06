import molli as ml
from somn.workflows.firstgen_calc_sub import main
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
) = preprocess.load_data()
real, rand = main()
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


def main(val_schema=""):
    for i, val in enumerate(combos):
        am, br = val.split("_")
        name_ = str(i + 1) + "_" + val + "_inval"
        if val_schema == "random":
            ### RANDOM SPLITS ###
            tr, va, te = preprocess.random_splits(dataset, validation=True, fold=11)
        ### OUT OF SAMPLE TEST, IN SAMPLE VAL ###
        # am_f,br_f,both,outsamp_handles = split_outsamp_reacts(data_df,amines=[44,38,32],bromides=[13],separate=True)
        elif val_schema == "to_vi" or val_schema == "vi_to":
            outsamp_handles = preprocess.split_outsamp_reacts(
                dataset, amines=[am], bromides=[br], separate=False
            )
            # tr,va,te,valm,testm = outsamp_splits(dataset,num_coup=5,save_mask=True,val_int=False,val_split=8,test_list=[uni_coup[i]])
            tr, te = preprocess.outsamp_by_handle(dataset, outsamp_handles)
            # print(tr.shape)
            tr, va = preprocess.random_splits(
                tr, validation=False, n_splits=1, fold=7
            )  # Comment out to only do train/test
        # Random features made on a component-basis
        x_tr = assemble_random_descriptors_from_handles(tr.index.tolist(), rand)
        x_va = assemble_random_descriptors_from_handles(va.index.tolist(), rand)
        x_te = assemble_random_descriptors_from_handles(te.index.tolist(), rand)
        x_tr_real = assemble_descriptors_from_handles(
            tr.index.tolist(), sub_am_dict, sub_br_dict
        )
        x_va_real = assemble_descriptors_from_handles(
            va.index.tolist(), sub_am_dict, sub_br_dict
        )
        x_te_real = assemble_descriptors_from_handles(
            te.index.tolist(), sub_am_dict, sub_br_dict
        )
        (x_tr_, x_va_, x_te_), (
            x_tr_re,
            x_va_re,
            x_te_re,
        ) = preprocess.new_mask_random_feature_arrays(
            (x_tr_real, x_va_real, x_te_real), (x_tr, x_va, x_te), _vt=0.04
        )  # Comment out to only do train/test
        # (x_tr_re,x_va_re,x_te_re),preprocess_mask = preprocess_feature_arrays((x_tr_real,x_va_real,x_te_real),save_mask=True,_vt=None) ##This is for getting back to early 2022 models    #Comment out to only do train/test
        # print(tr,te)
        # print(abs(x_tr_re-x_tr_).sum(axis=0).sum(axis=0),np.array_equal(x_tr_re,x_tr_))
        ### Note: append mode is possible, but only with PyTables, and this is supposed to be slower read/write
        ### For now, mask is only a reference "just in case", so we'll just write both.
        # mask_.to_hdf(outdir+name_+'_maskreference.hdf5','feature_mask',mode='w',format='fixed')
        # output = pd.concat((x_tr_,x_va_,x_te_),keys=['train','validate','test'],axis=1)
        # output.to_hdf(outdir+name_+'_partitions.hdf5','partitions',mode='w',format='fixed')
        # if i == 0:
        # print(x_tr_.transpose())
        # mask_ = pd.Series(vars,index=cols) #When saving variances of features
        # pd.Series(preprocess_mask).to_csv(outdir+name_+'_maskreference.csv') #A way to keep track of which features are removed in preprocessing
        # mask_.to_csv(outdir+name_+'_maskreference.csv',sep=',',header=['variance'],index=True)
        # These have been preprocessed just like the real features, but are randomized on a component-basis.
        x_tr_.to_feather(randout + name_ + "_xtr.feather")
        x_va_.to_feather(
            randout + name_ + "_xva.feather"
        )  # Comment out to only do train/test
        x_te_.to_feather(randout + name_ + "_xte.feather")
        # Save these in case we want direct comparison later
        x_tr_re.to_feather(realout + name_ + "_xtr.feather")
        x_va_re.to_feather(
            realout + name_ + "_xva.feather"
        )  # Comment out to only do train/test
        x_te_re.to_feather(realout + name_ + "_xte.feather")
        # print(tr,tr.transpose().reset_index(drop=True))
        ### Drop index for data because it is outside of the index/col level. This fails with .to_feather()
        tr.transpose().reset_index(drop=True).to_feather(
            randout + name_ + "_ytr.feather"
        )
        va.transpose().reset_index(drop=True).to_feather(
            randout + name_ + "_yva.feather"
        )  # Comment out to only do train/test
        te.transpose().reset_index(drop=True).to_feather(
            randout + name_ + "_yte.feather"
        )

        tr.transpose().reset_index(drop=True).to_feather(
            realout + name_ + "_ytr.feather"
        )
        va.transpose().reset_index(drop=True).to_feather(
            realout + name_ + "_yva.feather"
        )  # Comment out to only do train/test
        te.transpose().reset_index(drop=True).to_feather(
            realout + name_ + "_yte.feather"
        )
        # mask_.to_pickle(outdir+name_+'_maskreference.pkl')
        # output = pd.concat((x_tr_,x_va_,x_te_),keys=['train','validate','test'],axis=1)
        # output.to_pickle(outdir+name_+'_partitions.pkl')
