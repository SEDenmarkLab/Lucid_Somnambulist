import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

# tf.config.set_visible_devices([],'GPU')
from tensorflow import keras

# import numpy as np
import pandas as pd
from glob import glob

# import matplotlib.pyplot as plt
# from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector
# from sklearn.linear_model import RidgeCV
# from sklearn.ensemble import RandomForestRegressor
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error
from somn.learn import tf_organizer, tfDriver, model_inference, get_hps
from somn.build.assemble import assemble_descriptors_from_handles
from datetime import date
import pickle
import json

# import new_rdf_manual as desc
# import randfeat_rdf_manual as desc

# import molli as ml
# import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint


model_dir = r"/home/nir2/tfwork/ROCHE_ws/Nov-19-2022-00-00_realcont10_outval_outte/out/"
directory_ = r"/home/nir2/tfwork/ROCHE_ws/Nov-18-2022out_te_samepre_OUTsampval_075inc_vt03_maincont/real"

organ = tf_organizer("blah", partition_dir=directory_)
drive = tfDriver(organ)


date__ = date.today().strftime("%b-%d-%Y") + "oldmodels"

out_dir_path = (
    r"/home/nir2/tfwork/ROCHE_ws/nn_last_three_infer_" + date__ + "_forSI/out/"
)
os.makedirs(out_dir_path, exist_ok=True)


metrics = []
predictions = []

train_val_test = {}
train_val_test["train"] = []
train_val_test["val"] = []
train_val_test["test"] = []


### Should have all of this already done - this just needs to take in features and go.
# col1_ = ml.Collection.from_zip("descriptors/confs_am_optgfn2.zip")
# col2_ = ml.Collection.from_zip("descriptors/rocheval_amine_conf_am_optgfn2.zip")
# atomprop_am_dict = pickle.load(open("descriptors/amine_pickle_dict.p",'rb'))
# sub_am_dict = desc.retrieve_amine_rdf_descriptors(col1_,atomprop_am_dict,increment=0.75)
# new_am = desc.retrieve_amine_rdf_descriptors(col2_,atomprop_am_dict,increment=0.75)
# sub_am_dict.update(new_am)
# col1 = ml.Collection.from_zip("descriptors/confs_br_optgfn2.zip")
# col2 = ml.Collection.from_zip("descriptors/rocheval_amine_conf_br_optgfn2.zip")
# atomprop_dict = pickle.load(open("descriptors/bromide_pickle_dict.p",'rb'))
# sub_br_dict = desc.retrieve_bromide_rdf_descriptors(col1,atomprop_dict,increment=0.75)
# new_dict = desc.retrieve_bromide_rdf_descriptors(col2,atomprop_dict,increment=0.75)
# sub_br_dict.update(new_dict)
### Vestigial

# pred_str = sys.argv[1] # Needs to be for the new compounds - need to reconfigure this as a function with some basic logical checks.
pred_str = ""
x_p_df = assemble_descriptors_from_handles(pred_str, sub_am_dict, sub_br_dict)
pred_idx = x_p_df.columns.tolist()

with open("prediction_buffer_" + date__ + ".csv", "a") as g:
    g.write(",".join(pred_idx) + "\n")

with tf.device("/GPU:0"):
    for __k in range(len(drive.organizer.partitions)):
        train_val_test["train"] = []
        train_val_test["val"] = []
        train_val_test["test"] = []
        xtr, xval, xte, ytr, yval, yte = [f[0] for f in drive.x_y]
        name_ = str(drive.current_part_id)
        print(name_)
        partition_index = organ.partIDs.index(int(name_))
        tr, va, te, y1, y2, y3 = drive.organizer.partitions[partition_index]
        tr, va, te = drive._feather_to_np((tr, va, te))
        # print(tr[1])
        x_tr = assemble_descriptors_from_handles(
            tr[1].to_list(), sub_am_dict, sub_br_dict
        )
        x_va = assemble_descriptors_from_handles(
            va[1].to_list(), sub_am_dict, sub_br_dict
        )
        x_te = assemble_descriptors_from_handles(
            te[1].to_list(), sub_am_dict, sub_br_dict
        )
        (x_tr_, x_va_, x_te_, x_p_) = desc.preprocess_feature_arrays(
            (x_tr, x_va, x_te, x_p_df), save_mask=False
        )
        # print(x_tr_.transpose(),tr[0])
        # print(x_tr_.transpose().shape,tr[0].shape)
        # print(x_p_)
        # print(x_tr_,x_tr_.shape)
        # if x_tr_.shape[0] != tr[0].shape[0]:
        #     print(x_tr_,tr)
        #     raise Exception('feature preprocessing failed')
        models__ = glob(model_dir + name_ + "hpset*.h5")
        # print(models__)
        ytr = ytr.ravel()
        yval = yval.ravel()
        yte = yte.ravel()
        train_val_test["train"].append(ytr)
        train_val_test["val"].append(yval)
        train_val_test["test"].append(yte)
        for _model in models__:
            ### For new inferences
            model_config = keras.models.load_model(_model).get_config()
            # print(model_config)
            model_config["layers"][0]["config"]["batch_input_shape"] = (
                None,
                x_tr_.transpose().shape[1],
            )  # Update shape to match NEW preprocessing
            model = tf.keras.Sequential.from_config(model_config)
            model.compile(
                optimizer=tf.keras.optimizers.Adadelta(learning_rate=1),
                loss="mse",
                metrics=["accuracy", "mean_absolute_error", "mean_squared_error"],
            )
            history = model.fit(
                x_tr_.transpose().to_numpy(),
                ytr,
                batch_size=32,
                epochs=175,
                validation_data=(x_va_.transpose().to_numpy(), yval),
                workers=64,
                callbacks=[
                    ModelCheckpoint(
                        filepath=out_dir_path + name_ + "_best_model.h5",
                        monitor="val_loss",
                        save_best_only=True,
                    )
                ],
            )
            model.load_weights(out_dir_path + name_ + "_best_model.h5")
            ### For getting model values to plot
            # model = keras.models.load_model(_model)
            xtr = x_tr_.transpose().to_numpy()
            xval = x_va_.transpose().to_numpy()
            xte = x_te_.transpose().to_numpy()
            x_p = (
                x_p_.transpose().to_numpy()
            )  # Important to get index-instance column-features
            ytr_p, yval_p, yte_p, yp_p = model_inference(model, xtr, (xval, xte, x_p))
            predictions.append(yp_p)
            train_val_test["train"].append(ytr_p)
            train_val_test["val"].append(yval_p)
            train_val_test["test"].append(yte_p)
            mae_tr = mean_absolute_error(ytr, ytr_p)
            mae_te = mean_absolute_error(yte, yte_p)
            mae_val = mean_absolute_error(yval, yval_p)
            metrics.append([mae_tr, mae_val, mae_te])
            with open("prediction_buffer_" + date__ + ".csv", "a") as g:
                g.write(",".join([str(f) for f in yp_p]) + "\n")
        outdf = pd.DataFrame(train_val_test["train"]).transpose()
        outdf.index = x_tr_.columns
        valdf = pd.DataFrame(train_val_test["val"]).transpose()
        valdf.index = x_va_.columns
        tedf = pd.DataFrame(train_val_test["test"]).transpose()
        tedf.index = x_te_.columns
        outdf_ = pd.concat((outdf, valdf, tedf), axis=0, keys=["train", "val", "test"])
        outdf_.to_csv(
            out_dir_path
            + "output_models_part"
            + name_
            + "_"
            + "_".join([str(len(ytr)), str(len(yval)), str(len(yte))])
            + ".csv"
        )
        drive.get_next_part()

pred_df = pd.DataFrame(predictions, columns=pred_idx).transpose()
pred_df.to_csv(out_dir_path + "prophetic_output_" + date__ + ".csv")
