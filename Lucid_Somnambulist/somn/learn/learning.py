from glob import glob
import os
import pandas as pd
import tensorflow as tf

########### DEV ############
# tf.compat.v1.disable_eager_execution()  ## Trying to fix speed + memory issues DEV
### Note - the v1 compat option above forces keras-tuner tuner object to call a method get_updates() which doesn't exist
### with the TF2 current Adam optimizer. Maybe using the legacy version will fix it...but this seems not ideal.
import gc  ## Trying to fix speed + memory issues DEV

########### DEV ############
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, GaussianNoise
from keras.optimizers import Adam, Adadelta
from math import exp
import json

# from somn.workflows import PART_, OUTPUT_ ## Depreciated
import keras_tuner as kt
from keras.callbacks import (
    EarlyStopping,
    TerminateOnNaN,
    ReduceLROnPlateau,
    TensorBoard,
)
from somn.util.visualize import plot_results
import numpy as np
from somn.util.project import Project


class tf_organizer:
    """
    Track NN models:
        retrieve stored dataset partitions and features,
        keep all features and labels organized into iterable tuples,
        log progress on modeling

        NOT for use with multiple descriptor types. This should be
        instantiated with one, and it will keep track of that. Use descriptors
        of the same type in the partition directory.
    """

    def __init__(self, name, partition_dir="", validation=True, inference=False):
        self.name = name
        self.part_dir = partition_dir
        self.val = validation
        self.inference = inference
        if validation == True:
            (
                self.files,
                self.xtr,
                self.xval,
                self.xte,
                self.ytr,
                self.yval,
                self.yte,
            ) = self.get_partitions(val=validation)
        elif validation == False:
            self.files, self.xtr, self.xte, self.ytr, self.yte = self.get_partitions(
                val=validation
            )
        else:
            raise ValueError(
                "Must pass boolean for validation condition to instantiate class"
            )
        self.partitions, self.partIDs = self.organize_partitions()
        self.log = []
        self.results = {}  # This is for storing results to save later to a json
        if inference == True:
            self.masks = self.get_partition_masks()

    def get_partition_masks(self):
        """
        Used for inferencing - will retrieve variance threshold masks used to build partitions, and apply them to new feature arrays.
        """
        mask1 = sorted(glob(self.part_dir + "/*_constmask.csv"))
        mask2 = sorted(glob(self.part_dir + "/*_vtmask.csv"))
        mask3 = sorted(glob(self.part_dir + "/*_scale.csv"))
        return mask1, mask2, mask3

    def get_partitions(self, val=True):
        """
        retrieve partition files so that they can be opened iteratively
        """
        files = sorted(glob(self.part_dir + "/*.feather"))
        y_train = sorted(glob(self.part_dir + "/*ytr.feather"))
        y_te = sorted(glob(self.part_dir + "/*yte.feather"))
        x_train = sorted(glob(self.part_dir + "/*xtr.feather"))
        x_te = sorted(glob(self.part_dir + "/*xte.feather"))
        if val == True:
            y_val = sorted(glob(self.part_dir + "/*yva.feather"))
            x_val = sorted(glob(self.part_dir + "/*xva.feather"))
            return files, x_train, x_val, x_te, y_train, y_val, y_te
        elif val == False:
            return files, x_train, x_te, y_train, y_te

    def organize_partitions(self):
        """
        Takes partitions and sorts train/(val)/test feather filepaths. Also reports ID numbers ordered.
        """
        partitions = []
        part_IDS = []
        if self.val == True:
            for xtr, xva, xte, ytr, yva, yte in zip(
                self.xtr, self.xval, self.xte, self.ytr, self.yval, self.yte
            ):
                x_tr = self.get_partition_info(xtr)
                x_va = self.get_partition_info(xva)
                x_te = self.get_partition_info(xte)
                y_tr = self.get_partition_info(ytr)
                y_va = self.get_partition_info(yva)
                y_te = self.get_partition_info(yte)
                parts = [x_tr[0], x_va[0], x_te[0], y_tr[0], y_va[0], y_te[0]]
                if parts.count(parts[0]) != len(
                    parts
                ):  # All elements are the same; checking
                    raise Exception(
                        "filepath parsing FAILED during organize_partitions - check folder"
                    )
                partitions.append((xtr, xva, xte, ytr, yva, yte))
                part_IDS.append(int(x_tr[0]))
        elif self.val == False:
            for xtr, xte, ytr, yte in zip(self.xtr, self.xte, self.ytr, self.yte):
                x_tr = self.get_partition_info(xtr)
                x_te = self.get_partition_info(xte)
                y_tr = self.get_partition_info(ytr)
                y_te = self.get_partition_info(yte)
                parts = [x_tr[0], x_te[0], y_tr[0], y_te[0]]
                if parts.count(parts[0]) != len(
                    parts
                ):  # All elements are the same; checking
                    raise Exception("filepath parsing FAILED")
                partitions.append((xtr, xte, ytr, yte))
                part_IDS.append(int(x_tr[0]))
        return partitions, part_IDS

    def get_partition_info(self, pathstr: str):
        """
        retrieve partition number (first in string), and partition type string (unique identifier)
        """
        base = os.path.basename(pathstr)
        part, temp = base.split(
            "_", 1
        )  # Name is first; this has partition number _ date info. Second half has everything else
        unique, type_ = temp.rsplit(
            "_", 1
        )  # This is everything before final "ytr/yva/etc.feather" part of path
        return part, unique, type_.split(".")[0]

    def write_part_to_log(self, part_id: int = None):
        """
        writes to self.log, which serves as a buffer for keeping track of completed partitions
        """
        if part_id == None:
            raise ValueError("Did not pass partition id to write_to_log")
        elif type(part_id) == str:
            raise ValueError("Improper use of write to log func")
        elif type(part_id) == int:
            position = self.partIDs.index(part_id)
            length = len(self.log)
            if (
                length == position
            ):  # If the index in reference list is equal to length of growing list, then it is the next position in reference list
                self.log.append(part_id)
            else:
                raise Exception("Going out of order on partitions!")


class tfDriver:
    """
    FIRST isntantiate organizer, THEN instantiate driver. You can call organizer properties through driver.

    Class for driving tensorflow models. Mostly method-based. Accepts tuples of filepaths and:
        builds feature arrays,
        instantiates models,
        tunes models,
        reports validation metrics
    """

    def __init__(self, organizer: tf_organizer, prophetic_models=None):
        self.paths = organizer.partitions
        self.organizer = organizer
        self.get_next_part(iter_=False)
        self.x_y = self.prep_x_y()
        self.input_dim = self.x_y[0][0].shape[1]
        if self.organizer.inference is True:
            assert prophetic_models is not None
            # masks = self.prep_masks(self.organizer.masks)
            try:
                self.prophetic = self.organizer.prophetic_features
                self.sort_inference_models(prophetic_models)
                self.curr_models = self.models[0]
                self.curr_prophetic = self.prophetic[0]
            except:
                raise Exception(
                    "Tried to construct a tfDriver instance for inferencing without identifying\
                                models or prophetic features. Check input tf_organizer object."
                )

    def get_next_part(self, iter_=True):
        """
        Get next partition based on ones previously operated on.

        Returns TUPLE with FILEPATHS to .feathers of dataframes
        """
        if len(self.organizer.log) == 0:  # First iteration, get start.
            next = self.organizer.partIDs[0]
            curr_idx = 0
        elif self.organizer.log[-1] == self.organizer.partIDs[-1]:
            ## Condition is done; no more partitions, but this will prevent failure
            return 0
        else:
            current = self.organizer.log[
                -1
            ]  # Retrieve last one written to log, then get next value
            prev_idx = self.organizer.partIDs.index(current)
            curr_idx = prev_idx + 1
            next = self.organizer.partIDs[curr_idx]
        self.organizer.write_part_to_log(next)
        new_current = self.organizer.partitions[curr_idx]
        current_number = self.organizer.log[-1]
        self.current_part = new_current
        self.current_part_id = current_number
        if iter_ is True:
            self.x_y = self.prep_x_y()
            self.input_dim = self.x_y[0][0].shape[1]
            # self.current_mask = self.masks[curr_idx]
            if (
                self.organizer.inference is True
            ):  # When making predictions, iterate on models and preprocessed prophetic features
                try:
                    self.curr_models = self.models[curr_idx]
                    self.curr_prophetic = self.prophetic[curr_idx]
                except IndexError:
                    from warnings import warn

                    warn(
                        "Looks like prediction workflow ran out of pre-trained models before exhausting all partitions. Stopping now."
                    )
                    return 0  # Do not have models for ALL of the partitions; the earlier check can fail.
        print("Getting next partition", "\n\n", self.organizer.log[-1], new_current)
        # return new_current,current_number ### vestigial - no longer used

    ### Depreciated - this is done beforehand
    # def prep_masks(self, paths: tuple):
    #     out = []
    #     for pth in paths:
    #         df = pd.read_csv(pth, header=0, index_col=0)
    #         np_mask = df.to_numpy()
    #         out.append(np_mask)
    #     return tuple(out)
    ### Depreciated
    def load_prophetic_hypermodels_and_x(
        self,
    ):
        """
        Load current model and prophetic features

        model (compiled), and pd.DataFrame (index=instance, column=features)
        """
        assert self.organizer.inference is True
        model_paths = self.curr_models
        feat_path = self.curr_prophetic
        models = []
        for path in model_paths:
            # model = tf.keras.saving.load_model(path)
            model = tf.keras.models.load_model(path)
            models.append(model)
        feat = pd.read_feather(feat_path).transpose()
        return models, feat

    def sort_inference_models(self, allmodels):
        """
        pass all model paths (list of string paths from glob()) as an argument
        """
        assert self.organizer.inference is True
        # assert self.allmodels
        # all_models = self.allmodels
        model_info = [
            self.organizer.get_partition_info(m)[0].split("hpset")[0] for m in allmodels
        ]
        from collections import OrderedDict

        output = []
        sort = OrderedDict()
        for id, pa in zip(model_info, allmodels):
            if id in sort.keys():
                sort[id].append(pa)
            else:
                sort[id] = [pa]
        for id_, paths in sort.items():
            output.append(tuple(paths))
        self.models = output
        # return output

    def prep_x_y(self):
        """
        Prep x and y for modeling. Returns tuple of dataframes transposed so that
        INDEX labels are instances and COLUMNS are features
        """
        if self.organizer.val == True:
            xtr, xva, xte, ytr, yva, yte = self.current_part  # paths
            xtr_, xva_, xte_, ytr_, yva_, yte_ = self._feather_to_np(
                (xtr, xva, xte, ytr, yva, yte)
            )  # dfs
            return xtr_, xva_, xte_, ytr_, yva_, yte_
        elif self.organizer.val == False:
            xtr, xte, ytr, yte = self.current_part  # paths
            xtr_, xte_, ytr_, yte_ = self._feather_to_np((xtr, xte, ytr, yte))  # dfs
            return xtr_, xte_, ytr_, yte_

    def regression_model(self, hp):
        """
        This will look for one hidden layer NNs, with dropouts, and an output layer with no activation function
        It will allow changing activation functions between layers.

        NOTE: if interested in multiclass classification, use softmax with # nodes = # classes,
        multilabel classification use sigmoid with # nodes = number labels, and
        use linear with regression and one node

        """

        # input_dimension=self.input_dim
        model = Sequential()
        # model.add(Input(shape=input_dimension))
        hp_n_1 = hp.Int(
            "nodes_1", min_value=256, max_value=1024, step=16
        )  # 48 states possible
        hp_n_2 = hp.Int(
            "nodes_2", min_value=16, max_value=64, step=4
        )  # 48 states possible
        # hp_n_3 = hp.Int("nodes_3", min_value=8, max_value=256, step=8) ##DEV
        hp_noise = hp.Float("gaus_noise", min_value=0.005, max_value=0.08, step=0.005)
        hp_d_1 = hp.Float("dropout1", min_value=0.0, max_value=0.65)
        # hp_d_2 = hp.Float("dropout2", min_value=0.0, max_value=0.65) ##DEV
        # hp_a_1 = hp.Choice('act1',values=['relu','selu','softmax','tanh','gelu'])
        # hp_a_2 = hp.Choice('act2',values=['relu','selu','softmax','tanh','gelu'])
        # hp_a_3 = hp.Choice('act3',values=['relu','selu','softmax','tanh','gelu'])
        # hp_a_1 = hp.Choice("act1", values=["relu"])
        # hp_a_2 = hp.Choice("act2", values=["relu"])
        # hp_a_3 = hp.Choice("act3", values=["relu"])
        model.add(
            tf.keras.layers.GaussianNoise(
                # stddev=0.05,
                stddev=hp_noise,
                # seed=1234,
                input_shape=(self.input_dim,),
            )
        )
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(
            Dense(
                hp_n_1,
                # activation=hp_a_1,
                activation="relu",
                # activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4),
                # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4)
                # input_shape=(self.input_dim,)
            )
        )
        model.add(Dropout(hp_d_1))
        model.add(
            Dense(
                hp_n_2,
                # activation=hp_a_2,
                activation="relu",
                # activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4),
                # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4)
            )
        )
        # model.add(Dropout(hp_d_2))
        # model.add(
        #     Dense(
        #         hp_n_3,
        #         # activation=hp_a_3,
        #         activation="relu",
        #     )
        # )
        model.add(Dense(1, activation="linear"))
        # hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        # opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # opt = tf.keras.optimizers.SGD(learning_rate=1e-4)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=100, decay_rate=0.95
        )
        # opt = tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)
        # opt = tf.keras.optimizers.Adagrad(learning_rate=1e-4)
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule
        )  ## DEV added legacy - apparently needs it?
        model.compile(
            optimizer=opt,
            loss="mse",
            metrics=["mean_absolute_error", "mean_squared_error"],
            run_eagerly=False,
        )
        return model

    def mc_classification_model_5(self, hp):
        """
        This will look for one hidden layer NNs, with dropouts, and an output layer with no activation function
        It will allow changing activation functions between layers.

        NOTE: if interested in multiclass classification, use softmax with # nodes = # classes,
        multilabel classification use sigmoid with # nodes = number labels, and
        use linear with regression and one node

        """

        def top_2_acc(y_true, y_pred):
            return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

        def kl_div(y_true, y_pred):
            return tf.keras.metrics.kullback_leibler_divergence(y_true, y_pred)

        # input_dimension=self.input_dim
        model = Sequential()
        # model.add(Input(shape=input_dimension))
        hp_n_1 = hp.Int(
            "nodes_1", min_value=256, max_value=2496, step=64
        )  # 48 states possible
        hp_n_2 = hp.Int(
            "nodes_2", min_value=128, max_value=968, step=24
        )  # 48 states possible
        hp_n_3 = hp.Int("nodes_3", min_value=8, max_value=256, step=8)
        hp_noise = hp.Float("gaus_noise", min_value=0.005, max_value=0.08, step=0.005)
        hp_d_1 = hp.Float("dropout1", min_value=0.0, max_value=0.65)
        hp_d_2 = hp.Float("dropout2", min_value=0.0, max_value=0.65)
        # hp_a_1 = hp.Choice('act1',values=['relu','selu','softmax','tanh','gelu'])
        # hp_a_2 = hp.Choice('act2',values=['relu','selu','softmax','tanh','gelu'])
        # hp_a_3 = hp.Choice('act3',values=['relu','selu','softmax','tanh','gelu'])
        # hp_a_1 = hp.Choice("act1", values=["relu"])
        # hp_a_2 = hp.Choice("act2", values=["relu"])
        # hp_a_3 = hp.Choice("act3", values=["relu"])
        model.add(
            tf.keras.layers.GaussianNoise(
                # stddev=0.05,
                stddev=hp_noise,
                # seed=1234,
                input_shape=(self.input_dim,),
            )
        )
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(
            Dense(
                hp_n_1,
                # activation=hp_a_1,
                activation="relu",
                # activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4),
                # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4)
                # input_shape=(self.input_dim,)
            )
        )
        model.add(Dropout(hp_d_1))
        model.add(
            Dense(
                hp_n_2,
                # activation=hp_a_2,
                activation="relu",
                # activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4),
                # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4)
            )
        )
        model.add(Dropout(hp_d_2))
        model.add(
            Dense(
                hp_n_3,
                # activation=hp_a_3,
                activation="relu",
            )
        )
        model.add(Dense(5, activation="softmax"))
        # hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        # opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # opt = tf.keras.optimizers.SGD(learning_rate=1e-4)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=100, decay_rate=0.95
        )
        # opt = tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)
        # opt = tf.keras.optimizers.Adagrad(learning_rate=1e-4)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                "val_accuracy",
                tf.keras.metrics.CategoricalCrossentropy(),
                top_2_acc,
                kl_div,
                tf.keras.metrics.FalsePositives(
                    thresholds=0, name="false_pos", dtype=bool
                ),
                tf.keras.metrics.FalseNegatives(
                    thresholds=0, name="false_neg", dtype=bool
                ),
                tf.keras.metrics.TruePositives(
                    thresholds=0, name="true_pos", dtype=bool
                ),
                tf.keras.metrics.TrueNegatives(
                    thresholds=0, name="true_neg", dtype=bool
                ),
            ],
        )
        return model

    def upd_search_model(self, hp):
        """
        This will look for one hidden layer NNs, with dropouts, and an output layer with no activation function
        It will allow changing activation functions between layers.

        NOTE: if interested in multiclass classification, use softmax with # nodes = # classes,
        multilabel classification use sigmoid with # nodes = number labels, and
        use linear with regression and one node

        """

        input_dimension = self.input_dim
        model = Sequential()
        model.add(Input(shape=input_dimension))
        hp_n_1 = hp.Int(
            "nodes_1", min_value=64, max_value=512, step=16
        )  # 28 states possible
        hp_n_2 = hp.Int(
            "nodes_2", min_value=32, max_value=128, step=8
        )  # 28 states possible
        hp_n_3 = hp.Int("nodes_3", min_value=8, max_value=64, step=8)
        hp_noise = hp.Float("gaus_noise", min_value=0.005, max_value=0.08, step=0.005)
        hp_d_1 = hp.Float("dropout1", min_value=0.0, max_value=0.65)
        hp_d_2 = hp.Float("dropout2", min_value=0.0, max_value=0.65)
        hp_a_1 = hp.Choice("act1", values=["relu", "selu", "softmax", "gelu"])
        hp_a_2 = hp.Choice("act2", values=["relu", "selu", "softmax", "gelu"])
        hp_a_3 = hp.Choice("act3", values=["relu", "selu", "softmax", "gelu"])
        # hp_a_1 = hp.Choice("act1", values=["relu"])
        hp_a_2 = hp.Choice("act2", values=["relu"])
        hp_a_3 = hp.Choice("act3", values=["relu"])
        model.add(
            GaussianNoise(
                # stddev=0.05,
                stddev=hp_noise,
                # seed=1234,
                input_shape=(self.input_dim,),
            )
        )
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(
            Dense(
                hp_n_1,
                activation=hp_a_1,
                # activation = 'relu',
                # activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4),
                # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4)
                # input_shape=(self.input_dim,)
            )
        )
        model.add(Dropout(hp_d_1))
        model.add(
            Dense(
                hp_n_2,
                activation=hp_a_2,
                # activation='relu',
                # activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4),
                # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-4)
            )
        )
        model.add(Dropout(hp_d_2))
        model.add(
            Dense(
                hp_n_3,
                activation=hp_a_3,
                # activation = 'relu',
            )
        )
        model.add(Dense(1, activation="linear"))
        # hp_lr = hp.Choice('learning_rate',values=[1e-2,1e-3,1e-4,1e-5,1e-6])
        # opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
        # opt = tf.keras.optimizers.SGD(learning_rate=1e-4)
        opt = Adadelta(learning_rate=1)
        # opt = tf.keras.optimizers.Adagrad(learning_rate=1e-4)
        model.compile(
            optimizer=opt,
            loss="mse",
            metrics=["mean_absolute_error", "mean_squared_error"],
        )
        return model

    def _trans_xy_(cls, desc: pd.DataFrame):
        """
        Output feature arrays from input dataframes.

        Note: this is designed for an ecosystem which puts instances as columns and features as
        rows.

        Returns array and labels
        """
        transposition = desc.transpose()
        if type(transposition.index[0]) != str:
            raise ValueError("Check dataframe construction")
        feature_array = transposition.to_numpy()
        return feature_array, transposition.index

    def _feather_to_np(cls, paths: tuple):
        """
        Take objects and output dfs of objects in same order

        Transposes them so that instance names are now index (they are column labels in .feather storage)

        Gives back tuple of tuples: (array, index labels) for each entry

        """
        out = []
        idx_ = []
        for pth in paths:
            df = pd.read_feather(pth)
            df_tr = cls._trans_xy_(df)
            out.append(df_tr)
        return tuple(out)

    def _lr_sched(epoch, lr):
        """
        Depreciated - a simple learning rate scheduler
        """
        if epoch < 25:
            return lr
        elif 25 <= epoch < 50:
            return lr * exp(-0.1)
        elif 50 <= epoch < 100:
            return lr * exp(-0.05)


def get_mae_metrics(
    model: kt.HyperModel,
    X: np.array,
    inference_x: np.array,
    y: np.array,
    infer_labels: np.array,
):
    """
    Get inferences on partitions, then evaluate errors and report mae

    2/26/2022 VAL ONLY

    """
    yva, yte = infer_labels
    train_errors, val_errors, test_errors = compute_residuals(
        model, X, inference_x, y, (yva, yte)
    )
    trainmae = np.mean(train_errors)
    valmae = np.mean(val_errors)
    testmae = np.mean(test_errors)
    return trainmae, valmae, testmae


def get_mse_metrics(
    model: kt.HyperModel,
    X: np.array,
    inference_x: np.array,
    y: np.array,
    infer_labels: np.array,
):
    """
    Get inferences, then evaluate mse

    2/26/2022 VAL ONLY

    """
    yva, yte = infer_labels
    train_errors, val_errors, test_errors = compute_residuals(
        model, X, inference_x, y, (yva, yte)
    )
    trainmse = np.sqrt(np.sum(np.square(train_errors))) / len(train_errors)
    valmse = np.sqrt(np.sum(np.square(val_errors))) / len(val_errors)
    testmse = np.sqrt(np.sum(np.square(test_errors))) / len(test_errors)
    return trainmse, valmse, testmse


def compute_residuals(model, X, inference_x: np.array, y, infer_labels: np.array):
    """
    Get residual errors for partitions

    2/26/2022 VAL ONLY

    """
    yva, yte = infer_labels
    ytr_p, yva_p, yte_p = model_inference(model, X, inference_x)
    train_errors = abs(ytr_p - y)
    val_errors = abs(yva_p - yva)
    test_errors = abs(yte_p - yte)
    return train_errors, val_errors, test_errors


def model_inference(model, X, inference_x: np.array):
    """
    Takes inference tuple, and processes it. This has val , test, and prophetic X.

    Use trained model (or instantiated from identified parameters)

    Outputs predicted values based on descriptors
    """
    if len(inference_x) == 3:
        X_val, X_test, Xp = inference_x
        ytr_p = model.predict(X).ravel()
        yte_p = model.predict(X_test).ravel()
        yva_p = model.predict(X_val).ravel()
        ypr_p = model.predict(Xp).ravel()
        return ytr_p, yva_p, yte_p, ypr_p
    elif len(inference_x) == 2:
        X_val, X_test = inference_x
        ytr_p = model.predict(X).ravel()
        yte_p = model.predict(X_test).ravel()
        yva_p = model.predict(X_val).ravel()
        return ytr_p, yva_p, yte_p
    elif len(inference_x) == 1:
        X_test = inference_x
        ytr_p = model.predict(X).ravel()
        yte_p = model.predict(X_test).ravel()
        return ytr_p, yte_p
    else:
        raise Exception("Pass inference array; did not get proper number")


def save_model(model: Sequential):
    """
    Take trained model and save it and its weights

    Depreciated - now use .keras file
    """
    model_json = model.to_json()
    os.mkdirs("json_out/", exist_ok=True)
    os.mkdirs("weights_out/", exist_ok=True)
    with open("json_out/nn_" + model.name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("weights_out/nn_" + model.name + ".weights.h5")


def get_hps(hps: kt.HyperParameters()):
    # nodes_list = [hps.get("nodes_1"), hps.get("nodes_2"), hps.get("nodes_3")] ## 2 HLs
    # dropout_list = [hps.get("dropout1"), hps.get("dropout2")]## 2 HLs
    nodes_list = [hps.get("nodes_1"), hps.get("nodes_2")]  ## 1 HLs
    dropout_list = [hps.get("dropout1")]  ## 1 HLs

    # act_list = [hps.get("act1"), hps.get("act2"), hps.get("act3")]
    node_str = "_".join([f + "n" for f in map(str, nodes_list)])
    dropout_str = "_".join([f + "d" for f in map(str, dropout_list)])
    # act_str = "_".join([f + "a" for f in map(str, act_list)]) #shrinking search space
    # out = "_".join([node_str, dropout_str, act_str])
    out = "_".join([node_str, dropout_str])
    return out


def check_for_completed(drive: tfDriver):
    """
    Check if restarting a job. If so, proceed to the point where the last job left off.
    """
    out_dir_path = drive.model_out_path
    models = glob(f"{out_dir_path}*hpset*.keras")
    names = list(set([k.split("out/")[1].split("hpset")[0] for k in models]))
    completed = [
        str(f) for f in drive.organizer.partIDs if str(f) in names
    ]  # partIDs are integers, checking if they are done
    return completed, len(completed)


def hypermodel_search(
    experiment,
    max_val_cutoff=20,
    model_type="regression",
    tuner_objective="val_mean_absolute_error",
    deep_objective="val_mean_absolute_error",
    epoch_depth=200,
    num_hypermodels=3,
    cpu_testing=False,  # TEST
):
    project = Project()
    """
    Driver for hypermodel searching.

    experiment argument is required, and gives a label to the outputs - should be unique i.e. a date

    model_type must be "regression" or "classification" (multiclass for zero and yield quartiles, 5 total categories)

    tuner_objective is the objective for the HyperBand algorithm. Set to MAE or MSE on validation data

    deep_objective is used to identify the optimal epoch for the best hyperparameters identified by the tuner.

    max_val_cutoff is for whatever the deep_objective is; it will skip a partition if the best model is terrible

    epoch_depth is how deep a fit the best hypermodel will undergo. This searches for the best number of epochs using the deep objective.
        The deep objective should be validation mse or mae so that it will increase as the models begin to overfit. The best epoch will be selected
        from the minimum on the validation metric curve, ensuring a good fit.

    num_hypermodels is the integer number of hypermodel configurations to be saved from the HyperBand output. This helps to bolster the final ensemble.
        Assuming that multiple reasonably performant hypermodels are identified with each tuner instance, multiple models can be kept from the highest
        performing ones identified. Increasing this could include "bad" models when only a few performant hypermodels are identified. It should be tuned.
        5 is the default value here, and qualitatively, 10 was too high most of the time. This can be adjusted, but it should come with adjustments to the tuner
        parameters (such as more trials)

    This function is a procedure which will generate and serialize an output dictionary to a json file. The metrics can be parsed and analyzed later. The models are
        serialized as well. For Regression, a pred vs obs plot is saved as a .png, one for train vs val and one for train vs test. For classification, no plot is saved,
        but several classification metrics are saved to the JSON which are specific to classification.


    """
    if cpu_testing == False:
        config = tf.compat.v1.ConfigProto(
            device_count={"GPU": 0}
        )  ## DEV - trying to specify a specific GPU.
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)
    # elif cpu_testing == True:  # TEST - this was not the goal here. CPU is not going to be useful for this.
    #     config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    #     session = tf.compat.v1.Session(config=config)
    # import sys
    # exp = sys.argv[1]
    assert isinstance(experiment, str)
    # with tf.device("/GPU:0"):  ### Need a solution for this OR just don't and let it run in a container without specific hardware and multiple jobs
    # date__ = date.today().strftime("%b-%d-%Y-%H-%M") + "_" + experiment
    date__ = experiment  # Should not be a slash before or after
    tforg = tf_organizer(
        name="mdl_srch_" + experiment,
        partition_dir=f"{project.partitions}/real",  # noslash needed
        validation=True,
    )  # Need to name a subfolder for the partitions and add it here.

    drive = tfDriver(tforg)
    json_buffer_path = (
        f"{project.output}/{date__}/"  # There is already a slash after output
    )
    out_dir_path = f"{project.output}/{date__}/out/"  # Both slashes for out needed here
    drive.model_out_path = out_dir_path
    os.makedirs(json_buffer_path, exist_ok=True)
    os.makedirs(out_dir_path, exist_ok=True)
    # Directorys should be covered in above os.makedirs() call
    # splits = {}
    # csvs = glob(tforg.part_dir.rsplit("/", 1)[0] + "/*csv")
    # for f in csvs:
    #     key = f.rsplit("_", 1)[1].split(".")[0]
    #     val = pd.read_csv(f, header=0, index_col=0)
    #     splits[key] = val.to_dict(orient="list")
    # # print(len(drive.organizer.partitions))
    # # tf.debugging.set_log_device_placement(True)
    # # gpus = tf.config.list_logical_devices('GPU')
    # ### Restart; skipping already done partitions ####
    # # for i in range(6):
    # #     drive.get_next_part()
    completed, iter_ = check_for_completed(drive)
    if len(completed) > 0:  # Restarting - don't overwrite old file.
        import uuid

        logfile = open(f"{json_buffer_path}logfile{uuid.uuid1().hex}.txt", "w")
        print(
            f"WARNING  -  looks like model training was restarted, and {iter_} number of partitions were completed previously \
out of {len(drive.organizer.partitions)}. Attempting to detect which are complete and skip them. Completed partitions are\
{completed}"
        )
    else:  # First time. Load logfile and start results buffer.
        logfile = open(f"{json_buffer_path}logfile.txt", "w")
        with open(f"{json_buffer_path}tforg_results_buffer.json", "w") as k:
            k.write("{}")

    for __k in range(len(drive.organizer.partitions)):
        name_ = str(drive.current_part_id)
        if name_ in completed:
            drive.get_next_part()
            print(f"RESTART - SKIPPING PARTITION {name_}")
            continue
        if model_type == "regression":
            xtr, xval, xte, ytr, yval, yte = [
                f[0] for f in drive.x_y
            ]  # Normal procedure - get x,y data
            tuner = kt.tuners.Hyperband(
                drive.regression_model,
                objective=kt.Objective(tuner_objective, "min"),
                max_epochs=100,
                factor=3,
                # distribution_strategy=tf.distribute.MirroredStrategy(),
                # distribution_strategy=tf.distribute.MirroredStrategy(gpus),
                project_name="test_" + name_,
                directory=out_dir_path + name_,
                # directory=r'/home/nir2/tfwork/ROCHE_ws/Feb-26-2022-00-00/out/1000/',
                overwrite=False,
            )
        elif model_type == "classification":
            from somn.calculate.preprocess import prep_mc_labels

            ### Need to prep y data for multiclass labels. This is fast and avoids having to save separate partitions
            xtr_l, xval_l, xte_l, ytr_l, yval_l, yte_l = [
                f for f in drive.x_y
            ]  # Tuples (values, labels)
            ytr = prep_mc_labels(pd.DataFrame(ytr_l[0], index=ytr_l[1]))
            yval = prep_mc_labels(pd.DataFrame(yval_l[0], index=yval_l[1]))
            yte = prep_mc_labels(pd.DataFrame(yte_l[0], index=yte_l[1]))
            xtr = xtr_l[0]
            xval = xval_l[0]
            xte = xte_l[0]
            tuner = kt.tuners.Hyperband(
                drive.mc_classification_model_5,
                objective=kt.Objective(tuner_objective, "min"),
                max_epochs=120,
                factor=3,
                # distribution_strategy=tf.distribute.MirroredStrategy(),
                # distribution_strategy=tf.distribute.MirroredStrategy(gpus),
                project_name="test_" + name_,
                directory=out_dir_path + name_,
                # directory=r'/home/nir2/tfwork/ROCHE_ws/Feb-26-2022-00-00/out/1000/',
                overwrite=False,
            )
            try:
                assert ytr.shape[1] == 5
            except:
                raise Exception(
                    "Must pass properly formulated classification data. The y-data shape does not match the implemented classifier model\n \
The y data must be a vector with 5 columns corresponding to zero yield and yield quartiles. This can be prepared using\n \
the utility function somn.calculate.preprocess.prep_mc_labels"
                )
        stop_early = EarlyStopping(monitor=tuner_objective, patience=10)
        stop_nan = TerminateOnNaN()
        tensorboard = TensorBoard(log_dir=out_dir_path + name_, histogram_freq=1)
        tuner.search(
            xtr,
            ytr,
            # epochs=25,
            # validation_split=0.10,
            validation_data=(xval, yval),
            callbacks=[stop_early, stop_nan],
            batch_size=64,
            verbose=0,
        )
        tuner.results_summary()
        tuner.search_space_summary(extended=True)
        best_hp_list = tuner.get_best_hyperparameters(
            num_trials=num_hypermodels
        )  # Get top hypers
        best_hps = best_hp_list[0]
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
            xtr,
            ytr,
            batch_size=64,
            workers=64,
            use_multiprocessing=True,
            # validation_split=0.15,
            validation_data=(xval, yval),
            callbacks=[tensorboard, stop_nan],
            epochs=epoch_depth,
        )  ## CHANGE increased this to 200 with change in optimizer to Adam, variable learning rate, and now using data augmentation during train
        val_hist = history.history[deep_objective]
        if min(val_hist) > max_val_cutoff:
            print(f"validation error {min(val_hist)} was too large for part {name_}")
            drive.get_next_part()
            continue
        best_epoch = val_hist.index(min(val_hist))
        for i, hps in enumerate(best_hp_list):
            hypermodel = tuner.hypermodel.build(hps)
            history = hypermodel.fit(
                xtr,
                ytr,
                epochs=best_epoch,
                batch_size=32,
                workers=32,
                use_multiprocessing=True,
                # validation_split=0.15,
                validation_data=(xval, yval),
            )
            val_result = hypermodel.evaluate(xval, yval)
            tes_result = hypermodel.evaluate(xte, yte)
            if model_type == "regression":
                mse_result = get_mse_metrics(
                    hypermodel, xtr, (xval, xte), ytr, (yval, yte)
                )
                mae_result = get_mae_metrics(
                    hypermodel, xtr, (xval, xte), ytr, (yval, yte)
                )
            elif model_type == "classification":
                mse_result = ("classifier",)
                mae_result = "classifier"
            else:
                raise Exception(
                    "Must pass model type to hypermodel search function as regression or classification"
                )
            hypermodel.save(
                out_dir_path + name_ + "hpset" + str(i) + "_" + get_hps(hps) + ".keras",
                overwrite=False,
            )
            #### Not tested yet - for later development ####
            # if (
            #     mae_result[1] < 20.0
            # ):  # Validation must pass 20% MAE. Arbitrary cutoff for whether to save a model or not.
            #     ### Need to test to_json() and tf.keras.models.model_from_json() for just saving config.
            #     # tf.keras.saving.save_model(hypermodel,out_dir_path + name_ + "hpset" + str(i) + "_" + get_hps(hps) + ".h5",overwrite=False,save_format='h5',save_traces=False)
            #     ### Below is a prototype for json serialization of JUST the model configuration. This removes the needless
            #     ### save/load of weights that will just be retrained anyway.
            #     # json_model = hypermodel.to_json()
            #     # with open(f"{out_dir_path}{name_}_hpset{str(i)}_{get_hps(hps)}.json",'w') as g:
            #     #     json.dump(json_model,g)
            #### Untested development json serialization of models
            yval_p = hypermodel.predict(xval).ravel()
            yte_p = hypermodel.predict(xte).ravel()
            ytr_p = hypermodel.predict(xtr).ravel()
            val_h = history.history["val_loss"]
            train_h = history.history["loss"]
            if i == 0:  ## First iteration, write results dictionary entry
                tforg.results[name_] = {
                    "best_epoch": best_epoch,
                    "top_"
                    + str(len(best_hp_list))
                    + "_hyper_strings": [get_hps(f) for f in best_hp_list],
                    "val_history_deep": val_hist,
                    "val_result": {str(i + 1) + "hp_" + name_: val_result},
                    "test_result": {str(i + 1) + "hp_" + name_: tes_result},
                    "mse_result": {str(i + 1) + "hp_" + name_: mse_result},
                    "mae_result": {str(i + 1) + "hp_" + name_: mae_result},
                    "train_loss": {str(i + 1) + "hp_" + name_: train_h},
                    "val_loss": {str(i + 1) + "hp_" + name_: val_h},
                    # "test_correlation": {},
                    # "val_correlation": {},
                    "regression_stats": {},
                }
                if model_type == "classification":
                    tforg.results[name_]["class_metrics"] = {}
                # tforg.results[name_]["split"] = splits["spl" + name_]
            else:  ## Other iterations, append results to dictionary entry
                keys = [
                    "val_result",
                    "test_result",
                    "mse_result",
                    "mae_result",
                    "train_loss",
                    "val_loss",
                ]  # ordered based on keys above
                vals = [
                    val_result,
                    tes_result,
                    mse_result,
                    mae_result,
                    train_h,
                    val_h,
                ]  # ordered based on keys above
                for k, v in zip(
                    keys, vals
                ):  # requires above lists to be ordered correctly, but is fairly efficient
                    tforg.results[name_][k][str(i + 1) + "hp_" + name_] = v
            # plot_results(outdir=out_dir_path,expkey=name_+'_'+str(i)+'hps_valtest',train=(yval.ravel(),yval_p),test=(yte.ravel(),yte_p))
            if model_type == "regression":
                reg_lin_met = plot_results(
                    outdir=out_dir_path,
                    expkey=name_ + "test" + "_hp" + str(i),
                    train=(ytr.ravel(), ytr_p),
                    val=(yval.ravel(), yval_p),
                    test=(yte.ravel(), yte_p),
                )
                # val_lin_met = plot_results(
                #     outdir=out_dir_path,
                #     expkey=name_ + "val" + "_hp" + str(i),
                #     train=(ytr.ravel(), ytr_p),
                #     test=(yval.ravel(), yval_p),
                # )
                #### Linear regression metrics - slope, intercept, then R2 ####
                tforg.results[name_]["regression_stats"][i + 1] = reg_lin_met
                # tforg.results[name_]["val_correlation"][i + 1] = val_lin_met
            elif model_type == "classification":
                fp = history.history["false_pos"]
                fn = history.history["false_neg"]
                tp = history.history["true_pos"]
                tn = history.history["true_neg"]
                tforg.results[name_]["class_metrics"][f"{i+1}_false_pos"] = fp
                tforg.results[name_]["class_metrics"][f"{i+1}_false_neg"] = fn
                tforg.results[name_]["class_metrics"][f"{i+1}_true_pos"] = tp
                tforg.results[name_]["class_metrics"][f"{i+1}_true_neg"] = tn
                tforg.results[name_]["class_metrics"][f"{i+1}_val_top2"] = (
                    tf.keras.metrics.top_k_categorical_accuracy(yval, yval_p, k=2)
                )
                tforg.results[name_]["class_metrics"][f"{i+1}_test_top2"] = (
                    tf.keras.metrics.top_k_categorical_accuracy(yte, yte_p, k=2)
                )
                tforg.results[name_]["class_metrics"][f"{i+1}_val_kldiv"] = (
                    tf.keras.metrics.kullback_leibler_divergence(yval, yval_p)
                )
                tforg.results[name_]["class_metrics"][f"{i+1}_test_kldiv"] = (
                    tf.keras.metrics.kullback_leibler_divergence(yte, yte_p)
                )
            ### DEV CLEANUP ###
            del hypermodel
            ### Apparently, keras is notorious for leaving remnants in memory. gc.collect() is supposed to help.
            gc.collect()
            ###################
        buffer_log = json.dumps(tforg.results[name_])
        logfile.write(name_ + "," + str(__k) + buffer_log + "\n")
        logfile.flush()
        print("Completed partition " + str(__k) + "\n\n")
        with open(f"{json_buffer_path}tforg_results_buffer.json", "r") as p:
            results = json.load(p)
        results.update(tforg.results)
        with open(
            f"{json_buffer_path}tforg_results_buffer.json", "w"
        ) as j:  # Should just overwrite
            json.dump(results, j)
        tforg.results = {}
        ### Let's see if not carrying the tforg.results through the whole process is worthwhile.
        ### DEV CLEANUP ###
        del tuner
        gc.collect()
        ###################
        tf.keras.backend.clear_session()  # Cleanup - this is supposed to help lower memory leaks on iterations.
        drive.get_next_part()  # Iterate to next partition

    # with open(f"{json_buffer_path}final_complete_log{date__}.json", "w") as g:
    #     g.write(json.dumps(tforg.results))


# if __name__ == "__main__":
#     import sys

#     experiment = sys.argv[1]
#     hypermodel_search(experiment=experiment)
