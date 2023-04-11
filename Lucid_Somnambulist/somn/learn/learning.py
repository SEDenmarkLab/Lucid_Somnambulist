from glob import glob
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, GaussianNoise
from keras.optimizers import Adam, Adadelta
from math import exp
import json
from somn.workflows import PART_, OUTPUT_
import kerastuner as kt
from keras.callbacks import (
    EarlyStopping,
    TerminateOnNaN,
    ReduceLROnPlateau,
    TensorBoard,
)
from somn.depict.evaluate import plot_results
import numpy as np


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

    def __init__(self, name, partition_dir="", validation=True):
        self.name = name
        self.part_dir = partition_dir
        self.val = validation
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
                    raise Exception("filepath parsing FAILED")
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

    def __init__(self, organizer: tf_organizer):
        self.paths = organizer.partitions
        self.organizer = organizer
        self.get_next_part(iter_=False)
        self.x_y = self.prep_x_y()
        self.input_dim = self.x_y[0][0].shape[1]

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
            # print("DEBUG")
            return None
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
        if iter_ == True:
            self.x_y = self.prep_x_y()
            self.input_dim = self.x_y[0][0].shape[1]
        print("Getting next partition", "\n\n", self.organizer.log[-1], new_current)
        # return new_current,current_number ### vestigial - no longer used

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

    def legacy_search_model(self, hp):
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
            "nodes_1", min_value=256, max_value=3000, step=64
        )  # 48 states possible
        hp_n_2 = hp.Int(
            "nodes_2", min_value=128, max_value=1280, step=24
        )  # 48 states possible
        hp_n_3 = hp.Int("nodes_3", min_value=8, max_value=256, step=8)
        hp_noise = hp.Float("gaus_noise", min_value=0.005, max_value=0.08, step=0.005)
        hp_d_1 = hp.Float("dropout1", min_value=0.0, max_value=0.65)
        hp_d_2 = hp.Float("dropout2", min_value=0.0, max_value=0.65)
        # hp_a_1 = hp.Choice('act1',values=['relu','selu','softmax','tanh','gelu'])
        # hp_a_2 = hp.Choice('act2',values=['relu','selu','softmax','tanh','gelu'])
        # hp_a_3 = hp.Choice('act3',values=['relu','selu','softmax','tanh','gelu'])
        hp_a_1 = hp.Choice("act1", values=["relu"])
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
        hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        # opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        opt = Adam(learning_rate=hp_lr)
        # opt = tf.keras.optimizers.SGD(learning_rate=1e-4)
        # opt = tf.keras.optimizers.Adadelta(learning_rate=1e-4)
        # opt = tf.keras.optimizers.Adagrad(learning_rate=1e-4)
        model.compile(
            optimizer=opt,
            loss="mse",
            metrics=["mean_absolute_error", "mean_squared_error"],
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
        if epoch < 25:
            return lr
        elif 25 <= epoch < 50:
            return lr * exp(-0.1)
        elif 50 <= epoch < 100:
            return lr * exp(-0.05)


def get_mae_metrics(
    model: kt.HyperModel,
    X: np.array,
    inference_x: (np.array),
    y: np.array,
    infer_labels: (np.array),
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
    inference_x: (np.array),
    y: np.array,
    infer_labels: (np.array),
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


def compute_residuals(model, X, inference_x: (np.array), y, infer_labels: (np.array)):
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


def model_inference(model, X, inference_x: (np.array)):
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
    """
    model_json = model.to_json()
    os.mkdirs("json_out/", exist_ok=True)
    os.mkdirs("weights_out/", exist_ok=True)
    with open("json_out/nn_" + model.name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("weights_out/nn_" + model.name + ".weights.h5")


def get_hps(hps: kt.HyperParameters()):

    nodes_list = [hps.get("nodes_1"), hps.get("nodes_2"), hps.get("nodes_3")]
    dropout_list = [hps.get("dropout1"), hps.get("dropout2")]
    act_list = [hps.get("act1"), hps.get("act2"), hps.get("act3")]
    node_str = "_".join([f + "n" for f in map(str, nodes_list)])
    dropout_str = "_".join([f + "d" for f in map(str, dropout_list)])
    act_str = "_".join([f + "a" for f in map(str, act_list)])
    out = "_".join([node_str, dropout_str, act_str])
    return out


def main(experiment, min_epoch_cutoff=50):
    """
    Driver for hypermodel searching.

    experiment argument is required, and gives a label to the outputs - should be unique i.e. a date
    min_epoch_cutoff will reject a set of hypermodels after the hyperband search if that number of epochs or greater is not the best epoch.


    """
    # import sys
    # exp = sys.argv[1]
    assert isinstance(experiment, str)
    # with tf.device("/GPU:0"):  ### Need a solution for this OR just don't and let it run in a container without specific hardware and multiple jobs
    # date__ = date.today().strftime("%b-%d-%Y-%H-%M") + "_" + experiment
    date__ = experiment  # Should not be a slash before or after
    tforg = tf_organizer(
        name="mdl_srch_" + experiment,
        partition_dir=PART_,
        validation=True,
    )
    drive = tfDriver(tforg)
    json_buffer_path = OUTPUT_ + date__  # Slash after output
    out_dir_path = OUTPUT_ + date__ + "/out/"  # Both slashes for out made here
    os.makedirs(json_buffer_path, exist_ok=True)
    os.makedirs(out_dir_path, exist_ok=True)
    logfile = open(
        json_buffer_path + "/logfile.txt", "w"
    )  # Directorys should be covered in above os.makedirs() call
    splits = {}
    csvs = glob(tforg.part_dir.rsplit("/", 1)[0] + "/*csv")
    for f in csvs:
        key = f.rsplit("_", 1)[1].split(".")[0]
        val = pd.read_csv(f, header=0, index_col=0)
        splits[key] = val.to_dict(orient="list")
    # print(len(drive.organizer.partitions))
    # tf.debugging.set_log_device_placement(True)
    # gpus = tf.config.list_logical_devices('GPU')
    ### Restart; skipping already done partitions ####
    # for i in range(6):
    #     drive.get_next_part()

    for __k in range(len(drive.organizer.partitions)):
        xtr, xval, xte, ytr, yval, yte = [f[0] for f in drive.x_y]
        name_ = str(drive.current_part_id)
        # tuner = kt.tuners.Hyperband(drive.full_search_model,
        #             objective=kt.Objective('val_mean_squared_error','min'),
        #             max_epochs=120,
        #             factor=3,
        #             # distribution_strategy=tf.distribute.MirroredStrategy(),
        #             # distribution_strategy=tf.distribute.MirroredStrategy(gpus),
        #             project_name='test_'+name_,
        #             directory=out_dir_path+name_,
        #             # directory=r'/home/nir2/tfwork/ROCHE_ws/Feb-26-2022-00-00/out/1000/',
        #             overwrite=False
        #             )
        tuner = kt.tuners.BayesianOptimization(
            drive.legacy_search_model,
            objective=kt.Objective("val_mean_squared_error", "min"),
            max_trials=120,
            num_initial_points=5,
            alpha=1e-3,
            beta=2.8,
            # distribution_strategy=tf.distribute.MirroredStrategy(),
            # distribution_strategy=tf.distribute.MirroredStrategy(gpus),
            project_name="test_" + name_,
            directory=out_dir_path + name_,
            # directory=r'/home/nir2/tfwork/ROCHE_ws/Feb-26-2022-00-00/out/1000/',
            overwrite=True,
        )
        stop_early = EarlyStopping(monitor="val_mean_squared_error", patience=10)
        stop_nan = TerminateOnNaN()
        lr_sched = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=4,
        )
        tensorboard = TensorBoard(log_dir=out_dir_path + name_, histogram_freq=1)
        tuner.search(
            xtr,
            ytr,
            # epochs=25,
            # validation_split=0.10,
            validation_data=(xval, yval),
            callbacks=[stop_early, stop_nan],
            batch_size=16,
        )
        tuner.results_summary()
        tuner.search_space_summary(extended=True)
        best_hp_list = tuner.get_best_hyperparameters(num_trials=8)  # Get top hypers
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
            callbacks=[tensorboard, lr_sched, stop_nan],
            epochs=200,
        )  ## CHANGE increased this to 200 with change in optimizer to Adam, variable learning rate, and now using data augmentation during train
        val_hist = history.history["val_mean_squared_error"]
        # print(history.history.keys())
        # print(val_hist)
        # for i,j in list(enumerate(val_hist[75:]))[::-1]:
        #     # print(i,j)
        #     if j == min(val_hist[75:]):
        #         best_epoch=i+76
        #         break
        best_epoch = val_hist.index(min(val_hist))
        # print(best_epoch)
        if best_epoch < min_epoch_cutoff:
            logfile.write(name_ + " failed by not reaching minimum number of epochs\n")
            continue
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
            hypermodel.save(
                out_dir_path + name_ + "hpset" + str(i) + "_" + get_hps(hps) + ".h5"
            )
            # hypermodel.save(out_dir_path+name_+'hpset'+str(i)+'_partindex'+str(__k)+'.h5')
            val_result = hypermodel.evaluate(xval, yval)
            tes_result = hypermodel.evaluate(xte, yte)
            mse_result = get_mse_metrics(hypermodel, xtr, (xval, xte), ytr, (yval, yte))
            mae_result = get_mae_metrics(hypermodel, xtr, (xval, xte), ytr, (yval, yte))
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
                    + "_hyper_strings": [get_hps(f) for f in best_hp_list[1:]],
                    "val_history": val_hist,
                    "val_result": {i + 1: val_result},
                    "test_result": {i + 1: tes_result},
                    "mse_result": {i + 1: mse_result},
                    "mae_result": {i + 1: mae_result},
                    "train_loss": {i + 1: train_h},
                    "val_loss": {i + 1: val_h},
                }
                tforg.results[name_]["split"] = splits["spl" + name_]
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
            plot_results(
                outdir=out_dir_path,
                expkey=name_ + "test" + "_hp" + str(i),
                train=(ytr.ravel(), ytr_p),
                test=(yte.ravel(), yte_p),
            )
            plot_results(
                outdir=out_dir_path,
                expkey=name_ + "val" + "_hp" + str(i),
                train=(ytr.ravel(), ytr_p),
                test=(yval.ravel(), yval_p),
            )
        buffer_log = json.dumps(tforg.results[name_])
        logfile.write(buffer_log + "\n")
        logfile.flush()
        print("Completed partition " + str(__k))
        drive.get_next_part()

    with open(json_buffer_path + "final_complete_log" + date__ + ".json", "w") as g:
        g.write(json.dumps(tforg.results))


#### For reference - old model constructors

# def full_search_model(hp):
#     """
#     This will look for one hidden layer NNs, with dropouts, and an output layer with no activation function
#     It will allow changing activation functions between layers.

#     NOTE: if interested in multiclass classification, use softmax with # nodes = # classes,
#     multilabel classification use sigmoid with # nodes = number labels, and
#     use linear with regression and one node

#     """

#     # input_dimension=self.input_dim
#     model = Sequential()
#     # model.add(Input(shape=input_dimension))
#     hp_n_1 = hp.Int(
#         "nodes_1", min_value=256, max_value=6400, step=64
#     )  # 48 states possible
#     hp_n_2 = hp.Int(
#         "nodes_2", min_value=128, max_value=1280, step=24
#     )  # 48 states possible
#     hp_n_3 = hp.Int("nodes_3", min_value=8, max_value=256, step=8)
#     hp_d_1 = hp.Float("dropout1", min_value=0.0, max_value=0.95)
#     hp_d_2 = hp.Float("dropout2", min_value=0.0, max_value=0.95)
#     hp_a_1 = hp.Choice("act1", values=["relu", "softmax", "tanh", "exp"])
#     hp_a_2 = hp.Choice("act2", values=["relu", "softmax", "tanh"])
#     hp_a_3 = hp.Choice("act3", values=["relu", "softmax", "tanh"])
#     model.add(Dense(hp_n_1, activation=hp_a_1))
#     model.add(Dropout(hp_d_1))
#     model.add(Dense(hp_n_2, activation=hp_a_2))
#     model.add(Dropout(hp_d_2))
#     model.add(Dense(hp_n_3, activation=hp_a_3))
#     model.add(Dense(1), activation="linear")
#     hp_lr = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
#     model.compile(optimizer=Adam(learning_rate=hp_lr), loss="mse", metrics=["accuracy"])
#     return model


# def basic_2hl_model(X: np.array):
#     """
#     Early model framework that seemed to work reasonably well

#     Check input dimensionality to ensure that it is right

#     """
#     nodes_1 = 128
#     nodes_2 = 64
#     d1 = 0.23
#     d2 = 0.23
#     input_dimension = X.shape[1]
#     model = Sequential()
#     model.add(
#         GaussianNoise(
#             # stddev=0.05,
#             stddev=0.005,
#             # seed=1234,
#             input_shape=(input_dimension,),
#         )
#     )
#     model.add(Dense(nodes_1, activation="relu"))
#     model.add(Dropout(d1))
#     model.add(Dense(nodes_2, activation="relu"))
#     model.add(Dropout(d2))
#     model.add(Dense(1, activation="linear"))
#     model.compile(
#         optimizer=Adam(learning_rate=5e-3),
#         loss="mse",
#         metrics=["accuracy", "mse", "mae"],
#     )
#     return model
