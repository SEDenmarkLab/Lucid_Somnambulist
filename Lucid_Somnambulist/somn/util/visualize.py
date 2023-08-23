### Model evaluation - for feature development, model performance evaluation, etc


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import pandas as pd
from glob import glob
from somn.util.project import Project
from matplotlib import cm
from matplotlib.colors import ListedColormap


#### Depreciated
# def plot_results(outdir: str, expkey: str, train: (np.array), test: (np.array)):
#     """
#     Plot model predicted vs observed as an image.
#     """
#     observed, predicted = test
#     # print(observed,predicted)
#     traino, trainp = train
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax1.set_aspect(1)
#     plt.xlim([0, 100])
#     plt.ylim([0, 100])
#     plt.plot(traino, trainp, "ko")
#     plt.title("Train")
#     plt.ylabel("Predicted")
#     plt.xlabel("Observed")
#     plt.text(90, 3, str(mean_absolute_error(traino, trainp)))
#     k, b, r, p, _ = linregress(traino, trainp)
#     plt.plot([0, 100], [0 * k + b, 100 * k + b], alpha=0.65)
#     plt.text(
#         75.0, 7.0, f"$ R^2 = {r**2:0.4f} $ \n $ k = {k:0.3f} $ \n $ p = {p:1.3e} $"
#     )
#     ax2 = fig.add_subplot(1, 2, 2)
#     ax2.set_aspect(1)
#     plt.xlim([0, 100])
#     plt.ylim([0, 100])
#     plt.plot(observed, predicted, "rp")
#     plt.title("Test")
#     plt.xlabel("Observed")
#     plt.ylabel("Predicted")
#     plt.text(90, 3, str(mean_absolute_error(observed, predicted)))
#     k, b, r, p, _ = linregress(observed, predicted)
#     plt.plot([0, 100], [0 * k + b, 100 * k + b], alpha=0.60)
#     plt.text(
#         75.0, 7.0, f"$ R^2 = {r**2:0.4f} $ \n $ k = {k:0.3f} $ \n $ p = {p:1.3e} $"
#     )
#     plt.savefig(outdir + expkey + ".png")
#     return k,b,r**2
#### Depreciated


def plot_results(
    outdir: str, expkey: str, train: (np.array), val: (np.array), test: (np.array)
):
    """
    Plot model predicted vs observed as an image.

    Also returns regression statistics as a tuple with:
    (slope, intercept, R2, p-value)
    """
    grp = [
        pd.DataFrame(k, index=["observed", "predicted"]).transpose()
        for k in (train, val, test)
    ]
    df = pd.concat(
        grp,
        axis=0,
        keys=["train", "val", "test"],
    )
    df.reset_index(inplace=True)  # Gets keys to be level_0 column
    df["partition"] = df["level_0"]
    g = sns.lmplot(
        data=df,
        x="observed",
        y="predicted",
        col="partition",
        hue="partition",
        ci=85,
        height=5,
        aspect=1,
        palette="muted",
        scatter_kws={
            "s": 15,
            "alpha": 0.50,
        },
        legend_out=False,
        truncate=True,
        robust=True,
    )
    g.set(xlim=(-5, 110), ylim=(-5, 110))
    print(g.axes)
    for ax in g.axes.flatten():
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([20, 40, 60, 80, 100])
    out_stats = {}
    for lab, tup in zip(("train", "val", "test"), (train, val, test)):
        t, p = tup
        k, b, r, p, _ = linregress(t, p)
        out_stats[lab] = (k, b, r**2, p)
    plt.savefig(f"{outdir}{expkey}_plot_tvt.png", dpi=250, transparent=True)
    plt.clf()
    return out_stats


def get_cond_label(x_: int, pos):
    labels = [
        "K2CO3/Toluene",
        "K2CO3/Dioxane",
        "K2CO3/tAmOH",
        "NaOtBu/Toluene",
        "NaOtBu/Dioxane",
        "NaOtBu/tAmOH",
        "DBU/Toluene",
        "DBU/Dioxane",
        "DBU/tAmOH",
    ]
    # print(pos)
    # print(x_)
    return labels[pos]


def get_cat_label(x_, pos, preds: pd.DataFrame):
    # print(x_,pos)
    if type(pos) == None:
        print("error")
    y_ = np.arange(1, max(preds["catalyst"].values) + 1)
    y_ = np.delete(y_, 14)
    y_ = y_.tolist()
    return str(y_[pos])


def round_z(z_, pos):
    val = np.round_(z_, decimals=0)
    val = int(val)
    return val


def get_condition_components(str_):
    spl = str_.strip().split("_")
    cat = spl[2]
    solv = spl[3]
    base = spl[4]
    return cat, solv, base


def get_components(str_):
    spl = str_.strip().split("_")
    am = spl[0]
    br = spl[1]
    cat = spl[2]
    solv = spl[3]
    base = spl[4]
    return am, br, cat, solv, base


def get_handles_by_reactants(str_, handles_):
    out = []
    for k in handles_:
        # print(k.rsplit('_',3)[0])
        # print(str_)
        if k.rsplit("_", 3)[0] == str_:
            out.append(k)
    return out


def code_solvbase(strs_: tuple):
    solv, base = strs_
    if base == "a":
        if solv == "1":
            return 2  # Dioxane, moderately polar
        elif solv == "2":
            return 3  # tAmOH, most polar
        elif solv == "3":
            return 1  # Toluene, least polar
    elif base == "b":
        if solv == "1":
            return 5  # Dioxane, moderately polar
        elif solv == "2":
            return 6  # tAmOH, most polar
        elif solv == "3":
            return 4  # Toluene, least polar
    elif base == "c":
        if solv == "1":
            return 8  # Dioxane, moderately polar
        elif solv == "2":
            return 9  # tAmOH, most polar
        elif solv == "3":
            return 7  # Toluene, least polar


def get_unique_couplings(handles: list):
    unique_couplings = set([f[0] + "_" + f[1] for f in map(get_components, handles)])
    return unique_couplings


def load_predictions(prediction_experiment=None):
    """
    Load predictions from a particular experiment

    NOTE: must be called after a somn Project has been specified
    """
    assert isinstance(prediction_experiment, str)
    # from somn.util.project import Project

    project = Project()
    preds = pd.read_csv(
        f"{project.output}/{prediction_experiment}_rawpredictions.csv",
        index_col=0,
        header=0,
    )
    preds: pd.DataFrame
    arr = np.array(preds.values)
    mean_ = arr.mean(axis=1)
    stdev_ = arr.mean(axis=1)
    preds["average"] = mean_
    preds["stdev"] = stdev_
    return preds


def visualize_predictions(query=None, prediction_experiment=None, plot_type="average"):
    """
    Method to visualize a heatmap of predictions for a specific coupling

    Pass a query as "{amine}_{bromide}", and the prediction_experiment to locate the correct predictions.

    """
    ### Some simple checks before we get started
    assert isinstance(prediction_experiment, str) & isinstance(query, str)
    assert len(query.split("_")) == 2
    assert plot_type in ["average", "stdev"]
    project = Project()
    preds = load_predictions(prediction_experiment=prediction_experiment)
    pred_handles = preds.index.to_list()
    unique_couplings = get_unique_couplings(pred_handles)
    if query not in unique_couplings:
        raise Exception(
            f"Cannot visualize predictions {query} because \
            prediction experiment {prediction_experiment} does not appear to contain\
                any predictions for {query}"
        )
    query_handles = get_handles_by_reactants(query, pred_handles)
    preds_subset = preds.loc[query_handles, ["average", "stdev"]]
    ### Check if the predictions have already been processed/saved and serialize if appropriate
    file_check = glob(f"{project.output}/processed_{query}_predictions.csv")
    if not file_check:
        preds_subset.to_csv(
            f"{project.output}/processed_{query}_predictions.csv",
            header=True,
        )
    ### Some prep - these are used to format the heatmap
    cat_ = []
    solv_ = []
    base_ = []
    solv_base_ = []

    for k in preds_subset.index:
        cat, solv, base = get_condition_components(k)
        cat_.append(int(cat))
        solv_.append(solv)
        base_.append(base)
        solv_base_.append(code_solvbase((solv, base)))
    ### Categories in dataframe for seaborn plotting engine
    preds_subset["catalyst"] = cat_
    preds_subset["solvent"] = solv_
    preds_subset["base"] = base_
    preds_subset["code"] = solv_base_
    ### Labels for x and y
    x_ = np.arange(1, max(preds_subset["code"].values) + 1)
    y_ = np.arange(1, max(preds_subset["catalyst"].values) + 1)
    y_ = np.delete(y_, 14)
    ### Getting 2D array for 3D data organized
    z_pre = preds_subset[[plot_type, "catalyst", "code"]].sort_values(
        ["catalyst", "code"], inplace=False
    )
    temp_yld = z_pre[["code", "catalyst", plot_type]]
    yields = temp_yld[plot_type].to_list()
    z_ = []
    for k in range(len(y_)):  # y is catalysts
        temp = []
        for l in range(len(x_)):  # x is conditions
            idx = 9 * k + l
            yld = yields[idx]
            temp.append(yld)
        z_.append(temp)
    Z = np.array(z_)
    X, Y = np.meshgrid(x_, y_)
    sns.set_theme(style="white")
    Z_r = np.round(Z, decimals=0)
    data = pd.DataFrame(Z_r, index=y_, columns=x_)
    N = 512
    max_frac = np.max(Z_r) / 100.0
    if (
        max_frac >= 1.00
    ):  # This ensures that if max prediction is at (or above) 100, then we will just use the normal colormap
        cmap_ = "viridis"
    else:
        cbar_top = int(max_frac * N) + 1
        # print(cbar_top)
        other_frac = 1 - max_frac
        viridis = cm.get_cmap("viridis", cbar_top)
        newcolors = viridis(np.linspace(0, 1, cbar_top))
        # upper_frac = (100.0-np.max(Z_r))/100.0
        # num = int(np.round(upper_frac*N,decimals=0)-5)
        # print(num)
        # print(num/512)
        # print(np.max(Z_r))
        # print(N*other_frac)
        newcolors = np.append(
            newcolors,
            [
                np.array([78 / 256, 76 / 256, 43 / 256, 0.60])
                for f in range(int(N * other_frac))
            ],
            axis=0,
        )
        # print(newcolors)
        cmap_ = ListedColormap(newcolors)

    ax = sns.heatmap(
        data=data,
        # cmap='viridis',
        cmap=cmap_,
        square=True,
        vmin=0.0,
        vmax=100,
        # vmax=500,
        # vmax = np.max(Z_r),
        # center=np.max(Z_r)/2,
        cbar=True,
        cbar_kws={
            "shrink": 0.75,
            "extend": "max",
            "extendrect": True,
            "ticks": [0, 15, 30, 50, 75, 100],
            # "ticks": [100, 200, 300, 400, 500],  # For metric
            "spacing": "proportional",
            "label": "Predicted Yield",
            "location": "right",
            # 'extendfrac':(100.0-np.max(Z_r))/np.max(Z_r)
        },
        # center=50.0
    )
    # ax.yaxis.set_major_formatter(get_cat_label)
    ax.xaxis.set_major_formatter(get_cond_label)
    ax.tick_params("x", labelsize="small")
    plt.setp(
        ax.xaxis.get_majorticklabels(), rotation=-60, ha="left", rotation_mode="anchor"
    )
    plt.setp(
        ax.yaxis.get_majorticklabels(), rotation=0, ha="center", rotation_mode="anchor"
    )
    plt.subplots_adjust(bottom=0.25)
    # plt.show()
    # plt.savefig("rewrite_heatmaps/" + sys.argv[1] + "_heat.svg")
    plt.savefig(
        f"{project.output}/{prediction_experiment}_{query}_heatmap_{plot_type}.svg"
    )
    plt.show()
    plt.clf()


if __name__ == "__main__":
    ...
