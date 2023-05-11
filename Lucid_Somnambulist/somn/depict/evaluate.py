### Model evaluation - for feature development, model performance evaluation, etc


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import pandas as pd

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
    g = sns.lmplot(
        data=df,
        x="observed",
        y="predicted",
        col="level_0",
        ci=None,
        height=5,
        aspect=1,
        palette="muted",
        scatter_kws={
            "s": 20,
            "alpha": 0.75,
        },
    )
    g.set(xlim=(-5, 120), ylim=(-5, 120))
    print(g.axes)
    for ax in g.axes.flatten():
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([20, 40, 60, 80, 100])
    out_stats = {}
    for lab, tup in zip(("train", "val", "test"), (train, val, test)):
        t, p = tup
        k, b, r, p, _ = linregress(t, p)
        out_stats[lab] = (k, b, r**2, p)
    plt.savefig(f"{outdir}{expkey}_plot_trvate.png", dpi=250, transparent=True)
    return out_stats
