from sklearn.metrics import roc_curve, auc
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import KernelPCA
from config import RESULT_PATH


def plot_multiclass_roc(
    y_test,
    y_pred,
    n_classes,
    figsize,
    name,
    classes,
    combination,
):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_test_dummies = pd.get_dummies(y_test, drop_first=False).to_numpy()
    
    sel_classes = classes

    for i, cname in zip(range(n_classes), sel_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        print(cname, y_test_dummies[:, i].sum())

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_dummies.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name}", y=1.05, loc="left")

    for i, cname in zip(range(n_classes), sel_classes):
        ax.plot(
            fpr[i], tpr[i], label=f"{cname} ({np.round(roc_auc[i], 2)})", linewidth=2
        )

    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ({0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    ax.grid(alpha=0.4)
    sns.despine()
    plt.legend(loc=(1.05, 0.05), title="ROC Curves (AUC)")
    plt.savefig(
        os.path.join(RESULT_PATH, f"ROC_{combination}_{name}.tiff"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
    plt.close()


def plot_heatmap(
    normalized_array,
    classes,
    conf_matrix,
    name,
    Y,
    combination,
):
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        data=normalized_array,
        xticklabels=classes,
        yticklabels=classes,
        annot=conf_matrix,
        cmap="Blues",
        fmt="d",
        cbar=False,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{name}\n(n = {len(Y.index)})", y=1.05)
    plt.xticks(rotation=45, ha="right")
    plt.savefig(
        os.path.join(
            RESULT_PATH,
            f"conf_matrix_{combination}_{name}.tiff",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_pca(
    X,
    Y,
    selected_classes,
    name,
    combination,
    kernel_type="rbf",
    gamma_type=1,
    alpha_type=0.1,
):
    pca = KernelPCA(
        n_components=None,
        kernel=kernel_type,
        gamma=gamma_type,
        fit_inverse_transform=True,
        alpha=alpha_type,
    )
    components = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(5, 5))

    sns.scatterplot(
        x=components[:, 0], y=components[:, 1], hue=Y, hue_order=selected_classes, ax=ax,
    )
    ax.set_xlabel("Principal Component 0")
    ax.set_ylabel("Principal Component 1")
    ax.set_title(name, y=1.05)
    plt.legend(loc=(1.05, 0.0))
    plt.savefig(
        os.path.join(RESULT_PATH, f"PCA_{combination}_{name}.tiff"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_distribution(Y, name, combination):
    fig, ax = plt.subplots(figsize=(5, 5))

    Y.value_counts(sort=True, normalize=False).plot(
        ax=ax,
        kind="bar",
        color="#2d7dbb",
    )
    ax.set_ylabel("Number of Metastases")
    ax.set_xlabel("Primary")
    ax.set_title(f"{name} \n(n = {len(Y.index)})")
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.bar_label(ax.containers[0], label_type="edge")
    plt.savefig(
        os.path.join(RESULT_PATH, f"label_distribution_{combination}_{name}.tiff"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()