"""
Figure 12:
"""

import seaborn as sns
import pandas as pd
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((5, 4), (1, 1))
    
    subplotLabel(ax)

    plot_toppfun(ax[0])


    return f


def plot_toppfun(ax):
    """Plot GSEA results"""
    df = pd.read_csv("pf2/data/topp_fun_cmp9.csv", dtype=str)
    df = df.drop(columns=["ID", "Verbose ID"])
    category = df["Category"].to_numpy().astype(str)

    df = df.drop(columns=["Category"])
    df["Process"] = category
    df = df.iloc[:1000, :]
    df["Total Genes"] = df.iloc[:, 2:-1].astype(int).sum(axis=1).to_numpy()
    df = df.loc[df.loc[:, "Process"] == "GO: Biological Process"]
    df["pValue"] = df["pValue"].astype(float)

    sns.scatterplot(
        data=df.iloc[:10, :], x="pValue", y="Name", hue="Total Genes", ax=ax
    )
    ax.set(xscale="log")