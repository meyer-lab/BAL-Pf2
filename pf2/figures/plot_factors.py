import pickle
from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pf2.data_import import import_data
from pf2.tensor import build_tensor, run_parafac2

OPTIMAL_RANK = 4
REPO_PATH = dirname(abspath(__file__))


def main():
    data = import_data()
    tensor, patients = build_tensor(data)
    pf2 = run_parafac2(tensor, rank=OPTIMAL_RANK)
    with open(f"output/rank_{OPTIMAL_RANK}.pkl", "wb") as file:
        pickle.dump(pf2, file)

    factors = {}
    dims = ["Patient", "Cell State", "Gene"]
    for factor, dim in zip(pf2.factors, dims):
        factors[dim] = pd.DataFrame(
            factor / abs(factor).max(axis=0), columns=np.arange(pf2.rank) + 1
        )
        factors[dim].to_csv(f"{dim}_factor.csv")

    fig, axs = plt.subplots(
        1, len(factors), figsize=(8, 4), constrained_layout=True, dpi=300
    )
    axs = axs.flatten()
    for ax, dim in zip(axs, factors.keys()):
        factor = factors[dim]
        sns.heatmap(factor, vmin=-1, vmax=1, cmap="coolwarm", cbar=ax == axs[-1], ax=ax)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(dim)

    plt.savefig(join(REPO_PATH, "output", "factor_heatmaps.png"))
    plt.show()


if __name__ == "__main__":
    main()
