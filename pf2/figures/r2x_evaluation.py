from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from pf2.tensor import (build_tensor, get_variance_explained, import_data,
                        run_parafac2)

REPO_PATH = dirname(dirname(abspath(__file__)))


def main():
    data = import_data(log_transform=False, sum_one=False, normalize=True)
    tensor, patients = build_tensor(data)
    ranks = np.arange(1, 41)
    r2x = pd.Series(0, dtype=float, index=ranks)

    for rank in tqdm(ranks):
        pf2 = run_parafac2(tensor, rank)
        r2x.loc[rank] = get_variance_explained(pf2, tensor)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300, constrained_layout=True)
    ax.plot(r2x.index, r2x)
    ax.grid(True)

    ax.set_ylabel("R2X")
    ax.set_xlabel("Rank")

    plt.savefig(join(REPO_PATH, "output", "r2x_v_rank.png"))
    plt.show()


if __name__ == "__main__":
    main()
