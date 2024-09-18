# """Figure 2: R2X Curve"""

# import gc

# import numpy as np
# import pandas as pd

# from pf2.data_import import convert_to_patients, import_data, import_meta
from pf2.figures.common import getSetup
# from pf2.predict import predict_mortality
# from pf2.tensor import pf2


def makeFigure():
    #     meta = import_meta()
    #     data = import_data()
    #     conversions = convert_to_patients(data)

    axs, fig = getSetup((6, 6), (2, 1))

    #     ranks = np.arange(5, 65, 5)
    #     r2xs = pd.Series(0, dtype=float, index=ranks)
    #     accuracies = pd.Series(0, dtype=float, index=ranks)
    #     for rank in ranks:
    #         fac, r2x = pf2(data, rank, do_embedding=False)
    #         patient_factor = pd.DataFrame(
    #             fac.uns["Pf2_A"],
    #             index=conversions,
    #             columns=np.arange(fac.uns["Pf2_A"].shape[1]) + 1,
    #         )
    #         if meta.shape[0] != patient_factor.shape[0]:
    #             meta = meta.loc[patient_factor.index, :]

    #         acc, _ = predict_mortality(patient_factor, meta)
    #         r2xs.loc[rank] = r2x
    #         accuracies.loc[rank] = acc
    #         r2xs.to_csv("/home/jchin/BAL-Pf2/output/r2x_v_rank_no_ig.csv")
    #         accuracies.to_csv("/home/jchin/BAL-Pf2/output/acc_v_rank_no_ig.csv")

    #         gc.collect()

    #     # R2X Plots

    #     axs[0].plot(r2xs.index.astype(int), r2xs)
    #     axs[0].set_xticks(r2xs.index.astype(int))
    #     axs[0].grid(True)

    #     axs[0].set_ylabel("R2X")
    #     axs[0].set_xlabel("Rank")

    #     # Accuracy Plots

    #     axs[1].plot(accuracies.index, accuracies)
    #     axs[1].set_xticks(accuracies.index.astype(int))
    #     axs[1].grid(True)

    #     axs[1].set_ylabel("Accuracy")
    #     axs[1].set_xlabel("Rank")

    return fig
