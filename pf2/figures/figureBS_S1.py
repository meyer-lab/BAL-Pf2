"""Figure BS_S1: Description"""

import anndata
from .common import subplotLabel, getSetup
from ..data_import import import_meta, convert_to_patients
from ..tensor import correct_conditions


def makeFigure():
    ax, f = getSetup((8, 12), (2, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    X.uns["Pf2_A"] = correct_conditions(X)  
    condition_factors = X.uns["Pf2_A"]

    meta = import_meta(drop_duplicates=False)
    meta = meta.set_index("sample_id", drop=True)
    meta = meta.loc[~meta.index.duplicated(), :]

    sample_conversions = convert_to_patients(X, sample=True)
    meta = meta.loc[meta.index.isin(sample_conversions)]
    meta = meta.reindex(sample_conversions).dropna(axis=0, how="all")
    condition_factors = condition_factors[
        sample_conversions.isin(meta.index),
        :
    ]

    return f
