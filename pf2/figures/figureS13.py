"""Figure S12d"""

import anndata
from .common import getSetup
from ..data_import import meta_raw_df, bal_combine_bo_covid
from ..correlation import meta_groupings
import pandas as pd
import numpy as np
import gseapy as gp
from gseapy.plot import gseaplot2
from gseapy.plot import gseaplot
import matplotlib.pyplot as plt



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # ax, f = getSetup((20, 20), (5, 5))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    # X = X[:10000, :]
    
    df = pd.DataFrame([])
    df["Gene"] = X.var.index
    df["Rank"] = X.varm["Pf2_C"][:, 2]
    df = df.sort_values("Rank").reset_index(drop=True)
    print(df)
    pre_res = gp.prerank(rnk=df, gene_sets="GO_Biological_Process_2021", seed=0)
    
    out = []

    for term in list(pre_res.results):
        out.append([term,
                pre_res.results[term]['fdr'],
                pre_res.results[term]['es'],
                pre_res.results[term]['nes'],
                pre_res.results[term]['pval']])

    out_df = pd.DataFrame(out, columns = ['Term', 'fdr', 'es', 'nes', 'pval']).sort_values(by=["nes", "es"], ascending=False).reset_index(drop = True)
    print(df)
    print(out_df)
    term_to_plot = out_df['Term'][0]
    
    # hits = pre_res.results[term_to_plot]['hits']
    # res_values = pre_res.results[term_to_plot]['RES']


    hits = pre_res.results[term_to_plot]['hits']  # or the correct key for hit indices
    res_values = pre_res.results[term_to_plot]['RES']  # or the correct key for running enrichment scores
    nes = pre_res.results[term_to_plot]['nes']
    pval = pre_res.results[term_to_plot]['pval']
    fdr = pre_res.results[term_to_plot]['fdr']

    gplot = gseaplot(
    term=term_to_plot,
    hits=hits,
    nes=nes,
    pval=pval,
    fdr=fdr,
    RES=res_values,
    rank_metric=pre_res.ranking,  # This will show the ranked metric in a separate plot
    color="#388E3C",  # You can change the color if desired
    ofname="output/figureS13a.svg"
)



    # # Now call gseaplot2 with the correct parameters
    # gplot = gseaplot2(
    # terms=[term_to_plot],  # List of terms (just one in this case)
    # hits=[hits],           # List of hit indices
    # RESs=[res_values],     # List of RES values
    # rank_metric=pre_res.ranking,  # The ranking metric for the colorbar
    # ofname="output/figureS13a.svg"
    # )   

