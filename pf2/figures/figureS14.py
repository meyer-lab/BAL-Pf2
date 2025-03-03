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
import seaborn as sns



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((20, 20), (5, 5))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    # X = X[:10000, :]
    
    df = pd.DataFrame([])
    df["Gene"] = X.var.index
    df["Rank"] = X.varm["Pf2_C"][:, 2]
    df = df.sort_values("Rank").reset_index(drop=True)
    print(df)
    pre_res = gp.prerank(rnk=df, gene_sets="KEGG_2021_Human", seed=0)
    
    gsea_results = pre_res.res2d
    
    # gsea_results['FDR q-val'] = pd.to_numeric(gsea_results['FDR q-val'], errors='coerce')
    # print(gsea_results['FDR q-val'])
    # gsea_results['FDR q-val'] = gsea_results['FDR q-val'].fillna(1) 


    gsea_results['-log10(Qvalue)'] = -np.log10(gsea_results['FDR q-val']+1e-5)
    gsea_results['Rich Factor'] = gsea_results['ES'] / gsea_results['NES']

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=gsea_results.head(20), x='Rich Factor', y='Term', size='-log10(Qvalue)', hue='NES', palette='viridis', ax=ax[0])
    ax[0].set_title('Top 20 Pathway Enrichment')
    ax[0].set_xlabel('Rich Factor')
    ax[0].set_ylabel('Pathway Name')


#     term=term_to_plot,
#     hits=hits,
#     nes=nes,
#     pval=pval,
#     fdr=fdr,
#     RES=res_values,
#     rank_metric=pre_res.ranking,  # This will show the ranked metric in a separate plot
#     color="#388E3C",  # You can change the color if desired
#     ofname="output/figureS13a.svg"
# )



    # # Now call gseaplot2 with the correct parameters
    # gplot = gseaplot2(
    # terms=[term_to_plot],  # List of terms (just one in this case)
    # hits=[hits],           # List of hit indices
    # RESs=[res_values],     # List of RES values
    # rank_metric=pre_res.ranking,  # The ranking metric for the colorbar
    # ofname="output/figureS13a.svg"
    # )   

    return f