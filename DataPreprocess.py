import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from matplotlib.pyplot import rc_context
from umap import UMAP
import csv
# adata = sc.datasets.pbmc3k()
# print(adata)
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc.logging.print_header()

adata=sc.read('data/cortex.h5ad')
adata.raw = adata
print(adata)
# data=pd.DataFrame(adata.X)
# print(data)

# sc.pl.highest_expr_genes(adata, n_top=20, )
#
#
# adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
# sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'],
#                            percent_top=None, log1p=False, inplace=True)
#
# sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
#              jitter=0.4, multi_panel=True)
#
# sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
# sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# adata = adata[adata.obs.n_genes_by_counts < 2500, :]
# adata = adata[adata.obs.pct_counts_mt < 5, :]
# 1.Total-count normalize
sc.pp.normalize_total(adata, target_sum=1e4)
print(adata)
# 2.Logarithmize the data
# adata.raw = adata
# print(pd.DataFrame(adata.X))
sc.pp.log1p(adata)
print(adata)
# print(pd.DataFrame(adata.X))

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# sc.pl.highly_variable_genes(adata)
# adata = adata[:, adata.var.highly_variable]

# 3.Scale each gene to unit variance
sc.pp.scale(adata, max_value=10)
print(adata)
# 4.PCA
sc.tl.pca(adata, svd_solver='arpack')
print(adata)
# sc.pl.pca(adata, color=['label', 'label2'])
sc.pl.pca_variance_ratio(adata, log=True)
# 5.Computing the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
print(adata)
# 6.Embedding the neighborhood graph
# sc.tl.umap(adata)
# sc.pl.umap(adata)
# sc.tl.paga(adata)
# sc.pl.paga(adata)  # remove `plot=False` if you want to see the coarse-grained graph
# sc.tl.umap(adata, init_pos='paga')
# 7.Clustering the neighborhood graph
sc.tl.leiden(adata)
print(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['label', 'label2', 'leiden'])
# 8.Finding marker genes
print(adata)
# sc.settings.verbosity = 2  # reduce the verbosity
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

print(adata)
# data = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10)
# print(data)
# sc.pl.violin(adata, ['label', 'label2'], groupby='leiden')
# sc.pl.dotplot(adata, marker_genes, groupby='leiden')
with rc_context({'figure.figsize': (3, 3)}):
    sc.pl.umap(adata, color=['label'],
               s=50, frameon=True, ncols=4, vmax='p99')

# 通过切片查看观测值和变量
print(adata.obsm.tolist())


# results_file = 'data/cortex3.h5ad'
# adata.write('data/cortex3.h5ad')
# adata.write_csvs(results_file[:-5], skip_data=False)
