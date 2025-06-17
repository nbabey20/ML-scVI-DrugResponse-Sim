from pathlib import Path
import gzip, warnings
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scvi, gseapy as gp
from collections import defaultdict


cwd = Path(".").resolve()
pbmc_train = cwd / "train_pbmc.h5ad"
pbmc_valid = cwd / "valid_pbmc.h5ad"
gbm_h5 = next(cwd.glob("Targeted*_filtered_feature_bc_matrix.h5"))
meta_csv = cwd / "metadata.csv"
raw_glob = "GSM*_raw_counts_*.txt.gz"

out_dir = cwd / "results"
out_dir.mkdir(exist_ok=True)
sc.settings.figdir = out_dir
warnings.filterwarnings("ignore", category=FutureWarning)

#create metadata csv for GSM identifiers
def stub_metadata() -> None:
    gsm_ids = sorted({p.name.split("_")[0] for p in cwd.glob(raw_glob)})
    pd.DataFrame({"gsm": gsm_ids, "treatment": ["control"]*len(gsm_ids)}).to_csv(meta_csv, index=False)
    raise SystemExit(f"\nCreated {meta_csv}.  Edit 'treatment' and re-run.")

#load gene expression counts and convert each file into anndata object
def load_raw_counts(meta: pd.DataFrame) -> ad.AnnData:
    adatas = []
    for f in sorted(cwd.glob(raw_glob)):
        gsm = f.name.split("_")[0]
        trt = meta.loc[meta.gsm == gsm, "treatment"].item()
        if trt == "mixed":
            continue
        df = (pd.read_csv(gzip.open(f), sep="\t", index_col=0)
                .apply(pd.to_numeric, errors="coerce")
                .dropna(how="all")
                .astype(np.float32))
        adata = ad.AnnData(df.T)
        adata.var_names_make_unique()
        adata.obs["sample"] = gsm
        adata.obs["condition"] = trt
        adata.obs["dataset"] = "TMZ"
        adata.obs_names_make_unique()
        adatas.append(adata)
    if not adatas:
        raise RuntimeError("No usable GSM files (all 'mixed'?)")
    return ad.concat(adatas, join="outer")

def qc_keep_hvgs(a: ad.AnnData) -> None:
    #filters out low quality cells
    sc.pp.filter_cells(a, min_genes=50)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, subset=True, flavor="seurat_v3")

if __name__ == "__main__":

    #pbmc benchmark
    print("Loading PBMC benchmark")
    pbmc = sc.read_h5ad(pbmc_train).concatenate(sc.read_h5ad(pbmc_valid))
    pbmc.obs["condition"] = pbmc.obs["condition"].astype("category")
    pbmc.obs["dataset"] = "PBMC"
    pbmc.obs_names_make_unique()

    #tmz dataset
    if not meta_csv.exists():
        stub_metadata()
    meta = pd.read_csv(meta_csv)
    print("Loading TMZ dataset")
    tmz = load_raw_counts(meta)
    tmz.obs["condition"] = tmz.obs["condition"].astype("category")

    #gbm baseline
    print("Loading GBM baseline")
    gbm = sc.read_10x_h5(gbm_h5)
    gbm.var_names_make_unique()
    gbm.obs["condition"] = "GBM_baseline"
    gbm.obs["dataset"] = "GBM"
    gbm.obs_names_make_unique()

    #intersect genes
    shared = set(pbmc.var_names) & set(tmz.var_names) & set(gbm.var_names)
    pbmc, tmz, gbm = [a[:, list(shared)].copy() for a in (pbmc, tmz, gbm)]
    for a in (pbmc, tmz, gbm):
        qc_keep_hvgs(a)

    #expose gbm baseline label
    dummy = ad.AnnData(
        X=np.zeros((1, pbmc.shape[1]), dtype=np.float32),
        obs=pd.DataFrame({"condition": ["GBM_baseline"], "dataset": ["dummy"]}, index=["_dummy_cell_"]),
        var=pbmc.var.copy())
    pbmc_aug = pbmc.concatenate(dummy, join="outer", batch_key="dummy")

    #pre train
    print("Pre-training scVI")
    scvi.settings.seed = 0
    scvi.model.SCVI.setup_anndata(pbmc_aug, batch_key="condition")
    vae = scvi.model.SCVI(pbmc_aug, n_latent=30)
    vae.train(max_epochs=200)

    #fine tune
    print("Fine-tuning on TMZ")
    combo = pbmc.concatenate(tmz, join="outer", batch_key="dataset")
    scvi.model.SCVI.setup_anndata(combo, batch_key="condition")
    ft_model = scvi.model.SCVI.load_query_data(combo, vae)
    ft_model.train(max_epochs=100)

    #simulate
    print("Simulating TMZ response in GBM cells")
    sim = ft_model.get_normalized_expression(gbm, transform_batch="TMZ", return_mean=True)
    gbm_sim = ad.AnnData(sim.values, obs=gbm.obs.copy(), var=gbm.var.copy())
    gbm_sim.obs["condition"] = "GBM_sim_TMZ"
    gbm_sim.obs_names_make_unique()

    #joint umap
    combo_gbm = gbm.concatenate(
        gbm_sim,
        batch_key = "sim_status",
        batch_categories=["baseline", "sim_TMZ"],
        index_unique=None)

    sc.pp.pca(combo_gbm, n_comps=30)
    sc.pp.neighbors(combo_gbm, n_pcs=30)
    sc.tl.umap(combo_gbm)
    sc.pl.umap(combo_gbm, color="sim_status", frameon=False, save="_gbm_sim_vs_base.png")
    combo_gbm.write_h5ad(out_dir / "gbm_simulated.h5ad")

    #clusters on baseline
    sc.pp.pca(gbm, n_comps=30)
    sc.pp.neighbors(gbm, n_pcs=30)
    sc.tl.leiden(gbm, resolution=0.3, key_added="cluster")
    gbm_sim.obs["cluster"] = gbm.obs["cluster"].values

    combo_gbm.obs["cluster"] = pd.Categorical(
        list(gbm.obs["cluster"]) + list(gbm_sim.obs["cluster"]),
        categories=gbm.obs["cluster"].cat.categories, ordered=True)

    #DE per cluster
    de_tables: dict[str, pd.DataFrame] = {}
    for cl in combo_gbm.obs["cluster"].cat.categories:
        merged = combo_gbm[combo_gbm.obs["cluster"] == cl].copy()
        sc.tl.rank_genes_groups(merged, groupby="sim_status",
                                groups=["sim_TMZ"], reference="baseline",
                                method="wilcoxon")
        df = sc.get.rank_genes_groups_df(merged, group="sim_TMZ")
        df.to_csv(out_dir / f"DE_cluster{cl}.csv", index=False)
        de_tables[cl] = df
    print("Differential expression tables saved in", out_dir)

    #kegg ping
    for cl, df in de_tables.items():
        up = df.query("logfoldchanges > 0.5 & pvals_adj < 0.05") \
               .head(150)["names"].tolist()
        if not up:
            continue
        try:
            res = gp.enrichr(gene_list=up,
                             gene_sets="KEGG_2021_Human",
                             organism="Human",
                             outdir=None,
                             verbose=False).results
            if not res.empty:
                top = res.iloc[0]
                print(f"Cluster {cl}: {top.Term}  (adj p={top['Adjusted P-value']:.1e})")
        except Exception as e:
            print(f"Enrichr skipped for cluster {cl}: {e}")

    print("\nDone, see the results folder")
