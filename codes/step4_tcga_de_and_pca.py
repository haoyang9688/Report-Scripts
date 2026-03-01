#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


def read_logcpm(path_gz: str) -> pd.DataFrame:
    return pd.read_csv(path_gz, sep="\t", compression="gzip", index_col=0)


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _plot_volcano(de: pd.DataFrame, out_png: str, fdr_thr: float = 0.05, fc_thr: float = 1.0):
    x = de["logFC_tumor_minus_normal"].values
    y = -np.log10(np.clip(de["fdr"].values, 1e-300, 1.0))
    sig = (de["fdr"].values <= fdr_thr) & (np.abs(x) >= fc_thr)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=6, alpha=0.5)
    ax.scatter(x[sig], y[sig], s=8, alpha=0.8)
    ax.axhline(-np.log10(fdr_thr), linestyle="--")
    ax.axvline(-fc_thr, linestyle="--")
    ax.axvline(fc_thr, linestyle="--")
    ax.set_xlabel("logFC (tumor - normal) on logCPM")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("Volcano plot (Tumor vs Normal)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_pca(pca_df: pd.DataFrame, out_png: str):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    types = pca_df["sample_type"].astype(str).unique().tolist()
    for t in types:
        sub = pca_df[pca_df["sample_type"].astype(str) == t]
        ax.scatter(sub["PC1"], sub["PC2"], s=14, alpha=0.7, label=t)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA (top variable genes)")
    ax.legend(markerscale=1.2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_heatmap_top(de: pd.DataFrame, X: pd.DataFrame, sample_ids: list, sample_type: pd.Series,
                      out_png: str, top_k: int = 50):
    genes = de["gene"].head(top_k).tolist()
    mat = X.loc[genes, sample_ids].copy().fillna(0.0)

    mu = mat.mean(axis=1)
    sd = mat.std(axis=1).replace(0, 1.0)
    z = ((mat.T - mu) / sd).T

    order = np.argsort(sample_type.values.astype(str))
    z = z.iloc[:, order]

    fig = plt.figure(figsize=(max(8, len(sample_ids) * 0.06), max(6, top_k * 0.12)))
    ax = fig.add_subplot(111)
    im = ax.imshow(z.values, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=6)
    ax.set_xticks([])
    ax.set_title(f"Top {top_k} DE genes (z-score)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_umap(M: np.ndarray, labels: np.ndarray, out_png: str, seed: int = 0):
    try:
        import umap  # type: ignore
    except Exception as e:
        raise RuntimeError("UMAP requested but umap-learn is not installed. pip install umap-learn") from e

    reducer = umap.UMAP(n_components=2, random_state=seed)
    emb = reducer.fit_transform(M)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    uniq = pd.unique(labels.astype(str)).tolist()
    for u in uniq:
        idx = labels.astype(str) == u
        ax.scatter(emb[idx, 0], emb[idx, 1], s=14, alpha=0.7, label=u)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP (top variable genes)")
    ax.legend(markerscale=1.2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logcpm_symbol_gz", required=True)
    ap.add_argument("--sample_info_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_n", type=int, default=500)
    ap.add_argument("--top_var_genes", type=int, default=2000)
    ap.add_argument("--heatmap_topk", type=int, default=50)
    ap.add_argument("--do_umap", action="store_true")
    ap.add_argument("--umap_seed", type=int, default=0)
    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    fig_dir = os.path.join(args.out_dir, "figures")
    _ensure_dir(fig_dir)

    X = read_logcpm(args.logcpm_symbol_gz)  # genes x samples
    si = pd.read_csv(args.sample_info_tsv, sep="\t")
    for c in ["sample_id", "sample_type"]:
        if c not in si.columns:
            raise RuntimeError(f"sample_info_tsv must contain column: {c}")

    si = si[si["sample_id"].astype(str).isin(X.columns)].copy()
    si["sample_id"] = si["sample_id"].astype(str)

    tumor = si.loc[si["sample_type"] == "Primary Tumor", "sample_id"].tolist()
    normal = si.loc[si["sample_type"] == "Solid Tissue Normal", "sample_id"].tolist()
    if len(tumor) == 0 or len(normal) == 0:
        raise RuntimeError(f"Need both tumor and normal. Got tumor={len(tumor)}, normal={len(normal)}")

    Xt = X[tumor]
    Xn = X[normal]
    m_t = Xt.mean(axis=1)
    m_n = Xn.mean(axis=1)
    logFC = (m_t - m_n)

    tstat, pvals = stats.ttest_ind(Xt.values.T, Xn.values.T, equal_var=False, nan_policy="omit")
    if np.ndim(pvals) != 1 or len(pvals) != X.shape[0]:
        pvals = np.array([
            stats.ttest_ind(Xt.loc[g].values, Xn.loc[g].values, equal_var=False).pvalue
            for g in X.index
        ])
    fdr = multipletests(pvals, method="fdr_bh")[1]

    de = pd.DataFrame({
        "gene": X.index,
        "logFC_tumor_minus_normal": logFC.values,
        "pval": pvals,
        "fdr": fdr,
        "mean_logcpm_tumor": m_t.values,
        "mean_logcpm_normal": m_n.values,
        "n_tumor": len(tumor),
        "n_normal": len(normal),
    }).sort_values(["fdr", "pval"]).reset_index(drop=True)

    out_de = os.path.join(args.out_dir, "tcga_blca_DE_tumor_vs_normal.tsv")
    de.to_csv(out_de, sep="\t", index=False)
    print("[OK] wrote:", out_de, "rows=", len(de))

    top_genes = de["gene"].head(args.top_n).tolist()
    out_top = os.path.join(args.out_dir, f"tcga_blca_top{args.top_n}_DE_genes.txt")
    with open(out_top, "w", encoding="utf-8") as w:
        for g in top_genes:
            w.write(g + "\n")
    print("[OK] wrote:", out_top)

    sample_ids = si["sample_id"].tolist()
    var = X[sample_ids].var(axis=1).sort_values(ascending=False)
    top_var_genes = var.head(args.top_var_genes).index
    M = X.loc[top_var_genes, sample_ids].T.fillna(0.0).values

    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(M)
    pca_df = pd.DataFrame({
        "sample_id": sample_ids,
        "sample_type": si["sample_type"].astype(str).values,
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "explained_var_PC1": float(pca.explained_variance_ratio_[0]),
        "explained_var_PC2": float(pca.explained_variance_ratio_[1]),
    })
    out_pca = os.path.join(args.out_dir, "tcga_blca_PCA_samples.tsv")
    pca_df.to_csv(out_pca, sep="\t", index=False)
    print("[OK] wrote:", out_pca)

    _plot_volcano(de, os.path.join(fig_dir, "volcano.png"))
    _plot_pca(pca_df, os.path.join(fig_dir, "pca.png"))
    _plot_heatmap_top(de, X, sample_ids, si["sample_type"].astype(str),
                      os.path.join(fig_dir, "heatmap_top50.png"), top_k=args.heatmap_topk)
    print("[OK] wrote figures in:", fig_dir)

    if args.do_umap:
        _plot_umap(M, si["sample_type"].astype(str).values, os.path.join(fig_dir, "umap.png"), seed=args.umap_seed)
        print("[OK] wrote:", os.path.join(fig_dir, "umap.png"))


if __name__ == "__main__":
    main()