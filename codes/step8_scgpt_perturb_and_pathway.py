#!/usr/bin/env python3
# step8_scgpt_perturb_and_pathway.py

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# scGPT
from scgpt.tasks.cell_emb import get_batch_cell_embeddings  # scGPT docs API :contentReference[oaicite:2]{index=2}
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

def load_and_filter(adata_path: str, exclude_sample: str = "GSM5329919"):
    adata = sc.read_h5ad(adata_path)
    # 如果你的 obs 里有 sample/GSM 字段，请按你的字段名改这里：
    for col in ["sample", "GSM", "orig.ident"]:
        if col in adata.obs.columns:
            adata = adata[adata.obs[col].astype(str) != exclude_sample].copy()
            break
    return adata

def ensure_counts_layer(adata):
    # scGPT通常希望 raw counts；如果你有 adata.layers["counts"] 就用它
    if "counts" in adata.layers:
        X = adata.layers["counts"]
    else:
        # 否则退化用 adata.X（你需要确认它是不是 counts）
        X = adata.X
    return X

def compute_scgpt_embeddings(adata, batch_size=64, emb_mode="cls", seed=0):
    set_seed(seed)

    # 基础预处理（scGPT提供 Preprocessor；按你数据情况可调整）
    # 这里只做最“稳”的：不改变你现有 cluster，只为 scGPT 输入做必要格式
    pp = Preprocessor(
        normalize_total=1e4,
        log1p=True,
        # 也可加 hvg 选择，但为了“最小改动”，这里直接用现有 adata
    )
    adata_pp = adata.copy()
    pp(adata_pp)

    # 直接用预训练 scGPT_human 做 embedding（zero-shot） :contentReference[oaicite:3]{index=3}
    emb = get_batch_cell_embeddings(
        adata_pp,
        cell_embedding_mode=emb_mode,   # 'cls'
        model=None, vocab=None,
        batch_size=batch_size,
        max_length=1200,
        use_batch_labels=False,
    )
    adata.obsm["X_scgpt"] = emb
    return adata

def train_classifier(adata, label_col="leiden"):
    X = adata.obsm["X_scgpt"]
    y = adata.obs[label_col].astype(str).values
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=2000,
        n_jobs=8,
        C=1.0,
    )
    clf.fit(Xs, y)
    return clf, scaler

def perturb_gene_expr(adata, gene, factor, layer="counts"):
    ad = adata.copy()
    # 取表达矩阵
    X = ad.layers[layer] if layer in ad.layers else ad.X
    # 找基因
    if gene not in ad.var_names:
        return None
    gi = np.where(ad.var_names == gene)[0][0]
    # 稀疏矩阵处理
    if hasattr(X, "tocsc"):
        X = X.tocsc(copy=True)
        X[:, gi] = X[:, gi].multiply(factor)
        X = X.tocsr()
    else:
        X = X.copy()
        X[:, gi] = X[:, gi] * factor
    if layer in ad.layers:
        ad.layers[layer] = X
    else:
        ad.X = X
    return ad

def pathway_scores_scanpy(adata, pathways: dict, use_layer=None):
    # pathways: {"HALLMARK_EMT": [genes...], ...}
    scores = {}
    for name, genes in pathways.items():
        genes_use = [g for g in genes if g in adata.var_names]
        if len(genes_use) < 5:
            continue
        sc.tl.score_genes(adata, gene_list=genes_use, score_name=name, use_raw=False)
        scores[name] = adata.obs[name].values.astype(float)
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata_h5ad", required=True)
    ap.add_argument("--label_col", default="leiden")   # 或 celltype_pred
    ap.add_argument("--genes_tsv", required=True)      # 一列 gene
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude_sample", default="GSM5329919")
    ap.add_argument("--kd_factor", type=float, default=0.1)
    ap.add_argument("--oe_factor", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    adata = load_and_filter(args.adata_h5ad, args.exclude_sample)

    # 1) scGPT embedding
    adata = compute_scgpt_embeddings(adata, batch_size=args.batch_size)

    # 2) classifier on embeddings
    clf, scaler = train_classifier(adata, args.label_col)

    # baseline probs
    X0 = scaler.transform(adata.obsm["X_scgpt"])
    classes = clf.classes_
    p0 = clf.predict_proba(X0)  # [n_cells, n_classes]
    y_true = adata.obs[args.label_col].astype(str).values
    true_idx = np.array([np.where(classes == yy)[0][0] for yy in y_true])
    p0_true = p0[np.arange(len(true_idx)), true_idx]

    # pathways（示例：你可以换成 hallmark gene sets）
    pathways = {
        "EMT": ["VIM","EPCAM","CDH1","FN1","SNAI1","SNAI2","ZEB1","ZEB2"],
        "IFNG": ["IFNG","STAT1","IRF1","CXCL9","CXCL10","CXCL11"],
        "TNFA_NFKB": ["TNF","NFKB1","RELA","ICAM1","CXCL2","IL6"],
        "HYPOXIA": ["HIF1A","VEGFA","LDHA","SLC2A1","CA9"],
        "G2M": ["MKI67","TOP2A","CDC20","CCNB1","CDK1"],
    }

    genes = pd.read_csv(args.genes_tsv, sep="\t", header=None).iloc[:,0].astype(str).tolist()

    rows_dp = []
    rows_ds = []

    # compute baseline pathway scores
    base_scores = pathway_scores_scanpy(adata.copy(), pathways)

    # 3) perturb loop
    for gene in genes:
        for mode, factor in [("KD", args.kd_factor), ("OE", args.oe_factor)]:
            ad_p = perturb_gene_expr(adata, gene, factor, layer="counts")
            if ad_p is None:
                continue
            # recompute embeddings
            ad_p = compute_scgpt_embeddings(ad_p, batch_size=args.batch_size)
            Xp = scaler.transform(ad_p.obsm["X_scgpt"])
            pp = clf.predict_proba(Xp)
            pp_true = pp[np.arange(len(true_idx)), true_idx]
            dp = pp_true - p0_true

            # aggregate by label (cluster/celltype)
            for lab in np.unique(y_true):
                m = (y_true == lab)
                rows_dp.append({
                    "gene": gene, "mode": mode, "label": lab,
                    "dp_mean": float(np.mean(dp[m])),
                    "dp_abs_mean": float(np.mean(np.abs(dp[m]))),
                    "dp_max": float(np.max(dp[m])),
                    "n_cells": int(np.sum(m)),
                })

            # pathway delta scores (mean over label)
            pert_scores = pathway_scores_scanpy(ad_p.copy(), pathways)
            for pw in pert_scores:
                ds = pert_scores[pw] - base_scores[pw]
                for lab in np.unique(y_true):
                    m = (y_true == lab)
                    rows_ds.append({
                        "gene": gene, "mode": mode, "label": lab, "pathway": pw,
                        "dscore_mean": float(np.mean(ds[m])),
                        "n_cells": int(np.sum(m)),
                    })

    df_dp = pd.DataFrame(rows_dp)
    df_ds = pd.DataFrame(rows_ds)
    df_dp.to_csv(f"{args.out_dir}/perturb_delta_prob.tsv", sep="\t", index=False)
    df_ds.to_csv(f"{args.out_dir}/perturb_delta_pathway.tsv", sep="\t", index=False)

    print("[OK] wrote perturb_delta_prob.tsv and perturb_delta_pathway.tsv")

if __name__ == "__main__":
    main()