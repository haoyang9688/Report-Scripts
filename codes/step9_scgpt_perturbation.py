#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_scgpt_perturbation_v2.py
- Fixes KeyError: adata.var['id_in_vocab'] by creating vocab mapping from a pretrained scGPT checkpoint folder.
- Uses raw counts (adata.layers['counts'] preferred) for scGPT embedding + perturbation.
- Loads scGPT model ONCE, then reuses it for baseline and all perturbations (fast).

Required model_dir contents (from scGPT pretrained checkpoint folder):
  - vocab.json
  - args.json
  - best_model.pt

Outputs (in out_dir):
  - perturb_delta_prob.tsv
  - perturb_delta_pathway.tsv
  - Fig_sc1_deltaP_heatmap_KD.png
  - Fig_sc1_deltaP_heatmap_OE.png
  - Fig_sc2_pathway_heatmap_KD_<label>.png
  - Fig_sc2_pathway_heatmap_OE_<label>.png
"""

import argparse, os, json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch

# scGPT
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.tasks.cell_emb import get_batch_cell_embeddings


def ensure_dir(d): os.makedirs(d, exist_ok=True)

def exclude_sample(adata, sample_id="GSM5329919"):
    for col in ["GSM", "sample", "orig.ident", "batch", "Sample"]:
        if col in adata.obs.columns:
            m = adata.obs[col].astype(str) != sample_id
            if m.sum() < adata.n_obs:
                adata = adata[m].copy()
            return adata
    return adata

def pick_counts_matrix(adata):
    # Prefer raw counts layer if available
    if "counts" in adata.layers:
        return "counts"
    # Or use adata.raw if it exists and looks non-negative
    if adata.raw is not None:
        try:
            X = adata.raw.X
            x0 = X[:100, :100].A if hasattr(X, "A") else np.asarray(X[:100, :100])
            if np.nanmin(x0) >= 0:
                adata.X = adata.raw.X.copy()
                return None
        except Exception:
            pass
    raise RuntimeError(
        "No raw counts found. Please rerun preprocessing to save raw counts into adata.layers['counts'].\n"
        "Tip: in your step8 preprocessing, before normalization/log1p, add:\n"
        "    adata.layers['counts'] = adata.X.copy()\n"
    )

def load_scgpt_model(model_dir, device="cuda"):
    model_dir = os.path.abspath(model_dir)
    vocab_file = os.path.join(model_dir, "vocab.json")
    args_file  = os.path.join(model_dir, "args.json")
    model_file = os.path.join(model_dir, "best_model.pt")
    for f in [vocab_file, args_file, model_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file in model_dir: {f}")

    with open(args_file, "r") as f:
        model_configs = json.load(f)

    vocab = GeneVocab.from_file(vocab_file)
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[pad_token])

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("[WARN] CUDA not available; using CPU.")
    else:
        device = torch.device("cpu")

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=True,
        fast_transformer_backend="flash",
        pre_norm=False,
    )

    state = torch.load(model_file, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model_dict = model.state_dict()
        state = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(state)
        model.load_state_dict(model_dict)

    model.to(device)
    model.eval()
    return model, vocab, model_configs, device

def map_genes_to_vocab_and_filter(adata, vocab, gene_col="index"):
    if gene_col == "index":
        adata.var["index"] = adata.var.index.astype(str)
        gene_col = "index"
    else:
        if gene_col not in adata.var.columns:
            raise RuntimeError(f"gene_col={gene_col} not found in adata.var columns.")

    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var[gene_col].astype(str)]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"].values, dtype=int)
    print(f"[INFO] match {np.sum(gene_ids_in_vocab>=0)}/{len(gene_ids_in_vocab)} genes in scGPT vocab (size={len(vocab)}).")
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

    genes = adata.var[gene_col].astype(str).tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    return adata, gene_ids

def train_multinomial_lr(emb, labels):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(emb)
    clf = LogisticRegression(multi_class="multinomial", max_iter=3000, n_jobs=8, C=1.0)
    clf.fit(Xs, labels)
    return clf, scaler

def perturb_gene_matrix(adata, gene, factor):
    ad = adata.copy()
    X = ad.X
    if gene not in ad.var_names:
        return None
    gi = np.where(ad.var_names == gene)[0][0]
    if hasattr(X, "tocsc"):
        X = X.tocsc(copy=True)
        X[:, gi] = X[:, gi].multiply(factor)
        X = X.tocsr()
    else:
        X = np.array(X, copy=True)
        X[:, gi] = X[:, gi] * factor
    ad.X = X
    return ad

def pathway_scores_from_counts(adata_counts, pathways):
    tmp = adata_counts.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    scores = {}
    for name, genes in pathways.items():
        genes_use = [g for g in genes if g in tmp.var_names]
        if len(genes_use) < 5:
            continue
        sc.tl.score_genes(tmp, gene_list=genes_use, score_name=name, use_raw=False)
        scores[name] = tmp.obs[name].values.astype(float)
    return scores

def heatmap(df, index_col, col_col, val_col, out_png, title):
    pivot = df.pivot_table(index=index_col, columns=col_col, values=val_col, aggfunc="mean")
    fig = plt.figure(figsize=(10, max(4, 0.25 * pivot.shape[0])))
    ax = fig.add_subplot(111)
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_h5ad", required=True)
    ap.add_argument("--keep_genes_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_col", default="leiden")
    ap.add_argument("--exclude_gsm", default="GSM5329919")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--gene_col", default="index")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--kd_factor", type=float, default=0.1)
    ap.add_argument("--oe_factor", type=float, default=2.0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    adata = sc.read_h5ad(args.in_h5ad)
    adata = exclude_sample(adata, args.exclude_gsm)

    if args.label_col not in adata.obs.columns:
        raise RuntimeError(f"label_col={args.label_col} not found in adata.obs.")

    layer_name = pick_counts_matrix(adata)
    if layer_name:
        adata.X = adata.layers[layer_name].copy()

    model, vocab, model_configs, device = load_scgpt_model(args.model_dir, device=args.device)
    adata, gene_ids = map_genes_to_vocab_and_filter(adata, vocab, gene_col=args.gene_col)

    df_genes = pd.read_csv(args.keep_genes_tsv, sep="\t")
    gene_list = df_genes["gene"].astype(str).tolist()
    gene_list = [g for g in gene_list if g in adata.var_names]
    print(f"[INFO] bulk key genes present after vocab filtering: {len(gene_list)}/{len(df_genes)}")

    emb0 = get_batch_cell_embeddings(
        adata,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=1200,
        batch_size=args.batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )

    labels = adata.obs[args.label_col].astype(str).values
    clf, scaler = train_multinomial_lr(emb0, labels)
    X0 = scaler.transform(emb0)
    classes = clf.classes_
    p0 = clf.predict_proba(X0)
    true_idx = np.array([np.where(classes == yy)[0][0] for yy in labels])
    p0_true = p0[np.arange(len(true_idx)), true_idx]

    pathways = {
        "EMT": ["VIM","EPCAM","CDH1","FN1","SNAI1","SNAI2","ZEB1","ZEB2"],
        "IFNG": ["IFNG","STAT1","IRF1","CXCL9","CXCL10","CXCL11"],
        "TNFA_NFKB": ["TNF","NFKB1","RELA","ICAM1","CXCL2","IL6"],
        "HYPOXIA": ["HIF1A","VEGFA","LDHA","SLC2A1","CA9"],
        "G2M": ["MKI67","TOP2A","CDC20","CCNB1","CDK1"],
    }
    base_scores = pathway_scores_from_counts(adata, pathways)

    rows_dp, rows_ds = [], []
    for gene in gene_list:
        for mode, factor in [("KD", args.kd_factor), ("OE", args.oe_factor)]:
            ad_p = perturb_gene_matrix(adata, gene, factor)
            if ad_p is None:
                continue

            embp = get_batch_cell_embeddings(
                ad_p,
                cell_embedding_mode="cls",
                model=model,
                vocab=vocab,
                max_length=1200,
                batch_size=args.batch_size,
                model_configs=model_configs,
                gene_ids=gene_ids,
                use_batch_labels=False,
            )
            Xp = scaler.transform(embp)
            pp = clf.predict_proba(Xp)
            pp_true = pp[np.arange(len(true_idx)), true_idx]
            dp = pp_true - p0_true

            for lab in np.unique(labels):
                m = (labels == lab)
                rows_dp.append({
                    "gene": gene, "mode": mode, "label": lab,
                    "dp_mean": float(np.mean(dp[m])),
                    "dp_abs_mean": float(np.mean(np.abs(dp[m]))),
                    "dp_max": float(np.max(dp[m])),
                    "n_cells": int(np.sum(m)),
                })

            pert_scores = pathway_scores_from_counts(ad_p, pathways)
            for pw in pert_scores:
                ds = pert_scores[pw] - base_scores[pw]
                for lab in np.unique(labels):
                    m = (labels == lab)
                    rows_ds.append({
                        "gene": gene, "mode": mode, "label": lab, "pathway": pw,
                        "dscore_mean": float(np.mean(ds[m])),
                        "n_cells": int(np.sum(m)),
                    })

    df_dp = pd.DataFrame(rows_dp)
    df_ds = pd.DataFrame(rows_ds)
    df_dp.to_csv(os.path.join(args.out_dir, "perturb_delta_prob.tsv"), sep="\t", index=False)
    df_ds.to_csv(os.path.join(args.out_dir, "perturb_delta_pathway.tsv"), sep="\t", index=False)

    heatmap(df_dp[df_dp["mode"]=="KD"], "gene", "label", "dp_abs_mean",
            os.path.join(args.out_dir, "Fig_sc1_deltaP_heatmap_KD.png"),
            "KD: |Δp| by gene × label")
    heatmap(df_dp[df_dp["mode"]=="OE"], "gene", "label", "dp_abs_mean",
            os.path.join(args.out_dir, "Fig_sc1_deltaP_heatmap_OE.png"),
            "OE: |Δp| by gene × label")

    sens = df_dp.groupby(["mode","label"])["dp_abs_mean"].mean().reset_index()
    best_lab = sens.sort_values("dp_abs_mean", ascending=False).groupby("mode").head(1)
    best_map = dict(zip(best_lab["mode"], best_lab["label"]))
    print("[INFO] Most sensitive label (by mean |Δp|):", best_map)

    for mode in ["KD","OE"]:
        lab = best_map.get(mode, None)
        if lab is None:
            continue
        sub = df_ds[(df_ds["mode"]==mode) & (df_ds["label"]==lab)].copy()
        heatmap(sub, "gene", "pathway", "dscore_mean",
                os.path.join(args.out_dir, f"Fig_sc2_pathway_heatmap_{mode}_{lab}.png"),
                f"{mode}: Δpathway score on most sensitive label = {lab}")

    print("[OK] wrote outputs to:", args.out_dir)

if __name__ == "__main__":
    main()
