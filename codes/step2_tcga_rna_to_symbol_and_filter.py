#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step2_tcga_rna_to_symbol_and_filter.py

Input:
  - tcga_blca_counts.tsv.gz  (gene_id x samples)
  - download_dir + metadata_json (to locate an augmented_star_gene_counts.tsv and extract gene_id->gene_name)

Output:
  - tcga_blca_counts.symbol.tsv.gz   (gene_symbol x samples)
  - tcga_blca_logcpm.symbol.tsv.gz   (gene_symbol x samples)
  - tcga_gene_id_to_symbol.tsv       (gene_id -> gene_symbol mapping used)
  - tcga_blca_sample_info.tsv        (copied)
"""

import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def read_metadata_cart(metadata_json: str) -> pd.DataFrame:
    with open(metadata_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "data" in obj:
        obj = obj["data"]
    rows = []
    for it in obj:
        file_id = it.get("file_id") or it.get("id") or it.get("file_uuid")
        file_name = it.get("file_name") or it.get("filename")
        rows.append({"file_id": file_id, "file_name": file_name})
    df = pd.DataFrame(rows).dropna(subset=["file_id"]).drop_duplicates(subset=["file_id"])
    df["file_id"] = df["file_id"].astype(str)
    df["file_name"] = df["file_name"].astype(str)
    return df


def find_any_augmented_counts_file(download_dir: str, file_id: str) -> Optional[str]:
    """
    Try locate:
      download_dir/file_id/*.rna_seq.augmented_star_gene_counts.tsv
    """
    d = Path(download_dir) / file_id
    if not d.exists():
        return None
    cand = list(d.glob("*.rna_seq.augmented_star_gene_counts.tsv"))
    if len(cand) > 0:
        return str(cand[0])
    # fallback: any tsv in the folder
    cand2 = list(d.glob("*.tsv"))
    for p in cand2:
        if "augmented_star_gene_counts" in p.name:
            return str(p)
    return None


def build_gene_map_from_one_file(path: str) -> pd.DataFrame:
    """
    Parse augmented_star_gene_counts.tsv, extract gene_id -> gene_name.

    File often begins with:
      # gene-model: GENCODE v36
    so we use comment="#".
    """
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        engine="python",
        usecols=["gene_id", "gene_name"],
    )
    df["gene_id"] = df["gene_id"].astype(str).str.strip().str.replace(r"\.\d+$", "", regex=True)
    df["gene_name"] = df["gene_name"].astype(str).str.strip()

    # remove STAR summary rows if any slipped in (usually not in df because those rows have empty gene_name)
    df = df[~df["gene_id"].str.startswith("N_")]
    df = df[df["gene_id"].str.startswith("ENSG")]

    # empty gene_name -> keep gene_id as fallback later
    df = df.drop_duplicates(subset=["gene_id"], keep="first")
    return df


def build_gene_map(download_dir: str, metadata_json: str, target_gene_ids: pd.Index, max_try: int = 30) -> pd.DataFrame:
    meta = read_metadata_cart(metadata_json)

    tried = 0
    for fid in meta["file_id"].tolist():
        fp = find_any_augmented_counts_file(download_dir, fid)
        if not fp:
            continue
        tried += 1
        try:
            mp = build_gene_map_from_one_file(fp)
        except Exception:
            continue

        # check coverage
        covered = mp["gene_id"].isin(target_gene_ids).sum()
        if covered > 0:
            print(f"[INFO] Using gene map from: {fp}")
            print(f"[INFO] Map size={len(mp)}, covered_in_matrix={covered}/{len(target_gene_ids)}")
            return mp

        if tried >= max_try:
            break

    raise RuntimeError("Failed to build gene_id->gene_name mapping from downloaded augmented counts files.")


def read_counts_gz(path: str) -> pd.DataFrame:
    # gene_id in first column
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    # ensure gene_id no version
    df.index = df.index.astype(str).str.replace(r"\.\d+$", "", regex=True)
    # keep numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int64)
    return df


def make_symbol_matrix(
    counts: pd.DataFrame,
    gene_map: pd.DataFrame,
    collapse: str = "sum",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      counts_symbol: index=gene_symbol
      map_used: dataframe with gene_id, gene_symbol (aligned to original genes that appeared)
    """
    d = dict(zip(gene_map["gene_id"], gene_map["gene_name"]))

    gene_ids = counts.index.astype(str)
    symbols = []
    for g in gene_ids:
        s = d.get(g, "")
        if s is None:
            s = ""
        s = str(s).strip()
        if s == "" or s.lower() in {"nan", "na", "none"}:
            s = g  # fallback
        symbols.append(s)

    map_used = pd.DataFrame({"gene_id": gene_ids, "gene_symbol": symbols})
    counts2 = counts.copy()
    counts2.index = pd.Index(symbols)

    if collapse == "sum":
        counts2 = counts2.groupby(counts2.index).sum()
    elif collapse == "max":
        counts2 = counts2.groupby(counts2.index).max()
    elif collapse == "mean":
        # mean on counts is uncommon, but provided if user wants
        counts2 = counts2.groupby(counts2.index).mean().round().astype(np.int64)
    else:
        raise ValueError(f"Unknown collapse={collapse}")

    return counts2, map_used


def filter_low_expression(counts: pd.DataFrame, min_cpm: float, min_frac_samples: float) -> pd.DataFrame:
    if min_cpm <= 0 or min_frac_samples <= 0:
        return counts

    n = counts.shape[1]
    need = int(math.ceil(min_frac_samples * n))
    lib = counts.sum(axis=0).replace(0, np.nan)
    cpm = counts.div(lib, axis=1) * 1e6
    ok = (cpm >= min_cpm).sum(axis=1) >= need
    kept = int(ok.sum())
    print(f"[INFO] CPM filter: min_cpm={min_cpm}, min_frac={min_frac_samples} => keep {kept}/{counts.shape[0]} genes")
    return counts.loc[ok].copy()


def logcpm_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=0).replace(0, np.nan)
    cpm = counts.div(lib, axis=1) * 1e6
    return np.log2(cpm + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga_counts_gz", required=True, help="tcga_blca_counts.tsv.gz")
    ap.add_argument("--download_dir", required=True, help="GDC RNA download dir (contains file_id subdirs)")
    ap.add_argument("--metadata_json", required=True, help="metadata.cart.*.json used for RNA download")
    ap.add_argument("--sample_info_tsv", required=True, help="tcga_blca_sample_info.tsv (will be copied)")
    ap.add_argument("--out_dir", required=True, help="output dir")
    ap.add_argument("--collapse_dup_symbols", default="sum", choices=["sum", "max", "mean"])
    ap.add_argument("--min_cpm", type=float, default=1.0, help="low expression filter threshold (CPM)")
    ap.add_argument("--min_frac_samples", type=float, default=0.1, help="fraction of samples that must pass min_cpm")
    ap.add_argument("--keep_symbol_regex", default="", help="optional regex: keep symbols matching this regex")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Reading counts:", args.tcga_counts_gz)
    counts = read_counts_gz(args.tcga_counts_gz)
    print(f"[INFO] counts shape: {counts.shape} (genes x samples)")

    print("[INFO] Building gene_id -> gene_symbol map from augmented counts files ...")
    gene_map = build_gene_map(
        download_dir=args.download_dir,
        metadata_json=args.metadata_json,
        target_gene_ids=counts.index,
    )

    counts_sym, map_used = make_symbol_matrix(
        counts=counts,
        gene_map=gene_map,
        collapse=args.collapse_dup_symbols,
    )
    print(f"[INFO] After mapping+collapse: {counts_sym.shape}")

    # optional keep regex
    if args.keep_symbol_regex.strip():
        rx = args.keep_symbol_regex.strip()
        before = counts_sym.shape[0]
        counts_sym = counts_sym[counts_sym.index.to_series().str.contains(rx, regex=True)].copy()
        print(f"[INFO] keep_symbol_regex={rx}: {before} -> {counts_sym.shape[0]} genes")

    # low expression filter (on symbol-collapsed counts)
    counts_sym = filter_low_expression(counts_sym, args.min_cpm, args.min_frac_samples)

    # logcpm
    logcpm_sym = logcpm_from_counts(counts_sym)

    # write outputs
    out_counts = os.path.join(args.out_dir, "tcga_blca_counts.symbol.tsv.gz")
    out_logcpm = os.path.join(args.out_dir, "tcga_blca_logcpm.symbol.tsv.gz")
    out_map = os.path.join(args.out_dir, "tcga_gene_id_to_symbol.tsv")
    out_si = os.path.join(args.out_dir, "tcga_blca_sample_info.tsv")

    with gzip.open(out_counts, "wt", encoding="utf-8") as f:
        counts_sym.to_csv(f, sep="\t", index=True)
    with gzip.open(out_logcpm, "wt", encoding="utf-8") as f:
        logcpm_sym.to_csv(f, sep="\t", index=True)

    map_used.to_csv(out_map, sep="\t", index=False)

    # copy sample_info (no change needed)
    si = pd.read_csv(args.sample_info_tsv, sep="\t")
    si.to_csv(out_si, sep="\t", index=False)

    print("[OK] wrote:", out_counts, "shape=", counts_sym.shape)
    print("[OK] wrote:", out_logcpm, "shape=", logcpm_sym.shape)
    print("[OK] wrote:", out_map, "rows=", len(map_used))
    print("[OK] wrote:", out_si, "rows=", len(si))


if __name__ == "__main__":
    main()