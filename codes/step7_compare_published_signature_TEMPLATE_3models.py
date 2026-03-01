#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended Step7: reproduce and compare multiple published BLCA prognostic signatures.

Based on your original script `step7_compare_published_signature_TEMPLATE.py`, with:
  - Added 2 extra published IF>5 (recent) signatures (Jiang 2022; Sun 2022)
  - Added `--models` to run multiple models in one call
  - Added `--missing_gene_policy {error,fill0}`
  - Optional survival column overrides (TCGA + GSE)

Outputs:
  - If running ONE model (default): keeps the original output layout in `--out_dir/`
  - If running MULTIPLE models (`--models`): writes each model to `--out_dir/<model_key>/`
    and also produces `--out_dir/summary_metrics.tsv` and `.json`
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_gene import GeneAliasResolver, apply_resolver_to_index
from utils_survival import c_index, logrank_test, km_estimator

# Built-in published models (gene -> coef)
BUILTIN_MODELS = {
    "zou2021_aging_6gene": {
        "paper": "Zou et al., 2021, Aging (Albany NY) - 6-gene signature",
        "coefs": {
            "EMP1":    0.194173005532207,
            "RASGRP4": 0.300051365964018,
            "HSPA1L": -0.25824698561501,
            "AHNAK":   0.426348983855879,
            "SLC1A6":  0.29902852359222,
            "PRSS8":   0.29565935174874,
        }
    },
    "jiang2022_fimmu_6gene_tam": {
        "paper": "Jiang et al., 2022, Frontiers in Immunology - 6-gene TAM signature",
        "coefs": {
            "TBXAS1":  0.3703,
            "GYPC":    0.1809,
            "GAB3":   -1.2990,
            "HPGDS":   0.1220,
            "ADORA3":  0.3304,
            "FOLR2":   0.0236,
        }
    },
    "sun2022_oxidmed_5gene_lipid": {
        "paper": "Sun et al., 2022, Oxidative Medicine and Cellular Longevity - 5-gene lipid metabolism signature",
        "coefs": {
            "TM4SF1":  0.197,
            "KCNK5":  -0.373,
            "FASN":    0.408,
            "IMPDH1":  0.342,
            "KCNJ15": -0.171,
        }
    },
}

def read_expr(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, sep="\t", compression="gzip", index_col=0)
    return pd.read_csv(path, sep="\t", index_col=0)

def read_gse_expr(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        df = pd.read_csv(path, sep="\t", compression="gzip")
    else:
        df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={df.columns[0]: "ID"}).set_index("ID")
    return df

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def plot_km(out_png: str, time, event, group01, title: str):
    fig = plt.figure(figsize=(6.5, 5.2))
    ax = fig.add_subplot(111)
    for g, lab in [(0, "Low risk"), (1, "High risk")]:
        m = (group01 == g)
        t, s = km_estimator(time[m], event[m])
        ax.step(t, s, where="post", label=lab)
    chi2, p = logrank_test(time, event, group01)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{title}\nlog-rank p={p:.3g}, chi2={chi2:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return chi2, p

def parse_event_gse(ev_raw):
    ev_raw = pd.to_numeric(ev_raw, errors="coerce").values
    uniq = sorted(pd.unique(ev_raw[~np.isnan(ev_raw)]).tolist())
    if set(uniq) <= {0.0, 1.0}:
        return ev_raw.astype(int)
    if set(uniq) <= {1.0, 2.0}:
        return (ev_raw == 2.0).astype(int)
    return (ev_raw > 0).astype(int)

def make_groups(risk: np.ndarray, cutoff: float) -> np.ndarray:
    return (risk >= cutoff).astype(int)

def safe_cutoff_auto(risk_tcga: np.ndarray, risk_gse: np.ndarray):
    cutoff_tcga = float(np.median(risk_tcga))
    grp_gse = make_groups(risk_gse, cutoff_tcga)
    if len(np.unique(grp_gse)) < 2:
        cutoff_gse = float(np.median(risk_gse))
        note = "auto: tcga_median failed on GSE (single-group) -> gse cohort_median"
        return cutoff_tcga, cutoff_gse, note
    note = "auto: tcga_median used for both cohorts"
    return cutoff_tcga, cutoff_tcga, note

def run_one_model(model_key: str,
                  tcga: pd.DataFrame, surv: pd.DataFrame,
                  gse: pd.DataFrame, clin: pd.DataFrame,
                  time_col_tcga: str, event_col_tcga: str,
                  time_month_col_gse: str, event_col_gse: str,
                  args, out_dir: str, fig_dir: str):

    info = BUILTIN_MODELS[model_key]
    paper = info["paper"]
    coefs = info["coefs"]
    genes = list(coefs.keys())
    beta = np.array([coefs[g] for g in genes], float)

    # arrays
    time = surv[time_col_tcga].astype(float).values
    event = surv[event_col_tcga].astype(int).values
    time_gse = clin[time_month_col_gse].astype(float).values * 30.4375
    event_gse = parse_event_gse(clin[event_col_gse])

    # gene presence
    miss_tcga = [g for g in genes if g not in tcga.index]
    miss_gse  = [g for g in genes if g not in gse.index]

    if (miss_tcga or miss_gse) and args.missing_gene_policy == "error":
        raise RuntimeError(f"[{model_key}] Gene missing. TCGA missing={miss_tcga}, GSE missing={miss_gse}. "
                           f"Try --alias_tsv or --missing_gene_policy fill0.")

    if args.missing_gene_policy == "fill0":
        for g in miss_tcga:
            tcga.loc[g] = 0.0
        for g in miss_gse:
            gse.loc[g] = 0.0

    # risk
    risk_tcga = tcga.loc[genes].T.values @ beta
    risk_gse  = gse.loc[genes].T.values @ beta

    c_tcga = c_index(time, event, risk_tcga)
    c_gse  = c_index(time_gse, event_gse, risk_gse)

    # cutoff/groups
    cutoff_note = args.cutoff
    if args.cutoff == "tcga_median":
        cutoff_tcga = float(np.median(risk_tcga))
        cutoff_gse  = cutoff_tcga
    elif args.cutoff == "cohort_median":
        cutoff_tcga = float(np.median(risk_tcga))
        cutoff_gse  = float(np.median(risk_gse))
    else:
        cutoff_tcga, cutoff_gse, cutoff_note = safe_cutoff_auto(risk_tcga, risk_gse)

    grp_tcga = make_groups(risk_tcga, cutoff_tcga)
    grp_gse  = make_groups(risk_gse,  cutoff_gse)

    chi2_t, p_t = plot_km(os.path.join(fig_dir, "km_tcga.png"), time, event, grp_tcga, f"{model_key} on TCGA")
    chi2_g, p_g = plot_km(os.path.join(fig_dir, "km_gse13507.png"), time_gse, event_gse, grp_gse, f"{model_key} on GSE13507")

    # risk tables
    out_tcga = surv[["patient_id", time_col_tcga, event_col_tcga]].copy()
    out_tcga = out_tcga.rename(columns={time_col_tcga: "time_days", event_col_tcga: "event"})
    out_tcga["risk"] = risk_tcga
    out_tcga["group_high"] = grp_tcga
    out_tcga.to_csv(os.path.join(out_dir, "published_tcga_risk.tsv"), sep="\t", index=False)

    out_gse = clin[["GSM", time_month_col_gse, event_col_gse]].copy()
    out_gse = out_gse.rename(columns={time_month_col_gse: "survivalMonth", event_col_gse: "overall survival"})
    out_gse["time_days"] = time_gse
    out_gse["event"] = event_gse
    out_gse["risk"] = risk_gse
    out_gse["group_high"] = grp_gse
    out_gse.to_csv(os.path.join(out_dir, "published_gse13507_risk.tsv"), sep="\t", index=False)

    metrics = {
        "model": model_key,
        "paper": paper,
        "n_genes": int(len(genes)),
        "missing_tcga": miss_tcga,
        "missing_gse": miss_gse,
        "missing_policy": args.missing_gene_policy,
        "cutoff_rule": cutoff_note,
        "cutoff_tcga": float(cutoff_tcga),
        "cutoff_gse": float(cutoff_gse),
        "tcga": {"cindex": float(c_tcga), "logrank_p": float(p_t), "logrank_chi2": float(chi2_t), "n": int(len(time))},
        "gse13507": {"cindex": float(c_gse), "logrank_p": float(p_g), "logrank_chi2": float(chi2_g), "n": int(len(time_gse))},
        "genes": coefs,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as w:
        json.dump(metrics, w, indent=2, ensure_ascii=False)

    print("[MODEL]", model_key, "| cutoff_rule =", cutoff_note)
    print("[TCGA] C-index=", c_tcga, "logrank_p=", p_t)
    print("[GSE ] C-index=", c_gse,  "logrank_p=", p_g)
    print("[OK] wrote:", out_dir)

    return {
        "model": model_key,
        "paper": paper,
        "n_genes": int(len(genes)),
        "tcga_cindex": float(c_tcga),
        "gse_cindex": float(c_gse),
        "tcga_logrank_p": float(p_t),
        "gse_logrank_p": float(p_g),
        "cutoff_rule": cutoff_note,
        "missing_tcga": ",".join(miss_tcga) if miss_tcga else "",
        "missing_gse": ",".join(miss_gse) if miss_gse else "",
        "missing_policy": args.missing_gene_policy,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga_expr_gz", required=True)
    ap.add_argument("--tcga_survival_tsv", required=True)
    ap.add_argument("--gse_expr_tsv", required=True)
    ap.add_argument("--gse_clin_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alias_tsv", default=None)

    ap.add_argument("--model", default="zou2021_aging_6gene",
                    choices=list(BUILTIN_MODELS.keys()),
                    help="Run a single published model.")
    ap.add_argument("--models", default=None,
                    help='Run multiple models: "all" or comma-separated keys. If provided, overrides --model.')

    ap.add_argument("--cutoff", default="auto",
                    choices=["auto", "tcga_median", "cohort_median"],
                    help="grouping rule for KM/logrank (C-index does not depend on it)")
    ap.add_argument("--missing_gene_policy", default="error",
                    choices=["error", "fill0"],
                    help="How to handle missing genes: error or fill0.")

    ap.add_argument("--tcga_time_col", default=None,
                    help="TCGA time column (default auto: time_days else time)")
    ap.add_argument("--tcga_event_col", default=None,
                    help="TCGA event column (default: event)")
    ap.add_argument("--gse_time_month_col", default="survivalMonth",
                    help="GSE time column in months (default: survivalMonth)")
    ap.add_argument("--gse_event_col", default="overall survival",
                    help="GSE event column (default: overall survival)")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    if args.models:
        if args.models.strip().lower() == "all":
            model_keys = list(BUILTIN_MODELS.keys())
        else:
            model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        multi_mode = True
    else:
        model_keys = [args.model]
        multi_mode = False

    resolver = None
    if args.alias_tsv:
        resolver = GeneAliasResolver.from_tsv(args.alias_tsv)

    # TCGA
    tcga = read_expr(args.tcga_expr_gz)
    tcga.index = tcga.index.astype(str)
    tcga = apply_resolver_to_index(tcga, resolver)

    surv_all = pd.read_csv(args.tcga_survival_tsv, sep="\t").copy()
    if "patient_id" not in surv_all.columns:
        raise RuntimeError(f"TCGA survival file must contain 'patient_id'. Got: {list(surv_all.columns)}")

    time_col_tcga = args.tcga_time_col
    event_col_tcga = args.tcga_event_col
    if time_col_tcga is None:
        if "time_days" in surv_all.columns:
            time_col_tcga = "time_days"
        elif "time" in surv_all.columns:
            time_col_tcga = "time"
        else:
            raise RuntimeError(f"Cannot detect TCGA time column. Columns: {list(surv_all.columns)}")
    if event_col_tcga is None:
        if "event" in surv_all.columns:
            event_col_tcga = "event"
        else:
            raise RuntimeError(f"Cannot detect TCGA event column. Columns: {list(surv_all.columns)}")

    surv = surv_all.dropna(subset=["patient_id", time_col_tcga, event_col_tcga]).copy()
    surv["patient_id"] = surv["patient_id"].astype(str)
    common_pat = [p for p in surv["patient_id"] if p in tcga.columns]
    surv = surv[surv["patient_id"].isin(common_pat)].copy()
    tcga = tcga[common_pat]

    # GSE
    gse = read_gse_expr(args.gse_expr_tsv)
    gse.index = gse.index.astype(str)
    gse = apply_resolver_to_index(gse, resolver)

    clin_all = pd.read_csv(args.gse_clin_tsv, sep="\t").copy()
    if "GSM" not in clin_all.columns:
        raise RuntimeError(f"GSE clinical file must contain 'GSM'. Got: {list(clin_all.columns)}")
    clin_all["GSM"] = clin_all["GSM"].astype(str)
    if args.gse_time_month_col not in clin_all.columns:
        raise RuntimeError(f"GSE time column '{args.gse_time_month_col}' not found. Columns: {list(clin_all.columns)}")
    if args.gse_event_col not in clin_all.columns:
        raise RuntimeError(f"GSE event column '{args.gse_event_col}' not found. Columns: {list(clin_all.columns)}")

    clin = clin_all.dropna(subset=[args.gse_time_month_col]).copy()
    gsm_cols = [c for c in gse.columns if str(c).startswith("GSM") and c in set(clin["GSM"])]
    clin = clin[clin["GSM"].isin(gsm_cols)].copy().sort_values("GSM")
    gse = gse[clin["GSM"].tolist()]

    summary_rows = []

    for mk in model_keys:
        if multi_mode:
            out_m = os.path.join(args.out_dir, mk)
            ensure_dir(out_m)
            fig_dir = os.path.join(out_m, "figures")
            ensure_dir(fig_dir)
        else:
            out_m = args.out_dir
            fig_dir = os.path.join(args.out_dir, "figures")
            ensure_dir(fig_dir)

        row = run_one_model(
            model_key=mk,
            tcga=tcga,
            surv=surv,
            gse=gse,
            clin=clin,
            time_col_tcga=time_col_tcga,
            event_col_tcga=event_col_tcga,
            time_month_col_gse=args.gse_time_month_col,
            event_col_gse=args.gse_event_col,
            args=args,
            out_dir=out_m,
            fig_dir=fig_dir,
        )
        summary_rows.append(row)

    if multi_mode:
        df = pd.DataFrame(summary_rows)
        df.to_csv(os.path.join(args.out_dir, "summary_metrics.tsv"), sep="\t", index=False)
        with open(os.path.join(args.out_dir, "summary_metrics.json"), "w", encoding="utf-8") as w:
            json.dump(summary_rows, w, indent=2, ensure_ascii=False)
        print("[SUMMARY] wrote:", os.path.join(args.out_dir, "summary_metrics.tsv"))

if __name__ == "__main__":
    main()
