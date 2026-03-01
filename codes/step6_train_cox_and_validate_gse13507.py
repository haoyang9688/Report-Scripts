#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_train_cox_and_validate_gse13507_v2_multimodal_distill.py

What this adds on top of your existing step6 ridge-Cox script:
1) Optional MULTIMODAL teacher on TCGA: RNA + DNA methylation (mid-fusion gated network) trained with Cox loss.
2) Optional RNA-only student distilled from teacher (Cox loss + distillation loss).
3) Keeps the SAME key outputs used by step7:
   - tcga_patient_risk.tsv
   - gse13507_risk.tsv
   plus figures and a model card.

Why this helps you "beat published RNA-only signatures" on external GSE13507:
- You can use methylation to learn a stronger teacher on TCGA (where methylation exists),
  then distill that signal into an RNA-only student that still generalizes to GSE.

Input assumptions:
- TCGA RNA expr: genes x patients (TSV/TSV.GZ), logCPM already ok
- TCGA methylation: features x patients (TSV/TSV.GZ). Features can be gene-level or CpG-level.
  (If CpG-level, you should already have filtered/processed it; we will still do top-variance topK.)
- GSE RNA expr: geneSymbol x GSM (same as your current step6)
- Survival tables: same columns as current step6.

Model choices:
- --model ridge               : your original ridge Cox baseline (kept here)
- --model student_distill     : train teacher (RNA+meth) then train RNA-only student with distillation; evaluate student
- --model teacher_only        : train/evaluate teacher on TCGA only (no external unless GSE has methylation, which it usually doesn't)

Hyperparameter search:
- For ridge: your existing k-fold grid (max_genes, l2)
- For distill: optional random search over lr/weight_decay/dropout/latent_dim/alpha using a TCGA train/val split.

"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from utils_gene import GeneAliasResolver, apply_resolver_to_index
from utils_survival import (
    c_index,
    logrank_test,
    km_estimator,
    cox_ph_fit_ridge,
    cox_univariate_screen,
    standardize_train_apply,
    kfold_indices,
)

# ---- torch is only required for multimodal/student models ----
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    torch = None
    nn = None
    optim = None


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _read_expr(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, sep="\t", compression="gzip", index_col=0)
    return pd.read_csv(path, sep="\t", index_col=0)


def _read_gse_expr(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        df = pd.read_csv(path, sep="\t", compression="gzip")
    else:
        df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={df.columns[0]: "ID"}).set_index("ID")
    return df


def _parse_event_gse(series: pd.Series) -> np.ndarray:
    ev_raw = pd.to_numeric(series, errors="coerce").values
    uniq = sorted(pd.unique(ev_raw[~np.isnan(ev_raw)]).tolist())
    if set(uniq) <= {0.0, 1.0}:
        return ev_raw.astype(int)
    if set(uniq) <= {1.0, 2.0}:
        return (ev_raw == 2.0).astype(int)  # assume 2=dead
    return (ev_raw > 0).astype(int)


def _plot_km(out_png: str, time: np.ndarray, event: np.ndarray, group01: np.ndarray, title: str):
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


def _load_candidate_genes(path: str, resolver: GeneAliasResolver = None) -> list:
    with open(path, "r", encoding="utf-8") as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    if resolver is not None:
        genes = [resolver.resolve(g) for g in genes]
    genes = list(dict.fromkeys([g for g in genes if g]))
    return genes


# ---------------------- Cox loss (torch) ----------------------
def _cox_ph_loss_torch(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Negative log partial likelihood (Breslow), full-batch.
    risk: (n,) higher => higher hazard (worse survival)
    """
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order].float()
    log_cum = torch.logcumsumexp(r, dim=0)
    loss = -torch.sum((r - log_cum) * e) / (torch.sum(e) + 1e-8)
    return loss


# ---------------------- Models ----------------------
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        h = max(64, latent_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusionCox(nn.Module):
    """
    Mid-fusion gated network:
      z_rna = enc_rna(x_rna)
      z_m   = enc_meth(x_meth)
      g = sigmoid(W [z_rna; z_m])
      z = g*z_rna + (1-g)*z_m
      risk = w^T z
    """
    def __init__(self, rna_dim: int, meth_dim: int, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.enc_rna = MLPEncoder(rna_dim, latent_dim, dropout)
        self.enc_meth = MLPEncoder(meth_dim, latent_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x_rna, x_meth):
        z1 = self.enc_rna(x_rna)
        z2 = self.enc_meth(x_meth)
        g = self.gate(torch.cat([z1, z2], dim=1))
        z = g * z1 + (1.0 - g) * z2
        risk = self.head(z).squeeze(1)
        return risk


class RNAStudentCox(nn.Module):
    def __init__(self, rna_dim: int, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.enc = MLPEncoder(rna_dim, latent_dim, dropout)
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x_rna):
        z = self.enc(x_rna)
        risk = self.head(z).squeeze(1)
        return risk


# ---------------------- Helpers ----------------------
def _train_val_split(n: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val = idx[:n_val]
    tr = idx[n_val:]
    return tr, val


def _standardize_np_train_apply(Xtr: np.ndarray, Xte: np.ndarray | None):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xtr_z = (Xtr - mu) / sd
    if Xte is None:
        return Xtr_z, None, mu, sd
    return Xtr_z, (Xte - mu) / sd, mu, sd


def _top_variance_features(df: pd.DataFrame, topk: int) -> list:
    if topk <= 0 or topk >= df.shape[0]:
        return df.index.tolist()
    v = df.var(axis=1).sort_values(ascending=False)
    return v.head(topk).index.tolist()


def _fit_teacher_and_student(
    X_rna: np.ndarray,
    X_meth: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    seed: int,
    latent_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    alpha_distill: float,
    epochs: int,
    patience: int,
    device: str,
    do_student: bool = True,
):
    if torch is None:
        raise RuntimeError("PyTorch is required for --model teacher_only / student_distill, but torch import failed.")

    n = len(time)
    tr, va = _train_val_split(n, val_frac=0.2, seed=seed)

    # tensors
    Xr_tr = torch.tensor(X_rna[tr], dtype=torch.float32, device=device)
    Xm_tr = torch.tensor(X_meth[tr], dtype=torch.float32, device=device)
    t_tr = torch.tensor(time[tr], dtype=torch.float32, device=device)
    e_tr = torch.tensor(event[tr], dtype=torch.int64, device=device)

    Xr_va = torch.tensor(X_rna[va], dtype=torch.float32, device=device)
    Xm_va = torch.tensor(X_meth[va], dtype=torch.float32, device=device)
    t_va = torch.tensor(time[va], dtype=torch.float32, device=device)
    e_va = torch.tensor(event[va], dtype=torch.int64, device=device)

    teacher = GatedFusionCox(X_rna.shape[1], X_meth.shape[1], latent_dim=latent_dim, dropout=dropout).to(device)
    opt = optim.AdamW(teacher.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_c = -1.0
    bad = 0

    for ep in range(1, epochs + 1):
        teacher.train()
        opt.zero_grad()
        risk = teacher(Xr_tr, Xm_tr)
        loss = _cox_ph_loss_torch(risk, t_tr, e_tr)
        loss.backward()
        opt.step()

        teacher.eval()
        with torch.no_grad():
            rva = teacher(Xr_va, Xm_va).detach().cpu().numpy()
        cva = float(c_index(time[va], event[va], rva))

        if cva > best_val_c + 1e-4:
            best_val_c = cva
            best_state = {k: v.detach().cpu() for k, v in teacher.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        teacher.load_state_dict(best_state)

    # teacher risks (all TCGA)
    teacher.eval()
    with torch.no_grad():
        r_teacher = teacher(
            torch.tensor(X_rna, dtype=torch.float32, device=device),
            torch.tensor(X_meth, dtype=torch.float32, device=device),
        ).detach().cpu().numpy()

    # Optional student
    student = None
    r_student = None
    best_student_val_c = None

    if do_student:
        student = RNAStudentCox(X_rna.shape[1], latent_dim=latent_dim, dropout=dropout).to(device)
        opt_s = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

        best_s_state = None
        best_s_val = -1.0
        bad = 0

        r_teacher_tr = torch.tensor(r_teacher[tr], dtype=torch.float32, device=device)
        r_teacher_va = torch.tensor(r_teacher[va], dtype=torch.float32, device=device)

        for ep in range(1, epochs + 1):
            student.train()
            opt_s.zero_grad()
            rs = student(Xr_tr)
            loss_cox = _cox_ph_loss_torch(rs, t_tr, e_tr)
            loss_dist = torch.mean((rs - r_teacher_tr) ** 2)
            loss = loss_cox + alpha_distill * loss_dist
            loss.backward()
            opt_s.step()

            student.eval()
            with torch.no_grad():
                rva = student(Xr_va).detach().cpu().numpy()
            cva = float(c_index(time[va], event[va], rva))

            if cva > best_s_val + 1e-4:
                best_s_val = cva
                best_s_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_s_state is not None:
            student.load_state_dict(best_s_state)
        best_student_val_c = best_s_val

        student.eval()
        with torch.no_grad():
            r_student = student(torch.tensor(X_rna, dtype=torch.float32, device=device)).detach().cpu().numpy()

    return {
        "teacher": teacher,
        "student": student,
        "risk_teacher_tcga": r_teacher,
        "risk_student_tcga": r_student,
        "teacher_val_cindex": float(best_val_c),
        "student_val_cindex": None if best_student_val_c is None else float(best_student_val_c),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="ridge",
                    choices=["ridge", "teacher_only", "student_distill"],
                    help="ridge: original ridge Cox; teacher_only: RNA+meth gated Cox on TCGA; student_distill: teacher then RNA-only student")

    ap.add_argument("--tcga_expr_gz", required=True)
    ap.add_argument("--tcga_survival_tsv", required=True)
    ap.add_argument("--candidate_genes", required=True)

    ap.add_argument("--gse_expr_tsv", required=True)
    ap.add_argument("--gse_clin_tsv", required=True)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alias_tsv", default=None)

    # same restriction flags as your baseline
    ap.add_argument("--restrict_to_gse_genes", action="store_true")
    ap.add_argument("--no_restrict_to_gse_genes", action="store_true")

    # baseline ridge controls
    ap.add_argument("--univ_fdr", type=float, default=0.05)
    ap.add_argument("--no_cv", action="store_true")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2_grid", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    ap.add_argument("--max_genes_grid", default="10,20,30,40,50")
    ap.add_argument("--max_genes", type=int, default=39)
    ap.add_argument("--l2", type=float, default=1e-3)

    # methylation (teacher/student)
    ap.add_argument("--tcga_meth_tsv", default=None, help="required for teacher_only/student_distill unless --teacher_risk_tsv provided")
    ap.add_argument("--meth_topk", type=int, default=5000, help="top variance methylation features to keep (0=all)")
    ap.add_argument("--teacher_risk_tsv", default=None, help="optional: precomputed TCGA teacher risk per patient_id; skip teacher training")

    # neural training / search
    ap.add_argument("--device", default="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)

    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--alpha_distill", type=float, default=1.0)

    ap.add_argument("--hp_search", action="store_true", help="random search for student_distill over a small hyperparam space")
    ap.add_argument("--n_trials", type=int, default=20)

    args = ap.parse_args()
    _ensure_dir(args.out_dir)
    fig_dir = os.path.join(args.out_dir, "figures")
    _ensure_dir(fig_dir)

    # restriction default ON
    if args.no_restrict_to_gse_genes:
        restrict_to_gse = False
    elif args.restrict_to_gse_genes:
        restrict_to_gse = True
    else:
        restrict_to_gse = True

    resolver = None
    if args.alias_tsv:
        resolver = GeneAliasResolver.from_tsv(args.alias_tsv)
        print("[INFO] Loaded alias mapping:", args.alias_tsv, "n=", len(resolver.alias_to_symbol))

    # ---------------- TCGA RNA ----------------
    tcga_expr = _read_expr(args.tcga_expr_gz)  # genes x patients
    tcga_expr.index = tcga_expr.index.astype(str)
    tcga_expr = apply_resolver_to_index(tcga_expr, resolver)

    surv = pd.read_csv(args.tcga_survival_tsv, sep="\t")
    need = {"patient_id", "time_days", "event"}
    if not need.issubset(set(surv.columns)):
        raise RuntimeError(f"tcga_survival_tsv must contain columns: {sorted(need)}")
    surv = surv.dropna(subset=["patient_id", "time_days", "event"]).copy()
    surv["patient_id"] = surv["patient_id"].astype(str)

    common_pat = [p for p in surv["patient_id"].tolist() if p in tcga_expr.columns]
    surv = surv[surv["patient_id"].isin(common_pat)].copy()
    tcga_expr = tcga_expr[common_pat]
    time = surv["time_days"].astype(float).values
    event = surv["event"].astype(int).values
    print("[INFO] TCGA aligned patients:", len(common_pat), "genes:", tcga_expr.shape[0])

    # ---------------- GSE RNA ----------------
    gse = _read_gse_expr(args.gse_expr_tsv)  # genes x GSM
    gse.index = gse.index.astype(str)
    gse = apply_resolver_to_index(gse, resolver)

    clin = pd.read_csv(args.gse_clin_tsv, sep="\t")
    for c in ["GSM", "survivalMonth", "overall survival"]:
        if c not in clin.columns:
            raise RuntimeError("gse_clin_tsv must contain columns: GSM, survivalMonth, overall survival")
    clin["GSM"] = clin["GSM"].astype(str)
    clin = clin.dropna(subset=["survivalMonth"]).copy()

    gsm_cols = [c for c in gse.columns if str(c).startswith("GSM")]
    gsm_cols = [c for c in gsm_cols if c in set(clin["GSM"])]
    clin = clin[clin["GSM"].isin(gsm_cols)].copy().sort_values("GSM")
    gse = gse[clin["GSM"].tolist()]
    time_gse_days = clin["survivalMonth"].astype(float).values * 30.4375
    event_gse = _parse_event_gse(clin["overall survival"])
    print("[INFO] GSE aligned samples:", gse.shape[1], "genes:", gse.shape[0])

    # ---------------- Candidate genes (RNA) ----------------
    cand = _load_candidate_genes(args.candidate_genes, resolver=resolver)
    cand = [g for g in cand if g in tcga_expr.index]
    if restrict_to_gse:
        cand = [g for g in cand if g in gse.index]
        print("[INFO] restrict_to_gse_genes=ON, candidates in both:", len(cand))
    else:
        print("[INFO] restrict_to_gse_genes=OFF, candidates in TCGA:", len(cand))

    if len(cand) < 30:
        raise RuntimeError(f"Too few usable candidate genes: {len(cand)}. "
                           f"Check gene symbols / alias mapping or disable restriction.")

    Xtcga = tcga_expr.loc[cand].T
    vv = Xtcga.var(axis=0)
    Xtcga = Xtcga[vv[vv > 1e-8].index.tolist()]
    cand = Xtcga.columns.tolist()
    print("[INFO] usable candidates after variance filter:", len(cand))

    # ---------------- Ridge baseline branch ----------------
    if args.model == "ridge":
        # Keep your original logic (CV grid over max_genes, l2)
        do_cv = (not args.no_cv)
        best_max_genes = int(args.max_genes)
        best_l2 = float(args.l2)
        best = None

        if do_cv:
            l2_grid = [float(x) for x in args.l2_grid.split(",") if x.strip()]
            max_genes_grid = [int(x) for x in args.max_genes_grid.split(",") if x.strip()]
            folds = kfold_indices(len(time), k=args.kfold, seed=args.seed)
            rows = []

            for max_g in max_genes_grid:
                for l2 in l2_grid:
                    cs = []
                    used = []
                    for tr, te in folds:
                        Xtr_full = Xtcga.iloc[tr].copy()
                        Xte_full = Xtcga.iloc[te].copy()
                        ttr, etr = time[tr], event[tr]
                        tte, ete = time[te], event[te]

                        Xtr_z, Xte_z, _, _ = standardize_train_apply(Xtr_full, Xte_full)

                        uni = cox_univariate_screen(Xtr_z.values, list(Xtr_z.columns), ttr, etr)
                        uni["fdr"] = multipletests(uni["pval"].values, method="fdr_bh")[1]
                        uni = uni.sort_values(["fdr", "pval"]).reset_index(drop=True)

                        pass_genes = uni.loc[uni["fdr"] <= args.univ_fdr, "gene"].tolist()
                        if len(pass_genes) == 0:
                            pass_genes = uni["gene"].head(min(100, len(uni))).tolist()

                        genes_fold = pass_genes[:max_g]
                        used.append(len(genes_fold))

                        Xtr = Xtr_z[genes_fold]
                        Xte = Xte_z[genes_fold]

                        beta, _ = cox_ph_fit_ridge(Xtr.values, ttr, etr, l2=float(l2))
                        risk = Xte.values @ beta
                        cs.append(c_index(tte, ete, risk))

                    rows.append({
                        "max_genes": int(max_g),
                        "l2": float(l2),
                        "mean_cindex": float(np.mean(cs)),
                        "std_cindex": float(np.std(cs)),
                        "mean_genes_used_per_fold": float(np.mean(used)),
                        "kfold": int(args.kfold),
                        "seed": int(args.seed),
                        "univ_fdr": float(args.univ_fdr),
                    })

            cv_df = pd.DataFrame(rows).sort_values("mean_cindex", ascending=False).reset_index(drop=True)
            cv_path = os.path.join(args.out_dir, "cv_results.tsv")
            cv_df.to_csv(cv_path, sep="\t", index=False)
            print("[OK] wrote:", cv_path)

            best = cv_df.iloc[0].to_dict()
            best_max_genes = int(best["max_genes"])
            best_l2 = float(best["l2"])
            print("[INFO] best CV:", best)

        # Final fit on full TCGA
        Xtcga_z, _, mu_tcga, sd_tcga = standardize_train_apply(Xtcga, None)

        uni_full = cox_univariate_screen(Xtcga_z.values, list(Xtcga_z.columns), time, event)
        uni_full["fdr"] = multipletests(uni_full["pval"].values, method="fdr_bh")[1]
        uni_full = uni_full.sort_values(["fdr", "pval"]).reset_index(drop=True)
        uni_path = os.path.join(args.out_dir, "tcga_univariate_cox.tsv")
        uni_full.to_csv(uni_path, sep="\t", index=False)
        print("[OK] wrote:", uni_path)

        pass_genes = uni_full.loc[uni_full["fdr"] <= args.univ_fdr, "gene"].tolist()
        if len(pass_genes) == 0:
            pass_genes = uni_full["gene"].head(min(100, len(uni_full))).tolist()

        model_genes = pass_genes[:best_max_genes]
        pd.DataFrame({"gene": model_genes}).to_csv(os.path.join(args.out_dir, "model_genes.tsv"), sep="\t", index=False)
        print("[INFO] model genes:", len(model_genes))

        Xmv_z = Xtcga_z[model_genes]
        beta, info = cox_ph_fit_ridge(Xmv_z.values, time, event, l2=best_l2)
        print("[INFO] Cox fit:", info)

        coef = pd.DataFrame({"gene": model_genes, "beta": beta}).sort_values("beta", ascending=False)
        coef_path = os.path.join(args.out_dir, "tcga_multivariate_cox_coefs.tsv")
        coef.to_csv(coef_path, sep="\t", index=False)
        print("[OK] wrote:", coef_path)

        risk_tcga = Xmv_z.values @ beta
        c_tcga = c_index(time, event, risk_tcga)
        grp_tcga = (risk_tcga >= np.median(risk_tcga)).astype(int)
        chi2_tcga, p_tcga = _plot_km(os.path.join(fig_dir, "km_tcga.png"), time, event, grp_tcga, "TCGA-BLCA (train)")

        out_tcga = surv[["patient_id", "time_days", "event"]].copy()
        out_tcga["risk"] = risk_tcga
        out_tcga["group_high"] = grp_tcga
        out_tcga_path = os.path.join(args.out_dir, "tcga_patient_risk.tsv")
        out_tcga.to_csv(out_tcga_path, sep="\t", index=False)
        print("[OK] wrote:", out_tcga_path)
        print(f"[TCGA] C-index={c_tcga:.4f}  logrank p={p_tcga:.3g}")

        # External GSE
        common = [g for g in model_genes if g in gse.index]
        if len(common) != len(model_genes):
            print("[WARN] model genes missing in GSE:", len(model_genes) - len(common), " -> refit on intersection.")
            model_genes = common
            pd.DataFrame({"gene": model_genes}).to_csv(os.path.join(args.out_dir, "model_genes.tsv"), sep="\t", index=False)

            Xmv_z = Xtcga_z[model_genes]
            beta, info = cox_ph_fit_ridge(Xmv_z.values, time, event, l2=best_l2)
            coef = pd.DataFrame({"gene": model_genes, "beta": beta}).sort_values("beta", ascending=False)
            coef.to_csv(coef_path, sep="\t", index=False)

            risk_tcga = Xmv_z.values @ beta
            c_tcga = c_index(time, event, risk_tcga)
            grp_tcga = (risk_tcga >= np.median(risk_tcga)).astype(int)
            chi2_tcga, p_tcga = _plot_km(os.path.join(fig_dir, "km_tcga.png"), time, event, grp_tcga, "TCGA-BLCA (refit for GSE)")
            out_tcga["risk"] = risk_tcga
            out_tcga["group_high"] = grp_tcga
            out_tcga.to_csv(out_tcga_path, sep="\t", index=False)
            print(f"[TCGA-refit] C-index={c_tcga:.4f}  logrank p={p_tcga:.3g}")

        gseX = gse.loc[model_genes, clin["GSM"].tolist()].T
        mu2 = mu_tcga[model_genes]
        sd2 = sd_tcga[model_genes].replace(0, 1.0)
        gseX_z = (gseX - mu2) / sd2

        risk_gse = gseX_z.values @ beta
        c_gse = c_index(time_gse_days, event_gse, risk_gse)
        grp_gse = (risk_gse >= np.median(risk_gse)).astype(int)
        chi2_gse, p_gse = _plot_km(os.path.join(fig_dir, "km_gse13507.png"), time_gse_days, event_gse, grp_gse, "GSE13507 (external)")

        out_gse = clin[["GSM", "survivalMonth", "overall survival"]].copy()
        out_gse["time_days"] = time_gse_days
        out_gse["event"] = event_gse
        out_gse["risk"] = risk_gse
        out_gse["group_high"] = grp_gse
        out_gse_path = os.path.join(args.out_dir, "gse13507_risk.tsv")
        out_gse.to_csv(out_gse_path, sep="\t", index=False)
        print("[OK] wrote:", out_gse_path)
        print(f"[GSE13507] C-index={c_gse:.4f}  logrank p={p_gse:.3g}")

        card = {
            "model": "ridge_cox",
            "restrict_to_gse_genes": bool(restrict_to_gse),
            "tcga_patients": int(len(common_pat)),
            "gse_samples": int(gse.shape[1]),
            "candidate_genes_used": int(len(cand)),
            "univ_fdr": float(args.univ_fdr),
            "use_cv": bool(do_cv),
            "best": best,
            "best_max_genes": int(best_max_genes),
            "best_l2": float(best_l2),
            "tcga_cindex": float(c_tcga),
            "gse_cindex": float(c_gse),
            "tcga_logrank_p": float(p_tcga),
            "gse_logrank_p": float(p_gse),
            "genes": model_genes,
        }
        card_path = os.path.join(args.out_dir, "MODEL_CARD.json")
        with open(card_path, "w", encoding="utf-8") as w:
            json.dump(card, w, indent=2, ensure_ascii=False)
        print("[OK] wrote:", card_path)
        return

    # ---------------- Teacher/Student branch ----------------
    if args.model in ("teacher_only", "student_distill"):
        if torch is None:
            raise RuntimeError("PyTorch not available. Install torch or run --model ridge.")
        if args.teacher_risk_tsv is None and args.tcga_meth_tsv is None:
            raise RuntimeError("Need --tcga_meth_tsv for teacher, or provide --teacher_risk_tsv to skip teacher training.")

    # select RNA genes same way as ridge (univariate screening on TCGA z-scored candidates)
    Xtcga_z, _, mu_tcga, sd_tcga = standardize_train_apply(Xtcga, None)

    uni_full = cox_univariate_screen(Xtcga_z.values, list(Xtcga_z.columns), time, event)
    uni_full["fdr"] = multipletests(uni_full["pval"].values, method="fdr_bh")[1]
    uni_full = uni_full.sort_values(["fdr", "pval"]).reset_index(drop=True)

    pass_genes = uni_full.loc[uni_full["fdr"] <= args.univ_fdr, "gene"].tolist()
    if len(pass_genes) == 0:
        pass_genes = uni_full["gene"].head(min(200, len(uni_full))).tolist()

    model_genes = pass_genes[: int(args.max_genes)]
    pd.DataFrame({"gene": model_genes}).to_csv(os.path.join(args.out_dir, "model_genes.tsv"), sep="\t", index=False)
    print("[INFO] RNA genes used:", len(model_genes))

    # build TCGA RNA matrix (standardized)
    Xr_tcga = Xtcga[model_genes].values
    Xr_tcga_z, _, mu_rna, sd_rna = _standardize_np_train_apply(Xr_tcga, None)

    # GSE RNA standardized with TCGA params
    gseX = gse.loc[model_genes, clin["GSM"].tolist()].T
    mu2 = pd.Series(mu_tcga[model_genes].values, index=model_genes)
    sd2 = pd.Series(sd_tcga[model_genes].values, index=model_genes).replace(0, 1.0)
    Xr_gse_z = ((gseX - mu2) / sd2).values.astype(float)

    # teacher risks (either train teacher or load teacher_risk_tsv)
    teacher_risk_tcga = None
    teacher_obj = None
    student_obj = None
    best_trial = None

    if args.teacher_risk_tsv is not None:
        df = pd.read_csv(args.teacher_risk_tsv, sep="\t")
        if "patient_id" not in df.columns or "risk" not in df.columns:
            raise RuntimeError("--teacher_risk_tsv must have columns: patient_id, risk")
        df["patient_id"] = df["patient_id"].astype(str)
        df = df[df["patient_id"].isin(common_pat)].copy()
        df = df.set_index("patient_id")
        common_pat_all = list(common_pat)
        missing = [p for p in common_pat if p not in df.index]
        if missing:
            print("[WARN] %d TCGA patients missing in teacher risk; dropping: %s" % (len(missing), missing))
        common_pat = [p for p in common_pat if p in df.index]
        df = df.loc[common_pat]

        # --- Sync TCGA arrays to teacher-intersection patients (avoid index mismatch) ---
        if len(common_pat) != len(common_pat_all):
            keep_idx = [common_pat_all.index(p) for p in common_pat]
            try:
                X_tcga = X_tcga[keep_idx]
            except Exception:
                try:
                    X_tcga = X_tcga.iloc[keep_idx]
                except Exception:
                    pass
            for _name in ["time_tcga","event_tcga","tcga_time","tcga_event","y_time","y_event"]:
                try:
                    _arr = locals()[_name]
                except Exception:
                    continue
                try:
                    locals()[_name] = _arr[keep_idx]
                except Exception:
                    try:
                        locals()[_name] = _arr.iloc[keep_idx]
                    except Exception:
                        pass
            try:
                tcga_patients = [tcga_patients[i] for i in keep_idx]
            except Exception:
                pass
            print("[INFO] TCGA patients after teacher intersection: %d" % len(common_pat))

            # --- Re-align TCGA expr/survival/time/event after teacher intersection ---
            try:
                tcga_expr = tcga_expr[common_pat]
            except Exception:
                pass

            try:
                surv = surv.set_index("patient_id").loc[common_pat].reset_index()
                time = surv["time_days"].astype(float).values
                event = surv["event"].astype(int).values
            except Exception:
                pass

            print("[INFO] time/event length after teacher intersection:", len(time))

            # --- ALSO align RNA matrices used by student (Xr_tcga_z) to the same patient set ---

            try:

                _idx = [Xtcga.index.get_loc(p) for p in common_pat]

                try:

                    Xr_tcga = Xr_tcga[_idx]

                except Exception:

                    pass

                Xr_tcga_z = Xr_tcga_z[_idx]

            except Exception as e:

                print("[WARN] failed to align Xr_tcga_z after teacher intersection:", e)

            print("[INFO] Xr_tcga_z length after teacher intersection:", len(Xr_tcga_z))

        teacher_risk_tcga = df["risk"].astype(float).values
        print("[INFO] Loaded teacher risk:", args.teacher_risk_tsv)
    else:
        # load TCGA methylation and align patients
        meth = _read_expr(args.tcga_meth_tsv)  # features x patients
        meth.index = meth.index.astype(str)
        # do NOT alias-resolve methylation CpG IDs; only apply resolver if gene-level and user wants it
        # safest: only resolve if many features look like genes and resolver exists
        if resolver is not None:
            # heuristic: if >70% features are alnum/underscore without cg prefix, treat as gene-level
            feat = meth.index.to_series()
            gene_like = feat.str.match(r"^[A-Za-z0-9_.-]+$").mean()
            cpg_like = feat.str.startswith("cg").mean()
            if gene_like > 0.7 and cpg_like < 0.3:
                meth = apply_resolver_to_index(meth, resolver)

        # align patients intersection
        common_pat_m = [p for p in common_pat if p in meth.columns]
        if len(common_pat_m) < 50:
            raise RuntimeError(f"Too few TCGA patients with methylation aligned: {len(common_pat_m)}. "
                               f"Check methylation columns are patient_id and match survival.")
        # reorder all TCGA arrays to common_pat_m
        idx = [common_pat.index(p) for p in common_pat_m]
        time_m = time[idx]
        event_m = event[idx]
        Xr_tcga_m = Xr_tcga_z[idx]

        meth = meth[common_pat_m]
        # top variance features
        feats = _top_variance_features(meth, topk=int(args.meth_topk))
        meth = meth.loc[feats]
        Xm_tcga = meth.T.values.astype(float)
        Xm_tcga_z, _, mu_meth, sd_meth = _standardize_np_train_apply(Xm_tcga, None)

        # hyperparam search (random) - optimize STUDENT val c-index (or teacher if teacher_only)
        def one_run(latent_dim, dropout, lr, wd, alpha):
            out = _fit_teacher_and_student(
                X_rna=Xr_tcga_m,
                X_meth=Xm_tcga_z,
                time=time_m,
                event=event_m,
                seed=args.seed,
                latent_dim=latent_dim,
                dropout=dropout,
                lr=lr,
                weight_decay=wd,
                alpha_distill=alpha,
                epochs=args.epochs,
                patience=args.patience,
                device=args.device,
                do_student=(args.model == "student_distill"),
            )
            score = out["teacher_val_cindex"] if args.model == "teacher_only" else out["student_val_cindex"]
            return score, out, {
                "latent_dim": int(latent_dim),
                "dropout": float(dropout),
                "lr": float(lr),
                "weight_decay": float(wd),
                "alpha_distill": float(alpha),
                "meth_topk": int(args.meth_topk),
                "tcga_patients_with_meth": int(len(common_pat_m)),
                "meth_features_used": int(len(feats)),
            }, {
                "common_pat_m": common_pat_m,
                "feats": feats,
                "mu_meth": mu_meth,
                "sd_meth": sd_meth,
                "time_m": time_m,
                "event_m": event_m,
                "Xr_tcga_m": Xr_tcga_m,
                "Xm_tcga_z": Xm_tcga_z,
            }

        if args.hp_search:
            rng = np.random.default_rng(args.seed)
            best_score = -1.0
            best_payload = None
            for i in range(int(args.n_trials)):
                latent_dim = int(rng.choice([32, 64, 128, 256]))
                dropout = float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.5]))
                lr = float(10 ** rng.uniform(np.log10(1e-4), np.log10(3e-3)))
                wd = float(10 ** rng.uniform(np.log10(1e-6), np.log10(1e-2)))
                alpha = float(10 ** rng.uniform(np.log10(0.1), np.log10(10.0)))
                score, out, hp, cache = one_run(latent_dim, dropout, lr, wd, alpha)
                if score is None:
                    continue
                if float(score) > best_score:
                    best_score = float(score)
                    best_payload = (out, hp, cache)
                print(f"[HP] trial={i+1}/{args.n_trials} score={score:.4f} hp={hp}")
            if best_payload is None:
                raise RuntimeError("hp_search failed to produce a valid model.")
            out, hp, cache = best_payload
            best_trial = {"best_score": best_score, "hp": hp}
            print("[HP] best:", best_trial)
        else:
            score, out, hp, cache = one_run(args.latent_dim, args.dropout, args.lr, args.weight_decay, args.alpha_distill)
            best_trial = {"best_score": float(score), "hp": hp}

        teacher_obj = out["teacher"]
        student_obj = out["student"]
        # risks are only for the subset common_pat_m
        teacher_risk_tcga_m = out["risk_teacher_tcga"]
        student_risk_tcga_m = out["risk_student_tcga"]

        # Save torch models + preprocess info
        torch.save({"state_dict": teacher_obj.state_dict(), "hp": best_trial["hp"], "model_genes": model_genes},
                   os.path.join(args.out_dir, "teacher.pt"))
        if student_obj is not None:
            torch.save({"state_dict": student_obj.state_dict(), "hp": best_trial["hp"], "model_genes": model_genes},
                       os.path.join(args.out_dir, "student.pt"))

        # Expand risks back to full TCGA order (patients without meth get NaN)
        teacher_risk_tcga = np.full(len(common_pat), np.nan, float)
        student_risk_tcga = np.full(len(common_pat), np.nan, float)
        for j, p in enumerate(common_pat_m):
            k = common_pat.index(p)
            teacher_risk_tcga[k] = teacher_risk_tcga_m[j]
            if student_risk_tcga_m is not None:
                student_risk_tcga[k] = student_risk_tcga_m[j]

        # For evaluation/output we use:
        if args.model == "teacher_only":
            risk_tcga_use = teacher_risk_tcga
        else:
            risk_tcga_use = student_risk_tcga

        # Compute a simple linear map from student to all TCGA (including those without meth) by running student on RNA-only:
        if args.model == "student_distill" and student_obj is not None:
            student_obj.eval()
            with torch.no_grad():
                rs_all = student_obj(torch.tensor(Xr_tcga_z, dtype=torch.float32, device=args.device)).detach().cpu().numpy()
            risk_tcga_use = rs_all

        # Save risks now
        teacher_risk_tcga = risk_tcga_use

    # If we loaded teacher risk, and want student_distill, we train student using teacher risk as target (RNA-only)
    if args.teacher_risk_tsv is not None and args.model == "student_distill":
        # standardize teacher risk
        tr, va = _train_val_split(len(time), 0.2, seed=args.seed)
        # optional hp_search only over student
        def train_student(latent_dim, dropout, lr, wd, alpha):
            device = args.device
            Xtr = torch.tensor(Xr_tcga_z[tr], dtype=torch.float32, device=device)
            Xva = torch.tensor(Xr_tcga_z[va], dtype=torch.float32, device=device)
            ttr = torch.tensor(time[tr], dtype=torch.float32, device=device)
            etr = torch.tensor(event[tr], dtype=torch.int64, device=device)
            tva = time[va]
            eva = event[va]
            # --- distillation target normalization (train-split only; avoids huge-scale teacher dominating training) ---
            teach_tr = teacher_risk_tcga[tr].astype(float)
            teach_va = teacher_risk_tcga[va].astype(float)

            # handle NaNs defensively
            if np.isnan(teach_tr).any():
                fill = np.nanmedian(teach_tr)
                teach_tr = np.nan_to_num(teach_tr, nan=fill)
            if np.isnan(teach_va).any():
                fill = np.nanmedian(teach_tr)
                teach_va = np.nan_to_num(teach_va, nan=fill)

            # winsorize extreme outliers on train split (robust to heavy-tailed teacher risk)
            q_lo, q_hi = np.quantile(teach_tr, [0.01, 0.99])
            if np.isfinite(q_lo) and np.isfinite(q_hi) and q_hi > q_lo:
                teach_tr = np.clip(teach_tr, q_lo, q_hi)
                teach_va = np.clip(teach_va, q_lo, q_hi)

            # z-score using train split stats
            mu = float(np.mean(teach_tr))
            sd = float(np.std(teach_tr))
            if sd == 0.0 or (not np.isfinite(sd)):
                sd = 1.0
            teach_tr = (teach_tr - mu) / sd
            teach_va = (teach_va - mu) / sd

            rteach_tr = torch.tensor(teach_tr, dtype=torch.float32, device=device)
            rteach_va = teach_va

            stu = RNAStudentCox(Xr_tcga_z.shape[1], latent_dim=latent_dim, dropout=dropout).to(device)
            opt = optim.AdamW(stu.parameters(), lr=lr, weight_decay=wd)
            best_state = None
            best_val = -1.0
            bad = 0
            for ep in range(1, args.epochs + 1):
                stu.train()
                opt.zero_grad()
                rs = stu(Xtr)
                loss = _cox_ph_loss_torch(rs, ttr, etr) + alpha * torch.mean((rs - rteach_tr) ** 2)
                loss.backward()
                opt.step()

                stu.eval()
                with torch.no_grad():
                    rva = stu(Xva).detach().cpu().numpy()
                cva = float(c_index(tva, eva, rva))
                if cva > best_val + 1e-4:
                    best_val = cva
                    best_state = {k: v.detach().cpu() for k, v in stu.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            if best_state is not None:
                stu.load_state_dict(best_state)
            stu.eval()
            with torch.no_grad():
                rs_all = stu(torch.tensor(Xr_tcga_z, dtype=torch.float32, device=device)).detach().cpu().numpy()
            return best_val, stu, rs_all

        if args.hp_search:
            rng = np.random.default_rng(args.seed)
            best_val = -1.0
            best_pack = None
            for i in range(int(args.n_trials)):
                latent_dim = int(rng.choice([32, 64, 128, 256]))
                dropout = float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.5]))
                lr = float(10 ** rng.uniform(np.log10(1e-4), np.log10(3e-3)))
                wd = float(10 ** rng.uniform(np.log10(1e-6), np.log10(1e-2)))
                alpha = float(10 ** rng.uniform(np.log10(0.1), np.log10(10.0)))
                val, stu, rs = train_student(latent_dim, dropout, lr, wd, alpha)
                if val > best_val:
                    best_val = val
                    best_pack = (stu, rs, {"latent_dim": latent_dim, "dropout": dropout, "lr": lr, "weight_decay": wd, "alpha_distill": alpha})
                print(f"[HP-student] trial={i+1}/{args.n_trials} val_cindex={val:.4f}")
            student_obj, risk_tcga_use, hp = best_pack
            best_trial = {"best_score": float(best_val), "hp": hp}
        else:
            val, student_obj, risk_tcga_use = train_student(args.latent_dim, args.dropout, args.lr, args.weight_decay, args.alpha_distill)
            best_trial = {"best_score": float(val), "hp": {"latent_dim": args.latent_dim, "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay, "alpha_distill": args.alpha_distill}}

        torch.save({"state_dict": student_obj.state_dict(), "hp": best_trial["hp"], "model_genes": model_genes},
                   os.path.join(args.out_dir, "student.pt"))
        teacher_risk_tcga = risk_tcga_use

    # ---- Evaluate on TCGA (RNA-only risk used for output) ----
    risk_tcga = teacher_risk_tcga.astype(float)
    # risk direction sanity: keep as-is; step7 v2 can auto-fix sign if needed
    c_tcga = float(c_index(time, event, risk_tcga))
    grp_tcga = (risk_tcga >= np.nanmedian(risk_tcga)).astype(int)
    chi2_tcga, p_tcga = _plot_km(os.path.join(fig_dir, "km_tcga.png"), time, event, grp_tcga, f"TCGA-BLCA ({args.model})")

    out_tcga = surv[["patient_id", "time_days", "event"]].copy()
    out_tcga["risk"] = risk_tcga
    out_tcga["group_high"] = grp_tcga
    out_tcga_path = os.path.join(args.out_dir, "tcga_patient_risk.tsv")
    out_tcga.to_csv(out_tcga_path, sep="\t", index=False)
    print("[OK] wrote:", out_tcga_path)
    print(f"[TCGA] C-index={c_tcga:.4f}  logrank p={p_tcga:.3g}")

    # ---- External GSE (student is RNA-only; teacher_only cannot be evaluated unless methylation is present) ----
    if args.model == "teacher_only":
        print("[WARN] --model teacher_only cannot be evaluated on GSE13507 without methylation. Writing placeholder risk file.")
        out_gse = clin[["GSM", "survivalMonth", "overall survival"]].copy()
        out_gse["time_days"] = time_gse_days
        out_gse["event"] = event_gse
        out_gse["risk"] = np.nan
        out_gse["group_high"] = 0
        out_gse_path = os.path.join(args.out_dir, "gse13507_risk.tsv")
        out_gse.to_csv(out_gse_path, sep="\t", index=False)
        print("[OK] wrote:", out_gse_path)
        c_gse, p_gse = np.nan, np.nan
    else:
        # Run student on GSE RNA
        if student_obj is None:
            # if we trained teacher+student branch, student_obj should exist
            # but in the path where we created risk_tcga_use by running student on all tcga, we didn't keep object
            # safest: reload student.pt
            ck = torch.load(os.path.join(args.out_dir, "student.pt"), map_location=args.device)
            hp = ck["hp"]
            student_obj = RNAStudentCox(Xr_tcga_z.shape[1], latent_dim=int(hp["latent_dim"]), dropout=float(hp["dropout"])).to(args.device)
            student_obj.load_state_dict(ck["state_dict"])
            student_obj.eval()

        with torch.no_grad():
            risk_gse = student_obj(torch.tensor(Xr_gse_z, dtype=torch.float32, device=args.device)).detach().cpu().numpy()

        c_gse = float(c_index(time_gse_days, event_gse, risk_gse))
        grp_gse = (risk_gse >= np.median(risk_gse)).astype(int)
        chi2_gse, p_gse = _plot_km(os.path.join(fig_dir, "km_gse13507.png"), time_gse_days, event_gse, grp_gse, f"GSE13507 ({args.model})")

        out_gse = clin[["GSM", "survivalMonth", "overall survival"]].copy()
        out_gse["time_days"] = time_gse_days
        out_gse["event"] = event_gse
        out_gse["risk"] = risk_gse
        out_gse["group_high"] = grp_gse
        out_gse_path = os.path.join(args.out_dir, "gse13507_risk.tsv")
        out_gse.to_csv(out_gse_path, sep="\t", index=False)
        print("[OK] wrote:", out_gse_path)
        print(f"[GSE13507] C-index={c_gse:.4f}  logrank p={p_gse:.3g}")

    card = {
        "model": args.model,
        "restrict_to_gse_genes": bool(restrict_to_gse),
        "tcga_patients": int(len(common_pat)),
        "gse_samples": int(gse.shape[1]),
        "rna_genes_used": int(len(model_genes)),
        "univ_fdr": float(args.univ_fdr),
        "tcga_cindex": float(c_tcga),
        "gse_cindex": None if (isinstance(c_gse, float) and np.isnan(c_gse)) else float(c_gse),
        "tcga_logrank_p": float(p_tcga),
        "gse_logrank_p": None if (isinstance(p_gse, float) and np.isnan(p_gse)) else float(p_gse),
        "hp_search": bool(args.hp_search),
        "best_trial": best_trial,
        "notes": [
            "TCGA standardization params applied to GSE for RNA.",
            "If you see C-index<0.5 but KM significant, risk direction may be flipped; use step7 v2 --your_risk_sign auto to fix during comparison."
        ],
    }
    card_path = os.path.join(args.out_dir, "MODEL_CARD.json")
    with open(card_path, "w", encoding="utf-8") as w:
        json.dump(card, w, indent=2, ensure_ascii=False)
    print("[OK] wrote:", card_path)


if __name__ == "__main__":
    main()
