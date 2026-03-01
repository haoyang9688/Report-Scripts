#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stepM_tcga_methyl_fusion_v2.py

One-script pipeline for TCGA methylation + fusion Cox models.

Supports:
- methyl_input: a single matrix .txt OR a directory with many per-sample .txt
- optional mapping via --methyl_metadata_json (GDC metadata.cart.*.json) for UUID-named files
- methylation-only Cox (5-fold CV + KM/log-rank)
- fusion Cox: early / late / intermediate

Dependencies:
  pip install pandas numpy scikit-learn lifelines matplotlib
"""

import os, re, json, glob, argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_table_auto(path: str) -> pd.DataFrame:
    """
    Robust reader for txt/tsv/csv.
    - Skip binary (.parcel / pickle-like) by magic byte 0x80
    - Handle gzip even if extension is .txt.gz
    - Decode with utf-8 (errors=replace) fallback latin1
    - Use python engine for sep auto-detection
    """
    import gzip, io

    with open(path, "rb") as fb:
        head = fb.read(4)

    # binary (pickle/parcel) magic
    if head[:1] == b"\x80":
        raise ValueError(f"{path} looks like a binary (starts with 0x80), not a text table.")

    # gzip magic
    is_gz = head[:2] == b"\x1f\x8b"
    raw = gzip.open(path, "rb").read() if is_gz else open(path, "rb").read()

    try:
        txt = raw.decode("utf-8", errors="replace")
    except Exception:
        txt = raw.decode("latin1", errors="replace")

    buf = io.StringIO(txt)
    try:
        return pd.read_csv(buf, sep=None, engine="python")
    except Exception:
        buf.seek(0)
        try:
            return pd.read_csv(buf, sep="\t", engine="python")
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf, sep=",", engine="python")

def tcga_patient_id(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().replace(".", "-")
    return s[:12]


def find_tcga_barcode_in_text(text: str) -> str | None:
    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{3,4})", text)
    if m:
        return m.group(1)
    m2 = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", text)
    if m2:
        return m2.group(1)
    return None


def load_survival(surv_path: str) -> pd.DataFrame:
    df = read_table_auto(surv_path)
    cols = {c.lower(): c for c in df.columns}

    def pick(cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    pid = pick(["patient_id", "patient", "case_id", "submitter_id", "bcr_patient_barcode"])
    tcol = pick(["os_time", "os.time", "time", "days", "os", "days_to_death", "days_to_last_follow_up"])
    ecol = pick(["os_event", "os.event", "event", "status", "os_status", "vital_status"])

    if pid is None or tcol is None or ecol is None:
        raise ValueError(
            f"Cannot detect survival columns in {surv_path}. Need patient_id/time/event. Got: {list(df.columns)}"
        )

    out = df[[pid, tcol, ecol]].copy()
    out.columns = ["patient_id", "time", "event"]
    out["patient_id"] = out["patient_id"].astype(str).map(tcga_patient_id)
    out["time"] = pd.to_numeric(out["time"], errors="coerce")

    def norm_event(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ["dead", "deceased", "1", "true", "event"]:
                return 1
            if s in ["alive", "0", "false", "censored"]:
                return 0
        try:
            v = float(x)
            if v in [0.0, 1.0]:
                return int(v)
        except Exception:
            pass
        try:
            return int(float(x) != 0.0)
        except Exception:
            return np.nan

    out["event"] = out["event"].map(norm_event)
    out = out.dropna(subset=["patient_id", "time", "event"]).copy()
    out["event"] = out["event"].astype(int)
    return out.sort_values("patient_id")


def load_rna_expr_patientxgene(rna_path: str) -> pd.DataFrame:
    df = read_table_auto(rna_path)
    if df.columns[0].lower() in ["gene", "symbol", "id", "genes"] or "unnamed" in df.columns[0].lower():
        df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    def is_tcga(x):
        x = tcga_patient_id(x)
        return x.startswith("TCGA-") and len(x) == 12

    col_tcga = sum(is_tcga(c) for c in df.columns)
    idx_tcga = sum(is_tcga(i) for i in df.index)
    if col_tcga > idx_tcga:
        df = df.T

    df.index = df.index.map(tcga_patient_id)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return df


def build_gdc_file_to_patient_map(metadata_json: str) -> dict:
    with open(metadata_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("metadata_json is not a list; please provide metadata.cart.*.json")

    mp = {}
    for item in data:
        fid = str(item.get("file_id", "")).strip()
        fname = str(item.get("file_name", "")).strip()
        ents = item.get("associated_entities", []) or item.get("cases", []) or []

        tcga = None
        for ent in ents:
            for k in ["entity_submitter_id", "submitter_id", "case_id", "entity_id", "id"]:
                v = ent.get(k)
                if isinstance(v, str) and "TCGA-" in v:
                    tcga = v
                    break
            if tcga:
                break
        if not tcga:
            for k in ["submitter_id", "case_id", "id"]:
                v = item.get(k)
                if isinstance(v, str) and "TCGA-" in v:
                    tcga = v
                    break
        if not tcga:
            continue

        pid = tcga_patient_id(tcga)
        for key in [fid, fname]:
            if key:
                mp[key] = pid
                mp[os.path.splitext(key)[0]] = pid
    return mp


def detect_probe_beta_columns(df: pd.DataFrame):
    cols = list(df.columns)
    cols_l = [c.lower() for c in cols]

    probe_col = None
    for cand in ["composite element ref", "composite_element_ref", "probe", "cpg", "cg", "id"]:
        for c, cl in zip(cols, cols_l):
            if cand.replace(" ", "") in cl.replace(" ", ""):
                probe_col = c
                break
        if probe_col:
            break
    if probe_col is None:
        probe_col = cols[0]

    beta_col = None
    for cand in ["beta_value", "beta value", "beta", "value", "methylation"]:
        for c, cl in zip(cols, cols_l):
            if cand.replace(" ", "") in cl.replace(" ", ""):
                beta_col = c
                break
        if beta_col:
            break
    if beta_col is None:
        beta_col = cols[1] if len(cols) > 1 else cols[0]

    return probe_col, beta_col


def parse_one_methyl_txt(fp: str):
    df = read_table_auto(fp)
    probe_col, beta_col = detect_probe_beta_columns(df)
    df = df[[probe_col, beta_col]].copy()
    df.columns = ["probe", "beta"]
    df["probe"] = df["probe"].astype(str)
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df = df.dropna(subset=["beta"])
    s = df.set_index("probe")["beta"]
    barcode = ""
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            head = "".join([next(fh) for _ in range(30)])
        bc = find_tcga_barcode_in_text(head)
        barcode = bc or ""
    except Exception:
        pass
    return s, barcode


def load_methylation_dir_topk(methyl_dir: str, survival: pd.DataFrame, top_k: int,
                             metadata_json: str | None, use_m_value: bool):
    fps = sorted(glob.glob(os.path.join(methyl_dir, "*", "*.txt*")))
    fps = [f for f in fps if "/logs/" not in f and ".parcel" not in f]
    if len(fps) == 0:
        raise FileNotFoundError(f"No .txt files in {methyl_dir}")

    # file -> patient mapping
    file2pid = {}
    mp = build_gdc_file_to_patient_map(metadata_json) if metadata_json else {}
    surv_pids = set(survival["patient_id"].astype(str))

    for fp in fps:
        base = os.path.basename(fp)
        stem = os.path.splitext(base)[0]
        pid = ""
        if mp:
            uuid_dir = os.path.basename(os.path.dirname(fp))
        pid = mp.get(base) or mp.get(stem) or mp.get(uuid_dir) or ""
        if not pid:
            # try parse barcode from file name
            pid_try = tcga_patient_id(stem)
            if pid_try.startswith("TCGA-") and len(pid_try) == 12:
                pid = pid_try
        if not pid:
            # parse from file content
            _, bc = parse_one_methyl_txt(fp)
            pid = tcga_patient_id(bc) if bc else ""
        if pid and pid in surv_pids:
            file2pid[fp] = pid

    pairs = [(fp, pid) for fp, pid in file2pid.items()]
    if len(pairs) < 20:
        raise RuntimeError(
            f"Too few methylation files matched survival IDs ({len(pairs)}). Provide --methyl_metadata_json or check naming."
        )

    # probe universe from first file
    s0, _ = parse_one_methyl_txt(pairs[0][0])
    probes = s0.index.astype(str).tolist()
    P = len(probes)
    pos = {p: i for i, p in enumerate(probes)}

    n = np.zeros(P, dtype=np.int32)
    mean = np.zeros(P, dtype=np.float64)
    M2 = np.zeros(P, dtype=np.float64)

    def update(arr):
        mask = ~np.isnan(arr)
        idx = np.where(mask)[0]
        for i in idx:
            x = float(arr[i])
            ni = n[i] + 1
            delta = x - mean[i]
            mean[i] += delta / ni
            delta2 = x - mean[i]
            M2[i] += delta * delta2
            n[i] = ni

    # pass1: variance
    for fp, pid in pairs:
        s, _ = parse_one_methyl_txt(fp)
        arr = np.full(P, np.nan, dtype=np.float64)
        for pr, v in s.items():
            j = pos.get(str(pr))
            if j is not None:
                arr[j] = float(v)
        if use_m_value:
            eps = 1e-6
            b = np.clip(arr, eps, 1 - eps)
            arr = np.log2(b / (1 - b))
        update(arr)

    var = np.full(P, np.nan, dtype=np.float64)
    valid = n > 1
    var[valid] = M2[valid] / (n[valid] - 1)

    order = np.argsort(-np.nan_to_num(var, nan=-1.0))
    keep_idx = order[: min(top_k, P)]
    keep_probes = [probes[i] for i in keep_idx]

    # pass2: build matrix for keep probes
    rows, pids = [], []
    for fp, pid in pairs:
        s, _ = parse_one_methyl_txt(fp)
        vec = s.reindex(keep_probes).astype(float).values
        if use_m_value:
            eps = 1e-6
            b = np.clip(vec, eps, 1 - eps)
            vec = np.log2(b / (1 - b))
        rows.append(vec)
        pids.append(pid)

    X = pd.DataFrame(rows, index=pids, columns=keep_probes).groupby(level=0).mean()
    X.index.name = "patient_id"
    return X


def load_methylation(methyl_input: str, survival: pd.DataFrame, args) -> pd.DataFrame:
    if os.path.isdir(methyl_input):
        return load_methylation_dir_topk(
            methyl_dir=methyl_input,
            survival=survival,
            top_k=args.top_k_meth,
            metadata_json=args.methyl_metadata_json,
            use_m_value=args.use_m_value,
        )

    if not os.path.isfile(methyl_input):
        raise FileNotFoundError(methyl_input)

    df = read_table_auto(methyl_input)
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    col_tcga = sum(c.startswith("TCGA-") for c in df.columns)
    idx_tcga = sum(str(i).startswith("TCGA-") for i in df.index)
    if col_tcga > idx_tcga:
        df = df.T

    df.index = df.index.map(tcga_patient_id)
    df = df.loc[df.index.isin(set(survival["patient_id"]))].copy()
    df = df.apply(pd.to_numeric, errors="coerce").groupby(level=0).mean()

    if args.use_m_value:
        eps = 1e-6
        b = df.clip(eps, 1 - eps)
        df = np.log2(b / (1 - b))

    v = df.var(axis=0, skipna=True).sort_values(ascending=False)
    keep = v.head(min(args.top_k_meth, df.shape[1])).index.tolist()
    return df[keep]


def fit_ridge_cox(X: pd.DataFrame, y: pd.DataFrame, l2: float) -> CoxPHFitter:
    df = pd.concat([y[["time", "event"]], X], axis=1).dropna()
    cph = CoxPHFitter(penalizer=l2, l1_ratio=0.0)
    cph.fit(df, duration_col="time", event_col="event")
    return cph


def predict_risk(cph: CoxPHFitter, X: pd.DataFrame) -> pd.Series:
    """Return a *risk score* where larger values indicate higher hazard (worse prognosis).

    We prefer log-partial-hazard (linear predictor) to avoid numerical overflow from exp(lp).
    This is monotonic with partial hazard, so ranking/KM stratification are preserved.
    """
    X2 = X[cph.params_.index].dropna()
    # lifelines provides stable log partial hazard
    if hasattr(cph, "predict_log_partial_hazard"):
        r = cph.predict_log_partial_hazard(X2)
        return pd.Series(r.values.flatten(), index=X2.index, name="risk")
    # fallback: log(partial_hazard)
    r = cph.predict_partial_hazard(X2)
    rv = np.log(np.asarray(r).reshape(-1) + 1e-12)
    return pd.Series(rv, index=X2.index, name="risk")


def cv_cindex(X: pd.DataFrame, y: pd.DataFrame, l2: float, kfold: int, seed: int):
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    idx = np.arange(len(X))
    cidxs = []
    X_np = X.values

    for tr, va in kf.split(idx):
        Xtr = pd.DataFrame(X_np[tr], index=X.index[tr], columns=X.columns)
        Xva = pd.DataFrame(X_np[va], index=X.index[va], columns=X.columns)
        ytr = y.iloc[tr]
        yva = y.iloc[va]

        cph = fit_ridge_cox(Xtr, ytr, l2)
        risk = predict_risk(cph, Xva)
        yva2 = yva.set_index(Xva.index).loc[risk.index]
        c = concordance_index(yva2["time"], -risk, yva2["event"])
        cidxs.append(float(c))

    return float(np.mean(cidxs)), float(np.std(cidxs))




def write_risk_tsv(rdf: pd.DataFrame, out_tsv: str) -> None:
    """Write risk table with a stable ID column name.
    Ensures the first column is 'patient_id' instead of a generic 'index'.
    """
    rdf = rdf.copy()
    if rdf.index.name is None:
        rdf.index.name = "patient_id"
    out = rdf.reset_index().rename(columns={"index": "patient_id"})
    out.to_csv(out_tsv, sep="\t", index=False)

def km_plot(risk_df: pd.DataFrame, out_png: str, title: str):
    med = float(risk_df["risk"].median())
    low = risk_df[risk_df["risk"] <= med]
    high = risk_df[risk_df["risk"] > med]

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6.2, 4.6))
    ax = plt.gca()

    kmf.fit(low["time"], low["event"], label="Low risk")
    kmf.plot_survival_function(ax=ax)
    kmf.fit(high["time"], high["event"], label="High risk")
    kmf.plot_survival_function(ax=ax)

    lr = logrank_test(low["time"], high["time"], event_observed_A=low["event"], event_observed_B=high["event"])
    p = float(lr.p_value)
    chi2 = float(lr.test_statistic)

    plt.title(f"{title}\nlog-rank p={p:.4g}, chi2={chi2:.2f}")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return p, chi2


def late_fusion_cv(risk_rna: pd.Series, risk_meth: pd.Series, y: pd.DataFrame,
                  kfold: int, seed: int, steps: int):
    """Cross-validate late-fusion weight w on *standardized* modality risks.

    Important:
    - We standardize rr/rm (z-score) to handle scale mismatch between modalities.
    - lifelines.concordance_index assumes larger prediction => longer survival.
      Since our risk is larger => worse survival, we pass -pred.
    """
    common = sorted(set(risk_rna.index) & set(risk_meth.index) & set(y.index))
    rr = risk_rna.loc[common].astype(float)
    rm = risk_meth.loc[common].astype(float)
    yy = y.loc[common]

    def z(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        return (s - s.mean()) / (s.std() + 1e-12)

    rr_z = z(rr)
    rm_z = z(rm)

    ws = np.linspace(0, 1, steps)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)

    best_mean, best_std, best_w = -1e9, None, None
    for w in ws:
        cidxs = []
        for _, va in kf.split(common):
            va_ids = [common[i] for i in va]
            pred = w * rr_z.loc[va_ids] + (1 - w) * rm_z.loc[va_ids]
            yva = yy.loc[va_ids]
            c = concordance_index(yva["time"], -pred, yva["event"])
            cidxs.append(float(c))
        m = float(np.mean(cidxs))
        s = float(np.std(cidxs))
        if m > best_mean:
            best_mean, best_std, best_w = m, s, float(w)
    return best_mean, best_std, best_w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methyl_input", required=True, help="methylation: a .txt file OR a directory of .txt files")
    ap.add_argument("--methyl_metadata_json", default=None, help="optional: GDC metadata.cart.*.json for UUID-named files")
    ap.add_argument("--survival_tsv", required=True, help="TCGA survival table")
    ap.add_argument("--rna_expr_tsv", default=None, help="TCGA patient×gene RNA expr (needed for fusion)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", required=True, choices=["methyl_only", "early_fusion", "late_fusion", "intermediate_fusion"])

    ap.add_argument("--use_m_value", action="store_true")
    ap.add_argument("--top_k_meth", type=int, default=5000)
    ap.add_argument("--top_k_rna", type=int, default=2000)
    ap.add_argument("--rna_genes_list", default=None)

    ap.add_argument("--l2", type=float, default=3e-4)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--late_w_steps", type=int, default=21)
    ap.add_argument("--rna_pca_dim", type=int, default=30)
    ap.add_argument("--meth_pca_dim", type=int, default=30)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    y = load_survival(args.survival_tsv).set_index("patient_id").sort_index()

    # methylation
    Xm = load_methylation(args.methyl_input, y.reset_index(), args)
    Xm = Xm.loc[Xm.index.isin(y.index)].copy()
    Xm = Xm.apply(pd.to_numeric, errors="coerce").fillna(Xm.median(axis=0))

    sc_m = StandardScaler()
    Xm_z = pd.DataFrame(sc_m.fit_transform(Xm.values), index=Xm.index, columns=Xm.columns)

    with open(os.path.join(args.out_dir, "selected_features_meth.txt"), "w") as f:
        for c in Xm_z.columns:
            f.write(c + "\n")

    results = []

    if args.model == "methyl_only":
        yy = y.loc[Xm_z.index]
        mc, sd = cv_cindex(Xm_z, yy, args.l2, args.kfold, args.seed)
        cph = fit_ridge_cox(Xm_z, yy, args.l2)
        risk = predict_risk(cph, Xm_z)
        rdf = yy.loc[risk.index].copy()
        rdf["risk"] = risk
        p, chi2 = km_plot(rdf, os.path.join(args.out_dir, "km_methyl_only.png"), "TCGA methylation-only Cox")
        write_risk_tsv(rdf, os.path.join(args.out_dir, "risk_methyl_only.tsv"))
        results.append({"model": "methyl_only", "cv_cindex_mean": mc, "cv_cindex_std": sd, "km_p": p, "km_chi2": chi2,
                        "n_patients": len(rdf), "n_meth_features": Xm_z.shape[1]})

    else:
        if args.rna_expr_tsv is None:
            raise ValueError("--rna_expr_tsv is required for fusion models")

        Xr = load_rna_expr_patientxgene(args.rna_expr_tsv)
        if args.rna_genes_list:
            genes = [g.strip() for g in open(args.rna_genes_list, "r", encoding="utf-8") if g.strip()]
            keep = [g for g in genes if g in Xr.columns]
            Xr = Xr[keep]

        # align and select topK variance
        common_r = sorted(set(Xr.index) & set(y.index))
        Xr = Xr.loc[common_r].apply(pd.to_numeric, errors="coerce").fillna(Xr.median(axis=0))
        if Xr.shape[1] > args.top_k_rna:
            v = Xr.var(axis=0, skipna=True).sort_values(ascending=False)
            Xr = Xr[v.head(args.top_k_rna).index]
        sc_r = StandardScaler()
        Xr_z = pd.DataFrame(sc_r.fit_transform(Xr.values), index=Xr.index, columns=Xr.columns)

        common = sorted(set(Xr_z.index) & set(Xm_z.index) & set(y.index))
        Xr_z = Xr_z.loc[common]
        Xm_z2 = Xm_z.loc[common]
        yy = y.loc[common]

        if args.model == "early_fusion":
            X = pd.concat([Xr_z, Xm_z2], axis=1)
            mc, sd = cv_cindex(X, yy, args.l2, args.kfold, args.seed)
            cph = fit_ridge_cox(X, yy, args.l2)
            risk = predict_risk(cph, X)
            rdf = yy.loc[risk.index].copy()
            rdf["risk"] = risk
            p, chi2 = km_plot(rdf, os.path.join(args.out_dir, "km_early_fusion.png"), "TCGA early fusion (RNA+methyl)")
            write_risk_tsv(rdf, os.path.join(args.out_dir, "risk_early_fusion.tsv"))
            results.append({"model": "early_fusion", "cv_cindex_mean": mc, "cv_cindex_std": sd, "km_p": p, "km_chi2": chi2,
                            "n_patients": len(rdf), "n_total_features": X.shape[1]})

        elif args.model == "intermediate_fusion":
            pr = PCA(n_components=min(args.rna_pca_dim, Xr_z.shape[1]))
            pm = PCA(n_components=min(args.meth_pca_dim, Xm_z2.shape[1]))
            Xrp = pr.fit_transform(Xr_z.values)
            Xmp = pm.fit_transform(Xm_z2.values)
            X = pd.DataFrame(np.concatenate([Xrp, Xmp], axis=1), index=common,
                             columns=[f"RNA_PC{i+1}" for i in range(Xrp.shape[1])] + [f"METH_PC{i+1}" for i in range(Xmp.shape[1])])
            mc, sd = cv_cindex(X, yy, args.l2, args.kfold, args.seed)
            cph = fit_ridge_cox(X, yy, args.l2)
            risk = predict_risk(cph, X)
            rdf = yy.loc[risk.index].copy()
            rdf["risk"] = risk
            p, chi2 = km_plot(rdf, os.path.join(args.out_dir, "km_intermediate_fusion.png"), "TCGA intermediate fusion (PCA)")
            write_risk_tsv(rdf, os.path.join(args.out_dir, "risk_intermediate_fusion.tsv"))
            results.append({"model": "intermediate_fusion", "cv_cindex_mean": mc, "cv_cindex_std": sd, "km_p": p, "km_chi2": chi2,
                            "n_patients": len(rdf), "n_total_features": X.shape[1]})

        elif args.model == "late_fusion":
            cph_r = fit_ridge_cox(Xr_z, yy, args.l2)
            rr = predict_risk(cph_r, Xr_z)
            cph_m = fit_ridge_cox(Xm_z2, yy, args.l2)
            rm = predict_risk(cph_m, Xm_z2)

                        # CV weight selection is performed on standardized rr/rm (inside late_fusion_cv).
            # For final fused risk, apply the same standardization to avoid scale dominance.
            def z(s: pd.Series) -> pd.Series:
                s = s.astype(float)
                return (s - s.mean()) / (s.std() + 1e-12)

            rr_z = z(rr)
            rm_z = z(rm)

            mc, sd, w = late_fusion_cv(rr, rm, yy, args.kfold, args.seed, args.late_w_steps)
            risk = w * rr_z + (1 - w) * rm_z
            if w <= 1e-6 or (1 - w) <= 1e-6:
                print(f"[WARN] late fusion degenerated (w={w:.4f}); equivalent to a single modality after standardization.")
            rdf = yy.loc[risk.index].copy()
            rdf["risk"] = risk
            p, chi2 = km_plot(rdf, os.path.join(args.out_dir, "km_late_fusion.png"), f"TCGA late fusion (w={w:.2f})")
            write_risk_tsv(rdf, os.path.join(args.out_dir, "risk_late_fusion.tsv"))
            results.append({"model": "late_fusion", "cv_cindex_mean": mc, "cv_cindex_std": sd, "late_w": w, "km_p": p, "km_chi2": chi2,
                            "n_patients": len(rdf), "n_total_features": 2})

    pd.DataFrame(results).to_csv(os.path.join(args.out_dir, "cv_results.tsv"), sep="\t", index=False)
    print("Done. Summary saved to", os.path.join(args.out_dir, "cv_results.tsv"))


if __name__ == "__main__":
    main()
