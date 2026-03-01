#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
import time
import numpy as np
import pandas as pd
from urllib import request
from urllib.error import URLError, HTTPError

GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"

FIELDS = ",".join([
    "submitter_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.days_to_last_follow_up",
    "diagnoses.vital_status",
    "diagnoses.days_to_death",
    "diagnoses.days_to_last_follow_up",
    "follow_ups.days_to_last_follow_up",
    # 可选：一些临床协变量（后续建模可能用到）
    "demographic.gender",
    "diagnoses.age_at_diagnosis",
    "diagnoses.tumor_stage",
    "diagnoses.tumor_grade",
])

def tcga_patient_id(sample_id: str) -> str:
    s = str(sample_id)
    if s.startswith("TCGA-"):
        parts = s.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
    return s

def to_num(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"na","nan","none","null","not reported","unknown","--"}:
            return None
        return float(s)
    except Exception:
        return None

def norm_vs(x):
    if x is None:
        return ""
    return str(x).strip().lower()

def gdc_post_cases(patient_ids, timeout=120, max_retry=8, sleep_base=2.0):
    payload = {
        "filters": {"op":"in", "content":{"field":"submitter_id", "value": list(patient_ids)}},
        "fields": FIELDS,
        "format": "JSON",
        "size": len(patient_ids),
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(GDC_CASES_URL, data=data, headers={"Content-Type":"application/json"})
    for attempt in range(max_retry):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError) as e:
            if attempt == max_retry - 1:
                raise
            time.sleep(sleep_base + attempt)

def extract_os_from_hit(hit: dict):
    """
    OS 规则（稳健版）：
      - event=1: vital_status=Dead/Deceased 或 days_to_death 有值
      - time_days:
          - 若 event=1 -> days_to_death
          - 若 event=0 -> days_to_last_follow_up（demographic/diagnoses/follow_ups 取最大可用值）
      - 若 time_days 缺失 -> 丢弃该病人
    """
    pid = hit.get("submitter_id")

    demo = hit.get("demographic") if isinstance(hit.get("demographic"), dict) else {}
    diags = hit.get("diagnoses") if isinstance(hit.get("diagnoses"), list) else []
    d0 = diags[0] if (len(diags) > 0 and isinstance(diags[0], dict)) else {}

    # 1) vital_status 优先 demographic
    vs = demo.get("vital_status")
    if vs is None:
        vs = d0.get("vital_status")
    vs_n = norm_vs(vs)

    # 2) days_to_death 优先 demographic
    dtd = to_num(demo.get("days_to_death"))
    if dtd is None:
        dtd = to_num(d0.get("days_to_death"))

    # 3) days_to_last_follow_up：demographic/diagnoses/follow_ups 取最大可用值
    lfu_vals = []
    lfu_demo = to_num(demo.get("days_to_last_follow_up"))
    lfu_diag = to_num(d0.get("days_to_last_follow_up"))
    if lfu_demo is not None: lfu_vals.append(lfu_demo)
    if lfu_diag is not None: lfu_vals.append(lfu_diag)

    fus = hit.get("follow_ups") if isinstance(hit.get("follow_ups"), list) else []
    for fu in fus:
        if isinstance(fu, dict):
            v = to_num(fu.get("days_to_last_follow_up"))
            if v is not None:
                lfu_vals.append(v)
    lfu = max(lfu_vals) if lfu_vals else None

    # 4) event 判定
    dead_flag = (vs_n in {"dead","deceased"})
    if dtd is not None and dtd >= 0:
        dead_flag = True

    event = 1 if dead_flag else 0
    time_days = dtd if (event == 1) else lfu

    # Guardrail: OS time must be non-negative.
    # If vital_status indicates death but days_to_death is missing/negative,
    # we conservatively fall back to last_follow_up as censored (event=0) when available.
    if event == 1 and (time_days is None or time_days < 0):
        if lfu is not None and lfu >= 0:
            event = 0
            time_days = lfu
        else:
            return None

    if time_days is None or time_days < 0:
        return None


    if time_days is None:
        return None

    # 可选协变量
    gender = demo.get("gender")
    age = d0.get("age_at_diagnosis")
    stage = d0.get("tumor_stage")
    grade = d0.get("tumor_grade")

    return {
        "patient_id": pid,
        "time_days": float(time_days),
        "time_months": float(time_days) / 30.4375,
        "event": int(event),
        "vital_status": vs,
        "gender": gender,
        "age_at_diagnosis": age,
        "tumor_stage": stage,
        "tumor_grade": grade,
    }

def read_logcpm_symbol(path_gz: str) -> pd.DataFrame:
    with gzip.open(path_gz, "rt", encoding="utf-8", errors="ignore") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def collapse_sample_to_patient(expr: pd.DataFrame, mode="mean") -> pd.DataFrame:
    patients = [tcga_patient_id(c) for c in expr.columns]
    grp = {}
    for col, pid in zip(expr.columns, patients):
        grp.setdefault(pid, []).append(col)

    out = {}
    for pid, cols in grp.items():
        sub = expr[cols]
        if len(cols) == 1:
            out[pid] = sub.iloc[:, 0]
        else:
            if mode == "median":
                out[pid] = sub.median(axis=1, skipna=True)
            else:
                out[pid] = sub.mean(axis=1, skipna=True)

    mat = pd.DataFrame(out)
    return mat

def write_tsv_gz(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, sep="\t", index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logcpm_symbol_gz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--collapse", default="mean", choices=["mean","median"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Read logcpm symbol:", args.logcpm_symbol_gz)
    expr = read_logcpm_symbol(args.logcpm_symbol_gz)
    print("[INFO] expr shape (genes x samples):", expr.shape)

    print("[INFO] Collapse sample -> patient (mode=%s) ..." % args.collapse)
    expr_pat = collapse_sample_to_patient(expr, mode=args.collapse)
    print("[INFO] patient expr shape:", expr_pat.shape)

    patient_ids = list(expr_pat.columns)
    print("[INFO] Query GDC cases:", len(patient_ids))
    js = gdc_post_cases(patient_ids)
    hits = js.get("data", {}).get("hits", [])
    print("[INFO] GDC hits:", len(hits))

    rows = []
    for h in hits:
        r = extract_os_from_hit(h)
        if r is not None:
            rows.append(r)
    surv = pd.DataFrame(rows)

    if surv.empty:
        raise RuntimeError("No survival rows parsed from GDC. Check fields/connection.")

    surv = surv.sort_values("patient_id").reset_index(drop=True)
    print("[INFO] survival rows (non-missing time): %d/%d" % (len(surv), len(patient_ids)))
    print("[INFO] event counts:\n", surv["event"].value_counts(dropna=False))

    keep_p = surv["patient_id"].tolist()
    expr_pat = expr_pat[keep_p]

    out_expr = os.path.join(args.out_dir, "tcga_blca_logcpm.symbol.patient.tsv.gz")
    out_surv = os.path.join(args.out_dir, "tcga_blca_survival_os.tsv")

    write_tsv_gz(expr_pat, out_expr)
    surv.to_csv(out_surv, sep="\t", index=False)

    print("[OK] wrote:", out_expr, "shape=", expr_pat.shape)
    print("[OK] wrote:", out_surv, "rows=", len(surv))

if __name__ == "__main__":
    main()