#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import re
import time
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

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
        analysis = it.get("analysis") or {}
        wf = analysis.get("workflow_type") if isinstance(analysis, dict) else None

        cases = it.get("cases") or []
        case_submitter_id = None
        sample_submitter_id = None
        sample_type = None
        if cases and isinstance(cases, list):
            c0 = cases[0]
            case_submitter_id = c0.get("submitter_id") or c0.get("case_submitter_id")
            samples = c0.get("samples") or []
            if samples and isinstance(samples, list):
                s0 = samples[0]
                sample_submitter_id = s0.get("submitter_id") or s0.get("sample_submitter_id")
                sample_type = s0.get("sample_type")

        rows.append(
            {
                "file_id": file_id,
                "file_name": file_name,
                "workflow_type": wf,
                "case_submitter_id": case_submitter_id,
                "sample_barcode": sample_submitter_id,
                "patient_id": None,
                "sample_type": sample_type,
            }
        )
    df = pd.DataFrame(rows)
    return df


def find_downloaded_file(download_dir: str, file_id: str, file_name: str) -> str:
    d = Path(download_dir)
    p1 = d / file_id / file_name
    if p1.exists():
        return str(p1)
    p2 = d / file_name
    if p2.exists():
        return str(p2)
    p3 = d / file_id
    if p3.exists() and p3.is_dir():
        files = list(p3.glob("*"))
        if len(files) == 1:
            return str(files[0])
        for x in files:
            if file_name and file_name in x.name:
                return str(x)
    raise FileNotFoundError(f"Cannot locate file: file_id={file_id}, file_name={file_name}")


def parse_counts_file(path: str, counts_col: str) -> pd.Series:
    # augmented_star_gene_counts.tsv:
    # 1) comment line "# gene-model: ..."
    # 2) header: gene_id gene_name gene_type unstranded ...
    # We parse with header=0 and comment="#".
    try:
        df = pd.read_csv(path, sep="\t", header=0, comment="#", dtype=str, engine="python")
    except Exception:
        # fallback
        df = pd.read_csv(path, sep="\t", header=0, comment="#", dtype=str)

    need_cols = ["gene_id", "unstranded", "stranded_first", "stranded_second"]
    if "gene_id" not in df.columns:
        raise ValueError(f"Missing gene_id column in: {path}")

    col = counts_col
    if col not in df.columns:
        # some files might use different header names; try position-based fallback
        # gene_id is col0, unstranded usually col3
        if df.shape[1] >= 4:
            idx_map = {"unstranded": 3, "stranded_first": 4, "stranded_second": 5}
            idx = idx_map[counts_col]
            g = df.iloc[:, 0].astype(str)
            v = pd.to_numeric(df.iloc[:, idx], errors="coerce").fillna(0).astype(np.int64)
        else:
            raise ValueError(f"Cannot find counts column {counts_col} in {path}")
    else:
        g = df["gene_id"].astype(str)
        v = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)

    # remove summary rows
    bad = g.str.startswith("N_") | g.str.startswith("__")
    g = g[~bad]
    v = v[~bad]

    # drop Ensembl version suffix
    g = g.str.replace(r"\.\d+$", "", regex=True)

    s = pd.Series(v.values, index=g.values)
    s = s[~s.index.duplicated(keep="first")]
    return s


def write_tsv_gz(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        df.to_csv(f, sep="\t", index=True)


def normalize_tcga_sample_id(x: str) -> str:
    """Normalize any TCGA submitter_id into SAMPLE barcode (TCGA-XX-YYYY-01A / -11A)."""
    if not x or not isinstance(x, str) or not x.startswith("TCGA-"):
        return x
    parts = x.split("-")
    if len(parts) >= 4:
        return "-".join(parts[:4])
    return x


def tcga_patient_id_from_sample(sample_barcode: str) -> str:
    if not sample_barcode or not isinstance(sample_barcode, str) or not sample_barcode.startswith("TCGA-"):
        return None
    return "-".join(sample_barcode.split("-")[:3])


def gdc_post_json(url: str, payload: dict, timeout: int = 120, max_retry: int = 8) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    for attempt in range(max_retry):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError) as e:
            if attempt == max_retry - 1:
                raise
            time.sleep(2 + attempt)
    raise RuntimeError("GDC request failed unexpectedly.")


def gdc_fetch_file_meta(file_ids: list[str]) -> dict:
    """
    Return mapping: file_id -> dict(fields)
    Key fix: also fetch associated_entities because some files don't populate cases.samples.
    """
    url = "https://api.gdc.cancer.gov/files"
    fields = ",".join(
        [
            "file_id",
            "file_name",
            "cases.submitter_id",
            "cases.samples.submitter_id",
            "cases.samples.sample_type",
            "associated_entities.entity_type",
            "associated_entities.entity_submitter_id",
        ]
    )
    payload = {
        "filters": {"op": "in", "content": {"field": "file_id", "value": file_ids}},
        "fields": fields,
        "format": "JSON",
        "size": len(file_ids),
    }
    js = gdc_post_json(url, payload)
    hits = js.get("data", {}).get("hits", []) or []
    print(f"[INFO] GDC API hits: {len(hits)}")

    out = {}
    for h in hits:
        fid = h.get("file_id")
        if not fid:
            continue

        case_submitter_id = None
        sample_barcode = None
        sample_type = None

        cases = h.get("cases") or []
        if cases:
            c0 = cases[0] or {}
            case_submitter_id = c0.get("submitter_id")
            samples = c0.get("samples") or []
            # try match a TCGA sample id
            for s in samples:
                sid = s.get("submitter_id")
                if sid and isinstance(sid, str) and sid.startswith("TCGA-"):
                    sample_barcode = normalize_tcga_sample_id(sid)
                    st = s.get("sample_type")
                    if st:
                        sample_type = st
                    break
            # fallback: first sample
            if sample_barcode is None and samples:
                sid = samples[0].get("submitter_id")
                if sid:
                    sample_barcode = normalize_tcga_sample_id(sid)
                st = samples[0].get("sample_type")
                if st:
                    sample_type = st

        # if still missing, use associated_entities
        if sample_barcode is None:
            aes = h.get("associated_entities") or []
            # prefer entity_type == "sample"
            for ae in aes:
                et = ae.get("entity_type")
                es = ae.get("entity_submitter_id")
                if es and isinstance(es, str) and es.startswith("TCGA-"):
                    if et == "sample":
                        sample_barcode = normalize_tcga_sample_id(es)
                        break
            # fallback: any TCGA id, normalize to sample
            if sample_barcode is None:
                for ae in aes:
                    es = ae.get("entity_submitter_id")
                    if es and isinstance(es, str) and es.startswith("TCGA-"):
                        sample_barcode = normalize_tcga_sample_id(es)
                        break

        patient_id = tcga_patient_id_from_sample(sample_barcode) if sample_barcode else None

        out[fid] = {
            "file_name": h.get("file_name"),
            "case_submitter_id": case_submitter_id,
            "sample_barcode": sample_barcode,
            "patient_id": patient_id,
            "sample_type": sample_type,
        }
    return out


def dedup_by_sample(mat: pd.DataFrame, sample_ids: list[str], mode: str):
    if mode == "none":
        return mat, sample_ids

    sids = pd.Series(sample_ids)
    dup = sids.duplicated(keep=False)
    if not dup.any():
        return mat, sample_ids

    keep_idx = []
    for sid, idxs in sids.groupby(sids).groups.items():
        idxs = list(idxs)
        if len(idxs) == 1:
            keep_idx.append(idxs[0])
            continue
        if mode == "first":
            keep_idx.append(idxs[0])
        elif mode == "largest_lib":
            libs = mat.iloc[:, idxs].sum(axis=0).values
            keep_idx.append(idxs[int(np.argmax(libs))])
        else:
            keep_idx.append(idxs[0])

    keep_idx = sorted(keep_idx)
    mat2 = mat.iloc[:, keep_idx]
    sids2 = [sample_ids[i] for i in keep_idx]
    return mat2, sids2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download_dir", required=True)
    ap.add_argument("--metadata_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--counts_col", default="unstranded", choices=["unstranded", "stranded_first", "stranded_second"])
    ap.add_argument(
        "--keep_sample_types",
        default="Primary Tumor,Solid Tissue Normal",
        help="comma separated sample_type filter; empty => keep all",
    )
    ap.add_argument("--fill_from_api", action="store_true", help="query GDC API to fill sample_barcode/sample_type")
    ap.add_argument("--api_overwrite", action="store_true", help="(compat) same as --fill_from_api; always overwrite")
    ap.add_argument("--make_logcpm", action="store_true")
    ap.add_argument("--min_cpm", type=float, default=0.0)
    ap.add_argument("--min_frac_samples", type=float, default=0.0)
    ap.add_argument("--dedup_by_sample", default="largest_lib", choices=["none", "largest_lib", "first"])
    args = ap.parse_args()

    meta = read_metadata_cart(args.metadata_json)
    meta = meta.dropna(subset=["file_id"]).copy()

    if args.fill_from_api or args.api_overwrite:
        file_ids = meta["file_id"].astype(str).tolist()
        api_map = gdc_fetch_file_meta(file_ids)
        for col in ["file_name", "case_submitter_id", "sample_barcode", "patient_id", "sample_type"]:
            meta[col] = meta["file_id"].astype(str).map(lambda x: api_map.get(x, {}).get(col))

    # filter by sample_type if requested
    if args.keep_sample_types.strip():
        keep = set([x.strip() for x in args.keep_sample_types.split(",") if x.strip()])
        # keep rows whose sample_type in keep; if sample_type missing, drop
        meta = meta[meta["sample_type"].isin(keep)].copy()

    # parse counts
    series_list = []
    file_ids_used = []
    for _, r in meta.iterrows():
        fid = str(r["file_id"])
        fname = str(r["file_name"]) if pd.notna(r["file_name"]) else ""
        try:
            fp = find_downloaded_file(args.download_dir, fid, fname)
        except Exception as e:
            print(f"[WARN] skip file_id={fid}: {e}")
            continue
        try:
            s = parse_counts_file(fp, args.counts_col)
        except Exception as e:
            print(f"[WARN] parse failed: {fp} : {e}")
            continue
        series_list.append(s.rename(fid))
        file_ids_used.append(fid)

    if not series_list:
        raise RuntimeError("No counts files parsed. Check download_dir/metadata_json.")

    mat = pd.concat(series_list, axis=1).fillna(0).astype(np.int64)

    # build column names = TCGA sample barcode if available
    meta2 = meta[meta["file_id"].astype(str).isin(file_ids_used)].copy()
    meta2 = meta2.drop_duplicates(subset=["file_id"])
    fid2sb = dict(zip(meta2["file_id"].astype(str), meta2["sample_barcode"].astype(str)))
    # create sample_ids aligned to mat columns
    sample_ids = []
    for fid in mat.columns.tolist():
        sb = fid2sb.get(str(fid))
        if sb and sb != "nan" and sb.startswith("TCGA-"):
            sample_ids.append(normalize_tcga_sample_id(sb))
        else:
            sample_ids.append(str(fid))  # fallback

    # dedup if needed
    mat, sample_ids = dedup_by_sample(mat, sample_ids, args.dedup_by_sample)
    mat.columns = sample_ids

    # CPM filter optional
    if args.min_cpm > 0 and args.min_frac_samples > 0:
        lib = mat.sum(axis=0).replace(0, np.nan)
        cpm = mat.div(lib, axis=1) * 1e6
        need = int(np.ceil(args.min_frac_samples * mat.shape[1]))
        ok = (cpm >= args.min_cpm).sum(axis=1) >= need
        mat = mat.loc[ok]
        print(f"[INFO] CPM filter kept genes: {mat.shape[0]}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_counts = os.path.join(args.out_dir, "tcga_blca_counts.tsv.gz")
    write_tsv_gz(mat, out_counts)
    print("[OK] wrote:", out_counts, "shape=", mat.shape)

    # write sample_info aligned to final columns
    # rebuild meta for kept sample_ids
    # if we deduped, multiple file_ids collapse to one sample_id; keep first occurrence
    meta3 = meta2.copy()
    meta3["sample_id"] = meta3["sample_barcode"].map(lambda x: normalize_tcga_sample_id(str(x)) if pd.notna(x) else None)
    meta3["patient_id"] = meta3["sample_id"].map(tcga_patient_id_from_sample)
    meta3 = meta3.dropna(subset=["sample_id"])
    meta3 = meta3.drop_duplicates(subset=["sample_id"], keep="first")

    # keep only sample_ids we wrote
    keep_set = set(sample_ids)
    meta3 = meta3[meta3["sample_id"].isin(keep_set)].copy()

    out_si = os.path.join(args.out_dir, "tcga_blca_sample_info.tsv")
    meta3[["sample_id", "patient_id", "sample_type", "file_id", "file_name"]].to_csv(out_si, sep="\t", index=False)
    print("[OK] wrote:", out_si)
    if "sample_type" in meta3.columns:
        print(meta3["sample_type"].value_counts(dropna=False))

    if args.make_logcpm:
        lib = mat.sum(axis=0).replace(0, np.nan)
        cpm = mat.div(lib, axis=1) * 1e6
        logcpm = np.log2(cpm + 1.0)
        out_log = os.path.join(args.out_dir, "tcga_blca_logcpm.tsv.gz")
        write_tsv_gz(logcpm, out_log)
        print("[OK] wrote:", out_log)


if __name__ == "__main__":
    main()