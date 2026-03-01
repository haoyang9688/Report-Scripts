#!/usr/bin/env python3
import argparse
import re
import pandas as pd
from urllib import request

GPL_ACC = "GPL6102"
GPL_URL = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GPL_ACC}&targ=self&form=text&view=full"

def clean_symbol(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().strip('"')
    if s.lower() in {"", "na", "nan", "none", "null"}:
        return ""
    # 取第一个符号（常见分隔符）
    for sep in [" /// ", "///", " // ", "//", ";", ",", "|"]:
        if sep in s:
            s = s.split(sep)[0].strip()
            break
    # 去掉不合法字符（保守）
    s = re.sub(r"\s+", "", s)
    return s

def fetch_gpl_table(gpl_soft_path: str = None) -> pd.DataFrame:
    """
    读取 GPL6102 平台表，输出 DataFrame（至少包含 probe_id 和 gene_symbol）
    - 若提供 --gpl_soft_path，则从本地读取（避免联网）
    - 否则从 NCBI 直接下载 GPL6102 full SOFT text 并解析 platform table
    """
    if gpl_soft_path:
        text = open(gpl_soft_path, "r", encoding="utf-8", errors="ignore").read()
    else:
        with request.urlopen(GPL_URL, timeout=180) as resp:
            text = resp.read().decode("utf-8", errors="ignore")

    lines = text.splitlines()
    # 找 platform table begin/end
    tb = None
    te = None
    for i, l in enumerate(lines):
        if l.strip() == "!platform_table_begin":
            tb = i
        if l.strip() == "!platform_table_end":
            te = i
            break
    if tb is None or te is None or te <= tb + 1:
        raise RuntimeError("Failed to locate !platform_table_begin/end in GPL text.")

    header = lines[tb + 1].rstrip("\n").split("\t")
    data_lines = lines[tb + 2: te]
    rows = [ln.split("\t") for ln in data_lines if ln.strip()]

    gpl = pd.DataFrame(rows, columns=header)
    # probe id 列：一般叫 ID
    probe_col = None
    for c in gpl.columns:
        if c.strip().lower() == "id":
            probe_col = c
            break
    if probe_col is None:
        probe_col = gpl.columns[0]

    # gene symbol 列：不同 GPL 可能叫 Gene Symbol / Symbol / ILMN_Gene 等
    sym_col = None
    cand_cols = [c for c in gpl.columns]
    norm = {c: re.sub(r"[\s_]+", "", c.strip().lower()) for c in cand_cols}
    for c in cand_cols:
        if norm[c] in {"genesymbol", "symbol", "gene"}:
            sym_col = c
            break
    if sym_col is None:
        # 常见：ILMN_Gene / SYMBOL / Gene Symbol
        for c in cand_cols:
            if "symbol" in norm[c] or "ilmngene" in norm[c]:
                sym_col = c
                break

    if sym_col is None:
        raise RuntimeError(f"Cannot find a gene symbol column in GPL table. Columns sample: {gpl.columns[:30].tolist()}")

    out = gpl[[probe_col, sym_col]].copy()
    out.columns = ["probe_id", "gene_symbol_raw"]
    out["probe_id"] = out["probe_id"].astype(str).str.strip().str.strip('"')
    out["gene_symbol"] = out["gene_symbol_raw"].map(clean_symbol)
    out = out[out["probe_id"].notna() & (out["probe_id"] != "")]
    out = out.drop_duplicates("probe_id", keep="first")
    return out[["probe_id", "gene_symbol"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr_tsv", required=True, help="probe-level expr (first col ILMN_*, columns GSM...)")
    ap.add_argument("--out_tsv_gz", required=True, help="gene-symbol expr output (.tsv.gz)")
    ap.add_argument("--collapse", default="median", choices=["median", "mean", "max"])
    ap.add_argument("--gpl_soft_path", default=None, help="optional: local GPL6102 SOFT text file")
    ap.add_argument("--save_map_tsv", default=None, help="optional: save probe->symbol mapping tsv")
    args = ap.parse_args()

    print("[INFO] Loading probe-level expression:", args.expr_tsv)
    expr = pd.read_csv(args.expr_tsv, sep="\t")
    probe_col = expr.columns[0]
    expr = expr.rename(columns={probe_col: "probe_id"})
    expr["probe_id"] = expr["probe_id"].astype(str).str.strip().str.strip('"')
    expr = expr.set_index("probe_id")

    print("[INFO] Fetching GPL6102 probe->symbol map ...")
    mp = fetch_gpl_table(args.gpl_soft_path)
    if args.save_map_tsv:
        mp.to_csv(args.save_map_tsv, sep="\t", index=False)
        print("[OK] wrote mapping:", args.save_map_tsv, "rows=", len(mp))

    d = dict(zip(mp["probe_id"], mp["gene_symbol"]))
    symbols = expr.index.to_series().map(lambda x: d.get(x, ""))
    keep = symbols.ne("")
    expr2 = expr.loc[keep].copy()
    symbols2 = symbols.loc[keep]

    # 转数值
    expr2 = expr2.apply(pd.to_numeric, errors="coerce")

    # 按 gene symbol 合并多个 probes
    expr2["gene_symbol"] = symbols2.values
    if args.collapse == "median":
        out = expr2.groupby("gene_symbol").median(numeric_only=True)
    elif args.collapse == "mean":
        out = expr2.groupby("gene_symbol").mean(numeric_only=True)
    else:
        out = expr2.groupby("gene_symbol").max(numeric_only=True)

    out.to_csv(args.out_tsv_gz, sep="\t", compression="gzip")
    print("[OK] wrote:", args.out_tsv_gz, "shape=", out.shape)
    print("[INFO] mapped probes:", int(keep.sum()), "/", len(expr))

if __name__ == "__main__":
    main()