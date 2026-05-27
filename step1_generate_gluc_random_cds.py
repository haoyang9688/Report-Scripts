#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uniform random synonymous CDS generation and lowest-CAI selection.

Design
------
1. Fix Gluc AA sequence.
2. Uniformly sample synonymous codons at each AA position.
3. Generate N unique CDS sequences, e.g. 10000.
4. No CAI hard filter during generation.
5. Calculate HEK293T CAI / GC / Rare for all generated sequences.
6. Sort by CAI ascending.
7. Select the lowest-CAI TopK sequences for RL start.
8. Calculate MFE only for selected TopK sequences by default.

This is:
    unconditional uniform random synonymous generation
    + post-hoc lowest-CAI selection.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import RNA
    HAS_RNA = True
except ImportError:
    HAS_RNA = False


STOP_CODONS = {"TAA", "TAG", "TGA"}
ORDERED_STOP_CODONS = ("TAA", "TAG", "TGA")

CODON2AA = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",

    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",

    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",

    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

AA2CODONS: Dict[str, List[str]] = {}
for codon, aa in CODON2AA.items():
    if aa != "*":
        AA2CODONS.setdefault(aa, []).append(codon)
for aa in AA2CODONS:
    AA2CODONS[aa] = sorted(AA2CODONS[aa])

SENSE_CODONS = sorted([c for c, aa in CODON2AA.items() if aa != "*"])

DEFAULT_GLUC_AA = (
    "MGVKVLFALICIAVAEAKPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEM"
    "EANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPG"
    "FKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIK"
    "GAGGD"
)


def clean_aa(seq: str) -> str:
    seq = "".join(str(seq).split()).upper()
    if seq.endswith("*"):
        seq = seq[:-1]
    bad = sorted(set(x for x in seq if x not in AA2CODONS))
    if bad:
        raise ValueError(f"Invalid amino-acid symbols: {bad}")
    return seq


def clean_dna(seq: str) -> str:
    return "".join(str(seq).split()).upper().replace("U", "T")


def read_first_fasta_sequence(path: str) -> Tuple[str, str]:
    header = None
    seq_parts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:].strip()
            else:
                seq_parts.append(line)

    if header is None or not seq_parts:
        raise ValueError(f"No FASTA sequence found in: {path}")

    return header, "".join(seq_parts)


def split_codons(seq: str) -> List[str]:
    seq = clean_dna(seq)
    if len(seq) % 3 != 0:
        raise ValueError(f"CDS length must be multiple of 3, got {len(seq)}")
    return [seq[i:i + 3] for i in range(0, len(seq), 3)]


def split_cds_and_stop(seq: str) -> Tuple[List[str], Optional[str]]:
    codons = split_codons(seq)
    if codons and codons[-1] in STOP_CODONS:
        return codons[:-1], codons[-1]
    return codons, None


def translate_cds(seq: str) -> str:
    sense_codons, _ = split_cds_and_stop(seq)
    aas = []
    for i, codon in enumerate(sense_codons):
        aa = CODON2AA.get(codon)
        if aa is None:
            raise ValueError(f"Unknown codon {codon} at codon index {i}")
        if aa == "*":
            raise ValueError(f"Internal stop codon {codon} at codon index {i}")
        aas.append(aa)
    return "".join(aas)


def choose_stop(stop_mode: str, fixed_stop: str, idx: int, rng: random.Random) -> str:
    if stop_mode == "fixed":
        return fixed_stop
    if stop_mode == "random_three":
        return rng.choice(list(ORDERED_STOP_CODONS))
    if stop_mode == "balanced_three":
        return ORDERED_STOP_CODONS[int(idx) % len(ORDERED_STOP_CODONS)]
    raise ValueError(f"Unsupported stop_mode: {stop_mode}")


def load_weight_table(path: str) -> Dict[str, float]:
    weights = {}
    with open(path, "r", encoding="utf-8") as f:
        _ = next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                weights[p[0].upper()] = float(p[2])

    missing = [c for c in SENSE_CODONS if c not in weights]
    if missing:
        raise ValueError(f"CAI weight table missing {len(missing)} sense codons: {missing[:10]}")
    return weights


def calc_gc(seq: str) -> float:
    seq = clean_dna(seq)
    return float((seq.count("G") + seq.count("C")) / len(seq)) if seq else float("nan")


def calc_cai(seq: str, weights: Dict[str, float]) -> float:
    sense_codons, _ = split_cds_and_stop(seq)
    logs = []
    for codon in sense_codons:
        w = float(weights[codon])
        if w <= 0:
            raise ValueError(f"CAI weight must be > 0, got {w} for {codon}")
        logs.append(math.log(w))
    return float(math.exp(sum(logs) / len(logs))) if logs else float("nan")


def calc_rare(seq: str, weights: Dict[str, float], rare_threshold: float) -> float:
    sense_codons, _ = split_cds_and_stop(seq)
    if not sense_codons:
        return float("nan")
    return float(sum(1 for c in sense_codons if float(weights[c]) < rare_threshold) / len(sense_codons))


def calc_mfe(seq: str) -> float:
    if not HAS_RNA:
        return float("nan")
    _, mfe = RNA.fold(clean_dna(seq).replace("T", "U"))
    return float(mfe)


def codon_hamming_distance(seq1: str, seq2: str) -> int:
    c1 = split_codons(seq1)
    c2 = split_codons(seq2)
    if len(c1) != len(c2):
        return 10 ** 9
    return sum(a != b for a, b in zip(c1, c2))


def is_diverse_enough(seq: str, kept: List[str], min_codon_diff: int) -> bool:
    return all(codon_hamming_distance(seq, prev) >= min_codon_diff for prev in kept)


def generate_one_uniform_cds(
    aa_seq: str,
    rng: random.Random,
    stop_mode: str,
    fixed_stop: str,
    idx: int,
) -> str:
    codons = [rng.choice(AA2CODONS[aa]) for aa in aa_seq]
    stop = choose_stop(stop_mode, fixed_stop, idx, rng)
    return "".join(codons) + stop


def generate_random_pool(
    aa_seq: str,
    weights: Dict[str, float],
    n: int,
    seed: int,
    stop_mode: str,
    fixed_stop: str,
    rare_threshold: float,
    max_attempts: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    seen = set()
    rows = []
    attempts = 0

    while len(rows) < int(n) and attempts < int(max_attempts):
        attempts += 1

        seq = generate_one_uniform_cds(
            aa_seq=aa_seq,
            rng=rng,
            stop_mode=stop_mode,
            fixed_stop=fixed_stop,
            idx=len(rows),
        )

        if seq in seen:
            continue
        seen.add(seq)

        try:
            if translate_cds(seq) != aa_seq:
                continue

            sense_codons, stop = split_cds_and_stop(seq)

            rows.append({
                "pool_rank": len(rows) + 1,
                "proposal_index": attempts,
                "cds_sequence": seq,
                "sampling_mode": "uniform_random_synonymous",
                "stop_codon": stop,
                "length_nt": len(seq),
                "sense_codons": len(sense_codons),
                "gc": calc_gc(seq),
                "cai": calc_cai(seq, weights),
                "rare": calc_rare(seq, weights, rare_threshold),
            })

        except Exception as e:
            continue

        if len(rows) % 1000 == 0:
            print(f"[INFO] generated={len(rows)} / {n} | attempts={attempts}", flush=True)

    if len(rows) < int(n):
        raise RuntimeError(f"Only generated {len(rows)} / {n} unique sequences after {attempts} attempts.")

    return pd.DataFrame(rows)


def select_lowest_cai(
    df: pd.DataFrame,
    top_k: int,
    min_codon_diff: int,
) -> pd.DataFrame:
    work = df.sort_values(
        ["cai", "rare", "gc"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    selected_rows = []
    selected_seqs = []

    for _, row in work.iterrows():
        seq = str(row["cds_sequence"])
        if min_codon_diff > 0 and not is_diverse_enough(seq, selected_seqs, min_codon_diff):
            continue
        selected_rows.append(row.to_dict())
        selected_seqs.append(seq)
        if len(selected_rows) >= int(top_k):
            break

    # 如果 diversity 太严格导致不足 top_k，就放宽补齐
    if len(selected_rows) < int(top_k):
        used = set(selected_seqs)
        for _, row in work.iterrows():
            seq = str(row["cds_sequence"])
            if seq in used:
                continue
            selected_rows.append(row.to_dict())
            used.add(seq)
            if len(selected_rows) >= int(top_k):
                break

    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    out.insert(0, "start_rank_id", np.arange(1, len(out) + 1))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 10000 uniform random synonymous Gluc CDS and select lowest-CAI TopK."
    )

    p.add_argument("--aa_fasta", type=str, default=None)
    p.add_argument("--aa_sequence", type=str, default=None)
    p.add_argument("--use_default_gluc", action="store_true")

    p.add_argument("--cai_weight_file", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--prefix", type=str, default="Gluc_MF882921_uniformRandom10000_lowest10")

    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max_attempts", type=int, default=200000)

    p.add_argument("--rare_threshold", type=float, default=0.30)

    p.add_argument("--stop_mode", type=str, default="balanced_three", choices=["fixed", "random_three", "balanced_three"])
    p.add_argument("--stop_codon", type=str, default="TAA", choices=sorted(STOP_CODONS))

    p.add_argument("--min_codon_diff", type=int, default=8)

    p.add_argument("--calc_mfe_for_selected", action="store_true")
    p.add_argument("--calc_mfe_for_all", action="store_true")

    args = p.parse_args()

    if args.aa_sequence is None and args.aa_fasta is None and not args.use_default_gluc:
        raise ValueError("Please provide --aa_sequence, --aa_fasta, or --use_default_gluc.")

    if args.n <= 0:
        raise ValueError("--n must be positive.")

    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")

    if args.top_k > args.n:
        raise ValueError("--top_k cannot be greater than --n.")

    return args


def main() -> None:
    args = parse_args()

    if args.aa_sequence is not None:
        aa_header = "manual_input"
        aa_seq = clean_aa(args.aa_sequence)
    elif args.aa_fasta is not None:
        aa_header, aa_raw = read_first_fasta_sequence(args.aa_fasta)
        aa_seq = clean_aa(aa_raw)
    else:
        aa_header = "DEFAULT_GLUC_AA"
        aa_seq = clean_aa(DEFAULT_GLUC_AA)

    weights = load_weight_table(args.cai_weight_file)

    print("[INFO] ===== Uniform random CDS pool + lowest-CAI TopK selection =====")
    print(f"[INFO] AA source: {aa_header}")
    print(f"[INFO] AA length: {len(aa_seq)}")
    print(f"[INFO] Random pool size: {args.n}")
    print(f"[INFO] TopK selected by lowest CAI: {args.top_k}")
    print(f"[INFO] Stop mode: {args.stop_mode}")
    print("[INFO] Generation uses uniform synonymous codon sampling.")
    print("[INFO] No CAI weights are used during generation.")
    print("[INFO] CAI is only used for post-hoc ranking and TopK selection.")

    pool_df = generate_random_pool(
        aa_seq=aa_seq,
        weights=weights,
        n=int(args.n),
        seed=int(args.seed),
        stop_mode=str(args.stop_mode),
        fixed_stop=str(args.stop_codon),
        rare_threshold=float(args.rare_threshold),
        max_attempts=int(args.max_attempts),
    )

    if args.calc_mfe_for_all:
        if not HAS_RNA:
            raise RuntimeError("ViennaRNA is not available.")
        print("[INFO] Calculating MFE for all random pool sequences...")
        pool_df["mfe"] = pool_df["cds_sequence"].map(calc_mfe)
    else:
        pool_df["mfe"] = np.nan

    selected_df = select_lowest_cai(
        df=pool_df,
        top_k=int(args.top_k),
        min_codon_diff=int(args.min_codon_diff),
    )

    if args.calc_mfe_for_selected:
        if not HAS_RNA:
            raise RuntimeError("ViennaRNA is not available.")
        print("[INFO] Calculating MFE for selected TopK sequences...")
        selected_df["mfe"] = selected_df["cds_sequence"].map(calc_mfe)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_path = out_dir / f"{args.prefix}.random_pool.tsv"
    selected_path = out_dir / f"{args.prefix}.lowestCAI_top{args.top_k}.tsv"
    seq_path = out_dir / f"{args.prefix}.lowestCAI_top{args.top_k}_sequences.txt"
    summary_path = out_dir / f"{args.prefix}.summary.txt"

    pool_df.to_csv(pool_path, sep="\t", index=False)
    selected_df.to_csv(selected_path, sep="\t", index=False)

    with open(seq_path, "w", encoding="utf-8") as f:
        for seq in selected_df["cds_sequence"].astype(str).tolist():
            f.write(seq + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"AA source\t{aa_header}\n")
        f.write(f"AA length\t{len(aa_seq)}\n")
        f.write(f"random_pool_size\t{len(pool_df)}\n")
        f.write(f"top_k\t{args.top_k}\n")
        f.write(f"seed\t{args.seed}\n")
        f.write(f"generation\tuniform synonymous codon sampling; no CAI filter\n")
        f.write(f"selection\tlowest CAI TopK after generation\n\n")

        f.write("Random pool summary:\n")
        f.write(pool_df[["length_nt", "gc", "cai", "rare"]].describe().to_string())
        f.write("\n\nSelected TopK summary:\n")
        cols = ["length_nt", "gc", "cai", "rare", "mfe"]
        f.write(selected_df[[c for c in cols if c in selected_df.columns]].describe().to_string())
        f.write("\n\nSelected stop codon counts:\n")
        f.write(selected_df["stop_codon"].value_counts().sort_index().to_string())
        f.write("\n")

    print("[INFO] Saved:")
    print(f"  random pool      : {pool_path}")
    print(f"  selected TopK    : {selected_path}")
    print(f"  selected seq txt : {seq_path}")
    print(f"  summary          : {summary_path}")

    print("[INFO] Random pool CAI summary:")
    print(pool_df["cai"].describe().to_string())

    print("[INFO] Selected TopK:")
    show_cols = ["start_rank_id", "pool_rank", "cai", "gc", "rare", "mfe", "stop_codon"]
    print(selected_df[[c for c in show_cols if c in selected_df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
