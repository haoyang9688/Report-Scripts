#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HEK293T RL codon optimization (score.py-aligned scorer, score+CAI objective, center-screen version)

- training/search main objective: WEIGHT_SCORE * score_z + WEIGHT_CAI * cai_z
- center similarity is used only as a final deployment-time lower-bound screen
- training/search objective is score_z + cai_z (equal importance)
- biological center statistics are retained for logging and final screening, not penalty optimization
- center-related features are excluded from the policy/value network state input
- MFE enters the feasibility model as length-corrected residual:
    mfe_residual = raw_mfe - expected_mfe(length_nt)
- CSI is retained for logging/reporting only, but is not part of the main objective
- final export first applies a center lower-bound screen, then ranks by objective and translation score
- actor-critic training uses one-step TD / advantage: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
"""

import os
import sys
import time
import math
import json
import random
import subprocess
from copy import deepcopy
from functools import lru_cache
from collections import defaultdict, OrderedDict, deque
from typing import Dict, List, Optional, Tuple
from glob import glob

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# =========================================================
# 0. Optional ViennaRNA
# =========================================================
try:
    import RNA
    HAS_RNA = True
except ImportError:
    HAS_RNA = False

# =========================================================
# 1. Paths
# =========================================================
ORIGINAL_FILE = "/data/hyliu/HEK293T_RL/challenge_start_pool/Gluc_MF882921_uniformRandom_seed2026_top10_sequences"
MODEL_PATH = "/data/hyliu/HEK293T_RL/Models/best_model.p"
MODEL_CONFIG = "/data/hyliu/HEK293T_RL/Models/model_config.json"
HEK293T_CONDITION_NPZ = "/data/hyliu/HEK293T_RL/conditions/HEK293T_10552_RPKM.npz"

METRICS_FILE = "/data/hyliu/HEK293T_RL/metrics/HEK293T_full_high_expression_metrics.txt"
MFE_SOURCE_FILE = "/data/hyliu/HEK293T_RL/metrics/HEK293T_full_high_expression_CDS_with_MFE_only_cds_mfe.txt"
# Explicit HEK293T high-expression CDS reference for score_z / cai_z normalization.
# This avoids silently falling back to the 10 low-CAI start sequences.
NORMALIZATION_REF_CDS_FILE = MFE_SOURCE_FILE

CAI_WEIGHT_FILE = "/data/hyliu/HEK293T_RL/metrics/hek293t_codon_weights.txt"
CSI_WEIGHT_FILE = "/data/hyliu/HEK293T_RL/metrics/human_csi_weights.txt"

OUT_DIR = "/data/hyliu/HEK293T_RL/RL_results_Gluc"

# =========================================================
# 2. Global settings
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

MAX_EPISODES = 500
LR = 5e-5
DISCOUNT = 0.99
VALUE_COEF = 0.5
ENTROPY_COEF = 2e-3
PRINT_EVERY = 5
UPDATE_EVERY_STEPS = 1  # one-step TD update, aligned with the architecture diagram

OBJECTIVE_EPS = 1e-8

RARE_THRESHOLD = 0.30
USE_MFE_IN_REWARD = True

MAX_SEQ_LEN_NT = 4500
MRNA_DIM = 10552

# Keep the RL internal scorer strictly consistent with score.py.
# HEK293T_10552_RPKM.npz stores raw RPKM values.
# env_vec = log2(raw_RPKM + 1)
# scalar mRNA input = median(log2(raw_RPKM + 1))
CONDITION_KEY = "mRNA"
ENV_TRANSFORM = "log2p1"
RNA_SCALE = "rpkm"
MRNA_ABUNDANCE_STRATEGY = "median"

BIO_DEV_CLIP = 3.0
EPS = 1e-8

# main objective weights
WEIGHT_SCORE = 1.0
WEIGHT_CAI = 1.0

# adaptive constraint multipliers
MU_BIO_PRIOR_INIT = 0.0
MU_S_BIO_INIT = 0.0
MU_GC_INIT = 0.0
MU_RARE_INIT = 0.0
MU_MFE_INIT = 0.0
DUAL_LR = 0.02
DUAL_MAX = 10.0

# center lower-bound used ONLY at final deployment screening
CENTER_MIN = 0.70

# retained legacy constants (not used in the score+CAI training objective)
BIO_PRIOR_Z_MIN = 0.25
S_BIO_MIN = CENTER_MIN
DELTA_GC_MAX = 0.08
DELTA_RARE_MAX = 0.02
DELTA_MFE_RESIDUAL_MAX = 20.0

# moderately aggressive search
STEP_FRAC_OF_EDITABLE = 1.50
MIN_STEPS_PER_EPISODE = 32
MAX_STEPS_PER_EPISODE = 480
MAX_EDITS_PER_POSITION = 12

POSITION_COOLDOWN_STEPS = 0
ADAPTIVE_COOLDOWN_MIN_LEGAL_POSITIONS = 4

NO_IMPROVE_PATIENCE = 60
NEGATIVE_REWARD_PATIENCE = 999999

WARMUP_EPISODES = 10
ORIGINAL_RESTART_PROB = 0.20
RANDOM_ORIGINAL_RESTART_PROB = 0.35
GLOBAL_BEST_START_PROB = 0.20
RANDOM_GLOBAL_BEST_RESTART_PROB = 0.40

RANDOM_RESTART_EDIT_FRAC = 0.10
RANDOM_RESTART_MIN_EDITS = 2
RANDOM_RESTART_MAX_EDITS = 48

PARETO_SOFT_LIMIT = 2500
POLICY_WINDOW_CODONS = None
POLICY_FULL_ATTENTION_MAX_LEN = 2048
POLICY_CHUNK_LEN = 512

EARLY_MONITOR_EVERY_EPISODES = 100
EARLY_MONITOR_UNTIL_EPISODE = 200
EARLY_MONITOR_CANDIDATE_COUNT = 50

LATE_CHECKPOINT_CANDIDATES = {
    300: 250,
    450: 400,
}
FINAL_CANDIDATE_COUNT = 1000

CANDIDATE_DEDUP = True
CANDIDATE_MAX_ATTEMPT_MULTIPLIER = 25
CANDIDATE_TEMPERATURE = 1.45
CANDIDATE_TOPK_POS = 48
CANDIDATE_TOPK_ALT = 6
CANDIDATE_MAX_STEPS_SCALE = 1.5

FINAL_STAGE_FORCE_FILL = True
FINAL_STAGE_MAX_ATTEMPTS = 400000
FINAL_STAGE_MAX_STAGNANT_ATTEMPTS = 50000

AUTO_DISCOVER_ENSEMBLE = False
ENSEMBLE_MODEL_PATHS = [MODEL_PATH]
ENSEMBLE_UNCERTAINTY_ALPHA = 0.0

MIN_ENSEMBLE_MEMBERS = 2
ALLOW_SINGLE_MODEL_FALLBACK = True
MIN_MATCHED_TARGET_FRACTION = 0.99
WARN_MATCHED_SOURCE_FRACTION = 0.95

IGNORE_SOURCE_PREFIXES = (
    "reg_detector_list.",
    "reg_filter.",
    "reg_encoder.",
    "reg_fc.",
    "RPF_layer.",
    "TE_fc.",
    "TE_fc_count.",
    "RPF_fc_count.",
)
KEY_REMAP_RULES = {
    "mRNA_layer.4.": "mRNA_layer.3.",
    "mRNA_layer.6.": "mRNA_layer.5.",
}
REQUIRED_MODEL_PREFIXES = [
    "cds_detector_list",
    "cds_filter",
    "cds_encoder",
    "mRNA_layer",
    "attention_cds",
    "RPF_fc",
    "bias",
]
FULL_CDS_SCORE_WINDOW_NT = MAX_SEQ_LEN_NT
FULL_CDS_SCORE_STRIDE_NT = 1500
FULL_CDS_SCORE_REDUCTION = "mean"

BIO_GMM_MAX_COMPONENTS = 4
BIO_GMM_REG_COVAR = 1e-6
BIO_DENSITY_Q_LOW = 0.01
BIO_DENSITY_Q_HIGH = 0.99

# normalization reference sampling
NORM_REF_MAX_SEQS = 1000

# multi-start / multi-seed challenge setting
# 10 lowest-CAI uniform-random Gluc CDS starts × 1 run each = 10 independent RL runs
EXPECTED_START_COUNT = 10
N_RANDOM_SEEDS_PER_START = 1
RUN_SEED_BASE = SEED
FINAL_TOPK_PER_RUN = 10

# Parallel multi-start controller.
# Parent process launches one subprocess per start sequence.
# Each worker loads its own model/scorer and writes seq_XXXX outputs independently.
PARALLEL_MULTI_START = True
MAX_PARALLEL_START_JOBS = int(os.environ.get("RL_MAX_PARALLEL_START_JOBS", str(EXPECTED_START_COUNT)))
PARALLEL_LOG_DIR_NAME = "parallel_logs"
SINGLE_START_ENV = "RL_SINGLE_START_ID"

# Explicit fixed Gluc amino-acid sequence used as the synonymous codon constraint.
# The 10 input CDS starts are optimized only in the codon/CDS space;
# the target AA sequence defines the allowed synonymous codons at each position.
TARGET_AA_SEQUENCE = (
    "MGVKVLFALICIAVAEAKPTENNEDFNIVAVASNFATTDLDADRGKLPGKKLPLEVLKEM"
    "EANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEGDKESAQGGIGEAIVDIPEIPG"
    "FKDLEPMEQFIAQVDLCVDCTTGCLKGLANVQCSDLLKKWLPQRCATFASKIQGQVDKIK"
    "GAGGD"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def set_all_random_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================
# 3. Codon tables
# =========================================================
STOP_CODONS = {"TAA", "TAG", "TGA"}

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

AA2CODONS = defaultdict(list)
for codon, aa in CODON2AA.items():
    if aa != "*":
        AA2CODONS[aa].append(codon)

SENSE_CODONS = sorted([c for c, aa in CODON2AA.items() if aa != "*"])
CODON2IDX = {c: i for i, c in enumerate(SENSE_CODONS)}
IDX2CODON = {i: c for c, i in CODON2IDX.items()}
MAX_ALT = max(len(v) - 1 for v in AA2CODONS.values())

# =========================================================
# 4. Utilities
# =========================================================
def clean_seq(seq: str) -> str:
    return "".join(str(seq).split()).upper().replace("U", "T")


def clean_aa_sequence(seq: str) -> str:
    aa = "".join(str(seq).split()).upper()
    if aa.endswith("*"):
        aa = aa[:-1]
    invalid = sorted(set([x for x in aa if x not in AA2CODONS]))
    if invalid:
        raise ValueError(f"[ERROR] TARGET_AA_SEQUENCE contains invalid amino-acid symbols: {invalid}")
    return aa


def split_codons(seq: str) -> List[str]:
    seq = clean_seq(seq)
    if not seq:
        raise ValueError("Sequence is empty after cleaning")
    if len(seq) % 3 != 0:
        raise ValueError(f"Sequence length must be multiple of 3, got {len(seq)}")
    return [seq[i:i + 3] for i in range(0, len(seq), 3)]


def split_cds_and_stop(seq: str) -> Tuple[List[str], Optional[str]]:
    codons = split_codons(seq)
    if codons and codons[-1] in STOP_CODONS:
        return codons[:-1], codons[-1]
    return codons, None


def join_cds(sense_codons: List[str], stop_codon: Optional[str] = None) -> str:
    return "".join(sense_codons + ([stop_codon] if stop_codon is not None else []))


def translate_sense_codons(sense_codons: List[str]) -> str:
    aas = []
    for c in sense_codons:
        if c not in CODON2AA:
            raise ValueError(f"Unknown codon: {c}")
        aa = CODON2AA[c]
        if aa == "*":
            raise ValueError(f"Internal stop codon found in sense region: {c}")
        aas.append(aa)
    return "".join(aas)


def validate_full_cds(seq: str) -> None:
    codons = split_codons(seq)
    if len(codons) == 0:
        raise ValueError("Empty sequence")
    for i, c in enumerate(codons):
        if c not in CODON2AA:
            raise ValueError(f"Unknown codon '{c}' at codon index {i}")
        if i < len(codons) - 1 and CODON2AA[c] == "*":
            raise ValueError(f"Internal stop codon '{c}' at codon index {i}")


def synonymous_alts(codon: str) -> List[str]:
    aa = CODON2AA[codon]
    if aa == "*":
        return []
    return [c for c in AA2CODONS[aa] if c != codon]


def legal_positions(
    sense_codons: List[str],
    max_codon_index: Optional[int] = None,
    edited_counts: Optional[Dict[int, int]] = None,
    max_edits_per_position: Optional[int] = None,
    blocked_positions: Optional[List[int]] = None,
) -> List[int]:
    out = []
    blocked = set() if blocked_positions is None else set(blocked_positions)
    if max_codon_index is None:
        max_codon_index = len(sense_codons)
    for i, c in enumerate(sense_codons[:max_codon_index]):
        if i in blocked:
            continue
        if len(synonymous_alts(c)) == 0:
            continue
        if edited_counts is not None and max_edits_per_position is not None:
            if edited_counts.get(i, 0) >= max_edits_per_position:
                continue
        out.append(i)
    return out


def codon_tensor(sense_codons: List[str], device: str) -> torch.Tensor:
    idx = [CODON2IDX[c] for c in sense_codons]
    return torch.tensor(idx, dtype=torch.long, device=device)


def load_sequences_onecol(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    seqs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = clean_seq(line.strip())
            if not s:
                continue
            if len(s) % 3 != 0:
                print(f"[WARN] skip non-triplet-length sequence: len={len(s)}")
                continue
            try:
                validate_full_cds(s)
            except Exception as e:
                print(f"[WARN] skip invalid CDS: {e}")
                continue
            seqs.append(s)
    return seqs


def load_weight_table(path: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            d[parts[0].upper()] = float(parts[2])
    return d


def ensure_weight_table_complete(weights: Dict[str, float], table_name: str) -> None:
    missing = [c for c in SENSE_CODONS if c not in weights]
    if missing:
        raise ValueError(f"[ERROR] {table_name} missing {len(missing)} sense codons, examples: {missing[:10]}")


def values_to_logRNA(values: np.ndarray, scale: str = "rpkm") -> np.ndarray:
    """Convert raw RNA abundance values to log2(RPKM+1)-like model scalar inputs.

    This mirrors the standalone score.py logic. For the current HEK293T npz,
    values are raw RPKM and scale should remain 'rpkm'.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("RNA abundance array is empty after removing NaN/Inf")

    if scale == "rpkm":
        if np.any(arr < 0):
            raise ValueError("raw RNA abundance contains negative values")
        out = np.log2(arr + 1.0)
    elif scale == "log2p1":
        out = arr
    else:
        raise ValueError("RNA_SCALE must be 'rpkm' or 'log2p1'")

    if not np.all(np.isfinite(out)):
        raise ValueError("converted logRNA contains NaN or Inf")
    return out.astype(np.float32)


def choose_HEK293T_mRNA_abundance(logrna_values: np.ndarray, strategy: str = "median") -> Tuple[float, str]:
    """Derive the scalar mRNA input from the HEK293T mRNA abundance distribution.

    Default is median(log2(RPKM+1)), exactly matching the current score.py setup.
    """
    arr = np.asarray(logrna_values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("empty logRNA values for HEK293T mRNA abundance selection")

    if strategy == "median":
        val = float(np.median(arr))
    elif strategy == "mean":
        val = float(np.mean(arr))
    elif strategy == "q25":
        val = float(np.quantile(arr, 0.25))
    elif strategy == "q75":
        val = float(np.quantile(arr, 0.75))
    elif strategy == "q90":
        val = float(np.quantile(arr, 0.90))
    else:
        raise ValueError("MRNA_ABUNDANCE_STRATEGY must be one of: median, mean, q25, q75, q90")

    return val, strategy


def transform_HEK293T_env(raw_rpkm: np.ndarray, transform: str = "log2p1") -> np.ndarray:
    """Transform the fixed HEK293T raw RPKM vector into the model env_vec.

    Default is log2(RPKM+1), matching score.py.
    """
    raw = np.asarray(raw_rpkm, dtype=np.float32).reshape(-1)

    if raw.shape[0] != MRNA_DIM:
        raise ValueError(f"HEK293T env_vec dimension mismatch: got {raw.shape[0]}, expected {MRNA_DIM}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("HEK293T env vector contains NaN or Inf")
    if np.any(raw < 0):
        raise ValueError("HEK293T env vector contains negative RPKM values")

    if transform == "log2p1":
        return np.log2(raw + 1.0).astype(np.float32)
    if transform == "ribodecode":
        return np.log1p(raw * 5.0).astype(np.float32)
    if transform == "none":
        return raw.astype(np.float32)
    raise ValueError("ENV_TRANSFORM must be one of: log2p1, ribodecode, none")


def sliding_window_starts(total_len: int, window_len: int, stride: int) -> List[int]:
    if total_len <= 0:
        return [0]
    if total_len <= window_len:
        return [0]
    starts = list(range(0, max(total_len - window_len, 0) + 1, stride))
    last_start = total_len - window_len
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def choose_blocked_positions_adaptive(raw_legal_positions: List[int], recent_positions: deque) -> Optional[List[int]]:
    if POSITION_COOLDOWN_STEPS <= 0 or recent_positions is None:
        return None
    if len(raw_legal_positions) < ADAPTIVE_COOLDOWN_MIN_LEGAL_POSITIONS:
        return None
    blocked_positions = list(recent_positions)
    return blocked_positions if len(blocked_positions) > 0 else None


def compute_max_steps(editable_count: int) -> int:
    if editable_count <= 0:
        return 0
    steps = int(round(editable_count * STEP_FRAC_OF_EDITABLE))
    steps = max(MIN_STEPS_PER_EPISODE, steps)
    steps = min(MAX_STEPS_PER_EPISODE, steps)
    return steps



def similarity_to_legacy_penalty(s: float) -> float:
    return float(max(0.0, min(1.0, 1.0 - float(s))))


def write_table_txt(data, path: str) -> None:
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(path, sep="\t", index=False)


def write_json_txt(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_sequence_txt(seq: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(seq).strip() + "\n")


def _find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def load_reference_sequences_for_normalization(
    metrics_file: str,
    fallback_seqs: List[str],
    max_count: int = NORM_REF_MAX_SEQS,
    ref_cds_file: Optional[str] = None,
) -> List[str]:
    """
    Load CDS sequences used to compute score_z and cai_z reference statistics.

    Preferred behavior for this challenge run:
    1) Use an explicit HEK293T high-expression CDS reference file.
    2) Only if no explicit reference file is provided, try METRICS_FILE if it contains a CDS column.
    3) Fall back to start sequences only for legacy compatibility.

    With NORMALIZATION_REF_CDS_FILE set, this function is strict: if the explicit
    HEK293T high-expression reference file cannot provide valid CDS sequences,
    it raises an error instead of silently using the 10 low-CAI start sequences.
    """

    def _load_valid_cds_from_table(path: str, label: str) -> List[str]:
        if path is None or str(path).strip() == "":
            return []
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] {label} file not found: {path}")

        df = pd.read_csv(path, sep="\t")
        seq_col = _find_first_existing_col(df, ["cds_sequence", "sequence", "seq", "CDS", "cds"])
        if seq_col is None:
            print(f"[WARN] No CDS sequence column found in {label}: {path}", flush=True)
            return []

        seqs_local: List[str] = []
        seen = set()
        invalid_count = 0
        for raw in df[seq_col].astype(str).tolist():
            s = clean_seq(raw)
            if not s or s in seen:
                continue
            try:
                validate_full_cds(s)
            except Exception:
                invalid_count += 1
                continue
            seen.add(s)
            seqs_local.append(s)

        print(
            f"[INFO] Loaded {len(seqs_local)} valid CDS for normalization from {label} "
            f"column='{seq_col}' | invalid_skipped={invalid_count}",
            flush=True,
        )
        return seqs_local

    seqs: List[str] = []

    # Strict explicit HEK293T high-expression normalization reference.
    if ref_cds_file is not None:
        seqs = _load_valid_cds_from_table(ref_cds_file, "explicit HEK293T high-expression normalization reference")
        if len(seqs) == 0:
            raise ValueError(
                "[ERROR] Explicit NORMALIZATION_REF_CDS_FILE yielded 0 valid CDS sequences. "
                "Do not fall back to low-CAI start sequences for z-score normalization."
            )
    else:
        try:
            seqs = _load_valid_cds_from_table(metrics_file, "METRICS_FILE normalization reference")
        except Exception as e:
            print(f"[WARN] Failed to load normalization reference sequences from metrics file: {e}", flush=True)

        if len(seqs) == 0:
            print(
                "[WARN] Falling back to ORIGINAL_FILE sequences for normalization. "
                "For final analysis, prefer an explicit HEK293T high-expression CDS reference.",
                flush=True,
            )
            seqs = list(fallback_seqs)

    if len(seqs) > max_count:
        rng = random.Random(SEED)
        seqs = rng.sample(seqs, max_count)
        print(f"[INFO] Subsampled normalization reference CDS to n={len(seqs)}", flush=True)

    return seqs

# =========================================================
# 5. Metrics and bio GMM
# =========================================================
def calc_gc(seq: str) -> float:
    seq = clean_seq(seq)
    denom = sum(seq.count(x) for x in "ATGC")
    if denom == 0:
        return float("nan")
    return (seq.count("G") + seq.count("C")) / denom


def calc_geom_index(seq: str, weights: Dict[str, float], metric_name: str = "index") -> float:
    sense, _ = split_cds_and_stop(seq)
    logs = []
    for c in sense:
        if c not in weights:
            raise KeyError(f"{metric_name} weight missing for codon: {c}")
        w = weights[c]
        if w <= 0:
            raise ValueError(f"{metric_name} weight must be > 0, got {w} for codon {c}")
        logs.append(math.log(w))
    if len(logs) == 0:
        return float("nan")
    return float(math.exp(sum(logs) / len(logs)))


def calc_rare(seq: str, weights: Dict[str, float], threshold: float = 0.30) -> float:
    sense, _ = split_cds_and_stop(seq)
    vals = []
    for c in sense:
        if c not in weights:
            raise KeyError(f"rare-score weight missing for codon: {c}")
        vals.append(1.0 if weights[c] < threshold else 0.0)
    if len(vals) == 0:
        return float("nan")
    return float(sum(vals) / len(vals))


@lru_cache(maxsize=50000)
def calc_mfe(seq: str) -> float:
    if not HAS_RNA:
        raise RuntimeError("ViennaRNA not available while calc_mfe was requested")
    rna_seq = clean_seq(seq).replace("T", "U")
    _, mfe = RNA.fold(rna_seq)
    return float(mfe)


def compute_length_corrected_mfe(
    raw_mfe: float,
    length_nt: int,
    target_specs: Optional[Dict] = None,
) -> float:
    raw_mfe = float(raw_mfe)
    length_nt = int(length_nt)

    if target_specs is None:
        return raw_mfe

    slope = target_specs.get("mfe_length_slope", None)
    intercept = target_specs.get("mfe_length_intercept", None)

    if slope is None or intercept is None:
        return raw_mfe

    expected_mfe = float(slope) * float(length_nt) + float(intercept)
    return float(raw_mfe - expected_mfe)


def calc_all_metrics(
    seq: str,
    cai_weights: Dict[str, float],
    csi_weights: Dict[str, float],
    target_specs: Optional[Dict] = None,
) -> Dict[str, float]:
    seq = clean_seq(seq)
    length_nt = len(seq)

    out = {
        "length_nt": length_nt,
        "gc": calc_gc(seq),
        "csi": calc_geom_index(seq, csi_weights, metric_name="CSI"),
        "cai": calc_geom_index(seq, cai_weights, metric_name="CAI"),
        "rare": calc_rare(seq, cai_weights, threshold=RARE_THRESHOLD),
    }

    if USE_MFE_IN_REWARD:
        raw_mfe = calc_mfe(seq)
        out["mfe_raw"] = float(raw_mfe)
        out["mfe_residual"] = compute_length_corrected_mfe(
            raw_mfe=raw_mfe,
            length_nt=length_nt,
            target_specs=target_specs,
        )

    return out


def fit_bio_gmm_from_reference(metrics_file: str, mfe_file: Optional[str] = None) -> Dict:
    print(f"[INFO] Reading GMM metric table: {metrics_file}", flush=True)
    df = pd.read_csv(metrics_file, sep="\t")

    base_needed = ["gc", "rare"]
    for col in base_needed:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Column '{col}' not found in {metrics_file}")

    metric_seq_col = _find_first_existing_col(df, ["cds_sequence", "sequence", "seq", "CDS", "cds"])

    if "length_nt" not in df.columns:
        if metric_seq_col is not None:
            df["length_nt"] = df[metric_seq_col].astype(str).map(lambda x: len(clean_seq(x)))
        elif USE_MFE_IN_REWARD:
            raise ValueError(f"[ERROR] Column 'length_nt' not found in {metrics_file} and no sequence column is available")

    work = df[base_needed].copy()
    mfe_length_slope = None
    mfe_length_intercept = None
    mfe_alignment_mode = None

    if USE_MFE_IN_REWARD:
        if mfe_file is None:
            raise ValueError("[ERROR] USE_MFE_IN_REWARD=True but mfe_file is None")

        print(f"[INFO] Reading precomputed MFE table: {mfe_file}", flush=True)
        df_mfe = pd.read_csv(mfe_file, sep="\t")
        if "mfe" not in df_mfe.columns:
            raise ValueError(f"[ERROR] Column 'mfe' not found in {mfe_file}")

        mfe_seq_col = _find_first_existing_col(df_mfe, ["cds_sequence", "sequence", "seq", "CDS", "cds"])

        if metric_seq_col is not None and mfe_seq_col is not None:
            mfe_alignment_mode = "sequence_key"

            metric_merge = df[[metric_seq_col, "length_nt"] + base_needed].copy()
            metric_merge["_merge_cds"] = metric_merge[metric_seq_col].astype(str).map(clean_seq)
            metric_merge = metric_merge[metric_merge["_merge_cds"] != ""].copy()

            mfe_merge = df_mfe[[mfe_seq_col, "mfe"]].copy()
            mfe_merge["_merge_cds"] = mfe_merge[mfe_seq_col].astype(str).map(clean_seq)
            mfe_merge = mfe_merge[mfe_merge["_merge_cds"] != ""].copy()

            if metric_merge["_merge_cds"].duplicated(keep=False).any():
                dup_examples = (
                    metric_merge.loc[
                        metric_merge["_merge_cds"].duplicated(keep=False),
                        "_merge_cds"
                    ].head(5).tolist()
                )
                raise ValueError(f"[ERROR] Duplicate CDS keys found in metrics_file. Examples: {dup_examples}")

            if mfe_merge["_merge_cds"].duplicated(keep=False).any():
                dup_examples = (
                    mfe_merge.loc[
                        mfe_merge["_merge_cds"].duplicated(keep=False),
                        "_merge_cds"
                    ].head(5).tolist()
                )
                raise ValueError(f"[ERROR] Duplicate CDS keys found in mfe_file. Examples: {dup_examples}")

            merged = metric_merge.merge(
                mfe_merge[["_merge_cds", "mfe"]],
                on="_merge_cds",
                how="left",
                validate="one_to_one",
            )

            missing_mfe = int(merged["mfe"].isna().sum())
            if missing_mfe > 0:
                raise ValueError(f"[ERROR] {missing_mfe} metric rows could not be matched to MFE by cds_sequence")

            merged["length_nt"] = pd.to_numeric(merged["length_nt"], errors="coerce")
            merged["mfe"] = pd.to_numeric(merged["mfe"], errors="coerce")
            merged = merged.dropna(subset=["length_nt", "mfe"])

            x = merged["length_nt"].to_numpy(dtype=float)
            y = merged["mfe"].to_numpy(dtype=float)

            if len(np.unique(x)) < 2:
                mfe_length_slope = 0.0
                mfe_length_intercept = float(np.mean(y))
            else:
                mfe_length_slope, mfe_length_intercept = np.polyfit(x, y, 1)

            merged["mfe_residual"] = merged["mfe"] - (
                float(mfe_length_slope) * merged["length_nt"] + float(mfe_length_intercept)
            )

            work = merged[base_needed + ["mfe_residual"]].copy()

        else:
            mfe_alignment_mode = "row_order"

            if len(df) != len(df_mfe):
                raise ValueError(
                    "[ERROR] Cannot align MFE table: no shared CDS sequence column found, "
                    f"and row counts differ (metrics={len(df)}, mfe={len(df_mfe)})."
                )

            temp = df[["length_nt"] + base_needed].copy()
            temp["length_nt"] = pd.to_numeric(temp["length_nt"], errors="coerce")
            temp["mfe"] = pd.to_numeric(df_mfe["mfe"], errors="coerce")
            temp = temp.dropna(subset=["length_nt", "mfe"])

            x = temp["length_nt"].to_numpy(dtype=float)
            y = temp["mfe"].to_numpy(dtype=float)

            if len(np.unique(x)) < 2:
                mfe_length_slope = 0.0
                mfe_length_intercept = float(np.mean(y))
            else:
                mfe_length_slope, mfe_length_intercept = np.polyfit(x, y, 1)

            temp["mfe_residual"] = temp["mfe"] - (
                float(mfe_length_slope) * temp["length_nt"] + float(mfe_length_intercept)
            )

            work = temp[base_needed + ["mfe_residual"]].copy()

        print(
            f"[INFO] Length-corrected MFE enabled | mode={mfe_alignment_mode} | "
            f"slope={float(mfe_length_slope):.6f} | intercept={float(mfe_length_intercept):.6f}",
            flush=True
        )

    work = work.apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 20:
        raise ValueError(f"[ERROR] Too few valid rows for GMM after dropna: {len(work)}")

    feature_names = list(work.columns)
    X = work[feature_names].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_components = max(1, min(BIO_GMM_MAX_COMPONENTS, len(Xs) // 20, len(Xs)))
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=BIO_GMM_REG_COVAR,
        random_state=SEED,
    )
    gmm.fit(Xs)

    ll = gmm.score_samples(Xs)
    lo = float(np.quantile(ll, BIO_DENSITY_Q_LOW))
    hi = float(np.quantile(ll, BIO_DENSITY_Q_HIGH))
    if hi <= lo:
        hi = lo + EPS

    return {
        "feature_names": feature_names,
        "scaler": scaler,
        "gmm": gmm,
        "ll_low": lo,
        "ll_high": hi,
        "train_loglik_mean": float(np.mean(ll)),
        "train_loglik_std": float(np.std(ll, ddof=0)),
        "mfe_length_slope": None if mfe_length_slope is None else float(mfe_length_slope),
        "mfe_length_intercept": None if mfe_length_intercept is None else float(mfe_length_intercept),
        "mfe_alignment_mode": mfe_alignment_mode,
    }


def bio_density_score(metric_dict: Dict[str, float], bio_dist_model: Dict) -> float:
    feature_names = bio_dist_model["feature_names"]
    x = np.asarray([[metric_dict[k] for k in feature_names]], dtype=float)
    xs = bio_dist_model["scaler"].transform(x)
    ll = float(bio_dist_model["gmm"].score_samples(xs)[0])
    lo = float(bio_dist_model["ll_low"])
    hi = float(bio_dist_model["ll_high"])
    s = (ll - lo) / max(hi - lo, EPS)
    return float(max(0.0, min(1.0, s)))


def bio_loglik(metric_dict: Dict[str, float], bio_dist_model: Dict) -> float:
    feature_names = bio_dist_model["feature_names"]
    x = np.asarray([[metric_dict[k] for k in feature_names]], dtype=float)
    xs = bio_dist_model["scaler"].transform(x)
    return float(bio_dist_model["gmm"].score_samples(xs)[0])


def bio_prior_zscore(metric_dict: Dict[str, float], bio_dist_model: Dict) -> float:
    ll = bio_loglik(metric_dict, bio_dist_model)
    mu = float(bio_dist_model.get("train_loglik_mean", 0.0))
    sd = max(float(bio_dist_model.get("train_loglik_std", 1.0)), EPS)
    return float((ll - mu) / sd)


def score_to_z(score: float, score_ref_stats: Dict[str, float]) -> float:
    mu = float(score_ref_stats.get("mean", 0.0))
    sd = max(float(score_ref_stats.get("std", 1.0)), EPS)
    return float((float(score) - mu) / sd)


def compute_reference_score_stats(seqs: List[str], scorer) -> Dict[str, float]:
    vals = np.asarray([float(scorer.score(seq)) for seq in seqs], dtype=float)
    if len(vals) == 0:
        return {"mean": 0.0, "std": 1.0, "count": 0}
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0)) if len(vals) > 1 else 1.0
    if (not np.isfinite(sd)) or sd < EPS:
        sd = 1.0
    return {"mean": mu, "std": sd, "count": int(len(vals))}


def cai_to_z(cai: float, cai_ref_stats: Dict[str, float]) -> float:
    mu = float(cai_ref_stats.get("mean", 0.0))
    sd = max(float(cai_ref_stats.get("std", 1.0)), EPS)
    return float((float(cai) - mu) / sd)


def compute_reference_cai_stats(seqs: List[str], cai_weights: Dict[str, float]) -> Dict[str, float]:
    vals = np.asarray([float(calc_geom_index(seq, cai_weights, metric_name="CAI")) for seq in seqs], dtype=float)
    if len(vals) == 0:
        return {"mean": 0.0, "std": 1.0, "count": 0}
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0)) if len(vals) > 1 else 1.0
    if (not np.isfinite(sd)) or sd < EPS:
        sd = 1.0
    return {"mean": mu, "std": sd, "count": int(len(vals))}

# =========================================================
# 6. Frozen translation predictor
# =========================================================
class CNN_Encoder_Model(nn.Module):
    def __init__(self, model_config: Dict, input_shapes: List[List[int]]):
        super().__init__()
        _ = input_shapes
        self.model_config = model_config
        self.motif_depth = 4
        filters1 = 256
        dense4 = 512
        dense5 = 128
        dense6 = 32

        self.cds_detector_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=4, out_channels=filters1, kernel_size=5),
                nn.ReLU(),
                nn.BatchNorm1d(filters1),
            )
            for _ in range(self.motif_depth)
        ])
        self.cds_filter = nn.Sequential(nn.BatchNorm1d(filters1), nn.ReLU())
        self.cds_encoder = nn.Sequential(
            nn.Conv1d(in_channels=filters1, out_channels=64, kernel_size=30),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(1379),
            nn.Flatten(),
        )
        cds_flatten_size = self.cds_conv_shape()
        self.RPF_fc = nn.Sequential(
            nn.Linear(in_features=cds_flatten_size, out_features=dense5),
            nn.ReLU(),
            nn.BatchNorm1d(dense5),
            nn.Dropout(float(model_config.get("dropout5", 0.5))),
            nn.Linear(in_features=dense5, out_features=1),
        )
        self.attention_cds = nn.Sequential(
            nn.Linear(in_features=dense6 + 1, out_features=filters1 * self.motif_depth),
            nn.Tanh(),
        )
        self.mRNA_layer = nn.Sequential(
            nn.Linear(in_features=MRNA_DIM, out_features=dense4),
            nn.ReLU(),
            nn.BatchNorm1d(dense4),
            nn.Linear(in_features=dense4, out_features=dense6),
            nn.ReLU(),
            nn.BatchNorm1d(dense6),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def cds_motif_detection(self, sequence_input: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        results = [motif_detector(sequence_input) for motif_detector in self.cds_detector_list]
        avg_filter = torch.exp(a)
        features = torch.sum(torch.stack(results, 3) * avg_filter, 3) / torch.sum(avg_filter, 3)
        return self.cds_filter(features)

    def cds_conv_shape(self) -> int:
        x_input_1 = torch.zeros(1, 4, 4500)
        x_output = self.cds_motif_detection(x_input_1, torch.zeros([1, 256, 1, self.motif_depth]))
        x_output = self.cds_encoder(x_output)
        return int(np.prod(x_output.size()))

    def forward(self, cds_sequence: torch.Tensor, mRNA_array: torch.Tensor, mRNA_count: torch.Tensor) -> torch.Tensor:
        mRNA_features = self.mRNA_layer(mRNA_array)
        attention_cds = torch.reshape(
            self.attention_cds(torch.concat([mRNA_features, mRNA_count], 1)),
            [-1, 256, 1, self.motif_depth],
        )
        cds_output = self.cds_motif_detection(cds_sequence, attention_cds)
        cds_seq_features = self.cds_encoder(cds_output)
        return self.RPF_fc(cds_seq_features) * mRNA_count + self.bias


class FrozenTranslationScorer:
    input_shape = [[4, 5000], [4, 4500], [10552], [1], [1]]

    def __init__(self, model_config_path: str, model_path: str, condition_npz: str, device: str):
        self.device = device
        self.model_path = os.path.abspath(model_path)

        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"[ERROR] model_config not found: {model_config_path}")
        with open(model_config_path, "r", encoding="utf-8") as f:
            try:
                self.model_config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"[ERROR] Invalid JSON in model_config: {model_config_path} | {e}") from e

        self.model = CNN_Encoder_Model(model_config=self.model_config, input_shapes=self.input_shape)

        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            src_state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            src_state = ckpt
        else:
            raise TypeError(f"[ERROR] Unsupported checkpoint format for {model_path}")
        dst_state = self.model.state_dict()

        raw_state = src_state
        src_state = {}
        for k, v in raw_state.items():
            if k.startswith(IGNORE_SOURCE_PREFIXES):
                continue
            new_k = k
            for old_prefix, new_prefix in KEY_REMAP_RULES.items():
                if new_k.startswith(old_prefix):
                    new_k = new_prefix + new_k[len(old_prefix):]
                    break
            src_state[new_k] = v

        matched = {}
        skipped = []
        for k, v in src_state.items():
            if (k in dst_state) and (tuple(dst_state[k].shape) == tuple(v.shape)):
                matched[k] = v
            else:
                skipped.append(k)

        missing, unexpected = self.model.load_state_dict(matched, strict=False)
        src_total = max(len(src_state), 1)
        dst_total = max(len(dst_state), 1)
        matched_src_fraction = float(len(matched) / src_total)
        matched_dst_fraction = float((dst_total - len(missing)) / dst_total)
        matched_prefixes = {k.split(".")[0] for k in matched.keys()}
        missing_required_prefixes = [p for p in REQUIRED_MODEL_PREFIXES if p not in matched_prefixes]
        if len(missing_required_prefixes) > 0:
            raise RuntimeError(f"[ERROR] Predictor checkpoint missing required module prefixes for {self.model_path}: {missing_required_prefixes}")
        if matched_dst_fraction < MIN_MATCHED_TARGET_FRACTION:
            raise RuntimeError(f"[ERROR] Predictor checkpoint target coverage too low for {self.model_path}")
        if matched_src_fraction < WARN_MATCHED_SOURCE_FRACTION:
            print(f"[WARN] Predictor checkpoint source coverage is low for {self.model_path}: {matched_src_fraction:.3f}")
        if len(skipped) > 0:
            print(f"[WARN] skipped source tensors (show up to 20): {skipped[:20]}")
        if len(unexpected) > 0:
            print(f"[WARN] unexpected tensors (show up to 20): {list(unexpected)[:20]}")

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.mRNA_array, self.hek_mRNA_logRNA, self.condition_meta = self._load_condition_npz(condition_npz)
        self.mRNA_array = self.mRNA_array.to(self.device)
        self.mRNA_count = torch.tensor([[self.hek_mRNA_logRNA]], dtype=torch.float32, device=self.device)

        print(
            f"[INFO] RL scorer condition logic aligned with score.py | "
            f"env_transform={ENV_TRANSFORM} | "
            f"hek_mRNA_logRNA={self.hek_mRNA_logRNA:.6f} | "
            f"hek_mRNA_RPKM={self.condition_meta['hek_mRNA_RPKM']:.6f}",
            flush=True,
        )

    def _load_condition_npz(self, npz_path: str) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        npzfile = np.load(npz_path, allow_pickle=True)

        if CONDITION_KEY in npzfile.files:
            raw = npzfile[CONDITION_KEY].astype(np.float32).reshape(-1)
            source_key = CONDITION_KEY
        elif len(npzfile.files) == 1:
            source_key = npzfile.files[0]
            raw = npzfile[source_key].astype(np.float32).reshape(-1)
            print(
                f"[WARN] CONDITION_KEY={CONDITION_KEY!r} not found; using only available key {source_key!r}",
                flush=True,
            )
        else:
            raise KeyError(
                f"CONDITION_KEY={CONDITION_KEY!r} not found in {npz_path}. "
                f"Available keys: {npzfile.files}"
            )

        if raw.shape[0] != MRNA_DIM:
            raise ValueError(f"HEK293T condition vector length mismatch: got {raw.shape[0]}, expected {MRNA_DIM}")
        if not np.all(np.isfinite(raw)):
            raise ValueError("HEK293T condition vector contains NaN or Inf")
        if np.any(raw < 0):
            raise ValueError("HEK293T condition vector contains negative RPKM values")

        # Same as score.py:
        # env_vec = log2(raw RPKM + 1) when ENV_TRANSFORM='log2p1'.
        env_vec = transform_HEK293T_env(raw, transform=ENV_TRANSFORM)

        # Same as score.py:
        # scalar mRNA input = median(log2(raw RPKM + 1)) when strategy='median'.
        logrna_values = values_to_logRNA(raw, scale=RNA_SCALE)
        hek_mRNA_logRNA, strategy_name = choose_HEK293T_mRNA_abundance(
            logrna_values,
            strategy=MRNA_ABUNDANCE_STRATEGY,
        )

        hek_mRNA_RPKM = float((2.0 ** hek_mRNA_logRNA) - 1.0)

        meta = {
            "source_key": source_key,
            "raw_min": float(np.min(raw)),
            "raw_median": float(np.median(raw)),
            "raw_mean": float(np.mean(raw)),
            "raw_max": float(np.max(raw)),
            "env_min": float(np.min(env_vec)),
            "env_median": float(np.median(env_vec)),
            "env_mean": float(np.mean(env_vec)),
            "env_max": float(np.max(env_vec)),
            "logRNA_median": float(np.median(logrna_values)),
            "logRNA_mean": float(np.mean(logrna_values)),
            "hek_mRNA_logRNA": float(hek_mRNA_logRNA),
            "hek_mRNA_RPKM": float(hek_mRNA_RPKM),
            "hek_mRNA_n_genes": float(logrna_values.size),
        }

        print(f"[INFO] loaded HEK293T condition for RL scorer: {npz_path}::{source_key}", flush=True)
        print(
            f"[INFO] raw RPKM stats: min={meta['raw_min']:.6f} "
            f"median={meta['raw_median']:.6f} mean={meta['raw_mean']:.6f} "
            f"max={meta['raw_max']:.6f}",
            flush=True,
        )
        print(
            f"[INFO] env_transform={ENV_TRANSFORM} | "
            f"env median={meta['env_median']:.6f} mean={meta['env_mean']:.6f}",
            flush=True,
        )
        print(
            f"[INFO] HEK293T-derived scalar mRNA abundance = {hek_mRNA_logRNA:.6f} "
            f"by {strategy_name}; raw RNA equivalent={hek_mRNA_RPKM:.6f}; "
            f"n_genes={logrna_values.size}",
            flush=True,
        )

        env_vec = np.expand_dims(env_vec.astype(np.float32), 0)
        return torch.tensor(env_vec, dtype=torch.float32), float(hek_mRNA_logRNA), meta

    def _encode_cds(self, seq: str) -> torch.Tensor:
        seq = clean_seq(seq)
        if len(seq) < MAX_SEQ_LEN_NT:
            seq = seq + "N" * (MAX_SEQ_LEN_NT - len(seq))
        else:
            seq = seq[:MAX_SEQ_LEN_NT]
        dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        dna_I = np.eye(len(dna_vocab), dtype=np.float32)
        arr = np.array([dna_I[dna_vocab[ch]] for ch in seq], dtype=np.float32)
        arr = np.expand_dims(arr, 0)
        arr = torch.tensor(arr, dtype=torch.float32, device=self.device)
        arr = torch.transpose(arr, 2, 1)[:, :4, :]
        return arr

    def _window_starts_for_long_seq(self, seq_len_nt: int) -> List[int]:
        raw_starts = sliding_window_starts(seq_len_nt, FULL_CDS_SCORE_WINDOW_NT, FULL_CDS_SCORE_STRIDE_NT)
        starts = []
        max_start = max(0, seq_len_nt - FULL_CDS_SCORE_WINDOW_NT)
        max_start = (max_start // 3) * 3
        for s in raw_starts:
            s = min(s, max_start)
            s = (s // 3) * 3
            if len(starts) == 0 or s != starts[-1]:
                starts.append(s)
        return starts

    @torch.no_grad()
    def score(self, seq: str) -> float:
        seq = clean_seq(seq)
        if len(seq) <= FULL_CDS_SCORE_WINDOW_NT:
            cds_sequence = self._encode_cds(seq)
            prediction = self.model(cds_sequence, self.mRNA_array, self.mRNA_count)
            return float(prediction.squeeze().item())
        starts = self._window_starts_for_long_seq(len(seq))
        window_scores = []
        for s in starts:
            cds_sequence = self._encode_cds(seq[s:s + FULL_CDS_SCORE_WINDOW_NT])
            prediction = self.model(cds_sequence, self.mRNA_array, self.mRNA_count)
            window_scores.append(float(prediction.squeeze().item()))
        if len(window_scores) == 0:
            raise RuntimeError("[ERROR] No windows produced for long-sequence scoring")
        if FULL_CDS_SCORE_REDUCTION == "mean":
            return float(np.mean(window_scores))
        if FULL_CDS_SCORE_REDUCTION == "max":
            return float(np.max(window_scores))
        raise ValueError(f"Unsupported FULL_CDS_SCORE_REDUCTION: {FULL_CDS_SCORE_REDUCTION}")


class EnsembleTranslationScorer:
    def __init__(self, model_config_path: str, model_paths: List[str], condition_npz: str, device: str, uncertainty_alpha: float = ENSEMBLE_UNCERTAINTY_ALPHA):
        self.members = [FrozenTranslationScorer(model_config_path, mp, condition_npz, device) for mp in model_paths]
        self.uncertainty_alpha = float(uncertainty_alpha)
        print(f"[INFO] ensemble members: {len(self.members)} | alpha={self.uncertainty_alpha:.4f}")

    @torch.no_grad()
    def score(self, seq: str) -> float:
        vals = np.asarray([m.score(seq) for m in self.members], dtype=float)
        return float(np.mean(vals) - self.uncertainty_alpha * np.std(vals, ddof=0))

# =========================================================
# 7. RL environment (global-only)
# =========================================================
class SynonymousEnv:
    def __init__(
        self,
        original_seq: str,
        scorer,
        target_specs: Dict[str, Dict[str, float]],
        cai_weights: Dict[str, float],
        csi_weights: Dict[str, float],
        score_ref_stats: Dict[str, float],
        cai_ref_stats: Dict[str, float],
        target_aa_sequence: Optional[str] = None,
    ):
        self.original_seq = clean_seq(original_seq)
        self.orig_sense, self.stop_codon = split_cds_and_stop(self.original_seq)
        self.input_start_protein = translate_sense_codons(self.orig_sense)
        if target_aa_sequence is None:
            self.target_protein = self.input_start_protein
        else:
            self.target_protein = clean_aa_sequence(target_aa_sequence)
            if len(self.target_protein) != len(self.orig_sense):
                raise ValueError(
                    f"[ERROR] TARGET_AA_SEQUENCE length ({len(self.target_protein)}) "
                    f"does not match start CDS sense codon count ({len(self.orig_sense)})."
                )
        # Use the explicit target AA as the synonymous-search constraint.
        # No extra start-CDS-vs-target-AA equality check is performed here.
        self.orig_protein = self.target_protein
        self.scorer = scorer
        self.target_specs = target_specs
        self.cai_weights = cai_weights
        self.csi_weights = csi_weights
        self.score_ref_stats = score_ref_stats
        self.cai_ref_stats = cai_ref_stats
        self.policy_window_codons = len(self.orig_sense) if POLICY_WINDOW_CODONS is None else min(len(self.orig_sense), POLICY_WINDOW_CODONS)

        # original-sequence reference metrics for delta-constraints
        self.orig_metrics = calc_all_metrics(
            self.original_seq,
            cai_weights=self.cai_weights,
            csi_weights=self.csi_weights,
            target_specs=self.target_specs,
        )

        # adaptive dual variables
        self.mu_bio_prior = float(MU_BIO_PRIOR_INIT)
        self.mu_s_bio = float(MU_S_BIO_INIT)
        self.mu_gc = float(MU_GC_INIT)
        self.mu_rare = float(MU_RARE_INIT)
        self.mu_mfe = float(MU_MFE_INIT)

        self._reset_bookkeeping()
        self.reset()

    def _reset_bookkeeping(self) -> None:
        self.step_id = 0
        self.edited_counts: Dict[int, int] = defaultdict(int)
        self.recent_positions = deque(maxlen=POSITION_COOLDOWN_STEPS if POSITION_COOLDOWN_STEPS > 0 else None)
        self.no_improve_steps = 0
        self.negative_reward_steps = 0
        self.done_reason = None

    def _compute_objective(
        self,
        score_z: float,
        cai_z: float,
        bio_prior_z: float,
        s_bio: float,
        delta_gc: float,
        delta_rare: float,
        delta_mfe_residual: float,
    ) -> float:
        return float(compute_objective(
            score_z=score_z,
            cai_z=cai_z,
            bio_prior_z=bio_prior_z,
            s_bio=s_bio,
            delta_gc=delta_gc,
            delta_rare=delta_rare,
            delta_mfe_residual=delta_mfe_residual,
            mu_bio_prior=self.mu_bio_prior,
            mu_s_bio=self.mu_s_bio,
            mu_gc=self.mu_gc,
            mu_rare=self.mu_rare,
            mu_mfe=self.mu_mfe,
        ))

    def _validate_start_seq(self, seq: str) -> List[str]:
        seq = clean_seq(seq)
        validate_full_cds(seq)
        sense, stop_codon = split_cds_and_stop(seq)
        if len(sense) != len(self.target_protein):
            raise ValueError("Start sequence sense-codon length differs from TARGET_AA_SEQUENCE length")
        if stop_codon != self.stop_codon:
            raise ValueError("Start sequence stop codon differs from original_seq")
        return sense

    def _evaluate_seq(self, seq: str) -> Dict[str, float]:
        raw_score = float(self.scorer.score(seq))
        score_z = float(score_to_z(raw_score, self.score_ref_stats))
        metrics = calc_all_metrics(
            seq,
            self.cai_weights,
            self.csi_weights,
            target_specs=self.target_specs,
        )
        cai_z = float(cai_to_z(metrics["cai"], self.cai_ref_stats))
        s_bio = float(bio_density_score(metrics, self.target_specs))
        bio_ll = float(bio_loglik(metrics, self.target_specs))
        bio_prior = float(bio_prior_zscore(metrics, self.target_specs))
        delta_gc = float(metrics["gc"] - self.orig_metrics["gc"])
        delta_rare = float(metrics["rare"] - self.orig_metrics["rare"])
        if "mfe_residual" in metrics and "mfe_residual" in self.orig_metrics:
            delta_mfe_residual = float(metrics["mfe_residual"] - self.orig_metrics["mfe_residual"])
        else:
            delta_mfe_residual = 0.0
        objective = float(self._compute_objective(
            score_z=score_z,
            cai_z=cai_z,
            bio_prior_z=bio_prior,
            s_bio=s_bio,
            delta_gc=delta_gc,
            delta_rare=delta_rare,
            delta_mfe_residual=delta_mfe_residual,
        ))
        return {
            "score": raw_score,
            "score_z": score_z,
            "cai_z": cai_z,
            "s_bio": s_bio,
            "bio_loglik": bio_ll,
            "bio_prior_z": bio_prior,
            "delta_gc": delta_gc,
            "delta_rare": delta_rare,
            "delta_mfe_residual": delta_mfe_residual,
            "metrics": metrics,
            "objective": objective,
        }

    def _update_dual_variables(self, cur: Dict[str, float]) -> None:
        # Center is only used at final deployment screening in this version.
        # No adaptive constraint multipliers are updated during training.
        return None

    def _constraint_violations(self, cur: Dict[str, float]) -> Dict[str, float]:
        return {
            "bio_prior": max(0.0, BIO_PRIOR_Z_MIN - float(cur["bio_prior_z"])),
            "s_bio": max(0.0, S_BIO_MIN - float(cur["s_bio"])),
            "gc": max(0.0, abs(float(cur["delta_gc"])) - DELTA_GC_MAX),
            "rare": max(0.0, float(cur["delta_rare"]) - DELTA_RARE_MAX),
            "mfe": max(0.0, abs(float(cur["delta_mfe_residual"])) - DELTA_MFE_RESIDUAL_MAX),
        }

    def get_legal_positions(self) -> List[int]:
        out = []
        max_codon_index = self.policy_window_codons
        raw_positions = list(range(min(len(self.sense), max_codon_index)))
        blocked = choose_blocked_positions_adaptive(raw_positions, self.recent_positions)
        blocked_set = set() if blocked is None else set(blocked)

        for i, codon in enumerate(self.sense[:max_codon_index]):
            if i in blocked_set:
                continue
            if self.edited_counts.get(i, 0) >= MAX_EDITS_PER_POSITION:
                continue
            target_aa = self.target_protein[i]
            alts = [c for c in AA2CODONS[target_aa] if c != codon]
            if len(alts) == 0:
                continue
            out.append(i)
        return out

    def get_legal_alts(self, pos: int) -> List[str]:
        target_aa = self.target_protein[pos]
        return [c for c in AA2CODONS[target_aa] if c != self.sense[pos]]

    def reset(self, start_seq: Optional[str] = None) -> Dict:
        self._reset_bookkeeping()
        # reset adaptive dual variables at the beginning of each episode to keep
        # the optimization landscape comparable across episodes
        self.mu_bio_prior = float(MU_BIO_PRIOR_INIT)
        self.mu_s_bio = float(MU_S_BIO_INIT)
        self.mu_gc = float(MU_GC_INIT)
        self.mu_rare = float(MU_RARE_INIT)
        self.mu_mfe = float(MU_MFE_INIT)

        self.sense = self.orig_sense.copy() if start_seq is None else self._validate_start_seq(start_seq)
        self.curr_seq = join_cds(self.sense, self.stop_codon)
        cur = self._evaluate_seq(self.curr_seq)
        self.curr_score = float(cur["score"])
        self.curr_score_z = float(cur["score_z"])
        self.curr_cai_z = float(cur["cai_z"])
        self.curr_s_bio = float(cur["s_bio"])
        self.curr_bio_loglik = float(cur["bio_loglik"])
        self.curr_bio_prior_z = float(cur["bio_prior_z"])
        self.curr_delta_gc = float(cur["delta_gc"])
        self.curr_delta_rare = float(cur["delta_rare"])
        self.curr_delta_mfe_residual = float(cur["delta_mfe_residual"])
        self.curr_metrics = dict(cur["metrics"])
        self.curr_objective = float(cur["objective"])

        self.initial_score = self.curr_score
        self.initial_score_z = self.curr_score_z
        self.initial_cai_z = self.curr_cai_z
        self.initial_s_bio = self.curr_s_bio
        self.initial_bio_loglik = self.curr_bio_loglik
        self.initial_bio_prior_z = self.curr_bio_prior_z
        self.initial_objective = self.curr_objective
        self.initial_metrics = dict(self.curr_metrics)

        self.best_seq = self.curr_seq
        self.best_score = self.curr_score
        self.best_score_z = self.curr_score_z
        self.best_cai_z = self.curr_cai_z
        self.best_s_bio = self.curr_s_bio
        self.best_bio_loglik = self.curr_bio_loglik
        self.best_bio_prior_z = self.curr_bio_prior_z
        self.best_objective = self.curr_objective
        self.best_metrics = dict(self.curr_metrics)

        self.max_steps = compute_max_steps(len(self.get_legal_positions()))
        return self.get_state()

    def get_state(self) -> Dict:
        return {
            "sense_codons": self.sense.copy(),
            "policy_sense_codons": self.sense[:self.policy_window_codons].copy(),
            "full_seq": self.curr_seq,
            "score": self.curr_score,
            "score_z": self.curr_score_z,
            "cai_z": self.curr_cai_z,
            "s_bio": self.curr_s_bio,
            "bio_loglik": self.curr_bio_loglik,
            "bio_prior_z": self.curr_bio_prior_z,
            "p_bio": similarity_to_legacy_penalty(self.curr_s_bio),
            "objective": self.curr_objective,
            "current_delta_gc": self.curr_delta_gc,
            "current_delta_rare": self.curr_delta_rare,
            "current_delta_mfe_residual": self.curr_delta_mfe_residual,
            "initial_score": self.initial_score,
            "initial_score_z": self.initial_score_z,
            "initial_cai_z": self.initial_cai_z,
            "initial_s_bio": self.initial_s_bio,
            "initial_bio_prior_z": self.initial_bio_prior_z,
            "initial_objective": self.initial_objective,
            "best_score": self.best_score,
            "best_score_z": self.best_score_z,
            "best_cai_z": self.best_cai_z,
            "best_s_bio": self.best_s_bio,
            "best_bio_prior_z": self.best_bio_prior_z,
            "best_objective": self.best_objective,
            "step_id": self.step_id,
            "max_steps": self.max_steps,
            "mu_bio_prior": self.mu_bio_prior,
            "mu_s_bio": self.mu_s_bio,
            "mu_gc": self.mu_gc,
            "mu_rare": self.mu_rare,
            "mu_mfe": self.mu_mfe,
        }

    def step(self, pos: int, alt_codon: str) -> Tuple[Dict, float, bool, Dict]:
        if pos not in self.get_legal_positions():
            raise ValueError(f"illegal edit position: {pos}")
        if alt_codon not in self.get_legal_alts(pos):
            raise ValueError("illegal synonymous edit")

        old_score = self.curr_score
        old_score_z = self.curr_score_z
        old_cai_z = self.curr_cai_z
        old_s_bio = self.curr_s_bio
        old_bio_prior_z = self.curr_bio_prior_z
        old_objective = self.curr_objective
        old_metrics = dict(self.curr_metrics)
        old_seq = self.curr_seq
        old_codon = self.sense[pos]
        aa = self.target_protein[pos]

        self.sense[pos] = alt_codon
        self.edited_counts[pos] += 1
        if POSITION_COOLDOWN_STEPS > 0:
            self.recent_positions.append(pos)

        self.curr_seq = join_cds(self.sense, self.stop_codon)
        cur = self._evaluate_seq(self.curr_seq)
        self.curr_score = float(cur["score"])
        self.curr_score_z = float(cur["score_z"])
        self.curr_cai_z = float(cur["cai_z"])
        self.curr_s_bio = float(cur["s_bio"])
        self.curr_bio_loglik = float(cur["bio_loglik"])
        self.curr_bio_prior_z = float(cur["bio_prior_z"])
        self.curr_delta_gc = float(cur["delta_gc"])
        self.curr_delta_rare = float(cur["delta_rare"])
        self.curr_delta_mfe_residual = float(cur["delta_mfe_residual"])
        self.curr_metrics = dict(cur["metrics"])
        self.step_id += 1

        delta_t = float(self.curr_score - old_score)
        delta_score_z = float(self.curr_score_z - old_score_z)
        delta_cai_z = float(self.curr_cai_z - old_cai_z)
        delta_bio = float(self.curr_s_bio - old_s_bio)
        delta_bio_prior_z = float(self.curr_bio_prior_z - old_bio_prior_z)

        self.curr_objective = float(cur["objective"])
        reward = float(self.curr_objective - old_objective)

        improved = False
        if self.curr_objective > self.best_objective + OBJECTIVE_EPS:
            improved = True
            self.best_objective = self.curr_objective
            self.best_score = self.curr_score
            self.best_score_z = self.curr_score_z
            self.best_cai_z = self.curr_cai_z
            self.best_s_bio = self.curr_s_bio
            self.best_bio_loglik = self.curr_bio_loglik
            self.best_bio_prior_z = self.curr_bio_prior_z
            self.best_seq = self.curr_seq
            self.best_metrics = dict(self.curr_metrics)
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1


        if reward < 0:
            self.negative_reward_steps += 1
        else:
            self.negative_reward_steps = 0

        done = False
        if self.step_id >= self.max_steps:
            done = True
            self.done_reason = "max_steps"
        elif len(self.get_legal_positions()) == 0:
            done = True
            self.done_reason = "no_legal_actions"
        elif self.no_improve_steps >= NO_IMPROVE_PATIENCE:
            done = True
            self.done_reason = "no_improve_patience"
        elif self.negative_reward_steps >= NEGATIVE_REWARD_PATIENCE:
            done = True
            self.done_reason = "negative_reward_patience"

        info = {
            "delta_t": delta_t,
            "delta_score_z": delta_score_z,
            "delta_cai_z": delta_cai_z,
            "delta_bio": delta_bio,
            "delta_bio_prior_z": delta_bio_prior_z,
            "reward": reward,
            "metrics": dict(self.curr_metrics),
            "prev_metrics": old_metrics,
            "init_metrics": dict(self.initial_metrics),
            "best_metrics": dict(self.best_metrics),
            "current_score": self.curr_score,
            "current_score_z": self.curr_score_z,
            "current_cai_z": self.curr_cai_z,
            "current_s_bio": self.curr_s_bio,
            "current_bio_loglik": self.curr_bio_loglik,
            "current_bio_prior_z": self.curr_bio_prior_z,
            "current_p_bio": similarity_to_legacy_penalty(self.curr_s_bio),
            "current_objective": self.curr_objective,
            "current_delta_gc": self.curr_delta_gc,
            "current_delta_rare": self.curr_delta_rare,
            "current_delta_mfe_residual": self.curr_delta_mfe_residual,
            "best_score": self.best_score,
            "best_score_z": self.best_score_z,
            "best_cai_z": self.best_cai_z,
            "best_s_bio": self.best_s_bio,
            "best_bio_loglik": self.best_bio_loglik,
            "best_bio_prior_z": self.best_bio_prior_z,
            "best_p_bio": similarity_to_legacy_penalty(self.best_s_bio),
            "best_objective": self.best_objective,
            "current_seq": self.curr_seq,
            "best_seq": self.best_seq,
            "prev_seq": old_seq,
            "edit_pos": int(pos),
            "old_codon": old_codon,
            "new_codon": alt_codon,
            "aa": aa,
            "edited_count_at_pos": int(self.edited_counts[pos]),
            "legal_alt_count": int(len(self.get_legal_alts(pos))),
            "improved": improved,
            "mu_bio_prior": self.mu_bio_prior,
            "mu_s_bio": self.mu_s_bio,
            "mu_gc": self.mu_gc,
            "mu_rare": self.mu_rare,
            "mu_mfe": self.mu_mfe,
            "done_reason": self.done_reason,
        }
        return self.get_state(), reward, done, info

# =========================================================
# 8. Actor-Critic policy
# =========================================================
def mask_logits(logits: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
    x = logits.clone()
    x[~mask_bool] = -1e9
    return x


class ActorCriticPolicy(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, nlayers: int = 2, max_len: int = 2048, max_alt: int = 5, n_state_features: int = 6):
        super().__init__()
        self.max_alt = max_alt
        self.chunk_len = POLICY_CHUNK_LEN
        self.codon_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.state_proj = nn.Sequential(
            nn.Linear(n_state_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.alt_head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, max_alt))
        self.value_head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def encode(self, codon_idx: torch.Tensor) -> torch.Tensor:
        L = codon_idx.shape[0]
        if L <= POLICY_FULL_ATTENTION_MAX_LEN:
            pos_ids = torch.arange(L, device=codon_idx.device)
            x = self.codon_emb(codon_idx) + self.pos_emb(pos_ids)
            return self.encoder(x.unsqueeze(0)).squeeze(0)
        chunks = []
        for start in range(0, L, self.chunk_len):
            end = min(start + self.chunk_len, L)
            pos_ids = torch.arange(start, end, device=codon_idx.device)
            x = self.codon_emb(codon_idx[start:end]) + self.pos_emb(pos_ids)
            chunks.append(self.encoder(x.unsqueeze(0)).squeeze(0))
        return torch.cat(chunks, dim=0)

    def forward(self, codon_idx: torch.Tensor, legal_pos_mask: torch.Tensor, state_features: torch.Tensor, chosen_pos: Optional[int] = None, alt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        h = self.encode(codon_idx)
        pooled = h.mean(dim=0)
        s = self.state_proj(state_features.unsqueeze(0)).squeeze(0)
        L = h.shape[0]
        pos_feat = torch.cat([h, pooled.unsqueeze(0).expand(L, -1), s.unsqueeze(0).expand(L, -1)], dim=-1)
        pos_logits = mask_logits(self.pos_head(pos_feat).squeeze(-1), legal_pos_mask)
        value = self.value_head(torch.cat([pooled, s], dim=-1)).squeeze(-1)
        alt_logits = None
        if chosen_pos is not None:
            feat = torch.cat([pooled, h[chosen_pos], s], dim=-1)
            alt_logits = self.alt_head(feat)
            if alt_mask is not None:
                alt_logits = mask_logits(alt_logits, alt_mask)
        return pos_logits, alt_logits, value


def build_pos_mask(sense_codons: List[str], editable_positions: List[int], device: str) -> torch.Tensor:
    mask = torch.zeros(len(sense_codons), dtype=torch.bool, device=device)
    for p in editable_positions:
        mask[p] = True
    return mask


def build_alt_mask(n_alt: int, max_alt: int, device: str) -> torch.Tensor:
    mask = torch.zeros(max_alt, dtype=torch.bool, device=device)
    mask[:n_alt] = True
    return mask


def build_state_features(state: Dict, policy_seq_len: int, num_legal_positions: int, device: str) -> torch.Tensor:
    """
    Training-state features exclude center-related signals.
    The policy/value network only sees progress on the score+CAI objective
    plus a basic search-progress indicator. Center statistics are retained for
    logging and deployment-time screening only.
    """
    max_steps = max(int(state["max_steps"]), 1)
    step_ratio = float(state["step_id"]) / float(max_steps)

    score_z_gain = float(state["score_z"] - state["initial_score_z"])
    cai_z_gain = float(state["cai_z"] - state["initial_cai_z"])
    objective_gain = float(state["objective"] - state["initial_objective"])
    best_gap = float(state["best_objective"] - state["objective"])
    legal_frac = float(num_legal_positions) / float(max(policy_seq_len, 1))

    feats = torch.tensor(
        [
            step_ratio,
            score_z_gain,
            cai_z_gain,
            objective_gain,
            best_gap,
            legal_frac,
        ],
        dtype=torch.float32,
        device=device,
    )
    return feats


def estimate_state_value(policy: ActorCriticPolicy, env: SynonymousEnv, state: Dict) -> torch.Tensor:
    editable_positions = env.get_legal_positions()
    if len(editable_positions) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
    sense = state["policy_sense_codons"]
    codon_idx = codon_tensor(sense, DEVICE)
    pos_mask = build_pos_mask(sense, editable_positions, DEVICE)
    state_features = build_state_features(state, len(sense), len(editable_positions), DEVICE)
    with torch.no_grad():
        _, _, value = policy(codon_idx, pos_mask, state_features=state_features, chosen_pos=None, alt_mask=None)
    return value.detach()


def apply_actor_critic_update(
    policy: ActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    log_prob: torch.Tensor,
    value_s_t: torch.Tensor,
    reward: torch.Tensor,
    entropy: torch.Tensor,
    value_s_tp1: torch.Tensor,
    done: bool,
) -> Dict[str, float]:
    """
    One-step Actor-Critic update aligned with the architecture diagram.

    The reward evaluates the state transition caused by the selected edit action:
        r_t = J(C_{t+1}|R,E) - J(C_t|R,E)

    The critic provides the one-step TD / advantage signal:
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    This delta_t is used to:
      1) update the Actor policy by weighting log pi(a_t|s_t), and
      2) update the Critic by fitting V(s_t) to the one-step TD target.
    """
    reward = reward.to(value_s_t.device).float()
    value_s_tp1 = value_s_tp1.to(value_s_t.device).float().detach()
    not_done = 0.0 if bool(done) else 1.0

    td_target = reward + float(DISCOUNT) * value_s_tp1 * not_done
    td_error = td_target - value_s_t

    # Detach the TD error for the policy-gradient term so that the actor loss
    # does not backpropagate through the critic target/value calculation.
    advantage = td_error.detach()
    actor_loss = -(log_prob * advantage)
    critic_loss = F.smooth_l1_loss(value_s_t, td_target.detach())
    entropy_loss = -entropy
    loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "actor_loss": float(actor_loss.detach().item()),
        "critic_loss": float(critic_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "reward": float(reward.detach().item()),
        "value_s_t": float(value_s_t.detach().item()),
        "value_s_tp1": float(value_s_tp1.detach().item()),
        "td_target": float(td_target.detach().item()),
        "td_error": float(td_error.detach().item()),
    }


def summarize_episode_s_bio(trace_rows: List[Dict], episode_id: int, fallback_s_bio: float) -> float:
    episode_rows = [row for row in trace_rows if int(row.get("episode", -1)) == int(episode_id) and int(row.get("step", 0)) > 0]
    if len(episode_rows) == 0:
        return float(fallback_s_bio)
    vals = [float(row["current_s_bio"]) for row in episode_rows if "current_s_bio" in row]
    return float(np.mean(vals)) if len(vals) > 0 else float(fallback_s_bio)


def summarize_episode_metric(trace_rows: List[Dict], episode_id: int, key: str, fallback: float) -> float:
    episode_rows = [row for row in trace_rows if int(row.get("episode", -1)) == int(episode_id) and int(row.get("step", 0)) > 0]
    if len(episode_rows) == 0:
        return float(fallback)
    vals = [float(row[key]) for row in episode_rows if key in row]
    return float(np.mean(vals)) if len(vals) > 0 else float(fallback)


def make_episode_summary_row(
    *,
    episode: int,
    episode_start_source: str,
    num_steps: int,
    num_improvements: int,
    done_reason: str,
    final_score: float,
    final_score_z: float,
    final_s_bio: float,
    final_bio_prior_z: float,
    final_bio_loglik: float,
    final_objective: float,
    final_seq: str,
    best_score: float,
    best_score_z: float,
    best_s_bio: float,
    best_bio_prior_z: float,
    best_bio_loglik: float,
    best_objective: float,
    best_seq: str,
    episode_mean_s_bio: float,
    episode_mean_bio_prior_z: float,
) -> Dict:
    return {
        "episode": int(episode),
        "episode_start_source": str(episode_start_source),
        "num_steps": int(num_steps),
        "num_improvements": int(num_improvements),
        "done_reason": str(done_reason),
        "final_seq": str(final_seq),
        "final_score": float(final_score),
        "final_score_z": float(final_score_z),
        "final_s_bio": float(final_s_bio),
        "final_bio_prior_z": float(final_bio_prior_z),
        "final_bio_loglik": float(final_bio_loglik),
        "final_p_bio": similarity_to_legacy_penalty(final_s_bio),
        "final_objective": float(final_objective),
        "best_seq": str(best_seq),
        "best_score": float(best_score),
        "best_score_z": float(best_score_z),
        "best_s_bio": float(best_s_bio),
        "best_bio_prior_z": float(best_bio_prior_z),
        "best_bio_loglik": float(best_bio_loglik),
        "best_p_bio": similarity_to_legacy_penalty(best_s_bio),
        "best_objective": float(best_objective),
        "episode_mean_s_bio": float(episode_mean_s_bio),
        "episode_mean_bio_prior_z": float(episode_mean_bio_prior_z),
        "episode_mean_p_bio": similarity_to_legacy_penalty(episode_mean_s_bio),
    }


def make_candidate_record(
    *,
    sequence: str,
    score: float,
    score_z: float,
    s_bio: float,
    bio_prior_z: float,
    bio_loglik: float,
    cai_z: Optional[float] = None,
    delta_gc: float = 0.0,
    delta_rare: float = 0.0,
    delta_mfe_residual: float = 0.0,
    objective: Optional[float] = None,
    seq_id: Optional[int] = None,
    episode: Optional[int] = None,
    step: Optional[int] = None,
    source: str = "",
    source_role: str = "",
    candidate_start_source: str = "",
    candidate_stage: str = "",
    candidate_rollout_steps: Optional[int] = None,
    candidate_rank: Optional[int] = None,
    candidate_attempt: Optional[int] = None,
    candidate_done_reason: str = "",
    metrics: Optional[Dict[str, float]] = None,
) -> Dict:
    score = float(score)
    score_z = float(score_z)
    s_bio = float(s_bio)
    bio_prior_z = float(bio_prior_z)
    bio_loglik = float(bio_loglik)
    cai_z = float(cai_z) if cai_z is not None else float("nan")

    if objective is None:
        raise ValueError(
            "[ERROR] make_candidate_record requires explicit objective. "
            "This version uses deployment-time center screening, so objective must be passed explicitly."
        )

    objective = float(objective)
    if not np.isfinite(objective):
        raise ValueError(f"[ERROR] objective is not finite: {objective}")

    record = {
        "sequence": str(sequence),
        "objective": float(objective),
        "score": score,
        "score_z": score_z,
        "cai_z": cai_z,
        "s_bio": s_bio,
        "bio_prior_z": bio_prior_z,
        "bio_loglik": bio_loglik,
        "p_bio": similarity_to_legacy_penalty(s_bio),
        "delta_gc": float(delta_gc),
        "delta_rare": float(delta_rare),
        "delta_mfe_residual": float(delta_mfe_residual),
        "objective_definition": "score_z + cai_z",
        "seq_id": int(seq_id) if seq_id is not None else None,
        "episode": int(episode) if episode is not None else None,
        "step": int(step) if step is not None else None,
        "source": str(source),
        "source_role": str(source_role),
        "candidate_start_source": str(candidate_start_source),
        "candidate_stage": str(candidate_stage),
        "candidate_rollout_steps": int(candidate_rollout_steps) if candidate_rollout_steps is not None else None,
        "candidate_rank": int(candidate_rank) if candidate_rank is not None else None,
        "candidate_attempt": int(candidate_attempt) if candidate_attempt is not None else None,
        "candidate_done_reason": str(candidate_done_reason),
    }

    if metrics is not None:
        for k, v in metrics.items():
            record[f"metric_{k}"] = v

    return record

# =========================================================
# 9. Candidate helpers (global-only)
# =========================================================
def compute_constraint_penalty(
    *,
    bio_prior_z: float,
    s_bio: float,
    delta_gc: float,
    delta_rare: float,
    delta_mfe_residual: float,
    mu_bio_prior: float = MU_BIO_PRIOR_INIT,
    mu_s_bio: float = MU_S_BIO_INIT,
    mu_gc: float = MU_GC_INIT,
    mu_rare: float = MU_RARE_INIT,
    mu_mfe: float = MU_MFE_INIT,
) -> float:
    return 0.0


def compute_objective(
    score_z: float,
    cai_z: float,
    bio_prior_z: float,
    s_bio: float,
    delta_gc: float,
    delta_rare: float,
    delta_mfe_residual: float,
    mu_bio_prior: float = MU_BIO_PRIOR_INIT,
    mu_s_bio: float = MU_S_BIO_INIT,
    mu_gc: float = MU_GC_INIT,
    mu_rare: float = MU_RARE_INIT,
    mu_mfe: float = MU_MFE_INIT,
) -> float:
    return float(float(score_z) + float(cai_z))


def normalize_candidate_for_export(candidate: Dict) -> Dict:
    metrics = {}
    for k, v in candidate.items():
        if str(k).startswith("metric_"):
            metrics[str(k)[7:]] = v

    if "objective" not in candidate:
        keys_preview = sorted(list(candidate.keys()))
        raise KeyError(
            "[ERROR] Candidate is missing required field 'objective' before export normalization. "
            f"Available keys: {keys_preview}"
        )

    return make_candidate_record(
        sequence=str(candidate["sequence"]),
        score=float(candidate["score"]),
        score_z=float(candidate.get("score_z", candidate["score"])),
        cai_z=float(candidate.get("cai_z", float("nan"))),
        s_bio=float(candidate["s_bio"]),
        bio_prior_z=float(candidate.get("bio_prior_z", candidate.get("s_bio", 0.0))),
        bio_loglik=float(candidate.get("bio_loglik", 0.0)),
        delta_gc=float(candidate.get("delta_gc", 0.0)),
        delta_rare=float(candidate.get("delta_rare", 0.0)),
        delta_mfe_residual=float(candidate.get("delta_mfe_residual", 0.0)),
        objective=float(candidate["objective"]),
        seq_id=candidate.get("seq_id"),
        episode=candidate.get("episode"),
        step=candidate.get("step"),
        source=candidate.get("source", ""),
        source_role=candidate.get("source_role", ""),
        candidate_start_source=candidate.get("candidate_start_source", ""),
        candidate_stage=candidate.get("candidate_stage", ""),
        candidate_rollout_steps=candidate.get("candidate_rollout_steps"),
        candidate_rank=candidate.get("candidate_rank"),
        candidate_attempt=candidate.get("candidate_attempt"),
        candidate_done_reason=candidate.get("candidate_done_reason", ""),
        metrics=metrics if len(metrics) > 0 else None,
    )


def get_candidate_objective(candidate: Dict) -> float:
    """
    Final-clean behavior:
    objective is REQUIRED.
    Do not silently fall back to score_z / score, otherwise ranking may quietly
    degrade into a score-only policy when some stage forgets to pass objective.
    """
    if "objective" not in candidate:
        keys_preview = sorted(list(candidate.keys()))
        raise KeyError(
            "[ERROR] Candidate is missing required field 'objective'. "
            f"Available keys: {keys_preview}"
        )

    obj = candidate["objective"]
    try:
        obj = float(obj)
    except Exception as e:
        raise TypeError(f"[ERROR] Candidate objective is not numeric: {obj}") from e

    if not np.isfinite(obj):
        raise ValueError(f"[ERROR] Candidate objective is not finite: {obj}")

    return float(obj)


def get_candidate_metric(candidate: Dict, metric_name: str, default: float = float("-inf")) -> float:
    value = candidate.get(metric_name, default)
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def get_candidate_episode_key(candidate: Dict) -> Tuple[float, float, float, float]:
    """
    Center must not influence training-time candidate comparison in this final version.
    Episode-best comparison therefore uses only the score+CAI training objective and
    score/CAI-related tie-breakers.
    """
    return (
        -float(get_candidate_objective(candidate)),
        -float(candidate.get("score_z", candidate["score"])),
        -float(candidate.get("cai_z", float("-inf"))),
        -float(candidate["score"]),
    )


def get_candidate_archive_key(candidate: Dict) -> Tuple[float, float, float, float]:
    """
    Archive comparison also excludes center-related quantities so that center is only
    applied at final deployment screening.
    """
    return (
        -float(get_candidate_objective(candidate)),
        -float(candidate.get("score_z", candidate["score"])),
        -float(candidate.get("cai_z", float("-inf"))),
        -float(candidate["score"]),
    )


def is_better_candidate(candidate: Dict, incumbent: Optional[Dict], mode: str = "archive") -> bool:
    if incumbent is None:
        return True
    if mode == "episode":
        return bool(get_candidate_episode_key(candidate) < get_candidate_episode_key(incumbent))
    if mode == "archive":
        return bool(get_candidate_archive_key(candidate) < get_candidate_archive_key(incumbent))
    raise ValueError(f"Unsupported mode: {mode}")


# no adaptive beta update is used in the prior-based formulation


def should_add_to_final_pool(candidate_stage: str, episode: Optional[int] = None) -> bool:
    """
    final_candidate_pool is deployment-only:
    - keep baseline as a safe fallback
    - keep late_checkpoint/final generated candidates
    - keep episode_best only after the early-monitor phase
    - do NOT keep monitor-stage generated candidates
    """
    if candidate_stage == "baseline":
        return True
    if candidate_stage in {"late_checkpoint", "final"}:
        return True
    if candidate_stage == "episode_best":
        return episode is not None and int(episode) > EARLY_MONITOR_UNTIL_EPISODE
    return False


def update_archive(archive: List[Dict], candidate: Dict) -> List[Dict]:
    same_seq_idx = None
    for i, existing in enumerate(archive):
        if existing["sequence"] == candidate["sequence"]:
            same_seq_idx = i
            break

    if same_seq_idx is not None:
        if is_better_candidate(candidate, archive[same_seq_idx], mode="archive"):
            archive[same_seq_idx] = candidate
        return archive

    archive.append(candidate)
    return archive


def maybe_prune_archive(archive: List[Dict]) -> List[Dict]:
    if len(archive) <= PARETO_SOFT_LIMIT:
        return archive
    return sorted(archive, key=get_candidate_archive_key)[:PARETO_SOFT_LIMIT]


def _safe_z(v: float, mu: float, sd: float) -> Optional[float]:
    if not np.isfinite(v):
        return None
    if (not np.isfinite(sd)) or sd < EPS:
        return 0.0
    return float((v - mu) / sd)


def _apply_codon_usage_standardization(candidates: List[Dict]) -> List[Dict]:
    if len(candidates) == 0:
        return []
    enriched = [dict(c) for c in candidates]
    cai_vals = np.asarray(
        [get_candidate_metric(c, "metric_cai", default=float("nan")) for c in enriched],
        dtype=float
    )
    cai_finite = cai_vals[np.isfinite(cai_vals)]
    cai_mu = float(np.mean(cai_finite)) if len(cai_finite) > 0 else 0.0
    cai_sd = float(np.std(cai_finite, ddof=0)) if len(cai_finite) > 1 else 1.0

    for temp in enriched:
        cai_z = _safe_z(
            get_candidate_metric(temp, "metric_cai", default=float("nan")),
            cai_mu,
            cai_sd
        )
        temp["cai_rank_z"] = float(cai_z) if cai_z is not None else float("-inf")
    return enriched


def get_candidate_final_report_key(candidate: Dict) -> Tuple[float, float, float]:
    cai_rank_z = float(candidate.get("cai_rank_z", float("-inf")))
    return (
        -float(get_candidate_objective(candidate)),
        -float(candidate["score"]),
        -float(cai_rank_z),
    )


def is_candidate_feasible_for_deployment(candidate: Dict) -> bool:
    """
    Deployment-time center screen only.

    This version does NOT use center-related terms in the training penalty.
    Final deployment keeps only candidates whose center similarity is above
    the lower bound, then ranks them by score_z + cai_z.
    """
    s_bio = float(candidate.get("s_bio", float("-inf")))
    if not np.isfinite(s_bio):
        return False
    if s_bio < CENTER_MIN:
        return False
    return True


def count_feasible_final_candidates(candidates: List[Dict]) -> int:
    return int(sum(1 for c in candidates if is_candidate_feasible_for_deployment(c)))


def _prepare_ranked_candidates_for_final_report(candidates: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
        feasible_ranked, all_ranked

    We always rank both:
    - feasible_ranked: explicit deployment-feasible candidates only
    - all_ranked: all normalized candidates

    This lets final_best / final_topk make a strict choice while still exposing
    the full pool size for diagnostics.
    """
    if len(candidates) == 0:
        raise ValueError("Cannot rank candidates from an empty candidate list")

    enriched = [normalize_candidate_for_export(c) for c in candidates]
    enriched = _apply_codon_usage_standardization(enriched)

    all_ranked = sorted(enriched, key=get_candidate_final_report_key)
    feasible_ranked = [c for c in all_ranked if is_candidate_feasible_for_deployment(c)]

    return feasible_ranked, all_ranked


def select_final_best_candidate(
    candidates: List[Dict],
):
    feasible_ranked, all_ranked = _prepare_ranked_candidates_for_final_report(candidates)

    if len(feasible_ranked) == 0:
        raise ValueError(
            "[ERROR] No deployment-feasible candidate found in final pool. "
            "All candidates fall below the center lower bound."
        )

    final_best = dict(feasible_ranked[0])
    final_best["selection_pool_best_score"] = float(feasible_ranked[0]["score"])
    final_best["selection_rule"] = "center_screen_then_rank_J_eq_scorez_plus_caiz"
    final_best["ranking_pool_size"] = int(len(all_ranked))
    final_best["feasible_pool_size"] = int(len(feasible_ranked))

    return final_best, len(feasible_ranked)


def select_final_topk_candidates(
    candidates: List[Dict],
    topk: int,
):
    feasible_ranked, all_ranked = _prepare_ranked_candidates_for_final_report(candidates)

    if len(feasible_ranked) == 0:
        raise ValueError(
            "[ERROR] No deployment-feasible candidate found in final pool; cannot build final top-k."
        )

    out = []
    for cand in feasible_ranked[:topk]:
        temp = dict(cand)
        temp["selection_rule"] = "center_screen_then_rank_J_eq_scorez_plus_caiz"
        temp["ranking_pool_size"] = int(len(all_ranked))
        temp["feasible_pool_size"] = int(len(feasible_ranked))
        out.append(temp)
    return out


def get_candidate_generation_plan(ep: int) -> Tuple[int, Optional[str]]:
    if ep == MAX_EPISODES:
        return FINAL_CANDIDATE_COUNT, "final"
    if ep in LATE_CHECKPOINT_CANDIDATES:
        return int(LATE_CHECKPOINT_CANDIDATES[ep]), "late_checkpoint"
    if EARLY_MONITOR_EVERY_EPISODES > 0 and ep <= EARLY_MONITOR_UNTIL_EPISODE and ep % EARLY_MONITOR_EVERY_EPISODES == 0:
        return EARLY_MONITOR_CANDIDATE_COUNT, "monitor"
    return 0, None


def get_total_planned_candidate_targets() -> Tuple[int, int]:
    final_stage_target_count = 0
    total_target_count = 0
    for ep in range(1, MAX_EPISODES + 1):
        target_count, stage = get_candidate_generation_plan(ep)
        target_count = int(target_count)
        total_target_count += target_count
        if stage == "final":
            final_stage_target_count += target_count
    return int(final_stage_target_count), int(total_target_count)


def make_random_synonymous_restart(base_seq: str, max_codon_index: Optional[int] = None) -> str:
    seq = clean_seq(base_seq)
    sense, stop_codon = split_cds_and_stop(seq)
    editable_positions = legal_positions(sense, max_codon_index=max_codon_index)
    if len(editable_positions) == 0:
        return seq
    n_edits = int(round(len(editable_positions) * RANDOM_RESTART_EDIT_FRAC))
    n_edits = max(RANDOM_RESTART_MIN_EDITS, n_edits)
    n_edits = min(RANDOM_RESTART_MAX_EDITS, n_edits, len(editable_positions))
    if n_edits <= 0:
        return seq
    chosen_positions = random.sample(editable_positions, n_edits)
    new_sense = list(sense)
    for pos in chosen_positions:
        alts = synonymous_alts(new_sense[pos])
        if len(alts) > 0:
            new_sense[pos] = random.choice(alts)
    return join_cds(new_sense, stop_codon)


def choose_episode_start_seq(ep: int, original_seq: str, global_best_seq: str) -> Tuple[str, str]:
    if ep <= WARMUP_EPISODES:
        if ep % 2 == 0:
            return make_random_synonymous_restart(original_seq, max_codon_index=POLICY_WINDOW_CODONS), "random_original_warmup"
        return original_seq, "original_warmup"
    r = random.random()
    if r < ORIGINAL_RESTART_PROB:
        return original_seq, "original_restart"
    r -= ORIGINAL_RESTART_PROB
    if r < RANDOM_ORIGINAL_RESTART_PROB:
        return make_random_synonymous_restart(original_seq, max_codon_index=POLICY_WINDOW_CODONS), "random_original_restart"
    r -= RANDOM_ORIGINAL_RESTART_PROB
    if global_best_seq != original_seq and r < GLOBAL_BEST_START_PROB:
        return global_best_seq, "global_best"
    if global_best_seq != original_seq:
        return make_random_synonymous_restart(global_best_seq, max_codon_index=POLICY_WINDOW_CODONS), "random_global_best_restart"
    return make_random_synonymous_restart(original_seq, max_codon_index=POLICY_WINDOW_CODONS), "fallback_random_original_restart"


def choose_candidate_start_seq(sample_idx: int, original_seq: str, global_best_seq: str, episode_best_seq: str) -> Tuple[str, str]:
    mode = sample_idx % 4
    if mode == 0:
        return episode_best_seq, "episode_best"
    if mode == 1:
        return global_best_seq if global_best_seq is not None else episode_best_seq, "global_best"
    if mode == 2:
        base = global_best_seq if global_best_seq is not None else episode_best_seq
        return make_random_synonymous_restart(base, max_codon_index=POLICY_WINDOW_CODONS), "random_global_or_episode_best"
    return make_random_synonymous_restart(original_seq, max_codon_index=POLICY_WINDOW_CODONS), "random_original"



def generate_candidates_with_policy(
    policy: ActorCriticPolicy,
    original_seq: str,
    global_best_seq: str,
    episode_best_seq: str,
    seq_id: int,
    episode: int,
    stage: str,
    scorer,
    target_specs: Dict[str, Dict[str, float]],
    cai_weights: Dict[str, float],
    csi_weights: Dict[str, float],
    score_ref_stats: Dict[str, float],
    cai_ref_stats: Dict[str, float],
    target_count: int,
    existing_seen: Optional[set] = None,
    target_aa_sequence: Optional[str] = None,
) -> List[Dict]:
    if target_count <= 0:
        return []

    candidates = []
    seen = set(existing_seen) if (existing_seen is not None and CANDIDATE_DEDUP) else set()
    attempts = 0
    stagnant_attempts = 0

    default_max_attempts = max(int(target_count * CANDIDATE_MAX_ATTEMPT_MULTIPLIER), target_count)
    if stage == "final" and FINAL_STAGE_FORCE_FILL:
        max_attempts = max(FINAL_STAGE_MAX_ATTEMPTS, default_max_attempts)
    else:
        max_attempts = default_max_attempts

    policy.eval()
    try:
        while len(candidates) < target_count:
            if attempts >= max_attempts:
                print(
                    f"[WARN] stage={stage} stopped before reaching target_count "
                    f"(generated={len(candidates)}, target={target_count}, attempts={attempts})"
                )
                break

            if stage == "final" and FINAL_STAGE_FORCE_FILL and stagnant_attempts >= FINAL_STAGE_MAX_STAGNANT_ATTEMPTS:
                print(
                    f"[WARN] stage=final stagnated before reaching target_count "
                    f"(generated={len(candidates)}, target={target_count}, stagnant_attempts={stagnant_attempts})"
                )
                break

            attempts += 1
            start_seq, start_source = choose_candidate_start_seq(
                attempts - 1, original_seq, global_best_seq, episode_best_seq
            )

            env = SynonymousEnv(
                original_seq=original_seq,
                scorer=scorer,
                target_specs=target_specs,
                cai_weights=cai_weights,
                csi_weights=csi_weights,
                score_ref_stats=score_ref_stats,
                cai_ref_stats=cai_ref_stats,
                target_aa_sequence=target_aa_sequence,
            )
            state = env.reset(start_seq=start_seq)
            candidate_max_steps = max(1, int(round(env.max_steps * CANDIDATE_MAX_STEPS_SCALE)))
            done = False
            rollout_step = 0

            rollout_best_candidate = make_candidate_record(
                sequence=env.curr_seq,
                score=float(env.curr_score),
                score_z=float(env.curr_score_z),
                cai_z=float(env.curr_cai_z),
                s_bio=float(env.curr_s_bio),
                bio_prior_z=float(env.curr_bio_prior_z),
                bio_loglik=float(env.curr_bio_loglik),
                delta_gc=float(env.curr_delta_gc),
                delta_rare=float(env.curr_delta_rare),
                delta_mfe_residual=float(env.curr_delta_mfe_residual),
                objective=float(env.curr_objective),
                seq_id=int(seq_id),
                episode=int(episode),
                source=f"{stage}_generated",
                source_role="generated",
                candidate_start_source=start_source,
                candidate_stage=stage,
                candidate_rollout_steps=0,
                candidate_attempt=int(attempts),
                candidate_done_reason="",
            )

            with torch.no_grad():
                while not done and rollout_step < candidate_max_steps:
                    rollout_step += 1
                    sense = state["policy_sense_codons"]
                    editable_positions = env.get_legal_positions()
                    if len(editable_positions) == 0:
                        env.done_reason = "no_legal_actions"
                        break

                    codon_idx = codon_tensor(sense, DEVICE)
                    pos_mask = build_pos_mask(sense, editable_positions, DEVICE)
                    state_features = build_state_features(
                        state, len(sense), len(editable_positions), DEVICE
                    )

                    pos_logits, _, _ = policy(
                        codon_idx,
                        pos_mask,
                        state_features=state_features,
                        chosen_pos=None,
                        alt_mask=None,
                    )
                    pos_logits = pos_logits / CANDIDATE_TEMPERATURE

                    pos_k = min(CANDIDATE_TOPK_POS, len(editable_positions))
                    pos_values, pos_indices = torch.topk(pos_logits, k=pos_k)
                    pos_dist = Categorical(logits=pos_values)
                    pos = pos_indices[pos_dist.sample()]

                    alts = env.get_legal_alts(int(pos.item()))
                    if len(alts) == 0:
                        env.done_reason = "no_legal_alts"
                        break

                    alt_mask = build_alt_mask(len(alts), MAX_ALT, DEVICE)
                    _, alt_logits, _ = policy(
                        codon_idx,
                        pos_mask,
                        state_features=state_features,
                        chosen_pos=int(pos.item()),
                        alt_mask=alt_mask,
                    )
                    alt_logits = alt_logits / CANDIDATE_TEMPERATURE

                    alt_k = min(CANDIDATE_TOPK_ALT, len(alts))
                    alt_values, alt_indices = torch.topk(alt_logits, k=alt_k)
                    alt_dist = Categorical(logits=alt_values)
                    alt_choice = int(alt_indices[alt_dist.sample()].item())
                    alt_codon = alts[alt_choice]

                    state, _, done, _ = env.step(int(pos.item()), alt_codon)

                    current_candidate = make_candidate_record(
                        sequence=env.curr_seq,
                        score=float(env.curr_score),
                        score_z=float(env.curr_score_z),
                        cai_z=float(env.curr_cai_z),
                        s_bio=float(env.curr_s_bio),
                        bio_prior_z=float(env.curr_bio_prior_z),
                        bio_loglik=float(env.curr_bio_loglik),
                        delta_gc=float(env.curr_delta_gc),
                        delta_rare=float(env.curr_delta_rare),
                        delta_mfe_residual=float(env.curr_delta_mfe_residual),
                        objective=float(env.curr_objective),
                        seq_id=int(seq_id),
                        episode=int(episode),
                        source=f"{stage}_generated",
                        source_role="generated",
                        candidate_start_source=start_source,
                        candidate_stage=stage,
                        candidate_rollout_steps=int(rollout_step),
                        candidate_attempt=int(attempts),
                        candidate_done_reason=str(env.done_reason) if env.done_reason is not None else "",
                    )

                    if is_better_candidate(current_candidate, rollout_best_candidate, mode="episode"):
                        rollout_best_candidate = current_candidate

            seq = rollout_best_candidate["sequence"]

            if CANDIDATE_DEDUP and seq in seen:
                stagnant_attempts += 1
                continue

            metrics = calc_all_metrics(
                seq,
                cai_weights=cai_weights,
                csi_weights=csi_weights,
                target_specs=target_specs,
            )

            candidate = make_candidate_record(
                sequence=seq,
                score=float(rollout_best_candidate["score"]),
                score_z=float(rollout_best_candidate["score_z"]),
                cai_z=float(rollout_best_candidate.get("cai_z", float("nan"))),
                s_bio=float(rollout_best_candidate["s_bio"]),
                bio_prior_z=float(rollout_best_candidate["bio_prior_z"]),
                bio_loglik=float(rollout_best_candidate.get("bio_loglik", bio_loglik(metrics, target_specs))),
                delta_gc=float(rollout_best_candidate.get("delta_gc", 0.0)),
                delta_rare=float(rollout_best_candidate.get("delta_rare", 0.0)),
                delta_mfe_residual=float(rollout_best_candidate.get("delta_mfe_residual", 0.0)),
                objective=float(rollout_best_candidate.get("objective", rollout_best_candidate["score_z"])),
                seq_id=int(seq_id),
                episode=int(episode),
                step=None,
                source=f"{stage}_generated",
                source_role="generated",
                candidate_start_source=start_source,
                candidate_stage=stage,
                candidate_rollout_steps=int(rollout_step),
                candidate_attempt=int(attempts),
                candidate_done_reason=str(env.done_reason) if env.done_reason is not None else "",
                metrics=metrics,
            )
            if CANDIDATE_DEDUP:
                seen.add(seq)
            stagnant_attempts = 0
            candidates.append(candidate)

        for rank, cand in enumerate(candidates, start=1):
            cand["candidate_rank"] = rank

        return candidates

    finally:
        policy.train()

# =========================================================
# 10. Train one sequence
# =========================================================
def train_one_sequence(
    seq_id: int,
    original_seq: str,
    scorer,
    target_specs: Dict[str, Dict[str, float]],
    cai_weights: Dict[str, float],
    csi_weights: Dict[str, float],
    score_ref_stats: Dict[str, float],
    cai_ref_stats: Dict[str, float],
    target_aa_sequence: Optional[str] = None,
) -> Dict:
    policy = ActorCriticPolicy(
        vocab_size=len(CODON2IDX),
        d_model=128,
        nhead=4,
        nlayers=2,
        max_len=max(len(split_cds_and_stop(original_seq)[0]), 2048),
        max_alt=MAX_ALT,
        n_state_features=6,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    seq_out_dir = os.path.join(OUT_DIR, f"seq_{seq_id:04d}")
    os.makedirs(seq_out_dir, exist_ok=True)

    trace_rows: List[Dict] = []
    episode_summary_rows: List[Dict] = []
    archive: List[Dict] = []
    final_candidate_pool: List[Dict] = []
    final_seen_candidates = set()
    final_seen_all = set()
    final_candidate_generated_total_count = 0
    final_candidate_target_total_count = 0
    final_stage_generated_count = 0
    final_stage_target_count = 0
    global_seen_candidates = {original_seq}

    base_env = SynonymousEnv(
        original_seq=original_seq,
        scorer=scorer,
        target_specs=target_specs,
        cai_weights=cai_weights,
        csi_weights=csi_weights,
        score_ref_stats=score_ref_stats,
        cai_ref_stats=cai_ref_stats,
        target_aa_sequence=target_aa_sequence,
    )
    original_metrics = base_env.curr_metrics
    original_score = base_env.curr_score
    original_score_z = base_env.curr_score_z
    original_s_bio = base_env.curr_s_bio
    original_bio_prior_z = base_env.curr_bio_prior_z
    original_bio_loglik = base_env.curr_bio_loglik

    global_best_seq = original_seq
    global_best_score = original_score
    global_best_score_z = original_score_z
    global_best_cai_z = float(base_env.curr_cai_z)
    global_best_s_bio = original_s_bio
    global_best_bio_prior_z = original_bio_prior_z
    global_best_bio_loglik = original_bio_loglik
    global_best_objective = float(base_env.curr_objective)
    global_best_delta_gc = float(base_env.curr_delta_gc)
    global_best_delta_rare = float(base_env.curr_delta_rare)
    global_best_delta_mfe_residual = float(base_env.curr_delta_mfe_residual)
        
    base_candidate = make_candidate_record(
        sequence=original_seq,
        score=float(original_score),
        score_z=float(original_score_z),
        cai_z=float(base_env.curr_cai_z),
        s_bio=float(original_s_bio),
        bio_prior_z=float(original_bio_prior_z),
        bio_loglik=float(original_bio_loglik),
        delta_gc=float(base_env.curr_delta_gc),
        delta_rare=float(base_env.curr_delta_rare),
        delta_mfe_residual=float(base_env.curr_delta_mfe_residual),
        objective=float(base_env.curr_objective),
        seq_id=seq_id,
        episode=0,
        step=0,
        source="original_baseline",
        source_role="baseline",
        candidate_start_source="original",
        candidate_stage="baseline",
        candidate_rollout_steps=0,
        candidate_rank=0,
        metrics=original_metrics,
    )
    archive = update_archive(archive, base_candidate)
    archive = maybe_prune_archive(archive)
    final_candidate_pool = update_archive(final_candidate_pool, base_candidate)
    final_seen_candidates.add(base_candidate["sequence"])
    final_seen_all.add(base_candidate["sequence"])

    if base_env.max_steps == 0:
        trace_rows.append({
            "episode": 0,
            "episode_start_source": "original",
            "step": 0,
            "max_steps": int(base_env.max_steps),
            "reward": 0.0,
            "delta_t": 0.0,
            "delta_score_z": 0.0,
            "delta_bio": 0.0,
            "delta_bio_prior_z": 0.0,
            "delta_cai_z": 0.0,
            "current_score": float(base_env.curr_score),
            "current_score_z": float(base_env.curr_score_z),
            "current_cai_z": float(base_env.curr_cai_z),
            "current_s_bio": float(base_env.curr_s_bio),
            "current_bio_prior_z": float(base_env.curr_bio_prior_z),
            "current_bio_loglik": float(base_env.curr_bio_loglik),
            "current_p_bio": similarity_to_legacy_penalty(base_env.curr_s_bio),
            "current_objective": float(base_env.curr_objective),
            "current_delta_gc": float(base_env.curr_delta_gc),
            "current_delta_rare": float(base_env.curr_delta_rare),
            "current_delta_mfe_residual": float(base_env.curr_delta_mfe_residual),
            "best_score_in_episode": float(base_env.best_score),
            "best_score_z_in_episode": float(base_env.best_score_z),
            "best_s_bio_in_episode": float(base_env.best_s_bio),
            "best_bio_prior_z_in_episode": float(base_env.best_bio_prior_z),
            "current_seq": base_env.curr_seq,
            "best_seq_in_episode": base_env.best_seq,
            "done_reason": "no_editable_positions",
        })

        episode_summary = make_episode_summary_row(
            episode=0,
            episode_start_source="original",
            num_steps=0,
            num_improvements=0,
            done_reason="no_editable_positions",
            final_score=float(base_env.curr_score),
            final_score_z=float(base_env.curr_score_z),
            final_s_bio=float(base_env.curr_s_bio),
            final_bio_prior_z=float(base_env.curr_bio_prior_z),
            final_bio_loglik=float(base_env.curr_bio_loglik),
            final_objective=float(base_env.curr_objective),
            final_seq=base_env.curr_seq,
            best_score=float(base_env.best_score),
            best_score_z=float(base_env.best_score_z),
            best_s_bio=float(base_env.best_s_bio),
            best_bio_prior_z=float(base_env.best_bio_prior_z),
            best_bio_loglik=float(base_env.best_bio_loglik),
            best_objective=float(base_env.best_objective),
            best_seq=base_env.best_seq,
            episode_mean_s_bio=float(base_env.curr_s_bio),
            episode_mean_bio_prior_z=float(base_env.curr_bio_prior_z),
        )
        for k, v in original_metrics.items():
            episode_summary[f"final_metric_{k}"] = v
            episode_summary[f"best_metric_{k}"] = v
        episode_summary_rows.append(episode_summary)
        final_candidates = select_final_topk_candidates(list(final_candidate_pool), topk=FINAL_CANDIDATE_COUNT)
        final_best, candidate_pool_size = select_final_best_candidate(list(final_candidate_pool))
        final_top10 = select_final_topk_candidates(list(final_candidate_pool), topk=FINAL_TOPK_PER_RUN)
        write_table_txt(trace_rows, os.path.join(seq_out_dir, "trace.txt"))
        write_table_txt(episode_summary_rows, os.path.join(seq_out_dir, "episode_summary.txt"))
        write_sequence_txt(final_best["sequence"], os.path.join(seq_out_dir, "final_best.txt"))
        write_json_txt(final_best, os.path.join(seq_out_dir, "final_best_meta.txt"))
        write_table_txt(final_candidates, os.path.join(seq_out_dir, "final_candidates.txt"))
        write_table_txt(final_top10, os.path.join(seq_out_dir, "final_top10.txt"))
        remaining_final_pool_gap = max(0, FINAL_CANDIDATE_COUNT - count_feasible_final_candidates(final_candidate_pool))

        return {
              "seq_id": seq_id,
              "original_seq": original_seq,
              "best_sequence": final_best["sequence"],
              "best_score": final_best["score"],
              "best_score_z": final_best["score_z"],
              "best_s_bio": final_best["s_bio"],
              "best_bio_prior_z": final_best["bio_prior_z"],
              "best_objective": final_best["objective"],
              "candidate_pool_size": candidate_pool_size,
              "generated_candidates": 0,
              "target_candidates": remaining_final_pool_gap,
              "generated_candidates_total": 0,
              "target_candidates_total": remaining_final_pool_gap,
        "final_topk_per_run": FINAL_TOPK_PER_RUN,
          }

    for ep in range(1, MAX_EPISODES + 1):
        start_seq, start_source = choose_episode_start_seq(ep, original_seq, global_best_seq)
        env = SynonymousEnv(
            original_seq=original_seq,
            scorer=scorer,
            target_specs=target_specs,
            cai_weights=cai_weights,
            csi_weights=csi_weights,
            score_ref_stats=score_ref_stats,
            cai_ref_stats=cai_ref_stats,
            target_aa_sequence=target_aa_sequence,
        )
        state = env.reset(start_seq=start_seq)

        episode_best_candidate = make_candidate_record(
            sequence=env.curr_seq,
            score=float(env.curr_score),
            score_z=float(env.curr_score_z),
            cai_z=float(env.curr_cai_z),
            s_bio=float(env.curr_s_bio),
            bio_prior_z=float(env.curr_bio_prior_z),
            bio_loglik=float(env.curr_bio_loglik),
            delta_gc=float(env.curr_delta_gc),
            delta_rare=float(env.curr_delta_rare),
            delta_mfe_residual=float(env.curr_delta_mfe_residual),
            objective=float(env.curr_objective),
            seq_id=seq_id,
            episode=ep,
            source=f"{start_source}_episode_start",
            source_role="episode_start",
            candidate_start_source=start_source,
            candidate_stage="episode_start",
            candidate_rollout_steps=0,
            metrics=env.curr_metrics,
        )
        episode_best_metrics = dict(env.curr_metrics)

        done = False
        step_in_ep = 0
        num_improvements = 0

        while not done:
            step_in_ep += 1
            sense = state["policy_sense_codons"]
            editable_positions = env.get_legal_positions()
            if len(editable_positions) == 0:
                env.done_reason = "no_legal_actions"
                break

            codon_idx = codon_tensor(sense, DEVICE)
            pos_mask = build_pos_mask(sense, editable_positions, DEVICE)
            state_features = build_state_features(state, len(sense), len(editable_positions), DEVICE)
            pos_logits, _, value = policy(codon_idx, pos_mask, state_features=state_features, chosen_pos=None, alt_mask=None)
            pos_dist = Categorical(logits=pos_logits)
            pos = pos_dist.sample()

            alts = env.get_legal_alts(int(pos.item()))
            alt_mask = build_alt_mask(len(alts), MAX_ALT, DEVICE)
            _, alt_logits, _ = policy(codon_idx, pos_mask, state_features=state_features, chosen_pos=int(pos.item()), alt_mask=alt_mask)
            alt_dist = Categorical(logits=alt_logits)
            alt_idx = alt_dist.sample()

            chosen_alt = alts[int(alt_idx.item())]
            next_state, reward, done, info = env.step(int(pos.item()), chosen_alt)

            log_prob = pos_dist.log_prob(pos) + alt_dist.log_prob(alt_idx)
            entropy = pos_dist.entropy() + alt_dist.entropy()
            reward_t = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            value_s_tp1 = torch.tensor(0.0, dtype=torch.float32, device=DEVICE) if done else estimate_state_value(policy, env, next_state)
            update_stats = apply_actor_critic_update(
                policy=policy,
                optimizer=optimizer,
                log_prob=log_prob,
                value_s_t=value,
                reward=reward_t,
                entropy=entropy,
                value_s_tp1=value_s_tp1,
                done=done,
            )

            current_candidate = make_candidate_record(
                sequence=info["current_seq"],
                score=float(info["current_score"]),
                score_z=float(info["current_score_z"]),
                cai_z=float(info["current_cai_z"]),
                s_bio=float(info["current_s_bio"]),
                bio_prior_z=float(info["current_bio_prior_z"]),
                bio_loglik=float(info["current_bio_loglik"]),
                delta_gc=float(info["current_delta_gc"]),
                delta_rare=float(info["current_delta_rare"]),
                delta_mfe_residual=float(info["current_delta_mfe_residual"]),
                objective=float(info["current_objective"]),
                seq_id=seq_id,
                episode=ep,
                step=step_in_ep,
                source=f"{start_source}_train_step",
                source_role="train_step",
                candidate_start_source=start_source,
                candidate_stage="train_step",
                candidate_rollout_steps=step_in_ep,
                metrics=info["metrics"],
            )
            if is_better_candidate(current_candidate, episode_best_candidate, mode="episode"):
                episode_best_candidate = dict(current_candidate)
                episode_best_metrics = dict(info["metrics"])
            if info["improved"]:
                num_improvements += 1

            trace_rows.append({
                "episode": ep,
                "episode_start_source": start_source,
                "step": step_in_ep,
                "max_steps": int(env.max_steps),
                "reward": float(info["reward"]),
                "delta_t": float(info["delta_t"]),
                "delta_score_z": float(info["delta_score_z"]),
                "delta_cai_z": float(info["delta_cai_z"]),
                "delta_bio": float(info["delta_bio"]),
                "delta_bio_prior_z": float(info["delta_bio_prior_z"]),
                "current_score": float(info["current_score"]),
                "current_score_z": float(info["current_score_z"]),
                "current_cai_z": float(info["current_cai_z"]),
                "current_s_bio": float(info["current_s_bio"]),
                "current_bio_prior_z": float(info["current_bio_prior_z"]),
                "current_bio_loglik": float(info["current_bio_loglik"]),
                "current_p_bio": float(info["current_p_bio"]),
                "current_objective": float(info["current_objective"]),
                "current_delta_gc": float(info["current_delta_gc"]),
                "current_delta_rare": float(info["current_delta_rare"]),
                "current_delta_mfe_residual": float(info["current_delta_mfe_residual"]),
                "mu_bio_prior": float(info["mu_bio_prior"]),
                "mu_s_bio": float(info["mu_s_bio"]),
                "mu_gc": float(info["mu_gc"]),
                "mu_rare": float(info["mu_rare"]),
                "mu_mfe": float(info["mu_mfe"]),
                "best_score_in_episode": float(episode_best_candidate["score"]),
                "best_score_z_in_episode": float(episode_best_candidate["score_z"]),
                "best_cai_z_in_episode": float(episode_best_candidate.get("cai_z", float("nan"))),
                "best_s_bio_in_episode": float(episode_best_candidate["s_bio"]),
                "best_bio_prior_z_in_episode": float(episode_best_candidate["bio_prior_z"]),
                "current_seq": info["current_seq"],
                "best_seq_in_episode": episode_best_candidate["sequence"],
                "done_reason": info["done_reason"],
                "td_error": float(update_stats.get("td_error", float("nan"))),
                "td_target": float(update_stats.get("td_target", float("nan"))),
                "value_s_t": float(update_stats.get("value_s_t", float("nan"))),
                "value_s_tp1": float(update_stats.get("value_s_tp1", float("nan"))),
                "actor_loss": float(update_stats.get("actor_loss", float("nan"))),
                "critic_loss": float(update_stats.get("critic_loss", float("nan"))),
                "ac_loss": float(update_stats.get("loss", float("nan"))),
            })

            state = next_state

        episode_mean_s_bio = summarize_episode_s_bio(trace_rows, ep, env.curr_s_bio)
        episode_mean_bio_prior_z = summarize_episode_metric(trace_rows, ep, "current_bio_prior_z", env.curr_bio_prior_z)

        best_train_objective = get_candidate_objective(episode_best_candidate)

        episode_summary = make_episode_summary_row(
              episode=ep,
              episode_start_source=start_source,
              num_steps=int(step_in_ep),
              num_improvements=int(num_improvements),
              done_reason=env.done_reason if env.done_reason is not None else "",
              final_score=float(env.curr_score),
              final_score_z=float(env.curr_score_z),
              final_s_bio=float(env.curr_s_bio),
              final_bio_prior_z=float(env.curr_bio_prior_z),
              final_bio_loglik=float(env.curr_bio_loglik),
              final_objective=float(env.curr_objective),
              final_seq=env.curr_seq,
              best_score=float(episode_best_candidate["score"]),
              best_score_z=float(episode_best_candidate["score_z"]),
              best_s_bio=float(episode_best_candidate["s_bio"]),
              best_bio_prior_z=float(episode_best_candidate["bio_prior_z"]),
              best_bio_loglik=float(episode_best_candidate["bio_loglik"]),
              best_objective=float(best_train_objective),
              best_seq=episode_best_candidate["sequence"],
              episode_mean_s_bio=float(episode_mean_s_bio),
              episode_mean_bio_prior_z=float(episode_mean_bio_prior_z),
        )
        for k, v in env.curr_metrics.items():
            episode_summary[f"final_metric_{k}"] = v
        for k, v in episode_best_metrics.items():
            episode_summary[f"best_metric_{k}"] = v
        episode_summary_rows.append(episode_summary)

        episode_archive_candidate = make_candidate_record(
            sequence=str(episode_best_candidate["sequence"]),
            score=float(episode_best_candidate["score"]),
            score_z=float(episode_best_candidate["score_z"]),
            cai_z=float(episode_best_candidate.get("cai_z", float("nan"))),
            s_bio=float(episode_best_candidate["s_bio"]),
            bio_prior_z=float(episode_best_candidate["bio_prior_z"]),
            bio_loglik=float(episode_best_candidate["bio_loglik"]),
            delta_gc=float(episode_best_candidate.get("delta_gc", 0.0)),
            delta_rare=float(episode_best_candidate.get("delta_rare", 0.0)),
            delta_mfe_residual=float(episode_best_candidate.get("delta_mfe_residual", 0.0)),
            objective=float(episode_best_candidate.get("objective", best_train_objective)),
            seq_id=seq_id,
            episode=int(ep),
            step=int(step_in_ep),
            source=f"{start_source}_episode_best",
            source_role="episode_best",
            candidate_start_source=start_source,
            candidate_stage="episode_best",
            candidate_rollout_steps=int(step_in_ep),
            metrics=episode_best_metrics,
        )
        archive = update_archive(archive, episode_archive_candidate)
        archive = maybe_prune_archive(archive)
        if should_add_to_final_pool("episode_best", ep):
             global_seen_candidates.add(episode_archive_candidate["sequence"])
             final_seen_all.add(episode_archive_candidate["sequence"])
             if is_candidate_feasible_for_deployment(episode_archive_candidate):
                 final_candidate_pool = update_archive(final_candidate_pool, episode_archive_candidate)
                 final_seen_candidates.add(episode_archive_candidate["sequence"])

        current_global_best_candidate = make_candidate_record(
            sequence=global_best_seq,
            score=float(global_best_score),
            score_z=float(global_best_score_z),
            cai_z=float(global_best_cai_z),
            s_bio=float(global_best_s_bio),
            bio_prior_z=float(global_best_bio_prior_z),
            bio_loglik=float(global_best_bio_loglik),
            delta_gc=float(global_best_delta_gc),
            delta_rare=float(global_best_delta_rare),
            delta_mfe_residual=float(global_best_delta_mfe_residual),
            objective=float(global_best_objective),
        )
        if is_better_candidate(episode_best_candidate, current_global_best_candidate, mode="archive"):
            global_best_seq = str(episode_best_candidate["sequence"])
            global_best_score = float(episode_best_candidate["score"])
            global_best_score_z = float(episode_best_candidate["score_z"])
            global_best_cai_z = float(episode_best_candidate.get("cai_z", float("nan")))
            global_best_s_bio = float(episode_best_candidate["s_bio"])
            global_best_bio_prior_z = float(episode_best_candidate["bio_prior_z"])
            global_best_bio_loglik = float(episode_best_candidate.get("bio_loglik", 0.0))
            global_best_objective = float(episode_best_candidate.get("objective", get_candidate_objective(episode_best_candidate)))
            global_best_delta_gc = float(episode_best_candidate.get("delta_gc", 0.0))
            global_best_delta_rare = float(episode_best_candidate.get("delta_rare", 0.0))
            global_best_delta_mfe_residual = float(episode_best_candidate.get("delta_mfe_residual", 0.0))
                        
        target_count, stage = get_candidate_generation_plan(ep)
        if target_count > 0:
            effective_target_count = int(target_count)
            existing_seen_for_generation = None

            # final stage should fill the deployment-feasible pool to FINAL_CANDIDATE_COUNT
            # using deployment-pool uniqueness, not full history uniqueness
            if stage == "final":
                current_feasible_count = count_feasible_final_candidates(final_candidate_pool)
                effective_target_count = max(0, FINAL_CANDIDATE_COUNT - current_feasible_count)
                existing_seen_for_generation = set(final_seen_all)

            final_candidate_target_total_count += int(effective_target_count)
            if stage == "final":
                final_stage_target_count += int(effective_target_count)

            if effective_target_count > 0:
                generated = generate_candidates_with_policy(
                    policy=policy,
                    original_seq=original_seq,
                    global_best_seq=global_best_seq,
                    episode_best_seq=str(episode_best_candidate["sequence"]),
                    seq_id=seq_id,
                    episode=ep,
                    stage=stage,
                    scorer=scorer,
                    target_specs=target_specs,
                    cai_weights=cai_weights,
                    csi_weights=csi_weights,
                    score_ref_stats=score_ref_stats,
                    cai_ref_stats=cai_ref_stats,
                    target_count=effective_target_count,
                    existing_seen=existing_seen_for_generation,
                    target_aa_sequence=target_aa_sequence,
                )

                for cand in generated:
                      if stage == "final":
                          if cand["sequence"] in final_seen_all:
                              continue

                          final_seen_all.add(cand["sequence"])
                          archive = update_archive(archive, cand)
                          archive = maybe_prune_archive(archive)
                          global_seen_candidates.add(cand["sequence"])

                          if is_candidate_feasible_for_deployment(cand):
                              final_candidate_pool = update_archive(final_candidate_pool, cand)
                              final_seen_candidates.add(cand["sequence"])

                      else:
                            if cand["sequence"] in global_seen_candidates:
                                 continue

                            global_seen_candidates.add(cand["sequence"])

                            if should_add_to_final_pool(stage, ep):
                                 if cand["sequence"] not in final_seen_candidates:
                                      if is_candidate_feasible_for_deployment(cand):
                                          final_candidate_pool = update_archive(final_candidate_pool, cand)
                                          final_seen_candidates.add(cand["sequence"])
                                      final_seen_all.add(cand["sequence"])

                            archive = update_archive(archive, cand)
                            archive = maybe_prune_archive(archive)

                      final_candidate_generated_total_count += 1
                      if stage == "final":
                           final_stage_generated_count += 1

        if ep % PRINT_EVERY == 0:
            print(
                   f"[seq {seq_id:04d}] [EP {ep:04d}] "
                   f"best_obj={get_candidate_objective(episode_best_candidate):.4f} | "
                   f"score={float(episode_best_candidate['score']):.4f} | "
                   f"score_z={float(episode_best_candidate['score_z']):.4f} | "
                   f"bio_prior_z={float(episode_best_candidate['bio_prior_z']):.4f}"
             )

    final_candidates = select_final_topk_candidates(list(final_candidate_pool), topk=FINAL_CANDIDATE_COUNT)
    final_best, candidate_pool_size = select_final_best_candidate(list(final_candidate_pool))
    write_table_txt(trace_rows, os.path.join(seq_out_dir, "trace.txt"))
    write_table_txt(episode_summary_rows, os.path.join(seq_out_dir, "episode_summary.txt"))
    write_table_txt(final_candidates, os.path.join(seq_out_dir, "final_candidates.txt"))
    final_top10 = select_final_topk_candidates(list(final_candidate_pool), topk=FINAL_TOPK_PER_RUN)
    write_table_txt(final_top10, os.path.join(seq_out_dir, "final_top10.txt"))
    write_sequence_txt(final_best["sequence"], os.path.join(seq_out_dir, "final_best.txt"))
    write_json_txt(final_best, os.path.join(seq_out_dir, "final_best_meta.txt"))

    return {
        "seq_id": seq_id,
        "original_seq": original_seq,
        "best_sequence": final_best["sequence"],
        "best_score": final_best["score"],
        "best_score_z": final_best["score_z"],
        "best_s_bio": final_best["s_bio"],
        "best_bio_prior_z": final_best["bio_prior_z"],
        "best_objective": final_best["objective"],
        "candidate_pool_size": candidate_pool_size,
        "generated_candidates": final_stage_generated_count,
        "target_candidates": final_stage_target_count,
        "generated_candidates_total": final_candidate_generated_total_count,
        "target_candidates_total": final_candidate_target_total_count,
        "final_topk_per_run": FINAL_TOPK_PER_RUN,
    }

# =========================================================
# 11. Main
# =========================================================
def resolve_ensemble_model_paths(primary_model_path: str) -> List[str]:
    paths = []
    for p in ENSEMBLE_MODEL_PATHS:
        if os.path.exists(p):
            ap = os.path.abspath(p)
            if ap not in paths:
                paths.append(ap)
    if AUTO_DISCOVER_ENSEMBLE:
        model_dir = os.path.dirname(primary_model_path)
        patterns = [
            os.path.join(model_dir, "best_model*.p"),
            os.path.join(model_dir, "best_model*.pt"),
            os.path.join(model_dir, "best_model*.pth"),
            os.path.join(model_dir, "ensemble_*.pt"),
            os.path.join(model_dir, "ensemble_*.pth"),
        ]
        for patt in patterns:
            for p in sorted(glob(patt)):
                ap = os.path.abspath(p)
                if ap not in paths:
                    paths.append(ap)
    if len(paths) == 0:
        raise FileNotFoundError(f"No model checkpoints found for ensemble near: {primary_model_path}")
    if len(paths) < MIN_ENSEMBLE_MEMBERS and not ALLOW_SINGLE_MODEL_FALLBACK:
        raise RuntimeError(f"[ERROR] Ensemble guard triggered: found {len(paths)} checkpoint(s), but MIN_ENSEMBLE_MEMBERS={MIN_ENSEMBLE_MEMBERS}")
    return paths



def prepare_common_inputs_for_controller() -> None:
    """Parent-process lightweight preparation; heavy CUDA/model objects are loaded in workers."""
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] OUT_DIR ready: {OUT_DIR}", flush=True)
    print(f"[INFO] DEVICE(parent) = {DEVICE}", flush=True)
    print(f"[INFO] Start sequence file: {ORIGINAL_FILE}", flush=True)
    print("[INFO] Parallel multi-start design: 10 independent random lower-tail Gluc CDS starts × 1 run each", flush=True)
    print(f"[INFO] Max parallel start jobs: {MAX_PARALLEL_START_JOBS}", flush=True)
    print(f"[INFO] Final candidate count per run: {FINAL_CANDIDATE_COUNT}", flush=True)
    print(f"[INFO] Final TopK per start/run: {FINAL_TOPK_PER_RUN}", flush=True)

    cai_weights = load_weight_table(CAI_WEIGHT_FILE)
    ensure_weight_table_complete(cai_weights, "CAI weight table")

    original_seqs = load_sequences_onecol(ORIGINAL_FILE)
    if len(original_seqs) == 0:
        raise ValueError("[ERROR] No valid start sequences found.")
    print(f"[INFO] Start CDS sequences loaded: {len(original_seqs)}", flush=True)
    if len(original_seqs) != int(EXPECTED_START_COUNT):
        print(
            f"[WARN] Expected {EXPECTED_START_COUNT} start sequences, "
            f"but loaded {len(original_seqs)} from {ORIGINAL_FILE}",
            flush=True,
        )

    target_aa = clean_aa_sequence(TARGET_AA_SEQUENCE)
    write_sequence_txt(target_aa, os.path.join(OUT_DIR, "target_aa_sequence.txt"))
    print(f"[INFO] TARGET_AA_SEQUENCE loaded as synonymous constraint. AA length = {len(target_aa)}", flush=True)

    start_summary_rows = []
    for start_id, seq in enumerate(original_seqs, start=1):
        sense, stop = split_cds_and_stop(seq)
        row = {
            "start_id": int(start_id),
            "length_nt": int(len(seq)),
            "sense_codons": int(len(sense)),
            "stop_codon": stop,
            "target_aa_length": int(len(target_aa)),
            "cai": float(calc_geom_index(seq, cai_weights, metric_name="CAI")),
            "gc": float(calc_gc(seq)),
            "rare": float(calc_rare(seq, cai_weights, threshold=RARE_THRESHOLD)),
            "cds_sequence": seq,
        }
        start_summary_rows.append(row)
        print(
            f"[INFO] start_id={start_id:02d} | len_nt={len(seq)} | "
            f"sense_codons={len(sense)} | stop={stop} | target_aa_len={len(target_aa)} | "
            f"CAI={row['cai']:.6f} | GC={row['gc']:.6f} | Rare={row['rare']:.6f}",
            flush=True,
        )

    write_table_txt(start_summary_rows, os.path.join(OUT_DIR, "input_start_sequence_summary.txt"))


def load_common_heavy_objects(original_seqs: List[str]):
    """Worker-process heavy preparation; avoids CUDA/fork/pickling problems."""
    cai_weights = load_weight_table(CAI_WEIGHT_FILE)
    csi_weights = load_weight_table(CSI_WEIGHT_FILE)
    ensure_weight_table_complete(cai_weights, "CAI weight table")
    ensure_weight_table_complete(csi_weights, "CSI weight table")

    print(f"[INFO] Normalization reference CDS file: {NORMALIZATION_REF_CDS_FILE}", flush=True)
    norm_ref_seqs = load_reference_sequences_for_normalization(
        metrics_file=METRICS_FILE,
        fallback_seqs=original_seqs,
        max_count=NORM_REF_MAX_SEQS,
        ref_cds_file=NORMALIZATION_REF_CDS_FILE,
    )
    print(f"[INFO] Normalization reference sequences loaded: {len(norm_ref_seqs)}", flush=True)

    target_specs = fit_bio_gmm_from_reference(METRICS_FILE, MFE_SOURCE_FILE)
    print(
        f"[INFO] Bio GMM fitted. features={target_specs['feature_names']} | "
        f"loglik_mean={target_specs['train_loglik_mean']:.4f}",
        flush=True,
    )

    model_paths = resolve_ensemble_model_paths(MODEL_PATH)
    scorer = EnsembleTranslationScorer(
        model_config_path=MODEL_CONFIG,
        model_paths=model_paths,
        condition_npz=HEK293T_CONDITION_NPZ,
        device=DEVICE,
        uncertainty_alpha=ENSEMBLE_UNCERTAINTY_ALPHA,
    )

    print("[INFO] Computing score normalization statistics on HEK293T high-expression reference sequences...", flush=True)
    score_ref_stats = compute_reference_score_stats(norm_ref_seqs, scorer)
    print(
        f"[INFO] Score reference stats: mean={score_ref_stats['mean']:.6f} | "
        f"std={score_ref_stats['std']:.6f} | n={score_ref_stats['count']}",
        flush=True,
    )

    print("[INFO] Computing CAI normalization statistics on HEK293T high-expression reference sequences...", flush=True)
    cai_ref_stats = compute_reference_cai_stats(norm_ref_seqs, cai_weights)
    print(
        f"[INFO] CAI reference stats: mean={cai_ref_stats['mean']:.6f} | "
        f"std={cai_ref_stats['std']:.6f} | n={cai_ref_stats['count']}",
        flush=True,
    )

    return scorer, target_specs, cai_weights, csi_weights, score_ref_stats, cai_ref_stats


def run_single_start_worker(start_id: int) -> Dict:
    """Run exactly one start sequence in the current process."""
    os.makedirs(OUT_DIR, exist_ok=True)
    original_seqs = load_sequences_onecol(ORIGINAL_FILE)
    if len(original_seqs) == 0:
        raise ValueError("[ERROR] No valid start sequences found.")
    if start_id < 1 or start_id > len(original_seqs):
        raise ValueError(f"[ERROR] start_id={start_id} is out of range 1..{len(original_seqs)}")

    seq = original_seqs[start_id - 1]
    run_id = int(start_id)
    run_seed = int(RUN_SEED_BASE + start_id - 1)
    set_all_random_seeds(run_seed)

    print(f"[INFO] Worker PID={os.getpid()} | DEVICE={DEVICE} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(
        f"[INFO] Single-start worker running start_id={start_id}/{len(original_seqs)} | "
        f"run_id={run_id} | run_seed={run_seed}",
        flush=True,
    )

    scorer, target_specs, cai_weights, csi_weights, score_ref_stats, cai_ref_stats = load_common_heavy_objects(original_seqs)

    print("[INFO] Running smoke test score on this start sequence...", flush=True)
    print(f"[TEST SCORE] {scorer.score(seq):.6f}", flush=True)

    result = train_one_sequence(
        seq_id=run_id,
        original_seq=seq,
        scorer=scorer,
        target_specs=target_specs,
        cai_weights=cai_weights,
        csi_weights=csi_weights,
        score_ref_stats=score_ref_stats,
        cai_ref_stats=cai_ref_stats,
        target_aa_sequence=TARGET_AA_SEQUENCE,
    )

    result["start_id"] = int(start_id)
    result["run_id"] = int(run_id)
    result["run_seed"] = int(run_seed)
    result["independent_start"] = True
    result["parallel_worker"] = True
    result["worker_pid"] = int(os.getpid())
    result["start_sequence_file"] = ORIGINAL_FILE
    result["final_topk_per_run"] = FINAL_TOPK_PER_RUN

    seq_out_dir = os.path.join(OUT_DIR, f"seq_{run_id:04d}")
    os.makedirs(seq_out_dir, exist_ok=True)
    write_json_txt(result, os.path.join(seq_out_dir, "run_result_summary.json"))
    write_table_txt([result], os.path.join(seq_out_dir, "run_result_summary.txt"))
    print(f"[INFO] Worker start_id={start_id} done.", flush=True)
    return result


def _parse_visible_gpu_ids() -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip() != ""]


def _launch_parallel_workers() -> None:
    """Parent controller: launch one Python subprocess per start sequence."""
    prepare_common_inputs_for_controller()

    original_seqs = load_sequences_onecol(ORIGINAL_FILE)
    total_runs = len(original_seqs)
    max_jobs = max(1, min(int(MAX_PARALLEL_START_JOBS), total_runs))
    gpu_ids = _parse_visible_gpu_ids()
    log_dir = os.path.join(OUT_DIR, PARALLEL_LOG_DIR_NAME)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Launching true parallel workers: total_runs={total_runs}, max_jobs={max_jobs}", flush=True)
    if gpu_ids:
        print(f"[INFO] Parent visible GPU ids: {gpu_ids}; workers will be assigned round-robin if more than one GPU is visible.", flush=True)
    else:
        print("[INFO] CUDA_VISIBLE_DEVICES is empty in parent; workers inherit default GPU visibility.", flush=True)

    script_path = os.path.abspath(__file__)
    pending_start_ids = list(range(1, total_runs + 1))
    running = []
    failed = []
    completed = []

    def start_one(start_id: int):
        env = os.environ.copy()
        env[SINGLE_START_ENV] = str(start_id)
        env["RL_PARENT_PID"] = str(os.getpid())
        if len(gpu_ids) > 1:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[(start_id - 1) % len(gpu_ids)]
        log_path = os.path.join(log_dir, f"start_{start_id:02d}.log")
        log_fh = open(log_path, "w", encoding="utf-8")
        print(
            f"[INFO] Launch start_id={start_id:02d} | log={log_path} | "
            f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}",
            flush=True,
        )
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd(),
            text=True,
        )
        return {"start_id": start_id, "proc": proc, "log_fh": log_fh, "log_path": log_path, "t0": time.time()}

    while pending_start_ids or running:
        while pending_start_ids and len(running) < max_jobs:
            running.append(start_one(pending_start_ids.pop(0)))

        time.sleep(10)
        still_running = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            item["log_fh"].close()
            elapsed = time.time() - item["t0"]
            if rc == 0:
                completed.append(item["start_id"])
                print(f"[INFO] Completed start_id={item['start_id']:02d} | elapsed={elapsed:.1f}s", flush=True)
            else:
                failed.append((item["start_id"], rc, item["log_path"]))
                print(
                    f"[ERROR] Failed start_id={item['start_id']:02d} | rc={rc} | log={item['log_path']}",
                    flush=True,
                )
        running = still_running

    summary_rows = []
    for start_id in range(1, total_runs + 1):
        summary_json = os.path.join(OUT_DIR, f"seq_{start_id:04d}", "run_result_summary.json")
        if os.path.exists(summary_json):
            with open(summary_json, "r", encoding="utf-8") as f:
                summary_rows.append(json.load(f))
        else:
            summary_rows.append({
                "start_id": int(start_id),
                "run_id": int(start_id),
                "status": "missing_summary",
                "summary_json": summary_json,
            })

    write_table_txt(summary_rows, os.path.join(OUT_DIR, "all_sequence_summary.txt"))
    write_json_txt(
        {
            "total_runs": int(total_runs),
            "max_parallel_jobs": int(max_jobs),
            "completed": completed,
            "failed": failed,
            "log_dir": log_dir,
        },
        os.path.join(OUT_DIR, "parallel_run_summary.json"),
    )

    if failed:
        raise RuntimeError(f"[ERROR] Some parallel workers failed: {failed}")

    print("[INFO] All parallel workers completed successfully.", flush=True)
    print(f"[INFO] Combined summary: {os.path.join(OUT_DIR, 'all_sequence_summary.txt')}", flush=True)


def main() -> None:
    single_start = os.environ.get(SINGLE_START_ENV, "").strip()
    if single_start:
        run_single_start_worker(int(single_start))
        return

    if PARALLEL_MULTI_START:
        _launch_parallel_workers()
        return

    # Fallback sequential mode, mainly for debugging.
    os.makedirs(OUT_DIR, exist_ok=True)
    original_seqs = load_sequences_onecol(ORIGINAL_FILE)
    if len(original_seqs) == 0:
        raise ValueError("[ERROR] No valid start sequences found.")
    scorer, target_specs, cai_weights, csi_weights, score_ref_stats, cai_ref_stats = load_common_heavy_objects(original_seqs)
    summary_rows = []
    for start_id, seq in enumerate(original_seqs, start=1):
        run_id = int(start_id)
        run_seed = int(RUN_SEED_BASE + start_id - 1)
        set_all_random_seeds(run_seed)
        result = train_one_sequence(
            seq_id=run_id,
            original_seq=seq,
            scorer=scorer,
            target_specs=target_specs,
            cai_weights=cai_weights,
            csi_weights=csi_weights,
            score_ref_stats=score_ref_stats,
            cai_ref_stats=cai_ref_stats,
            target_aa_sequence=TARGET_AA_SEQUENCE,
        )
        result["start_id"] = int(start_id)
        result["run_id"] = int(run_id)
        result["run_seed"] = int(run_seed)
        result["independent_start"] = True
        result["parallel_worker"] = False
        summary_rows.append(result)
    write_table_txt(summary_rows, os.path.join(OUT_DIR, "all_sequence_summary.txt"))
    print("[INFO] Done.", flush=True)

if __name__ == "__main__":
    main()
