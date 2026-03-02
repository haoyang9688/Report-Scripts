# 🌿🎈🌿🎈🌿🎈🌿 The Computational biology test 🌿🎈🌿🎈🌿🎈🌿


This repository contains an end-to-end, **reproducible** pipeline to:

1. **Merge TCGA STAR gene counts** (BLCA), generate **logCPM**, map **Ensembl → gene symbols**, and filter low-expression genes.
2. Build **patient-level OS survival table** from the GDC API.
3. Perform **tumor vs normal differential expression (DE)** and derive a candidate gene list.
4. Train a **Cox model** on TCGA and **validate on GSE13507**, optionally using **multimodal distillation** (methylation teacher → RNA student).
5. Compare against **published BLCA signatures**.
6. Process **GSE135337 scRNA-seq** and run **scGPT-based in silico perturbation** (KD/OE) to quantify cell-type and pathway sensitivity.

---

## 😊Repository layout

Key scripts:

- `step1_tcga_merge_star_counts.py` — merge STAR gene-level counts into one matrix (optionally logCPM).
- `step2_tcga_rna_to_symbol_and_filter.py` — Ensembl → symbol + CPM filtering.
- `step3_tcga_make_patient_expr_and_survival.py` — sample→patient collapse + OS survival from GDC API.
- `step4_tcga_de_and_pca.py` — tumor vs normal DE + PCA (+ figures).
- `step5_gse13507_probe_to_symbol.py` — probe → gene symbol for GSE13507 (GPL6102).
- `step6_train_cox_and_validate_gse13507.py` — Cox training on TCGA and validation on GSE13507 (ridge / teacher_only / student_distill).
- `step7_compare_published_signature_TEMPLATE_3models.py` — compare published signatures and summarize metrics.
- `step8_gse135337_merge_and_preprocess.py` — merge and preprocess GSE135337 scRNA; keep HVGs ∪ key genes; save `layers["counts"]`.
- `step9_scgpt_perturbation.py` — scGPT perturbation (KD/OE) + Δp heatmaps + pathway perturbation heatmaps.
- `stepM_tcga_methyl_fusion.py` — (optional) methylation-only and fusion Cox models.
- `utils_gene.py`, `utils_survival.py` — gene alias + survival utilities.

---

## 🤗Requirements

### 🤗System
- Linux recommended
- Python **3.9+** (3.10 suggested)

### 🤗Python packages
Bulk pipeline (Steps 1–7):
- numpy, pandas, scipy
- scikit-learn
- statsmodels
- matplotlib
- lifelines (for methylation fusion script)

Single-cell pipeline (Steps 8–9):
- scanpy, anndata, leidenalg, igraph
- torch (GPU recommended)
- scgpt
- scikit-learn, matplotlib

> Tip: keep **two conda envs**: one for bulk (`bulk_env`) and one for scGPT (`scgpt_env`).

Example environment setup (edit versions to match your machine):
```bash
# bulk
conda create -n bulk_env python=3.10 -y
conda activate bulk_env
pip install numpy pandas scipy scikit-learn statsmodels matplotlib lifelines

# single-cell + scGPT
conda create -n scgpt_env python=3.10 -y
conda activate scgpt_env
pip install numpy pandas scipy scikit-learn matplotlib scanpy anndata leidenalg igraph torch scgpt
```

---

## 👉👉👉Data preparation

🤗🤗🤗This section describes **exactly which datasets are used** and how to organize them for this pipeline.

**Download source (pre-downloaded files provided by this study)**
```text
https://drive.google.com/drive/folders/1lxx6em_xachidsaOuJBURZaBc_NCNkEk?usp=drive_link
```

### 1) Training set: TCGA-BLCA (bulk RNA-seq + clinical survival; DNA methylation optional)

**Required**
- **Bulk RNA-seq gene expression (STAR-Counts)** for the **TCGA-BLCA** cohort
- **Clinical survival information** (overall survival), used in Step 3 (queried automatically from GDC API by patient ID)

**Optional**
- **DNA methylation** for TCGA-BLCA (used only if you run `stepM_tcga_methyl_fusion.py`)

**Download portal**
```text
https://portal.gdc.cancer.gov/projects/TCGA-BLCA
```

**What to download (recommended filters in GDC portal)**
- Project: `TCGA-BLCA`
- Data Category: `Transcriptome Profiling`
- Data Type: `Gene Expression Quantification`
- Workflow Type: `STAR - Counts`
- Sample Types: `Primary Tumor`, `Solid Tissue Normal`

After adding files to the cart, export:
- the **downloaded STAR count files** (GDC Data Transfer Tool output), and
- the corresponding `metadata.cart.*.json` (used by Step 1 / Step 2).

**Expected local layout (example)**
```text
Training/
  Bulk_RNA/
    metadata.cart.YYYY-MM-DD.json
    <file_id_1>/
      *.star_gene_counts.tsv
    <file_id_2>/
      *.star_gene_counts.tsv
```

If you run methylation (optional), prepare:
```text
Training/
  DNA_methylation/
    metadata.cart.YYYY-MM-DD.json
    <file_id_x>/
      *.txt   (beta values / methylation table as provided by GDC)
```

---

### 2) External test set: GSE13507 (GEO; 165 BLCA tumors, GSM340605–GSM340769)

This dataset is used as an **external validation cohort** in Step 6 and Step 7.

**Download source**
```text
https://www.ncbi.nlm.nih.gov/geo/
```

**Dataset**
- Series: **GSE13507**
- Samples: **GSM340605–GSM340769** (165 bladder tumor expression profiles)

**What you need for this pipeline**
- Expression matrix (probe-level or gene-level)
- Clinical table containing survival time and event status (your `--gse_clin_tsv`)

If the expression matrix is probe-level (typical microarray), run `step5_gse13507_probe_to_symbol.py` to map probes → gene symbols and collapse duplicates.

**Expected local layout (example)**
```text
Test/
  GSE13507/
    GSE13507_expr_*.tsv(.gz)                 # expression (probe or gene symbol)
    GSE13507_clinical_info.*.tsv             # survival / clinical info used by Step6/7
```

---

### 3) Single-cell BLCA cohort for perturbation: GSE135337 (7 tumors, exclude GSM5329919)

This dataset is used only for **scGPT in silico perturbation** (Step 8–9) to assess cell-type and pathway sensitivity of the prognostic genes.

**Dataset**
- Series: **GSE135337**
- Use: **7 bladder tumor samples**, **excluding GSM5329919**


**Expected input format for Step 8**
Place the downloaded per-sample tables under:
```text
BLCA/
  GSE135337/
    GSM*_gene_cell_exprs_table.txt.gz        # genes × cells count table per sample
```

Step 8 merges all samples into a single `*.h5ad` and preserves raw counts in `adata.layers["counts"]` for Step 9 perturbation.


---

## 🤗Full pipeline (commands)

Below are the commands used for the complete run (adapt paths to your system).

> **Convention used below**
> - Project root: `/data2/liu_hy/Hebei`
> - Outputs written to `processed/`

---

## 🤗Step 1 — Merge TCGA STAR counts (and logCPM)

```bash
cd /data2/liu_hy/Hebei
python3 step1_tcga_merge_star_counts.py \
  --download_dir /data2/liu_hy/Hebei/Training/Bulk_RNA \
  --metadata_json /data2/liu_hy/Hebei/Training/Bulk_RNA/metadata.cart.2026-02-27.json \
  --out_dir /data2/liu_hy/Hebei/processed/tcga_rna \
  --counts_col unstranded \
  --keep_sample_types "Primary Tumor,Solid Tissue Normal" \
  --fill_from_api \
  --dedup_by_sample largest_lib \
  --make_logcpm
```

Outputs (in `--out_dir`):
- `tcga_blca_counts.tsv.gz`
- `tcga_blca_logcpm.tsv.gz` (if `--make_logcpm`)
- `tcga_blca_sample_info.tsv`

Quick QC (tumor vs normal counts):
```bash
python3 - <<'PY'
import gzip, re
f="/data2/liu_hy/Hebei/processed/tcga_rna/tcga_blca_counts.tsv.gz"
with gzip.open(f,"rt") as fh:
    cols=fh.readline().rstrip("\n").split("\t")[1:]
print("n_cols:", len(cols))
print("tumor(-01):", sum(bool(re.search(r"-01[A-Z]", c)) for c in cols))
print("normal(-11):", sum(bool(re.search(r"-11[A-Z]", c)) for c in cols))
PY
```

---

## 🤗Step 2 — Ensembl → gene symbol + CPM filter

```bash
python3 step2_tcga_rna_to_symbol_and_filter.py \
  --tcga_counts_gz /data2/liu_hy/Hebei/processed/tcga_rna/tcga_blca_counts.tsv.gz \
  --download_dir   /data2/liu_hy/Hebei/Training/Bulk_RNA \
  --metadata_json  /data2/liu_hy/Hebei/Training/Bulk_RNA/metadata.cart.2026-02-27.json \
  --sample_info_tsv /data2/liu_hy/Hebei/processed/tcga_rna/tcga_blca_sample_info.tsv \
  --out_dir        /data2/liu_hy/Hebei/processed/tcga_rna_symbol \
  --collapse_dup_symbols sum \
  --min_cpm 1.0 \
  --min_frac_samples 0.1
```

Outputs:
- `tcga_blca_counts.symbol.tsv.gz`
- `tcga_blca_logcpm.symbol.tsv.gz`
- `tcga_gene_id_to_symbol.tsv`
- `tcga_blca_sample_info.tsv` (copied)

QC:
```bash
python3 - <<'PY'
import gzip, re
f="/data2/liu_hy/Hebei/processed/tcga_rna_symbol/tcga_blca_logcpm.symbol.tsv.gz"
with gzip.open(f,"rt") as fh:
    cols=fh.readline().rstrip("\n").split("\t")[1:]
print("n_cols:", len(cols))
print("TCGA-like:", sum(c.startswith("TCGA-") for c in cols))
print("tumor(-01):", sum(bool(re.search(r"-01[A-Z]", c)) for c in cols))
print("normal(-11):", sum(bool(re.search(r"-11[A-Z]", c)) for c in cols))
print("example:", cols[:5])
PY
```

---

## 🤗Step 3 — Patient-level expression + OS survival (GDC API)

This step:
- collapses sample → patient (`--collapse mean|median`)
- queries GDC cases API to build OS table

```bash
python3 step3_tcga_make_patient_expr_and_survival.py \
  --logcpm_symbol_gz /data2/liu_hy/Hebei/processed/tcga_rna_symbol/tcga_blca_logcpm.symbol.tsv.gz \
  --out_dir /data2/liu_hy/Hebei/processed/tcga_patient_os \
  --collapse mean
```

Outputs:
- `tcga_blca_logcpm.symbol.patient.tsv.gz`
- `tcga_blca_survival_os.tsv`

> Note: requires Internet access to query `https://api.gdc.cancer.gov/cases`.

---

## 🤗Step 4 — Tumor vs Normal DE + PCA (candidate gene list)

```bash
python3 step4_tcga_de_and_pca.py \
  --logcpm_symbol_gz /data2/liu_hy/Hebei/processed/tcga_rna_symbol/tcga_blca_logcpm.symbol.tsv.gz \
  --sample_info_tsv  /data2/liu_hy/Hebei/processed/tcga_rna_symbol/tcga_blca_sample_info.tsv \
  --out_dir          /data2/liu_hy/Hebei/processed/tcga_de_pca \
  --top_n 500
```

Outputs:
- `tcga_blca_DE_tumor_vs_normal.tsv`
- `tcga_blca_top500_DE_genes.txt`
- `tcga_blca_PCA_samples.tsv`
- figures in `figures/`: `volcano.png`, `pca.png`, `heatmap_top50.png`, `umap.png`

---

## 🤗Step 5 — GSE13507 probe → gene symbol (GPL6102)

If you start from probe-level microarray expression:

```bash
python3 step5_gse13507_probe_to_symbol.py \
  --expr_tsv /path/to/GSE13507_probe_expr.tsv \
  --out_tsv_gz /data2/liu_hy/Hebei/Test/GSE13507/GSE13507_expr_geneSymbol.tsv.gz \
  --collapse median
```

Sanity check (gene × sample):
```bash
python3 - <<'PY'
import gzip, pandas as pd
f="/data2/liu_hy/Hebei/Test/GSE13507/GSE13507_expr_geneSymbol_GSM340605_340769.aliasfix.tsv.gz"
with gzip.open(f,"rt") as fh:
    header = fh.readline().rstrip("\n").split("\t")
print("n_cols (incl gene col):", len(header))
print("first 5 columns:", header[:5])

with gzip.open(f,"rt") as fh:
    for i in range(5):
        ln = fh.readline().rstrip("\n").split("\t")
        print("row", i, "gene:", ln[0], "first_value:", ln[1])
PY
```

---

## 🤗Optional — Methylation teacher / fusion models (StepM)

`stepM_tcga_methyl_fusion.py` supports:
- methyl_only
- early_fusion
- late_fusion
- intermediate_fusion (PCA then concat)

Example: methyl-only
```bash
python3 stepM_tcga_methyl_fusion.py \
  --methyl_input "/data2/liu_hy/Hebei/Training/DNA_methylation" \
  --methyl_metadata_json "/data2/liu_hy/Hebei/Training/DNA_methylation/metadata.cart.2026-02-27.json" \
  --survival_tsv "/data2/liu_hy/Hebei/processed/tcga_patient_os/tcga_blca_survival_os.simple.tsv" \
  --out_dir "/data2/liu_hy/Hebei/processed/methyl_fusion_runs/methyl_only" \
  --model methyl_only \
  --top_k_meth 2000 \
  --l2 1e-2
```

Outputs (in `--out_dir`):
- `risk_methyl_only.tsv`
- `cv_results.tsv`
- `km_methyl_only.png`

If you use the methylation teacher risk as input to Step6, ensure it intersects the TCGA patients used for OS:
```bash
python3 - <<'PY'
import pandas as pd, gzip

tcga_expr = "/data2/liu_hy/Hebei/processed/tcga_patient_os/tcga_blca_logcpm.symbol.patient.tsv.gz"
tcga_surv = "/data2/liu_hy/Hebei/processed/tcga_patient_os/tcga_blca_survival_os.tsv"
teacher   = "/data2/liu_hy/Hebei/processed/methyl_fusion_runs/methyl_only/risk_methyl_only.tsv"
out       = "/data2/liu_hy/Hebei/processed/methyl_fusion_runs/methyl_only/risk_methyl_only.intersect324.tsv"

with gzip.open(tcga_expr, "rt") as f:
    head = f.readline().rstrip("\n").split("\t")
expr_ids = set(head[1:])

surv = pd.read_csv(tcga_surv, sep="\t", usecols=["patient_id"])
keep = set(surv["patient_id"].astype(str)) & expr_ids

tea = pd.read_csv(teacher, sep="\t")
tea["patient_id"] = tea["patient_id"].astype(str)
tea = tea[tea["patient_id"].isin(keep)].copy()
tea = tea.sort_values("patient_id").drop_duplicates("patient_id", keep="first")

tea.to_csv(out, sep="\t", index=False)
print("wrote:", out, "N=", tea.shape[0])
PY
```

---

## 🤗Step 6 — Train Cox model (TCGA) and validate on GSE13507

Student distillation (RNA student) with a precomputed teacher risk:
```bash
python3 step6_train_cox_and_validate_gse13507.py \
  --model student_distill \
  --tcga_expr_gz /data2/liu_hy/Hebei/processed/tcga_patient_os/tcga_blca_logcpm.symbol.patient.tsv.gz \
  --tcga_survival_tsv /data2/liu_hy/Hebei/processed/tcga_patient_os/tcga_blca_survival_os.tsv \
  --candidate_genes /data2/liu_hy/Hebei/processed/tcga_de_pca/tcga_blca_top500_DE_genes.txt \
  --teacher_risk_tsv /data2/liu_hy/Hebei/processed/methyl_fusion_runs/methyl_only/risk_methyl_only.intersect324.tsv \
  --hp_search --n_trials 20 \
  --univ_fdr 0.2 \
  --max_genes 39 \
  --gse_expr_tsv /data2/liu_hy/Hebei/Test/GSE13507/GSE13507_expr_geneSymbol_GSM340605_340769.aliasfix.tsv.gz \
  --gse_clin_tsv /data2/liu_hy/Hebei/Test/GSE13507/GSE13507_clinical_info.GSM340605_340769.tsv \
  --out_dir /data2/liu_hy/Hebei/processed/cox_tcga_train_gse_test
```

Main outputs:
- `model_genes.tsv` (selected genes; default up to 39)
- `tcga_patient_risk.tsv`, `gse13507_risk.tsv`
- `tcga_univariate_cox.tsv`
- `tcga_multivariate_cox_coefs.tsv`
- `cv_results.tsv`
- figures: `km_tcga.png`, `km_gse13507.png`
- `MODEL_CARD.json`

---

## 🤗Step 7 — Compare with published signatures

```bash
python3 step7_compare_published_signature_TEMPLATE_3models.py \
  --models all \
  --cutoff auto \
  --missing_gene_policy fill0 \
  --tcga_expr_gz /data2/liu_hy/Hebei/processed/tcga_patient_os/tcga_blca_logcpm.symbol.patient.tsv.gz \
  --tcga_survival_tsv /data2/liu_hy/Hebei/processed/tcga_patient_os/tcga_blca_survival_os.tsv \
  --gse_expr_tsv /data2/liu_hy/Hebei/Test/GSE13507/GSE13507_expr_geneSymbol_GSM340605_340769.aliasfix.tsv.gz \
  --gse_clin_tsv /data2/liu_hy/Hebei/Test/GSE13507/GSE13507_clinical_info.GSM340605_340769.tsv \
  --out_dir /data2/liu_hy/Hebei/processed/published_3models_compare_with_ours
```

Outputs:
- per-model folders under `--out_dir/`
- summary tables: `summary_metrics.tsv`, `summary_metrics.json`

---

## 🤗Step 8 — GSE135337 scRNA merge + preprocess (Scanpy)

```bash
python3 step8_gse135337_merge_and_preprocess.py \
  --in_dir /data2/liu_hy/Hebei/BLCA/GSE135337 \
  --keep_genes_tsv /data2/liu_hy/Hebei/processed/cox_tcga_train_gse_test/model_genes.tsv \
  --out_h5ad /data2/liu_hy/Hebei/processed/gse135337_sc_processed_keepkey.h5ad \
  --out_dir  /data2/liu_hy/Hebei/processed/gse135337_sc_processed_keepkey \
  --n_hvg 8000 \
  --min_genes 200 --min_cells 3 --max_mt_pct 20 \
  --n_pcs 30 --leiden_res 0.8
```

Outputs:
- `gse135337_sc_processed_keepkey.h5ad`
- QC figures (in `--out_dir/figures/`): `qc_violin.png`, `umap_leiden_sample.png`

> Important: this script saves raw counts into `adata.layers["counts"]`, which Step9 requires.

---

## 🤗Step 9 — scGPT in silico perturbation (KD/OE)

### Step 9a — write `celltype_pred` into the h5ad (recommended)

`step9_scgpt_perturbation.py` can stratify sensitivity by `--label_col`.  
To avoid using Leiden clusters (too granular/noisy), create coarse labels first:

```bash
python3 - <<'PY'
import os, scanpy as sc, matplotlib.pyplot as plt

in_h5ad="/data2/liu_hy/Hebei/processed/gse135337_sc_processed_keepkey.h5ad"
out_dir="/data2/liu_hy/Hebei/processed/gse135337_sc_processed_keepkey/step8b_markers"
os.makedirs(out_dir, exist_ok=True)

adata=sc.read_h5ad(in_h5ad)

# marker genes (cluster-level)
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=True)
sc.pl.rank_genes_groups(adata, n_genes=10, show=False)
plt.savefig(os.path.join(out_dir,"rank_genes_groups_top10.png"), dpi=200, bbox_inches="tight"); plt.close()

# coarse cell type annotation
marker_sets = {
  "epithelial_like": ["EPCAM","KRT8","KRT18","KRT19"],
  "immune_like": ["PTPRC","CD3D","CD3E","NKG7"],
  "myeloid_like": ["LYZ","S100A8","S100A9","FCGR3A"],
  "fibroblast_like": ["COL1A1","COL1A2","DCN","LUM"],
  "endothelial_like": ["PECAM1","VWF","KDR"],
}

for ct, genes in marker_sets.items():
    genes=[g for g in genes if (adata.raw is not None and g in adata.raw.var_names)]
    if len(genes) >= 2:
        sc.tl.score_genes(adata, gene_list=genes, score_name=f"score_{ct}", use_raw=True)

score_cols=[c for c in adata.obs.columns if c.startswith("score_")]
adata.obs["celltype_pred"]=adata.obs[score_cols].idxmax(axis=1).str.replace("score_","", regex=False)

sc.pl.umap(adata, color=["leiden","celltype_pred","sample"], show=False)
plt.savefig(os.path.join(out_dir,"umap_celltype_pred.png"), dpi=200, bbox_inches="tight"); plt.close()

adata.write(in_h5ad)
print("[OK] wrote celltype_pred into:", in_h5ad)
print("[OK] figures in:", out_dir)
PY
```

### Step 9b — choose Top10 genes (|coef|) that exist in scRNA

```bash
python3 - <<'PY'
import pandas as pd, scanpy as sc

tab="/data2/liu_hy/Hebei/processed/cox_tcga_train_gse_test/signature_plots/signature_39gene_cox_table.tsv"
h5ad="/data2/liu_hy/Hebei/processed/gse135337_sc_processed_keepkey.h5ad"
out="/data2/liu_hy/Hebei/processed/cox_tcga_train_gse_test/key_genes_top10_sc_present.tsv"

df=pd.read_csv(tab, sep="\t")
df["abs_coef"]=df["coef"].abs()
top=df.sort_values("abs_coef", ascending=False).head(10)[["gene"]].copy()

adata=sc.read_h5ad(h5ad)
genes=set(adata.raw.var_names) if adata.raw is not None else set(adata.var_names)
top=top[top["gene"].isin(genes)].copy()

top.to_csv(out, sep="\t", index=False)
print("[OK] wrote:", out, "n=", top.shape[0])
print("genes:", top["gene"].tolist())
PY
```

### Step 9c — run scGPT perturbation

```bash
conda activate scgpt_env

python3 step9_scgpt_perturbation.py \
  --in_h5ad /data2/liu_hy/Hebei/processed/gse135337_sc_processed_keepkey.h5ad \
  --keep_genes_tsv /data2/liu_hy/Hebei/processed/cox_tcga_train_gse_test/key_genes_top10_sc_present.tsv \
  --out_dir /data2/liu_hy/Hebei/processed/gse135337_scgpt_perturb_top10 \
  --label_col celltype_pred \
  --exclude_gsm GSM4006644_BC1 \
  --model_dir /data2/liu_hy/Hebei/scgpt_models \
  --device cuda \
  --batch_size 128 \
  --kd_factor 0.1 \
  --oe_factor 2.0
```

Outputs (in `--out_dir`):
- `perturb_delta_prob.tsv` (Δp by gene × label)
- `perturb_delta_pathway.tsv` (Δpathway score by gene × pathway)
- `Fig_sc1_deltaP_heatmap_KD.png`
- `Fig_sc1_deltaP_heatmap_OE.png`
- `Fig_sc2_pathway_heatmap_KD_<label>.png`
- `Fig_sc2_pathway_heatmap_OE_<label>.png`

---

## 🤗Troubleshooting

### Step3 cannot query GDC API
- Ensure Internet access.
- If behind a firewall, try running Step3 on a machine that can access `api.gdc.cancer.gov`.

### Step8/Step9: “No raw counts found”
- Step9 prefers `adata.layers["counts"]`.  
  Step8 already writes it; if you used another preprocessing script, make sure you saved raw counts:
  ```python
  adata.layers["counts"] = adata.X.copy()
  ```

### Step9: missing files in `--model_dir`
Ensure the folder contains exactly:
- `vocab.json`
- `args.json`
- `best_model.pt`

### Memory / CUDA OOM in Step9
- reduce `--batch_size` (e.g., 32 or 64，👉this work:128)
- use `--device cpu` for testing

---

## 😊Notes on reproducibility
- Scripts use explicit random seeds where applicable (see `--seed` in Step6).
- To compare runs fairly, keep the same gene lists and preprocessing parameters.

---

## 🤗Citation
If you use this pipeline, please cite:
- TCGA-BLCA (GDC / TCGA)
- GEO: **GSE13507** and **GSE135337**
- scGPT (pretrained model / method)
