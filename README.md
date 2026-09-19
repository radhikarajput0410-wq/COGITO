# Advanced Gene Regulatory Network (GRN) Inference
**GAT+JEN: Prior-Informed Joint Multi-Regulator Framework**

This repository provides a rigorously audited framework for inferring Gene Regulatory Networks (GRNs) from steady-state transcriptomic data by integrating biological prior knowledge (via an Expression-feature Graph Attention Network) with joint sparse regression constraints (via Joint Multi-Regulator Elastic Net). 

## 1. Project Purpose
GRN inference from transcriptomic data is challenging due to correlated regulators, indirect associations, high-dimensional feature spaces, and incomplete prior knowledge. The objective of this project is to evaluate whether a prior-informed framework combining non-linear structural encoding (GAT) with joint sparse regression (JEN) can improve the held-out recovery of regulatory edges compared to candidate-wise correlation heuristics.

## 2. Final Methodology
The final evaluated workflow consists of the following components:
1. **Prior Regulatory Graph:** Derived from RegulonDB.
2. **Expression-feature GAT:** Performs structural message passing over the known regulatory prior graph.
3. **Joint Multi-Regulator Elastic Net (JEN):** Evaluates multivariate regulatory constraints directly on expression data.
4. **Rank-based Fusion:** Combines the decoupled structural and regression ranking signals into a unified prediction score.

## 3. Dataset Requirements
Due to file size constraints, raw biological datasets are not included directly in this repository. 
The authoritative benchmarking dataset utilized for the final quantitative claims is the **PRECISE-1K** E. coli dataset.
- E. coli expression matrix: 1035 samples × 4257 genes.
- RegulonDB prior network.

All final validated evaluations were strictly conducted on the frozen split provided in the data directory.

## 4. Frozen Benchmark Data
The final immutable PRECISE-1K test split and evaluation universe are permanently stored in:
`data/processed/precise1k_regulondb_frozen_split/`

This directory contains:
- `prior_edges.csv`: 4701 known training edges.
- `hidden_test_edges.csv`: 3134 structurally isolated testing edges (0 overlap with prior).
- `evaluation_universe.csv`: 500 label-blind, variance-selected target genes yielding 249,413 candidate pairs.

**Do NOT alter these files.** Doing so will compromise the integrity of the benchmark and invalidate the final results.

## 5. How to Run the Pipeline
To execute the final benchmark evaluation:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the unbiased precise-1k baseline (Pearson / Topology)
python src/baselines/precise1k_baseline.py

# 3. Train and evaluate the Graph Attention Network (GAT)
python src/gat_precise1k.py

# 4. Train and evaluate the Joint Elastic Net (JEN)
python src/models/gat_guided_joint_elasticnet.py

# 5. Execute Multi-Seed robustness and final fusion
python src/gat_jen_multi_seed.py

# 6. Generate final reports and figures
python scripts/generate_final_reports.py
python scripts/generate_final_figures.py
```

## 6. Final Results
All final validated results, metrics, and paper-ready narrative reports are permanently stored in:
`results/final/`

Authoritative metric and leakage audits verifying the isolation of test labels are located in:
`results/audits/`

Generated visual evidence is located in:
`results/figures/`

**Key Result:**
The GAT+JEN fusion model demonstrated consistent superiority over both its isolated GAT and JEN components across multiple random initializations on the frozen PRECISE-1K benchmark (Mean AUPRC: 0.019209). 

## 7. gNAISTO Reproducibility Status
A direct numerical comparison with gNAISTO (a candidate-wise block-coordinate descent Graphical Lasso) could not be completed. The official implementation was not reproducibly executable at the dimensional scale required by the frozen PRECISE-1K benchmark within the available computational environment. Numerical singularity and $O(p^3)$ intractability over the singular dense covariance matrix prevented a faithful unified benchmark result. Therefore, direct numerical superiority over gNAISTO remains **INCONCLUSIVE**.

## 8. Important Limitations
1. Dependence on the quality and coverage of the prior network.
2. Linear assumptions constrained within the Joint Elastic Net component.
3. High-dimensional sample-to-feature ratios present computational challenges for dense joint models.
4. The benchmark strictly evaluates held-out edge recovery within a known network space; it does not constitute independent biological discovery.
5. Steady-state transcriptomic data evaluates statistical dependencies and cannot establish biochemical causality.

## 9. Historical / Superseded Experiments
During the development phase, several iterations involving alternative datasets (e.g., DREAM4, PlantRegMap/Arabidopsis) and intermediate strategies (e.g., hard Top-K pruning) were explored. These explorations were structurally superseded by the final PRECISE-1K benchmark protocol.
All associated historical scripts, obsolete data artifacts, and invalid metric claims have been quarantined in the `archive/` directory to preserve auditable project history without contaminating the final scientific claims.


================================================================================

# DATASET DOCUMENTATION


## From README_ARABIDOPSIS.md

# Arabidopsis thaliana & PlantRegMap Setup

This benchmark supports plant GRN inference using Arabidopsis expression data and PlantRegMap regulatory priors.

## Data Availability
The raw files are heavy and **not included** in this repository.

## Reproducible Download Instructions

To execute the Arabidopsis benchmark, you must download the official dataset and PlantRegMap prior.

1. **Official Sources**:
   - **PlantRegMap**: Download the Arabidopsis TF-target interactions.
   - **Expression Data**: A standardized microarray or RNA-seq expression matrix for Arabidopsis (e.g., from GEO or Araport).
2. **Expected Files**:
   - `Ath_TF_target.txt.gz` (PlantRegMap Prior)
   - `arabidopsis_expr.tsv` (Expression Matrix)
3. **Location**: Place the downloaded files exactly here:
   - `data/external/Ath_TF_target.txt.gz`
   - `data/external/arabidopsis_expr.tsv`

## Processing Pipeline
Once the files are located in `data/external/`, run the processing script:
```bash
python src/datasets/process_arabidopsis.py
```
This script will:
- Extract TF and Target IDs from the zipped PlantRegMap database.
- Drop duplicates and standardize formatting.
- Generate the final `data/external/arabidopsis_plantregmap_priors.tsv` prior network.

Ensure the identifiers in `arabidopsis_expr.tsv` (e.g., AGI locus codes like `AT1G01010`) match the identifiers output from PlantRegMap to guarantee correct alignment in the benchmark runner.


## From README_DREAM4.md

# DREAM4 In-Silico Network Challenge

This benchmark evaluates methods on the DREAM4 in-silico network challenge (Size 100, multifactorial).

## Current Status
Currently, only one network is pre-installed in the repository:
- `data/raw/dream4/insilico_size100_1_multifactorial.tsv`
- `data/raw/dream4/insilico_size100_1_goldstandard.tsv`

## Reproducible Download Instructions

To evaluate on the remaining networks (Networks 2-5), you must obtain them from the official DREAM challenge repository.

1. **Official Source**: Synapse (DREAM Challenges) or standard bioinformatics repositories hosting DREAM4 archives.
2. **Expected Files**:
   - `insilico_size100_2_multifactorial.tsv`
   - `insilico_size100_2_goldstandard.tsv`
   - `insilico_size100_3_multifactorial.tsv`
   - `insilico_size100_3_goldstandard.tsv`
   - `insilico_size100_4_multifactorial.tsv`
   - `insilico_size100_4_goldstandard.tsv`
   - `insilico_size100_5_multifactorial.tsv`
   - `insilico_size100_5_goldstandard.tsv`
3. **Location**: Place all downloaded files directly into the `data/raw/dream4/` directory.

The `src.datasets.process_geo.py` module and benchmark runner will automatically detect and iterate over any properly named DREAM4 files placed in this directory.


## From README_PRECISE1K.md

# PRECISE-1K (E. coli) Dataset Setup

The PRECISE-1K dataset provides high-quality RNA-seq compendia for Escherichia coli K-12 MG1655, along with a curated Transcriptional Regulatory Network (TRN).

## Data Availability
**GOOD NEWS**: The heavy raw datasets for PRECISE-1K *are* actively included in this repository and do not need to be downloaded externally.

1. **Official Source**: iModulonDB (PRECISE-1K dataset)
2. **Provided Files**:
   - `data/external/precise1k/precise1k/data/precise1k/log_tpm_norm_qc.csv` (Expression Matrix)
   - `data/external/precise1k/precise1k/data/annotation/TRN.csv` (Prior Network)
   - `data/external/precise1k/precise1k/data/annotation/gene_info.csv` (Gene annotations and locus tags)

## Processing Pipeline
Once the raw files are downloaded, run the automated processing script:
```bash
python src/datasets/process_precise1k.py
```
This script will:
- Map gene symbols (e.g., `accB`) to standard locus tags (`b3255`) for consistency.
- Generate `data/external/precise1k_expr.tsv` (Expression).
- Generate `data/external/ecoli_regulondb.tsv` (Prior network).

