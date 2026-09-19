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


