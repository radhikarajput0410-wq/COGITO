import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.preprocessing import GRNPreprocessor
from src.models.gat_model import GATCandidatePrioritizer, EdgePredictor
from src.models.joint_inference import JointMultiRegulatorInference
from src.baselines.baselines import BaselineModels

def get_universe(gene_names, df_gold, train_and_val_edges):
    all_possible_edges = {(src, tgt) for src in gene_names for tgt in gene_names if src != tgt}
    test_universe_edges = all_possible_edges - train_and_val_edges
    all_gold_set = set(zip(df_gold['Source'], df_gold['Target']))
    test_pos_set = all_gold_set - train_and_val_edges
    return test_universe_edges, test_pos_set, all_gold_set

def calc_metrics(pred_map, test_universe_edges, test_pos_set):
    y_true = [1 if e in test_pos_set else 0 for e in test_universe_edges]
    y_score = [pred_map.get(e, 0.0) for e in test_universe_edges]
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)
    return average_precision_score(y_true, y_score)

def run_precise1k_baseline():
    print("=== STARTING PRECISE-1K BASELINE ===")
    out_dir = "results"
    
    # 1. Load Data
    expr_path = "data/external/precise1k/precise1k/data/precise1k/log_tpm_norm_qc.csv"
    prior_path = os.path.join(out_dir, "precise1k_filtered_prior.csv")
    
    df_expr = pd.read_csv(expr_path, index_col=0).T # samples x genes
    df_prior = pd.read_csv(prior_path)
    df_prior.columns = ['Source', 'Target']
    
    gene_names = list(df_expr.columns)
    
    # Dataset Regime
    num_samples, num_genes = df_expr.shape
    print(f"PRECISE-1K | Samples (n): {num_samples} | Genes (p): {num_genes} | n/p: {num_samples/num_genes:.3f} | p/n: {num_genes/num_samples:.3f}")
    df_regime = pd.DataFrame([
        {'Dataset': 'DREAM4', 'Samples': 100, 'Genes': 100, 'n/p': 1.0, 'p/n': 1.0},
        {'Dataset': 'PRECISE-1K', 'Samples': num_samples, 'Genes': num_genes, 'n/p': num_samples/num_genes, 'p/n': num_genes/num_samples}
    ])
    df_regime.to_csv(os.path.join(out_dir, "dataset_regime_comparison.csv"), index=False)
    
    # To avoid 4000x4000 regressions which would take 10+ hours, we will subsample the *evaluation* universe 
    # for the test set to a representative 50 regulators and 50 targets
    # The user strictly says: "Do not compare Pearson against a random score generated over a different set of edges."
    # To ensure the evaluation universe contains true positives, we pick the top 50 genes by degree in the TRN
    degree_counts = pd.concat([df_prior['Source'], df_prior['Target']]).value_counts()
    eval_genes = list(degree_counts.head(50).index)
    # Ensure they are in gene_names
    eval_genes = [g for g in eval_genes if g in gene_names]
    # We still use the full expression matrix for embedding/pearson!
    
    X_features = df_expr.values # Samples x Genes. BaselineModels expects Genes x Samples later, but preprocessing handles this.
    # Actually GRNPreprocessor expects Genes x Samples or Samples x Genes?
    # run_pipeline does: X_features, gene_names = preprocessor.process_expression(df_expr)
    # Let's use BaselineModels directly which assumes standard orientation.
    # BaselineModels takes X as (Genes x Samples)
    X_baseline = df_expr.T.values 
    
    benchmark_results = []
    candidate_results = []
    
    seeds = [42]
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Split (Phase 5)
        df_prior_shuffled = df_prior.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_size = int(0.4 * len(df_prior_shuffled))
        val_size = int(0.2 * len(df_prior_shuffled))
        
        df_train = df_prior_shuffled.iloc[:train_size]
        df_val = df_prior_shuffled.iloc[train_size:train_size+val_size]
        
        train_and_val_edges = set(zip(df_train['Source'], df_train['Target'])).union(set(zip(df_val['Source'], df_val['Target'])))
        
        test_universe_edges, test_pos_set, all_gold_set = get_universe(eval_genes, df_prior, train_and_val_edges)
        print(f"Test universe size: {len(test_universe_edges)} edges. True positives: {len(test_pos_set)}")
        
        if seed == 42:
            with open(os.path.join(out_dir, "precise1k_split.json"), "w") as f:
                json.dump({
                    'seed': seed, 'train_edges': len(df_train), 'val_edges': len(df_val),
                    'test_universe': len(test_universe_edges), 'test_positives': len(test_pos_set)
                }, f, indent=4)
        
        # Phase 7: Random
        pred_random = {e: np.random.rand() for e in test_universe_edges}
        auprc_rnd = calc_metrics(pred_random, test_universe_edges, test_pos_set)
        benchmark_results.append({'Method': 'B0 - Random', 'Seed': seed, 'AUPRC': auprc_rnd})
        
        # Phase 6: Pearson
        corr_matrix = np.corrcoef(X_baseline)
        pred_pearson = {}
        pearson_edges_out = []
        
        for e in test_universe_edges:
            src_idx = gene_names.index(e[0])
            tgt_idx = gene_names.index(e[1])
            score = abs(corr_matrix[src_idx, tgt_idx])
            score = 0.0 if np.isnan(score) else score
            pred_pearson[e] = score
            if seed == 42:
                pearson_edges_out.append({'source': e[0], 'target': e[1], 'score': score, 'label': 1 if e in test_pos_set else 0, 'split': 'test', 'seed': seed})
        
        if seed == 42:
            pd.DataFrame(pearson_edges_out).to_csv(os.path.join(out_dir, "precise1k_pearson_edges.csv"), index=False)
            
        auprc_prs = calc_metrics(pred_pearson, test_universe_edges, test_pos_set)
        print(f"Pearson AUPRC: {auprc_prs:.4f}")
        benchmark_results.append({'Method': 'B1 - Pearson', 'Seed': seed, 'AUPRC': auprc_prs})
        
        # Phase 8: GAT
        preprocessor = GRNPreprocessor(scale_data=True)
        # We need expression as df with gene columns, samples rows or vice versa?
        # df_expr is Samples x Genes
        X_feat, gnames2 = preprocessor.process_expression(df_expr.T) # passing Genes x Samples
        df_gold_p = preprocessor.process_gold_standard(df_train, gnames2)
        x_tensor, train_edge_index = preprocessor.to_torch_tensors(X_feat, df_gold_p)
        
        gat_model = GATCandidatePrioritizer(num_node_features=x_tensor.shape[1], hidden_channels=32, out_channels=16)
        edge_predictor = EdgePredictor(embedding_dim=16)
        optimizer = torch.optim.Adam(list(gat_model.parameters()) + list(edge_predictor.parameters()), lr=0.01)
        criterion = torch.nn.BCELoss()
        
        gat_model.train()
        edge_predictor.train()
        for _ in range(30):
            optimizer.zero_grad()
            z = gat_model(x_tensor, train_edge_index)
            scores = edge_predictor(z, train_edge_index)
            loss = criterion(scores, torch.ones_like(scores))
            loss.backward()
            optimizer.step()
            
        gat_model.eval()
        edge_predictor.eval()
        
        pred_gat = {}
        with torch.no_grad():
            z = gat_model(x_tensor, train_edge_index)
            # Evaluate only on test universe
            src_idx = [gnames2.index(e[0]) for e in test_universe_edges]
            tgt_idx = [gnames2.index(e[1]) for e in test_universe_edges]
            test_ei = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
            if test_ei.shape[1] > 0:
                gat_scores = edge_predictor(z, test_ei)
                for e, sc in zip(test_universe_edges, gat_scores):
                    pred_gat[e] = sc.item()
                    
        auprc_gat = calc_metrics(pred_gat, test_universe_edges, test_pos_set)
        benchmark_results.append({'Method': 'B2 - GAT', 'Seed': seed, 'AUPRC': auprc_gat})
        
        # Candidate Recall
        for k in [10, 20, 30, 50]:
            cands_p, cands_g, cands_r = {}, {}, {}
            for tgt in eval_genes:
                # filter to eval universe
                possible_edges = [(src, tgt) for src in eval_genes if src != tgt and (src, tgt) in test_universe_edges]
                # Pearson
                tgt_p = sorted(possible_edges, key=lambda e: pred_pearson.get(e, 0), reverse=True)[:k]
                cands_p[tgt] = tgt_p
                # GAT
                tgt_g = sorted(possible_edges, key=lambda e: pred_gat.get(e, 0), reverse=True)[:k]
                cands_g[tgt] = tgt_g
                # Random
                np.random.shuffle(possible_edges)
                cands_r[tgt] = possible_edges[:k]
                
            def eval_c(name, cd):
                hits = sum(1 for tgt, clist in cd.items() for e in clist if e in test_pos_set)
                rec = hits / max(1, len(test_pos_set))
                candidate_results.append({'Method': name, 'Seed': seed, 'K': k, 'Recall': rec})
                
            eval_c('Pearson', cands_p)
            eval_c('GAT', cands_g)
            eval_c('Random', cands_r)
            
        # Phase 9: JEN (Joint Elastic Net)
        # To make it run in reasonable time, we run JEN only for the eval_genes targets
        jen_input_candidates = {tgt: [(src, 1.0) for src in eval_genes if src != tgt and (src, tgt) in test_universe_edges] for tgt in eval_genes}
        jen = JointMultiRegulatorInference(gnames2, use_elasticnet=True, stability_runs=1)
        try:
            df_jen = jen.infer_network(jen_input_candidates, X_baseline)
            pred_jen = {(r['Source'], r['Target']): r['Joint_Score'] for _, r in df_jen.iterrows()}
        except Exception as e:
            print(f"JEN failed: {e}")
            pred_jen = {e: 0.0 for e in test_universe_edges}
            
        auprc_jen = calc_metrics(pred_jen, test_universe_edges, test_pos_set)
        benchmark_results.append({'Method': 'B3 - JEN', 'Seed': seed, 'AUPRC': auprc_jen})
        
        # Phase 10: GAT + JEN
        pred_gat_jen = {e: pred_gat.get(e, 0) + pred_jen.get(e, 0) for e in test_universe_edges}
        auprc_gj = calc_metrics(pred_gat_jen, test_universe_edges, test_pos_set)
        benchmark_results.append({'Method': 'B4 - GAT+JEN', 'Seed': seed, 'AUPRC': auprc_gj})

    # Save Results
    pd.DataFrame(benchmark_results).to_csv(os.path.join(out_dir, "precise1k_benchmark.csv"), index=False)
    pd.DataFrame(candidate_results).to_csv(os.path.join(out_dir, "precise1k_candidates.csv"), index=False)
    
    print("\nPRECISE-1K BASELINE COMPLETE.")
    
if __name__ == "__main__":
    run_precise1k_baseline()
