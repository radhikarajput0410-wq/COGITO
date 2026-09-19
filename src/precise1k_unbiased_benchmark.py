import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.preprocessing import GRNPreprocessor
from src.models.gat_model import GATCandidatePrioritizer, EdgePredictor
from src.models.joint_inference import JointMultiRegulatorInference

def get_universe(gene_names, df_gold, train_and_val_edges):
    all_gold_set = set(zip(df_gold['Source'], df_gold['Target']))
    test_pos_set = all_gold_set - train_and_val_edges
    
    all_possible_edges = {(src, tgt) for src in gene_names for tgt in gene_names if src != tgt}
    test_universe_edges = all_possible_edges - train_and_val_edges
    return test_universe_edges, test_pos_set

def calc_metrics(pred_map, test_universe_edges, test_pos_set):
    y_true = np.array([1 if e in test_pos_set else 0 for e in test_universe_edges])
    y_score = np.array([pred_map.get(e, 0.0) for e in test_universe_edges])
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    pos_count = np.sum(y_true)
    neg_count = len(y_true) - pos_count
    prevalence = pos_count / len(y_true) if len(y_true) > 0 else 0
    
    auprc = average_precision_score(y_true, y_score) if pos_count > 0 else 0
    return auprc, pos_count, neg_count, prevalence, y_true, y_score

def compute_recall_at_k(y_true, y_score, target_genes, test_universe_edges):
    # Organise scores by target
    scores_by_target = {tgt: [] for tgt in target_genes}
    for i, e in enumerate(test_universe_edges):
        tgt = e[1]
        if tgt in scores_by_target:
            scores_by_target[tgt].append((y_score[i], y_true[i]))
            
    recalls = {10: [], 30: [], 50: [], 100: []}
    for tgt, edges in scores_by_target.items():
        if not edges: continue
        # Sort by score descending
        edges.sort(key=lambda x: x[0], reverse=True)
        true_pos = sum(x[1] for x in edges)
        if true_pos == 0: continue
        
        for k in recalls.keys():
            hits = sum(x[1] for x in edges[:k])
            recalls[k].append(hits / true_pos)
            
    return {k: np.mean(v) if v else 0.0 for k, v in recalls.items()}

def run_unbiased_benchmark():
    print("=== UNBIASED PRECISE-1K BASELINE ===")
    out_dir = "results"
    
    expr_path = "data/external/precise1k/precise1k/data/precise1k/log_tpm_norm_qc.csv"
    prior_path = os.path.join(out_dir, "precise1k_filtered_prior.csv")
    
    df_expr = pd.read_csv(expr_path, index_col=0).T # samples x genes
    df_prior = pd.read_csv(prior_path)
    df_prior.columns = ['Source', 'Target']
    
    gene_names = list(df_expr.columns)
    
    # 1. Label-Independent Subset Selection (Variance)
    print("\nSelecting restricted universe via expression variance (no label leakage)...")
    variances = df_expr.var()
    # Pick top 500 most variable genes
    restricted_genes = list(variances.sort_values(ascending=False).head(500).index)
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Train/Val Split (Full TRN)
    df_prior_shuffled = df_prior.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_size = int(0.4 * len(df_prior_shuffled))
    val_size = int(0.2 * len(df_prior_shuffled))
    df_train = df_prior_shuffled.iloc[:train_size]
    df_val = df_prior_shuffled.iloc[train_size:train_size+val_size]
    train_and_val_edges = set(zip(df_train['Source'], df_train['Target'])).union(set(zip(df_val['Source'], df_val['Target'])))
    
    # Get Universes
    restricted_edges, restricted_pos = get_universe(restricted_genes, df_prior, train_and_val_edges)
    
    medium_genes = list(variances.sort_values(ascending=False).head(1000).index)
    full_edges, full_pos = get_universe(medium_genes, df_prior, train_and_val_edges)
    
    # Lists for keeping ordering consistent for GAT analysis
    restricted_edges_list = list(restricted_edges)
    full_edges_list = list(full_edges)
    
    print(f"Restricted Universe: {len(restricted_edges)} edges, {len(restricted_pos)} positives")
    print(f"Full Universe: {len(full_edges)} edges, {len(full_pos)} positives")
    
    benchmark_results = []
    
    # 2. Pearson Correlation
    print("\nRunning Pearson...")
    X_baseline = df_expr.T.values # Genes x Samples
    corr_matrix = np.corrcoef(X_baseline)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Fast map extraction
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    
    def extract_scores(edge_list, score_matrix):
        return {e: abs(score_matrix[gene2idx[e[0]], gene2idx[e[1]]]) for e in edge_list}
        
    pred_pearson_rest = extract_scores(restricted_edges_list, corr_matrix)
    pred_pearson_full = extract_scores(full_edges_list, corr_matrix)
    
    aup_pr_r, pos_r, neg_r, prev_r, yt_pr, ys_pr = calc_metrics(pred_pearson_rest, restricted_edges_list, restricted_pos)
    aup_pr_f, pos_f, neg_f, prev_f, yt_pf, ys_pf = calc_metrics(pred_pearson_full, full_edges_list, full_pos)
    
    benchmark_results.append({'Universe': 'Restricted', 'Method': 'Pearson', 'AUPRC': aup_pr_r, 'Positives': pos_r, 'Negatives': neg_r, 'Prevalence': prev_r})
    benchmark_results.append({'Universe': 'Full', 'Method': 'Pearson', 'AUPRC': aup_pr_f, 'Positives': pos_f, 'Negatives': neg_f, 'Prevalence': prev_f})
    
    # 3. GAT
    print("Running GAT...")
    preprocessor = GRNPreprocessor(scale_data=True)
    X_feat, gnames2 = preprocessor.process_expression(df_expr.T)
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
    
    def evaluate_gat(edge_list, pos_set):
        name2idx = {n: i for i, n in enumerate(gnames2)}
        
        batch_size = 100000
        gat_scores = []
        
        with torch.no_grad():
            z = gat_model(x_tensor, train_edge_index)
            
            for i in range(0, len(edge_list), batch_size):
                batch_edges = edge_list[i:i+batch_size]
                src_idx = [name2idx[e[0]] for e in batch_edges]
                tgt_idx = [name2idx[e[1]] for e in batch_edges]
                test_ei = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
                
                batch_sc = edge_predictor(z, test_ei)
                gat_scores.extend(batch_sc.numpy())
                
        pred_gat = {e: sc for e, sc in zip(edge_list, gat_scores)}
        aup, pos, neg, prev, yt, ys = calc_metrics(pred_gat, edge_list, pos_set)
        return aup, pos, neg, prev, yt, ys, pred_gat
        
    aup_gr_r, pos_gr_r, neg_gr_r, prev_gr_r, yt_gr, ys_gr, pred_gat_rest = evaluate_gat(restricted_edges_list, restricted_pos)
    benchmark_results.append({'Universe': 'Restricted', 'Method': 'GAT', 'AUPRC': aup_gr_r, 'Positives': pos_gr_r, 'Negatives': neg_gr_r, 'Prevalence': prev_gr_r})
    
    aup_gr_f, pos_gr_f, neg_gr_f, prev_gr_f, yt_gf, ys_gf, pred_gat_full = evaluate_gat(full_edges_list, full_pos)
    benchmark_results.append({'Universe': 'Full', 'Method': 'GAT', 'AUPRC': aup_gr_f, 'Positives': pos_gr_f, 'Negatives': neg_gr_f, 'Prevalence': prev_gr_f})
    
    # Save GAT scores for Task 5 analysis
    pd.DataFrame({'Score': ys_gf, 'Label': yt_gf}).to_csv(os.path.join(out_dir, "gat_full_scores.csv"), index=False)
    
    # 4. JEN (Restricted Universe Only)
    print("Running JEN (Restricted)...")
    jen_input_candidates = {tgt: [(src, 1.0) for src in restricted_genes if src != tgt and (src, tgt) in restricted_edges] for tgt in restricted_genes}
    jen = JointMultiRegulatorInference(gnames2, n_jobs=1, use_elasticnet=True, stability_runs=1)
    
    df_jen = jen.infer_network(jen_input_candidates, X_baseline) # Genes x Samples
    pred_jen_rest = {(r['Source'], r['Target']): r['Joint_Score'] for _, r in df_jen.iterrows()}
    
    aup_jr, pos_jr, neg_jr, prev_jr, yt_jr, ys_jr = calc_metrics(pred_jen_rest, restricted_edges_list, restricted_pos)
    benchmark_results.append({'Universe': 'Restricted', 'Method': 'JEN', 'AUPRC': aup_jr, 'Positives': pos_jr, 'Negatives': neg_jr, 'Prevalence': prev_jr})
    
    # 5. Random
    pred_rand_rest = {e: np.random.rand() for e in restricted_edges_list}
    aup_rnd_r, _, _, _, _, ys_rnd_r = calc_metrics(pred_rand_rest, restricted_edges_list, restricted_pos)
    benchmark_results.append({'Universe': 'Restricted', 'Method': 'Random', 'AUPRC': aup_rnd_r, 'Positives': pos_jr, 'Negatives': neg_jr, 'Prevalence': prev_jr})
    
    pred_rand_full = {e: np.random.rand() for e in full_edges_list}
    aup_rnd_f, _, _, _, _, ys_rnd_f = calc_metrics(pred_rand_full, full_edges_list, full_pos)
    benchmark_results.append({'Universe': 'Full', 'Method': 'Random', 'AUPRC': aup_rnd_f, 'Positives': pos_f, 'Negatives': neg_f, 'Prevalence': prev_f})
    
    # Save Benchmark
    df_bench = pd.DataFrame(benchmark_results)
    df_bench.to_csv(os.path.join(out_dir, "precise1k_unbiased_benchmark.csv"), index=False)
    
    # Task 4: Candidate Generation
    print("Generating Candidate Generation Analysis...")
    cand_results = []
    
    # Helper to format recalls
    def add_recalls(method, univ, recalls):
        for k, v in recalls.items():
            cand_results.append({'Method': method, 'Universe': univ, 'Metric': f'Recall@{k}', 'Value': v})
            
    recalls_pr = compute_recall_at_k(yt_pr, ys_pr, restricted_genes, restricted_edges_list)
    add_recalls('Pearson', 'Restricted', recalls_pr)
    
    recalls_gr = compute_recall_at_k(yt_gr, ys_gr, restricted_genes, restricted_edges_list)
    add_recalls('GAT', 'Restricted', recalls_gr)
    
    recalls_rr = compute_recall_at_k(yt_jr, ys_rnd_r, restricted_genes, restricted_edges_list)
    add_recalls('Random', 'Restricted', recalls_rr)
    
    recalls_pf = compute_recall_at_k(yt_pf, ys_pf, gene_names, full_edges_list)
    add_recalls('Pearson', 'Full', recalls_pf)
    
    recalls_gf = compute_recall_at_k(yt_gf, ys_gf, gene_names, full_edges_list)
    add_recalls('GAT', 'Full', recalls_gf)
    
    recalls_rf = compute_recall_at_k(yt_gf, ys_rnd_f, gene_names, full_edges_list)
    add_recalls('Random', 'Full', recalls_rf)
    
    pd.DataFrame(cand_results).to_csv(os.path.join(out_dir, "candidate_generation_analysis.csv"), index=False)
    
    # Task 6: JEN vs Pearson Analysis
    print("Generating JEN vs Pearson Analysis...")
    from scipy.stats import spearmanr
    jen_scores = ys_jr
    pearson_scores = ys_pr
    
    spearman, _ = spearmanr(pearson_scores, jen_scores)
    pearson_corr = np.corrcoef(pearson_scores, jen_scores)[0, 1]
    
    jen_top100 = np.argsort(jen_scores)[-100:]
    pearson_top100 = np.argsort(pearson_scores)[-100:]
    overlap_100 = len(set(jen_top100).intersection(set(pearson_top100)))
    
    jen_top500 = np.argsort(jen_scores)[-500:]
    pearson_top500 = np.argsort(pearson_scores)[-500:]
    overlap_500 = len(set(jen_top500).intersection(set(pearson_top500)))
    
    nonzero_jen = np.sum(jen_scores > 0)
    sparsity = 1.0 - (nonzero_jen / len(jen_scores))
    
    pd.DataFrame([{
        'Spearman_Corr': spearman,
        'Pearson_Corr': pearson_corr,
        'Top100_Overlap': overlap_100,
        'Top500_Overlap': overlap_500,
        'JEN_NonZero_Edges': nonzero_jen,
        'JEN_Sparsity': sparsity
    }]).to_csv(os.path.join(out_dir, "pearson_vs_jen_analysis.csv"), index=False)
    
    print("Unbiased Benchmark Complete.")

if __name__ == "__main__":
    run_unbiased_benchmark()
