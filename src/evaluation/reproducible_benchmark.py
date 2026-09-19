import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_pipeline import load_config
from src.utils.preprocessing import GRNPreprocessor
from src.models.gat_model import GATCandidatePrioritizer, EdgePredictor
from src.models.joint_inference import JointMultiRegulatorInference
from src.baselines.baselines import BaselineModels
from src.evaluation.edge_scoring import FinalEdgeScorer

def get_universe(gene_names, df_gold, train_and_val_edges):
    all_possible_edges = {(src, tgt) for src in gene_names for tgt in gene_names if src != tgt}
    test_universe_edges = all_possible_edges - train_and_val_edges
    
    all_gold_set = set(zip(df_gold['Source'], df_gold['Target']))
    test_pos_set = all_gold_set - train_and_val_edges
    test_neg_set = test_universe_edges - all_gold_set
    
    return test_universe_edges, test_pos_set, test_neg_set, all_possible_edges, all_gold_set

def calc_metrics(pred_map, test_universe_edges, test_pos_set):
    y_true = [1 if e in test_pos_set else 0 for e in test_universe_edges]
    y_score = [pred_map.get(e, 0.0) for e in test_universe_edges]
    
    # Handle NaN and inf in scores
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    auprc = average_precision_score(y_true, y_score)
    return auprc, y_true, y_score

def get_indegree_groups(df_gold, gene_names):
    indegrees = {g: 0 for g in gene_names}
    for tgt in df_gold['Target']:
        indegrees[tgt] = indegrees.get(tgt, 0) + 1
        
    vals = list(indegrees.values())
    low_thresh = np.percentile(vals, 33)
    high_thresh = np.percentile(vals, 66)
    
    groups = {'Low': [], 'Medium': [], 'High': []}
    for g, deg in indegrees.items():
        if deg <= low_thresh:
            groups['Low'].append(g)
        elif deg <= high_thresh:
            groups['Medium'].append(g)
        else:
            groups['High'].append(g)
    return groups, indegrees

def evaluate_high_indegree(pred_map, test_universe_edges, test_pos_set, target_group):
    # filter universe and pos_set for only these targets
    group_universe = [e for e in test_universe_edges if e[1] in target_group]
    if not group_universe: return np.nan
    group_pos = {e for e in test_pos_set if e[1] in target_group}
    if not group_pos: return np.nan
    
    y_true = [1 if e in group_pos else 0 for e in group_universe]
    y_score = [pred_map.get(e, 0.0) for e in group_universe]
    
    return average_precision_score(y_true, y_score)

def run_benchmark():
    out_dir = "results/benchmark"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=== STARTING REPRODUCIBLE GRN BENCHMARK ===")
    
    config = load_config("dream4")
    df_expr = pd.read_csv(config.expr_path, sep='\t')
    df_gold = pd.read_csv(config.gold_path, sep='\t', header=None)
    if len(df_gold.columns) == 2:
        df_gold.columns = ['Source', 'Target']
    else:
        df_gold.columns = ['Source', 'Target', 'Weight'] + list(df_gold.columns[3:])

    # Task 9: Sample to Gene ratio
    num_samples = df_expr.shape[0]
    num_genes = df_expr.shape[1]
    print(f"Dataset: DREAM4 | Genes (p): {num_genes} | Samples (n): {num_samples} | Ratio (n/p): {num_samples/num_genes:.2f}")
    if num_samples < num_genes:
        print("NOTE: This is a strongly high-dimensional (p >> n) regime. JEN performance may collapse due to collinearity noise.")

    preprocessor = GRNPreprocessor(scale_data=True)
    X_features, gene_names = preprocessor.process_expression(df_expr)
    df_gold_processed = preprocessor.process_gold_standard(df_gold, gene_names)
    x_tensor, all_gold_edge_index = preprocessor.to_torch_tensors(X_features, df_gold_processed)
    num_gold = all_gold_edge_index.shape[1]
    
    groups, indegrees = get_indegree_groups(df_gold_processed, gene_names)
    
    seeds = [42, 123, 2024]
    
    benchmark_results = []
    candidate_results = []
    edge_scores = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Split
        shuffled_indices = np.random.permutation(num_gold)
        train_size = int(0.4 * num_gold)
        val_size = int(0.2 * num_gold)
        
        train_prior_edge_index = all_gold_edge_index[:, shuffled_indices[:train_size]]
        val_prior_edge_index = all_gold_edge_index[:, shuffled_indices[train_size:train_size + val_size]]
        
        train_prior_set = set(zip([gene_names[i] for i in train_prior_edge_index[0].numpy()], 
                                  [gene_names[i] for i in train_prior_edge_index[1].numpy()]))
        val_prior_set = set(zip([gene_names[i] for i in val_prior_edge_index[0].numpy()], 
                                [gene_names[i] for i in val_prior_edge_index[1].numpy()]))
                                
        train_and_val_edges = train_prior_set.union(val_prior_set)
        test_universe_edges, test_pos_set, test_neg_set, all_possible_edges, all_gold_set = get_universe(gene_names, df_gold_processed, train_and_val_edges)
        
        # Helper to log edge scores
        def log_edges(method, pred_map):
            for e in test_universe_edges:
                edge_scores.append({
                    'source': e[0], 'target': e[1], 'edge_id': f"{e[0]}->{e[1]}",
                    'score': pred_map.get(e, 0.0), 'label': 1 if e in test_pos_set else 0,
                    'split': 'test', 'method': method, 'seed': seed
                })
        
        def evaluate_and_log(method, pred_map):
            auprc, _, _ = calc_metrics(pred_map, test_universe_edges, test_pos_set)
            high_auprc = evaluate_high_indegree(pred_map, test_universe_edges, test_pos_set, groups['High'])
            med_auprc = evaluate_high_indegree(pred_map, test_universe_edges, test_pos_set, groups['Medium'])
            low_auprc = evaluate_high_indegree(pred_map, test_universe_edges, test_pos_set, groups['Low'])
            
            benchmark_results.append({
                'Dataset': 'DREAM4', 'Network': 'insilico_size100_1', 'Method': method, 'Seed': seed,
                'AUPRC': auprc, 'High-Indegree AUPRC': high_auprc, 'Medium-Indegree AUPRC': med_auprc, 'Low-Indegree AUPRC': low_auprc
            })
            log_edges(method, pred_map)
            
        # B0: Random
        pred_random = {e: np.random.rand() for e in test_universe_edges}
        evaluate_and_log('B0 - Random', pred_random)
        
        # B1: Pearson
        df_corr = BaselineModels(gene_names).run_correlation(X_features, method='pearson')
        pred_pearson = {(row['Source'], row['Target']): abs(row['Score']) for _, row in df_corr.iterrows()}
        evaluate_and_log('B1 - Pearson', pred_pearson)
        
        # B2: GAT
        gat_model = GATCandidatePrioritizer(num_node_features=x_tensor.shape[1], hidden_channels=32, out_channels=16)
        edge_predictor = EdgePredictor(embedding_dim=16)
        optimizer = torch.optim.Adam(list(gat_model.parameters()) + list(edge_predictor.parameters()), lr=0.01)
        criterion = torch.nn.BCELoss()
        
        gat_model.train()
        edge_predictor.train()
        for _ in range(50):
            optimizer.zero_grad()
            z = gat_model(x_tensor, train_prior_edge_index)
            scores = edge_predictor(z, train_prior_edge_index)
            loss = criterion(scores, torch.ones_like(scores))
            loss.backward()
            optimizer.step()
            
        gat_model.eval()
        edge_predictor.eval()
        
        all_sources_idx, all_targets_idx = [], []
        for i in range(len(gene_names)):
            for j in range(len(gene_names)):
                if i != j:
                    all_sources_idx.append(i)
                    all_targets_idx.append(j)
        all_edge_index_full = torch.tensor([all_sources_idx, all_targets_idx], dtype=torch.long)
        
        with torch.no_grad():
            z = gat_model(x_tensor, train_prior_edge_index)
            all_gat_scores = edge_predictor(z, all_edge_index_full)
            
        pred_gat = {}
        for src_idx, tgt_idx, sc in zip(all_edge_index_full[0], all_edge_index_full[1], all_gat_scores):
            pred_gat[(gene_names[src_idx.item()], gene_names[tgt_idx.item()])] = sc.item()
            
        evaluate_and_log('B2 - GAT', pred_gat)
        
        # Task 7: Controlled Candidate Exp (K=10, 20, 30, 50)
        for k in [10, 20, 30, 50]:
            # Pearson candidates
            cands_p = {}
            for tgt in gene_names:
                tgt_corr = df_corr[df_corr['Target'] == tgt].nlargest(k, 'Score')
                cands_p[tgt] = [(row['Source'], abs(row['Score'])) for _, row in tgt_corr.iterrows()]
            
            # GAT candidates
            cands_g = {}
            for tgt in gene_names:
                tgt_gat = sorted([(src, pred_gat.get((src, tgt), 0)) for src in gene_names if src != tgt], key=lambda x: x[1], reverse=True)[:k]
                cands_g[tgt] = tgt_gat
                
            # Random candidates
            cands_r = {}
            for tgt in gene_names:
                possible = [g for g in gene_names if g != tgt]
                chosen = np.random.choice(possible, size=k, replace=False)
                cands_r[tgt] = [(c, np.random.rand()) for c in chosen]
                
            def eval_c(name, cd):
                hits = sum(1 for tgt, clist in cd.items() for src, _ in clist if (src, tgt) in test_pos_set)
                rec = hits / len(test_pos_set) if test_pos_set else 0
                candidate_results.append({'Method': name, 'Seed': seed, 'K': k, 'Recall': rec})
                
            eval_c('Pearson', cands_p)
            eval_c('GAT', cands_g)
            eval_c('Random', cands_r)
        
        # B3: JEN
        # JEN run on ALL edges to avoid candidate bias
        flat_all = {tgt: [(src, 1.0) for src in gene_names if src != tgt] for tgt in gene_names}
        df_jen = JointMultiRegulatorInference(gene_names, use_elasticnet=True, stability_runs=1).infer_network(flat_all, X_features)
        pred_jen = {(r['Source'], r['Target']): r['Joint_Score'] for _, r in df_jen.iterrows()}
        evaluate_and_log('B3 - JEN', pred_jen)
        
        # B4: GAT + JEN (Average)
        pred_gat_jen = {}
        for e in test_universe_edges:
            g = pred_gat.get(e, 0.0)
            j = pred_jen.get(e, 0.0)
            # Normalize them roughly by ranking or just simple sum
            pred_gat_jen[e] = g + j
        evaluate_and_log('B4 - GAT+JEN', pred_gat_jen)
        
        # B5: GAT + RF (Clean train)
        # Train strictly on val_prior_set (y=1) and disjoint negative sample (y=0)
        true_negatives = list(all_possible_edges - all_gold_set)
        train_negatives = set(np.random.choice(len(true_negatives), size=500, replace=False))
        train_neg_edges = {true_negatives[i] for i in train_negatives}
        
        rows = []
        for e in test_universe_edges.union(val_prior_set).union(train_neg_edges):
            g_sc = pred_gat.get(e, 0.0)
            j_sc = pred_jen.get(e, 0.0)
            if e in val_prior_set:
                rows.append({'Source': e[0], 'Target': e[1], 'GAT': g_sc, 'JEN': j_sc, 'y': 1, 'is_train': True})
            elif e in train_neg_edges:
                rows.append({'Source': e[0], 'Target': e[1], 'GAT': g_sc, 'JEN': j_sc, 'y': 0, 'is_train': True})
            else:
                rows.append({'Source': e[0], 'Target': e[1], 'GAT': g_sc, 'JEN': j_sc, 'y': 0, 'is_train': False})
                
        df_f = pd.DataFrame(rows)
        df_train = df_f[df_f['is_train']]
        
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        X_tr = df_train[['GAT', 'JEN']].values
        y_tr = df_train['y'].values
        # standardize
        mu = np.mean(X_tr, axis=0)
        sigma = np.std(X_tr, axis=0) + 1e-8
        rf.fit((X_tr - mu) / sigma, y_tr)
        
        X_test = df_f[['GAT', 'JEN']].values
        probs = rf.predict_proba((X_test - mu) / sigma)[:, 1]
        
        pred_rf = {(r['Source'], r['Target']): p for (_, r), p in zip(df_f.iterrows(), probs)}
        evaluate_and_log('B5 - GAT+RF', pred_rf)

    # Save outputs
    pd.DataFrame(benchmark_results).to_csv(os.path.join(out_dir, 'benchmark_results.csv'), index=False)
    pd.DataFrame(edge_scores).to_csv(os.path.join(out_dir, 'edge_scores.csv'), index=False)
    pd.DataFrame(candidate_results).to_csv(os.path.join(out_dir, 'candidate_results.csv'), index=False)
    
    # Generate Figures
    plt.figure()
    df_summ = pd.DataFrame(benchmark_results)
    sns.barplot(data=df_summ, x='Method', y='AUPRC')
    plt.xticks(rotation=45)
    plt.title('Baseline Method Comparison (AUPRC)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'method_auprc_comparison.png'))
    
    plt.figure()
    df_cand = pd.DataFrame(candidate_results)
    sns.lineplot(data=df_cand, x='K', y='Recall', hue='Method', marker='o')
    plt.title('Candidate Selection (GAT vs Random vs Pearson)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'candidate_k_vs_recall.png'))
    
    plt.figure()
    df_melt = pd.melt(df_summ, id_vars=['Method', 'Seed'], value_vars=['Low-Indegree AUPRC', 'Medium-Indegree AUPRC', 'High-Indegree AUPRC'], var_name='Indegree', value_name='Group_AUPRC')
    sns.barplot(data=df_melt, x='Indegree', y='Group_AUPRC', hue='Method')
    plt.xticks(rotation=15)
    plt.title('High-Indegree Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'high_indegree_auprc.png'))
    
    print("\nBENCHMARK EXECUTED SUCCESSFULLY.")

if __name__ == "__main__":
    run_benchmark()
