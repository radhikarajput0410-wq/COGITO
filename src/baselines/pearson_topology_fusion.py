import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import average_precision_score

def calc_metrics(y_score, test_universe_edges, test_pos_set):
    y_true = np.array([1 if e in test_pos_set else 0 for e in test_universe_edges], dtype=np.int8)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    pos_count = np.sum(y_true)
    auprc = average_precision_score(y_true, y_score) if pos_count > 0 else 0
    return auprc, pos_count, y_true, y_score

def compute_recall_at_k(y_true, y_score, target_genes, test_universe_edges):
    scores_by_target = {tgt: [] for tgt in target_genes}
    for i, e in enumerate(test_universe_edges):
        tgt = e[1]
        if tgt in scores_by_target:
            scores_by_target[tgt].append((y_score[i], y_true[i]))
            
    recalls = {10: [], 30: [], 50: [], 100: []}
    for tgt, edges in scores_by_target.items():
        if not edges: continue
        edges.sort(key=lambda x: x[0], reverse=True)
        true_pos = sum(x[1] for x in edges)
        if true_pos == 0: continue
        
        for k in recalls.keys():
            hits = sum(x[1] for x in edges[:k])
            recalls[k].append(hits / true_pos)
            
    return {k: np.mean(v) if v else 0.0 for k, v in recalls.items()}

def run_experiment():
    print("=== RUNNING PEARSON + TOPOLOGY FUSION EXPERIMENT ===")
    
    out_dir = "data/processed/precise1k_regulondb_frozen_split"
    results_dir = "results"
    
    # LOAD DATA
    df_expr = pd.read_csv("data/external/precise1k_expr.tsv", sep='\t', index_col=0)
    gene_names = list(df_expr.index.str.lower())
    df_expr.index = gene_names
    
    df_univ = pd.read_csv(os.path.join(out_dir, "evaluation_universe.csv"))
    restricted_edges_list = list(zip(df_univ['Regulator'], df_univ['Target']))
    restricted_genes = list(set(df_univ['Regulator']).union(set(df_univ['Target'])))
    
    df_prior = pd.read_csv(os.path.join(out_dir, "prior_edges.csv"))
    prior_edges_list = list(zip(df_prior['Source'], df_prior['Target']))
    
    df_test = pd.read_csv(os.path.join(out_dir, "hidden_test_edges.csv"))
    test_pos_set = set(zip(df_test['Source'], df_test['Target']))
    
    seed = 42
    results = []
    
    def add_res(meth, alpha, aup, recs):
        res = {
            'method': meth,
            'alpha': alpha,
            'auprc': aup,
            'recall_at_10': recs[10],
            'recall_at_30': recs[30],
            'recall_at_50': recs[50],
            'recall_at_100': recs[100],
            'n_candidates': len(restricted_edges_list),
            'n_test_positives': len(test_pos_set.intersection(set(restricted_edges_list))),
            'random_seed': seed
        }
        results.append(res)
    
    # 1. Pearson Score
    print("Computing Pearson Baseline...")
    df_restricted = df_expr.loc[restricted_genes]
    corr_matrix = np.corrcoef(df_restricted.values)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    gene2idx = {g: i for i, g in enumerate(restricted_genes)}
    
    raw_pearson = np.array([abs(corr_matrix[gene2idx[e[0]], gene2idx[e[1]]]) for e in restricted_edges_list])
    
    # 2. Topology Score (Common Neighbors - Undirected projection)
    print("Computing Common Neighbors Baseline...")
    G = nx.DiGraph()
    G.add_edges_from(prior_edges_list)
    G_undirected = G.to_undirected(as_view=True)
    
    raw_topology = np.zeros(len(restricted_edges_list))
    undirected_cache = {}
    for g in restricted_genes:
        undirected_cache[g] = set(G_undirected.neighbors(g)) if G_undirected.has_node(g) else set()
        
    for i, (u, v) in enumerate(restricted_edges_list):
        raw_topology[i] = len(undirected_cache[u].intersection(undirected_cache[v]))
        
    # Add minimal deterministic noise to break ties consistently
    np.random.seed(seed)
    noise1 = np.random.rand(len(raw_pearson)) * 1e-6
    noise2 = np.random.rand(len(raw_topology)) * 1e-6
    
    score_pearson = raw_pearson + noise1
    score_topology = raw_topology + noise2
    
    # 3. Rank Normalization
    print("Rank Normalizing...")
    # rankdata assigns 1 to smallest, N to largest. We divide by N so max is 1.0 (percentile)
    rank_pearson = rankdata(score_pearson) / len(score_pearson)
    rank_topology = rankdata(score_topology) / len(score_topology)
    
    # 4. Complementarity
    rho, _ = spearmanr(rank_pearson, rank_topology)
    print(f"Spearman correlation (Pearson vs Topology): {rho:.4f}")
    
    # Random Baseline
    y_score_rnd = np.random.rand(len(restricted_edges_list))
    aup_rnd, _, yt_rnd, ys_rnd = calc_metrics(y_score_rnd, restricted_edges_list, test_pos_set)
    add_res("Random", None, aup_rnd, compute_recall_at_k(yt_rnd, ys_rnd, restricted_genes, restricted_edges_list))
    
    # 5. Pre-registered Grid Fusion
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    for a in alphas:
        fusion_score = a * rank_pearson + (1.0 - a) * rank_topology
        
        # Name handling based on alpha
        if a == 1.0:
            method_name = "Pearson"
        elif a == 0.0:
            method_name = "Common Neighbors"
        else:
            method_name = "Fusion"
            
        aup, _, yt, ys = calc_metrics(fusion_score, restricted_edges_list, test_pos_set)
        recs = compute_recall_at_k(yt, ys, restricted_genes, restricted_edges_list)
        add_res(method_name, a, aup, recs)
        
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(results_dir, "PEARSON_TOPOLOGY_FUSION.csv"), index=False)
    
    # 6. Top-K Overlap Analysis
    print("\nTop-K Overlap Analysis (Pearson vs Topology):")
    idx_pearson = np.argsort(rank_pearson)[::-1]
    idx_topology = np.argsort(rank_topology)[::-1]
    
    for K in [10, 30, 50, 100]:
        top_p = set(idx_pearson[:K])
        top_t = set(idx_topology[:K])
        inter = len(top_p.intersection(top_t))
        union = len(top_p.union(top_t))
        jaccard = inter / union if union > 0 else 0
        print(f"  K={K}: Intersection={inter}, Jaccard={jaccard:.4f}")

    print("Done. Results saved to PEARSON_TOPOLOGY_FUSION.csv")

if __name__ == "__main__":
    run_experiment()
