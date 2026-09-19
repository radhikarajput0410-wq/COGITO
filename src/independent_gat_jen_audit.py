import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr, rankdata
import networkx as nx

class DirectedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32, heads=4):
        super(DirectedGAT, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=heads, concat=False)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x, edge_index, query_edges):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        src_emb = x[query_edges[0]]
        tgt_emb = x[query_edges[1]]
        pair_emb = torch.cat([src_emb, tgt_emb], dim=1)
        scores = self.mlp(pair_emb).squeeze(-1)
        return scores

def independent_recall(y_true, y_score, candidate_edges, K):
    """Ground-up, independent Per-Target Recall@K implementation"""
    # 1. Group by target
    tgt_dict = {}
    for i, e in enumerate(candidate_edges):
        tgt = e[1]
        if tgt not in tgt_dict:
            tgt_dict[tgt] = []
        tgt_dict[tgt].append((y_score[i], y_true[i]))
        
    # 2. Score each target
    recalls = []
    for tgt, cands in tgt_dict.items():
        true_pos = sum(x[1] for x in cands)
        if true_pos == 0:
            continue
        # Sort candidates descending by score
        cands.sort(key=lambda x: x[0], reverse=True)
        hits = sum(x[1] for x in cands[:K])
        recalls.append(hits / true_pos)
        
    return np.mean(recalls) if recalls else 0.0

def run_audit():
    print("==================================================")
    print("INDEPENDENT METRIC AUDIT: GAT-GUIDED JEN")
    print("==================================================")
    
    out_dir = "data/processed/precise1k_regulondb_frozen_split"
    results_dir = "results"
    
    # LOAD DATA EXACTLY
    df_expr = pd.read_csv("data/external/precise1k_expr.tsv", sep='\t', index_col=0)
    gene_names = list(df_expr.index.str.lower())
    df_expr.index = gene_names
    
    df_univ = pd.read_csv(os.path.join(out_dir, "evaluation_universe.csv"))
    candidate_edges = list(zip(df_univ['Regulator'], df_univ['Target']))
    genes = list(set(df_univ['Regulator']).union(set(df_univ['Target'])))
    
    df_prior = pd.read_csv(os.path.join(out_dir, "prior_edges.csv"))
    prior_edges_list = list(zip(df_prior['Source'], df_prior['Target']))
    
    df_test = pd.read_csv(os.path.join(out_dir, "hidden_test_edges.csv"))
    hidden_test_set = set(zip(df_test['Source'], df_test['Target']))
    
    # 1. Universe Audit
    print("\n[1] Universe Audit")
    n_cands = len(candidate_edges)
    n_unique_cands = len(set(candidate_edges))
    print(f"Total candidates: {n_cands}")
    print(f"Unique candidates: {n_unique_cands}")
    print(f"Duplicates: {n_cands - n_unique_cands}")
    if n_cands != 249413 or n_cands != n_unique_cands:
        print("FAIL: Universe alignment error!")
        sys.exit(1)
        
    y_true = np.array([1 if e in hidden_test_set else 0 for e in candidate_edges], dtype=np.int8)
    n_pos = np.sum(y_true)
    print(f"Evaluated positives: {n_pos}")
    print(f"Positive prevalence: {n_pos/n_cands:.6f}")
    
    # 2. Leakage Audit
    print("\n[2] Leakage Audit")
    intersect = len(set(prior_edges_list).intersection(hidden_test_set))
    print(f"Prior / Test Intersection: {intersect}")
    if intersect != 0:
        print("FAIL: Prior leakage!")
        sys.exit(1)
        
    # Reconstruct predictions exactly
    print("\n[3] Reconstructing Predictions (Seed 42)...")
    df_restricted = df_expr.loc[genes]
    corr_matrix = np.nan_to_num(np.corrcoef(df_restricted.values), nan=0.0)
    gene2idx = {g: i for i, g in enumerate(genes)}
    score_pearson = np.array([abs(corr_matrix[gene2idx[e[0]], gene2idx[e[1]]]) for e in candidate_edges])
    
    G = nx.DiGraph()
    G.add_edges_from(prior_edges_list)
    G_undir = G.to_undirected(as_view=True)
    undirected_cache = {g: set(G_undir.neighbors(g)) if G_undir.has_node(g) else set() for g in genes}
    score_topology = np.array([len(undirected_cache[e[0]].intersection(undirected_cache[e[1]])) for e in candidate_edges])
    np.random.seed(42)
    score_topology = score_topology + np.random.rand(n_cands) * 1e-6
    
    # Train GAT
    global_node2idx = {node: i for i, node in enumerate(gene_names)}
    expr_mean = df_expr.mean(axis=1).values
    expr_std = df_expr.std(axis=1).values
    expr_var = df_expr.var(axis=1).values
    pca = PCA(n_components=16, random_state=42)
    expr_pca = pca.fit_transform(df_expr.values)
    expr_features = np.hstack([expr_mean[:, None], expr_std[:, None], expr_var[:, None], expr_pca])
    
    def z_norm(arr):
        mean = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True) + 1e-8
        return (arr - mean) / std

    x_tensor = torch.tensor(z_norm(expr_features), dtype=torch.float)
    prior_src = [global_node2idx[u] for u, v in prior_edges_list if u in global_node2idx and v in global_node2idx]
    prior_tgt = [global_node2idx[v] for u, v in prior_edges_list if u in global_node2idx and v in global_node2idx]
    edge_index = torch.tensor([prior_src, prior_tgt], dtype=torch.long)
    
    test_q_src = [global_node2idx[u] for u, v in candidate_edges]
    test_q_tgt = [global_node2idx[v] for u, v in candidate_edges]
    test_queries = torch.tensor([test_q_src, test_q_tgt], dtype=torch.long)
    
    torch.manual_seed(42)
    np.random.seed(42)
    pos_edges = list(zip(prior_src, prior_tgt))
    num_pos = len(pos_edges)
    eval_edges_idx = list(zip(test_q_src, test_q_tgt))
    np.random.shuffle(eval_edges_idx)
    neg_edges = eval_edges_idx[:num_pos]
    
    pos_edges = np.array(pos_edges)
    neg_edges = np.array(neg_edges)
    np.random.shuffle(pos_edges)
    np.random.shuffle(neg_edges)
    val_size = int(0.2 * num_pos)
    train_queries = np.vstack([pos_edges[val_size:], neg_edges[val_size:]])
    train_labels = np.concatenate([np.ones(len(pos_edges[val_size:])), np.zeros(len(neg_edges[val_size:]))])
    val_queries = np.vstack([pos_edges[:val_size], neg_edges[:val_size]])
    val_labels = np.concatenate([np.ones(len(pos_edges[:val_size])), np.zeros(len(neg_edges[:val_size]))])
    p = np.random.permutation(len(train_queries))
    train_queries, train_labels = train_queries[p], train_labels[p]
    
    model = DirectedGAT(in_channels=x_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_val_auprc = -1
    best_state = None
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tensor, edge_index, torch.tensor(train_queries.T, dtype=torch.long))
        loss = criterion(logits, torch.tensor(train_labels, dtype=torch.float))
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x_tensor, edge_index, torch.tensor(val_queries.T, dtype=torch.long))).numpy()
            val_auprc = average_precision_score(val_labels, val_probs)
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state = model.state_dict().copy()
                
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score_gat = torch.sigmoid(model(x_tensor, edge_index, test_queries)).numpy()
        
    # JEN Reconstruct
    K = 100
    p_dict = {tgt: [] for tgt in genes}
    t_dict = {tgt: [] for tgt in genes}
    g_dict = {tgt: [] for tgt in genes}
    for i, e in enumerate(candidate_edges):
        p_dict[e[1]].append((e[0], score_pearson[i]))
        t_dict[e[1]].append((e[0], score_topology[i]))
        g_dict[e[1]].append((e[0], score_gat[i]))
        
    score_jen_gat = np.zeros(n_cands)
    edge2idx = {e: i for i, e in enumerate(candidate_edges)}
    
    for tgt in genes:
        p_cands = [x[0] for x in sorted(p_dict[tgt], key=lambda x: x[1], reverse=True)[:K]]
        t_cands = [x[0] for x in sorted(t_dict[tgt], key=lambda x: x[1], reverse=True)[:K]]
        g_cands = [x[0] for x in sorted(g_dict[tgt], key=lambda x: x[1], reverse=True)[:K]]
        
        union_cands = list(set(p_cands).union(set(t_cands)).union(set(g_cands)))
        if len(union_cands) == 0: continue
            
        X = df_restricted.loc[union_cands].values.T
        y = df_restricted.loc[tgt].values
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42)
        en.fit(X_scaled, y_scaled)
        
        for i, cand in enumerate(union_cands):
            if (cand, tgt) in edge2idx:
                score_jen_gat[edge2idx[(cand, tgt)]] = abs(en.coef_[i])
                
    # Rank fusion
    rank_gat = rankdata(score_gat) / len(score_gat)
    rank_jen = rankdata(score_jen_gat) / len(score_jen_gat)
    
    score_fusion_75 = 0.75 * rank_gat + 0.25 * rank_jen
    
    # 4. Independent AUPRC / Recall Validation
    print("\n[4] Metric Validation")
    def run_val(name, s, reported_auprc, reported_r30):
        auprc = average_precision_score(y_true, s)
        r30 = independent_recall(y_true, s, candidate_edges, 30)
        
        print(f"--- {name} ---")
        print(f"Ind. AUPRC: {auprc:.6f} | Rep. AUPRC: {reported_auprc:.6f} | Diff: {abs(auprc-reported_auprc):.1e}")
        print(f"Ind. R@30: {r30:.6f} | Rep. R@30: {reported_r30:.6f} | Diff: {abs(r30-reported_r30):.1e}")
        return {'Model': name, 'Ind_AUPRC': auprc, 'Rep_AUPRC': reported_auprc, 'Ind_R@30': r30, 'Rep_R@30': reported_r30}

    metrics = []
    np.random.seed(42)
    score_rnd = np.random.rand(n_cands)
    metrics.append(run_val("Random", score_rnd, 0.000250, 0.057692))
    metrics.append(run_val("Pearson", score_pearson, 0.003087, 0.560897))
    metrics.append(run_val("GAT", score_gat, 0.013204, 1.000000))
    metrics.append(run_val("JEN", score_jen_gat, 0.001056, 0.256410))
    metrics.append(run_val("GAT + JEN (a=0.75)", score_fusion_75, 0.020892, 1.000000))
    
    # 5. Fusion Validation
    print("\n[5] Fusion Validation (Spearman & Jaccard)")
    rho, _ = spearmanr(rank_gat, rank_jen)
    print(f"Spearman(GAT, JEN): {rho:.4f}")
    
    idx_gat = np.argsort(rank_gat)[::-1]
    idx_jen = np.argsort(rank_jen)[::-1]
    
    for K in [10, 30, 50, 100]:
        t_gat = set(idx_gat[:K])
        t_jen = set(idx_jen[:K])
        inter = len(t_gat.intersection(t_jen))
        print(f"K={K}: Jaccard={inter / len(t_gat.union(t_jen)):.4f}")
        
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(os.path.join(results_dir, "PRECISE1K_GAT_JEN_INDEPENDENT_AUDIT.csv"), index=False)
    
if __name__ == "__main__":
    run_audit()
