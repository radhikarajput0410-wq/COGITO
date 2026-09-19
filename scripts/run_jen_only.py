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

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# GAT ARCHITECTURE (Expression-only Configuration)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# EXPERIMENT PIPELINE
# ---------------------------------------------------------
def run_experiment():
    print("==================================================")
    print("GAT-GUIDED JOINT MULTI-REGULATOR ELASTIC NET")
    print("==================================================")
    
    out_dir = "data/processed/precise1k_regulondb_frozen_split"
    results_dir = "results"
    
    # 1. LOAD FROZEN BENCHMARK
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
    
    print(f"Benchmark: PRECISE-1K frozen benchmark")
    print(f"Evaluation universe: {len(restricted_edges_list)} pairs")
    
    # Leakage Audits
    intersect_prior_test = len(set(prior_edges_list).intersection(test_pos_set))
    if intersect_prior_test != 0:
        print("STATUS: FAIL. Leakage detected between prior and test sets.")
        sys.exit(1)
        
    print("Leakage audit: PASS")

    # 2. BASELINE: PEARSON & TOPOLOGY
    print("Generating Pearson and Topology baselines...")
    df_restricted = df_expr.loc[restricted_genes]
    corr_matrix = np.nan_to_num(np.corrcoef(df_restricted.values), nan=0.0)
    gene2idx = {g: i for i, g in enumerate(restricted_genes)}
    
    score_pearson = np.array([abs(corr_matrix[gene2idx[e[0]], gene2idx[e[1]]]) for e in restricted_edges_list])
    
    G = nx.DiGraph()
    G.add_edges_from(prior_edges_list)
    G_undir = G.to_undirected(as_view=True)
    undirected_cache = {g: set(G_undir.neighbors(g)) if G_undir.has_node(g) else set() for g in restricted_genes}
    score_topology = np.array([len(undirected_cache[e[0]].intersection(undirected_cache[e[1]])) for e in restricted_edges_list])
    
    # Break ties for ranking consistently
    np.random.seed(42)
    score_topology = score_topology + np.random.rand(len(score_topology)) * 1e-6
    
    # 3. BASELINE: GAT (Expression-feature)
    print("Training GAT for candidate ranking (Seed 42)...")
    node_list = gene_names
    global_node2idx = {node: i for i, node in enumerate(node_list)}
    
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
    
    test_q_src = [global_node2idx[u] for u, v in restricted_edges_list]
    test_q_tgt = [global_node2idx[v] for u, v in restricted_edges_list]
    test_queries = torch.tensor([test_q_src, test_q_tgt], dtype=torch.long)
    
    # Train GAT
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
    
    train_q_t = torch.tensor(train_queries.T, dtype=torch.long)
    train_y_t = torch.tensor(train_labels, dtype=torch.float)
    val_q_t = torch.tensor(val_queries.T, dtype=torch.long)
    val_y_t = torch.tensor(val_labels, dtype=torch.float)
    
    model = DirectedGAT(in_channels=x_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auprc = -1
    best_state = None
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tensor, edge_index, train_q_t)
        loss = criterion(logits, train_y_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x_tensor, edge_index, val_q_t)).numpy()
            val_auprc = average_precision_score(val_y_t.numpy(), val_probs)
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state = model.state_dict().copy()
                
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score_gat = torch.sigmoid(model(x_tensor, edge_index, test_queries)).numpy()

    # 4. JOINT ELASTIC NET
    print("Preparing candidate unions for Joint Elastic Net...")
    K = 100
    
    def build_score_dict(scores):
        d = {tgt: [] for tgt in restricted_genes}
        for i, e in enumerate(restricted_edges_list):
            d[e[1]].append((e[0], scores[i]))
        return d
        
    p_dict = build_score_dict(score_pearson)
    t_dict = build_score_dict(score_topology)
    g_dict = build_score_dict(score_gat)
    
    def run_jen(use_gat=False):
        jen_scores = np.zeros(len(restricted_edges_list))
        edge2idx = {e: i for i, e in enumerate(restricted_edges_list)}
        
        for tgt in restricted_genes:
            # Sort candidates
            p_cands = [x[0] for x in sorted(p_dict[tgt], key=lambda x: x[1], reverse=True)[:K]]
            t_cands = [x[0] for x in sorted(t_dict[tgt], key=lambda x: x[1], reverse=True)[:K]]
            
            union_cands = set(p_cands).union(set(t_cands))
            if use_gat:
                g_cands = [x[0] for x in sorted(g_dict[tgt], key=lambda x: x[1], reverse=True)[:K]]
                union_cands = union_cands.union(set(g_cands))
                
            union_cands = list(union_cands)
            if len(union_cands) == 0:
                continue
                
            # X matrix
            X = df_restricted.loc[union_cands].values.T
            y = df_restricted.loc[tgt].values
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Fit EN
            en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42)
            en.fit(X_scaled, y_scaled)
            
            # Map back coefficients
            for i, cand in enumerate(union_cands):
                e = (cand, tgt)
                if e in edge2idx:
                    jen_scores[edge2idx[e]] = abs(en.coef_[i])
                    
        return jen_scores
        
    print("Running Standard JEN (Pearson U Topology)...")
    score_jen_standard = run_jen(use_gat=False)
    
    print("Running GAT-Guided JEN (Pearson U Topology U GAT)...")
    score_jen_gat = run_jen(use_gat=True)

    # 5. RANK FUSION
    print("Computing Rank Fusions...")
    rank_gat = rankdata(score_gat) / len(score_gat)
    rank_jen = rankdata(score_jen_gat) / len(score_jen_gat)
    
    score_fusion_25 = 0.25 * rank_gat + 0.75 * rank_jen
    score_fusion_50 = 0.50 * rank_gat + 0.50 * rank_jen
    score_fusion_75 = 0.75 * rank_gat + 0.25 * rank_jen

    # 6. EVALUATION
    results = []
    
    np.random.seed(42)
    score_random = np.random.rand(len(restricted_edges_list))
    
    models = {
        "Random": score_random,
        "Pearson": score_pearson,
        "Common Neighbors": score_topology,
        "GAT": score_gat,
        "JEN": score_jen_standard,
        "GAT-guided JEN": score_jen_gat,
        "GAT + JEN (a=0.25)": score_fusion_25,
        "GAT + JEN (a=0.50)": score_fusion_50,
        "GAT + JEN (a=0.75)": score_fusion_75
    }
    
    print("\n--- RESULTS ---")
    y_true_test = np.array([1 if e in test_pos_set else 0 for e in restricted_edges_list], dtype=np.int8)
    
    for name, s in models.items():
        aup, pos_count, yt, ys = calc_metrics(s, restricted_edges_list, test_pos_set)
        recs = compute_recall_at_k(yt, ys, restricted_genes, restricted_edges_list)
        results.append({
            'Model': name,
            'AUPRC': aup,
            'R@10': recs[10],
            'R@30': recs[30],
            'R@50': recs[50],
            'R@100': recs[100]
        })
        print(f"{name:20s} AUPRC: {aup:.6f} | R@30: {recs[30]:.6f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(results_dir, "PRECISE1K_GAT_GUIDED_JEN_RESULTS.csv"), index=False)

    # 7. HIGH-MULTI-REGULATOR ANALYSIS
    print("\n--- HIGH-MULTI-REGULATOR ANALYSIS (GAT-guided JEN) ---")
    target_pos_counts = {tgt: 0 for tgt in restricted_genes}
    for e in test_pos_set:
        if e in restricted_edges_list:
            target_pos_counts[e[1]] += 1
            
    group_1 = [tgt for tgt, c in target_pos_counts.items() if c == 1]
    group_23 = [tgt for tgt, c in target_pos_counts.items() if c in [2, 3]]
    group_4p = [tgt for tgt, c in target_pos_counts.items() if c >= 4]
    
    def evaluate_group(tgt_group, group_name):
        if not tgt_group: return
        group_edges = [e for e in restricted_edges_list if e[1] in tgt_group]
        group_y_true = np.array([1 if e in test_pos_set else 0 for e in group_edges], dtype=np.int8)
        
        edge2idx = {e: i for i, e in enumerate(restricted_edges_list)}
        group_indices = [edge2idx[e] for e in group_edges]
        group_score = score_jen_gat[group_indices]
        
        aup = average_precision_score(group_y_true, group_score) if np.sum(group_y_true) > 0 else 0
        recs = compute_recall_at_k(group_y_true, group_score, tgt_group, group_edges)
        print(f"{group_name:10s} (N={len(tgt_group):3d}) | AUPRC: {aup:.6f} | R@30: {recs[30]:.6f}")

    evaluate_group(group_1, "1 Reg")
    evaluate_group(group_23, "2-3 Regs")
    evaluate_group(group_4p, "4+ Regs")

    print("Done.")

if __name__ == "__main__":
    run_experiment()
