import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
from collections import defaultdict
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
# MODEL
# ---------------------------------------------------------
class DirectedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32, heads=4):
        super(DirectedGAT, self).__init__()
        # GATv2 layers natively support directed graphs via edge_index directionality
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
        # query_edges: [2, num_queries] (src, tgt)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        
        # Node embeddings
        src_emb = x[query_edges[0]]
        tgt_emb = x[query_edges[1]]
        
        pair_emb = torch.cat([src_emb, tgt_emb], dim=1)
        scores = self.mlp(pair_emb).squeeze(-1)
        return scores

# ---------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------
def run_experiment():
    print("=== RUNNING LEAKAGE-CONTROLLED GAT EXPERIMENT ===")
    
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
    
    # Global Node mapping across all expression genes for PCA and graph alignment
    node_list = gene_names
    node2idx = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)
    
    # -----------------------------------------------------
    # NODE FEATURES
    # -----------------------------------------------------
    print("Constructing Expression Node Features...")
    # Summary stats
    expr_mean = df_expr.mean(axis=1).values
    expr_std = df_expr.std(axis=1).values
    expr_var = df_expr.var(axis=1).values
    
    # PCA
    pca = PCA(n_components=16, random_state=42)
    expr_pca = pca.fit_transform(df_expr.values)
    
    expr_features = np.hstack([
        expr_mean[:, None],
        expr_std[:, None],
        expr_var[:, None],
        expr_pca
    ])
    
    print("Constructing Structural Node Features (from Prior)...")
    in_deg = np.zeros(num_nodes)
    out_deg = np.zeros(num_nodes)
    
    for u, v in prior_edges_list:
        if u in node2idx and v in node2idx:
            out_deg[node2idx[u]] += 1
            in_deg[node2idx[v]] += 1
            
    tot_deg = in_deg + out_deg
    
    struct_features = np.hstack([
        in_deg[:, None],
        out_deg[:, None],
        tot_deg[:, None],
        np.log1p(in_deg)[:, None],
        np.log1p(out_deg)[:, None],
        np.log1p(tot_deg)[:, None]
    ])
    
    # Normalized feature groups
    def z_norm(arr):
        mean = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True) + 1e-8
        return (arr - mean) / std

    expr_features = z_norm(expr_features)
    struct_features = z_norm(struct_features)
    
    feat_tensors = {
        'expr': torch.tensor(expr_features, dtype=torch.float),
        'struct': torch.tensor(struct_features, dtype=torch.float),
        'combined': torch.tensor(np.hstack([expr_features, struct_features]), dtype=torch.float)
    }
    
    # -----------------------------------------------------
    # GRAPH CONSTRUCTION
    # -----------------------------------------------------
    prior_src = [node2idx[u] for u, v in prior_edges_list if u in node2idx and v in node2idx]
    prior_tgt = [node2idx[v] for u, v in prior_edges_list if u in node2idx and v in node2idx]
    edge_index = torch.tensor([prior_src, prior_tgt], dtype=torch.long)
    
    # -----------------------------------------------------
    # EVALUATION PREP
    # -----------------------------------------------------
    test_q_src = [node2idx[u] for u, v in restricted_edges_list]
    test_q_tgt = [node2idx[v] for u, v in restricted_edges_list]
    test_queries = torch.tensor([test_q_src, test_q_tgt], dtype=torch.long)
    
    # Compute true label arrays
    y_true_test = np.array([1 if e in test_pos_set else 0 for e in restricted_edges_list], dtype=np.int8)

    # -----------------------------------------------------
    # NEGATIVE SAMPLING & SPLITS (Strictly Prior-Driven)
    # -----------------------------------------------------
    def generate_training_data(seed):
        np.random.seed(seed)
        
        # Positives are the prior edges
        pos_edges = list(zip(prior_src, prior_tgt))
        num_pos = len(pos_edges)
        
        # Negatives are sampled from evaluation candidate universe (NOT prior)
        eval_edges_idx = list(zip(test_q_src, test_q_tgt))
        np.random.shuffle(eval_edges_idx)
        
        neg_edges = eval_edges_idx[:num_pos] # 1:1 ratio
        
        # Split 80/20 for training/validation
        pos_edges = np.array(pos_edges)
        neg_edges = np.array(neg_edges)
        
        np.random.shuffle(pos_edges)
        np.random.shuffle(neg_edges)
        
        val_size = int(0.2 * num_pos)
        train_pos, val_pos = pos_edges[val_size:], pos_edges[:val_size]
        train_neg, val_neg = neg_edges[val_size:], neg_edges[:val_size]
        
        train_queries = np.vstack([train_pos, train_neg])
        train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
        
        val_queries = np.vstack([val_pos, val_neg])
        val_labels = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))])
        
        # Shuffle train
        p = np.random.permutation(len(train_queries))
        train_queries, train_labels = train_queries[p], train_labels[p]
        
        return (
            torch.tensor(train_queries.T, dtype=torch.long),
            torch.tensor(train_labels, dtype=torch.float),
            torch.tensor(val_queries.T, dtype=torch.long),
            torch.tensor(val_labels, dtype=torch.float)
        )

    # -----------------------------------------------------
    # TRAINING FUNCTION
    # -----------------------------------------------------
    def train_model(feat_key, seed):
        torch.manual_seed(seed)
        
        x = feat_tensors[feat_key]
        train_q, train_y, val_q, val_y = generate_training_data(seed)
        
        model = DirectedGAT(in_channels=x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auprc = -1
        best_state = None
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            
            logits = model(x, edge_index, train_q)
            loss = criterion(logits, train_y)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_logits = model(x, edge_index, val_q)
                val_probs = torch.sigmoid(val_logits).numpy()
                val_auprc = average_precision_score(val_y.numpy(), val_probs)
                
                if val_auprc > best_val_auprc:
                    best_val_auprc = val_auprc
                    best_state = model.state_dict().copy()
        
        model.load_state_dict(best_state)
        model.eval()
        
        with torch.no_grad():
            test_logits = model(x, edge_index, test_queries)
            test_probs = torch.sigmoid(test_logits).numpy()
            
        aup, _, yt, ys = calc_metrics(test_probs, restricted_edges_list, test_pos_set)
        recs = compute_recall_at_k(yt, ys, restricted_genes, restricted_edges_list)
        return aup, recs

    # -----------------------------------------------------
    # EXECUTE ABLATIONS
    # -----------------------------------------------------
    results = []
    seeds = [42, 123, 2026]
    models = [
        ("Expression-only GAT", "expr"),
        ("Structure-only GAT", "struct"),
        ("Expression+Structure GAT", "combined")
    ]
    
    for model_name, feat_key in models:
        print(f"\nTraining {model_name}...")
        for seed in seeds:
            print(f"  Seed {seed}...")
            aup, recs = train_model(feat_key, seed)
            results.append({
                'model': model_name,
                'seed': seed,
                'auprc': aup,
                'recall_at_10': recs[10],
                'recall_at_30': recs[30],
                'recall_at_50': recs[50],
                'recall_at_100': recs[100]
            })

    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(results_dir, "GAT_PRECISE1K_RESULTS.csv"), index=False)
    
    # Aggregation for printing
    print("\n--- GAT RESULTS ---")
    agg = df_res.groupby('model').agg(['mean', 'std'])
    print(agg[['auprc', 'recall_at_30']])
    print("Done. Results saved to GAT_PRECISE1K_RESULTS.csv")

if __name__ == "__main__":
    run_experiment()
