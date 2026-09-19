import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATCandidatePrioritizer(nn.Module):
    """
    A Graph Attention Network (GAT) designed to rank candidate regulators for GRN expansion.
    It learns structural embeddings by passing transcriptomic node features over the 
    known (prior) biological network, leveraging attention to weigh neighborhood importance.
    """
    def __init__(self, num_node_features: int, hidden_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.2):
        """
        Args:
            num_node_features (int): Dimensionality of the input node features (e.g., number of samples).
            hidden_channels (int): Dimensionality of the hidden representation.
            out_channels (int): Dimensionality of the final node embedding.
            heads (int): Number of multi-head attentions.
            dropout (float): Dropout probability to prevent overfitting.
        """
        super(GATCandidatePrioritizer, self).__init__()
        
        # First GAT layer (expands channels via multi-head attention)
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=dropout)
        
        # Second GAT layer (aggregates heads back down to out_channels)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute node embeddings.
        
        Args:
            x: Node features tensor (num_genes, num_samples).
            edge_index: The prior known GRN edges (2, num_prior_edges).
            
        Returns:
            Node embeddings of shape (num_genes, out_channels).
        """
        # Pass features and prior structure through first GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Pass through second GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x

class EdgePredictor(nn.Module):
    """
    Takes the GAT node embeddings and scores the likelihood of a regulatory edge 
    between any two genes using a bilinear or MLP scoring function.
    """
    def __init__(self, embedding_dim: int):
        super(EdgePredictor, self).__init__()
        # We use a simple MLP to project the concatenated source and target embeddings to an edge score
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid() # Outputs a score between 0 and 1
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Scores specific edges.
        
        Args:
            z: Node embeddings from the GAT (num_genes, embedding_dim).
            edge_index: The candidate edges to score (2, num_candidate_edges).
            
        Returns:
            Scores for each candidate edge (num_candidate_edges, 1).
        """
        # Extract embeddings for the source and target nodes of the specified edges
        source_embeddings = z[edge_index[0]]
        target_embeddings = z[edge_index[1]]
        
        # Concatenate them
        edge_features = torch.cat([source_embeddings, target_embeddings], dim=-1)
        
        # Predict score
        return self.mlp(edge_features).squeeze(-1)

def get_top_k_candidates(scores: torch.Tensor, candidate_edge_index: torch.Tensor, gene_names: list, top_k: int = 50) -> dict:
    """
    Converts raw tensor scores into a dictionary mapping {target: [top K candidate sources]}.
    """
    scores_np = scores.detach().cpu().numpy()
    sources_np = candidate_edge_index[0].cpu().numpy()
    targets_np = candidate_edge_index[1].cpu().numpy()
    
    target_to_candidates = {}
    
    # Iterate over all scored edges
    for src_idx, tgt_idx, score in zip(sources_np, targets_np, scores_np):
        tgt_name = gene_names[tgt_idx]
        src_name = gene_names[src_idx]
        
        if tgt_name not in target_to_candidates:
            target_to_candidates[tgt_name] = []
            
        target_to_candidates[tgt_name].append((src_name, score))
        
    # Sort by score and keep Top-K
    final_dict = {}
    for tgt_name, c_list in target_to_candidates.items():
        c_list.sort(key=lambda x: x[1], reverse=True)
        final_dict[tgt_name] = [(src, float(score)) for src, score in c_list[:top_k]]
        
    return final_dict

import numpy as np
import scipy.stats

def get_adaptive_k_candidates(scores: torch.Tensor, candidate_edge_index: torch.Tensor, gene_names: list, 
                              k_min: int = 20, k_max: int = 100) -> dict:
    """
    V2: Adaptive Candidate Selection
    Instead of a fixed Top-K, K scales dynamically based on the GAT score entropy.
    Targets with high score entropy (high uncertainty / many viable candidates) get a larger K.
    """
    scores_np = scores.detach().cpu().numpy()
    sources_np = candidate_edge_index[0].cpu().numpy()
    targets_np = candidate_edge_index[1].cpu().numpy()
    
    target_to_candidates = {}
    
    for src_idx, tgt_idx, score in zip(sources_np, targets_np, scores_np):
        tgt_name = gene_names[tgt_idx]
        src_name = gene_names[src_idx]
        
        if tgt_name not in target_to_candidates:
            target_to_candidates[tgt_name] = []
            
        target_to_candidates[tgt_name].append((src_name, score))
        
    final_dict = {}
    for tgt_name, c_list in target_to_candidates.items():
        # Sort by score
        c_list.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate Entropy of the top K_max scores to measure uncertainty
        top_scores = [float(x[1]) for x in c_list[:k_max]]
        if sum(top_scores) > 0:
            probs = np.array(top_scores) / sum(top_scores)
            entropy = scipy.stats.entropy(probs)
        else:
            entropy = 0
            
        # Max entropy for K_max uniform distribution is ln(K_max)
        max_entropy = np.log(k_max)
        entropy_ratio = min(1.0, entropy / max_entropy) if max_entropy > 0 else 0
        
        # Scale K dynamically
        adaptive_k = int(k_min + (k_max - k_min) * entropy_ratio)
        
        final_dict[tgt_name] = [(src, float(score)) for src, score in c_list[:adaptive_k]]
        
    return final_dict

class CandidateUnion:
    """
    V2: Multi-Source Candidate Recovery
    Merges GAT candidates, Pearson correlation top targets, and known Prior edges
    to maximize initial recall before soft-prior penalization.
    """
    def __init__(self, gene_names, train_prior_set):
        self.gene_names = gene_names
        self.train_prior_set = train_prior_set

    def merge_candidates(self, gat_candidates_dict, df_correlation, corr_top_k=10):
        final_candidates = {}
        
        for target in self.gene_names:
            candidate_set = {} # Map source -> (max_score, source_type)
            
            # 1. GAT Candidates
            for src, score in gat_candidates_dict.get(target, []):
                candidate_set[src] = score
                
            # 2. Prior Edges (Force inclusion with high score if not present)
            for src in self.gene_names:
                if (src, target) in self.train_prior_set:
                    if src not in candidate_set:
                        candidate_set[src] = 1.0 # High confidence for prior
                        
            # 3. Correlation Candidates (Top N by absolute Pearson)
            if df_correlation is not None:
                tgt_corr = df_correlation[df_correlation['Target'] == target].nlargest(corr_top_k, 'Score')
                for _, row in tgt_corr.iterrows():
                    src = row['Source']
                    if src not in candidate_set:
                        candidate_set[src] = abs(row['Score']) # Use correlation as pseudo-score
                        
            # Sort final merged set by score
            sorted_candidates = sorted(candidate_set.items(), key=lambda x: x[1], reverse=True)
            final_candidates[target] = sorted_candidates
            
        return final_candidates
