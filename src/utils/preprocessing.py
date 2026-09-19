import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

class GRNPreprocessor:
    """
    Handles the preprocessing of gene expression data and ground-truth edge lists
    to prepare them for downstream Graph Neural Network (GAT) and Joint Inference modeling.
    """
    def __init__(self, scale_data=True):
        """
        Args:
            scale_data (bool): Whether to standardize gene expression (Z-score normalization).
        """
        self.scale_data = scale_data
        self.scaler = StandardScaler()
        
    def process_expression(self, df_expr: pd.DataFrame, train_indices=None) -> tuple[np.ndarray, list]:
        """
        Processes the expression matrix. Converts it to a Genes x Samples orientation 
        if necessary, ensuring it's ready to be used as node features in a GNN.
        
        Args:
            df_expr: DataFrame containing expression data.
            train_indices: Optional list of sample indices to fit the scaler on, preventing 
                           information leak from test samples into the training normalization parameters.
                           If None, fits on all provided data (standard for steady-state GRN tasks where 
                           edges, not samples, are the train/test split).
                           
        Returns:
            X_node_features (np.ndarray): Shape (num_genes, num_samples).
            gene_names (list): List of gene identifiers in order.
        """
        # 1. Identify and drop metadata columns (like 'Time' in DREAM4)
        metadata_cols = [col for col in df_expr.columns if col.lower() in ['time', 'sample', 'condition']]
        df_clean = df_expr.drop(columns=metadata_cols)
        
        # 2. Determine orientation. We want columns to be genes for scaling.
        # Heuristic: if rows > cols, it's likely Genes x Samples. We want Samples x Genes for StandardScaler.
        if df_clean.shape[0] > df_clean.shape[1]:
            df_clean = df_clean.transpose()
            
        gene_names = list(df_clean.columns)
        expr_matrix = df_clean.values # Shape: (Samples, Genes)
        
        # 3. Variance scaling (Z-score normalization per gene across samples)
        # We explicitly verify whether rows correspond to genes or samples before constructing the ML input tensor.
        if self.scale_data:
            if train_indices is not None:
                # Fit ONLY on training samples to strictly prevent data leakage
                self.scaler.fit(expr_matrix[train_indices])
                expr_matrix = self.scaler.transform(expr_matrix)
            else:
                # Fit and transform on all samples (when evaluating topological edge splits)
                expr_matrix = self.scaler.fit_transform(expr_matrix)
                
        # 4. Convert to Node Features for GNN
        # GNNs expect node features of shape (num_nodes, num_features).
        # Here, nodes are genes, and features are the expression levels across samples.
        X_node_features = expr_matrix.T # Shape: (Genes, Samples)
        
        return X_node_features, gene_names

    def process_gold_standard(self, df_edges: pd.DataFrame, gene_names: list) -> pd.DataFrame:
        """
        Processes the gold standard network, mapping gene names to integer indices 
        corresponding to the rows in X_node_features.
        
        Args:
            df_edges: DataFrame of edges (Source, Target, Weight).
            gene_names: List of gene names ordered as in the node features.
            
        Returns:
            Mapped DataFrame with integer indices for 'source_idx' and 'target_idx'.
        """
        # Create a mapping from gene name to integer index (0 to N-1)
        gene2idx = {gene: idx for idx, gene in enumerate(gene_names)}
        
        # Ensure column names are standard
        if len(df_edges.columns) >= 3:
            df_edges.columns = ['Source', 'Target', 'Weight'] + list(df_edges.columns[3:])
        elif len(df_edges.columns) == 2:
            df_edges.columns = ['Source', 'Target']
            df_edges['Weight'] = 1.0 # Assume unweighted edges exist
            
        # Filter edges to only include genes present in our expression matrix
        df_edges = df_edges[df_edges['Source'].isin(gene2idx) & df_edges['Target'].isin(gene2idx)].copy()
        
        # Map to integer indices for PyTorch Geometric / ML indexing
        df_edges['source_idx'] = df_edges['Source'].map(gene2idx)
        df_edges['target_idx'] = df_edges['Target'].map(gene2idx)
        
        return df_edges

    def to_torch_tensors(self, X_node_features: np.ndarray, df_edges: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts the processed numpy arrays into PyTorch tensors.
        
        Returns:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices in COO format (2, num_edges).
        """
        x = torch.tensor(X_node_features, dtype=torch.float32)
        
        sources = df_edges['source_idx'].values
        targets = df_edges['target_idx'].values
        
        # PyTorch Geometric expects edge_index in [2, num_edges] format
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        
        return x, edge_index
