import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LassoCV, ElasticNetCV
import warnings
from sklearn.exceptions import ConvergenceWarning

class BaselineModels:
    """
    Implements standard candidate-wise Gene Regulatory Network (GRN) inference algorithms.
    These serve as the baseline against which the joint multi-regulator approach will be evaluated.
    """
    def __init__(self, gene_names: list, n_jobs=-1):
        """
        Args:
            gene_names (list): List of gene identifiers corresponding to the rows of the expression matrix.
            n_jobs (int): Number of cores to use for cross-validation in penalized regression.
        """
        self.gene_names = gene_names
        self.num_genes = len(gene_names)
        self.n_jobs = n_jobs

    def _format_results(self, edge_scores, method_name):
        """
        Converts an NxN adjacency matrix of scores into a flattened edge list DataFrame.
        """
        # Create a boolean mask to exclude self-loops (diagonal)
        mask = ~np.eye(self.num_genes, dtype=bool)
        
        sources, targets = np.where(mask)
        scores = edge_scores[mask]
        
        df = pd.DataFrame({
            'Source': np.array(self.gene_names)[sources],
            'Target': np.array(self.gene_names)[targets],
            'Score': np.abs(scores), # Absolute value for ranking edge confidence
            'Direction': np.sign(scores),
            'Method': method_name
        })
        
        # Sort by absolute score descending
        return df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    def run_correlation(self, X: np.ndarray, method='pearson') -> pd.DataFrame:
        """
        Computes undirected edges based on simple pairwise correlation.
        
        Args:
            X: Node features array of shape (num_genes, num_samples).
            method: 'pearson' or 'spearman'.
            
        Returns:
            DataFrame of ranked edges.
        """
        if method == 'pearson':
            # Vectorized Pearson correlation
            # np.corrcoef expects variables as rows, observations as columns
            corr_matrix = np.corrcoef(X)
            # Replace NaNs with 0 (zero variance genes)
            edge_scores = np.nan_to_num(corr_matrix, nan=0.0)
            # Zero out diagonal
            np.fill_diagonal(edge_scores, 0.0)
        else:
            edge_scores = np.zeros((self.num_genes, self.num_genes))
            for i in range(self.num_genes):
                for j in range(self.num_genes):
                    if i != j:
                        r, _ = spearmanr(X[i], X[j])
                        edge_scores[i, j] = r if not np.isnan(r) else 0.0
                        
        return self._format_results(edge_scores, method.capitalize())

    def run_candidate_wise_lasso(self, X: np.ndarray) -> pd.DataFrame:
        """
        Performs standard candidate-wise LASSO regression.
        For each target gene Y, it uses all other genes as independent variables X.
        This is a 'global' joint regression, which is computationally expensive (O(p^3)) 
        and often dilutes localized signals compared to a targeted prior-informed joint inference.
        
        Args:
            X: Node features array of shape (num_genes, num_samples).
            
        Returns:
            DataFrame of ranked directed edges.
        """
        # Suppress convergence warnings during cross-validation hyperparameter search
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        edge_scores = np.zeros((self.num_genes, self.num_genes))
        
        # Transpose X to (samples, genes) for sklearn
        X_samples = X.T 
        
        for target_idx in range(self.num_genes):
            # Target expression vector
            y = X_samples[:, target_idx]
            
            # Predictor matrix (all other genes)
            mask = np.ones(self.num_genes, dtype=bool)
            mask[target_idx] = False
            X_predictors = X_samples[:, mask]
            
            # Fit Lasso with Cross-Validation for optimal alpha (regularization strength)
            # This is standard candidate-wise inference where no biological prior restricts the candidates.
            lasso = LassoCV(cv=5, n_jobs=self.n_jobs, random_state=42)
            lasso.fit(X_predictors, y)
            
            # Store coefficients back into the full NxN matrix
            edge_scores[mask, target_idx] = lasso.coef_
            
        return self._format_results(edge_scores, 'Candidate-wise LASSO')
    
    def run_candidate_wise_elasticnet(self, X: np.ndarray) -> pd.DataFrame:
        """
        Performs candidate-wise Elastic Net regression, adding L2 penalty to handle 
        highly correlated candidate regulators (collinearity) better than pure LASSO.
        """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        edge_scores = np.zeros((self.num_genes, self.num_genes))
        X_samples = X.T 
        
        for target_idx in range(self.num_genes):
            y = X_samples[:, target_idx]
            mask = np.ones(self.num_genes, dtype=bool)
            mask[target_idx] = False
            X_predictors = X_samples[:, mask]
            
            enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv=5, n_jobs=self.n_jobs, random_state=42)
            enet.fit(X_predictors, y)
            
            edge_scores[mask, target_idx] = enet.coef_
            
        return self._format_results(edge_scores, 'Candidate-wise ElasticNet')
