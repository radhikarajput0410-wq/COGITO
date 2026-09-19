import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning

class JointMultiRegulatorInference:
    """
    V2: Prior-Informed Joint Multi-Regulator Inference model with Soft Priors and Stability Selection.
    """
    def __init__(self, gene_names: list, n_jobs=-1, use_elasticnet=True, stability_runs=50, stability_threshold=0.5):
        self.gene_names = gene_names
        self.num_genes = len(gene_names)
        self.gene2idx = {g: i for i, g in enumerate(gene_names)}
        self.n_jobs = n_jobs
        self.use_elasticnet = use_elasticnet
        self.stability_runs = stability_runs
        self.stability_threshold = stability_threshold

    def _fit_single_run(self, X_predictors, y, weights, random_state=None):
        # 1. Soft Prior Implementation (Feature Rescaling)
        # Minimize ||Y - X \beta||^2 + lambda \sum w_i |\beta_i|
        # Equivalent to: X' = X / w, fit beta', beta = beta' / w
        eps = 1e-4
        safe_weights = np.clip(weights, eps, 1.0)
        X_rescaled = X_predictors / safe_weights
        
        # 2. Bootstrap sampling
        if random_state is not None:
            np.random.seed(random_state)
            n_samples = X_rescaled.shape[0]
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_rescaled = X_rescaled[indices]
            y = y[indices]
            
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_rescaled, y)
            
        # 3. Rescale coefficients back
        actual_coefs = model.coef_ / safe_weights
        return actual_coefs

    def fit_target(self, target_gene: str, candidate_genes: list, X: np.ndarray) -> dict:
        target_idx = self.gene2idx.get(target_gene)
        if target_idx is None:
            raise ValueError(f"Target gene {target_gene} not found.")
            
        if candidate_genes and isinstance(candidate_genes[0], tuple):
            candidate_names = [c[0] for c in candidate_genes if c[0] != target_gene and c[0] in self.gene2idx]
            candidate_weights = np.array([c[1] for c in candidate_genes if c[0] != target_gene and c[0] in self.gene2idx])
        else:
            return {} # Requires scored candidates for V2 Soft Priors
            
        if not candidate_names:
            return {}
            
        candidate_indices = [self.gene2idx[c] for c in candidate_names]
        
        X_samples = X.T
        y = X_samples[:, target_idx]
        X_predictors = X_samples[:, candidate_indices]
        
        # Stability Selection Bootstrapping
        # We run the regression stability_runs times with different bootstraps
        if self.stability_runs > 1:
            all_coefs = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_single_run)(X_predictors, y, candidate_weights, random_state=i) 
                for i in range(self.stability_runs)
            )
            
            coef_matrix = np.vstack(all_coefs)
            
            # Selection probability: fraction of runs where coef != 0
            selection_prob = np.mean(coef_matrix != 0.0, axis=0)
            
            # Mean coefficient across all runs (or just stable runs)
            mean_coefs = np.mean(coef_matrix, axis=0)
            
            results = {}
            for gene_name, prob, mean_c in zip(candidate_names, selection_prob, mean_coefs):
                if prob >= self.stability_threshold:
                    results[gene_name] = {
                        'coef': mean_c,
                        'stability': prob
                    }
        else:
            # Single run (V1 mode with soft priors)
            coefs = self._fit_single_run(X_predictors, y, candidate_weights, random_state=None)
            results = {}
            for gene_name, coef in zip(candidate_names, coefs):
                if coef != 0.0:
                    results[gene_name] = {
                        'coef': coef,
                        'stability': 1.0
                    }
                    
        return results

    def _process_target(self, target, candidates, X):
        results = self.fit_target(target, candidates, X)
        edges = []
        for source, metrics in results.items():
            edges.append({
                'Source': source,
                'Target': target,
                'Joint_Score': abs(metrics['coef']),
                'Stability': metrics['stability'],
                'Direction': np.sign(metrics['coef']),
            })
        return edges

    def infer_network(self, target_to_candidates: dict, X: np.ndarray) -> pd.DataFrame:
        target_items = list(target_to_candidates.items())
        all_edges_lists = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_target)(target, candidates, X)
            for target, candidates in target_items
        )
        
        all_edges = [edge for sublist in all_edges_lists for edge in sublist]
                
        if not all_edges:
            return pd.DataFrame(columns=['Source', 'Target', 'Joint_Score', 'Stability', 'Direction'])
            
        df = pd.DataFrame(all_edges)
        return df.sort_values(by='Joint_Score', ascending=False).reset_index(drop=True)
