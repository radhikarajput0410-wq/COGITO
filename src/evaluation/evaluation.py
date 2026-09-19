import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, f1_score

class GRNEvaluator:
    """
    Handles the evaluation of predicted gene regulatory networks against a known ground truth.
    Prioritizes Area Under the Precision-Recall Curve (AUPRC) due to severe class imbalance.
    """
    def __init__(self, ground_truth_df: pd.DataFrame, all_genes: list):
        """
        Args:
            ground_truth_df: DataFrame containing the true edges (Source, Target, Weight).
                             Assumes Weight > 0 implies an edge exists.
            all_genes: List of all gene names in the dataset to calculate true negatives.
        """
        self.all_genes = all_genes
        self.num_genes = len(all_genes)
        
        # Create a set of true edges for fast lookup
        self.true_edges_set = set(
            zip(ground_truth_df['Source'], ground_truth_df['Target'])
        )
        
        # Calculate positive-class prevalence (Baseline AUPRC for a random classifier)
        self.total_possible_edges = self.num_genes * (self.num_genes - 1)
        self.total_true_edges = len(self.true_edges_set)
        self.prevalence = self.total_true_edges / self.total_possible_edges if self.total_possible_edges > 0 else 0.0

    def evaluate_predictions(self, predicted_edges_df: pd.DataFrame, k_thresholds=[50, 100, 500]) -> dict:
        """
        Evaluates the predicted edges against the ground truth.
        
        Args:
            predicted_edges_df: DataFrame of predicted edges, sorted by 'Score' descending.
                                Expected columns: ['Source', 'Target', 'Score'].
            k_thresholds: List of K values for Precision@K and Recall@K.
            
        Returns:
            Dictionary containing evaluation metrics (AUPRC, AUROC, Precision@K, etc.).
        """
        # Ensure predictions are sorted by score
        preds = predicted_edges_df.sort_values(by='Score', ascending=False).copy()
        
        # Map true labels to the predicted edges
        preds['y_true'] = preds.apply(lambda row: 1 if (row['Source'], row['Target']) in self.true_edges_set else 0, axis=1)
        
        y_true_sorted = preds['y_true'].values
        y_scores_sorted = preds['Score'].values
        
        # If the predictor didn't output scores for all possible pairs, we assume the rest are 0
        missing_edges = self.total_possible_edges - len(y_true_sorted)
        if missing_edges > 0:
            # The missing true labels depend on how many true edges were completely missed by the predictor
            missed_true_edges = self.total_true_edges - y_true_sorted.sum()
            y_true_full = np.concatenate([y_true_sorted, np.ones(missed_true_edges), np.zeros(missing_edges - missed_true_edges)])
            y_scores_full = np.concatenate([y_scores_sorted, np.zeros(missing_edges)])
        else:
            y_true_full = y_true_sorted
            y_scores_full = y_scores_sorted
            
        # 1. AUPRC - Area Under Precision-Recall Curve (Primary Metric)
        precision, recall, _ = precision_recall_curve(y_true_full, y_scores_full)
        auprc = auc(recall, precision)
        # Alternatively, average_precision_score handles step interpolation slightly differently
        ap_score = average_precision_score(y_true_full, y_scores_full)
        
        # 2. AUROC - Area Under Receiver Operating Characteristic Curve
        fpr, tpr, _ = roc_curve(y_true_full, y_scores_full)
        auroc = auc(fpr, tpr)
        
        # 3. Precision@K and Recall@K
        metrics = {
            'AUPRC': auprc,
            'Average_Precision': ap_score,
            'AUROC': auroc,
            'Prevalence': self.prevalence,
            'Total_True_Edges': self.total_true_edges
        }
        
        for k in k_thresholds:
            if k > len(y_true_sorted):
                continue
            
            top_k_true = y_true_sorted[:k]
            prec_at_k = top_k_true.sum() / k
            rec_at_k = top_k_true.sum() / self.total_true_edges if self.total_true_edges > 0 else 0
            
            metrics[f'Precision@{k}'] = prec_at_k
            metrics[f'Recall@{k}'] = rec_at_k
            
        # 4. Global Precision, Recall, F1 (Using a default score threshold, e.g., Top N edges where N = True Edges)
        # This is a standard heuristic in GRN inference when no hard threshold is known.
        threshold_idx = min(self.total_true_edges, len(y_scores_sorted) - 1)
        if threshold_idx > 0:
            hard_threshold = y_scores_sorted[threshold_idx]
            y_pred_binary = (y_scores_full >= hard_threshold).astype(int)
            
            # Recalculate using sklearn
            metrics['F1_Score'] = f1_score(y_true_full, y_pred_binary)
            
        return metrics

    def evaluate_candidate_recall(self, top_k_dict: dict) -> dict:
        """
        Calculates Candidate Recall@K for the GAT prioritization step to measure
        the 'Missing Regulator Problem'.
        
        Args:
            top_k_dict: Dictionary mapping target to list of candidate tuples (source, score).
            
        Returns:
            Dictionary containing Candidate Recall metrics.
        """
        if self.total_true_edges == 0:
            return {'Candidate_Recall@K': 0.0}
            
        true_regulators_captured = 0
        
        for target, candidates in top_k_dict.items():
            candidate_sources = {c[0] for c in candidates}
            for source in candidate_sources:
                if (source, target) in self.true_edges_set:
                    true_regulators_captured += 1
                    
        recall = true_regulators_captured / self.total_true_edges
        return {'Candidate_Recall@K': recall}

    def generate_report(self, metrics: dict, model_name: str) -> str:
        """
        Formats the metrics dictionary into a readable text report.
        """
        report = f"--- Evaluation Report: {model_name} ---\n"
        report += f"Positive-Class Prevalence (Random Baseline AUPRC): {metrics['Prevalence']:.5f}\n"
        report += f"AUPRC (Area Under PR Curve): {metrics['AUPRC']:.4f}\n"
        report += f"AUROC: {metrics['AUROC']:.4f}\n"
        if 'F1_Score' in metrics:
            report += f"F1-Score (Top-N heuristic): {metrics['F1_Score']:.4f}\n"
        
        for key in metrics.keys():
            if '@' in key:
                report += f"{key}: {metrics[key]:.4f}\n"
                
        report += "--------------------------------------\n"
        return report
