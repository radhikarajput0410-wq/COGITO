import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class FinalEdgeScorer:
    """
    V2: Meta-Model Evidence Fusion
    Integrates GAT Score, Joint Coefficient, Stability Score, and Prior Evidence into a final edge probability.
    Supports Logistic Regression (V1) or Random Forest (V2) to preserve score variance.
    """
    def __init__(self, fusion_mode='rf'):
        """
        Args:
            fusion_mode (str): 'lr' for Logistic Regression, 'rf' for Random Forest
        """
        self.fusion_mode = fusion_mode
        if self.fusion_mode == 'rf':
            # Use shallow RF to prevent overfitting and preserve variance
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
        else:
            self.model = LogisticRegression(class_weight='balanced', random_state=42)
            
        self.scaler = StandardScaler()

    def _prepare_features(self, top_k_dict, df_joint_scores, df_temporal_scores, prior_set):
        # Convert dictionaries/dataframes into a unified feature matrix
        all_edges = set()
        
        gat_map = {}
        for tgt, c_list in top_k_dict.items():
            for src, score in c_list:
                edge = (src, tgt)
                all_edges.add(edge)
                gat_map[edge] = score
                
        joint_map = {}
        stability_map = {}
        if df_joint_scores is not None and not df_joint_scores.empty:
            for _, row in df_joint_scores.iterrows():
                edge = (row['Source'], row['Target'])
                all_edges.add(edge)
                joint_map[edge] = row['Joint_Score']
                stability_map[edge] = row.get('Stability', 0.0)
                
        temporal_map = {}
        if df_temporal_scores is not None and not df_temporal_scores.empty:
            for _, row in df_temporal_scores.iterrows():
                edge = (row['Source'], row['Target'])
                all_edges.add(edge)
                temporal_map[edge] = row['Temporal_Score']
                
        data = []
        for src, tgt in all_edges:
            data.append({
                'Source': src,
                'Target': tgt,
                'GAT_Score': gat_map.get((src, tgt), 0.0),
                'Joint_Score': joint_map.get((src, tgt), 0.0),
                'Stability_Score': stability_map.get((src, tgt), 0.0),
                'Temporal_Score': temporal_map.get((src, tgt), 0.0),
                'Prior_Evidence': 1.0 if (src, tgt) in prior_set else 0.0
            })
            
        return pd.DataFrame(data)

    def fit_and_score(self, top_k_dict, df_joint_scores, df_temporal_scores, train_prior_set, val_prior_set):
        
        # 1. Prepare Validation Data to train the fusion model
        df_features = self._prepare_features(top_k_dict, df_joint_scores, df_temporal_scores, train_prior_set)
        
        feature_cols = ['GAT_Score', 'Joint_Score', 'Stability_Score', 'Temporal_Score', 'Prior_Evidence']
        
        # Target: 1 if in val_prior_set else 0
        df_features['y'] = df_features.apply(lambda row: 1 if (row['Source'], row['Target']) in val_prior_set else 0, axis=1)
        
        X = df_features[feature_cols].values
        y = df_features['y'].values
        
        # Must have both classes to train
        if len(np.unique(y)) > 1:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            # Predict probabilities
            probs = self.model.predict_proba(X_scaled)[:, 1]
            df_features['Final_Score'] = probs
            
            if self.fusion_mode == 'lr':
                print("\nLearned Meta-Model Fusion Weights (Logistic Regression Coefficients):")
                for name, coef in zip(feature_cols, self.model.coef_[0]):
                    print(f"{name}: {coef:.4f}")
            elif self.fusion_mode == 'rf':
                print("\nLearned Meta-Model Feature Importances (Random Forest):")
                for name, imp in zip(feature_cols, self.model.feature_importances_):
                    print(f"{name}: {imp:.4f}")
        else:
            print("Warning: Meta-model could not be trained due to class collapse in validation set. Falling back to Joint_Score or GAT_Score.")
            # Fallback
            if 'Joint_Score' in df_features.columns and df_features['Joint_Score'].sum() > 0:
                df_features['Final_Score'] = df_features['Joint_Score']
            else:
                df_features['Final_Score'] = df_features['GAT_Score']
                
        df_final = df_features[['Source', 'Target', 'Final_Score']].sort_values(by='Final_Score', ascending=False)
        return df_final
