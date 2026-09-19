import pandas as pd
import numpy as np

class Precise1KAdapter:
    """
    Adapter for the PRECISE-1K E. coli dataset.
    This module inspects, cleans, and standardizes real transcriptomic data 
    from PRECISE-1K and prior knowledge from RegulonDB.
    """
    def __init__(self, expr_path: str, gold_path: str):
        self.expr_path = expr_path
        self.gold_path = gold_path

    def load_and_standardize(self):
        """
        Loads the data, assuming standard matrix format, and handles missing values/duplicates.
        Returns:
            df_expr: Cleaned expression DataFrame (Genes x Samples or Samples x Genes)
            df_gold: Cleaned prior knowledge DataFrame (Source, Target, Weight)
        """
        print("Loading PRECISE-1K (Real Transcriptomic Benchmark) dataset...")
        
        try:
            # We assume a standard TSV for expression: index=genes, columns=samples
            df_expr = pd.read_csv(self.expr_path, sep='\t', index_col=0)
            
            # Handle missing values by median imputation
            if df_expr.isnull().values.any():
                print("Imputing missing values in PRECISE-1K expression data...")
                df_expr = df_expr.fillna(df_expr.median())
                
            # Remove duplicate genes if any exist
            df_expr = df_expr[~df_expr.index.duplicated(keep='first')]
            
            # Ensure it's returned as Samples x Genes for the preprocessor
            df_expr = df_expr.transpose()
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Expression file not found: {self.expr_path}. Please download PRECISE-1K.")

        try:
            # We assume a standard TSV for priors without header: Source, Target, (optional Weight)
            df_gold = pd.read_csv(self.gold_path, sep='\t', header=None)
            if df_gold.shape[1] == 2:
                df_gold[2] = 1.0 # Add a weight column of 1.0 if missing
        except FileNotFoundError:
            raise FileNotFoundError(f"Prior network file not found: {self.gold_path}. Please download RegulonDB priors.")
            
        return df_expr, df_gold
