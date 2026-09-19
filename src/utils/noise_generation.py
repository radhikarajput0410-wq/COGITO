import numpy as np
import pandas as pd

class NoiseGenerator:
    """
    Systematically degrades a transcriptomic expression matrix to simulate 
    realistic biological and technical measurement errors. Used to evaluate 
    the robustness of the GRN inference pipeline.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def add_gaussian_noise(self, X: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Adds Gaussian white noise to the expression matrix.
        
        Args:
            X: Original expression matrix (numpy array).
            noise_level: Float between 0.0 and 1.0 representing the standard deviation 
                         of the noise relative to the standard deviation of the data.
                         (e.g., 0.10 means 10% noise).
                         
        Returns:
            Noisy expression matrix of the same shape.
        """
        if noise_level == 0.0:
            return X.copy()
            
        # Calculate the standard deviation of the original data to scale the noise appropriately
        data_std = np.std(X)
        
        # Generate Gaussian noise with mean=0 and scaled standard deviation
        noise = np.random.normal(0, data_std * noise_level, X.shape)
        
        # Add noise and ensure we don't drop below 0 (since gene expression cannot be negative)
        X_noisy = X + noise
        X_noisy = np.clip(X_noisy, a_min=0.0, a_max=None)
        
        return X_noisy

    def add_sparse_outliers(self, X: np.ndarray, outlier_fraction: float, multiplier: float = 3.0) -> np.ndarray:
        """
        Simulates technical artifacts (like RNA-seq amplification bias or batch effects)
        by randomly selecting a fraction of data points and artificially inflating them.
        
        Args:
            X: Original expression matrix.
            outlier_fraction: Fraction of total elements in X to corrupt (0.0 to 1.0).
            multiplier: How much to multiply the corrupted elements by.
            
        Returns:
            Corrupted expression matrix.
        """
        if outlier_fraction == 0.0:
            return X.copy()
            
        X_corrupted = X.copy()
        
        # Calculate how many elements to corrupt
        total_elements = X.size
        num_outliers = int(total_elements * outlier_fraction)
        
        # Randomly select indices
        indices = np.random.choice(total_elements, num_outliers, replace=False)
        
        # Convert 1D indices to 2D coordinates
        row_indices, col_indices = np.unravel_index(indices, X.shape)
        
        # Inflate the selected values
        X_corrupted[row_indices, col_indices] *= multiplier
        
        return X_corrupted

    def generate_noisy_datasets(self, df_expr: pd.DataFrame, noise_levels: list = [0.05, 0.10, 0.20, 0.30]) -> dict:
        """
        Convenience function to generate a suite of noisy DataFrames from a clean one.
        
        Args:
            df_expr: Clean pandas DataFrame of expression data.
            noise_levels: List of Gaussian noise levels to apply.
            
        Returns:
            Dictionary mapping noise level (float) to noisy DataFrame.
        """
        # We only add noise to numeric columns. Metadata like 'Time' must be preserved.
        numeric_cols = df_expr.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df_expr.select_dtypes(exclude=[np.number]).columns
        
        X_clean = df_expr[numeric_cols].values
        
        noisy_datasets = {}
        
        for level in noise_levels:
            X_noisy = self.add_gaussian_noise(X_clean, level)
            
            # Reconstruct DataFrame
            df_noisy = pd.DataFrame(X_noisy, columns=numeric_cols, index=df_expr.index)
            
            # Reattach non-numeric metadata
            for col in non_numeric_cols:
                df_noisy[col] = df_expr[col]
                
            # Reorder columns to match original
            df_noisy = df_noisy[df_expr.columns]
            noisy_datasets[level] = df_noisy
            
        return noisy_datasets
