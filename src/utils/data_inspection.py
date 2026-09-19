import os
import glob
import pandas as pd
import numpy as np

def inspect_dataset_directory(data_dir: str):
    """
    Scans a directory for dataset files and automatically infers their 
    structure, dimensions, separators, and missing values.
    
    Args:
        data_dir (str): Path to the directory containing raw data files.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    files = glob.glob(os.path.join(data_dir, "*.*"))
    if not files:
        print(f"No files found in '{data_dir}'.")
        return

    print(f"--- Data Format Inspection: {data_dir} ---")
    print(f"Found {len(files)} files.\n")

    for file_path in files:
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        
        print(f"File: {filename}")
        print(f"Extension: {ext}")
        
        # We only attempt to parse text-based tabular files.
        if ext in ['.csv', '.tsv', '.txt']:
            # Try to infer the separator by reading the first line
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    sep = '\t'
                    print("Inferred separator: TAB (\\t)")
                elif ',' in first_line:
                    sep = ','
                    print("Inferred separator: COMMA (,)")
                elif ';' in first_line:
                    sep = ';'
                    print("Inferred separator: SEMICOLON (;)")
                else:
                    sep = r'\s+'
                    print("Inferred separator: WHITESPACE")
            
            try:
                # Load the dataframe to inspect dimensions and data
                df = pd.read_csv(file_path, sep=sep, engine='python')
                
                print(f"Dimensions (Rows x Cols): {df.shape[0]} x {df.shape[1]}")
                print(f"Columns: {list(df.columns)[:5]} ... (truncated)")
                
                missing_values = df.isnull().sum().sum()
                print(f"Total Missing Values: {missing_values}")
                print(f"Data Types:\n{df.dtypes.value_counts().to_string()}")
                
                # We need to determine if the matrix is genes x samples or samples x genes.
                # In many biological datasets, genes are rows. However, in ML, samples are usually rows.
                # We apply a heuristic: if column names look like gene identifiers (e.g., 'G1', 'gene_1') 
                # or if the number of columns > number of rows, it might be samples x genes.
                if df.shape[1] > 2: # Exclude edge lists (typically 2-3 columns: source, target, weight)
                    # Heuristic for orientation
                    col_str = str(list(df.columns)).lower()
                    if 'gene' in col_str or 'g' in col_str:
                        # But are they in columns or rows?
                        if df.shape[1] > df.shape[0]:
                            print("Orientation heuristic: Likely [Samples x Genes] based on dimensions.")
                        else:
                            print("Orientation heuristic: Likely [Genes x Samples] based on dimensions.")
                    else:
                         print("Orientation heuristic: Unclear, please verify manually.")
                         
                print("First 2 rows:")
                print(df.head(2).to_string(index=False))
                
            except Exception as e:
                print(f"Failed to parse {filename} as tabular data. Error: {e}")
        else:
            print("Skipping detailed parse (not a recognized tabular format).")
            
        print("-" * 40)

if __name__ == "__main__":
    # Target the raw data directory for inspection
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "dream4")
    inspect_dataset_directory(raw_dir)
