import os
import pandas as pd

def process_precise1k():
    raw_expr_path = os.path.join("data", "external", "precise1k", "precise1k", "data", "precise1k", "log_tpm_norm_qc.csv")
    raw_trn_path = os.path.join("data", "external", "precise1k", "precise1k", "data", "annotation", "TRN.csv")
    raw_annot_path = os.path.join("data", "external", "precise1k", "precise1k", "data", "annotation", "gene_info.csv")
    
    out_expr_path = os.path.join("data", "external", "precise1k_expr.tsv")
    out_trn_path = os.path.join("data", "external", "ecoli_regulondb.tsv")
    
    print("Loading Annotation Map...")
    df_annot = pd.read_csv(raw_annot_path)
    # Some gene_names might be NaN, so drop them
    df_annot = df_annot.dropna(subset=['gene_name'])
    # Map gene symbol (e.g., accB) to locus_tag (e.g., b3255). We lowercase everything to ensure matches.
    symbol_to_locus = {str(k).lower(): v for k, v in zip(df_annot['gene_name'], df_annot['locus_tag'])}
    
    print("Processing PRECISE-1K TRN (Prior)...")
    df_trn = pd.read_csv(raw_trn_path)
    df_prior = df_trn[['regulator', 'gene_name']].drop_duplicates().copy()
    
    # Map to b-numbers
    df_prior['regulator_b'] = df_prior['regulator'].astype(str).str.lower().map(symbol_to_locus)
    df_prior['target_b'] = df_prior['gene_name'].astype(str).str.lower().map(symbol_to_locus)
    
    # Drop rows where we couldn't find a mapping
    df_prior = df_prior.dropna(subset=['regulator_b', 'target_b'])
    
    df_mapped = df_prior[['regulator_b', 'target_b']].copy()
    df_mapped['weight'] = 1.0
    
    df_mapped.to_csv(out_trn_path, sep='\t', index=False, header=False)
    print(f"Saved {len(df_mapped)} prior edges (after ID mapping) to {out_trn_path}")
    
    print("Processing PRECISE-1K Expression...")
    # Expression matrix: Genes x Samples. The CSV has gene IDs or names as the first column.
    df_expr = pd.read_csv(raw_expr_path, index_col=0)
    
    # We should match the genes in expression with the genes in TRN.
    # PRECISE-1K index_col=0 in log_tpm_norm_qc is usually 'gene_id' (b-numbers) or 'gene_name' (e.g. crp)
    # Let's inspect what index the expression matrix uses.
    
    df_expr.to_csv(out_expr_path, sep='\t')
    print(f"Saved expression matrix shape {df_expr.shape} to {out_expr_path}")

if __name__ == "__main__":
    process_precise1k()
