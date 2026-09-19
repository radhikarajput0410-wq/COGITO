import os
import pandas as pd
import numpy as np

def audit_precise1k():
    print("=== PRECISE-1K DATA INTEGRITY AUDIT ===")
    
    expr_path = "data/external/precise1k/precise1k/data/precise1k/log_tpm_norm_qc.csv"
    trn_path = "data/external/precise1k/precise1k/data/annotation/TRN.csv"
    annot_path = "data/external/precise1k/precise1k/data/annotation/gene_info.csv"
    meta_path = "data/external/precise1k/precise1k/data/precise1k/metadata.csv"
    
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Metadata
    print("\n[Metadata Audit]")
    df_meta = pd.read_csv(meta_path)
    sample_count = len(df_meta)
    print(f"Sample count: {sample_count}")
    print(f"Missing metadata: {df_meta.isnull().sum().sum()}")
    print(f"Duplicated sample IDs: {df_meta.duplicated(subset=df_meta.columns[0]).sum()}")
    
    # 2. Expression
    print("\n[Expression Audit]")
    df_expr = pd.read_csv(expr_path, index_col=0) # index is locus_tag (b-numbers)
    # The matrix is genes x samples. Transpose to samples x genes
    df_expr = df_expr.T
    
    num_samples, num_genes = df_expr.shape
    print(f"Expression matrix dimensions: {num_samples} samples x {num_genes} genes")
    print(f"Missing values: {df_expr.isnull().sum().sum()}")
    print(f"Duplicated genes: {df_expr.columns.duplicated().sum()}")
    print(f"Duplicated samples: {df_expr.index.duplicated().sum()}")
    
    variances = df_expr.var()
    constant_genes = (variances == 0).sum()
    print(f"Constant genes (zero variance): {constant_genes}")
    print(f"Variance distribution: Min={variances.min():.4f}, Max={variances.max():.4f}, Mean={variances.mean():.4f}")
    
    # 3. Regulatory Network
    print("\n[Regulatory Network Audit]")
    df_trn = pd.read_csv(trn_path)
    print(f"Raw TRN shape: {df_trn.shape}")
    print(df_trn.head())
    # The TRN has columns like regulator, gene_id, gene_name
    # regulator is often symbol (cra), gene_id is locus (b0002)
    # Let's map everything to locus_tags for robust evaluation.
    
    # Load Annotation
    df_annot = pd.read_csv(annot_path)
    # map symbol to locus tag
    df_annot_clean = df_annot.dropna(subset=['gene_name', 'locus_tag'])
    symbol_to_locus = {str(row['gene_name']).lower(): str(row['locus_tag']) for _, row in df_annot_clean.iterrows()}
    
    mapping_results = []
    
    # Map Regulators
    mapped_regs = []
    for reg in df_trn['regulator']:
        reg_str = str(reg).lower()
        if reg_str in symbol_to_locus:
            mapped_regs.append(symbol_to_locus[reg_str])
            mapping_results.append({'original_id': reg, 'mapped_id': symbol_to_locus[reg_str], 'source': 'TRN_regulator', 'mapping_status': 'Mapped'})
        else:
            mapped_regs.append(np.nan)
            mapping_results.append({'original_id': reg, 'mapped_id': np.nan, 'source': 'TRN_regulator', 'mapping_status': 'Unmapped'})
            
    df_trn['regulator_b'] = mapped_regs
    
    # Map Targets (gene_id is already locus tag in most cases, but let's verify)
    mapped_targs = []
    for tgt, tgt_name in zip(df_trn['gene_id'], df_trn['gene_name']):
        if pd.notna(tgt) and str(tgt).startswith('b'):
            mapped_targs.append(str(tgt))
            mapping_results.append({'original_id': tgt, 'mapped_id': str(tgt), 'source': 'TRN_target', 'mapping_status': 'Mapped_Direct'})
        else:
            tgt_str = str(tgt_name).lower()
            if tgt_str in symbol_to_locus:
                mapped_targs.append(symbol_to_locus[tgt_str])
                mapping_results.append({'original_id': tgt_name, 'mapped_id': symbol_to_locus[tgt_str], 'source': 'TRN_target', 'mapping_status': 'Mapped_Symbol'})
            else:
                mapped_targs.append(np.nan)
                mapping_results.append({'original_id': tgt_name, 'mapped_id': np.nan, 'source': 'TRN_target', 'mapping_status': 'Unmapped'})
                
    df_trn['target_b'] = mapped_targs
    
    df_mapping = pd.DataFrame(mapping_results)
    df_mapping.to_csv(os.path.join(out_dir, "precise1k_gene_mapping.csv"), index=False)
    
    df_prior = df_trn.dropna(subset=['regulator_b', 'target_b'])
    df_prior = df_prior[['regulator_b', 'target_b']].drop_duplicates()
    
    print(f"Number of edges (mapped): {len(df_prior)}")
    unique_regs = df_prior['regulator_b'].nunique()
    unique_targs = df_prior['target_b'].nunique()
    print(f"Unique regulators: {unique_regs}")
    print(f"Unique targets: {unique_targs}")
    self_loops = sum(df_prior['regulator_b'] == df_prior['target_b'])
    print(f"Self-loops: {self_loops}")
    print(f"Duplicated edges: {df_prior.duplicated().sum()}")
    
    # 4. Intersection
    print("\n[Intersection]")
    expr_genes = set(df_expr.columns)
    net_genes = set(df_prior['regulator_b']).union(set(df_prior['target_b']))
    
    common_genes = expr_genes.intersection(net_genes)
    print(f"Expression genes: {len(expr_genes)}")
    print(f"Network genes: {len(net_genes)}")
    print(f"Common genes: {len(common_genes)}")
    
    genes_lost_expr = len(expr_genes) - len(common_genes)
    genes_lost_net = len(net_genes) - len(common_genes)
    print(f"Genes lost from expression: {genes_lost_expr}")
    print(f"Genes lost from network: {genes_lost_net}")
    
    # Filter prior to common genes
    df_prior_filtered = df_prior[df_prior['regulator_b'].isin(common_genes) & df_prior['target_b'].isin(common_genes)]
    print(f"Final network edges after intersection filtering: {len(df_prior_filtered)}")
    
    df_prior_filtered.to_csv(os.path.join(out_dir, "precise1k_filtered_prior.csv"), index=False)
    
    print("\nAudit completed.")

if __name__ == "__main__":
    audit_precise1k()
