"""
Gram-Schmidt Orthogonalization for VAD Direction Vectors

This script reads the direction vectors from the results directory,
applies Gram-Schmidt orthogonalization to remove leakage between dimensions,
and saves the orthogonalized vectors for each model and language.

The Gram-Schmidt process ensures that the Valence, Arousal, and Dominance vectors are 
mutually orthogonal while preserving the direction of the first vector (Valence).
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Configuration
languages = ['arabic', 'english']
DIMENSION_ORDER = ['V', 'A', 'D']  # Order for orthogonalization (V is preserved)


def gram_schmidt(vectors):
    """
    Apply Gram-Schmidt orthogonalization to a list of vectors.
    
    Args:
        vectors: List of numpy arrays representing vectors to orthogonalize
        
    Returns:
        List of orthogonalized (and normalized) numpy arrays
    """
    orthogonal_vectors = []
    
    for i, v in enumerate(vectors):
        # Start with the current vector
        u = v.copy().astype(np.float64)
        
        # Subtract projections onto all previous orthogonal vectors
        for prev_u in orthogonal_vectors:
            if np.dot(prev_u, prev_u) > 1e-10:
                projection = np.dot(u, prev_u) / np.dot(prev_u, prev_u)
                u = u - projection * prev_u
        
        # Normalize the vector
        norm = np.linalg.norm(u)
        if norm > 1e-10:  # Avoid division by zero
            u = u / norm
        
        orthogonal_vectors.append(u)
    
    return orthogonal_vectors


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def compute_orthogonality_metrics(vectors, labels):
    """
    Compute orthogonality metrics for a set of vectors.
    
    Args:
        vectors: List of numpy arrays
        labels: List of labels for the vectors (V, A, D)
        
    Returns:
        Dictionary containing orthogonality metrics
    """
    metrics = {}
    
    for i, (v1, label1) in enumerate(zip(vectors, labels)):
        for j, (v2, label2) in enumerate(zip(vectors, labels)):
            if i < j:  # Only compute upper triangle
                cos_sim = cosine_similarity(v1, v2)
                angle_rad = np.arccos(np.clip(cos_sim, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)
                
                key = f"{label1}_{label2}"
                metrics[key] = {
                    'cosine_similarity': cos_sim,
                    'angle_deg': angle_deg
                }
    
    return metrics


def process_language(language, results_dir):
    """
    Process all models for a given language.
    
    Args:
        language: Language name ('arabic' or 'english')
        results_dir: Path to results directory
        
    Returns:
        Tuple of (orthogonalized_df, metrics_df)
    """
    lang_dir = results_dir / language
    input_file = lang_dir / 'direction_vectors.csv'
    
    if not input_file.exists():
        print(f"  Warning: Direction vectors not found: {input_file}")
        return None, None
    
    # Read direction vectors
    df = pd.read_csv(input_file)
    
    # Get unique models
    models = df['model'].unique()
    print(f"  Found {len(models)} models")
    
    # Prepare output data
    output_rows = []
    metrics_rows = []
    
    # Process each model
    for model in models:
        model_df = df[df['model'] == model].copy()
        
        # Get embedding dimension for this model
        embedding_dim = int(model_df['embedding_dim'].iloc[0])
        dim_cols = [f'dim_{i}' for i in range(embedding_dim)]
        
        # Extract vectors in order: V, A, D
        vectors = []
        for dim in DIMENSION_ORDER:
            row = model_df[model_df['dimension'] == dim]
            if len(row) == 0:
                print(f"    Warning: {dim} not found for {model}")
                continue
            
            # Extract only the valid dimension columns
            valid_cols = [c for c in dim_cols if c in row.columns]
            vector = row[valid_cols].values[0]
            vectors.append(vector)
        
        if len(vectors) != 3:
            print(f"    Error: Expected 3 vectors, found {len(vectors)}. Skipping {model}.")
            continue
        
        # Compute original orthogonality metrics
        orig_metrics = compute_orthogonality_metrics(vectors, DIMENSION_ORDER)
        
        # Apply Gram-Schmidt orthogonalization
        orthogonal_vectors = gram_schmidt(vectors)
        
        # Compute orthogonalized metrics
        ortho_metrics = compute_orthogonality_metrics(orthogonal_vectors, DIMENSION_ORDER)
        
        # Calculate improvement
        orig_mean_cos = np.mean([abs(m['cosine_similarity']) for m in orig_metrics.values()])
        ortho_mean_cos = np.mean([abs(m['cosine_similarity']) for m in ortho_metrics.values()])
        
        print(f"    {model}: mean|cos| {orig_mean_cos:.4f} -> {ortho_mean_cos:.6f}")
        
        # Save orthogonalized vectors
        for dim, ortho_vec in zip(DIMENSION_ORDER, orthogonal_vectors):
            row_data = {
                'model': model,
                'dimension': dim,
                'embedding_dim': embedding_dim
            }
            for i, val in enumerate(ortho_vec):
                row_data[f'dim_{i}'] = val
            output_rows.append(row_data)
        
        # Save metrics (before and after)
        for metric_type, metrics in [('original', orig_metrics), ('orthogonalized', ortho_metrics)]:
            metric_row = {
                'model': model,
                'type': metric_type,
            }
            for key, vals in metrics.items():
                metric_row[f'cos_{key}'] = vals['cosine_similarity']
                metric_row[f'angle_{key}'] = vals['angle_deg']
            
            # Add summary metrics
            metric_row['mean_abs_cos'] = np.mean([abs(m['cosine_similarity']) for m in metrics.values()])
            metric_row['orthogonality_score'] = 1 - metric_row['mean_abs_cos']
            metrics_rows.append(metric_row)
    
    # Create output DataFrames
    output_df = pd.DataFrame(output_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    
    return output_df, metrics_df


def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results"
    
    print("=" * 60)
    print("Gram-Schmidt Orthogonalization for VAD Direction Vectors")
    print("=" * 60)
    print(f"\nDimension order (first is preserved): {' -> '.join(DIMENSION_ORDER)}")
    
    # Process each language
    for language in languages:
        print(f"\n{'-' * 60}")
        print(f"Processing {language.upper()}")
        print(f"{'-' * 60}")
        
        output_df, metrics_df = process_language(language, results_dir)
        
        if output_df is None:
            continue
        
        lang_dir = results_dir / language
        
        # Save orthogonalized vectors
        output_file = lang_dir / "orthogonalized_direction_vectors.csv"
        output_df.to_csv(output_file, index=False)
        print(f"\n  Saved orthogonalized vectors: {output_file}")
        
        # Save metrics
        metrics_file = lang_dir / "orthogonalization_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"  Saved metrics: {metrics_file}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    
    for language in languages:
        lang_dir = results_dir / language
        metrics_file = lang_dir / "orthogonalization_metrics.csv"
        
        if not metrics_file.exists():
            continue
        
        metrics_df = pd.read_csv(metrics_file)
        
        # Compare original vs orthogonalized
        orig = metrics_df[metrics_df['type'] == 'original']
        ortho = metrics_df[metrics_df['type'] == 'orthogonalized']
        
        print(f"\n{language.upper()}:")
        print(f"  Original mean|cos|:        {orig['mean_abs_cos'].mean():.6f}")
        print(f"  Orthogonalized mean|cos|:  {ortho['mean_abs_cos'].mean():.6f}")
        improvement = (1 - ortho['mean_abs_cos'].mean() / orig['mean_abs_cos'].mean()) * 100
        print(f"  Improvement:               {improvement:.2f}%")
    
    print(f"\n{'=' * 60}")
    print("Orthogonalization complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
