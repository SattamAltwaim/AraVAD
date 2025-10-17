"""
Gram-Schmidt Orthogonalization for VAD Direction Vectors

This script reads the orthogonality analysis results, extracts the VAD direction vectors 
for each model, applies Gram-Schmidt orthogonalization to remove leakage between dimensions,
and saves the orthogonalized vectors to a new CSV file.

The Gram-Schmidt process ensures that the Valence, Arousal, and Dominance vectors are 
mutually orthogonal while preserving the direction of the first vector (Valence).
"""

import pandas as pd
import numpy as np
from pathlib import Path


def gram_schmidt(vectors):
    """
    Apply Gram-Schmidt orthogonalization to a list of vectors.
    
    Args:
        vectors: List of numpy arrays representing vectors to orthogonalize
        
    Returns:
        List of orthogonalized numpy arrays
    """
    orthogonal_vectors = []
    
    for i, v in enumerate(vectors):
        # Start with the current vector
        u = v.copy()
        
        # Subtract projections onto all previous orthogonal vectors
        for prev_u in orthogonal_vectors:
            projection = np.dot(u, prev_u) / np.dot(prev_u, prev_u)
            u = u - projection * prev_u
        
        # Normalize the vector
        norm = np.linalg.norm(u)
        if norm > 1e-10:  # Avoid division by zero
            u = u / norm
        
        orthogonal_vectors.append(u)
    
    return orthogonal_vectors


def compute_orthogonality_metrics(vectors, labels):
    """
    Compute orthogonality metrics for a set of vectors.
    
    Args:
        vectors: List of numpy arrays
        labels: List of labels for the vectors
        
    Returns:
        Dictionary containing orthogonality metrics
    """
    metrics = {}
    
    for i, (v1, label1) in enumerate(zip(vectors, labels)):
        for j, (v2, label2) in enumerate(zip(vectors, labels)):
            if i < j:  # Only compute upper triangle
                dot_product = np.dot(v1, v2)
                angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)
                
                key = f"{label1}_{label2}"
                metrics[key] = {
                    'dot_product': dot_product,
                    'angle_deg': angle_deg
                }
    
    return metrics


def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "Datasets" / "orthogonality_analysis_results.csv"
    output_file = base_dir / "Datasets" / "orthogonalized_vad_vectors.csv"
    
    print(f"Reading data from: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter to only direction vectors
    direction_df = df[df['vector_type'] == 'direction'].copy()
    
    # Get unique models
    models = direction_df['model'].unique()
    print(f"Found models: {', '.join(models)}")
    
    # Prepare output data
    output_rows = []
    metrics_rows = []
    
    # Process each model
    for model in models:
        print(f"\nProcessing {model}...")
        
        # Extract direction vectors for this model
        model_df = direction_df[direction_df['model'] == model].copy()
        
        # Get the three direction vectors in order: V, A, D
        vector_order = ['dir_V', 'dir_A', 'dir_D']
        vectors = []
        
        for vec_name in vector_order:
            row = model_df[model_df['vector_name'] == vec_name]
            if len(row) == 0:
                print(f"  Warning: {vec_name} not found for {model}")
                continue
            
            # Extract vector dimensions (all columns starting with 'dim_')
            dim_cols = [col for col in row.columns if col.startswith('dim_')]
            vector = row[dim_cols].values[0]
            vectors.append(vector)
        
        if len(vectors) != 3:
            print(f"  Error: Expected 3 vectors, found {len(vectors)}. Skipping {model}.")
            continue
        
        # Compute original orthogonality metrics
        print("  Original orthogonality:")
        orig_metrics = compute_orthogonality_metrics(vectors, ['V', 'A', 'D'])
        for key, vals in orig_metrics.items():
            print(f"    {key}: dot={vals['dot_product']:.6f}, angle={vals['angle_deg']:.2f}°")
        
        # Apply Gram-Schmidt orthogonalization
        orthogonal_vectors = gram_schmidt(vectors)
        
        # Compute orthogonalized metrics
        print("  After Gram-Schmidt orthogonalization:")
        ortho_metrics = compute_orthogonality_metrics(orthogonal_vectors, ['V', 'A', 'D'])
        for key, vals in ortho_metrics.items():
            print(f"    {key}: dot={vals['dot_product']:.6f}, angle={vals['angle_deg']:.2f}°")
        
        # Save orthogonalized vectors
        for vec_name, ortho_vec in zip(vector_order, orthogonal_vectors):
            row_data = {
                'model': model,
                'vector_name': vec_name,
            }
            # Add vector dimensions
            for i, val in enumerate(ortho_vec):
                row_data[f'dim_{i}'] = val
            
            output_rows.append(row_data)
        
        # Save metrics
        for metric_type, prefix in [('original', 'orig'), ('orthogonalized', 'ortho')]:
            metrics = orig_metrics if metric_type == 'original' else ortho_metrics
            metric_row = {
                'model': model,
                'metric_type': metric_type,
            }
            for key, vals in metrics.items():
                metric_row[f'{key}_dot'] = vals['dot_product']
                metric_row[f'{key}_angle'] = vals['angle_deg']
            metrics_rows.append(metric_row)
    
    # Create output DataFrames
    output_df = pd.DataFrame(output_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    
    # Save orthogonalized vectors
    print(f"\nSaving orthogonalized vectors to: {output_file}")
    output_df.to_csv(output_file, index=False)
    
    # Save metrics
    metrics_file = base_dir / "Datasets" / "orthogonalization_metrics.csv"
    print(f"Saving metrics to: {metrics_file}")
    metrics_df.to_csv(metrics_file, index=False)
    
    print("\n✓ Orthogonalization complete!")
    print(f"  - Orthogonalized vectors: {output_file}")
    print(f"  - Metrics: {metrics_file}")


if __name__ == "__main__":
    main()

