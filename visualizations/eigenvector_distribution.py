#!/usr/bin/env python3
"""
Generate distribution of eigenvector centrality plot in EPS format.

This script recreates the plot from the Jupyter notebook analysis,
using the same styling and format.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set matplotlib to use LaTeX-style fonts (matching other visualization scripts)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
})

def main():
    # Paths
    root = Path(__file__).resolve().parents[1]
    centrality_file = root / 'network_results' / 'centrality_rankings.csv'
    output_file = root / 'visualizations' / 'eigenvector_distribution.eps'
    
    # Load centrality rankings
    print(f"Loading centrality rankings from: {centrality_file}")
    df = pd.read_csv(centrality_file)
    
    # Filter out zero eigenvector centrality values
    filtered = df[df['eigenvector'] > 0].copy()
    
    if len(filtered) == 0:
        print("Warning: No non-zero eigenvector centrality values found!")
        return
    
    # Set up the figure (matching notebook: figsize=(6.5, 4.5))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    # Plot histogram and KDE using log-transformed x-values
    log_values = np.log10(filtered['eigenvector'])
    
    sns.histplot(
        log_values,
        bins=30,
        kde=True,
        stat="density",
        color="skyblue",
        edgecolor="black",
        linewidth=0.6,
        ax=ax
    )
    
    # Set axes and titles (matching notebook font sizes, but scaled up for publication)
    ax.set_xlabel("log10(Eigenvector Centrality Score)", fontweight='bold', fontsize=22)
    ax.set_ylabel("Density", fontweight='bold', fontsize=22)
    ax.set_title("Distribution of Eigenvector Centrality (log scale)", fontsize=22, fontweight='bold')
    
    # Clean styling (matching notebook)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    
    # Final layout
    plt.tight_layout()
    
    # Save as EPS
    print(f"Saving figure to: {output_file}")
    plt.savefig(output_file, format='eps', dpi=300, bbox_inches='tight')
    print(f"Figure saved successfully!")
    
    # Print summary
    print(f"\nSummary statistics:")
    print(f"  Total nodes: {len(df)}")
    print(f"  Nodes with eigenvector > 0: {len(filtered)}")
    print(f"  Mean log10(eigenvector): {log_values.mean():.6f}")
    print(f"  Median log10(eigenvector): {log_values.median():.6f}")
    print(f"  Std log10(eigenvector): {log_values.std():.6f}")

if __name__ == '__main__':
    main()

