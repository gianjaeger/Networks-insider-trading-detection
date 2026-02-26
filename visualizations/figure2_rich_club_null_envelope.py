#!/usr/bin/env python3
"""
Figure 2: The "Rich Club" Null Envelope

This figure visualizes the separation between empirical values and null distributions
for key network metrics to refute "data mining" critiques.

Plot Type: 2x2 Subplot with Distribution Histograms
Each subplot shows:
- Histogram (Grey): Distribution of 100 Calibrated Null results
- Histogram (Blue): Distribution of 100 Insider Shuffle results
- Vertical Line (Red): The Empirical Value
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set matplotlib to use LaTeX-style fonts
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
    'font.size': 20,  # Base font size increased
    'axes.labelsize': 22,  # Bigger axis labels
    'axes.titlesize': 22,
    'xtick.labelsize': 20,  # Bigger tick labels
    'ytick.labelsize': 20,
    'legend.fontsize': 18,  # Bigger legend
    'figure.titlesize': 24,  # Bigger figure title
})

def load_data():
    """Load null model results and real network metrics."""
    root = Path(__file__).resolve().parents[1]
    
    # Load null model results
    calibrated_null = pd.read_csv(
        root / "network_results" / "null_analysis_full" / "calibrated_null_v2" / "metrics_null.csv"
    )
    shuffle_null = pd.read_csv(
        root / "network_results" / "null_analysis_full" / "insider_shuffle_null_Q" / "metrics_null.csv"
    )
    
    # Load real metrics
    real_metrics = pd.read_csv(
        root / "network_results" / "null_analysis_full" / "metrics_real.csv"
    )
    
    return calibrated_null, shuffle_null, real_metrics

def create_figure2():
    """Create the Rich Club Null Envelope figure with 2x2 subplots."""
    cal_null, shuffle_null, real_metrics = load_data()
    
    # Define metrics to plot
    metrics = [
        ('num_nodes', 'Number of Nodes', 'a'),
        ('num_edges', 'Number of Edges', 'b'),
        ('num_components', 'Number of Connected Components', 'c'),
        ('ultra_strong_ties', 'Ultra Strong Ties', 'd')
    ]
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, metric_label, subplot_label) in enumerate(metrics):
        ax = axes[idx]
        
        # Extract values
        cal_values = cal_null[metric].values
        shuffle_values = shuffle_null[metric].values
        real_value = real_metrics[metric].iloc[0]
        
        # Create bins
        all_values = np.concatenate([cal_values, shuffle_values])
        bins = np.linspace(0, max(all_values) * 1.1, 30)
        
        # Create histograms
        ax.hist(cal_values, bins=bins, alpha=0.7, color='gray', edgecolor='black', 
                linewidth=0.5, label=f'Calibrated Null (μ={np.mean(cal_values):.1f})',
                density=False)
        ax.hist(shuffle_values, bins=bins, alpha=0.7, color='#1f77b4', edgecolor='black', 
                linewidth=0.5, label=f'Insider Shuffle (μ={np.mean(shuffle_values):.1f})',
                density=False)
        
        # Add vertical line for empirical value
        ax.axvline(real_value, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Empirical ({real_value:,.0f})', zorder=10)
        
        # Add subplot label (a, b, c, d)
        ax.text(0.02, 0.98, f'({subplot_label})', transform=ax.transAxes,
                fontsize=22, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))
        
        # Customize axes
        ax.set_xlabel(metric_label, fontweight='bold', fontsize=22)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=22)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=18)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Format x-axis for large numbers
        if real_value > 1000:
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
            # Or use comma formatting
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add break indicator if empirical value is far from distributions
        max_null = max(max(cal_values), max(shuffle_values))
        if real_value > max_null * 2:
            # Add break indicator
            xlim = ax.get_xlim()
            break_pos = max_null * 1.15
            ax.text(break_pos, -ax.get_ylim()[1] * 0.05, '//', ha='center', va='top',
                   fontsize=26, color='black', fontweight='bold')
    
    plt.tight_layout()
    
    # Save as EPS
    output_path = Path(__file__).parent / "figure2_rich_club_null_envelope.eps"
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    create_figure2()
