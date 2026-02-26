#!/usr/bin/env python3
"""
Figure 2 v2: The "Rich Club" Null Envelope (Log-Log Scale)

This figure visualizes the separation between empirical values and null distributions
for key network metrics to refute "data mining" critiques.

Plot Type: 2x2 Subplot with Distribution Histograms (Log-Log Scale)
Each subplot shows:
- Histogram (Grey): Distribution of 1000 Calibrated Null results
- Histogram (Blue): Distribution of 1000 Insider Shuffle results
- Vertical Line (Red): The Empirical Value
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set matplotlib to use LaTeX-style fonts with larger sizes
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
    'font.size': 18,  # Increased base font size
    'axes.labelsize': 20,  # Larger axis labels
    'axes.titlesize': 20,
    'xtick.labelsize': 18,  # Larger tick labels
    'ytick.labelsize': 18,
    'legend.fontsize': 16,  # Larger legend
    'figure.titlesize': 22,
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

def create_figure2_v2():
    """Create the Rich Club Null Envelope figure with 2x2 subplots (log-log scale)."""
    cal_null, shuffle_null, real_metrics = load_data()
    
    # Define metrics to plot
    metrics = [
        ('num_nodes', 'Number of Nodes', 'a'),
        ('num_edges', 'Number of Edges', 'b'),
        ('num_components', 'Number of Connected Components', 'c'),
        ('ultra_strong_ties', 'Ultra Strong Ties', 'd')
    ]
    
    # Create 2x2 subplot figure with more spacing
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (metric, metric_label, subplot_label) in enumerate(metrics):
        ax = axes[idx]
        
        # Extract values
        cal_values = cal_null[metric].values
        shuffle_values = shuffle_null[metric].values
        real_value = real_metrics[metric].iloc[0]
        
        # For log scale, ensure all values are positive (add 1 to handle zeros)
        # This is standard practice for log scale with potential zeros
        epsilon = 1.0
        cal_values_plot = np.maximum(cal_values, 0) + epsilon
        shuffle_values_plot = np.maximum(shuffle_values, 0) + epsilon
        real_value_plot = max(real_value, 0) + epsilon
        
        # Create bins for log scale
        all_values_plot = np.concatenate([cal_values_plot, shuffle_values_plot])
        min_val = max(epsilon, np.min(all_values_plot) * 0.9)
        max_val = np.max(all_values_plot) * 1.1
        bins_log = np.logspace(np.log10(min_val), np.log10(max_val), 30)
        
        # Plot histograms with log scale
        ax.hist(cal_values_plot, bins=bins_log, alpha=0.7, color='gray', edgecolor='black', 
                linewidth=0.5, label=f'Calibrated Null (μ={np.mean(cal_values):.1f})',
                density=False)
        ax.hist(shuffle_values_plot, bins=bins_log, alpha=0.7, color='#1f77b4', edgecolor='black', 
                linewidth=0.5, label=f'Insider Shuffle (μ={np.mean(shuffle_values):.1f})',
                density=False)
        
        # Add vertical line for empirical value
        ax.axvline(real_value_plot, color='red', linestyle='--', linewidth=3, 
                   label=f'Empirical ({real_value:,.0f})', zorder=10)
        
        # Add subplot label (a, b, c, d) - positioned to avoid overlap
        ax.text(0.02, 0.98, f'({subplot_label})', transform=ax.transAxes,
                fontsize=20, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.5))
        
        # Set log-log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Customize axes with larger fonts
        ax.set_xlabel(metric_label, fontweight='bold', fontsize=20)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=20)
        
        # Legend with larger font and better positioning to avoid overlap
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                          fontsize=16, framealpha=0.95)
        legend.get_frame().set_linewidth(1.5)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', axis='both', which='both')
        
        # Format x-axis for log scale - show actual values (subtract epsilon from displayed value)
        def format_log_label(x, pos):
            val = x - epsilon
            if val >= 1000:
                return f'{val:,.0f}'
            elif val >= 1:
                return f'{val:.0f}'
            elif val >= 0.1:
                return f'{val:.1f}'
            else:
                return f'{val:.2f}'
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_log_label))
        
        # Adjust y-axis to avoid overlap
        ax.tick_params(axis='both', which='major', labelsize=18, pad=8)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        
        # Add break indicator if empirical value is far from distributions
        max_null = max(max(cal_values_plot), max(shuffle_values_plot))
        if real_value_plot > max_null * 2:
            # Add break indicator
            break_pos = max_null * 1.2
            ylim = ax.get_ylim()
            ax.text(break_pos, ylim[0] * 1.5, '//', ha='center', va='bottom',
                   fontsize=26, color='black', fontweight='bold')
    
    # Increase spacing between subplots to avoid overlap
    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)
    
    # Save as EPS
    output_path = Path(__file__).parent / "figure2_rich_club_null_envelope_v2.eps"
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"Figure 2 v2 saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    create_figure2_v2()

