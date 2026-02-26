#!/usr/bin/env python3
"""
Figure 1: The "Forensic Fingerprint" (Comparative Significance Profile)

This figure demonstrates that while the weights of the edges are normal,
the presence of Strong Ties is not.

Plot Type: Grouped Bar Chart
X-Axis: Metrics (Nodes, Edges, Avg Weight, Ultra Strong Ties)
Y-Axis: Log-Scale Normalized Z-Score (SP_i)
Series A (Red Bars): Structural Z-Scores (Calibrated Null)
Series B (Blue Bars): Placebo Z-Scores (Insider Shuffle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set matplotlib to use LaTeX-style fonts
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})

def load_summaries():
    """Load summary data from both null models."""
    root = Path(__file__).resolve().parents[1]
    
    calibrated_summary = pd.read_csv(
        root / "network_results" / "null_analysis_full" / "calibrated_null_v2" / "metrics_summary.csv"
    )
    shuffle_summary = pd.read_csv(
        root / "network_results" / "null_analysis_full" / "insider_shuffle_null_Q" / "metrics_summary.csv"
    )
    
    return calibrated_summary, shuffle_summary

def create_figure1():
    """Create the Forensic Fingerprint figure."""
    cal_summary, shuffle_summary = load_summaries()
    
    # Select metrics of interest
    metrics = ['num_nodes', 'num_edges', 'avg_edge_weight', 'ultra_strong_ties']
    metric_labels = ['Nodes', 'Edges', 'Avg Edge Weight', 'Ultra Strong Ties']
    
    # Extract z-scores
    cal_z_scores = []
    shuffle_z_scores = []
    
    for metric in metrics:
        cal_row = cal_summary[cal_summary['metric'] == metric].iloc[0]
        shuffle_row = shuffle_summary[shuffle_summary['metric'] == metric].iloc[0]
        
        cal_z = abs(cal_row['z_score'])
        shuffle_z = abs(shuffle_row['z_score'])
        
        # Handle inf values (set to a large number for visualization)
        if np.isinf(cal_z):
            cal_z = 1e6
        if np.isinf(shuffle_z):
            shuffle_z = 1e6
        
        cal_z_scores.append(cal_z)
        shuffle_z_scores.append(shuffle_z)
    
    # Normalize to log-scale (add 1 to avoid log(0))
    cal_z_log = np.log10(np.array(cal_z_scores) + 1)
    shuffle_z_log = np.log10(np.array(shuffle_z_scores) + 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, cal_z_log, width, label='Calibrated Null (Structural)', 
                   color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, shuffle_z_log, width, label='Insider Shuffle (Placebo)', 
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize axes
    ax.set_xlabel('Network Metrics', fontweight='bold')
    ax.set_ylabel('Log-Scale Normalized Z-Score ($SP_i$)', fontweight='bold')
    ax.set_title('Forensic Fingerprint: Comparative Significance Profile', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=0, ha='center')
    ax.set_yscale('linear')  # Linear scale for log-transformed values
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add original z-score values as annotations (not log-transformed)
    for i, (bar1, bar2, cal_z, shuffle_z) in enumerate(zip(bars1, bars2, cal_z_scores, shuffle_z_scores)):
        # Format large numbers
        if cal_z >= 1000:
            cal_label = f'{cal_z/1000:.1f}K'
        else:
            cal_label = f'{cal_z:.1f}'
        
        if shuffle_z >= 1000:
            shuffle_label = f'{shuffle_z/1000:.1f}K'
        else:
            shuffle_label = f'{shuffle_z:.1f}'
        
        ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.05,
               f'Z={cal_label}', ha='center', va='bottom', fontsize=8, 
               color='#d62728', fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.05,
               f'Z={shuffle_label}', ha='center', va='bottom', fontsize=8,
               color='#1f77b4', fontweight='bold')
    
    plt.tight_layout()
    
    # Save as EPS
    output_path = Path(__file__).parent / "figure1_forensic_fingerprint.eps"
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')
    print(f"Figure 1 saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    create_figure1()



