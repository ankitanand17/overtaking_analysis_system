"""
plot_distance_errors.py

This script processes distance calibration data and generates a professional double-pane 
graphical report validating estimated sensor distances against true physical distances.
It plots distance estimation accuracy on the left and absolute/relative error analytics 
on the right.
"""

import os
import matplotlib
# Use 'Agg' non-interactive backend for server compatibility (avoids GUI window errors in headless mode)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    Main entry point. Assembles calibration datasets, builds dual-axis visual 
    charts, customize styling parameters (grid, titles, color palette), and saves 
    the result as an image.
    """
    # Style configuration for a clean, professional aesthetic
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create the figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
    
    # Harmonies Color palette
    COLOR_IDEAL = '#64748B'      # Sleek Slate Gray
    COLOR_EST = '#4F46E5'        # Indigo
    COLOR_ABS_ERR = '#0D9488'    # Deep Teal
    COLOR_PCT_ERR = '#E11D48'    # Vibrant Rose/Coral
    
    # Calibration/Evaluation dataset arrays
    actual_dist = np.array([5, 10, 15, 20, 25, 30])
    est_dist = np.array([5.1, 10.2, 14.7, 20.5, 24.4, 30.8])
    abs_err = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.8])
    err_pct = np.array([2.0, 2.0, 2.0, 2.5, 2.4, 2.7])
    
    # --------------------------------------------------------------------------
    # Subplot 1: Actual vs. Estimated Distance
    # --------------------------------------------------------------------------
    # 1. Ideal line y=x reference representation
    ax1.plot([4, 32], [4, 32], color=COLOR_IDEAL, linestyle='--', linewidth=1.5, label='Ideal Reference (y=x)')
    # 2. Estimated distance line
    ax1.plot(actual_dist, est_dist, color=COLOR_EST, marker='o', markersize=8, 
             markeredgecolor='white', markeredgewidth=1.5, linewidth=2.5, label='Estimated Distance')
    
    # Labels and visual styling configs
    ax1.set_title('Distance Estimation Accuracy', fontsize=14, fontweight='bold', pad=15, color='#1E293B')
    ax1.set_xlabel('Actual Distance (m)', fontsize=12, labelpad=8, color='#334155')
    ax1.set_ylabel('Estimated Distance (m)', fontsize=12, labelpad=8, color='#334155')
    ax1.set_xlim(4, 32)
    ax1.set_ylim(4, 32)
    ax1.set_xticks(actual_dist)
    ax1.set_yticks(actual_dist)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax1.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', shadow=False, loc='upper left', fontsize=10)
    
    # Clean spines borders
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    ax1.spines['left'].set_color('#94A3B8')
    ax1.spines['bottom'].set_color('#94A3B8')
    
    # --------------------------------------------------------------------------
    # Subplot 2: Absolute and Percentage Error (Dual Axes representation)
    # --------------------------------------------------------------------------
    # Left axis (Absolute Error)
    ax2.set_xlabel('Actual Distance (m)', fontsize=12, labelpad=8, color='#334155')
    ax2.set_ylabel('Absolute Error (m)', color=COLOR_ABS_ERR, fontsize=12, labelpad=8)
    line1 = ax2.plot(actual_dist, abs_err, color=COLOR_ABS_ERR, marker='s', markersize=8, 
                     markeredgecolor='white', markeredgewidth=1.5, linewidth=2.5, label='Absolute Error (m)')
    ax2.tick_params(axis='y', labelcolor=COLOR_ABS_ERR)
    ax2.set_xticks(actual_dist)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    
    # Right axis (Percentage Error)
    ax2_pct = ax2.twinx()
    ax2_pct.set_ylabel('Relative Error (%)', color=COLOR_PCT_ERR, fontsize=12, labelpad=8)
    line2 = ax2_pct.plot(actual_dist, err_pct, color=COLOR_PCT_ERR, marker='^', markersize=8, 
                         markeredgecolor='white', markeredgewidth=1.5, linewidth=2.5, label='Error (%)')
    ax2_pct.tick_params(axis='y', labelcolor=COLOR_PCT_ERR)
    ax2_pct.set_ylim(0, 4.0)
    
    # Combine legends from both left and right axis systems
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, frameon=True, facecolor='white', edgecolor='#E2E8F0', shadow=False, loc='upper left', fontsize=10)
    
    ax2.set_title('Absolute & Relative Error Analysis', fontsize=14, fontweight='bold', pad=15, color='#1E293B')
    
    # Clean spines borders for second plot
    for spine in ['top']:
        ax2.spines[spine].set_visible(False)
        ax2_pct.spines[spine].set_visible(False)
    ax2.spines['left'].set_color('#94A3B8')
    ax2.spines['bottom'].set_color('#94A3B8')
    ax2_pct.spines['right'].set_color('#94A3B8')
    
    # Main title layout adjustment
    plt.suptitle('Distance Sensor Calibration & Validation Report', fontsize=16, fontweight='bold', y=0.98, color='#0F172A')
    plt.tight_layout()
    
    # Save target output file path
    output_filename = 'distance_calibration_metrics.png'
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SUCCESS: Graph successfully saved to {output_path}")

if __name__ == "__main__":
    main()
