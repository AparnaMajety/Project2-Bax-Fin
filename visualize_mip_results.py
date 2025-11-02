"""
Visualize MIP Optimization Results
Loads results from JSON and creates visualization plots.

Usage:
    python visualize_mip_results.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = "mip_results.json"

def load_results():
    """Load results from JSON file."""
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    return data

def create_visualizations():
    """Create all visualization plots."""
    
    # Load data
    data = load_results()
    results = data['results']
    
    # Convert to DataFrame for easier manipulation
    results_list = []
    for key, value in results.items():
        if isinstance(value, dict) and 'm' in value:
            results_list.append({
                'm': value['m'],
                'L1_2024': value.get('L1_2024'),
                'L1_2025': value.get('L1_2025'),
                'status': value.get('status'),
                'solve_time': value.get('solve_time')
            })
    
    df = pd.DataFrame(results_list).sort_values('m')
    
    # Filter out None values
    df = df[df['L1_2024'].notna()]
    
    if len(df) == 0:
        print("No results to visualize!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MIP Optimization Results', fontsize=16, fontweight='bold')
    
    # Plot 1: L1 Tracking Error vs m
    ax1 = axes[0, 0]
    ax1.plot(df['m'], df['L1_2024'], marker='o', label='In-sample (2024)', linewidth=2)
    ax1.plot(df['m'], df['L1_2025'], marker='s', label='Out-of-sample (2025)', linewidth=2)
    ax1.set_xlabel('Number of Stocks (m)')
    ax1.set_ylabel('L1 Tracking Error')
    ax1.set_title('Tracking Error vs Number of Stocks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Solve Time vs m
    ax2 = axes[0, 1]
    if df['solve_time'].notna().any():
        ax2.bar(df['m'], df['solve_time'], alpha=0.7, color='steelblue')
        ax2.set_xlabel('Number of Stocks (m)')
        ax2.set_ylabel('Solve Time (seconds)')
        ax2.set_title('Optimization Time vs Number of Stocks')
        ax2.grid(True, alpha=0.3, axis='y')
        # Add hour markers
        max_time = df['solve_time'].max()
        if max_time > 3600:
            ax2.axhline(y=3600, color='r', linestyle='--', alpha=0.5, label='1 hour')
            ax2.legend()
    
    # Plot 3: Improvement (Out-of-sample vs In-sample)
    ax3 = axes[1, 0]
    if df['L1_2024'].notna().any() and df['L1_2025'].notna().any():
        improvement = df['L1_2024'] - df['L1_2025']
        colors = ['green' if x > 0 else 'red' for x in improvement]
        ax3.bar(df['m'], improvement, alpha=0.7, color=colors)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Number of Stocks (m)')
        ax3.set_ylabel('L1 Improvement (2024 - 2025)')
        ax3.set_title('Out-of-Sample vs In-Sample Performance')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Status Summary
    ax4 = axes[1, 1]
    status_counts = df['status'].value_counts()
    if len(status_counts) > 0:
        ax4.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Optimization Status Distribution')
    
    plt.tight_layout()
    plt.savefig('mip_results_visualization.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: mip_results_visualization.png")
    plt.show()
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY TABLE")
    print("="*70)
    
    summary = df[['m', 'L1_2024', 'L1_2025', 'status', 'solve_time']].copy()
    summary.columns = ['m', 'In-Sample L1 (2024)', 'Out-of-Sample L1 (2025)', 'Status', 'Time (s)']
    summary['Time (s)'] = summary['Time (s)'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    summary['In-Sample L1 (2024)'] = summary['In-Sample L1 (2024)'].apply(
        lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
    )
    summary['Out-of-Sample L1 (2025)'] = summary['Out-of-Sample L1 (2025)'].apply(
        lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
    )
    
    print(summary.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    try:
        create_visualizations()
    except FileNotFoundError:
        print(f"Error: {RESULTS_FILE} not found!")
        print("Run run_mip_optimizations.py first to generate results.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

