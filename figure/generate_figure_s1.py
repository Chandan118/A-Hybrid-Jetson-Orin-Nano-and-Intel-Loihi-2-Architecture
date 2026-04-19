#!/usr/bin/env python3
"""
Figure S1. Neuromorphic Acceleration: Throughput and Energy Efficiency.
Bar charts comparing Jetson-only (GPU) vs Jetson+Loihi 2 (Hybrid) performance.
Shows +69.6% throughput improvement and -23.6% energy reduction per frame.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from hybrid_comparison_results.json
categories = ['Throughput\n(FPS)', 'Energy Efficiency\n(inverse J/frame)']
jetson_only = [49.73, 60.61]  # Standardized metrics
hybrid = [84.34, 79.32]       # Neuromorphic hybrid

# Calculate improvements
fps_improvement = ((84.34 - 49.73) / 49.73) * 100
energy_improvement = ((79.32 - 60.61) / 60.61) * 100

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Color palette
colors = ['#3498db', '#2ecc71']
jetson_color = '#e74c3c'
hybrid_color = '#27ae60'

# ========== Throughput (FPS) ==========
x = np.arange(1)
width = 0.35

bars1 = ax1.bar(x - width/2, [jetson_only[0]], width, label='Jetson Only', color=jetson_color, edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x + width/2, [hybrid[0]], width, label='Jetson + Loihi 2', color=hybrid_color, edgecolor='black', linewidth=1.2)

ax1.set_ylabel('Frames Per Second (FPS)', fontsize=12, fontweight='bold')
ax1.set_title('Throughput Comparison', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(['FPS'])
ax1.legend(loc='upper left', fontsize=10)
ax1.set_ylim(0, 100)

# Add value labels
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{jetson_only[0]:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{hybrid[0]:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement annotation
ax1.annotate(f'+{fps_improvement:.1f}%', xy=(0, hybrid[0]), xytext=(0.3, hybrid[0] + 8),
             fontsize=14, fontweight='bold', color='#27ae60',
             arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

ax1.grid(axis='y', alpha=0.3, linestyle='-')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ========== Energy Efficiency ==========
bars3 = ax2.bar(x - width/2, [jetson_only[1]], width, label='Jetson Only', color=jetson_color, edgecolor='black', linewidth=1.2)
bars4 = ax2.bar(x + width/2, [hybrid[1]], width, label='Jetson + Loihi 2', color=hybrid_color, edgecolor='black', linewidth=1.2)

ax2.set_ylabel('Efficiency Score (higher = better)', fontsize=12, fontweight='bold')
ax2.set_title('Energy Efficiency Comparison', fontsize=13, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(['Efficiency'])
ax2.legend(loc='upper left', fontsize=10)
ax2.set_ylim(0, 100)

# Add value labels
for bar in bars3:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{jetson_only[1]:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
for bar in bars4:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{hybrid[1]:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement annotation
ax2.annotate(f'+{energy_improvement:.1f}%', xy=(0, hybrid[1]), xytext=(0.3, hybrid[1] + 8),
             fontsize=14, fontweight='bold', color='#27ae60',
             arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

ax2.grid(axis='y', alpha=0.3, linestyle='-')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Main title
fig.suptitle('Figure S1. Neuromorphic Acceleration: Throughput and Energy Efficiency', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save outputs
output_path = '/Users/chandansheikder/Documents/Bio-Inspired Paper/Manuscript/Bio/Trends in Biotechnology/jetson_slam_experiments/figure/S1'

fig.savefig(f'{output_path}.pdf', format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{output_path}.png', format='png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{output_path}.svg', format='svg', bbox_inches='tight', facecolor='white')

print("Figure S1 generated successfully!")
plt.close()
