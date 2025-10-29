"""
Generate Architecture Diagram as PNG/PDF
Creates a publication-quality diagram of the hybrid BiLSTM-Transformer architecture.
*** MODIFIED for better layout and spacing ***
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os # Added to manage save path

# Set up the figure with high DPI for publication quality
fig, ax = plt.subplots(1, 1, figsize=(16, 13), dpi=300) # Increased height slightly
ax.set_xlim(0, 20)
ax.set_ylim(-1, 25) # Adjusted Y limits for more space
ax.axis('off')

# Define colors (matching TikZ version)
colors = {
    'input': '#E6F5FF',      # Light blue
    'embed': '#FFF0E6',      # Light orange
    'lstm': '#E6FFE6',       # Light green
    'attention': '#F0E6FF',  # Light purple
    'transformer': '#FFFACD', # Light yellow
    'output': '#FFE6E6',     # Light red
    'technical': '#FFE6F0',  # Light pink
    'edge': '#333333'        # Dark gray for borders
}

def draw_box(ax, x, y, width, height, text, color, fontsize=10, bold=True):
    """Draw a colored box with text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1",
        edgecolor=colors['edge'],
        facecolor=color,
        linewidth=1.5 # Slightly thinner lines
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, wrap=True)

    return (x, y)

def draw_arrow(ax, x1, y1, x2, y2, style='->', width=2, label='', bend=0):
    """Draw an arrow between two points, optionally curved"""
    connectionstyle = f"arc3,rad={bend}" if bend != 0 else "arc3"
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=colors['edge'],
        linewidth=width,
        mutation_scale=20,
        connectionstyle=connectionstyle,
        zorder=1
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        # Adjust label position slightly based on bend
        offset_x = 0.5 + abs(bend)*2
        offset_y = 0.1
        ax.text(mid_x + offset_x, mid_y + offset_y, label, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Title
ax.text(10, 24, 'Hybrid Stock Prediction Architecture', # Moved title higher
        ha='center', va='top', fontsize=18, weight='bold')

# Vertical spacing constant
V_SPACE = 2.4 # Increased spacing

# ==================== LEFT BRANCH: TEXT PROCESSING ====================
branch_x_l = 5
y_start_l = 22 # Start higher
ax.text(branch_x_l, y_start_l + 1.0, 'Text Processing Branch', # Adjusted title position
        ha='center', va='center', fontsize=13, weight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=0.3))

# Input: News Articles
y = y_start_l
pos1 = draw_box(ax, branch_x_l, y, 4, 1.1, # Slightly wider box
                'Financial News\n(60-day sequence)',
                colors['input'], fontsize=9)

# FinBERT Embeddings
y -= V_SPACE
pos2 = draw_box(ax, branch_x_l, y, 4, 1.1,
                'FinBERT Embeddings\n768 dimensions',
                colors['embed'], fontsize=9)
draw_arrow(ax, pos1[0], pos1[1]-0.55, pos2[0], pos2[1]+0.55, width=2.5)
ax.text(branch_x_l + 2.2, y, '768d', fontsize=8, style='italic')

# Sentiment Features
y -= V_SPACE
pos3 = draw_box(ax, branch_x_l, y, 4, 1.1,
                'Sentiment Extraction\n18 features',
                colors['embed'], fontsize=9)
draw_arrow(ax, pos2[0], pos2[1]-0.55, pos3[0], pos3[1]+0.55, width=2.5)
ax.text(branch_x_l + 2.2, y, '18d', fontsize=8, style='italic')

# Concatenate
y -= V_SPACE * 0.8 # Slightly less space before concat
pos4 = draw_box(ax, branch_x_l, y, 4, 0.9,
                'Concatenate: 768 + 18 = 786',
                colors['input'], fontsize=8)
draw_arrow(ax, pos3[0], pos3[1]-0.55, pos4[0], pos4[1]+0.45, width=2.5)

# BiLSTM
y -= V_SPACE * 1.1 # More space after concat
pos5 = draw_box(ax, branch_x_l, y, 4, 1.3, # Taller box for more text
                'BiLSTM (3 layers)\n512 hidden × 2 directions\n1024 dimensions',
                colors['lstm'], fontsize=9)
draw_arrow(ax, pos4[0], pos4[1]-0.45, pos5[0], pos5[1]+0.65, width=2.5)
ax.text(branch_x_l + 2.2, y, '1024d', fontsize=8, style='italic')

# Multi-Head Attention
y -= V_SPACE
pos6 = draw_box(ax, branch_x_l, y, 4, 1.1,
                'Multi-Head Attention\n16 heads',
                colors['attention'], fontsize=9)
draw_arrow(ax, pos5[0], pos5[1]-0.65, pos6[0], pos6[1]+0.55, width=2.5)

# Transformer Encoder
y -= V_SPACE
pos7 = draw_box(ax, branch_x_l, y, 4, 1.1,
                'Transformer Encoder\n4 layers',
                colors['transformer'], fontsize=9)
draw_arrow(ax, pos6[0], pos6[1]-0.55, pos7[0], pos7[1]+0.55, width=2.5)

# Text Features
y -= V_SPACE
pos8 = draw_box(ax, branch_x_l, y, 4, 1.1,
                'Text Features\n512 dimensions',
                colors['lstm'], fontsize=9)
draw_arrow(ax, pos7[0], pos7[1]-0.55, pos8[0], pos8[1]+0.55, width=2.5)
ax.text(branch_x_l + 2.2, y, '512d', fontsize=8, style='italic')
text_branch_end_y = pos8[1] - 0.55 # Store y coordinate for arrow

# ==================== RIGHT BRANCH: NUMERICAL PROCESSING ====================
branch_x_r = 15
y_start_r = y_start_l # Align vertically
ax.text(branch_x_r, y_start_r + 1.0, 'Numerical Processing Branch', # Adjusted title position
        ha='center', va='center', fontsize=13, weight='bold',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3, pad=0.3))

# Input: Stock Data
y = y_start_r
pos1_r = draw_box(ax, branch_x_r, y, 4, 1.1,
                  'Price & Context Data\n(60-day sequence)',
                  colors['input'], fontsize=9)

# Feature Engineering
y -= V_SPACE
pos2_r = draw_box(ax, branch_x_r, y, 4, 1.1,
                  'Feature Engineering\n(83 features)',
                  colors['technical'], fontsize=9)
draw_arrow(ax, pos1_r[0], pos1_r[1]-0.55, pos2_r[0], pos2_r[1]+0.55, width=2.5)
ax.text(branch_x_r - 2.2, y, '83d', fontsize=8, style='italic')

# Normalization
y -= V_SPACE
pos5_r = draw_box(ax, branch_x_r, y, 4, 1.1,
                  'Normalization\n(Standard Scaler)',
                  colors['technical'], fontsize=9)
draw_arrow(ax, pos2_r[0], pos2_r[1]-0.55, pos5_r[0], pos5_r[1]+0.55, width=2.5)

# Projection Layer
y -= V_SPACE
pos6_r = draw_box(ax, branch_x_r, y, 4, 1.1,
                  'Projection Layer\n83 → 256',
                  colors['technical'], fontsize=9)
draw_arrow(ax, pos5_r[0], pos5_r[1]-0.55, pos6_r[0], pos6_r[1]+0.55, width=2.5)

# Numerical Features
y -= V_SPACE
pos7_r = draw_box(ax, branch_x_r, y, 4, 1.1,
                  'Numerical Features\n256 dimensions',
                  colors['technical'], fontsize=9)
draw_arrow(ax, pos6_r[0], pos6_r[1]-0.55, pos7_r[0], pos7_r[1]+0.55, width=2.5)
ax.text(branch_x_r - 2.2, y, '256d', fontsize=8, style='italic')
num_branch_end_y = pos7_r[1] - 0.55 # Store y coordinate for arrow

# ==================== FUSION AND OUTPUT ====================
y_fusion = text_branch_end_y - V_SPACE * 1.5 # Position fusion relative to branches
x_fusion = 10
pos_fusion = draw_box(ax, x_fusion, y_fusion, 4.5, 1.3, # Wider box
                      'Late Fusion\nConcatenate 512 + 256 = 768\nLayer Norm + Dropout',
                      colors['output'], fontsize=9)

# Arrows from both branches to fusion with slight curve
draw_arrow(ax, pos8[0], text_branch_end_y, x_fusion - 2.25, y_fusion, width=3, bend=-0.2)
draw_arrow(ax, pos7_r[0], num_branch_end_y, x_fusion + 2.25, y_fusion, width=3, bend=0.2)

# FC Layer 1
y_fusion -= V_SPACE
pos_fc1 = draw_box(ax, x_fusion, y_fusion, 4.5, 1.1,
                   'Fully Connected 1\n768 → 256 + ReLU',
                   colors['output'], fontsize=9)
draw_arrow(ax, x_fusion, pos_fusion[1]-0.65, x_fusion, pos_fc1[1]+0.55, width=2.5)
ax.text(x_fusion + 2.7, y_fusion, '256d', fontsize=8, style='italic')

# FC Layer 2
y_fusion -= V_SPACE * 0.9 # Slightly less space
pos_fc2 = draw_box(ax, x_fusion, y_fusion, 4.5, 1.1,
                   'Fully Connected 2\n256 → 1 (Logit)',
                   colors['output'], fontsize=9)
draw_arrow(ax, x_fusion, pos_fc1[1]-0.55, x_fusion, pos_fc2[1]+0.55, width=2.5)
ax.text(x_fusion + 2.7, y_fusion, '1d', fontsize=8, style='italic')

# Output
y_fusion -= V_SPACE * 0.9 # Slightly less space
pos_output = draw_box(ax, x_fusion, y_fusion, 4.5, 0.9,
                      'Binary Prediction\n(0=Down/Neutral, 1=Up)',
                      colors['input'], fontsize=9, bold=True)
draw_arrow(ax, x_fusion, pos_fc2[1]-0.55, x_fusion, pos_output[1]+0.45, width=2.5)

# ==================== TRAINING INFO BOX ====================
info_x, info_y = 2.0, 0.5 # Adjusted position
info_text = (
    'Training Configuration:\n'
    '• Loss: Focal Loss (γ=2.0)\n'
    '• Optimizer: AdamW (lr=1e-4)\n'
    '• Scheduler: Cosine Annealing\n'
    '• Precision: Mixed (FP16)\n'
    '• Parameters: ~15M'
)
ax.text(info_x, info_y, info_text,
        fontsize=8, va='bottom', ha='left', # Increased font size slightly
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor=colors['edge'], linewidth=1.5))

# ==================== PARAMETER COUNT BOX ====================
param_x, param_y = 18.0, 0.5 # Adjusted position
param_text = (
    'Parameter Breakdown:\n'
    '• BiLSTM: 8.4M (56%)\n'
    '• Attention: 2.1M (14%)\n'
    '• Transformer: 4.2M (28%)\n'
    '• Classifier: 0.3M (2%)\n'
    '• Total: ~15M'
)
ax.text(param_x, param_y, param_text,
        fontsize=8, va='bottom', ha='right', # Increased font size slightly
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen',
                  edgecolor=colors['edge'], linewidth=1.5))

# ==================== LEGEND ====================
legend_elements = [
    mpatches.Patch(facecolor=colors['input'], edgecolor=colors['edge'], label='Input/Output'),
    mpatches.Patch(facecolor=colors['embed'], edgecolor=colors['edge'], label='Embeddings/Features'),
    mpatches.Patch(facecolor=colors['lstm'], edgecolor=colors['edge'], label='LSTM Layers'),
    mpatches.Patch(facecolor=colors['attention'], edgecolor=colors['edge'], label='Attention'),
    mpatches.Patch(facecolor=colors['transformer'], edgecolor=colors['edge'], label='Transformer'),
    mpatches.Patch(facecolor=colors['output'], edgecolor=colors['edge'], label='Fusion/Classifier'),
    mpatches.Patch(facecolor=colors['technical'], edgecolor=colors['edge'], label='Numerical Processing'), # Added label for pink
]
ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=9, frameon=True) # Adjusted y position and columns

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjusted rect for legend space

# --- Save to current directory ---
output_dir = '.' # Save to the directory where the script is run
try:
    # Save as high-res PNG
    png_path = os.path.join(output_dir, 'architecture_diagram.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {os.path.abspath(png_path)} (300 DPI)")

    # Save as PDF (vector, perfect for LaTeX)
    pdf_path = os.path.join(output_dir, 'architecture_diagram.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {os.path.abspath(pdf_path)} (vector)")

    # Save as high-res PNG for presentations
    png_highres_path = os.path.join(output_dir, 'architecture_diagram_highres.png')
    plt.savefig(png_highres_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {os.path.abspath(png_highres_path)} (600 DPI)")

    print("\n🎉 Done! Files saved to your current directory.")

except Exception as e:
    print(f"❌ Error saving files: {e}")
    print("Please ensure you have write permissions in the script's directory.")

# plt.show() # Uncomment to display the plot directly