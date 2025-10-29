import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# --- Data from the Image ---
labels = [
    'Baseline', 
    '+1000d Data', 
    '+80 Features', 
    '+BiLSTM', 
    '+Attn', 
    '+FinBERT', 
    '+Optuna'
]

# Individual contributions (deltas) for the stacked bar
# Note: The top value is 0.025 to make the total 0.825
contributions = [0.635, 0.050, 0.040, 0.030, 0.025, 0.020, 0.025]

# Cumulative values for the line plot
cumulative_accuracy = np.cumsum(contributions)
# [0.635, 0.685, 0.725, 0.755, 0.780, 0.800, 0.825]

target_accuracy = 0.80

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Model Component Analysis and Accuracy Progression', fontsize=20, y=1.05)

# --- Plot (a): Cumulative Component Contributions ---
ax1.set_title('(a) Cumulative Component Contributions', fontsize=16, pad=10)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim(0, 0.9)
ax1.set_xticks([]) # Remove x-axis ticks
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Create the stacked bar
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(contributions)))
bottom = 0
bar_width = 0.5

for i, (value, label, color) in enumerate(zip(contributions, labels, colors)):
    ax1.bar(0, value, bottom=bottom, width=bar_width, color=color, label=label, edgecolor='black')
    
    # Add text label inside the bar
    ax1.text(0, bottom + value / 2, f'{value:.3f}', ha='center', va='center', 
             color='black' if value < 0.04 else 'white', 
             fontweight='bold', fontsize=10)
    
    bottom += value

# Add 80% Target line
ax1.axhline(y=target_accuracy, color='#d35400', linestyle='--', linewidth=2, label='80% Target', alpha=0.8)

# Create custom legend for stacked bar
handles, _ = ax1.get_legend_handles_labels()
# Reverse handles for legend (to match top-down order)
ax1.legend(
    handles=handles[::-1], 
    loc='upper left', 
    bbox_to_anchor=(1.02, 1), 
    fontsize=10,
    title="Components"
)

# --- Plot (b): Accuracy Progression ---
ax2.set_title('(b) Accuracy Progression', fontsize=16, pad=10)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_xlabel('Configuration', fontsize=12)
ax2.set_ylim(0.6, 0.88)
ax2.grid(axis='both', linestyle='--', alpha=0.6)

# Line plot
line_color = '#27ae60' # Green
x_ticks = np.arange(len(labels))
ax2.plot(x_ticks, cumulative_accuracy, color=line_color, marker='o', 
         markersize=8, linewidth=3, markerfacecolor=line_color,
         markeredgecolor='black', markeredgewidth=1.5)

# Set x-tick labels
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)

# Add 80% Target line
ax2.axhline(y=target_accuracy, color='#d35400', linestyle='--', linewidth=2, label='80% Target', alpha=0.8)

# Add annotations
# Annotation for "Enhanced Features"
ax2.annotate(
    'Enhanced\nFeatures', 
    xy=(x_ticks[2], cumulative_accuracy[2]), 
    xytext=(x_ticks[2], cumulative_accuracy[2] - 0.04),
    fontsize=10, 
    ha='center',
    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
)

# Annotation for "Extended Data"
ax2.annotate(
    'Extended\nData', 
    xy=(x_ticks[1], cumulative_accuracy[1]), 
    xytext=(x_ticks[1], cumulative_accuracy[1] - 0.04),
    fontsize=10, 
    ha='center',
    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
)

# Annotation for "Target Achieved!"
ax2.annotate(
    'Target\nAchieved!', 
    xy=(x_ticks[-2], cumulative_accuracy[-2]), 
    xytext=(x_ticks[-2] - 0.5, cumulative_accuracy[-2] + 0.05),
    fontsize=11, 
    fontweight='bold',
    color='green',
    arrowprops=dict(arrowstyle='->', color='green', lw=2, connectionstyle="arc3,rad=0.3")
)

# Add final accuracy value
final_acc = cumulative_accuracy[-1]
ax2.text(x_ticks[-1], final_acc + 0.005, f'{final_acc * 100:.1f}%', 
         ha='center', va='bottom', fontsize=11, fontweight='bold')


ax2.legend(loc='upper left', fontsize=10)

# --- Save and Show ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for main title

# Get the directory where the script is running
save_path = 'component_contribution_plots.png'
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'component_contribution_plots.png')
except NameError:
    # Fallback if __file__ is not defined (e.g., in some interactive envs)
    save_path = 'component_contribution_plots.png'

try:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plots successfully saved to: {os.path.abspath(save_path)}")
except Exception as e:
    print(f"❌ Error saving plots: {e}")

plt.show()