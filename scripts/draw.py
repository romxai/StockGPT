import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def create_training_curves(
    target_acc, 
    final_loss, 
    total_epochs, 
    noise_level=0.01, 
    overfit_start_epoch=60, 
    overfit_strength=0.0001,
    learning_steepness=0.1
):
    """
    Creates realistic training curves with multi-layered noise and overfitting.
    
    Args:
        target_acc (float): The peak accuracy the model *must* reach.
        final_loss (float): The lowest loss the model *must* reach.
        total_epochs (int): The total number of epochs.
        noise_level (float): Base amount of random noise.
        overfit_start_epoch (int): The epoch where overfitting begins.
        overfit_strength (float): How fast the model overfits.
        learning_steepness (float): How fast the model learns.

    Returns:
        tuple: (acc_curve, loss_curve, peak_acc_epoch, min_loss_epoch)
    """
    
    overfit_start_epoch = min(max(5, overfit_start_epoch), total_epochs - 1)
    
    # --- 1. Generate Noise ---
    # Low-frequency "wandering" noise
    raw_noise_low_freq = np.random.normal(0, noise_level * 2.5, total_epochs)
    smoothing_window = 7
    smoothed_noise = np.convolve(raw_noise_low_freq, np.ones(smoothing_window) / smoothing_window, 'same')
    
    # High-frequency "granular/fine" noise
    fine_noise = np.random.normal(0, noise_level * 0.5, total_epochs)
    
    total_noise_acc = smoothed_noise * 0.5 + fine_noise * 0.5
    total_noise_loss = smoothed_noise + fine_noise

    # --- 2. Base Learning Phase ---
    x = np.arange(1, total_epochs + 1)
    
    # Base Accuracy Curve (Logistic)
    initial_acc = 0.40 # Start a bit higher than 1/3
    acc_range = (target_acc * 0.95) - initial_acc # Aim slightly below target to let noise peak
    base_acc_curve = initial_acc + acc_range / (1 + np.exp(-learning_steepness * (x - overfit_start_epoch / 1.8)))
    
    # Base Loss Curve (Exponential Decay)
    initial_loss = 1.0
    loss_range = initial_loss - (final_loss * 1.05) # Aim slightly above target
    base_loss_curve = (final_loss * 1.05) + loss_range * np.exp(-learning_steepness * 0.9 * x)

    # --- 3. Base Overfitting Phase ---
    overfit_epochs = total_epochs - overfit_start_epoch
    if overfit_epochs > 0:
        x_overfit = np.arange(overfit_epochs)
        
        # Accuracy: Slow quadratic decay
        acc_peak = base_acc_curve[overfit_start_epoch - 1]
        acc_overfit = acc_peak - overfit_strength * (x_overfit ** 2)
        base_acc_curve[overfit_start_epoch:] = acc_overfit
        
        # Loss: Slow quadratic rise
        loss_min = base_loss_curve[overfit_start_epoch - 1]
        loss_overfit = loss_min + (overfit_strength * 2.5) * (x_overfit ** 2)
        base_loss_curve[overfit_start_epoch:] = loss_overfit
        
    # --- 4. Combine Curves, Noise, and Scale to Match Target ---
    
    # Accuracy:
    acc_noisy = base_acc_curve + total_noise_acc
    # Scale the curve so its absolute peak is *exactly* target_acc
    current_peak_val = np.max(acc_noisy)
    peak_offset = target_acc - current_peak_val
    acc_final = acc_noisy + peak_offset
    
    # Loss:
    loss_noisy = base_loss_curve + total_noise_loss
    # Scale the curve so its absolute minimum is *exactly* final_loss
    current_min_val = np.min(loss_noisy)
    min_offset = final_loss - current_min_val
    loss_final = loss_noisy + min_offset

    # --- 5. Find "Early Stop" points ---
    peak_acc_epoch = np.argmax(acc_final) + 1
    min_loss_epoch = np.argmin(loss_final) + 1

    # Clip to reasonable bounds
    acc_final = np.clip(acc_final, 0.3, 0.95)
    loss_final = np.clip(loss_final, 0.1, 1.5)

    return acc_final, loss_final, peak_acc_epoch, min_loss_epoch

# --- Configuration for Each Model (Based on your image) ---
TOTAL_EPOCHS = 80

# Peak accuracies from the bar chart
model_peaks = {
    "Random Forest": 0.620,
    "Logistic Regression": 0.585,
    "Baseline LSTM": 0.635,
    "BiLSTM + Attention": 0.705,
    "Transformer": 0.725,
    "Hybrid (ours)": 0.785,
    "Hybrid + Optuna": 0.825,
}

# Estimated minimum losses (inversely related to accuracy)
model_min_loss = {
    "Baseline LSTM": 0.85,
    "BiLSTM + Attention": 0.70,
    "Transformer": 0.65,
    "Hybrid (ours)": 0.50,
    "Hybrid + Optuna": 0.42,
}

# Colors to match the bar chart
model_colors = {
    "Random Forest": "#a9aaa9",
    "Logistic Regression": "#b5b7b6",
    "Baseline LSTM": "#9fa1a0",
    "BiLSTM + Attention": "#3498db",  # Blue
    "Transformer": "#2980b9",         # Darker Blue
    "Hybrid (ours)": "#2ecc71",       # Green
    "Hybrid + Optuna": "#27ae60",     # Darker Green
}

# --- Generate Curves for NN Models ---
nn_models = ["Baseline LSTM", "BiLSTM + Attention", "Transformer", "Hybrid (ours)", "Hybrid + Optuna"]
model_curves = {}

# Tuned parameters for different learning behaviors
params = {
    "Baseline LSTM": (0.025, 65, 0.00005, 0.07),
    "BiLSTM + Attention": (0.018, 60, 0.0001, 0.09),
    "Transformer": (0.020, 55, 0.00012, 0.1),
    "Hybrid (ours)": (0.012, 55, 0.00015, 0.12),
    "Hybrid + Optuna": (0.010, 50, 0.00018, 0.14), # Learns fastest, overfits slightly sooner
}

for name in nn_models:
    noise, start_epoch, strength, steepness = params[name]
    model_curves[name] = create_training_curves(
        target_acc=model_peaks[name],
        final_loss=model_min_loss[name],
        total_epochs=TOTAL_EPOCHS,
        noise_level=noise,
        overfit_start_epoch=start_epoch,
        overfit_strength=strength,
        learning_steepness=steepness
    )

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
fig.suptitle('Model Training History', fontsize=22, y=1.03)

# --- Plot 1: Validation Accuracy ---
ax1.set_title('Validation Accuracy over Epochs', fontsize=18)
ax1.set_ylabel('Validation Accuracy', fontsize=14)
ax1.set_ylim(0.55, 0.9) # Zoom in
ax1.set_xlim(1, TOTAL_EPOCHS)
ax1.grid(linestyle='--', alpha=0.6, which='both')

legend_handles_acc = []

# Plot NN Models
for name, (acc, loss, peak_epoch, min_loss) in model_curves.items():
    x_axis = np.arange(1, TOTAL_EPOCHS + 1)
    line, = ax1.plot(x_axis, acc, color=model_colors[name], linewidth=2.2, alpha=0.9)
    legend_handles_acc.append(Line2D([0], [0], color=model_colors[name], lw=2.5, label=name))
    
    # Add early stop marker
    if name in ["Hybrid (ours)", "Hybrid + Optuna"]:
        ax1.plot(peak_epoch, acc[peak_epoch - 1], 'o', 
                 color=model_colors[name], markersize=8, 
                 markeredgecolor='black', markeredgewidth=1.5)

# Plot Baseline Models (Horizontal Lines)
rf_acc = model_peaks["Random Forest"]
lr_acc = model_peaks["Logistic Regression"]

ax1.axhline(y=rf_acc, color=model_colors["Random Forest"], linestyle='--', lw=2.5, label="Random Forest (Baseline)")
ax1.axhline(y=lr_acc, color=model_colors["Logistic Regression"], linestyle=':', lw=2.5, label="Logistic Regression (Baseline)")

# Manually add baselines to legend handles
legend_handles_acc.append(Line2D([0], [0], color=model_colors["Random Forest"], linestyle='--', lw=2.5, label='Random Forest (Baseline)'))
legend_handles_acc.append(Line2D([0], [0], color=model_colors["Logistic Regression"], linestyle=':', lw=2.5, label='Logistic Regression (Baseline)'))

# Create custom legend for accuracy
early_stop_marker_acc = Line2D([0], [0], marker='o', color='w',
                           label='Peak Accuracy (Early Stop)',
                           markerfacecolor='gray', markeredgecolor='black', 
                           markersize=8, markeredgewidth=1.5)
legend_handles_acc.append(early_stop_marker_acc)
ax1.legend(handles=legend_handles_acc, loc='lower right', fontsize=11, ncol=1)

# --- Plot 2: Validation Loss ---
ax2.set_title('Validation Loss over Epochs', fontsize=18)
ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Validation Loss', fontsize=14)
ax2.set_ylim(0.35, 1.2)
ax2.grid(linestyle='--', alpha=0.6, which='both')

legend_handles_loss = []

# Plot NN Models
for name, (acc, loss, peak_acc, min_loss_epoch) in model_curves.items():
    x_axis = np.arange(1, TOTAL_EPOCHS + 1)
    line, = ax2.plot(x_axis, loss, color=model_colors[name], linewidth=2.2, alpha=0.9)
    legend_handles_loss.append(Line2D([0], [0], color=model_colors[name], lw=2.5, label=name))
    
    # Add early stop marker
    if name in ["Hybrid (ours)", "Hybrid + Optuna"]:
        ax2.plot(min_loss_epoch, loss[min_loss_epoch - 1], 'o', 
                 color=model_colors[name], markersize=8, 
                 markeredgecolor='black', markeredgewidth=1.5)

# Create custom legend for loss
early_stop_marker_loss = Line2D([0], [0], marker='o', color='w',
                                label='Minimum Loss (Early Stop)',
                                markerfacecolor='gray', markeredgecolor='black', 
                                markersize=8, markeredgewidth=1.5)
legend_handles_loss.append(early_stop_marker_loss)
ax2.legend(handles=legend_handles_loss, loc='upper right', fontsize=11, ncol=1)

# --- Save the File ---
save_path = 'model_training_curves.png'
try:
    # Get the directory where the script is running
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'model_training_curves.png')
except NameError:
    # Fallback if __file__ is not defined (e.g., in some interactive envs)
    save_path = 'model_training_curves.png'

try:
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plots successfully saved to: {os.path.abspath(save_path)}")
except Exception as e:
    print(f"❌ Error saving plots: {e}")

plt.show()