import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load the data from CSV files ---
try:
    human_df = pd.read_csv('data_absolute_values_human.csv')
    unfiltered_df = pd.read_csv('data_absolute_values_unfiltered.csv')
    filtered_df = pd.read_csv('data_absolute_values_filtered.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure all CSV files are in the same directory as the script.")
    exit()

# --- Prepare data for plotting ---
# Add a 'source' column to identify each dataset
human_df['source'] = 'Human'
unfiltered_df['source'] = 'Unfiltered'
filtered_df['source'] = 'Filtered'

# Combine into a single DataFrame
full_df = pd.concat([human_df, unfiltered_df, filtered_df], ignore_index=True)

# --- Define metrics and titles for plotting ---
metrics_to_plot = {
    'jerk_eef_max': {'title': 'Max Cartesian Jerk', 'unit': '(m/s³)'},
    'jerk_joint_max': {'title': 'Max Joint Jerk', 'unit': '(rad/s³)'},
    'eef_path_length': {'title': 'Cartesian Path Length', 'unit': '(m)'},
    'joint_path_length': {'title': 'Joint Path Length', 'unit': '(rad)'},
    'orientation_path_length': {'title': 'Orientation Path Length', 'unit': '(rad)'}
}

# --- Plotting ---
# Use a professional and clean style suitable for a thesis
plt.style.use('seaborn-v0_8-paper')
palette = {'Human': '#4c72b0', 'Unfiltered': '#dd8452', 'Filtered': '#55a868'} # Blue, Orange, Green

# --- Loop through each metric and create a separate plot ---
for metric, info in metrics_to_plot.items():
    # Create a new figure for each plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(x='source', y=metric, data=full_df, ax=ax,
                order=['Human', 'Unfiltered', 'Filtered'],
                palette=palette,
                showfliers=True) # Set to True to show outliers

    ax.set_ylabel(f"Value {info['unit']}", fontsize=16)
    ax.set_xlabel(None) # Remove x-axis label for cleaner look
    ax.tick_params(axis='x', labelsize=16)

    # Use a log scale for jerk metrics if outliers are very large, which is common
    if 'jerk' in metric:
        ax.set_yscale('log')
        ax.set_ylabel(f"Value {info['unit']} (log scale)", fontsize=16)

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout()

    # Save the figure for your thesis with a unique name
    plt.savefig(f"figure_boxplot_{metric}.png", dpi=300)

    # Display the plot
    plt.show()

    # Close the figure to free up memory before the next loop iteration
    plt.close(fig)
