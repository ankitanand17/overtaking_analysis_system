"""
plots.py

This module provides functions to generate graphical plots from the generated 
telemetry data. It maps driving actions to integer codes and plots action 
timelines and counts using Matplotlib.
"""

import os
import matplotlib
# Use 'Agg' non-interactive backend for server compatibility (avoids GUI window errors in headless mode)
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# Action Name to Numeric Code
# ==============================================================================

# Map suggested string actions to numeric values for y-axis plotting representation.
ACTION_MAP = {
    'NO_ACTION': 0,
    'HOLD': 1,
    'YOU_CAN_CHANGE_LANE': 2,
    'OVERTAKE': 3,
    'BACK_TO_ORIGINAL_LANE': 4,
    'GO_TO_ORIGINAL_LANE': 5,
    'OVERTAKE_NEXT': 6
}


# ==============================================================================
# Generate Action Timeline Plot
# ==============================================================================

def generate_action_timeline_plot(
    dataframe,
    plots_dir
):
    """
    Creates a scatter plot showing recommended ADAS actions over the duration 
    of the video (frame by frame).

    Args:
        dataframe (pd.DataFrame): Telemetry log dataframe.
        plots_dir (str): Directory where the timeline plot image will be saved.
    """
    # Map suggested_action labels to action code integers
    dataframe['action_code'] = (
        dataframe['suggested_action']
        .map(ACTION_MAP)
        .fillna(0)
        .astype(int)
    )

    plt.figure(figsize=(10, 4))

    # Scatter plot representing frames vs action code integers
    plt.scatter(
        dataframe['frame'],
        dataframe['action_code'],
        c=dataframe['action_code'],
        s=6
    )

    # Label y-ticks with the original action names
    plt.yticks(
        list(ACTION_MAP.values()),
        list(ACTION_MAP.keys())
    )

    plt.xlabel('Frame')
    plt.ylabel('Action')
    plt.title('Action Timeline')
    plt.tight_layout()

    save_path = os.path.join(
        plots_dir,
        'action_timeline.png'
    )

    plt.savefig(save_path)
    plt.close()
    print(f"Saved timeline plot: {save_path}")


# ==============================================================================
# Generate Action Count Plot
# ==============================================================================

def generate_action_count_plot(
    dataframe,
    plots_dir
):
    """
    Creates a bar plot summarizing the total counts of each recommended ADAS action.

    Args:
        dataframe (pd.DataFrame): Telemetry log dataframe.
        plots_dir (str): Directory where the bar chart image will be saved.
    """
    plt.figure(figsize=(6, 4))

    # Bar chart showing action occurrences frequency count
    dataframe['suggested_action'] \
        .value_counts() \
        .plot(kind='bar')

    plt.title('Action Counts')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.tight_layout()

    save_path = os.path.join(
        plots_dir,
        'action_counts.png'
    )

    plt.savefig(save_path)
    plt.close()
    print(f"Saved count plot: {save_path}")


# ==============================================================================
# Generate All Plots
# ==============================================================================

def generate_all_plots(
    dataframe,
    plots_dir
):
    """
    Unified entry point to construct and save all statistical evaluation plots.

    Args:
        dataframe (pd.DataFrame): Frame telemetry log dataframe.
        plots_dir (str): Output folder path where plots will be saved.
    """
    os.makedirs(
        plots_dir,
        exist_ok=True
    )

    try:
        generate_action_timeline_plot(
            dataframe,
            plots_dir
        )

        generate_action_count_plot(
            dataframe,
            plots_dir
        )

        print(f"All plots saved in: {plots_dir}")

    except Exception as error:
        print(f"Plot generation failed: {error}")