"""
summary.py

This module compiles statistical metadata summaries from the generated 
overtaking logs. It outputs reports containing action count statistics, frame 
counts, and evaluates whether a maneuver is classified as a "harsh overtaking" 
event based on safety boundaries.
"""

import os
import json
import pandas as pd

from config.settings import T_MARGIN


# ==============================================================================
# Count Suggested Actions
# ==============================================================================

def get_action_statistics(dataframe):
    """
    Computes value counts for all suggested actions in the telemetry log.

    Args:
        dataframe (pd.DataFrame): Telemetry log dataframe.

    Returns:
        dict: A dictionary mapping action names to their occurrence count.
    """
    if dataframe.empty:
        return {}

    action_counts = (
        dataframe['suggested_action']
        .value_counts()
        .to_dict()
    )

    return action_counts


# ==============================================================================
# Detect Harsh Overtaking
# ==============================================================================

def detect_harsh_overtaking(dataframe):
    """
    Analyzes the telemetry logs to detect aggressive or risky overtaking actions.
    A maneuver is considered harsh if:
      - The time gap (oncoming TTC - required passing time) drops below 80% of T_MARGIN.
      - The host vehicle is forced to perform an emergency merge-back (aborted overtaking).

    Args:
        dataframe (pd.DataFrame): Frame telemetry log dataframe.

    Returns:
        tuple: (harsh_flag_boolean, list_of_warning_logs)
    """
    harsh = False
    notes = []

    # --------------------------------------------------------------------------
    # Check unsafe timing gaps during lane changes
    # --------------------------------------------------------------------------
    starts = dataframe[
        dataframe['suggested_action']
        ==
        'YOU_CAN_CHANGE_LANE'
    ]

    for _, row in starts.iterrows():
        t_oncoming = (
            row['t_oncoming_s']
            if (
                't_oncoming_s' in row
                and not pd.isna(row['t_oncoming_s'])
            )
            else 1e6
        )

        t_required = (
            row['t_req_s']
            if (
                't_req_s' in row
                and not pd.isna(row['t_req_s'])
            )
            else 1e6
        )

        # Unsafe margin threshold calculation (80% of our configured safety margin)
        if (
            t_oncoming - t_required
        ) < (0.8 * T_MARGIN):
            harsh = True
            notes.append(
                f"Close timing warning at frame {int(row['frame'])}: "
                f"TTC oncoming ({t_oncoming:.1f}s) - required time ({t_required:.1f}s) "
                f"is below safe threshold ({0.8 * T_MARGIN:.1f}s)."
            )

    # --------------------------------------------------------------------------
    # Check for emergency lane aborts (BACK_TO_ORIGINAL_LANE)
    # --------------------------------------------------------------------------
    aborts = dataframe[
        dataframe['suggested_action']
        ==
        'BACK_TO_ORIGINAL_LANE'
    ]

    if len(aborts) > 0:
        harsh = True
        notes.append(
            f"Emergency lane return (abort) events detected: {len(aborts)} frame(s)."
        )

    return harsh, notes


# ==============================================================================
# Create Summary Dictionary
# ==============================================================================

def create_summary(
    dataframe,
    input_video,
    total_frames
):
    """
    Compiles action statistics and safety evaluations into a summary dictionary.

    Args:
        dataframe (pd.DataFrame): Frame telemetry logs.
        input_video (str): File name/path of the analyzed video.
        total_frames (int): Total number of frames analyzed.

    Returns:
        dict: Standardized summary report map.
    """
    action_counts = get_action_statistics(
        dataframe
    )

    harsh, notes = detect_harsh_overtaking(
        dataframe
    )

    summary = {
        "video":
            os.path.basename(input_video),
        "frames":
            int(total_frames),
        "actions_counts":
            action_counts,
        "harsh_overtaking":
            bool(harsh),
        "notes":
            notes
    }

    return summary


# ==============================================================================
# Save Summary JSON
# ==============================================================================

def save_summary(
    summary,
    summary_json_path
):
    """
    Saves the summary dictionary as a formatted JSON file.

    Args:
        summary (dict): Summarized stats dictionary.
        summary_json_path (str): Output filepath to write JSON to.
    """
    with open(summary_json_path, 'w') as file:
        json.dump(
            summary,
            file,
            indent=2
        )

    print(f"Saved summary JSON report: {summary_json_path}")