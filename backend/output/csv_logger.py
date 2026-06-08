"""
csv_logger.py

This module contains utilities to assemble telemetry records frame-by-frame and 
export them as structured Pandas DataFrames or CSV files. This data is utilized 
by the front-end dashboard for generating plots and charts.
"""

import pandas as pd
from config.settings import CLASS_HEIGHTS


# ==============================================================================
# Create Single CSV Row
# ==============================================================================

def create_log_row(
    frame_idx,
    track_id,
    is_target,
    info,
    bbox,
    centroid,
    smoothed_distance,
    raw_distance,
    speed_mps,
    t_req,
    t_oncoming,
    suggested_action,
    reason,
    count_flag,
    decision_taken,
    return_candidate,
    approx_distance_traveled,
    min_return_distance,
    motion_direction,
    motion_color
):
    """
    Constructs a dictionary row representing the telemetry metrics for a specific 
    tracked object in a single video frame.

    Args:
        frame_idx (int): Current video frame index.
        track_id (int): Unique track identifier assigned by tracker.
        is_target (int): Flag (0 or 1) indicating if target is the locked vehicle.
        info (dict): Class and grouping info.
        bbox (tuple): Bounding box coordinates [x1, y1, x2, y2].
        centroid (tuple): Centroid coordinate projection (cx, cy).
        smoothed_distance (float or None): Smoothed EMA distance.
        raw_distance (float or None): Raw calculated distance.
        speed_mps (float): Calculated target speed in m/s.
        t_req (float or None): Passing duration required.
        t_oncoming (float or None): TTC for oncoming vehicle.
        suggested_action (str): Suggested driving action string.
        reason (str): Reason for the driving action suggestion.
        count_flag (int): Lane index flag indicator.
        decision_taken (str): Final standardized decision badge name.
        return_candidate (int): Flag indicating if merging checklists are active.
        approx_distance_traveled (float): Distance traveled since starting return.
        min_return_distance (float): Calculated minimum safe return distance.
        motion_direction (str): Motion class direction.
        motion_color (str): Motion display color coding label.

    Returns:
        dict: Populated row conforming to the output dataset schema.
    """
    bx1, by1, bx2, by2 = bbox
    cx, cy = centroid
    group = info.get('group', 'unknown')

    row = {
        # ----------------------------------------------------------------------
        # Frame Information
        # ----------------------------------------------------------------------
        "frame": frame_idx,
        "track_id": int(track_id),
        "is_target": int(is_target),

        # ----------------------------------------------------------------------
        # Vehicle Information
        # ----------------------------------------------------------------------
        "vehicle_class_name":
            info.get('class_name', 'unknown'),
        "vehicle_class_group":
            group,

        # ----------------------------------------------------------------------
        # Bounding Box Coordinates
        # ----------------------------------------------------------------------
        "bbox_x1": int(bx1),
        "bbox_y1": int(by1),
        "bbox_x2": int(bx2),
        "bbox_y2": int(by2),

        # ----------------------------------------------------------------------
        # Centroid Points
        # ----------------------------------------------------------------------
        "centroid_x": int(cx),
        "centroid_y": int(cy),

        # ----------------------------------------------------------------------
        # Distance / Speed
        # ----------------------------------------------------------------------
        "distance_m":
            float(smoothed_distance)
            if smoothed_distance is not None
            else None,

        "raw_distance_m":
            float(raw_distance)
            if raw_distance is not None
            else None,

        "speed_mps":
            float(speed_mps),

        # ----------------------------------------------------------------------
        # Vehicle Safety Thresholds
        # ----------------------------------------------------------------------
        "class_height_m":
            float(
                CLASS_HEIGHTS.get(
                    group,
                    CLASS_HEIGHTS['unknown']
                )
            ),

        "class_d_pass_m":
            float(
                CLASS_HEIGHTS.get(
                    group,
                    CLASS_HEIGHTS['unknown']
                ) * 2.0
            ),

        "class_safe_follow_m":
            float(
                CLASS_HEIGHTS.get(
                    group,
                    CLASS_HEIGHTS['unknown']
                )
            ),

        # ----------------------------------------------------------------------
        # Time Metrics (Seconds)
        # ----------------------------------------------------------------------
        "t_req_s":
            float(t_req)
            if t_req is not None
            else None,

        "t_oncoming_s":
            float(t_oncoming)
            if t_oncoming is not None
            else None,

        # ----------------------------------------------------------------------
        # Decision States
        # ----------------------------------------------------------------------
        "suggested_action":
            suggested_action,
        "reason":
            reason,
        "count_flag":
            int(count_flag),
        "decision_taken":
            decision_taken,
        "return_candidate":
            int(return_candidate),

        # ----------------------------------------------------------------------
        # Return Distance Clearing
        # ----------------------------------------------------------------------
        "approx_distance_traveled_since_return_m":
            float(approx_distance_traveled),
        "min_return_distance_required_m":
            float(min_return_distance),

        # ----------------------------------------------------------------------
        # Motion Direction Parameters
        # ----------------------------------------------------------------------
        "relative_speed_mps":
            float(speed_mps),
        "motion_direction":
            motion_direction,
        "motion_color":
            motion_color,
    }

    return row


# ==============================================================================
# Save CSV File
# ==============================================================================

def save_csv(rows, csv_path):
    """
    Saves the list of telemetry log rows as a CSV file using Pandas.

    Args:
        rows (list of dict): Collected frame telemetry logs.
        csv_path (str): Destination file path.

    Returns:
        pd.DataFrame: Created dataframe object.
    """
    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        csv_path,
        index=False
    )

    print(f"Saved telemetry log CSV: {csv_path}")
    return dataframe