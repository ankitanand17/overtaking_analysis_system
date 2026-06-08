"""
helpers.py

This module contains various mathematics, geometry, and mapping helper functions
utilized by the overtaking decision logic. Includes pinhole camera model-based 
distance estimation, exponential moving average filters, required passing time, 
oncoming collision time, and IoU functions.
"""

import math

from config.settings import (
    CLASS_HEIGHTS,
    YOLO_TO_GROUP,
    EMA_ALPHA
)

# ==============================================================================
# Map YOLO Class Name
# ==============================================================================

def map_yolo_name(name: str):
    """
    Standardizes YOLO class labels into general vehicle groups.

    Args:
        name (str): Raw class name label output from YOLOv8 model.

    Returns:
        str: General class group ('car', 'truck', 'van', etc.) or 'unknown'.
    """
    if name is None:
        return 'unknown'

    return YOLO_TO_GROUP.get(name.lower(), 'unknown')


# ==============================================================================
# Estimate Distance Using Vehicle Height
# ==============================================================================

def estimate_distance_by_height(
    bbox_h_px,
    class_group='car',
    focal_px=1500.0
):
    """
    Calculates estimated target vehicle distance from camera sensor using a 
    standard pinhole camera model formula: 
        d = (H * f) / h_px
    where 'H' is the reference class height, 'f' is camera focal length, and 
    'h_px' is the detected bounding box pixel height.

    Args:
        bbox_h_px (float): Bounding box pixel height.
        class_group (str): Standardized class group (e.g. 'car', 'truck').
        focal_px (float): Camera focal length in pixels.

    Returns:
        float or None: Estimated distance in meters, or None if height is invalid.
    """
    if bbox_h_px is None or bbox_h_px <= 0:
        return None

    # Retrieve physical height 'H' for class group
    real_height = CLASS_HEIGHTS.get(
        class_group,
        CLASS_HEIGHTS['unknown']
    )

    # Perform distance calculation
    distance = (real_height * float(focal_px)) / float(bbox_h_px)
    return distance


# ==============================================================================
# Exponential Moving Average Smoothing
# ==============================================================================

def ema_smooth(
    previous_value,
    new_value,
    alpha=EMA_ALPHA
):
    """
    Applies Exponential Moving Average (EMA) smoothing to reduce noise in distance estimations.
    Formula:
        smoothed_t = alpha * raw_t + (1 - alpha) * smoothed_{t-1}

    Args:
        previous_value (float or None): Previous frame's smoothed value.
        new_value (float or None): Current frame's raw estimation value.
        alpha (float): Interpolation factor weight. Defaults to config EMA_ALPHA.

    Returns:
        float or None: Smoothed output, or fallback value if inputs are None.
    """
    if previous_value is None:
        return new_value

    if new_value is None:
        return previous_value

    smoothed = (
        alpha * new_value
        +
        (1.0 - alpha) * previous_value
    )

    return float(smoothed)


# ==============================================================================
# Compute Required Overtaking Time
# ==============================================================================

def compute_t_req(
    distance_to_target_m,
    host_speed_mps,
    relative_speed_mps,
    d_pass
):
    """
    Calculates the estimated duration required to safely complete the overtaking 
    maneuver. Based on relative speed and safety passing distance boundary.
    Formula:
        t_req = d_pass / relative_speed

    Args:
        distance_to_target_m (float): Distance in meters to locked vehicle.
        host_speed_mps (float): Host vehicle speed in m/s.
        relative_speed_mps (float): Speed difference (host - target) in m/s.
        d_pass (float): Total longitudinal distance required to pass target.

    Returns:
        float: Estimated overtaking duration in seconds.
    """
    epsilon = 0.1
    # Avoid zero division and negative relative speeds (ensure positive moving delta)
    relative_speed = max(relative_speed_mps, epsilon)
    t_req = float(d_pass) / relative_speed
    return t_req


# ==============================================================================
# Compute Oncoming Vehicle Time
# ==============================================================================

def compute_t_oncoming(
    distance_oncoming_m,
    oncoming_speed_mps
):
    """
    Calculates the time-to-collision (TTC) for the nearest oncoming vehicle in
    the overtaking lane.
    Formula:
        t_oncoming = distance_oncoming_m / oncoming_speed_mps

    Args:
        distance_oncoming_m (float): Distance to oncoming vehicle.
        oncoming_speed_mps (float): Absolute speed of oncoming vehicle.

    Returns:
        float: Time-to-collision in seconds.
    """
    epsilon = 0.01
    t_oncoming = (
        float(distance_oncoming_m)
        /
        max(oncoming_speed_mps, epsilon)
    )
    return t_oncoming


# ==============================================================================
# Intersection Over Union (IOU)
# ==============================================================================

def iou(boxA, boxB):
    """
    Calculates Intersection over Union (IoU) overlap score between two boxes boxA and boxB.

    Args:
        boxA (list/tuple): Coordinates [x1, y1, x2, y2].
        boxB (list/tuple): Coordinates [x1, y1, x2, y2].

    Returns:
        float: Overlap score between 0.0 and 1.0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Intersecting rectangle width and height
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute areas
    boxA_area = max(
        0,
        (boxA[2] - boxA[0]) *
        (boxA[3] - boxA[1])
    )

    boxB_area = max(
        0,
        (boxB[2] - boxB[0]) *
        (boxB[3] - boxB[1])
    )

    # Union area calculation
    union = (
        boxA_area +
        boxB_area -
        inter_area
    )

    if union <= 0:
        return 0.0

    return inter_area / union