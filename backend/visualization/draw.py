"""
draw.py

This module provides helper utilities for drawing OpenCV visualizations on top 
of the processed video frames. It displays bounding boxes, track ID labels, 
ROI boundary corridors, active host statuses, and active target vehicle telemetry metrics.
"""

import cv2


# ==============================================================================
# Draw Bounding Box
# ==============================================================================

def draw_bbox(
    frame,
    bbox,
    color=(0, 255, 0),
    thickness=2
):
    """
    Draws a 2D rectangle bounding box on the image frame.

    Args:
        frame (np.ndarray): Image array in BGR format.
        bbox (tuple/list): Coordinate box bounds [x1, y1, x2, y2].
        color (tuple): Box outline color (B, G, R). Defaults to green (0, 255, 0).
        thickness (int): Outline thickness width. Defaults to 2.
    """
    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness
    )


# ==============================================================================
# Draw Detection Label
# ==============================================================================

def draw_label(
    frame,
    text,
    position,
    color=(255, 255, 255),
    scale=0.5,
    thickness=1
):
    """
    Helper function to write text label overlays on the frame.

    Args:
        frame (np.ndarray): Target image frame.
        text (str): String message to display.
        position (tuple): Coordinate position (x, y) for text base line.
        color (tuple): Text RGB color. Defaults to white.
        scale (float): Font scale multiplier factor. Defaults to 0.5.
        thickness (int): Text thickness width. Defaults to 1.
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


# ==============================================================================
# Draw All Detections
# ==============================================================================

def draw_detections(
    frame,
    detections
):
    """
    Overlays all raw object detections as thin green bounding boxes and confidence labels.

    Args:
        frame (np.ndarray): Target BGR image frame.
        detections (list): Standardized detection list: [[x1, y1, x2, y2, score, class], ...]
    """
    for det in detections:
        x1, y1, x2, y2, score, cls_name = det

        # Draw box bounds
        draw_bbox(
            frame,
            (x1, y1, x2, y2),
            color=(0, 180, 0),
            thickness=1
        )

        label = f"{cls_name}:{score:.2f}"

        # Draw label header
        draw_label(
            frame,
            label,
            (x1, max(10, y1 - 6)),
            color=(0, 255, 0)
        )


# ==============================================================================
# Draw Track ID
# ==============================================================================

def draw_track_id(
    frame,
    track_id,
    bbox
):
    """
    Draws the unique tracking ID text overlay above a tracked vehicle.

    Args:
        frame (np.ndarray): BGR image frame.
        track_id (int): Assigned track ID.
        bbox (tuple): Target bounding box [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    text = f"ID {track_id}"

    draw_label(
        frame,
        text,
        (int(x1), max(12, int(y1) - 6)),
        color=(255, 255, 255)
    )


# ==============================================================================
# Draw Host Vehicle Centroid
# ==============================================================================

def draw_host_centroid(
    frame,
    centroid
):
    """
    Draws a visual center dot marker indicating host vehicle center projection on road.

    Args:
        frame (np.ndarray): BGR frame.
        centroid (tuple): Centroid pixel point (cx, cy).
    """
    cx, cy = centroid

    cv2.circle(
        frame,
        (int(cx), int(cy)),
        5,
        (255, 0, 0),
        -1
    )


# ==============================================================================
# Draw ROI Lane Lines
# ==============================================================================

def draw_roi_lines(
    frame,
    x1,
    x2,
    height
):
    """
    Draws vertical safety corridor boundary lines (orange) dividing left, center, 
    and right lanes inside the image.

    Args:
        frame (np.ndarray): Frame array.
        x1 (float): Left boundary x-pixel index.
        x2 (float): Right boundary x-pixel index.
        height (int): Frame pixel height.
    """
    # Left corridor boundary
    cv2.line(
        frame,
        (int(x1), 0),
        (int(x1), height),
        (0, 140, 255),
        2
    )

    # Right corridor boundary
    cv2.line(
        frame,
        (int(x2), 0),
        (int(x2), height),
        (0, 140, 255),
        2
    )


# ==============================================================================
# Draw Status Information
# ==============================================================================

def draw_status(
    frame,
    frame_number,
    count_flag,
    return_candidate,
    locked_target,
    status_label=""
):
    """
    Draws structural ADAS engine system metrics on the top left of the screen,
    including active frame, merge counter state, and action guidelines.

    Args:
        frame (np.ndarray): BGR frame.
        frame_number (int): Active frame index.
        count_flag (int): Lane index state tracker.
        return_candidate (bool): True if returning checklist is active.
        locked_target (int or None): Active locked track ID.
        status_label (str): Text suggestion to overlay (e.g. 'HOLD', 'you_can_change the lane').
    """
    text = (
        f"F:{frame_number} "
        f"count:{count_flag} "
        f"ret:{int(return_candidate)} "
        f"lock:{str(locked_target)}"
    )

    # Telemetry parameters block
    draw_label(
        frame,
        text,
        (8, 24),
        color=(0, 255, 0),
        scale=0.6,
        thickness=2
    )

    # Highlight suggestion text in yellow/cyan
    if status_label:
        cv2.putText(
            frame,
            status_label,
            (8, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


# ==============================================================================
# Draw Vehicle Information
# ==============================================================================

def draw_vehicle_info(
    frame,
    bbox,
    track_id,
    distance,
    speed,
    action,
    reason,
    motion_color,
    is_target=False
):
    """
    Draws detailed HUD telemetry lines (ID, distance, relative velocity, action, reason) 
    above a tracked vehicle. Highlights target vehicles with a magenta border box.
    """
    x1, y1, x2, y2 = bbox

    # --------------------------------------------------------------------------
    # Select Color based on vehicle direction classification
    # --------------------------------------------------------------------------
    color = (0, 255, 255)

    if motion_color == "RED":
        color = (0, 0, 255) # Oncoming vehicle threat

    elif motion_color == "GREEN":
        color = (0, 255, 0) # Safe target vehicle moving faster

    elif motion_color == "YELLOW":
        color = (0, 255, 255) # Same direction caution

    # --------------------------------------------------------------------------
    # Highlight target box if it's the locked target
    # --------------------------------------------------------------------------
    if is_target:
        # Draw magenta frame
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "TARGET",
            (x1, y1 - 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

    # --------------------------------------------------------------------------
    # Info Text Block
    # --------------------------------------------------------------------------
    info_lines = [
        f"ID: {track_id}",
        f"Dist: {distance:.1f}m" if distance is not None else "Dist: N/A",
        f"Speed: {speed:.1f}m/s",
        f"Action: {action}",
        f"Reason: {reason}",
        f"Motion: {motion_color}"
    ]

    # --------------------------------------------------------------------------
    # Draw Text Lines
    # --------------------------------------------------------------------------
    for i, text in enumerate(info_lines):
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10 - (i * 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2
        )