"""
tracker_utils.py

This module provides helper utilities for object tracking. Specifically, it maps 
the output trajectories (tracks) produced by the SORT Kalman Filter tracker to 
the raw YOLO object detector outputs based on bounding box overlap (IoU).
"""

import numpy as np
from utils.helpers import iou


# ==============================================================================
# Associate SORT Tracks with YOLO Detections
# ==============================================================================

def associate_tracks_to_dets(tracks, detections):
    """
    Associates active SORT tracks to the raw YOLO detections for the current frame.
    This step allows us to retrieve class labels (e.g. 'car', 'truck') for 
    tracks that are mathematically tracked by the Kalman filter.

    Args:
        tracks (np.ndarray): Array of active tracks [[x1, y1, x2, y2, track_id], ...]
        detections (np.ndarray): Array of raw YOLO detections [[x1, y1, x2, y2, confidence, class], ...]

    Returns:
        dict: A mapping from track ID (int) to detection index (int or None).
    """
    mapping = {}

    if tracks is None:
        return mapping

    # --------------------------------------------------------------------------
    # Convert detection boxes to list of tuples for quick indexing
    # --------------------------------------------------------------------------
    if detections.size:
        detection_boxes = [
            tuple(map(float, detections[i, :4]))
            for i in range(detections.shape[0])
        ]
    else:
        detection_boxes = []

    # Keep track of which detections have already been matched to prevent 
    # multiple tracks mapping to the same physical detection box.
    used_detections = set()

    # --------------------------------------------------------------------------
    # Match each track with its best overlapping detection
    # --------------------------------------------------------------------------
    for track in tracks:
        track_box = tuple(map(float, track[:4]))
        best_detection_index = None
        best_iou = 0.0

        for i, det_box in enumerate(detection_boxes):
            # If this detection is already matched, skip it
            if i in used_detections:
                continue

            iou_value = iou(track_box, det_box)

            # Keep the detection with the highest IoU overlap
            if iou_value > best_iou:
                best_iou = iou_value
                best_detection_index = i

        # --------------------------------------------------------------------------
        # Save mapping if IoU is above our confidence threshold
        # --------------------------------------------------------------------------
        # An IoU threshold of 0.2 ensures we only match overlapping boxes
        if (
            best_detection_index is not None
            and best_iou > 0.2
        ):
            mapping[int(track[4])] = best_detection_index
            # Mark detection as used
            used_detections.add(best_detection_index)
        else:
            # No matching detection found for this track
            mapping[int(track[4])] = None

    return mapping