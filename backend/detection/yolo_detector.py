"""
yolo_detector.py

This module handles object detection for the overtaking analysis system. It attempts 
to use Ultralytics YOLOv8 for deep learning-based vehicle classification and localization. 
If Ultralytics is not installed or available, it falls back to a classical computer 
vision motion-and-contour detector.
"""

import cv2
import numpy as np

# Try importing YOLO from the ultralytics library for deep-learning-based vehicle detection
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ==============================================================================
# Load YOLO Model
# ==============================================================================

def load_yolo_model(weights_path='yolov8n.pt'):
    """
    Initializes and loads the YOLOv8 model from the given weights file path.

    Args:
        weights_path (str): File path to weights (e.g. 'yolov8n.pt').

    Returns:
        YOLO or None: Loaded YOLO model object, or None if YOLO is not available.
    """
    if YOLO is None:
        print("YOLO not available. Fallback detector will be used instead.")
        return None

    # Load YOLO model with pre-trained weights
    model = YOLO(weights_path)
    return model


# ==============================================================================
# Run YOLO Detection
# ==============================================================================

def run_yolo_detection(
    model,
    frame,
    conf_threshold=0.35
):
    """
    Performs object detection on a single video frame using the loaded YOLO model.

    Args:
        model (YOLO): The loaded YOLOv8 model object.
        frame (np.ndarray): BGR image frame from opencv video source.
        conf_threshold (float): Minimum confidence threshold to accept a detection.

    Returns:
        list of list: A list of detections, where each detection is:
                      [x1, y1, x2, y2, confidence_score, class_name]
    """
    detections = []

    if model is None:
        return detections

    # Perform inference.
    # imgsz=640 downscales or scales the frame to 640 for optimized performance
    results = model(
        frame,
        imgsz=640,
        conf=conf_threshold
    )[0]

    # Extract bounding boxes, class labels, and confidence scores
    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        # Get coordinates [x1, y1, x2, y2]
        boxes = results.boxes.xyxy.cpu().numpy()
        # Get confidence scores
        scores = results.boxes.conf.cpu().numpy()
        # Get class indices as integers
        class_ids = (
            results.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )
        # Get class name lookup map
        names = (
            model.names
            if hasattr(model, 'names')
            else {}
        )

        # Map predictions to output format
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )
            score = float(scores[i])
            class_id = int(class_ids[i])
            class_name = names.get(
                class_id,
                str(class_id)
            )

            detections.append([
                x1,
                y1,
                x2,
                y2,
                score,
                class_name
            ])

    return detections


# ==============================================================================
# Fallback Motion Detector
# ==============================================================================

def fallback_detection(frame):
    """
    Classical Computer Vision fallback detector using difference thresholding and contours.
    Used when YOLO is unavailable. Identifies regions of changes (motion/objects).

    Args:
        frame (np.ndarray): BGR input frame.

    Returns:
        list of list: Bounding boxes under format [x1, y1, x2, y2, score=0.5, class_name='unknown'].
    """
    detections = []

    # Convert image to grayscale for intensity-based processing
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Median blur filter to reduce noise and keep edge boundaries clean
    blur = cv2.medianBlur(gray, 7)

    # Calculate absolute pixel difference between the raw and blurred grayscale image
    diff = cv2.absdiff(gray, blur)

    # Threshold the difference to create a binary mask of localized variations
    _, threshold = cv2.threshold(
        diff,
        30,
        255,
        cv2.THRESH_BINARY
    )

    # Extract contour boundaries from binary mask
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Loop through contours and filter out small noise blobs
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Skip small contour bounding boxes (below 2000 area size)
        if w * h < 2000:
            continue

        # Add as a detection
        detections.append([
            x,
            y,
            x + w,
            y + h,
            0.5,        # Assign default score of 0.5 for fallback matches
            'unknown'   # Class name is unknown under classical CV fallback
        ])

    return detections


# ==============================================================================
# Main Detection Wrapper
# ==============================================================================

def detect_objects(
    frame,
    model=None,
    conf_threshold=0.35
):
    """
    Unified entry point for object detection. Delegates to YOLO or Fallback.

    Args:
        frame (np.ndarray): BGR video frame.
        model (YOLO, optional): Loaded YOLOv8 model instance. Defaults to None.
        conf_threshold (float): Inference confidence threshold. Defaults to 0.35.

    Returns:
        list of list: Standardized list of detections [[x1, y1, x2, y2, score, class_name], ...]
    """
    # Use YOLO deep learning inference if model weights are loaded
    if model is not None:
        return run_yolo_detection(
            model,
            frame,
            conf_threshold
        )

    # Fallback to background subtraction/contour thresholding otherwise
    return fallback_detection(frame)