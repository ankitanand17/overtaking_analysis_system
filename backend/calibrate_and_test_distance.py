"""
calibrate_and_test_distance.py

Usage:
  python calibrate_and_test_distance.py --source 0
  python calibrate_and_test_distance.py --source test_video.mp4

Controls (while running):
  - 'c' : run calibration step (you will be prompted to place a known object and enter its real width & distance).
  - 'r' : re-run calibration (overwrite previous f).
  - 't' : record a test sample (you'll type the true/ground-truth distance for that frame; it'll log estimated vs true).
  - 'm' : toggle manual ROI selection for calibration (if YOLO can't find the object).
  - 'q' or ESC : quit and print/save results.

Outputs:
  - Prints live estimated distance on top-left of frame.
  - Saves test samples to distance_test_log.csv and prints MAE/RMSE on exit.
"""

"""
calibrate_and_test_distance_gpu.py

Same functionality as the CPU script but forces YOLO inference to run on GPU (if available).

Usage:
  python calibrate_and_test_distance_gpu.py --source 0
  python calibrate_and_test_distance_gpu.py --source test_video.mp4
"""
import argparse
import time
import csv
import math
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
import os
import torch

# ------------------------------------------------------------------------------
# Utility Helpers
# ------------------------------------------------------------------------------

def bbox_width(bbox):
    """
    Computes width of the 2D bounding box.

    Args:
        bbox (tuple/list): Coordinates [x1, y1, x2, y2].

    Returns:
        float: Bounding box pixel width.
    """
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1)

def estimate_distance(real_width_m, focal_px, width_px):
    """
    Pinhole camera model distance formula using width in pixels:
        d = (W_m * f) / w_px

    Args:
        real_width_m (float): Ground-truth width of the target object in meters.
        focal_px (float): Calibrated camera focal length in pixels.
        width_px (float): Bounding box pixel width.

    Returns:
        float or None: Estimated distance in meters.
    """
    if width_px <= 0 or focal_px is None or focal_px <= 0:
        return None
    return (real_width_m * focal_px) / width_px

def compute_metrics(records):
    """
    Computes evaluation metrics (Mean Absolute Error and Root Mean Squared Error) 
    comparing estimated distance values against true/ground-truth inputs.

    Args:
        records (list of dict): Recorded test data points.

    Returns:
        dict: Evaluated results keys: MAE, RMSE, and count.
    """
    errs = [r['est'] - r['true'] for r in records if r['est'] is not None and r['true'] is not None]
    if not errs:
        return None
    mae = sum(abs(e) for e in errs) / len(errs)
    rmse = math.sqrt(sum(e*e for e in errs) / len(errs))
    return {'MAE': mae, 'RMSE': rmse, 'count': len(errs)}

# ------------------------------------------------------------------------------
# Main Calibration/Testing Loop
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file path (default=0)")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model (ultralytics) path or name")
    ap.add_argument("--target-class", default="car", help="target object class name for calibration/estimation (default=car)")
    args = ap.parse_args()

    # Determine execution device: CUDA GPU if hardware supports it, else CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Parse video stream source (camera index or clip filepath)
    source = int(args.source) if args.source.isdigit() else args.source

    # Load YOLO Model
    model = YOLO(args.model)

    # Resolve class name string into class id index
    class_name_to_id = {name: idx for idx, name in model.names.items()}
    target_class_name = args.target_class
    target_class_id = class_name_to_id.get(target_class_name, None)
    if target_class_id is None:
        print(f"Warning: target class '{target_class_name}' not found in YOLO class names. Using detections of any class.")
    else:
        print(f"Using target class '{target_class_name}' (id={target_class_id}) for auto-detection.")

    # Initialize video capture source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: cannot open source:", source)
        return

    focal_px = None
    calib_real_width_m = None
    manual_roi_mode = False

    # Records array to aggregate test outputs
    records = []

    print("\nControls: 'c' calibrate, 'r' recalibrate, 'm' manual ROI toggle, 't' record test sample, ESC/q quit\n")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream / cannot fetch frame.")
            break
        frame_id += 1
        timestamp = time.time()

        # Run single-frame inference on GPU (using device argument)
        try:
            results = model.predict(frame, conf=0.35, device=device, verbose=False)
        except Exception as e:
            # Fallback to CPU if GPU driver fails
            print("Warning: model.predict on device failed, falling back to CPU. Error:", e)
            results = model.predict(frame, conf=0.35, device="cpu", verbose=False)

        chosen_bbox = None
        chosen_conf = 0.0
        chosen_class = None

        # Iterate predictions to find matching target class candidate
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = bbox_width((x1, y1, x2, y2))
                
                if target_class_id is not None:
                    if cls == target_class_id and conf > chosen_conf:
                        chosen_conf = conf
                        chosen_bbox = (x1, y1, x2, y2)
                        chosen_class = cls
                else:
                    # Select the largest box if class filters are disabled
                    if w > (bbox_width(chosen_bbox) if chosen_bbox else 0):
                        chosen_bbox = (x1, y1, x2, y2)
                        chosen_conf = conf
                        chosen_class = cls

        # Overlay all general predictions on stream
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item()); conf = float(box.conf[0].item())
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1,y1), (x2,y2), (200,200,0), 1)
                cv2.putText(frame, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0), 1)

        # Highlight target vehicle box
        if chosen_bbox is not None:
            x1,y1,x2,y2 = [int(v) for v in chosen_bbox]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(frame, f"TARGET {model.names[chosen_class]} conf:{chosen_conf:.2f}", (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

        # Overlay focal length & calculated distance outputs on screen
        info_lines = []
        if focal_px is None:
            info_lines.append("Focal: NOT calibrated (press 'c' to calibrate)")
        else:
            info_lines.append(f"Focal (px): {focal_px:.2f}")
            w_px = bbox_width(chosen_bbox) if chosen_bbox is not None else 0
            est = estimate_distance(calib_real_width_m, focal_px, w_px) if (calib_real_width_m and w_px>0) else None
            if est is not None:
                info_lines.append(f"Estimated distance: {est:.2f} m (object width={calib_real_width_m} m)")
            else:
                info_lines.append("Estimated distance: N/A (no valid bbox)")

        y0 = 30
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Distance Calibration & Test (GPU)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        if key == ord('m'):
            manual_roi_mode = not manual_roi_mode
            print("Manual ROI mode:", manual_roi_mode)

        if key == ord('c') or key == ord('r'):
            # Calibration routine logic: calculate focal length based on known width and distance
            print("\n--- CALIBRATION ---")
            print("Place the calibration object (of known real width) in view and press ENTER.")
            print("You can toggle manual ROI mode with 'm' before calibration.")
            input("Press Enter when ready to capture a frame for calibration...")

            ret2, frame2 = cap.read()
            if not ret2:
                print("Failed to capture frame for calibration.")
                continue

            # Query real physical specifications from CLI console input
            while True:
                try:
                    real_w = float(input("Enter the real object width in meters (e.g., car width 1.8): ").strip())
                    break
                except:
                    print("Invalid number. Try again.")
            while True:
                try:
                    known_dist = float(input("Enter the known distance to object in meters (e.g., 10): ").strip())
                    break
                except:
                    print("Invalid number. Try again.")

            # Perform detection on captured frame
            try:
                results2 = model.predict(frame2, conf=0.35, device=device, verbose=False)
            except Exception as e:
                print("GPU predict failed during calibration, falling back to CPU. Error:", e)
                results2 = model.predict(frame2, conf=0.35, device="cpu", verbose=False)

            chosen2 = None
            chosen_conf2 = 0.0
            for r in results2:
                for box in r.boxes:
                    cls = int(box.cls[0].item()); conf = float(box.conf[0].item())
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    w_px = x2 - x1
                    if target_class_id is not None and cls == target_class_id:
                        if conf > chosen_conf2:
                            chosen_conf2 = conf
                            chosen2 = (x1,y1,x2,y2)
                    else:
                        if chosen2 is None or w_px > (chosen2[2]-chosen2[0]):
                            chosen2 = (x1,y1,x2,y2)

            # Manual ROI selection popup window if requested or auto-detect fails
            if manual_roi_mode or chosen2 is None:
                print("No suitable detection found OR manual ROI requested.")
                print("Select ROI manually. After selecting, press ENTER or SPACE.")
                r = cv2.selectROI("Select calibration ROI", frame2, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select calibration ROI")
                x, y, w, h = r
                if w == 0 or h == 0:
                    print("ROI selection canceled. Aborting calibration.")
                    continue
                chosen2 = (x, y, x + w, y + h)

            w_px = bbox_width(chosen2)
            if w_px <= 0:
                print("Invalid bbox width from calibration. Aborting.")
                continue

            # Calculate focal length: f = (d * w_px) / W_m
            f_est = (known_dist * w_px) / real_w
            focal_px = f_est
            calib_real_width_m = real_w
            print(f"Calibration done: focal_px = {focal_px:.2f} px (using width {real_w} m at {known_dist} m).")
            print("-------------------\n")

            # Flush cache files from GPU memory
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if key == ord('t'):
            # Record test validation data point
            if focal_px is None:
                print("Calibration not done yet. Press 'c' first.")
                continue

            if chosen_bbox is None:
                print("No detection available for this frame.")
                r = cv2.selectROI("Select object ROI for test", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select object ROI for test")
                x,y,w,h = r
                if w == 0 or h == 0:
                    print("ROI canceled; skipping test.")
                    continue
                bbox = (x, y, x+w, y+h)
            else:
                bbox = chosen_bbox

            est_dist = estimate_distance(calib_real_width_m, focal_px, bbox_width(bbox))
            if est_dist is None:
                print("Could not estimate distance for this test sample.")
                continue

            # Prompt user for true distance to record estimation delta
            while True:
                try:
                    true_d = float(input("Enter the TRUE distance (meters) of the object in this frame: ").strip())
                    break
                except:
                    print("Invalid number. Try again.")

            rec = {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'est': est_dist,
                'true': true_d,
            }
            records.append(rec)
            err = rec['est'] - rec['true']
            print(f"Recorded test: est={rec['est']:.3f} m, true={rec['true']:.3f} m, err={err:.3f} m")

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    cap.release()
    cv2.destroyAllWindows()

    # Save log spreadsheet and print summary statistical results
    if records:
        df = pd.DataFrame(records)
        out_csv = "distance_test_log.csv"
        df.to_csv(out_csv, index=False)
        metrics = compute_metrics(records)
        if metrics:
            print("\n=== Distance estimation metrics ===")
            print(f"Samples: {metrics['count']}")
            print(f"MAE: {metrics['MAE']:.3f} m")
            print(f"RMSE: {metrics['RMSE']:.3f} m")
            print(f"Saved test records to {out_csv}")
    else:
        print("No test records saved. Calibration state:")
        print(f"focal_px: {focal_px}, calib_real_width_m: {calib_real_width_m}")

if __name__ == "__main__":
    main()
