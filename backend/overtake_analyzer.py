"""
overtake_analyzer.py

This is the main computer vision pipeline coordinator script. It processes input 
driving videos frame-by-frame, performs vehicle detection using YOLO/Fallback, 
maintains track identities using the SORT algorithm, computes vehicle distances/velocities, 
runs the ADAS state machine logic, and exports telemetry reports and plots.
"""

import os
import argparse
from collections import defaultdict, deque

import cv2
import numpy as np

# ==============================================================================
# CONFIG
# ==============================================================================

from config.settings import *

# =========================================================
# DETECTION
# =========================================================

from detection.yolo_detector import (
    load_yolo_model,
    detect_objects
)

# =========================================================
# TRACKING
# =========================================================

from tracking.sort import Sort

from tracking.tracker_utils import (
    associate_tracks_to_dets
)

# =========================================================
# HELPERS
# =========================================================

from utils.helpers import (
    map_yolo_name,
    estimate_distance_by_height,
    ema_smooth,
    compute_t_req
)

# =========================================================
# TARGET LOCK
# =========================================================

from decision.target_lock import (
    lock_target,
    update_locked_target,
    recover_target,
    release_target
)

# =========================================================
# OVERTAKING LOGIC
# =========================================================

from decision.overtaking_logic import (
    get_nearest_oncoming_vehicle,
    start_return_candidate,
    can_merge_back,
    calculate_return_distance,
    get_suggested_action,
    resolve_decision_name
)

# =========================================================
# VISUALIZATION
# =========================================================

from visualization.draw import (
    draw_detections,
    draw_track_id,
    draw_host_centroid,
    draw_roi_lines,
    draw_status,
    draw_vehicle_info
)

# =========================================================
# OUTPUT
# =========================================================

from output.csv_logger import (
    create_log_row,
    save_csv
)

from output.summary import (
    create_summary,
    save_summary
)

from output.plots import (
    generate_all_plots
)


# =========================================================
# MAIN PROCESS FUNCTION
# =========================================================

def process_video(
    input_path,
    output_dir,
    yolo_weights='yolov8n.pt',
    conf=0.35,
    focal_px=DEFAULT_FOCAL_PX,
    save_video=True,
    show_preview=False
):
    """
    Core video processing pipeline. Operates headlessly under REST server invocation, 
    or can run with interactive previews locally via CLI.

    Args:
        input_path (str): Path to input mp4/avi video file.
        output_dir (str): Folder path where results will be exported.
        yolo_weights (str): Weights name for YOLO model initialization.
        conf (float): Object detector confidence threshold.
        focal_px (float): Calibrated camera focal length in pixels.
        save_video (bool): Flag to write annotated visual output frames to video.
        show_preview (bool): Flag to open live OpenCV rendering windows (GUI).

    Returns:
        dict: Paths to exported logs, summaries, plots, and video resources.
    """
    os.makedirs(output_dir, exist_ok=True)

    plots_dir = os.path.join(output_dir, 'plots')
    csv_path = os.path.join(
        output_dir,
        'frames_data.csv'
    )
    summary_json = os.path.join(
        output_dir,
        'summary.json'
    )
    annotated_video = os.path.join(
        output_dir,
        'annotated_video.mp4'
    )

    # =====================================================
    # LOAD YOLO
    # =====================================================

    model = load_yolo_model(yolo_weights)

    # =====================================================
    # VIDEO
    # =====================================================

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {input_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # =====================================================
    # VIDEO WRITER
    # =====================================================

    writer = None

    if save_video:

        # Use avc1 (H.264) for HTML5 video tag compatibility in modern browsers
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        writer = cv2.VideoWriter(
            annotated_video,
            fourcc,
            fps,
            (width, height)
        )

    # =====================================================
    # ROI
    # =====================================================

    x1 = width / 3.0
    x2 = 2.0 * width / 3.0

    hysteresis_px = 0.05 * (width / 3.0)

    x1_inner = x1 + hysteresis_px
    x2_inner = x2 - hysteresis_px

    # =====================================================
    # TRACKER
    # =====================================================

    tracker = Sort(
        max_age=60,
        min_hits=3,
        iou_threshold=0.5
    )

    # =====================================================
    # VARIABLES
    # =====================================================

    rows = []

    history = defaultdict(
        lambda: deque(maxlen=8)
    )

    raw_history = defaultdict(
        lambda: deque(maxlen=8)
    )

    roi_counters = {
        'LEFT': 0,
        'CENTER': 0,
        'RIGHT': 0
    }

    count_flag = 0

    frame_idx = 0

    target_locked_id = None

    target_missing_counter = 0

    return_candidate = False

    return_start_frame = 0

    return_last_rel_speed = 3.0

    host_speed_est = 15.0

    # ==========================================================================
    # PROCESS LOOP
    # ==============================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        display = frame.copy()

        # ======================================================================
        # HOST CENTROID
        # ======================================================================
        # Host projection point: representing where the user vehicle's center aligns
        host_centroid = (
            width // 2,
            int(height * 0.92)
        )

        draw_host_centroid(
            display,
            host_centroid
        )

        # ======================================================================
        # DETECTION
        # ======================================================================
        # Feed the frame to YOLO/Fallback detector to extract 2D boxes
        detections = detect_objects(
            frame,
            model=model,
            conf_threshold=conf
        )

        draw_detections(
            display,
            detections
        )

        # =================================================
        # TRACKING
        # =================================================

        if len(detections) == 0:

            dets_np = np.empty((0, 5))

        else:

            dets_np = np.array([

                [d[0], d[1], d[2], d[3], d[4]]

                for d in detections

            ])

        tracked = tracker.update(dets_np)

        dets_obj = np.array(
            detections,
            dtype=object
        )

        mapping = associate_tracks_to_dets(
            tracked,
            dets_obj
        )

        tracks_map = {}

        # =================================================
        # TRACK PROCESSING
        # =================================================

        center_targets = []

        right_oncoming = []

        left_lane_objs = []

        for tr in tracked:

            bx1, by1, bx2, by2, tid = tr

            tid = int(tid)

            cls_name = None

            if (
                tid in mapping
                and mapping[tid] is not None
            ):

                det_index = mapping[tid]

                cls_name = dets_obj[det_index, 5]

            group = map_yolo_name(cls_name)

            tracks_map[tid] = {

                'bbox': [
                    int(bx1),
                    int(by1),
                    int(bx2),
                    int(by2)
                ],

                'class_name': cls_name,

                'group': group
            }

            draw_track_id(
                display,
                tid,
                (bx1, by1, bx2, by2)
            )

            # =============================================
            # DISTANCE
            # =============================================

            bbox_h = by2 - by1

            raw_distance = estimate_distance_by_height(
                bbox_h,
                class_group=group,
                focal_px=focal_px
            )

            raw_history[tid].append(
                (frame_idx, raw_distance)
            )

            prev_smoothed = (

                history[tid][-1][1]

                if len(history[tid]) > 0

                else None
            )

            smoothed = ema_smooth(
                prev_smoothed,
                raw_distance
            )

            history[tid].append(
                (frame_idx, smoothed)
            )

            # =============================================
            # SPEED
            # =============================================

            prev_speed = 0.0

            if len(history[tid]) >= 2:

                f0, d0 = history[tid][-2]

                f1, d1 = history[tid][-1]

                dt = max(
                    (f1 - f0) / fps,
                    1e-3
                )

                if d0 is not None and d1 is not None:

                    prev_speed = (
                        d0 - d1
                    ) / dt

            # =============================================
            # MOTION CLASSIFICATION
            # =============================================

            motion_direction = "SAME_DIRECTION"

            motion_color = "YELLOW"

            if prev_speed > OPPOSITE_SPEED_THRESHOLD:

                motion_direction = "OPPOSITE_DIRECTION"

                motion_color = "RED"

            else:

                motion_direction = "SAME_DIRECTION"

                if prev_speed < 0:

                    motion_color = "GREEN"

                else:

                    motion_color = "YELLOW"

            # =============================================
            # ROI
            # =============================================

            cx = int((bx1 + bx2) / 2.0)

            if cx < x1_inner:

                left_lane_objs.append(
                    (
                        tid,
                        tracks_map[tid],
                        smoothed,
                        prev_speed,
                        group
                    )
                )

            elif cx >= x2_inner:

                right_oncoming.append(
                    (
                        tid,
                        tracks_map[tid],
                        smoothed,
                        prev_speed,
                        group
                    )
                )

            else:

                center_targets.append(
                    (
                        tid,
                        tracks_map[tid],
                        smoothed,
                        prev_speed,
                        group
                    )
                )

        # ======================================================================
        # TARGET LOCK State Machine
        # ======================================================================
        selected_target = None

        # Lock target vehicle if center lane corridor holds candidates and lock is free
        target_locked_id, selected_target = lock_target(
            center_targets,
            target_locked_id,
            count_flag,
            frame_idx
        )

        if (
            target_locked_id is not None
            and target_locked_id in tracks_map
        ):
            target_missing_counter = 0
            selected_target = update_locked_target(
                target_locked_id,
                tracks_map,
                history
            )

            # State transition validation:
            # If the locked target shifts from center corridor to left corridor (cx < x1_inner), 
            # we infer that the host vehicle has changed lanes to the right (overtaking lane initiated)
            bx1, by1, bx2, by2 = tracks_map[target_locked_id]['bbox']
            tid_cx = int((bx1 + bx2) / 2.0)
            if tid_cx < x1_inner and count_flag == 0:
                count_flag = 1
                print(f"[{frame_idx}] Inferred lane-change: locked target {target_locked_id} moved to LEFT ROI (cx={tid_cx}) -> count=1")

        elif target_locked_id is not None:
            # If the locked target is missing from the current frame tracking,
            # attempt to recover it by checking if another track ID occupies 
            # its last known proximity area.
            previous_target_bbox = None
            if (
                target_locked_id in tracks_map
            ):
                previous_target_bbox = (
                    tracks_map[target_locked_id]["bbox"]
                )

            target_locked_id, recovered = recover_target(
                target_locked_id,
                center_targets,
                history,
                frame_idx
            )

            if recovered:
                target_missing_counter = 0

            if not recovered:
                target_missing_counter += 1
                # Permanently release target lock if target has been missing consecutively 
                # for more than MAX_TARGET_MISSING frames
                lost = release_target(
                    target_missing_counter,
                    MAX_TARGET_MISSING,
                    frame_idx
                )

                if lost:
                    target_locked_id = None
                    target_missing_counter = 0

        # ======================================================================
        # ONCOMING VEHICLE ANALYSIS
        # ======================================================================
        # Locate the closest oncoming threat vehicle in the right lane corridor
        nearest_oncoming, nearest_t_oncoming = \
            get_nearest_oncoming_vehicle(
                right_oncoming
            )

        # ======================================================================
        # HOST ROI COUNTERS & STATE UPDATES
        # ======================================================================
        # Determine the host vehicle's current lane relative to vertical ROI lanes
        hx = host_centroid[0]
        if hx < x1_inner:
            host_roi = 'LEFT'
        elif hx >= x2_inner:
            host_roi = 'RIGHT'
        else:
            host_roi = 'CENTER'

        # Maintain lane occupancy stabilization counter values
        for k in ('LEFT', 'CENTER', 'RIGHT'):
            if k == host_roi:
                roi_counters[k] += 1
            else:
                roi_counters[k] = 0

        # If host stabilizes in the right corridor for N_CONFIRM frames,
        # update lane change flag count to 1 (actively overtaking)
        if roi_counters['RIGHT'] >= N_CONFIRM and count_flag == 0:
            count_flag = 1
            return_candidate = False
            return_start_frame = 0
            return_last_rel_speed = 0.0
            print(f"[{frame_idx}] Host entered RIGHT ROI -> count=1")

        # ======================================================================
        # RETURN CANDIDATE EVALUATION
        # ======================================================================
        # If overtaking lane is active and target vehicle has been passed/cleared,
        # set state to return_candidate (evaluating returning merges)
        prev_return_candidate = return_candidate
        return_candidate, decision_taken = \
            start_return_candidate(
                count_flag,
                selected_target,
                return_candidate,
                frame_idx
            )

        if return_candidate and not prev_return_candidate:
            return_start_frame = frame_idx
            return_last_rel_speed = 3.0

        # ======================================================================
        # RETURN DISTANCE CHECK
        # ======================================================================
        # Calculate time elapsed and distance cleared since return phase started
        elapsed_time = (
            frame_idx - return_start_frame
        ) / fps

        (
            min_return_distance,
            approx_distance_traveled

        ) = calculate_return_distance(
            return_last_rel_speed,
            elapsed_time,
            host_speed_est
        )

        left_clear = len(left_lane_objs) == 0

        # If original center lane is clear, wait buffers have elapsed, and host has cleared
        # the overtaken vehicle, merge back is authorized.
        merge_allowed = can_merge_back(
            roi_counters['CENTER'] >= N_CONFIRM,
            left_clear,
            approx_distance_traveled,
            min_return_distance,
            return_candidate
        )

        if merge_allowed:
            # Re-initialize lane index variables upon returning to original lane
            count_flag = 0
            return_candidate = False

        # =================================================
        # LOGGING
        # =================================================

        if len(tracks_map) == 0:

            row = create_log_row(

                frame_idx,
                track_id=-1,
                is_target=0,
                info={'class_name': 'unknown', 'group': 'unknown', 'bbox': [0, 0, 0, 0]},
                bbox=(0, 0, 0, 0),
                centroid=(0, 0),
                smoothed_distance=None,
                raw_distance=None,
                speed_mps=0.0,
                t_req=None,
                t_oncoming=nearest_t_oncoming,
                suggested_action='NO_ACTION',
                reason='no targets detected',
                count_flag=count_flag,
                decision_taken='NO_ACTION',
                return_candidate=return_candidate,
                approx_distance_traveled=approx_distance_traveled,
                min_return_distance=min_return_distance,
                motion_direction='SAME_DIRECTION',
                motion_color='YELLOW'

            )

            rows.append(row)

        locked_target_t_req = None
        for tid, info in tracks_map.items():

            bx1, by1, bx2, by2 = info['bbox']

            cx = int((bx1 + bx2) / 2)

            group = info['group']

            smoothed = history[tid][-1][1]

            raw_distance = raw_history[tid][-1][1]

            prev_speed = 0.0

            if len(history[tid]) >= 2:

                f0, d0 = history[tid][-2]
                f1, d1 = history[tid][-1]

                dt = max((f1 - f0) / fps, 1e-3)

                if d0 is not None and d1 is not None:
                    prev_speed = (d0 - d1) / dt

            is_target = (
                selected_target is not None
                and tid == selected_target['id']
            )

            t_req_val = None

            if is_target and smoothed is not None:

                d_pass = (
                    CLASS_HEIGHTS[group]
                    * 2.0
                )

                rel_speed = host_speed_est - prev_speed

                t_req_val = compute_t_req(

                    smoothed,
                    host_speed_est,
                    rel_speed,
                    d_pass

                )

            if is_target:
                locked_target_t_req = t_req_val

            suggested_action, reason = \
                get_suggested_action(

                    count_flag,
                    is_target,
                    smoothed,
                    group,
                    right_oncoming,
                    return_candidate,
                    left_clear,
                    approx_distance_traveled,
                    min_return_distance,
                    roi_counters['CENTER'] >= N_CONFIRM,
                    t_req_val=t_req_val,
                    nearest_t_oncoming=nearest_t_oncoming

                )

            decision_taken = resolve_decision_name(
                suggested_action
            )

            row = create_log_row(

                frame_idx,
                tid,
                is_target,
                info,
                (bx1, by1, bx2, by2),
                (cx, (by1 + by2) // 2),
                smoothed,
                raw_distance,
                prev_speed,
                t_req_val,
                nearest_t_oncoming,
                suggested_action,
                reason,
                count_flag,
                decision_taken,
                return_candidate,
                approx_distance_traveled,
                min_return_distance,
                motion_direction,
                motion_color
            )

            rows.append(row)

            # =============================================
            # DRAW VEHICLE INFORMATION
            # =============================================

            draw_vehicle_info(

                display,

                (bx1, by1, bx2, by2),

                tid,

                smoothed,

                prev_speed,

                suggested_action,

                reason,

                motion_color,

                is_target

            )

        # =================================================
        # DRAWING
        # =================================================

        draw_roi_lines(
            display,
            x1,
            x2,
            height
        )

        # Determine the big status label
        status_label = ""
        if target_locked_id is not None and count_flag == 0:
            is_safe = True
            if nearest_t_oncoming is not None:
                time_to_overtake = locked_target_t_req if locked_target_t_req is not None else 5.0
                if nearest_t_oncoming < (time_to_overtake + T_MARGIN):
                    is_safe = False
            
            if is_safe:
                status_label = "you_can_change the lane"
            else:
                status_label = "HOLD"
        elif return_candidate:
            if nearest_t_oncoming is not None:
                time_to_overtake = locked_target_t_req if locked_target_t_req is not None else 5.0
                if nearest_t_oncoming < (time_to_overtake + T_MARGIN):
                    status_label = "GO BACK TO ORIGINAL LANE"
                else:
                    status_label = "GO TO ORIGINAL LANE"
            else:
                status_label = "GO TO ORIGINAL LANE"
        elif count_flag > 0:
            if nearest_t_oncoming is not None:
                time_to_overtake = locked_target_t_req if locked_target_t_req is not None else 5.0
                if nearest_t_oncoming < (time_to_overtake + T_MARGIN):
                    status_label = "GO BACK TO ORIGINAL LANE"
                else:
                    status_label = "overtaking"
            else:
                status_label = "overtaking"

        draw_status(
            display,
            frame_idx,
            count_flag,
            return_candidate,
            target_locked_id,
            status_label=status_label
        )

        # =================================================
        # SAVE VIDEO
        # =================================================

        if writer is not None:
            writer.write(display)

        if show_preview:
            try:
                cv2.imshow("Smart Overtaking System", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception as e:
                print(f"Skipping preview window: {e}")

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    if writer is not None:
        writer.release()

    if show_preview:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # =====================================================
    # SAVE CSV
    # =====================================================

    df = save_csv(
        rows,
        csv_path
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary = create_summary(

        dataframe=df,

        input_video=input_path,

        total_frames=frame_idx

    )

    save_summary(
        summary,
        summary_json
    )

    # =====================================================
    # PLOTS
    # =====================================================

    generate_all_plots(
        df,
        plots_dir
    )

    print("Processing complete.")

    return {

        "csv_path": csv_path,
        "video_path": annotated_video,
        "summary_path": summary_json,
        "plots_dir": plots_dir

    }


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        required=True
    )

    parser.add_argument(
        '--output_dir',
        default='./results'
    )

    parser.add_argument(
        '--save_video',
        action='store_true'
    )

    parser.add_argument(
        '--show_preview',
        action='store_true'
    )

    parser.add_argument(
        '--yolo_weights',
        default='yolov8n.pt'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.35
    )

    parser.add_argument(
        '--focal_px',
        type=float,
        default=DEFAULT_FOCAL_PX
    )

    args = parser.parse_args()

    process_video(

        input_path=args.input,

        output_dir=args.output_dir,

        yolo_weights=args.yolo_weights,

        conf=args.conf,

        focal_px=args.focal_px,

        save_video=args.save_video,

        show_preview=args.show_preview

    )