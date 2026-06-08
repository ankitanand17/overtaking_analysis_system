"""
target_lock.py

This module manages the "target lock" lifecycle for the overtaking system.
It locks onto the nearest vehicle in the center lane corridor, tracks it across frames,
recovers it using distance proximity if its track ID changes (due to temporary occlusion
or detector noise), and releases the lock if the vehicle remains missing for too long.
"""

# ==============================================================================
# Lock Initial Target
# ==============================================================================

def lock_target(
    center_targets,
    target_locked_id,
    count_flag,
    frame_idx
):
    """
    Identifies and locks onto a vehicle in the center corridor when there is no
    active lock, and the host vehicle is still in its original lane (count_flag == 0).

    Args:
        center_targets (list): Candidates in the center corridor:
                               [(tid, info, dist, speed, group), ...]
        target_locked_id (int or None): Current locked vehicle ID.
        count_flag (int): Lane state flag (0 = original lane, 1 = overtaking lane).
        frame_idx (int): Current frame index.

    Returns:
        tuple: (updated_target_locked_id, selected_target_dictionary or None)
    """
    selected_target = None

    # We only lock onto a new target if we aren't currently locked, and 
    # the host is still in the original lane (before initiating overtaking)
    if (
        target_locked_id is None
        and count_flag == 0
    ):
        if len(center_targets) > 0:
            # Sort center candidates by distance to target (lock onto the closest vehicle)
            center_targets = sorted(
                center_targets,
                key=lambda x:
                x[2] if x[2] is not None else 1e6
            )

            # Select nearest candidate
            sel = center_targets[0]
            (
                sel_tid,
                sel_info,
                sel_dist,
                sel_speed,
                sel_group
            ) = sel

            target_locked_id = int(sel_tid)

            # Build metadata dictionary for the locked target
            selected_target = {
                "id": target_locked_id,
                "bbox": sel_info["bbox"],
                "dist_m": sel_dist,
                "speed_mps": sel_speed,
                "group": sel_group
            }

            print(
                f"[{frame_idx}] "
                f"Locked target {target_locked_id} (Dist: {sel_dist:.1f}m)"
            )

    return target_locked_id, selected_target


# ==============================================================================
# Update Locked Target
# ==============================================================================

def update_locked_target(
    target_locked_id,
    tracks_map,
    history
):
    """
    Updates coordinate bounding boxes and smoothed distance metrics for the active 
    locked target vehicle.

    Args:
        target_locked_id (int): Active locked track ID.
        tracks_map (dict): Coordinates mapping for active tracks in the frame.
        history (dict): Distance history arrays for active tracks.

    Returns:
        dict or None: Updated target details, or None if target ID is no longer active.
    """
    if target_locked_id not in tracks_map:
        return None

    bbox = tracks_map[target_locked_id]["bbox"]
    group = tracks_map[target_locked_id].get(
        "group",
        "unknown"
    )

    smoothed_distance = None

    # Retrieve most recent smoothed distance value from EMA history
    if (
        target_locked_id in history
        and len(history[target_locked_id]) > 0
    ):
        smoothed_distance = (
            history[target_locked_id][-1][1]
        )

    selected_target = {
        "id": target_locked_id,
        "bbox": bbox,
        "dist_m": smoothed_distance,
        "speed_mps": None, # Velocity is computed dynamically in main analyzer loop
        "group": group
    }

    return selected_target


# ==============================================================================
# Recover Missing Target
# ==============================================================================

def recover_target(
    target_locked_id,
    center_targets,
    history,
    frame_idx
):
    """
    Attempts to recover a lost target if its track ID suddenly changed in the 
    current frame. Matches target's last known distance against current candidates
    within a threshold of 2.0 meters.

    Args:
        target_locked_id (int): The last locked target track ID.
        center_targets (list): Candidates currently detected in center lane.
        history (dict): Tracks history database.
        frame_idx (int): Current frame index.

    Returns:
        tuple: (recovered_target_id, success_boolean)
    """
    recovered = False

    if len(center_targets) == 0:
        return target_locked_id, recovered

    if target_locked_id not in history:
        return target_locked_id, recovered

    if len(history[target_locked_id]) == 0:
        return target_locked_id, recovered

    # Retrieve last known smoothed distance of the locked target
    previous_distance = (
        history[target_locked_id][-1][1]
    )

    if previous_distance is None:
        return target_locked_id, recovered

    best_candidate = None
    best_error = 999999

    # Find the candidate vehicle closest in distance to the lost target
    for (
        cand_tid,
        cand_info,
        cand_dist,
        cand_speed,
        cand_group
    ) in center_targets:

        if cand_dist is None:
            continue

        # Distance discrepancy calculation
        distance_error = abs(
            previous_distance -
            cand_dist
        )

        # Skip candidates with high distance variation (must be within 2.0 meters)
        if distance_error > 2.0:
            continue

        if distance_error < best_error:
            best_error = distance_error
            best_candidate = cand_tid

    # If a match is verified, update the lock to this new track ID
    if best_candidate is not None:
        print(
            f"[{frame_idx}] "
            f"Recovered target {target_locked_id} -> mapping to new track ID {best_candidate} (Error: {best_error:.2f}m)"
        )
        target_locked_id = int(
            best_candidate
        )
        recovered = True

    return target_locked_id, recovered


# ==============================================================================
# Release Lost Target
# ==============================================================================

def release_target(
    target_missing_counter,
    max_target_missing,
    frame_idx
):
    """
    Checks if a target has been missing for too long and needs to be released.

    Args:
        target_missing_counter (int): Consecutive frames target has been missing.
        max_target_missing (int): Threshold limit to release target.
        frame_idx (int): Current frame index.

    Returns:
        bool: True if target lock is released, False otherwise.
    """
    if target_missing_counter > max_target_missing:
        print(
            f"[{frame_idx}] "
            f"Target permanently lost after {target_missing_counter} frames. Releasing lock."
        )
        return True

    return False