"""
overtaking_logic.py

This module contains the core ADAS decision-making logic for overtaking maneuvers.
It processes tracking states, distance inputs, and oncoming vehicle speeds to recommend 
safe overtaking maneuvers (HOLD, YOU_CAN_CHANGE_LANE, OVERTAKE, GO_TO_ORIGINAL_LANE, etc.)
and monitors safety corridors.
"""

from utils.helpers import (
    compute_t_req,
    compute_t_oncoming
)

from config.settings import (
    T_MARGIN,
    RETURN_WAIT_TIME,
    CLASS_HEIGHTS
)


# ==============================================================================
# Find Nearest Oncoming Vehicle
# ==============================================================================

def get_nearest_oncoming_vehicle(right_oncoming):
    """
    Identifies the nearest oncoming vehicle in the opposite lane (right lane corridor).
    Calculates its time-to-collision (TTC) based on speed and relative distance.

    Args:
        right_oncoming (list): List of oncoming vehicles in the right lane:
                               [(tid, info, dist, speed, group), ...]

    Returns:
        tuple: (nearest_vehicle_tuple, oncoming_ttc_seconds or None)
    """
    if len(right_oncoming) == 0:
        return None, None

    # Sort oncoming vehicles by distance to identify the closest threat
    right_oncoming = sorted(
        right_oncoming,
        key=lambda x:
        x[2] if x[2] is not None else 1e6
    )

    nearest = right_oncoming[0]
    distance_oncoming = nearest[2]
    # Ensure speed is positive for time-to-collision calculations
    speed_oncoming = abs(nearest[3])

    t_oncoming = compute_t_oncoming(
        distance_oncoming,
        speed_oncoming
    )

    return nearest, t_oncoming


# ==============================================================================
# Start Return Candidate
# ==============================================================================

def start_return_candidate(
    count_flag,
    selected_target,
    return_candidate,
    frame_idx
):
    """
    State transition checker: initiates the lane-return checking phase once the host 
    vehicle is in the overtaking lane and has fully passed the target vehicle.

    Args:
        count_flag (int): Lane state tracker (1 = currently in overtaking lane).
        selected_target (dict or None): Current active target vehicle information.
        return_candidate (bool): Active return candidate state flag.
        frame_idx (int): Current frame index.

    Returns:
        tuple: (updated_return_candidate_flag, decision_taken_string)
    """
    decision = "NONE"

    # If the host is in the overtaking lane, and the target vehicle is no longer in 
    # the camera field of view, we can initiate the return-merge-back check
    if (
        count_flag == 1
        and selected_target is None
        and not return_candidate
    ):
        return_candidate = True
        decision = "START_RETURN_CANDIDATE"
        print(
            f"[{frame_idx}] "
            f"Overtaken target has cleared. Initiating return candidate merge checklist."
        )

    return return_candidate, decision


# ==============================================================================
# Check Merge Back Permission
# ==============================================================================

def can_merge_back(
    roi_center_confirmed,
    left_clear,
    approx_distance_traveled,
    min_return_distance,
    return_candidate
):
    """
    Evaluates safety conditions to permit merging back to the original center lane.

    Args:
        roi_center_confirmed (bool): True if host has re-aligned near center.
        left_clear (bool): True if no vehicle occupies the safety zone.
        approx_distance_traveled (float): Distance traveled since starting return.
        min_return_distance (float): Calculated minimum safe return distance threshold.
        return_candidate (bool): True if return candidate state is active.

    Returns:
        bool: True if safe to merge back, False otherwise.
    """
    if (
        roi_center_confirmed
        and left_clear
        and approx_distance_traveled >= min_return_distance
        and return_candidate
    ):
        return True

    return False


# ==============================================================================
# Calculate Return Distance
# ==============================================================================

def calculate_return_distance(
    return_last_relative_speed,
    elapsed_time,
    host_speed
):
    """
    Calculates safety clearance margins during the merging back phase.

    Args:
        return_last_relative_speed (float): Relative speed when return phase started.
        elapsed_time (float): Time elapsed in seconds since starting return phase.
        host_speed (float): Est speed of host vehicle.

    Returns:
        tuple: (min_return_distance_required, approx_distance_traveled)
    """
    # Safe return distance is relative velocity * wait time buffer
    min_return_distance = (
        abs(return_last_relative_speed)
        *
        RETURN_WAIT_TIME
    )

    # Approximate distance traversed during the return maneuver
    approx_distance_traveled = (
        elapsed_time
        *
        max(host_speed, 0.1)
    )

    return (
        min_return_distance,
        approx_distance_traveled
    )


# ==============================================================================
# Suggested Driving Action
# ==============================================================================

def get_suggested_action(
    count_flag,
    is_target,
    smoothed_distance,
    vehicle_group,
    right_oncoming,
    return_candidate,
    left_clear,
    approx_distance_traveled,
    min_return_distance,
    roi_center_confirmed,
    t_req_val=None,
    nearest_t_oncoming=None
):
    """
    Main state machine decision logic. Generates driving actions based on 
    lane positions, oncoming vehicles, and corridor distance safety margins.

    Returns:
        tuple: (suggested_action_string, explanation_reason_string)
    """
    # Get physical safe distance threshold for the target vehicle type
    safe_distance = CLASS_HEIGHTS.get(
        vehicle_group,
        CLASS_HEIGHTS['unknown']
    )

    # Check for oncoming threats in the opposite lane
    oncoming_threat = False
    time_to_overtake = 5.0
    if nearest_t_oncoming is not None:
        time_to_overtake = t_req_val if t_req_val is not None else 5.0
        # If oncoming vehicle time-to-collision is below required passing time + buffer
        if nearest_t_oncoming < (time_to_overtake + T_MARGIN):
            oncoming_threat = True

    # --------------------------------------------------------------------------
    # Case 1: Host vehicle is in the original lane (count_flag == 0)
    # --------------------------------------------------------------------------
    if count_flag == 0:
        if is_target:
            # If oncoming traffic is too close, hold lane position
            if oncoming_threat:
                return (
                    "HOLD",
                    f"oncoming threat (TTC {nearest_t_oncoming:.1f}s < req {time_to_overtake:.1f}s + margin)"
                )

            # Recommend lane change if target distance is greater than the safety threshold
            if (
                smoothed_distance is not None
                and smoothed_distance >= safe_distance
            ):
                return (
                    "YOU_CAN_CHANGE_LANE",
                    f"dist {smoothed_distance:.1f} >= safe {safe_distance:.1f}"
                )
            else:
                return (
                    "HOLD",
                    f"dist {smoothed_distance:.1f} < safe {safe_distance:.1f}"
                )

        return (
            "NO_ACTION",
            "not target"
        )

    # --------------------------------------------------------------------------
    # Case 2: Host vehicle is in the overtaking lane (count_flag == 1)
    # --------------------------------------------------------------------------
    else:
        if len(right_oncoming) == 0:
            action = "OVERTAKE"
            reason = "right lane clear"
        else:
            action = "BACK_TO_ORIGINAL_LANE"
            reason = "oncoming vehicle detected"

    # --------------------------------------------------------------------------
    # Case 3: Overtaken vehicle has cleared; returning to original lane
    # --------------------------------------------------------------------------
    if return_candidate:
        if (
            left_clear
            and approx_distance_traveled >= min_return_distance
            and roi_center_confirmed
        ):
            return (
                "GO_TO_ORIGINAL_LANE",
                "safe to merge back"
            )
        else:
            return (
                "HOLD",
                "waiting before merge back"
            )

    return action, reason


# ==============================================================================
# Final Decision Name
# ==============================================================================

def resolve_decision_name(suggested_action):
    """
    Standardizes recommended visual badge string outputs for dashboard integration.

    Args:
        suggested_action (str): Suggested decision output from the decision engine.

    Returns:
        str: Final UI standardized action name label.
    """
    if suggested_action == "YOU_CAN_CHANGE_LANE":
        return "PREPARE_TO_CHANGE"

    elif suggested_action == "HOLD":
        return "HOLD"

    elif suggested_action == "OVERTAKE":
        return "OVERTAKE"

    elif suggested_action == "BACK_TO_ORIGINAL_LANE":
        return "BACK_TO_ORIGINAL_LANE"

    elif suggested_action == "GO_TO_ORIGINAL_LANE":
        return "MERGE_LEFT"

    elif suggested_action == "OVERTAKE_NEXT":
        return "OVERTAKE_NEXT"

    return "NO_ACTION"