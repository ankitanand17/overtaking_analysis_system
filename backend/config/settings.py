# ==============================================================================
# Camera / Distance Settings
# ==============================================================================

# Default camera focal length in pixels. This is calibrated for distance estimation.
# Used in the formula: distance = (real_height * focal_px) / bounding_box_height_px
DEFAULT_FOCAL_PX = 1500.0

# ==============================================================================
# Lane Change Parameters
# ==============================================================================

# Number of consecutive frames the vehicle must be detected in a lane (LEFT/RIGHT/CENTER)
# to confirm it has stabilized in that lane. Used to prevent noisy lane boundary triggers.
N_CONFIRM = 5

# Safe safety buffer time margin (in seconds) between the time-to-collision (TTC) with oncoming 
# vehicle and the time required to complete the overtaking maneuver. 
T_MARGIN = 3.0

# Time (in seconds) to wait after overtaking has finished before attempting to merge back
# to the original lane, ensuring enough buffer space is cleared.
RETURN_WAIT_TIME = 1.5

# ==============================================================================
# Host Vehicle Template Matching
# ==============================================================================

# Number of frames to use when matching templates for tracking the host vehicle's lane positioning.
HOST_TEMPLATE_FRAMES = 6

# Width in pixels of the host vehicle template cropped box.
HOST_TEMPLATE_W = 160

# Height in pixels of the host vehicle template cropped box.
HOST_TEMPLATE_H = 80

# Minimum threshold score for template matching (0.0 to 1.0).
# Scores above this value indicate a successful match.
TEMPLATE_MATCH_THRESHOLD = 0.35

# ==============================================================================
# Smoothing Parameters
# ==============================================================================

# Alpha smoothing factor for the Exponential Moving Average (EMA) distance filter.
# Must be between 0.0 and 1.0. Lower value means smoother but higher latency.
EMA_ALPHA = 0.2

# ==============================================================================
# Tracking Parameters
# ==============================================================================

# Maximum number of consecutive frames a locked target can be missing/undetected
# before the system considers it permanently lost and releases the target lock.
MAX_TARGET_MISSING = 12

# ==============================================================================
# Motion / Speed Parameters
# ==============================================================================

# Minimum optical flow shift in pixels to classify motion.
FLOW_SHIFT_PX_MIN = 3.0

# Threshold (in m/s) to classify if a tracked object is moving in the opposite direction.
# Speed calculations above this threshold indicate oncoming traffic.
OPPOSITE_SPEED_THRESHOLD = 8.0

# ==============================================================================
# Vehicle Heights (meters)
# ==============================================================================

# Reference physical height of vehicle groups in meters, used as ground-truth height 'H' 
# in the pinhole distance estimation camera model.
CLASS_HEIGHTS = {
    'motorcycle': 1.0,
    'bicycle': 1.0,
    'car': 1.4,
    'suv': 1.7,
    'van': 1.8,
    'truck': 3.0,
    'bus': 3.0,
    'unknown': 1.5
}

# ==============================================================================
# Vehicle Lengths (meters)
# ==============================================================================

# Reference physical length of vehicle groups in meters.
# Used for safe passing distance thresholds during the return/merging phase.
CLASS_LENGTHS = {
    'motorcycle': 2.0,
    'car': 4.5,
    'suv': 4.8,
    'van': 5.5,
    'truck': 12.0,
    'bus': 11.0,
}

# ==============================================================================
# YOLO Class Mapping
# ==============================================================================

# Maps YOLOv8 default dataset class name labels to our internal classification groups.
YOLO_TO_GROUP = {
    'motorbike': 'motorcycle',
    'motorcycle': 'motorcycle',
    'bicycle': 'bicycle',
    'bike': 'motorcycle',

    'car': 'car',
    'truck': 'truck',
    'bus': 'truck',
    'van': 'van',
    'suv': 'suv',

    'person': 'unknown'
}