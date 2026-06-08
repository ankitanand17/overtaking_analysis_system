# Smart Overtaking Decision Analysis System - Backend REST API

This is the machine learning backend and decision-making server for the Smart Overtaking Decision Analysis System. Built with **Flask** and integrated with **YOLOv8** (deep learning object detection) and the **SORT (Simple Online and Realtime Tracking)** algorithm, this server processes road footage, estimates vehicle boundaries, tracks vehicles, and provides driving safety recommendations in real-time.

---

## 📁 Backend Directory Architecture

```txt
backend/
├── app.py                      # Flask REST API endpoints and CORS configuration
├── overtake_analyzer.py        # Core video processor orchestrating detection and tracking loops
├── calibrate_and_test_distance.py  # Interactive distance sensor calibration script
├── plot_distance_errors.py     # Accuracy evaluation chart generator (RMSE/MAE)
├── config/
│   └── settings.py             # Reference heights, lengths, and safety margins configuration
├── decision/
│   ├── target_lock.py          # Tracks nearest targets, handles occlusion recovery
│   └── overtaking_logic.py     # ADAS driving action recommendations state machine
├── detection/
│   └── yolo_detector.py        # YOLOv8 object detector and classical CV fallback
├── tracking/
│   ├── sort.py                 # Kalman Box Filter tracking algorithm
│   └── tracker_utils.py        # Associates SORT tracks to YOLO detections
├── utils/
│   └── helpers.py              # Camera pinhole formulas, EMA smoothing, and math utilities
├── output/
│   ├── csv_logger.py           # Frame-by-frame telemetry log exporter (CSV)
│   ├── plots.py                # Action timelines and counts scatter/bar chart generator
│   └── summary.py              # Statistics compile and harsh overtaking detector
├── static/                     # Processed video runs and temporary uploads directories
├── videos/                     # Sample footage library
└── yolov8n.pt                  # YOLOv8 pre-trained weights
```

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10+
- PyTorch (configured for GPU/CUDA acceleration if compatible graphics card is available)
- OpenCV (headless-compatible)

### Running the API Server
Execute the server using the workspace GPU virtual environment:
```bash
# From the project root folder
venv_gpu\Scripts\python.exe backend/app.py
```
*The Flask application will mount in debug mode on `http://127.0.0.1:5000`.*

---

## 🔌 API Endpoints Documentation

All requests interact with the prefix `/api/`.

### 1. List Sample Video Library
- **Endpoint**: `GET /api/videos`
- **Description**: Scans the pre-loaded video library folder and returns all supported video filenames.
- **Response Format**: `JSON` list of strings:
  ```json
  [
    "sample_highway.mp4",
    "sample_overtaking.avi"
  ]
  ```

### 2. Upload and Analyze Video
- **Endpoint**: `POST /api/analyze`
- **Description**: Triggers a computer vision processing run for a custom uploaded video file or a selected sample video name.
- **Request Format**: 
  - **Uploaded File**: Multipart form-data with the key `video`.
  - **Sample Selector**: JSON payload: `{"videoName": "sample_highway.mp4"}`.
- **Response Payload Example**:
  ```json
  {
    "success": true,
    "videoName": "sample_highway.mp4",
    "isSample": true,
    "summary": {
      "video": "sample_highway.mp4",
      "frames": 245,
      "actions_counts": {
        "HOLD": 82,
        "YOU_CAN_CHANGE_LANE": 120,
        "OVERTAKE": 43
      },
      "harsh_overtaking": false,
      "notes": []
    },
    "videoUrl": "/api/static/results/sample_highway/annotated_video.mp4",
    "csvUrl": "/api/static/results/sample_highway/frames_data.csv",
    "summaryUrl": "/api/static/results/sample_highway/summary.json",
    "plots": {
      "actionTimeline": "/api/static/results/sample_highway/plots/action_timeline.png",
      "actionCounts": "/api/static/results/sample_highway/plots/action_counts.png"
    }
  }
  ```

### 3. Secure File Retrieval
- **Endpoint**: `GET /api/download`
- **Description**: Securely serves generated CSV tables or JSON summaries as attachments. Incorporates route verification checks to prevent directory traversal.
- **Parameters**: `path` (the local URL returned by the analysis payload).
- **Example Usage**: `http://localhost:5000/api/download?path=/api/static/results/sample_highway/frames_data.csv`

---

## 📏 Distance Camera Sensor Calibration

The backend includes calibration scripts to compute and evaluate the camera focal length pixels parameter used in pinhole model distance estimations.

### 1. Perform Distance Calibration
To run interactive calibration or capture test samples:
```bash
python backend/calibrate_and_test_distance.py --source 0 --model yolov8n.pt
```
- **Controls (CV window active)**:
  - `c`: Capture calibration frame (enter ground-truth width/distance to calculate focal length).
  - `m`: Toggle manual ROI selection (use mouse selector box if YOLO fails).
  - `t`: Save current frame estimation to log (enter true distance).
  - `q`: Save CSV log and quit.

### 2. Generate Accuracy Validation Report
To generate absolute and relative error plots from the recorded test samples:
```bash
python backend/plot_distance_errors.py
```
- **Output**: Generates double-pane accuracy graph saved to `backend/distance_calibration_metrics.png`.
