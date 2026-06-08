# Project Restructuring & React ADAS Dashboard Implementation Guide

This guide details the structural reorganization, backend modernization, and React frontend implementation completed for the Smart Overtaking Decision Analysis System.

---

## 📁 Decoupled Folder Architecture

The codebase has been refactored from a single-folder hybrid structure into a decoupled architecture separating the machine learning backend and the interactive client frontend.

```
workspace/
  ├── backend/                     # Machine Learning Server (Flask API)
  │     ├── app.py                 # REST API endpoints & CORS handler
  │     ├── overtake_analyzer.py   # Headless-configured Overtaking Analyzer
  │     ├── yolov8n.pt             # YOLOv8 weights file
  │     ├── config/                # Settings & vehicle constraints
  │     ├── decision/              # Target lock & merging decision modules
  │     ├── detection/             # Object detectors (YOLO / Fallback)
  │     ├── tracking/              # SORT track association
  │     ├── utils/                 # Geometry & distance calculation helper
  │     ├── visualization/         # Text & bounding box draw modules
  │     ├── output/                # Plot generation & CSV logging
  │     ├── videos/                # Directory containing sample video files
  │     └── static/                # Directory for uploaded and processed results
  │
  └── frontend/                    # Client Dashboard (Vite + React)
        ├── package.json           # React dependencies & scripts
        ├── vite.config.js         # Local dev proxy config
        ├── index.html             # Client entry point
        └── src/
              ├── main.jsx         # DOM Mounting script
              ├── index.css        # Futuristic cockpit layout styling
              └── App.jsx          # Telemetry HUD dashboard component
```

---

## 📋 Implementation Plan Followed

The restructuring was executed according to the approved implementation plan:

1. **Backend Separation**: Move all existing computer vision modules, configs, and weights into the `backend/` directory to prevent directory conflicts.
2. **Headless OpenCV Safety**: Modify the video processor `process_video` function in `overtake_analyzer.py` to bypass GUI elements (`cv2.imshow` and `cv2.waitKey`) under server mode, preventing deployment crashes.
3. **REST API Modernization**: Update the Flask server `app.py` to communicate exclusively via JSON REST payloads. Implement CORS handling to permit communication with Vite on separate ports.
4. **Futuristic React Dashboard**: Scafold a Vite-React project under `frontend/` and configure a telemetry dashboard styled with clean cockpit aesthetics and smooth micro-animations.
5. **E2E Integration & Verification**: Validate endpoints and launch local server pipelines.

---

## 🛠 Exactly What Has Been Done

### 1. Headless OpenCV Processing
- **File**: [overtake_analyzer.py](file:///c:/Users/getan/Desktop/Final/backend/overtake_analyzer.py)
- **Modifications**: Added a boolean `show_preview` parameter (default: `False`). Visual windows (`cv2.imshow`) and keyboard listening loops (`cv2.waitKey`) are only initialized when `show_preview` is explicitly passed as `True`.
- **Impact**: Server execution no longer depends on visual environment resources, resolving Flask connection drops.

### 2. Modernized REST API App
- **File**: [app.py](file:///c:/Users/getan/Desktop/Final/backend/app.py)
- **Modifications**: 
  - Exposed `GET /api/videos` to dynamically inspect and list sample clips.
  - Exposed `POST /api/analyze` to trigger the AI analysis headlessly and return JSON results containing metrics, parsed summaries, and file download pointer URLs.
  - Exposed `/api/static/<path>` to serve the generated annotated videos, CSV logs, and timeline plots.
  - Exposed `/api/download` to securely retrieve logs without path traversal risks.
  - Appended CORS compliance headers directly via a custom `@app.after_request` filter.

### 3. State-of-the-Art React ADAS Cockpit HUD
- **Files**: [App.jsx](file:///c:/Users/getan/Desktop/Final/frontend/src/App.jsx) and [index.css](file:///c:/Users/getan/Desktop/Final/frontend/src/index.css)
- **Features Completed**:
  - **Video Source Selector**: Beautiful selection grids for pre-loaded videos and dropzones for local video uploads.
  - **Active HUD metrics**: Telemetry badges displaying recommended safe actions (Hold, Overtake, Prepare to merge), real-time risk indicators (Low, Med, High), and locked target information.
  - **Live ADAS Logs terminal**: Telemetry progress monitor outputting live tracking and distance boundaries.
  - **Plot toggler card**: Interactive tab frames rendering active timeline scatter charts and action counts.
  - **Data export Center**: Quick-action buttons to download frame spreadsheets and JSON parameters directly from the REST engine.

---

## 🚀 How to Run the Project

Ensure you are in the workspace root directory before proceeding.

### 1. Start the Flask REST Backend
Launch the Python server utilizing your GPU virtual environment:
```bash
# In the workspace root folder
venv_gpu\Scripts\python.exe backend/app.py
```
*The server will boot in debug mode on `http://127.0.0.1:5000`.*

### 2. Start the Vite-React Frontend
Install client dependencies and launch the dev environment:
```bash
# Navigate to the frontend directory
cd frontend
npm install
npm run dev
```
*The Vite engine will load the cockpit interface on `http://localhost:5173/`.*
