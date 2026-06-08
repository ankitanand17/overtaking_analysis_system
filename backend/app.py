"""
app.py

This is the Flask REST API server for the Smart Overtaking System. It acts as the gateway 
between the interactive React dashboard frontend and the Python computer vision AI engine. 
It supports video uploading, running analysis pipelines headlessly, listing sample libraries, 
serving static plots/videos, and downloading files securely.
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory, send_file
from overtake_analyzer import process_video

# ==============================================================================
# FLASK REST API SERVER
# ==============================================================================

app = Flask(__name__)

# Base workspace paths relative to this backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory for temporary uploaded user clips
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
# Directory for generated timeline charts, logs, and processed clips
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
# Directory containing pre-loaded sample footage
VIDEOS_FOLDER = os.path.join(BASE_DIR, "videos")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==============================================================================
# CORS SUPPORT
# ==============================================================================

@app.after_request
def add_cors_headers(response):
    """
    Appends Cross-Origin Resource Sharing (CORS) compliance headers to allow 
    communication with the frontend dev server running on a separate port.
    """
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS,PUT,DELETE")
    return response

# ==============================================================================
# STATIC ASSETS SERVING
# ==============================================================================

@app.route("/api/static/<path:filename>")
def serve_static(filename):
    """
    Serves static analysis results (processed videos, Matplotlib charts, logs)
    to the frontend player.
    """
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)

# ==============================================================================
# LIST SAMPLE VIDEOS
# ==============================================================================

@app.route("/api/videos", methods=["GET"])
def list_videos():
    """
    Lists all available video files in the sample videos folder.
    Used by the dashboard footage library tab.

    Returns:
        JSON response: List of video filenames.
    """
    try:
        if not os.path.exists(VIDEOS_FOLDER):
            return jsonify([])
        files = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================================================================
# UPLOAD AND ANALYZE VIDEO
# ==============================================================================

@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """
    Main API endpoint. Accepts a video file upload or a selected sample filename.
    Triggers the headless overtaking computer vision analyzer pipeline and returns 
    the parsed results, charts, and downloadable URLs.

    Returns:
        JSON response containing the analysis summary, plots links, and annotated video URL.
    """
    try:
        is_sample = False
        video_path = None
        filename = None

        # Case 1: Video File Upload from dashboard dropzone
        if "video" in request.files:
            file = request.files["video"]
            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400
            
            filename = file.filename
            video_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(video_path)
        
        # Case 2: Select Pre-loaded Sample Video from library list
        else:
            data = request.json or {}
            video_name = data.get("videoName")
            if not video_name:
                return jsonify({"error": "No video uploaded or sample video selected"}), 400
            
            filename = video_name
            video_path = os.path.join(VIDEOS_FOLDER, video_name)
            if not os.path.exists(video_path):
                return jsonify({"error": f"Sample video '{video_name}' not found"}), 404
            is_sample = True

        # Unique run subfolder based on video base name to prevent cross-run overwriting
        video_base_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(RESULT_FOLDER, video_base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Execute computer vision algorithm headlessly (no windows, no blocking UI loops)
        result = process_video(
            input_path=video_path,
            output_dir=output_dir,
            save_video=True,
            show_preview=False
        )

        # Read the generated summary stats file
        summary_data = {}
        summary_path = result.get("summary_path")
        if summary_path and os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary_data = json.load(f)

        # Build final response payload
        response_data = {
            "success": True,
            "videoName": filename,
            "isSample": is_sample,
            "summary": summary_data,
            "videoUrl": f"/api/static/results/{video_base_name}/annotated_video.mp4",
            "csvUrl": f"/api/static/results/{video_base_name}/frames_data.csv",
            "summaryUrl": f"/api/static/results/{video_base_name}/summary.json",
            "plots": {
                "actionTimeline": f"/api/static/results/{video_base_name}/plots/action_timeline.png",
                "actionCounts": f"/api/static/results/{video_base_name}/plots/action_counts.png"
            }
        }
        
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==============================================================================
# SECURE DIRECT DOWNLOAD
# ==============================================================================

@app.route("/api/download", methods=["GET"])
def download_file():
    """
    Downloads log files or summary reports securely. Includes validation 
    checks to prevent directory traversal attacks.
    """
    path_param = request.args.get("path")
    if not path_param:
        return jsonify({"error": "Path parameter is required"}), 400
    
    # Strip any leading "/api/" or "api/" prefix to map to local static folder
    clean_path = path_param
    if clean_path.startswith("/api/"):
        clean_path = clean_path[5:]
    elif clean_path.startswith("api/"):
        clean_path = clean_path[4:]
    
    # Resolve absolute paths and validate they are contained within backend directory
    resolved_path = os.path.abspath(os.path.join(BASE_DIR, clean_path.lstrip("/\\")))
    if not resolved_path.startswith(os.path.abspath(BASE_DIR)):
        return jsonify({"error": "Access unauthorized"}), 403
        
    if not os.path.exists(resolved_path):
        return jsonify({"error": "File not found"}), 404
        
    return send_file(resolved_path, as_attachment=True)

# ==============================================================================
# START SERVER
# ==============================================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )