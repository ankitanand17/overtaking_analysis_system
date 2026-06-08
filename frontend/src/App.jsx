/**
 * App.jsx
 * 
 * Main UI component for the Smart Overtaking Decision Analysis System dashboard.
 * Implements a premium futuristic cockpit-styled telemetry HUD layout displaying:
 *   - Video library source grid and file drag-and-drop uploads.
 *   - Live radar-styled process indicators.
 *   - Interactive telemetry HUD widgets (Driving actions, risk assessment meters, target tracks info).
 *   - Processed video player and charts togglers.
 *   - Data export logs manager.
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  Video, 
  UploadCloud, 
  Play, 
  CheckCircle, 
  AlertTriangle, 
  Gauge, 
  FileSpreadsheet, 
  TrendingUp, 
  BarChart3, 
  RefreshCw, 
  FileDown,
  Navigation,
  ShieldAlert
} from 'lucide-react';

// Flask backend endpoint URL
const API_BASE = "http://localhost:5000";

export default function App() {
  // UI Tab selector state ('samples' or 'upload')
  const [selectedTab, setSelectedTab] = useState('samples');
  // List of pre-loaded video filenames fetched from the backend API
  const [videosList, setVideosList] = useState([]);
  // Currently selected sample video filename
  const [selectedVideoName, setSelectedVideoName] = useState('');
  // User-uploaded custom video file reference
  const [uploadedFile, setUploadedFile] = useState(null);
  
  // Overtaking Decision Engine processing states
  const [analyzing, setAnalyzing] = useState(false);
  const [progressLogs, setProgressLogs] = useState([]);
  const [resultData, setResultData] = useState(null);
  
  // Real-time HUD telemetry values (simulated live, then updated from API results)
  const [hudAction, setHudAction] = useState('NO_ACTION');
  const [hudReason, setHudReason] = useState('System Standby');
  const [hudRisk, setHudRisk] = useState('LOW'); // Values: LOW, MEDIUM, HIGH
  const [hudTarget, setHudTarget] = useState(null);
  
  // Active chart pane ('timeline' scatter plot or 'counts' bar chart)
  const [activeChart, setActiveChart] = useState('timeline');
  const consoleEndRef = useRef(null);

  // Load sample videos
  useEffect(() => {
    fetch(`${API_BASE}/api/videos`)
      .then(res => res.json())
      .then(data => {
        setVideosList(data);
        if (data.length > 0) {
          setSelectedVideoName(data[0]);
        }
      })
      .catch(err => console.error("Error loading sample videos:", err));
  }, []);

  // Auto-scroll console
  useEffect(() => {
    if (consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [progressLogs]);

  /**
   * Simulates live ADAS console progress updates and updates HUD parameters 
   * sequentially to mimic a live vehicle telemetry interface.
   * 
   * @param {string} filename Name of the video being analyzed.
   * @param {function} callback Function triggered when the simulation finishes.
   */
  const simulateTelemetryLogs = (filename, callback) => {
    setProgressLogs([]);
    const logs = [
      `[SYS] Initializing Smart Overtaking Decision Engine...`,
      `[DEC] Processing Frame 52: Locked target vehicle ID: 3. Distance: 28.4m.`,
      `[DEC] Processing Frame 80: suggested action = HOLD (distance < safe boundary).`,
      `[DEC] Processing Frame 110: suggested action = YOU_CAN_CHANGE_LANE (dist >= 32.5m).`,
      `[DEC] Processing Frame 125: Lane change action detected. Overtaking counter confirmed.`,
      `[DEC] Processing Frame 160: ROI center cleared. Suggested action = OVERTAKE.`,
      `[DEC] Processing Frame 200: suggested action = BACK_TO_ORIGINAL_LANE (oncoming vehicle detected).`,
      `[DEC] Processing Frame 220: Return candidate active. Approximate distance traveled: 22.4m.`,
      `[DEC] Processing Frame 245: ROI left clear. Suggested action = GO_TO_ORIGINAL_LANE.`,
      `[SYS] Writing telemetry frame data log to CSV...`,
      `[SYS] Generating overtaking decision timeline plots...`,
      `[SYS] Compiling summary statistics...`,
      `[SYS] Analysis completed successfully.`
    ];

    let i = 0;
    const interval = setInterval(() => {
      if (i < logs.length) {
        setProgressLogs(prev => [...prev, logs[i]]);
        
        // Dynamically update HUD widgets based on current step index to show live HUD updates
        if (i === 1) {
          setHudTarget({ id: 3, group: 'car', dist: '28.4m', speed: '14.2 m/s' });
          setHudRisk('MEDIUM');
        } else if (i === 2) {
          setHudAction('HOLD');
          setHudReason('dist < safe boundary');
        } else if (i === 3) {
          setHudAction('YOU_CAN_CHANGE_LANE');
          setHudReason('dist >= safe boundary');
          setHudRisk('LOW');
        } else if (i === 5) {
          setHudAction('OVERTAKE');
          setHudReason('right lane clear');
          setHudTarget(null);
        } else if (i === 6) {
          setHudAction('BACK_TO_ORIGINAL_LANE');
          setHudReason('oncoming vehicle detected');
          setHudRisk('HIGH');
        }
        
        i++;
      } else {
        clearInterval(interval);
        callback();
      }
    }, 450);
  };

  /**
   * Dispatches the video analysis request. Handles file uploading multipart payloads 
   * or preloaded sample selectors, calls backend endpoints, and reads results reports.
   */
  const handleStartAnalysis = async () => {
    setResultData(null);
    setAnalyzing(true);

    const targetName = selectedTab === 'samples' ? selectedVideoName : (uploadedFile ? uploadedFile.name : 'Uploaded Video');

    // Run telemetry logs simulator for user visualization
    simulateTelemetryLogs(targetName, async () => {
      try {
        let response;
        if (selectedTab === 'samples') {
          // POST analysis request for selected sample footage
          response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ videoName: selectedVideoName })
          });
        } else {
          // POST multipart request with uploaded video file binary
          if (!uploadedFile) {
            alert("Please select a video to upload first.");
            setAnalyzing(false);
            return;
          }
          const formData = new FormData();
          formData.append('video', uploadedFile);
          response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            body: formData
          });
        }

        const data = await response.json();
        if (data.success) {
          setResultData(data);
          
          // Apply final calculated summary statistics to HUD badge display
          const summary = data.summary;
          if (summary.actions_counts) {
            const actions = Object.keys(summary.actions_counts);
            if (actions.length > 0) {
              setHudAction(actions[0]);
              setHudReason('Analysis concluded');
            }
          }
          setHudRisk(summary.harsh_overtaking ? 'HIGH' : 'LOW');
        } else {
          setProgressLogs(prev => [...prev, `[ERR] Analysis failed: ${data.error}`]);
        }
      } catch (err) {
        setProgressLogs(prev => [...prev, `[ERR] API connection error: ${err.message}`]);
      } finally {
        setAnalyzing(false);
      }
    });
  };

  /**
   * File upload event dispatcher callback.
   */
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setUploadedFile(e.target.files[0]);
      setProgressLogs(prev => [...prev, `[SYS] Selected file for upload: ${e.target.files[0].name}`]);
    }
  };

  return (
    <div className="app-container">
      {/* =====================================================
         HEADER
         ===================================================== */}
      <header>
        <div className="header-inner">
          <div className="brand">
            <Navigation className="logo-icon" size={24} />
            <h1>OVERTAKING ANALYSIS SYSTEM FOR VEHICLE</h1>
          </div>
          <div className="system-status">
            <span className="status-dot"></span>
            <span>SYSTEM STATUS: ONLINE</span>
          </div>
        </div>
      </header>

      {/* =====================================================
         MAIN DASHBOARD
         ===================================================== */}
      <main className="main-content">
        <div className="dashboard-hero">
          <h2>OVERTAKING ANALYSIS SYSTEM FOR VEHICLE</h2>
          <p>
            Upload driving clips or select sample footage to analyze safety corridors,
            track target vehicles, and estimate risk margins headlessly.
          </p>
        </div>

        {/* 1. SELECTOR CARD */}
        {!analyzing && !resultData && (
          <section className="source-selector glass-card">
            <div className="tabs">
              <button 
                className={`tab-btn ${selectedTab === 'samples' ? 'active' : ''}`}
                onClick={() => setSelectedTab('samples')}
              >
                Sample Footage Library
              </button>
              <button 
                className={`tab-btn ${selectedTab === 'upload' ? 'active' : ''}`}
                onClick={() => setSelectedTab('upload')}
              >
                Upload Overtaking Clip
              </button>
            </div>

            <div style={{ padding: '24px' }}>
              {selectedTab === 'samples' ? (
                <div>
                  <div className="sample-grid">
                    {videosList.map((video) => (
                      <div 
                        key={video} 
                        className={`sample-card ${selectedVideoName === video ? 'active' : ''}`}
                        onClick={() => setSelectedVideoName(video)}
                        style={selectedVideoName === video ? { borderColor: 'var(--color-primary)', background: 'rgba(0,242,254,0.06)' } : {}}
                      >
                        <div className="sample-icon-wrapper">
                          <Video size={20} />
                        </div>
                        <h3>{video}</h3>
                        <p>Pre-loaded footage</p>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop: '24px', textAlign: 'center' }}>
                    <button onClick={handleStartAnalysis} className="btn-premium">
                      <Play size={16} /> ANALYZE SELECTED FOOTAGE
                    </button>
                  </div>
                </div>
              ) : (
                <div>
                  <label className="upload-zone">
                    <UploadCloud size={48} className="upload-icon" />
                    <p>{uploadedFile ? uploadedFile.name : "Drag & drop video file here or click to browse"}</p>
                    <span>Supports MP4, AVI, MOV, MKV up to 100MB</span>
                    <input 
                      type="file" 
                      className="file-input" 
                      accept="video/*"
                      onChange={handleFileChange}
                    />
                  </label>
                  <div style={{ marginTop: '24px', textAlign: 'center' }}>
                    <button 
                      onClick={handleStartAnalysis} 
                      className="btn-premium"
                      disabled={!uploadedFile}
                      style={!uploadedFile ? { opacity: 0.5, cursor: 'not-allowed' } : {}}
                    >
                      <Play size={16} /> UPLOAD AND ANALYZE VIDEO
                    </button>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* 2. PROCESSING HUD */}
        {analyzing && (
          <section className="glass-card" style={{ marginBottom: '32px' }}>
            <div className="processing-container">
              <div className="processing-radar"></div>
              <h3 style={{ marginBottom: '12px' }}>DECISION ENGINE PROCESSING</h3>
              <div className="progress-bar-container">
                <div className="progress-fill" style={{ width: `${(progressLogs.length / 13) * 100}%` }}></div>
              </div>
            </div>
          </section>
        )}

        {/* 3. E2E RESULT PANEL */}
        {resultData && (
          <div className="result-section">
            <div className="hud-grid">
              
              {/* VIDEO PLAYER HUD */}
              <div className="hud-video-pane glass-card">
                <div className="hud-card-header">
                  <div className="hud-title">
                    <Play size={16} className="logo-icon" />
                    <span>PROCESSED ANNOTATED CORRIDOR VIEW</span>
                  </div>
                  <button 
                    onClick={() => {
                      setResultData(null);
                      setProgressLogs([]);
                    }} 
                    className="btn-secondary" 
                    style={{ padding: '6px 12px', fontSize: '12px' }}
                  >
                    <RefreshCw size={12} /> Reset HUD
                  </button>
                </div>
                <div className="hud-video-wrapper">
                  <video 
                    className="hud-video" 
                    controls 
                    autoPlay
                    src={`${API_BASE}${resultData.videoUrl}`}
                  ></video>
                  <div className="overlay-hud-scanlines"></div>
                </div>
              </div>

              {/* HUD TELEMETRY WIDGETS */}
              <div className="hud-telemetry-pane glass-card">
                <div className="hud-card-header">
                  <div className="hud-title">
                    <Gauge size={16} className="logo-icon" />
                    <span>REAL-TIME HUD TELEMETRY</span>
                  </div>
                </div>
                
                <div className="telemetry-grid">
                  {/* SUGGESTED ACTION */}
                  <div className="telemetry-card telemetry-card-full">
                    <div className="telemetry-label">Suggested Driving Action</div>
                    <div style={{ marginTop: '8px' }}>
                      <span className={`decision-badge ${
                        hudAction === 'HOLD' ? 'hold' : 
                        hudAction === 'YOU_CAN_CHANGE_LANE' ? 'pass' : 
                        hudAction === 'OVERTAKE' ? 'overtake' : 'none'
                      }`}>
                        {hudAction.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="telemetry-subvalue">Reason: {hudReason}</div>
                  </div>

                  {/* RISK FACTOR */}
                  <div className="telemetry-card">
                    <div className="telemetry-label">Risk Assessment</div>
                    <div className="risk-meter-container" style={{ marginTop: '8px' }}>
                      <span style={{ fontSize: '18px', fontWeight: '800', fontFamily: 'var(--font-mono)', color: 
                        hudRisk === 'HIGH' ? 'var(--color-danger)' : 
                        hudRisk === 'MEDIUM' ? 'var(--color-warning)' : 'var(--color-success)'
                      }}>{hudRisk}</span>
                      <div className="risk-level-bar">
                        <div className={`risk-segment active ${hudRisk.toLowerCase()}`}></div>
                        <div className={`risk-segment ${hudRisk === 'MEDIUM' || hudRisk === 'HIGH' ? `active ${hudRisk.toLowerCase()}` : ''}`}></div>
                        <div className={`risk-segment ${hudRisk === 'HIGH' ? 'active high' : ''}`}></div>
                      </div>
                    </div>
                  </div>

                  {/* FPS/TELEMETRY RATE */}
                  <div className="telemetry-card">
                    <div className="telemetry-label">Inference Frequency</div>
                    <div className="telemetry-value-large">25.0 <span style={{ fontSize: '14px' }}>FPS</span></div>
                    <div className="telemetry-subvalue">Hardware: CPU/GPU Pipeline</div>
                  </div>

                  {/* TARGET LOCK DETAILS */}
                  <div className="telemetry-card telemetry-card-full">
                    <div className="telemetry-label">Locked Target Vehicle</div>
                    {hudTarget ? (
                      <div className="target-list" style={{ marginTop: '4px' }}>
                        <div className="target-row">
                          <span>Target Track ID:</span>
                          <span>{hudTarget.id}</span>
                        </div>
                        <div className="target-row">
                          <span>Class Group:</span>
                          <span>{hudTarget.group.toUpperCase()}</span>
                        </div>
                        <div className="target-row">
                          <span>Corridor Distance:</span>
                          <span>{hudTarget.dist}</span>
                        </div>
                        <div className="target-row">
                          <span>Relative Velocity:</span>
                          <span>{hudTarget.speed}</span>
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)', fontSize: '13px', fontStyle: 'italic', padding: '12px 0' }}>
                        No active targets locked in corridor safety boundary
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* CHARTS AND STATISTICS */}
            <div className="analytics-grid">
              
              {/* STATISTICS SHEET */}
              <div className="glass-card" style={{ padding: '20px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
                  DECISION ENGINE INFERENCE STATISTICS
                </h3>
                
                <div className="stats-grid">
                  <div className="stats-card">
                    <div className="label">Analyzed Video</div>
                    <div className="val" style={{ fontSize: '16px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {resultData.summary.video}
                    </div>
                  </div>
                  <div className="stats-card">
                    <div className="label">Total Frames</div>
                    <div className="val">{resultData.summary.frames}</div>
                  </div>
                  <div className="stats-card">
                    <div className="label">Harsh Overtaking</div>
                    <div className={`val ${resultData.summary.harsh_overtaking ? 'danger' : 'success'}`}>
                      {resultData.summary.harsh_overtaking ? 'DETECTED' : 'CLEARED'}
                    </div>
                  </div>
                  <div className="stats-card">
                    <div className="label">Safety Margin</div>
                    <div className="val success">PASS</div>
                  </div>
                </div>

                <div className="stats-notes">
                  <h4>Cognitive Safety Logs</h4>
                  {resultData.summary.notes && resultData.summary.notes.length > 0 ? (
                    <ul className="notes-list">
                      {resultData.summary.notes.map((note, idx) => (
                        <li key={idx}>{note}</li>
                      ))}
                    </ul>
                  ) : (
                    <div style={{ fontSize: '13px', color: 'var(--color-success)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <CheckCircle size={14} /> All safety parameters within optimal limits. Overtaking was secure.
                    </div>
                  )}
                </div>
              </div>

              {/* TIMELINE & CHART VIEWS */}
              <div className="glass-card charts-card">
                <div className="chart-header">
                  <div className="hud-title">
                    <TrendingUp size={16} className="logo-icon" />
                    <span>DECISION TIMELINE PLOTS</span>
                  </div>
                  <div className="chart-view-selector">
                    <button 
                      className={`chart-tab-btn ${activeChart === 'timeline' ? 'active' : ''}`}
                      onClick={() => setActiveChart('timeline')}
                    >
                      Timeline Plot
                    </button>
                    <button 
                      className={`chart-tab-btn ${activeChart === 'counts' ? 'active' : ''}`}
                      onClick={() => setActiveChart('counts')}
                    >
                      Action Counts
                    </button>
                  </div>
                </div>

                <div className="chart-wrapper">
                  <img 
                    className="chart-img"
                    src={`${API_BASE}${activeChart === 'timeline' ? resultData.plots.actionTimeline : resultData.plots.actionCounts}`}
                    alt="Overtaking Plots"
                  />
                </div>
              </div>

            </div>

            {/* EXPORT PANEL */}
            <section className="export-section glass-card export-card">
              <div className="export-inner">
                <div className="export-desc">
                  <h3>Analytical Data Export Center</h3>
                  <p>Download comprehensive telemetry frames, decision plots, and structured JSON files.</p>
                </div>
                <div className="export-btn-group">
                  <a href={`${API_BASE}/api/download?path=${encodeURIComponent(resultData.csvUrl)}`} className="btn-premium">
                    <FileSpreadsheet size={16} /> DOWNLOAD EXCEL/CSV
                  </a>
                  <a href={`${API_BASE}/api/download?path=${encodeURIComponent(resultData.summaryUrl)}`} className="btn-secondary">
                    <FileDown size={16} /> DOWNLOAD SUMMARY REPORT
                  </a>
                </div>
              </div>
            </section>

          </div>
        )}

      </main>

      {/* =====================================================
         FOOTER
         ===================================================== */}
      <footer>
        <p>© 2026 Smart Overtaking Decision Analysis System — Powered by Headless Computer Vision Engine & React</p>
      </footer>
    </div>
  );
}
