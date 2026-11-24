"""
Standalone App-Mode Live Cockpit Window (Zero Address Bar).
Launches an ultra-fast multi-threaded streaming server and opens Chrome/Edge in --app mode,
providing a clean native desktop application window with AAA-tier Cyber Robotics telemetry.
"""
import os
import sys
import time
import json
import subprocess
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np
import gymnasium as gym
import envs
from stable_baselines3 import PPO
from render.visual_engine import VisualEngine

# Global simulation state & frame buffer
latest_jpeg = None
frame_lock = threading.Lock()
running = True
camera_mode = "front_overview"
speed_scale = 1.0
is_paused = False
force_reset = False

current_telemetry = {
    "step": 0,
    "phase": 0,
    "phase_name": "PHASE 0: SECURE GRASP",
    "speed": 0.0,
    "dist": 0.0,
    "reward": 0.0,
    "contact": False,
    "is_success": False,
    "camera": "front_overview",
    "speed_scale": 1.0,
    "paused": False
}
telemetry_lock = threading.Lock()

VALID_CAMERAS = {
    "front_overview": "Front Studio",
    "side_cinematic": "Side Cinematic",
    "top_tactical": "Top Tactical",
    "left_wrist_cam": "Left Wrist",
    "right_wrist_cam": "Right Wrist"
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ALOHA 14-DOF Bimanual Dynamic Toss & Catch // Mission Control Cockpit</title>
  <style>
    :root {
      --bg-base: #06080d;
      --bg-panel: rgba(13, 17, 24, 0.88);
      --bg-header: rgba(9, 13, 19, 0.95);
      --border-subtle: #1e293b;
      --border-glow: rgba(0, 240, 255, 0.35);
      --neon-cyan: #00f0ff;
      --neon-green: #00ff88;
      --neon-magenta: #ff007f;
      --neon-amber: #ffb700;
      --text-main: #f1f5f9;
      --text-muted: #8492a6;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body {
      background: var(--bg-base);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: 100vh;
      user-select: none;
    }

    /* =========================================================================
       TOP HEADER: Mission Status & Realtime Telemetry HUD
       ========================================================================= */
    header {
      background: var(--bg-header);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 0 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 52px;
      z-index: 100;
    }

    .brand-section {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 7px;
      background: rgba(0, 255, 136, 0.1);
      border: 1px solid rgba(0, 255, 136, 0.3);
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 700;
      color: var(--neon-green);
      letter-spacing: 0.8px;
    }

    .live-dot {
      width: 8px;
      height: 8px;
      background: var(--neon-green);
      border-radius: 50%;
      box-shadow: 0 0 10px var(--neon-green);
      animation: pulse 1.4s ease-in-out infinite;
    }

    @keyframes pulse {
      0% { transform: scale(0.9); opacity: 0.5; }
      50% { transform: scale(1.15); opacity: 1; }
      100% { transform: scale(0.9); opacity: 0.5; }
    }

    .brand-title {
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.8px;
      background: linear-gradient(90deg, #00f0ff, #38bdf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    /* Telemetry HUD Readouts */
    .telemetry-strip {
      display: flex;
      align-items: center;
      gap: 20px;
    }

    .tel-item {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 2px;
    }

    .tel-label {
      font-size: 9px;
      font-weight: 700;
      color: var(--text-muted);
      letter-spacing: 0.6px;
      text-transform: uppercase;
    }

    .tel-val {
      font-size: 12px;
      font-weight: 700;
      font-family: 'Consolas', 'Courier New', monospace;
      color: #fff;
    }

    .val-cyan { color: var(--neon-cyan); }
    .val-green { color: var(--neon-green); }
    .val-magenta { color: var(--neon-magenta); }
    .val-amber { color: var(--neon-amber); }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    /* =========================================================================
       MAIN VIEWPORT: Centered 16:9 Arena Canvas
       ========================================================================= */
    #viewport-container {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      background: radial-gradient(circle at center, #0a0e16 0%, #030406 100%);
      position: relative;
      padding: 12px 16px 84px 16px; /* Bottom padding reserves room for floating dock */
      overflow: hidden;
    }

    .viewport-frame {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid rgba(0, 240, 255, 0.22);
      box-shadow: 0 15px 50px rgba(0, 0, 0, 0.9), 0 0 25px rgba(0, 240, 255, 0.08);
      max-width: 100%;
      max-height: 100%;
      aspect-ratio: 16 / 9;
    }

    #stream-img {
      display: block;
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 140px);
      object-fit: contain;
    }

    .cam-overlay-tag {
      position: absolute;
      top: 14px;
      left: 14px;
      background: rgba(8, 12, 18, 0.8);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(0, 240, 255, 0.35);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 800;
      color: var(--neon-cyan);
      letter-spacing: 0.8px;
      pointer-events: none;
    }

    /* =========================================================================
       BOTTOM CONTROL DOCK: Ergonomic Floating Glassmorphism Toolbar
       ========================================================================= */
    .bottom-dock {
      position: fixed;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(13, 18, 27, 0.9);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.75), 0 0 20px rgba(0, 240, 255, 0.1);
      border-radius: 12px;
      padding: 8px 18px;
      display: flex;
      align-items: center;
      gap: 18px;
      z-index: 100;
    }

    .dock-divider {
      width: 1px;
      height: 24px;
      background: rgba(255, 255, 255, 0.12);
    }

    .dock-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .dock-label {
      font-size: 10px;
      font-weight: 800;
      color: var(--text-muted);
      margin-right: 4px;
      letter-spacing: 0.6px;
      text-transform: uppercase;
    }

    .btn {
      background: #141b26;
      border: 1px solid #243044;
      color: #cbd5e1;
      padding: 6px 12px;
      font-size: 11px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .btn:hover {
      background: #1e293b;
      color: #fff;
      border-color: var(--neon-cyan);
      box-shadow: 0 0 10px rgba(0, 240, 255, 0.25);
    }

    .btn.active {
      background: rgba(0, 240, 255, 0.15);
      color: var(--neon-cyan);
      border-color: var(--neon-cyan);
      box-shadow: 0 0 12px rgba(0, 240, 255, 0.35);
      font-weight: 700;
    }

    .btn-action {
      background: #1a1e28;
      border-color: #334155;
    }
    .btn-action:hover {
      border-color: var(--neon-amber);
      color: var(--neon-amber);
      box-shadow: 0 0 10px rgba(255, 183, 0, 0.3);
    }

    .btn-danger {
      background: #200f14;
      border-color: #551925;
      color: #fca5a5;
    }
    .btn-danger:hover {
      background: #3b1219;
      border-color: #ef4444;
      color: #fff;
      box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
    }

    .key-badge {
      display: inline-block;
      background: rgba(255, 255, 255, 0.12);
      padding: 1px 5px;
      border-radius: 3px;
      font-size: 9px;
      color: #94a3b8;
      font-family: monospace;
      font-weight: 700;
    }

    /* Toast Notification */
    #toast {
      position: fixed;
      top: 68px;
      right: 24px;
      background: rgba(13, 18, 27, 0.94);
      backdrop-filter: blur(12px);
      border: 1px solid var(--neon-cyan);
      box-shadow: 0 0 20px rgba(0, 240, 255, 0.3);
      color: #fff;
      padding: 10px 18px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      opacity: 0;
      transform: translateY(-8px);
      transition: all 0.25s ease;
      pointer-events: none;
      z-index: 1000;
    }
    #toast.show {
      opacity: 1;
      transform: translateY(0);
    }

    /* Shortcuts Help Modal */
    #help-modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(6px);
      z-index: 2000;
      align-items: center;
      justify-content: center;
    }
    #help-modal.open {
      display: flex;
    }
    .modal-box {
      background: #0d121c;
      border: 1px solid var(--border-glow);
      border-radius: 10px;
      width: 440px;
      padding: 24px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.9);
    }
    .modal-title {
      font-size: 15px;
      font-weight: 800;
      color: var(--neon-cyan);
      margin-bottom: 16px;
      letter-spacing: 0.5px;
    }
    .key-list {
      display: grid;
      grid-template-columns: 85px 1fr;
      gap: 10px 16px;
      font-size: 13px;
      color: #cbd5e1;
    }
  </style>
</head>
<body>
  <!-- Top Header: Branding & Live Telemetry -->
  <header>
    <div class="brand-section">
      <div class="status-badge">
        <div class="live-dot"></div>
        <span>LIVE // 60 FPS</span>
      </div>
      <div class="brand-title">ALOHA 14-DOF BIMANUAL // MISSION COCKPIT</div>
    </div>

    <!-- Live Telemetry Readouts -->
    <div class="telemetry-strip">
      <div class="tel-item">
        <span class="tel-label">PHASE</span>
        <span class="tel-val val-cyan" id="tel-phase">P0: GRASP</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">STEP</span>
        <span class="tel-val" id="tel-step">0000 / 250</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">PROJ SPEED</span>
        <span class="tel-val val-magenta" id="tel-speed">0.00 m/s</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">CATCH DIST</span>
        <span class="tel-val val-amber" id="tel-dist">0.00 m</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">CONTACT</span>
        <span class="tel-val val-green" id="tel-contact">FREE</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">REWARD</span>
        <span class="tel-val" id="tel-reward">+0.00</span>
      </div>
    </div>

    <div class="header-actions">
      <button class="btn btn-action" onclick="toggleHelp()"><span class="key-badge">?</span> HOTKEYS</button>
      <button class="btn btn-danger" onclick="window.close()">EXIT</button>
    </div>
  </header>

  <!-- Centered Simulation Arena Canvas -->
  <div id="viewport-container">
    <div class="viewport-frame" id="viewport-frame">
      <div class="cam-overlay-tag" id="cam-tag">FRONT STUDIO (OVERVIEW)</div>
      <img id="stream-img" src="/video_feed" alt="Live MuJoCo Stream">
    </div>
  </div>

  <!-- Bottom Floating Glassmorphism Control Dock -->
  <div class="bottom-dock">
    <!-- Camera Switcher -->
    <div class="dock-group">
      <span class="dock-label">CAMERA:</span>
      <button class="btn cam-btn active" data-cam="front_overview" onclick="setCam('front_overview')">
        <span class="key-badge">1</span> Front
      </button>
      <button class="btn cam-btn" data-cam="side_cinematic" onclick="setCam('side_cinematic')">
        <span class="key-badge">2</span> Side
      </button>
      <button class="btn cam-btn" data-cam="top_tactical" onclick="setCam('top_tactical')">
        <span class="key-badge">3</span> Top
      </button>
      <button class="btn cam-btn" data-cam="left_wrist_cam" onclick="setCam('left_wrist_cam')">
        <span class="key-badge">4</span> L-Wrist
      </button>
      <button class="btn cam-btn" data-cam="right_wrist_cam" onclick="setCam('right_wrist_cam')">
        <span class="key-badge">5</span> R-Wrist
      </button>
    </div>

    <div class="dock-divider"></div>

    <!-- Playback Speed -->
    <div class="dock-group">
      <span class="dock-label">SPEED:</span>
      <button class="btn speed-btn" data-speed="0.25" onclick="setSpeed(0.25)">0.25x</button>
      <button class="btn speed-btn" data-speed="0.5" onclick="setSpeed(0.5)">0.5x</button>
      <button class="btn speed-btn active" data-speed="1.0" onclick="setSpeed(1.0)">1.0x</button>
    </div>

    <div class="dock-divider"></div>

    <!-- Simulation Actions -->
    <div class="dock-group">
      <button class="btn btn-action" id="pause-btn" onclick="togglePause()">
        <span class="key-badge">Space</span> <span id="pause-text">Pause</span>
      </button>
      <button class="btn btn-action" onclick="resetEpisode()">
        <span class="key-badge">R</span> Reset
      </button>
      <button class="btn btn-action" onclick="toggleFullscreen()">
        <span class="key-badge">F</span> Fullscreen
      </button>
    </div>
  </div>

  <!-- Toast Notification -->
  <div id="toast">Notification</div>

  <!-- Hotkeys Help Modal -->
  <div id="help-modal" onclick="if(event.target===this) toggleHelp()">
    <div class="modal-box">
      <div class="modal-title">⚡ MISSION COCKPIT HOTKEYS</div>
      <div class="key-list">
        <div><span class="key-badge">1 ~ 5</span></div><div>Switch Camera Angle (Front, Side, Top, L-Wrist, R-Wrist)</div>
        <div><span class="key-badge">Space</span></div><div>Pause / Resume Physics Simulation</div>
        <div><span class="key-badge">R</span></div><div>Immediate Episode Reset (Retoss)</div>
        <div><span class="key-badge">[ / ]</span></div><div>Adjust Playback Speed (0.25x / 1.0x)</div>
        <div><span class="key-badge">F</span></div><div>Toggle Fullscreen Viewport</div>
        <div><span class="key-badge">?</span></div><div>Toggle Hotkeys Help Window</div>
      </div>
      <div style="margin-top:20px; text-align:right;">
        <button class="btn btn-action" onclick="toggleHelp()">Close</button>
      </div>
    </div>
  </div>

  <script>
    // Force maximize standalone app window to screen dimensions
    window.addEventListener('DOMContentLoaded', () => {
      try {
        if (window.screen) {
          window.moveTo(0, 0);
          window.resizeTo(window.screen.availWidth, window.screen.availHeight);
        }
      } catch (e) {}
    });

    const CAM_NAMES = {
      'front_overview': 'FRONT STUDIO (OVERVIEW)',
      'side_cinematic': 'SIDE CINEMATIC (TRAJECTORY)',
      'top_tactical': 'TOP TACTICAL (APEX)',
      'left_wrist_cam': 'LEFT WRIST (THROWER)',
      'right_wrist_cam': 'RIGHT WRIST (CATCHER)'
    };

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      toast.classList.add('show');
      clearTimeout(window._toastTimer);
      window._toastTimer = setTimeout(() => toast.classList.remove('show'), 1600);
    }

    function setCam(cam) {
      fetch('/api/camera?mode=' + cam)
        .then(r => r.json())
        .then(data => {
          document.querySelectorAll('.cam-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.cam === cam);
          });
          const label = CAM_NAMES[cam] || cam;
          document.getElementById('cam-tag').innerText = label;
          showToast('Camera switched to: ' + (VALID_CAMERAS[cam] || cam));
        })
        .catch(err => console.error(err));
    }

    const VALID_CAMERAS = {
      'front_overview': 'Front Studio',
      'side_cinematic': 'Side Cinematic',
      'top_tactical': 'Top Tactical',
      'left_wrist_cam': 'Left Wrist',
      'right_wrist_cam': 'Right Wrist'
    };

    function setSpeed(scale) {
      fetch('/api/speed?scale=' + scale)
        .then(r => r.json())
        .then(data => {
          document.querySelectorAll('.speed-btn').forEach(b => {
            b.classList.toggle('active', parseFloat(b.dataset.speed) === scale);
          });
          showToast('Playback speed: ' + scale + 'x');
        })
        .catch(err => console.error(err));
    }

    function togglePause() {
      fetch('/api/pause')
        .then(r => r.json())
        .then(data => {
          const paused = data.paused;
          document.getElementById('pause-text').innerText = paused ? 'Resume' : 'Pause';
          document.getElementById('pause-btn').classList.toggle('active', paused);
          showToast(paused ? 'Simulation Paused' : 'Simulation Resumed');
        })
        .catch(err => console.error(err));
    }

    function resetEpisode() {
      fetch('/api/reset')
        .then(r => r.json())
        .then(data => {
          showToast('Episode Reset');
        })
        .catch(err => console.error(err));
    }

    function toggleFullscreen() {
      const el = document.getElementById('viewport-container');
      if (!document.fullscreenElement) {
        el.requestFullscreen().catch(err => console.error(err));
      } else {
        document.exitFullscreen();
      }
    }

    function toggleHelp() {
      document.getElementById('help-modal').classList.toggle('open');
    }

    // Telemetry Poller (10Hz)
    setInterval(() => {
      fetch('/api/telemetry')
        .then(r => r.json())
        .then(t => {
          if (!t) return;
          document.getElementById('tel-phase').innerText = t.phase_name || ('P' + t.phase);
          document.getElementById('tel-step').innerText = String(t.step).padStart(4, '0') + ' / 250';
          document.getElementById('tel-speed').innerText = t.speed.toFixed(2) + ' m/s';
          document.getElementById('tel-dist').innerText = t.dist.toFixed(2) + ' m';
          document.getElementById('tel-contact').innerText = t.contact ? 'LOCKED' : 'FREE';
          document.getElementById('tel-contact').className = 'tel-val ' + (t.contact ? 'val-green' : 'val-amber');
          document.getElementById('tel-reward').innerText = (t.reward >= 0 ? '+' : '') + t.reward.toFixed(2);
        })
        .catch(() => {});
    }, 100);

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === '1') setCam('front_overview');
      else if (e.key === '2') setCam('side_cinematic');
      else if (e.key === '3') setCam('top_tactical');
      else if (e.key === '4') setCam('left_wrist_cam');
      else if (e.key === '5') setCam('right_wrist_cam');
      else if (e.code === 'Space') { e.preventDefault(); togglePause(); }
      else if (e.key === 'r' || e.key === 'R') resetEpisode();
      else if (e.key === 'f' || e.key === 'F') toggleFullscreen();
      else if (e.key === '?') toggleHelp();
      else if (e.key === '[') setSpeed(0.25);
      else if (e.key === ']') setSpeed(1.0);
    });
  </script>
</body>
</html>
"""

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global camera_mode, speed_scale, is_paused, force_reset
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            while running:
                with frame_lock:
                    frame_bytes = latest_jpeg
                if frame_bytes is not None:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    except Exception:
                        break
                time.sleep(0.016)

        elif self.path.startswith("/api/camera"):
            for cam_key in VALID_CAMERAS:
                if f"mode={cam_key}" in self.path:
                    camera_mode = cam_key
                    break
            self._send_json({"ok": True, "camera": camera_mode, "name": VALID_CAMERAS.get(camera_mode, "")})

        elif self.path.startswith("/api/speed"):
            if "scale=0.25" in self.path:
                speed_scale = 0.25
            elif "scale=0.5" in self.path:
                speed_scale = 0.5
            else:
                speed_scale = 1.0
            self._send_json({"ok": True, "speed": speed_scale})

        elif self.path.startswith("/api/pause"):
            is_paused = not is_paused
            self._send_json({"ok": True, "paused": is_paused})

        elif self.path.startswith("/api/reset"):
            force_reset = True
            self._send_json({"ok": True, "reset": True})

        elif self.path.startswith("/api/telemetry"):
            with telemetry_lock:
                data_copy = current_telemetry.copy()
            self._send_json(data_copy)

        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def simulation_worker(model_path="checkpoints/aloha_toss_catch_final.zip"):
    global latest_jpeg, running, camera_mode, speed_scale, is_paused, force_reset, current_telemetry
    env = gym.make("AlohaTossCatch-v0")
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"[SimEngine] Loaded trained policy: {model_path}")
    else:
        model = None
        print(f"[SimEngine] Checkpoint not found, using policy sampling.")

    unwrapped = env.unwrapped
    engine = VisualEngine(unwrapped.model, unwrapped.data, width=1280, height=720)

    phase_label_map = {
        0: "PHASE 0: SECURE GRASP",
        1: "PHASE 1: BALLISTIC TOSS",
        2: "PHASE 2: AIR INTERCEPT",
        3: "PHASE 3: CUSHION LOCK"
    }

    ep = 0
    while running:
        ep += 1
        obs, info = env.reset(seed=int(time.time() * 1000) % 100000)
        step = 0
        force_reset = False

        while running:
            if force_reset:
                break

            if is_paused:
                time.sleep(0.04)
                continue

            step += 1
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            proj_vel = unwrapped.data.qvel[16:19].copy()
            proj_speed = float(np.linalg.norm(proj_vel))
            phase_idx = int(info.get("phase", 0))
            is_success = bool(info.get("is_success", False))

            phase_str = "MISSION SUCCESS: CATCH SECURED" if is_success else phase_label_map.get(phase_idx, f"PHASE {phase_idx}")

            # Telemetry for HUD & Web Poller
            telemetry = {
                "step": step,
                "phase": phase_idx,
                "phase_name": phase_str,
                "proj_pos": unwrapped.data.site_xpos[unwrapped.site_projectile].copy(),
                "proj_vel": proj_vel,
                "speed": proj_speed,
                "dist": float(info.get("dist_proj_to_catch", 0.0)),
                "dist_to_catch": float(info.get("dist_proj_to_catch", 0.0)),
                "reward": float(reward),
                "contact": bool(unwrapped.data.sensordata[0] > 0.05),
                "is_success": is_success,
                "camera": camera_mode,
                "speed_scale": speed_scale,
                "paused": is_paused
            }

            with telemetry_lock:
                current_telemetry.update({
                    "step": step,
                    "phase": phase_idx,
                    "phase_name": phase_str,
                    "speed": proj_speed,
                    "dist": telemetry["dist"],
                    "reward": float(reward),
                    "contact": telemetry["contact"],
                    "is_success": is_success,
                    "camera": camera_mode,
                    "speed_scale": speed_scale,
                    "paused": is_paused
                })

            # Render frame with active camera mode
            frame_bgr = engine.render_frame(telemetry, camera_name=camera_mode)
            _, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

            with frame_lock:
                latest_jpeg = encoded.tobytes()

            if speed_scale <= 0.25:
                time.sleep(0.06)
            elif speed_scale <= 0.5:
                time.sleep(0.03)
            else:
                time.sleep(0.012)

            if terminated or truncated:
                time.sleep(0.4)
                break

    engine.close()
    env.close()


def launch_app_window(port=8080):
    url = f"http://127.0.0.1:{port}/"
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    browser_bin = None
    if os.path.exists(chrome_path):
        browser_bin = chrome_path
    elif os.path.exists(edge_path):
        browser_bin = edge_path

    # Detect user's actual monitor resolution via Windows user32 API
    screen_w, screen_h = 1920, 1080
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except Exception:
        pass

    if browser_bin:
        print(f"[AppWindow] Launching full widescreen app-mode window ({screen_w}x{screen_h}) via: {os.path.basename(browser_bin)}")
        cmd = [
            browser_bin,
            f"--app={url}",
            f"--window-size={screen_w},{screen_h}",
            "--window-position=0,0",
            "--start-maximized"
        ]
        subprocess.Popen(cmd)
    else:
        import webbrowser
        webbrowser.open(url)


def main():
    global running
    port = 8080
    server = ThreadingHTTPServer(("127.0.0.1", port), StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    model_path = "checkpoints/aloha_toss_catch_final.zip"
    sim_thread = threading.Thread(target=simulation_worker, args=(model_path,), daemon=True)
    sim_thread.start()

    time.sleep(0.8)
    launch_app_window(port)

    print("\n=================================================================")
    print("🚀 Standalone Live Cockpit Window is RUNNING! (No Address Bar)")
    print(f"• Local URL   : http://127.0.0.1:{port}/")
    print("• Multi-Thread: ThreadingHTTPServer Active (Instant API Response)")
    print("• Control Dock: Floating Ergonomic Bottom Toolbar")
    print("• Cameras     : 5-Angle Full Studio (1:Front, 2:Side, 3:Top, 4:L-Wrist, 5:R-Wrist)")
    print("• Shortcuts   : 1~5 (Cam), Space (Pause), R (Reset), F (Fullscreen)")
    print("• Press Ctrl+C in this terminal to shut down.")
    print("=================================================================\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Cockpit] Shutting down...")
        running = False
        server.shutdown()


if __name__ == "__main__":
    main()
