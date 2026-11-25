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

# Gamification & Interactive Control State
game_mode = "manual"             # "manual", "auto", "chaos"
shot_style = "bullet"            # "bullet", "lob", "hook", "beast"
is_waiting_trigger = True        # Starts paused in lobby waiting for user [LAUNCH] trigger!
auto_loop = False                # Controlled by game_mode
target_toss_yaw_deg = 0.0        # User-chosen yaw angle (-15 deg to +15 deg)
target_toss_pitch_deg = 0.0      # User-chosen pitch variation (-3 deg to +5 deg)
toss_mode = "manual"             # "manual" or "chaos"
combo_streak = 0
total_score = 0
last_event = "READY TO LAUNCH"

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
    "paused": False,
    "is_waiting_trigger": True,
    "is_waiting": True,
    "auto_loop": False,
    "game_mode": "manual",
    "shot_style": "bullet",
    "target_yaw": 0.0,
    "toss_mode": "manual",
    "combo": 0,
    "score": 0,
    "last_event": "READY TO LAUNCH"
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
  <title>ALOHA 14-DOF Bimanual Dynamic Toss & Catch | HWIHWA LAB</title>
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
      border: 1px solid rgba(0, 240, 255, 0.35);
      box-shadow: 0 15px 50px rgba(0, 0, 0, 0.9), 0 0 30px rgba(0, 240, 255, 0.12);
      max-width: 100%;
      max-height: 100%;
      aspect-ratio: 16 / 9;
    }
    .viewport-frame::before {
      content: "";
      position: absolute;
      top: 8px; left: 8px; width: 14px; height: 14px;
      border-top: 2px solid var(--neon-cyan);
      border-left: 2px solid var(--neon-cyan);
      pointer-events: none;
      z-index: 10;
    }
    .viewport-frame::after {
      content: "";
      position: absolute;
      bottom: 8px; right: 8px; width: 14px; height: 14px;
      border-bottom: 2px solid var(--neon-cyan);
      border-right: 2px solid var(--neon-cyan);
      pointer-events: none;
      z-index: 10;
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
      top: 9px;
      left: 14px;
      height: 24px;
      display: inline-flex;
      align-items: center;
      background: rgba(8, 12, 18, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(0, 240, 255, 0.45);
      padding: 0 10px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 800;
      color: var(--neon-cyan);
      letter-spacing: 0.8px;
      white-space: nowrap;
      pointer-events: none;
      z-index: 15;
    }

    /* =========================================================================
       BOTTOM CONTROL DOCK: Ultra-Slim Ergonomic Floating Cyber HUD Toolbar
       ========================================================================= */
    .bottom-dock {
      position: fixed;
      bottom: 12px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(10, 15, 24, 0.94);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(0, 240, 255, 0.28);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.85), 0 0 20px rgba(0, 240, 255, 0.15);
      border-radius: 6px;
      padding: 4px 12px;
      height: 38px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      z-index: 9999;
      pointer-events: auto;
      white-space: nowrap;
      max-width: 98vw;
      box-sizing: border-box;
    }

    .dock-divider {
      width: 1px;
      height: 18px;
      background: rgba(255, 255, 255, 0.15);
      flex-shrink: 0;
    }

    .dock-group {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      flex-shrink: 0;
    }

    .dock-label {
      font-size: 9px;
      font-weight: 800;
      color: #64748b;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      margin-right: 2px;
      flex-shrink: 0;
      user-select: none;
    }

    .btn {
      background: #111722;
      border: 1px solid #1e293b;
      color: #94a3b8;
      height: 26px;
      padding: 0 8px;
      font-size: 10px;
      font-weight: 700;
      border-radius: 4px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      letter-spacing: 0.4px;
      text-transform: uppercase;
      transition: all 0.12s ease;
      white-space: nowrap;
      flex-shrink: 0;
      user-select: none;
      pointer-events: auto;
    }

    .btn:hover {
      background: #1a2336;
      color: #fff;
      border-color: var(--neon-cyan);
      box-shadow: 0 0 8px rgba(0, 240, 255, 0.3);
    }

    .btn:active {
      transform: scale(0.96);
    }

    .btn.active {
      background: rgba(0, 240, 255, 0.18);
      color: var(--neon-cyan);
      border-color: var(--neon-cyan);
      box-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
      font-weight: 800;
    }

    .btn-action {
      background: #141a26;
      border-color: #2a3749;
    }
    .btn-action:hover {
      border-color: var(--neon-amber);
      color: var(--neon-amber);
      box-shadow: 0 0 8px rgba(255, 183, 0, 0.35);
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
      box-shadow: 0 0 10px rgba(239, 68, 68, 0.45);
    }

    .key-badge {
      display: inline-block;
      background: rgba(255, 255, 255, 0.12);
      padding: 0 4px;
      border-radius: 2px;
      font-size: 8.5px;
      color: #94a3b8;
      font-family: monospace;
      font-weight: 700;
      line-height: 14px;
      pointer-events: none;
    }

    /* Giant Fire / Launch Button: Slim Cyberpunk Launcher */
    .btn-fire {
      background: linear-gradient(135deg, #ff007f 0%, #e11d48 100%);
      border: 1px solid #ff4da6;
      box-shadow: 0 0 15px rgba(255, 0, 127, 0.6), inset 0 0 6px rgba(255, 255, 255, 0.3);
      color: #fff;
      font-weight: 800;
      font-size: 10.5px;
      letter-spacing: 0.8px;
      height: 28px;
      padding: 0 12px;
      border-radius: 4px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s ease;
      white-space: nowrap;
      flex-shrink: 0;
      text-transform: uppercase;
    }
    .btn-fire.ready-pulse {
      animation: fire-glow 1.0s infinite alternate;
    }
    @keyframes fire-glow {
      0% { transform: scale(1.0); box-shadow: 0 0 10px rgba(255, 0, 127, 0.5); }
      100% { transform: scale(1.03); box-shadow: 0 0 24px rgba(255, 0, 127, 0.95); }
    }
    .btn-fire:hover {
      transform: scale(1.04);
      box-shadow: 0 0 28px rgba(255, 0, 127, 1.0);
    }

    /* Interactive Angle Control Deck: Slim */
    .angle-slider-group {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: rgba(0, 240, 255, 0.05);
      height: 26px;
      padding: 0 8px;
      border-radius: 4px;
      border: 1px solid rgba(0, 240, 255, 0.22);
      flex-shrink: 0;
    }
    .slider-tag {
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.5px;
      color: var(--neon-cyan);
      white-space: nowrap;
      user-select: none;
    }
    .slider-bound {
      font-size: 9px;
      font-family: 'Consolas', monospace;
      color: #64748b;
      user-select: none;
      font-weight: 700;
    }
    .angle-slider {
      width: 65px;
      height: 4px;
      accent-color: var(--neon-cyan);
      cursor: pointer;
    }
    .angle-val-badge {
      font-family: 'Consolas', monospace;
      font-size: 10px;
      font-weight: 800;
      color: var(--neon-cyan);
      min-width: 34px;
      text-align: right;
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

    /* Shortcuts Help Modal: Wide Glassmorphism Cyber HUD */
    #help-modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(4, 8, 16, 0.82);
      backdrop-filter: blur(10px);
      z-index: 2000;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }
    #help-modal.open {
      display: flex;
    }
    .modal-box {
      background: linear-gradient(135deg, rgba(13, 20, 35, 0.97) 0%, rgba(8, 12, 22, 0.99) 100%);
      border: 1px solid rgba(0, 240, 255, 0.35);
      border-radius: 12px;
      width: 720px;
      max-width: 95vw;
      padding: 24px 28px;
      box-shadow: 0 25px 80px rgba(0, 0, 0, 0.95), 0 0 35px rgba(0, 240, 255, 0.15);
      position: relative;
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 12px;
    }
    .modal-title {
      font-size: 15px;
      font-weight: 800;
      color: var(--neon-cyan);
      letter-spacing: 0.8px;
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 0;
    }
    .modal-close-btn {
      background: transparent;
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #94a3b8;
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 11px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
    }
    .modal-close-btn:hover {
      border-color: var(--neon-magenta);
      color: #fff;
      box-shadow: 0 0 10px rgba(255, 0, 127, 0.3);
    }
    .modal-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .modal-card {
      background: rgba(255, 255, 255, 0.025);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 8px;
      padding: 14px 16px;
    }
    .card-title {
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.5px;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
      text-transform: uppercase;
    }
    .card-title.accent-cyan { color: var(--neon-cyan); }
    .card-title.accent-magenta { color: var(--neon-magenta); }
    .card-title.accent-green { color: var(--neon-green); }
    .card-title.accent-amber { color: var(--neon-amber); }
    .key-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 12px;
      color: #cbd5e1;
      padding: 5px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    .key-row:last-child {
      border-bottom: none;
    }
    .key-row .desc {
      font-size: 11px;
      color: #94a3b8;
      text-align: right;
    }
    .shot-guide-item {
      font-size: 11px;
      color: #cbd5e1;
      padding: 3px 0;
      line-height: 1.35;
    }
    .shot-guide-tag {
      font-weight: 800;
      color: var(--neon-cyan);
      margin-right: 4px;
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
      <div class="brand-title">ALOHA 14-DOF Bimanual Dynamic Toss & Catch | HWIHWA LAB</div>
    </div>

    <!-- Live Telemetry Readouts -->
    <div class="telemetry-strip">
      <div class="tel-item">
        <span class="tel-label">STREAK</span>
        <span class="tel-val val-magenta" id="tel-combo">0X 🔥</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">SCORE</span>
        <span class="tel-val val-cyan" id="tel-score">0 PTS</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">SHOT</span>
        <span class="tel-val val-cyan" id="tel-shot">🎯 BULLET</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">PHASE</span>
        <span class="tel-val val-cyan" id="tel-phase">P0: LOBBY</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">STEP</span>
        <span class="tel-val" id="tel-step">0000 / 250</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">SPEED</span>
        <span class="tel-val val-magenta" id="tel-speed">0.00 m/s</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">DIST</span>
        <span class="tel-val val-amber" id="tel-dist">0.00 m</span>
      </div>
      <div class="tel-item">
        <span class="tel-label">STATUS</span>
        <span class="tel-val val-green" id="tel-contact">READY</span>
      </div>
    </div>

    <div class="header-actions">
      <button class="btn btn-action active" id="header-sfx-btn" onclick="toggleSfx()">
        <span id="header-sfx-icon">🔊</span> SFX: ON
      </button>
      <button class="btn btn-action" onclick="toggleFullscreen()">
        <span class="key-badge">F</span> FULL
      </button>
      <button class="btn btn-action" onclick="toggleHelp()">
        <span class="key-badge">?</span> HOTKEYS
      </button>
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

  <!-- Bottom Floating Ultra-Slim Cyber HUD Toolbar -->
  <div class="bottom-dock">
    <!-- Capsule 1: Playback Action Cluster (LAUNCH, PAUSE, RESET) -->
    <div class="dock-group">
      <button class="btn-fire ready-pulse" id="btn-fire" onclick="fireToss()">
        <span class="key-badge" style="background:rgba(0,0,0,0.35);color:#fff;">SPACE</span>
        <span id="btn-fire-text">🚀 LAUNCH</span>
      </button>
      <button class="btn btn-action" id="pause-btn" onclick="togglePause()">
        <span class="key-badge">P</span> <span id="pause-text">PAUSE</span>
      </button>
      <button class="btn btn-action" id="reset-btn" onclick="resetEpisode()">
        <span class="key-badge">R</span> RESET
      </button>
    </div>

    <div class="dock-divider"></div>

    <!-- Capsule 2: Game Mode (3-Segment Pill) -->
    <div class="dock-group">
      <span class="dock-label">MODE:</span>
      <button class="btn active" id="btn-mode-manual" onclick="setGameMode('manual')">MANUAL</button>
      <button class="btn" id="btn-mode-auto" onclick="setGameMode('auto')">AUTO</button>
      <button class="btn" id="btn-mode-chaos" onclick="setGameMode('chaos')" style="color:var(--neon-magenta);border-color:rgba(255,0,127,0.35);">CHAOS RUSH ⚡</button>
    </div>

    <div class="dock-divider"></div>

    <!-- Capsule 3: 4 Signature Shot Styles + Micro Slider -->
    <div class="dock-group" id="shot-deck">
      <span class="dock-label">SHOT:</span>
      <button class="btn active" id="btn-shot-bullet" onclick="setShotStyle('bullet')">🎯 BULLET</button>
      <button class="btn" id="btn-shot-lob" onclick="setShotStyle('lob')">🌈 LOB</button>
      <button class="btn" id="btn-shot-hook" onclick="setShotStyle('hook')">↩️ HOOK</button>
      <button class="btn" id="btn-shot-beast" onclick="setShotStyle('beast')">⚡ BEAST</button>
      <div class="angle-slider-group" title="Horizontal Toss Aiming Angle (-15° Left to +15° Right)">
        <span class="slider-tag">AIM YAW:</span>
        <span class="slider-bound">L</span>
        <input type="range" min="-15" max="15" step="1" value="0" class="angle-slider" id="angle-slider" oninput="onYawChange(this.value)">
        <span class="slider-bound">R</span>
        <span class="angle-val-badge" id="angle-val-badge">0.0°</span>
      </div>
    </div>

    <div class="dock-divider"></div>

    <!-- Capsule 4: 5 Multi-Angle Camera Studio -->
    <div class="dock-group">
      <span class="dock-label">CAM:</span>
      <button class="btn cam-btn active" data-cam="front_overview" onclick="setCam('front_overview')"><span class="key-badge">1</span> FRONT</button>
      <button class="btn cam-btn" data-cam="side_cinematic" onclick="setCam('side_cinematic')"><span class="key-badge">2</span> SIDE</button>
      <button class="btn cam-btn" data-cam="top_tactical" onclick="setCam('top_tactical')"><span class="key-badge">3</span> TOP</button>
      <button class="btn cam-btn" data-cam="left_wrist_cam" onclick="setCam('left_wrist_cam')"><span class="key-badge">4</span> L-WRIST</button>
      <button class="btn cam-btn" data-cam="right_wrist_cam" onclick="setCam('right_wrist_cam')"><span class="key-badge">5</span> R-WRIST</button>
    </div>
  </div>

  <!-- Toast Notification -->
  <div id="toast">Notification</div>

  <!-- Hotkeys Help Modal: Wide Glassmorphism Cyber HUD -->
  <div id="help-modal" onclick="if(event.target===this) toggleHelp()">
    <div class="modal-box">
      <div class="modal-header">
        <div class="modal-title">⚡ MISSION COCKPIT CONTROLS & MANUAL</div>
        <button class="modal-close-btn" onclick="toggleHelp()"><span class="key-badge">ESC</span> CLOSE</button>
      </div>

      <div class="modal-grid">
        <!-- Card 1: Playback & Flight Controls -->
        <div class="modal-card">
          <div class="card-title accent-cyan">🚀 PLAYBACK & FLIGHT</div>
          <div class="key-row">
            <div><span class="key-badge">Space</span> <span class="key-badge">Enter</span></div>
            <div class="desc">LAUNCH in lobby / Pause in flight</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">P</span></div>
            <div class="desc">Pause / Resume physics loop</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">R</span></div>
            <div class="desc">Instant reset & back to lobby</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">A</span></div>
            <div class="desc">Toggle AUTO infinite benchmarking</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">C</span></div>
            <div class="desc">CHAOS RUSH (infinite random fire)</div>
          </div>
        </div>

        <!-- Card 2: Aim & Trajectory Controls -->
        <div class="modal-card">
          <div class="card-title accent-magenta">🎯 AIM & TRAJECTORY (탄도 및 조준)</div>
          <div class="key-row">
            <div><span class="key-badge">←</span> <span class="key-badge">→</span></div>
            <div class="desc">Adjust Aim Yaw by 1.0° (Fine-tune)</div>
          </div>
          <div style="margin-top:6px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.06);">
            <div class="shot-guide-item"><span class="shot-guide-tag">AIM YAW:</span> -15° (Left) ~ +15° (Right) slider</div>
            <div class="shot-guide-item"><span class="shot-guide-tag">🎯 BULLET:</span> Flat fast laser (0.0°, 2.3m/s)</div>
            <div class="shot-guide-item"><span class="shot-guide-tag">🌈 LOB:</span> High rainbow arc (+1.5° pitch)</div>
            <div class="shot-guide-item"><span class="shot-guide-tag">↩️ HOOK:</span> Sharp left curve (-10.0° yaw)</div>
            <div class="shot-guide-item"><span class="shot-guide-tag">⚡ BEAST:</span> Wide right extreme (+10.0° yaw)</div>
          </div>
        </div>

        <!-- Card 3: Camera Studio -->
        <div class="modal-card">
          <div class="card-title accent-green">🎥 MULTI-ANGLE CAMERA STUDIO</div>
          <div class="key-row">
            <div><span class="key-badge">1</span></div>
            <div class="desc">FRONT Studio (Full arena overview)</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">2</span></div>
            <div class="desc">SIDE Cinematic (Height & arc profile)</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">3</span></div>
            <div class="desc">TOP Tactical (Left/Right spread angle)</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">4</span></div>
            <div class="desc">L-WRIST (Toss robot 1st-person POV)</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">5</span></div>
            <div class="desc">R-WRIST (Catch robot 1st-person POV)</div>
          </div>
        </div>

        <!-- Card 4: System & Display -->
        <div class="modal-card">
          <div class="card-title accent-amber">⚙️ SYSTEM & DISPLAY</div>
          <div class="key-row">
            <div><span class="key-badge">S</span> <span class="key-badge">M</span></div>
            <div class="desc">Toggle Arcade SFX audio on/off</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">F</span></div>
            <div class="desc">Toggle Fullscreen theater mode</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">?</span></div>
            <div class="desc">Toggle Hotkeys manual modal</div>
          </div>
          <div class="key-row">
            <div><span class="key-badge">ESC</span></div>
            <div class="desc">Close this help window</div>
          </div>
        </div>
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

    const VALID_CAMERAS = {
      'front_overview': 'Front Studio',
      'side_cinematic': 'Side Cinematic',
      'top_tactical': 'Top Tactical',
      'left_wrist_cam': 'Left Wrist',
      'right_wrist_cam': 'Right Wrist'
    };

    let isWaitingLaunch = true;
    let prevCombo = 0;
    let prevLastEvent = '';
    let sfxEnabled = true;
    let audioCtx = null;

    // Zero-dependency Pure Web Audio API Sound Synthesizer (Instant 0MB SFX)
    function initAudio() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
    }

    function playSfx(type) {
      if (!sfxEnabled) return;
      try {
        initAudio();
        if (audioCtx.state === 'suspended') {
          audioCtx.resume();
        }
        const t = audioCtx.currentTime;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);

        if (type === 'click') {
          // Sharp UI mechanical switch tick
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(1400, t);
          osc.frequency.exponentialRampToValueAtTime(320, t + 0.035);
          gain.gain.setValueAtTime(0.06, t);
          gain.gain.linearRampToValueAtTime(0.001, t + 0.035);
          osc.start(t);
          osc.stop(t + 0.035);
        } else if (type === 'launch') {
          // Sci-Fi railgun toss laser sweep
          osc.type = 'sine';
          osc.frequency.setValueAtTime(280, t);
          osc.frequency.exponentialRampToValueAtTime(920, t + 0.16);
          gain.gain.setValueAtTime(0.20, t);
          gain.gain.linearRampToValueAtTime(0.001, t + 0.20);
          osc.start(t);
          osc.stop(t + 0.20);
        } else if (type === 'catch') {
          // High-dopamine two-tone arcade coin chime
          const osc2 = audioCtx.createOscillator();
          const gain2 = audioCtx.createGain();
          osc2.connect(gain2);
          gain2.connect(audioCtx.destination);

          osc.type = 'sine';
          osc.frequency.setValueAtTime(587.33, t); // D5
          osc.frequency.setValueAtTime(880.00, t + 0.08); // A5
          gain.gain.setValueAtTime(0.22, t);
          gain.gain.exponentialRampToValueAtTime(0.001, t + 0.38);

          osc2.type = 'triangle';
          osc2.frequency.setValueAtTime(1174.66, t + 0.08); // D6
          gain2.gain.setValueAtTime(0.14, t + 0.08);
          gain2.gain.exponentialRampToValueAtTime(0.001, t + 0.38);

          osc.start(t);
          osc2.start(t + 0.08);
          osc.stop(t + 0.38);
          osc2.stop(t + 0.38);
        } else if (type === 'miss') {
          // Low-pitch glitch buzz
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(220, t);
          osc.frequency.linearRampToValueAtTime(75, t + 0.25);
          gain.gain.setValueAtTime(0.18, t);
          gain.gain.linearRampToValueAtTime(0.001, t + 0.25);
          osc.start(t);
          osc.stop(t + 0.25);
        }
      } catch (e) {}
    }

    function toggleSfx() {
      sfxEnabled = !sfxEnabled;
      const btn = document.getElementById('header-sfx-btn');
      const icon = document.getElementById('header-sfx-icon');
      if (btn) btn.classList.toggle('active', sfxEnabled);
      if (icon) icon.innerText = sfxEnabled ? '🔊' : '🔇';
      showToast(sfxEnabled ? 'SFX Audio: ENABLED' : 'SFX Audio: MUTED');
      if (sfxEnabled) playSfx('click');
    }

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.innerText = msg;
      toast.classList.add('show');
      clearTimeout(window._toastTimer);
      window._toastTimer = setTimeout(() => toast.classList.remove('show'), 1600);
    }

    function fireToss() {
      playSfx('launch');
      fetch('/api/fire')
        .then(r => r.json())
        .then(data => {
          showToast('🚀 BALL LAUNCHED! INTERCEPTING...');
          const btn = document.getElementById('btn-fire');
          btn.classList.remove('ready-pulse');
          document.getElementById('btn-fire-text').innerText = 'IN FLIGHT...';
        })
        .catch(err => console.error(err));
    }

    function setGameMode(mode) {
      playSfx('click');
      fetch('/api/mode?mode=' + mode)
        .then(r => r.json())
        .then(data => {
          ['manual', 'auto', 'chaos'].forEach(m => {
            const b = document.getElementById('btn-mode-' + m);
            if (b) b.classList.toggle('active', m === mode);
          });
          const shotDeck = document.getElementById('shot-deck');
          if (mode === 'chaos') {
            shotDeck.style.opacity = '0.35';
            shotDeck.style.pointerEvents = 'none';
            showToast('🔥 CHAOS RUSH: CONTINUOUS RANDOM STREAM ACTIVATED!');
          } else {
            shotDeck.style.opacity = '1.0';
            shotDeck.style.pointerEvents = 'auto';
            showToast('Game Mode: ' + mode.toUpperCase());
          }
        })
        .catch(err => console.error(err));
    }

    function setShotStyle(style) {
      playSfx('click');
      fetch('/api/shot_style?style=' + style)
        .then(r => r.json())
        .then(data => {
          ['bullet', 'lob', 'hook', 'beast'].forEach(s => {
            const b = document.getElementById('btn-shot-' + s);
            if (b) b.classList.toggle('active', s === style);
          });
          document.getElementById('angle-slider').value = data.yaw;
          updateBadge(data.yaw);
          const styleDescriptions = {
            'bullet': '🎯 BULLET: Flat Fast Laser (Yaw 0.0°, Pitch -3.0°)',
            'lob': '🌈 LOB: High Rainbow Arc (Yaw 0.0°, Pitch +1.5°)',
            'hook': '↩️ HOOK: Sharp Left Curve (Yaw -10.0°)',
            'beast': '⚡ BEAST: Wide Right Extreme (Yaw +10.0°)'
          };
          showToast(styleDescriptions[style] || ('Shot: ' + style.toUpperCase()));
        })
        .catch(err => console.error(err));
    }

    function onYawChange(val) {
      const yaw = parseFloat(val);
      updateBadge(yaw);
      ['bullet', 'lob', 'hook', 'beast'].forEach(s => {
        const b = document.getElementById('btn-shot-' + s);
        if (b) b.classList.remove('active');
      });
      fetch('/api/set_angle?yaw=' + yaw)
        .catch(err => console.error(err));
    }

    function updateBadge(yaw) {
      const badge = document.getElementById('angle-val-badge');
      badge.innerText = (yaw > 0 ? '+' : '') + yaw.toFixed(1) + '°';
    }

    const CAMS_ORDER = ['front_overview', 'side_cinematic', 'top_tactical', 'left_wrist_cam', 'right_wrist_cam'];
    function setCam(cam) {
      playSfx('click');
      fetch('/api/camera?mode=' + cam)
        .then(r => r.json())
        .then(data => {
          document.querySelectorAll('.cam-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.cam === cam);
          });
          const label = CAM_NAMES[cam] || cam;
          document.getElementById('cam-tag').innerText = label;
          showToast('Camera: ' + (VALID_CAMERAS[cam] || cam).toUpperCase());
        })
        .catch(err => console.error(err));
    }

    function togglePause() {
      playSfx('click');
      fetch('/api/pause')
        .then(r => r.json())
        .then(data => {
          const paused = data.paused;
          const txt = document.getElementById('pause-text');
          const btn = document.getElementById('pause-btn');
          if (txt) txt.innerText = paused ? 'RESUME' : 'PAUSE';
          if (btn) btn.classList.toggle('active', paused);
          showToast(paused ? 'Simulation Paused' : 'Simulation Resumed');
        })
        .catch(err => console.error(err));
    }

    function resetEpisode() {
      playSfx('click');
      fetch('/api/reset')
        .then(r => r.json())
        .then(data => {
          showToast('Episode Reset & Ready');
        })
        .catch(err => console.error(err));
    }

    function toggleFullscreen() {
      playSfx('click');
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => console.error(err));
      } else {
        document.exitFullscreen();
      }
    }

    function toggleHelp() {
      playSfx('click');
      document.getElementById('help-modal').classList.toggle('open');
    }

    // Telemetry Poller (10Hz)
    setInterval(() => {
      fetch('/api/telemetry')
        .then(r => r.json())
        .then(t => {
          if (!t) return;
          document.getElementById('tel-combo').innerText = t.combo + 'X 🔥';
          document.getElementById('tel-score').innerText = t.score + ' PTS';
          const shotMap = {
            'bullet': '🎯 BULLET',
            'lob': '🌈 LOB',
            'hook': '↩️ HOOK',
            'beast': '⚡ BEAST'
          };
          const shotEl = document.getElementById('tel-shot');
          if (shotEl && t.shot_style) {
            shotEl.innerText = shotMap[t.shot_style] || t.shot_style.toUpperCase();
          }
          document.getElementById('tel-phase').innerText = t.phase_name || ('P' + t.phase);
          document.getElementById('tel-step').innerText = String(t.step).padStart(4, '0') + ' / 250';
          document.getElementById('tel-speed').innerText = t.speed.toFixed(2) + ' m/s';
          document.getElementById('tel-dist').innerText = t.dist.toFixed(2) + ' m';
          
          const contactEl = document.getElementById('tel-contact');
          if (t.is_waiting) {
            contactEl.innerText = 'LOBBY READY';
            contactEl.className = 'tel-val val-green';
          } else {
            contactEl.innerText = t.contact ? 'LOCKED 🎯' : 'FREE';
            contactEl.className = 'tel-val ' + (t.contact ? 'val-green' : 'val-amber');
          }

          // Sound triggers on game events
          if (t.combo > prevCombo) {
            playSfx('catch');
          } else if (t.last_event && t.last_event.includes('MISSED') && t.last_event !== prevLastEvent) {
            playSfx('miss');
          }
          prevCombo = t.combo;
          prevLastEvent = t.last_event;

          // Update launch button appearance
          const fireBtn = document.getElementById('btn-fire');
          const fireTxt = document.getElementById('btn-fire-text');
          if (t.is_waiting) {
            if (!isWaitingLaunch) {
              isWaitingLaunch = true;
              fireBtn.classList.add('ready-pulse');
              fireTxt.innerText = '🚀 LAUNCH';
            }
          } else {
            if (isWaitingLaunch) {
              isWaitingLaunch = false;
              fireBtn.classList.remove('ready-pulse');
              fireTxt.innerText = 'IN FLIGHT...';
            }
          }
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
      else if (e.code === 'Space' || e.code === 'Enter') {
        e.preventDefault();
        if (isWaitingLaunch) {
          fireToss();
        } else {
          togglePause();
        }
      }
      else if (e.key === 'a' || e.key === 'A') toggleAutoLoop();
      else if (e.key === 'r' || e.key === 'R') resetEpisode();
      else if (e.key === 'p' || e.key === 'P') togglePause();
      else if (e.key === 'c' || e.key === 'C') setGameMode('chaos');
      else if (e.key === 's' || e.key === 'S' || e.key === 'm' || e.key === 'M') toggleSfx();
      else if (e.key === 'f' || e.key === 'F') toggleFullscreen();
      else if (e.key === '?' || (e.key === '/' && e.shiftKey)) toggleHelp();
      else if (e.key === 'Escape') {
        const modal = document.getElementById('help-modal');
        if (modal && modal.classList.contains('open')) toggleHelp();
      }
      else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        const slider = document.getElementById('angle-slider');
        if (slider) {
          const newYaw = Math.max(-15, Math.round(parseFloat(slider.value) - 1));
          slider.value = newYaw;
          onYawChange(newYaw);
        }
      }
      else if (e.key === 'ArrowRight') {
        e.preventDefault();
        const slider = document.getElementById('angle-slider');
        if (slider) {
          const newYaw = Math.min(15, Math.round(parseFloat(slider.value) + 1));
          slider.value = newYaw;
          onYawChange(newYaw);
        }
      }
    });
  </script>
</body>
</html>
"""

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global camera_mode, speed_scale, is_paused, force_reset, is_waiting_trigger, target_toss_yaw_deg, target_toss_pitch_deg, toss_mode, auto_loop, game_mode, shot_style
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
            with telemetry_lock:
                current_telemetry["camera"] = camera_mode
            self._send_json({"ok": True, "camera": camera_mode, "name": VALID_CAMERAS.get(camera_mode, "")})

        elif self.path.startswith("/api/mode"):
            if "mode=chaos" in self.path:
                game_mode = "chaos"
                toss_mode = "chaos"
                auto_loop = True
                is_waiting_trigger = False
                target_toss_yaw_deg = float(np.degrees(np.random.uniform(-0.16, 0.16)))
            elif "mode=auto" in self.path:
                game_mode = "auto"
                toss_mode = "manual"
                auto_loop = True
                is_waiting_trigger = False
                target_toss_yaw_deg = 0.0
                target_toss_pitch_deg = 0.0
            else:  # manual
                game_mode = "manual"
                auto_loop = False
                toss_mode = "manual"
                is_waiting_trigger = True
            with telemetry_lock:
                current_telemetry["game_mode"] = game_mode
                current_telemetry["auto_loop"] = auto_loop
                current_telemetry["toss_mode"] = toss_mode
                current_telemetry["is_waiting_trigger"] = is_waiting_trigger
                current_telemetry["is_waiting"] = is_waiting_trigger
                current_telemetry["target_yaw"] = target_toss_yaw_deg
            self._send_json({"ok": True, "mode": game_mode, "auto_loop": auto_loop, "is_waiting": is_waiting_trigger})

        elif self.path.startswith("/api/shot_style"):
            if "style=bullet" in self.path:
                shot_style = "bullet"
                target_toss_yaw_deg = 0.0
                target_toss_pitch_deg = -3.0
            elif "style=lob" in self.path:
                shot_style = "lob"
                target_toss_yaw_deg = 0.0
                target_toss_pitch_deg = 1.5
            elif "style=hook" in self.path:
                shot_style = "hook"
                target_toss_yaw_deg = -10.0
                target_toss_pitch_deg = -0.5
            elif "style=beast" in self.path:
                shot_style = "beast"
                target_toss_yaw_deg = 10.0
                target_toss_pitch_deg = 1.0
            toss_mode = "manual"
            with telemetry_lock:
                current_telemetry["shot_style"] = shot_style
                current_telemetry["target_yaw"] = target_toss_yaw_deg
                current_telemetry["target_pitch"] = target_toss_pitch_deg
                current_telemetry["toss_mode"] = toss_mode
                current_telemetry["phase_name"] = f"READY: {shot_style.upper()} (YAW {target_toss_yaw_deg:+.1f}°, PITCH {target_toss_pitch_deg:+.1f}°) [SPACE]"
            self._send_json({"ok": True, "style": shot_style, "yaw": target_toss_yaw_deg, "pitch": target_toss_pitch_deg})

        elif self.path.startswith("/api/speed"):
            if "scale=0.25" in self.path:
                speed_scale = 0.25
            elif "scale=0.5" in self.path:
                speed_scale = 0.5
            else:
                speed_scale = 1.0
            with telemetry_lock:
                current_telemetry["speed_scale"] = speed_scale
            self._send_json({"ok": True, "speed": speed_scale})

        elif self.path.startswith("/api/pause"):
            is_paused = not is_paused
            with telemetry_lock:
                current_telemetry["paused"] = is_paused
            self._send_json({"ok": True, "paused": is_paused})

        elif self.path.startswith("/api/fire"):
            is_waiting_trigger = False
            with telemetry_lock:
                current_telemetry["is_waiting_trigger"] = False
                current_telemetry["is_waiting"] = False
            self._send_json({"ok": True, "firing": True})

        elif self.path.startswith("/api/set_angle"):
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            if "yaw" in qs:
                target_toss_yaw_deg = float(qs["yaw"][0])
            if "pitch" in qs:
                target_toss_pitch_deg = float(qs["pitch"][0])
            toss_mode = "manual"
            with telemetry_lock:
                current_telemetry["target_yaw"] = target_toss_yaw_deg
                current_telemetry["toss_mode"] = toss_mode
                current_telemetry["phase_name"] = f"READY: YAW {target_toss_yaw_deg:+.1f}° (CLICK LAUNCH OR SPACE)"
            self._send_json({"ok": True, "yaw": target_toss_yaw_deg, "pitch": target_toss_pitch_deg, "mode": toss_mode})

        elif self.path.startswith("/api/preset"):
            if "mode=left" in self.path:
                target_toss_yaw_deg = -8.0
                target_toss_pitch_deg = 0.0
                toss_mode = "manual"
            elif "mode=right" in self.path:
                target_toss_yaw_deg = 8.0
                target_toss_pitch_deg = 0.0
                toss_mode = "manual"
            elif "mode=center" in self.path or "mode=straight" in self.path:
                target_toss_yaw_deg = 0.0
                target_toss_pitch_deg = 0.0
                toss_mode = "manual"
            elif "mode=chaos" in self.path:
                toss_mode = "chaos"
                target_toss_yaw_deg = float(np.degrees(np.random.uniform(-0.16, 0.16)))
            with telemetry_lock:
                current_telemetry["target_yaw"] = target_toss_yaw_deg
                current_telemetry["toss_mode"] = toss_mode
                current_telemetry["phase_name"] = f"READY: YAW {target_toss_yaw_deg:+.1f}° (CLICK LAUNCH OR SPACE)"
            self._send_json({"ok": True, "mode": toss_mode, "yaw": target_toss_yaw_deg})

        elif self.path.startswith("/api/autoloop"):
            if "enable=1" in self.path or "enable=true" in self.path:
                auto_loop = True
            elif "enable=0" in self.path or "enable=false" in self.path:
                auto_loop = False
            else:
                auto_loop = not auto_loop
            self._send_json({"ok": True, "auto_loop": auto_loop})

        elif self.path.startswith("/api/reset"):
            force_reset = True
            is_waiting_trigger = True
            with telemetry_lock:
                current_telemetry["is_waiting_trigger"] = True
                current_telemetry["is_waiting"] = True
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
    global is_waiting_trigger, auto_loop, target_toss_yaw_deg, target_toss_pitch_deg, toss_mode, combo_streak, total_score, last_event, game_mode, shot_style
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
        
        # Configure user-directed or continuous chaos angle
        if game_mode == "chaos" or toss_mode == "chaos":
            chosen_yaw = float(np.random.uniform(-0.16, 0.16))
            chosen_pitch = float(np.random.uniform(-0.02, 0.05))
            target_toss_yaw_deg = float(np.degrees(chosen_yaw))
            target_toss_pitch_deg = float(np.degrees(chosen_pitch))
            is_waiting_trigger = False  # Continuous fire in Chaos Rush!
        elif game_mode == "auto":
            chosen_yaw = 0.0
            chosen_pitch = 0.0
            is_waiting_trigger = False
        else:
            chosen_yaw = float(np.radians(target_toss_yaw_deg))
            chosen_pitch = float(np.radians(target_toss_pitch_deg))
            
        obs, info = env.reset(options={"toss_yaw": chosen_yaw, "toss_pitch": chosen_pitch})
        step = 0
        force_reset = False

        # If waiting for user fire trigger (and auto_loop is off), stay in ready stance
        ready_telemetry = {
            "step": 0,
            "phase": 0,
            "phase_name": f"READY TO TOSS (YAW {np.degrees(chosen_yaw):+.1f}°)",
            "proj_pos": unwrapped.data.site_xpos[unwrapped.site_projectile].copy(),
            "proj_vel": np.zeros(3),
            "speed": 0.0,
            "dist": float(np.linalg.norm(unwrapped.data.site_xpos[unwrapped.site_projectile] - unwrapped.data.site_xpos[unwrapped.site_right_catch])),
            "dist_to_catch": float(np.linalg.norm(unwrapped.data.site_xpos[unwrapped.site_projectile] - unwrapped.data.site_xpos[unwrapped.site_right_catch])),
            "reward": 0.0,
            "contact": False,
            "is_success": False,
            "camera": camera_mode,
            "speed_scale": speed_scale,
            "paused": True
        }
        
        # Render and hold ready frame
        ready_frame = engine.render_frame(ready_telemetry, camera_name=camera_mode)
        _, encoded = cv2.imencode(".jpg", ready_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with frame_lock:
            latest_jpeg = encoded.tobytes()

        last_rendered_cam = camera_mode
        last_rendered_yaw = chosen_yaw
        last_rendered_pitch = chosen_pitch
        last_render_time = time.time()

        while running and is_waiting_trigger and not auto_loop:
            if force_reset:
                force_reset = False
                break

            # Live update toss angle & pitch while waiting in lobby!
            current_target_yaw = float(np.radians(target_toss_yaw_deg))
            current_target_pitch = float(np.radians(target_toss_pitch_deg))
                
            if abs(current_target_yaw - chosen_yaw) > 1e-4 or abs(current_target_pitch - chosen_pitch) > 1e-4:
                chosen_yaw = current_target_yaw
                chosen_pitch = current_target_pitch
                obs, info = env.reset(options={"toss_yaw": chosen_yaw, "toss_pitch": chosen_pitch})

            now = time.time()
            if camera_mode != last_rendered_cam or abs(chosen_yaw - last_rendered_yaw) > 1e-4 or abs(chosen_pitch - last_rendered_pitch) > 1e-4 or (now - last_render_time > 0.05):
                ready_telemetry["camera"] = camera_mode
                ready_telemetry["phase_name"] = f"READY: {shot_style.upper()} ({np.degrees(chosen_yaw):+.1f}°, {np.degrees(chosen_pitch):+.1f}°) [SPACE]"
                ready_telemetry["proj_pos"] = unwrapped.data.site_xpos[unwrapped.site_projectile].copy()
                ready_telemetry["dist"] = float(np.linalg.norm(unwrapped.data.site_xpos[unwrapped.site_projectile] - unwrapped.data.site_xpos[unwrapped.site_right_catch]))
                ready_telemetry["dist_to_catch"] = ready_telemetry["dist"]
                ready_frame = engine.render_frame(ready_telemetry, camera_name=camera_mode)
                _, encoded = cv2.imencode(".jpg", ready_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with frame_lock:
                    latest_jpeg = encoded.tobytes()
                last_rendered_cam = camera_mode
                last_rendered_yaw = chosen_yaw
                last_rendered_pitch = chosen_pitch
                last_render_time = now

            with telemetry_lock:
                current_telemetry.update({
                    "step": 0,
                    "phase": 0,
                    "phase_name": f"READY: {shot_style.upper()} ({np.degrees(chosen_yaw):+.1f}°) [SPACE]",
                    "speed": 0.0,
                    "dist": ready_telemetry["dist"],
                    "reward": 0.0,
                    "contact": False,
                    "is_success": False,
                    "camera": camera_mode,
                    "speed_scale": speed_scale,
                    "paused": True,
                    "is_waiting_trigger": is_waiting_trigger,
                    "is_waiting": is_waiting_trigger,
                    "auto_loop": auto_loop,
                    "shot_style": shot_style,
                    "target_yaw": float(np.degrees(chosen_yaw)),
                    "target_pitch": float(np.degrees(chosen_pitch)),
                    "toss_mode": toss_mode,
                    "combo": combo_streak,
                    "score": total_score,
                    "last_event": last_event
                })
            time.sleep(0.02)

        is_waiting_trigger = False
        with telemetry_lock:
            current_telemetry["is_waiting_trigger"] = False
            current_telemetry["is_waiting"] = False

        # Simulation Step Loop
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
                    "paused": is_paused,
                    "is_waiting_trigger": is_waiting_trigger,
                    "is_waiting": is_waiting_trigger,
                    "auto_loop": auto_loop,
                    "target_yaw": float(np.degrees(chosen_yaw)),
                    "toss_mode": toss_mode,
                    "combo": combo_streak,
                    "score": total_score,
                    "last_event": last_event
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
                # Update Gamification Streak & Score
                if is_success:
                    combo_streak += 1
                    gained = 100 * combo_streak
                    total_score += gained
                    last_event = f"🔥 PERFECT CATCH! {combo_streak}X STREAK (+{gained} PTS)"
                else:
                    combo_streak = 0
                    last_event = "💔 MISSED! STREAK RESET"
                
                with telemetry_lock:
                    current_telemetry["combo"] = combo_streak
                    current_telemetry["score"] = total_score
                    current_telemetry["last_event"] = last_event
                    
                time.sleep(0.35)
                # Next episode: if game_mode is manual and not auto_loop, wait for next user trigger!
                if game_mode == "manual" and not auto_loop:
                    is_waiting_trigger = True
                else:
                    is_waiting_trigger = False
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
    import argparse
    parser = argparse.ArgumentParser(description="ALOHA 14-DOF Bimanual Dynamic Toss & Catch | HWIHWA LAB")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"), help="Web server host (default: 127.0.0.1 or 0.0.0.0 in Docker/Spaces)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860 if os.environ.get("SPACE_ID") else 8080)), help="Web server port (default: 8080 or 7860 in Spaces)")
    parser.add_argument("--no-browser", action="store_true", help="Do not launch browser window automatically")
    args = parser.parse_args()

    host = args.host
    port = args.port
    server = None
    for p in range(port, port + 10):
        try:
            server = ThreadingHTTPServer((host, p), StreamingHandler)
            port = p
            break
        except OSError:
            continue
    if server is None:
        server = ThreadingHTTPServer((host, 0), StreamingHandler)
        port = server.server_port

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    model_path = "checkpoints/aloha_toss_catch_final.zip"
    sim_thread = threading.Thread(target=simulation_worker, args=(model_path,), daemon=True)
    sim_thread.start()

    should_launch_browser = not args.no_browser and not os.environ.get("SPACE_ID")
    if should_launch_browser:
        time.sleep(0.8)
        launch_app_window(port)

    display_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print("\n=================================================================")
    print("🚀 ALOHA 14-DOF Bimanual Dynamic Toss & Catch | HWIHWA LAB")
    print(f"• Local URL   : http://{display_host}:{port}/")
    print("• Multi-Thread: ThreadingHTTPServer Active (Instant API Response)")
    print("• Control Dock: Floating Ergonomic Bottom Toolbar")
    print("• Cameras     : 5-Angle Full Studio (1:Front, 2:Side, 3:Top, 4:L-Wrist, 5:R-Wrist)")
    print("• Shortcuts   : Space/Enter (Fire/Pause), A (Auto), R (Reset), P (Pause), F (Fullscreen)")
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
