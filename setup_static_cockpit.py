"""
Sets up the 100% Free Zero-Cold-Start Static Cyber Cockpit for Hugging Face Spaces.
Generates:
  - index.html
  - cockpit_sim.js
"""
from pathlib import Path
import json

root = Path(__file__).resolve().parent

# 1. Read existing HTML from run_live_cockpit.py
with open(root / "run_live_cockpit.py", "r", encoding="utf-8") as f:
    text = f.read()

body_start = text.find('</head>')
body_end = text.find('<script>')
body_html = text[body_start:body_end].strip()

# Replace img with canvas in body
body_html = body_html.replace(
    '<img id="stream-img" src="/video_feed" alt="Live MuJoCo Stream">',
    '<canvas id="simCanvas" width="1280" height="720" style="width:100%; height:100%; object-fit:contain; display:block;"></canvas>'
)

# Update action buttons with direct links
body_html = body_html.replace(
    'href="#" onclick="showToast(\'Download bundle from Hugging Face repository\'); return false;"',
    'href="https://huggingface.co/hwihwalab/aloha-bimanual-toss-catch-model/resolve/main/aloha_bimanual_toss_catch_bundle.zip" target="_blank" download'
)
body_html = body_html.replace(
    'href="https://huggingface.co/"',
    'href="https://huggingface.co/hwihwalab/aloha-bimanual-toss-catch-model"'
)
body_html = body_html.replace(
    'href="https://github.com/"',
    'href="https://github.com/Hwihwa-Lab/aloha-bimanual-toss-catch"'
)

index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ALOHA 14-DOF Bimanual Dynamic Toss & Catch | HWIHWA LAB</title>
  <link rel="stylesheet" href="style.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
{body_html}
  <script src="cockpit_sim.js"></script>
</body>
</html>
"""

with open(root / "index.html", "w", encoding="utf-8") as f:
    f.write(index_html)
print("[✓] index.html generated successfully.")

# 2. Generate cockpit_sim.js
cockpit_js = """/**
 * ALOHA 14-DOF Bimanual Dynamic Toss & Catch - Interactive Cyber Cockpit Simulator
 * 60 FPS Photorealistic Web Canvas Engine // HWIHWA LAB
 */

(function () {
  'use strict';

  // Canvas & Context
  const canvas = document.getElementById('simCanvas');
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const CW = 1280;
  const CH = 720;

  // Grid Config (High-Fidelity 960x540)
  const FW = 960;
  const FH = 540;
  const COLS = 6;
  const ROWS = 6;
  const TOTAL_FRAMES = 36;

  // Simulation State
  let telemetryData = null;
  let sprites = {}; // key: `${style}_${cam}` -> Image
  let isReady = false;
  let isWaitingTrigger = true;
  let isPlaying = false;
  let isPaused = false;
  let currentFrame = 0;
  let animTimer = null;
  let lastFrameTime = performance.now();

  // User Selection
  let gameMode = 'manual'; // 'manual', 'auto', 'chaos'
  let shotStyle = 'bullet'; // 'bullet', 'lob', 'hook', 'beast'
  let currentCam = 'front_overview';
  let aimYaw = 0.0;
  let comboStreak = 0;
  let totalScore = 0;
  let lastEvent = 'READY TO LAUNCH';
  let sfxEnabled = true;

  const STYLES = ['bullet', 'lob', 'hook', 'beast'];
  const CAMS = ['front_overview', 'side_cinematic', 'top_tactical'];
  const CAM_NAMES = {
    'front_overview': 'Front Overview',
    'side_cinematic': 'Side Cinematic',
    'top_tactical': 'Top Tactical',
    'left_wrist_cam': 'Left Wrist PiP',
    'right_wrist_cam': 'Right Wrist PiP'
  };

  // Web Audio SFX Synthesizer
  let audioCtx = null;
  function getAudioCtx() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    return audioCtx;
  }

  function playSfx(type) {
    if (!sfxEnabled) return;
    try {
      const actx = getAudioCtx();
      const t = actx.currentTime;

      if (type === 'launch') {
        const osc = actx.createOscillator();
        const gain = actx.createGain();
        osc.connect(gain);
        gain.connect(actx.destination);
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(120, t);
        osc.frequency.exponentialRampToValueAtTime(780, t + 0.22);
        gain.gain.setValueAtTime(0.2, t);
        gain.gain.linearRampToValueAtTime(0.001, t + 0.22);
        osc.start(t);
        osc.stop(t + 0.22);
      } else if (type === 'catch') {
        // Futuristic success chord
        [523.25, 659.25, 783.99, 1046.5].forEach((freq, i) => {
          const osc = actx.createOscillator();
          const gain = actx.createGain();
          osc.connect(gain);
          gain.connect(actx.destination);
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(freq, t + i * 0.04);
          gain.gain.setValueAtTime(0.18, t + i * 0.04);
          gain.gain.exponentialRampToValueAtTime(0.001, t + i * 0.04 + 0.35);
          osc.start(t + i * 0.04);
          osc.stop(t + i * 0.04 + 0.35);
        });
      } else if (type === 'click') {
        const osc = actx.createOscillator();
        const gain = actx.createGain();
        osc.connect(gain);
        gain.connect(actx.destination);
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, t);
        gain.gain.setValueAtTime(0.08, t);
        gain.gain.linearRampToValueAtTime(0.001, t + 0.04);
        osc.start(t);
        osc.stop(t + 0.04);
      }
    } catch (e) {}
  }

  // Load Metadata & Sprites
  function initAssets() {
    fetch('assets/telemetry_data.json')
      .then(r => r.json())
      .then(data => {
        telemetryData = data.telemetry;
        loadSprites();
      })
      .catch(err => {
        console.error('Failed to load telemetry JSON:', err);
      });
  }

  function loadSprites() {
    let toLoad = STYLES.length * CAMS.length;
    let loaded = 0;

    STYLES.forEach(style => {
      CAMS.forEach(cam => {
        const key = `${style}_${cam}`;
        const img = new Image();
        img.src = `assets/frames/${key}.jpg`;
        img.onload = () => {
          loaded++;
          if (loaded >= toLoad) {
            isReady = true;
            renderFrame(0);
            updateUI();
            console.log('[Cockpit] All 60fps trajectory sprites loaded successfully!');
          }
        };
        img.onerror = () => {
          loaded++;
          if (loaded >= toLoad) {
            isReady = true;
            renderFrame(0);
          }
        };
        sprites[key] = img;
      });
    });
  }

  // Draw Specific Frame to 1280x720 Canvas
  function renderFrame(frameIdx) {
    if (!isReady) {
      ctx.fillStyle = '#06080d';
      ctx.fillRect(0, 0, CW, CH);
      ctx.fillStyle = '#00f0ff';
      ctx.font = '700 16px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('INITIALIZING 60FPS CYBER TELEMETRY ENGINE...', CW / 2, CH / 2);
      return;
    }

    const activeCam = (currentCam === 'left_wrist_cam' || currentCam === 'right_wrist_cam') ? 'front_overview' : currentCam;
    const key = `${shotStyle}_${activeCam}`;
    const sprite = sprites[key] || sprites[`bullet_front_overview`];

    if (!sprite || !sprite.complete) return;

    const col = frameIdx % COLS;
    const row = Math.floor(frameIdx / COLS);
    const sx = col * FW;
    const sy = row * FH;

    ctx.drawImage(sprite, sx, sy, FW, FH, 0, 0, CW, CH);

    // Update Telemetry Bar
    updateTelemetryHUD(frameIdx);
  }

  function updateTelemetryHUD(frameIdx) {
    const list = telemetryData ? telemetryData[shotStyle] : null;
    let t = null;
    if (list && frameIdx > 0 && frameIdx <= list.length) {
      t = list[frameIdx - 1];
    }

    const phaseEl = document.getElementById('tel-phase');
    const speedEl = document.getElementById('tel-speed');
    const distEl = document.getElementById('tel-dist');
    const comboEl = document.getElementById('tel-combo');
    const scoreEl = document.getElementById('tel-score');
    const eventEl = document.getElementById('tel-event');

    if (t) {
      if (phaseEl) phaseEl.innerText = t.phase_name;
      if (speedEl) speedEl.innerText = t.speed.toFixed(2) + ' m/s';
      if (distEl) distEl.innerText = t.dist.toFixed(3) + ' m';
    } else {
      if (phaseEl) phaseEl.innerText = isWaitingTrigger ? `READY: ${shotStyle.toUpperCase()} (${aimYaw >= 0 ? '+' : ''}${aimYaw.toFixed(1)}°) [SPACE]` : 'INITIALIZING';
      if (speedEl) speedEl.innerText = '0.00 m/s';
      if (distEl) distEl.innerText = '0.528 m';
    }

    if (comboEl) comboEl.innerText = comboStreak + 'X';
    if (scoreEl) scoreEl.innerText = totalScore.toLocaleString();
    if (eventEl) eventEl.innerText = lastEvent;
  }

  // Animation Loop (60 FPS)
  function stepSimulation(now) {
    if (!isPlaying) return;

    if (isPaused) {
      animTimer = requestAnimationFrame(stepSimulation);
      return;
    }

    const elapsed = now - lastFrameTime;
    const frameInterval = 28; // ~35 FPS playback gives perfect physical hanging time

    if (elapsed >= frameInterval) {
      lastFrameTime = now;
      currentFrame++;

      const list = telemetryData ? telemetryData[shotStyle] : null;
      const maxF = list ? list.length : TOTAL_FRAMES;

      if (currentFrame >= maxF) {
        // Episode Completed
        currentFrame = maxF - 1;
        isPlaying = false;
        isWaitingTrigger = (gameMode === 'manual');

        // Combo & Score Update
        comboStreak++;
        const gain = 100 * comboStreak;
        totalScore += gain;
        lastEvent = `🔥 PERFECT CATCH! ${comboStreak}X STREAK (+${gain} PTS)`;
        playSfx('catch');
        showToast(lastEvent);

        renderFrame(currentFrame);
        updateUI();

        // Auto Loop or Chaos Next Round Trigger
        if (gameMode === 'auto' || gameMode === 'chaos') {
          setTimeout(() => {
            if (gameMode === 'chaos') {
              // Pick random shot & angle in chaos mode
              const randomStyle = STYLES[Math.floor(Math.random() * STYLES.length)];
              setShotStyle(randomStyle, false);
              const randYaw = (Math.random() * 20.0 - 10.0);
              onYawChange(randYaw);
            }
            fireToss();
          }, 450);
        }
        return;
      }

      renderFrame(currentFrame);
    }

    animTimer = requestAnimationFrame(stepSimulation);
  }

  // User Actions
  function fireToss() {
    if (isPlaying) return;
    playSfx('launch');
    isWaitingTrigger = false;
    isPlaying = true;
    isPaused = false;
    currentFrame = 0;
    lastFrameTime = performance.now();
    showToast('🚀 BALL LAUNCHED! INTERCEPTING...');

    const btn = document.getElementById('btn-fire');
    if (btn) btn.classList.remove('ready-pulse');
    const txt = document.getElementById('btn-fire-text');
    if (txt) txt.innerText = 'IN FLIGHT...';

    if (animTimer) cancelAnimationFrame(animTimer);
    animTimer = requestAnimationFrame(stepSimulation);
  }

  function togglePause() {
    playSfx('click');
    isPaused = !isPaused;
    const txt = document.getElementById('pause-text');
    const btn = document.getElementById('pause-btn');
    if (txt) txt.innerText = isPaused ? 'RESUME' : 'PAUSE';
    if (btn) btn.classList.toggle('active', isPaused);
    showToast(isPaused ? 'Simulation Paused' : 'Simulation Resumed');
  }

  function resetEpisode() {
    playSfx('click');
    if (animTimer) cancelAnimationFrame(animTimer);
    isPlaying = false;
    isPaused = false;
    isWaitingTrigger = true;
    currentFrame = 0;
    lastEvent = 'EPISODE RESET: READY TO LAUNCH';
    renderFrame(0);
    updateUI();
    showToast('Episode Reset & Ready');
  }

  function setGameMode(mode) {
    playSfx('click');
    gameMode = mode;
    ['manual', 'auto', 'chaos'].forEach(m => {
      const b = document.getElementById('btn-mode-' + m);
      if (b) b.classList.toggle('active', m === mode);
    });

    const shotDeck = document.getElementById('shot-deck');
    if (mode === 'chaos') {
      if (shotDeck) {
        shotDeck.style.opacity = '0.35';
        shotDeck.style.pointerEvents = 'none';
      }
      showToast('🔥 CHAOS RUSH: CONTINUOUS RANDOM STREAM ACTIVATED!');
      if (!isPlaying) fireToss();
    } else if (mode === 'auto') {
      if (shotDeck) {
        shotDeck.style.opacity = '1.0';
        shotDeck.style.pointerEvents = 'auto';
      }
      showToast('Game Mode: AUTO LOOP');
      if (!isPlaying) fireToss();
    } else {
      if (shotDeck) {
        shotDeck.style.opacity = '1.0';
        shotDeck.style.pointerEvents = 'auto';
      }
      showToast('Game Mode: MANUAL');
    }
  }

  function setShotStyle(style, notify = true) {
    if (notify) playSfx('click');
    shotStyle = style;
    STYLES.forEach(s => {
      const b = document.getElementById('btn-shot-' + s);
      if (b) b.classList.toggle('active', s === style);
    });

    const defaultYaws = { 'bullet': 0.0, 'lob': 0.0, 'hook': -10.0, 'beast': 10.0 };
    aimYaw = defaultYaws[style] || 0.0;
    const slider = document.getElementById('angle-slider');
    if (slider) slider.value = aimYaw;
    updateBadge(aimYaw);

    const descriptions = {
      'bullet': '🎯 BULLET: Flat Fast Laser (Yaw 0.0°, Pitch -3.0°)',
      'lob': '🌈 LOB: High Rainbow Arc (Yaw 0.0°, Pitch +1.5°)',
      'hook': '↩️ HOOK: Sharp Left Curve (Yaw -10.0°)',
      'beast': '⚡ BEAST: Wide Right Extreme (Yaw +10.0°)'
    };
    if (notify) showToast(descriptions[style] || ('Shot: ' + style.toUpperCase()));

    if (!isPlaying) renderFrame(0);
  }

  function onYawChange(val) {
    aimYaw = parseFloat(val);
    updateBadge(aimYaw);
    STYLES.forEach(s => {
      const b = document.getElementById('btn-shot-' + s);
      if (b) b.classList.remove('active');
    });
    if (!isPlaying) renderFrame(0);
  }

  function updateBadge(yaw) {
    const badge = document.getElementById('angle-val-badge');
    if (badge) badge.innerText = (yaw >= 0 ? '+' : '') + yaw.toFixed(1) + '°';
  }

  function setCam(cam) {
    playSfx('click');
    currentCam = cam;
    document.querySelectorAll('.cam-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.cam === cam);
    });
    const tag = document.getElementById('cam-tag');
    if (tag) tag.innerText = CAM_NAMES[cam] || cam;
    showToast('Camera: ' + (CAM_NAMES[cam] || cam).toUpperCase());
    renderFrame(currentFrame);
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

  function toggleHelp() {
    playSfx('click');
    const modal = document.getElementById('help-modal');
    if (modal) {
      modal.classList.toggle('open');
    }
  }

  function toggleFullscreen() {
    playSfx('click');
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => console.error(err));
    } else {
      document.exitFullscreen();
    }
  }

  function showToast(msg) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.innerText = msg;
    toast.classList.add('show');
    clearTimeout(window._toastTimer);
    window._toastTimer = setTimeout(() => toast.classList.remove('show'), 1600);
  }

  function updateUI() {
    const fireBtn = document.getElementById('btn-fire');
    const fireTxt = document.getElementById('btn-fire-text');
    if (fireBtn) {
      if (isWaitingTrigger && !isPlaying) {
        fireBtn.classList.add('ready-pulse');
        if (fireTxt) fireTxt.innerText = 'LAUNCH';
      } else {
        fireBtn.classList.remove('ready-pulse');
        if (fireTxt) fireTxt.innerText = isPlaying ? 'IN FLIGHT...' : 'READY';
      }
    }
    const pauseTxt = document.getElementById('pause-text');
    if (pauseTxt) pauseTxt.innerText = isPaused ? 'RESUME' : 'PAUSE';
  }

  // Keyboard Shortcuts Binding
  function bindHotkeys() {
    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      const k = e.key;

      if (k === ' ' || k === 'Enter') {
        e.preventDefault();
        if (isWaitingTrigger || !isPlaying) fireToss();
        else togglePause();
      } else if (k === 'p' || k === 'P') {
        e.preventDefault();
        togglePause();
      } else if (k === 'r' || k === 'R') {
        e.preventDefault();
        resetEpisode();
      } else if (k === 'a' || k === 'A') {
        e.preventDefault();
        setGameMode(gameMode === 'auto' ? 'manual' : 'auto');
      } else if (k === 'c' || k === 'C') {
        e.preventDefault();
        setGameMode(gameMode === 'chaos' ? 'manual' : 'chaos');
      } else if (k === '1') {
        setCam('front_overview');
      } else if (k === '2') {
        setCam('side_cinematic');
      } else if (k === '3') {
        setCam('top_tactical');
      } else if (k === '4') {
        setCam('left_wrist_cam');
      } else if (k === '5') {
        setCam('right_wrist_cam');
      } else if (k === 'ArrowLeft') {
        e.preventDefault();
        onYawChange(Math.max(-15.0, aimYaw - 1.0));
      } else if (k === 'ArrowRight') {
        e.preventDefault();
        onYawChange(Math.min(15.0, aimYaw + 1.0));
      } else if (k === 's' || k === 'S' || k === 'm' || k === 'M') {
        toggleSfx();
      } else if (k === 'f' || k === 'F') {
        toggleFullscreen();
      } else if (k === '?' || (k === '/' && e.shiftKey)) {
        e.preventDefault();
        toggleHelp();
      } else if (k === 'Escape') {
        const modal = document.getElementById('help-modal');
        if (modal && modal.classList.contains('open')) {
          e.preventDefault();
          toggleHelp();
        }
      }
    });
  }

  // Event Listeners Initialization
  function initListeners() {
    // Fire button
    const fireBtn = document.getElementById('btn-fire');
    if (fireBtn) fireBtn.onclick = fireToss;

    // Pause & Reset
    const pauseBtn = document.getElementById('pause-btn');
    if (pauseBtn) pauseBtn.onclick = togglePause;
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) resetBtn.onclick = resetEpisode;

    // Game Modes
    ['manual', 'auto', 'chaos'].forEach(m => {
      const b = document.getElementById('btn-mode-' + m);
      if (b) b.onclick = () => setGameMode(m);
    });

    // Shot Styles
    STYLES.forEach(s => {
      const b = document.getElementById('btn-shot-' + s);
      if (b) b.onclick = () => setShotStyle(s);
    });

    // Angle Slider
    const slider = document.getElementById('angle-slider');
    if (slider) slider.oninput = (e) => onYawChange(e.target.value);

    // Cameras
    document.querySelectorAll('.cam-btn').forEach(b => {
      b.onclick = () => setCam(b.dataset.cam);
    });

    // Header buttons
    const sfxBtn = document.getElementById('header-sfx-btn');
    if (sfxBtn) sfxBtn.onclick = toggleSfx;

    // Hotkeys Help Modal toggle buttons
    document.querySelectorAll('.header-actions button').forEach(btn => {
      if (btn.textContent.includes('HOTKEYS') || btn.textContent.includes('?')) {
        btn.onclick = toggleHelp;
      }
    });
    const modalClose = document.querySelector('.modal-close-btn');
    if (modalClose) {
      modalClose.onclick = toggleHelp;
    }

    bindHotkeys();
  }

  // Expose global functions for inline HTML onclick handlers
  window.toggleHelp = toggleHelp;
  window.fireToss = fireToss;
  window.togglePause = togglePause;
  window.resetEpisode = resetEpisode;
  window.setGameMode = setGameMode;
  window.setShotStyle = setShotStyle;
  window.onYawChange = onYawChange;
  window.setCam = setCam;
  window.toggleSfx = toggleSfx;
  window.toggleFullscreen = toggleFullscreen;

  // Start initialization
  initListeners();
  initAssets();

})();
"""

with open(root / "cockpit_sim.js", "w", encoding="utf-8") as f:
    f.write(cockpit_js)
print("[✓] cockpit_sim.js generated successfully.")

