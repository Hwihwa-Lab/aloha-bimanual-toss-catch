"""
Extracts photorealistic 60fps MuJoCo HUD frames and telemetry sequences for Static Web Cockpit.
Generates:
  - assets/frames/{style}_{cam}.jpg (Sprite grid of HUD-rendered frames)
  - assets/telemetry_data.json (Step-by-step real physics telemetry for all 4 shot styles)
"""
import os
import json
import time
from pathlib import Path
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

import envs
from render.visual_engine import VisualEngine

STYLES = {
    "bullet": {"yaw": 0.0, "pitch": -3.0, "name": "BULLET LASER"},
    "lob": {"yaw": 0.0, "pitch": 1.5, "name": "RAINBOW LOB"},
    "hook": {"yaw": -10.0, "pitch": -0.5, "name": "LEFT HOOK"},
    "beast": {"yaw": 10.0, "pitch": 1.0, "name": "RIGHT BEAST"}
}

CAMERAS = ["front_overview", "side_cinematic", "top_tactical"]

def extract_all():
    root_dir = Path(__file__).resolve().parent
    out_dir = root_dir / "assets" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading trained PPO model & environment for 60fps frame extraction...")
    model_path = root_dir / "checkpoints" / "aloha_toss_catch_final.zip"
    model = PPO.load(str(model_path))

    # High-Fidelity 4x Crisp Resolution (960x540 per frame, 92% JPEG Quality)
    fw, fh = 960, 540
    cols = 6
    rows = 6
    max_frames = cols * rows  # 36 frames per episode

    all_telemetry = {}

    for style_key, cfg in STYLES.items():
        yaw_rad = float(np.radians(cfg["yaw"]))
        pitch_rad = float(np.radians(cfg["pitch"]))
        print(f"\n=======================================================")
        print(f"🎯 Capturing Ballistic Trajectory: {style_key.upper()} ({cfg['name']})")
        print(f"   Yaw: {cfg['yaw']:+.1f} deg, Pitch: {cfg['pitch']:+.1f} deg")
        print("=======================================================")

        # Run rollout once to record actions and telemetry
        env = gym.make("AlohaTossCatch-v0")
        obs, info = env.reset(options={"toss_yaw": yaw_rad, "toss_pitch": pitch_rad})
        unwrapped = env.unwrapped

        actions = []
        step_telemetry = []

        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < max_frames:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            proj_vel = unwrapped.data.qvel[16:19].copy()
            proj_speed = float(np.linalg.norm(proj_vel))
            phase_idx = int(info.get("phase", 0))
            is_success = bool(info.get("is_success", False))

            step_telemetry.append({
                "step": step,
                "phase": phase_idx,
                "phase_name": "MISSION SUCCESS: CATCH SECURED" if is_success else f"PHASE {phase_idx}",
                "speed": round(proj_speed, 2),
                "dist": round(float(info.get("dist_proj_to_catch", 0.0)), 3),
                "reward": round(float(reward), 2),
                "contact": bool(unwrapped.data.sensordata[0] > 0.05),
                "is_success": is_success
            })

        env.close()
        num_steps = len(actions)
        print(f"  -> Trajectory completed in {num_steps} steps. Success: {step_telemetry[-1]['is_success']}")
        all_telemetry[style_key] = step_telemetry

        # Now render each camera view with VisualEngine
        for cam_name in CAMERAS:
            print(f"  📸 Rendering camera [{cam_name}] with 60fps HUD overlays...")
            env = gym.make("AlohaTossCatch-v0")
            obs, info = env.reset(options={"toss_yaw": yaw_rad, "toss_pitch": pitch_rad})
            unwrapped = env.unwrapped
            engine = VisualEngine(unwrapped.model, unwrapped.data, width=1280, height=720)

            frames = []

            # 1. Lobby ready frame (frame 0)
            ready_tel = {
                "step": 0,
                "phase": 0,
                "phase_name": f"READY: {style_key.upper()} ({cfg['yaw']:+.1f}°) [SPACE]",
                "proj_pos": unwrapped.data.site_xpos[unwrapped.site_projectile].copy(),
                "proj_vel": np.zeros(3),
                "speed": 0.0,
                "dist": float(np.linalg.norm(unwrapped.data.site_xpos[unwrapped.site_projectile] - unwrapped.data.site_xpos[unwrapped.site_right_catch])),
                "dist_to_catch": float(np.linalg.norm(unwrapped.data.site_xpos[unwrapped.site_projectile] - unwrapped.data.site_xpos[unwrapped.site_right_catch])),
                "reward": 0.0,
                "contact": False,
                "is_success": False,
                "camera": cam_name,
                "speed_scale": 1.0,
                "paused": True
            }
            f_bgr = engine.render_frame(ready_tel, camera_name=cam_name)
            frames.append(cv2.resize(f_bgr, (fw, fh)))

            # 2. Episode rollout frames
            for s_idx, act in enumerate(actions):
                obs, reward, _, _, info = env.step(act)
                tel = {
                    "step": s_idx + 1,
                    "phase": step_telemetry[s_idx]["phase"],
                    "phase_name": step_telemetry[s_idx]["phase_name"],
                    "proj_pos": unwrapped.data.site_xpos[unwrapped.site_projectile].copy(),
                    "proj_vel": unwrapped.data.qvel[16:19].copy(),
                    "speed": step_telemetry[s_idx]["speed"],
                    "dist": step_telemetry[s_idx]["dist"],
                    "dist_to_catch": step_telemetry[s_idx]["dist"],
                    "reward": step_telemetry[s_idx]["reward"],
                    "contact": step_telemetry[s_idx]["contact"],
                    "is_success": step_telemetry[s_idx]["is_success"],
                    "camera": cam_name,
                    "speed_scale": 1.0,
                    "paused": False
                }
                f_bgr = engine.render_frame(tel, camera_name=cam_name)
                frames.append(cv2.resize(f_bgr, (fw, fh)))

            engine.close()
            env.close()

            # Pad or truncate to max_frames
            while len(frames) < max_frames:
                frames.append(frames[-1].copy())
            frames = frames[:max_frames]

            # Assemble grid sprite
            grid_img = np.zeros((rows * fh, cols * fw, 3), dtype=np.uint8)
            for idx, frame in enumerate(frames):
                r = idx // cols
                c = idx % cols
                grid_img[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] = frame

            sprite_path = out_dir / f"{style_key}_{cam_name}.jpg"
            cv2.imwrite(str(sprite_path), grid_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            size_kb = sprite_path.stat().st_size / 1024
            print(f"     -> Saved sprite: {sprite_path.name} ({size_kb:.1f} KB)")

    # Save complete telemetry JSON
    meta = {
        "fw": fw,
        "fh": fh,
        "cols": cols,
        "rows": rows,
        "total_frames": max_frames,
        "styles": list(STYLES.keys()),
        "cameras": CAMERAS,
        "telemetry": all_telemetry
    }
    json_path = root_dir / "assets" / "telemetry_data.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[✓] Telemetry metadata saved to: {json_path}")
    print("[🎉] All 60fps trajectory sprites generated successfully!")

if __name__ == "__main__":
    extract_all()

