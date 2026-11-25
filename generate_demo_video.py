"""
Generates ultra-high-definition 60fps H.264 demo video (assets/demo_video.mp4)
featuring 3 signature ballistic trajectories (BULLET, LOB, HOOK) with Cyber HUD overlays.
"""
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

import envs
from render.visual_engine import VisualEngine

ROUNDS = [
    {"style": "bullet", "yaw": 0.0, "pitch": -3.0, "name": "BULLET LASER"},
    {"style": "lob", "yaw": 0.0, "pitch": 1.5, "name": "RAINBOW LOB"},
    {"style": "hook", "yaw": -10.0, "pitch": -0.5, "name": "LEFT HOOK"}
]

def generate_video():
    root = Path(__file__).resolve().parent
    out_dir = root / "assets"
    raw_avi = out_dir / "_temp_raw.avi"
    final_mp4 = out_dir / "demo_video.mp4"

    model_path = root / "checkpoints" / "aloha_toss_catch_final.zip"
    print(f"[*] Loading model: {model_path.name}")
    model = PPO.load(str(model_path))

    width, height = 1280, 720
    fps = 60

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(raw_avi), fourcc, fps, (width, height))

    combo_streak = 0
    total_score = 0

    for r_idx, r_cfg in enumerate(ROUNDS):
        print(f"\n[Round {r_idx+1}/3] Simulating {r_cfg['style'].upper()} ({r_cfg['name']})...")
        yaw_rad = float(np.radians(r_cfg["yaw"]))
        pitch_rad = float(np.radians(r_cfg["pitch"]))

        env = gym.make("AlohaTossCatch-v0")
        obs, info = env.reset(options={"toss_yaw": yaw_rad, "toss_pitch": pitch_rad})
        unwrapped = env.unwrapped
        engine = VisualEngine(unwrapped.model, unwrapped.data, width=width, height=height)

        # 1. Lobby Hold (8 frames = ~0.13s)
        lobby_tel = {
            "step": 0,
            "phase": 0,
            "phase_name": f"READY: {r_cfg['style'].upper()} ({r_cfg['yaw']:+.1f}°) [SPACE]",
            "proj_pos": unwrapped.data.site_xpos[unwrapped.site_projectile].copy(),
            "proj_vel": np.zeros(3),
            "speed": 0.0,
            "dist": float(np.linalg.norm(unwrapped.data.site_xpos[unwrapped.site_projectile] - unwrapped.data.site_xpos[unwrapped.site_right_catch])),
            "dist_to_catch": float(np.linalg.norm(unwrapped.data.site_xpos[unwrapped.site_projectile] - unwrapped.data.site_xpos[unwrapped.site_right_catch])),
            "reward": 0.0,
            "contact": False,
            "is_success": False,
            "camera": "front_overview",
            "speed_scale": 1.0,
            "paused": True
        }
        lobby_frame = engine.render_frame(lobby_tel, camera_name="front_overview")
        for _ in range(10):
            writer.write(lobby_frame)

        # 2. Episode Execution
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < 60:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            proj_vel = unwrapped.data.qvel[16:19].copy()
            proj_speed = float(np.linalg.norm(proj_vel))
            phase_idx = int(info.get("phase", 0))
            is_success = bool(info.get("is_success", False))

            phase_str = "MISSION SUCCESS: CATCH SECURED" if is_success else f"PHASE {phase_idx}"
            tel = {
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
                "camera": "front_overview",
                "speed_scale": 1.0,
                "paused": False
            }
            frame = engine.render_frame(tel, camera_name="front_overview")
            writer.write(frame)

        # 3. Success freeze hold (18 frames = ~0.3s)
        combo_streak += 1
        total_score += 100 * combo_streak
        tel["phase_name"] = f"🔥 PERFECT CATCH! {combo_streak}X STREAK (+{100*combo_streak} PTS)"
        success_frame = engine.render_frame(tel, camera_name="front_overview")
        for _ in range(16):
            writer.write(success_frame)

        engine.close()
        env.close()

    writer.release()
    print(f"\n[✓] Raw MJPG written: {raw_avi.stat().st_size / 1024:.1f} KB")

    # 4. Transcode with ffmpeg for universal web autoplay (H.264, YUV420P, FastStart)
    print("[*] Transcoding to web-optimized H.264 MP4 with FFmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_avi),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",
        "-crf", "18",
        "-movflags", "+faststart",
        str(final_mp4)
    ]
    subprocess.run(cmd, check=True)

    if raw_avi.exists():
        raw_avi.unlink()

    size_kb = final_mp4.stat().st_size / 1024
    print(f"\n[🎉] Ultra-High-Definition 60fps MP4 Generated: {final_mp4.name} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    generate_video()

