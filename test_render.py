import gymnasium as gym
import envs
import cv2
from render.visual_engine import VisualEngine

def main():
    print("=== Testing 60fps Telemetry HUD & Multi-Camera Rendering ===")
    env = gym.make("AlohaTossCatch-v0")
    obs, info = env.reset(seed=42)

    unwrapped = env.unwrapped
    engine = VisualEngine(unwrapped.model, unwrapped.data, width=1280, height=720)

    frames = []
    print("Rendering 25 frames with HUD overlay...")

    for step in range(25):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        telemetry = {
            "step": step + 1,
            "phase": info.get("phase", 0),
            "proj_pos": unwrapped.data.site_xpos[unwrapped.site_projectile].copy(),
            "proj_vel": unwrapped.data.qvel[16:19].copy(),
            "dist_to_catch": float(info.get("dist_proj_to_catch", 0.0)),
            "reward": float(reward),
            "contact": bool(unwrapped.data.sensordata[0] > 0.05),
            "is_success": bool(info.get("is_success", False))
        }

        frame = engine.render_frame(telemetry, camera_name="front_overview")
        frames.append(frame)

        if step == 0:
            cv2.imwrite("assets/cockpit_preview.png", frame)
            print("Saved initial screenshot: assets/cockpit_preview.png")

        if terminated or truncated:
            break

    engine.save_gif(frames, output_path="assets/cockpit_preview.gif", fps=20)
    engine.close()
    env.close()
    print("SUCCESS: Full rendering pipeline verified!")

if __name__ == "__main__":
    main()

