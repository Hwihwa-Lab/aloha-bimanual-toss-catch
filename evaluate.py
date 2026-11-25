"""
Evaluation, Empirical Benchmarking & High-Fidelity Rendering Script.
Evaluates trained PPO policy over N episodes, computes real physical metrics,
generates 60fps Telemetry Cockpit GIF/Video, and optionally updates README metrics with ground truth.
"""
import os
import re
import argparse
import numpy as np
import gymnasium as gym
import envs
from stable_baselines3 import PPO
from render.visual_engine import VisualEngine

def update_readme_files(success_rate, mean_time, mean_jerk, peak_speed):
    """
    Safely injects empirically measured metrics into README.md and README_KR.md.
    """
    target_files = ["README.md", "README_KR.md"]
    for fname in target_files:
        if not os.path.exists(fname):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Update YAML Frontmatter metrics
        content = re.sub(
            r"(name:\s*Catch Intercept Rate\s*\n\s*type:\s*success_rate\s*\n\s*value:\s*)[\d\.]+",
            rf"\g<1>{success_rate:.1f}",
            content
        )
        content = re.sub(
            r"(name:\s*Time to Intercept\s*\n\s*type:\s*time_to_success\s*\n\s*value:\s*)[\d\.]+",
            rf"\g<1>{mean_time:.2f}",
            content
        )
        content = re.sub(
            r"(name:\s*Torque Smoothness \(Jerk\)\s*\n\s*type:\s*jerk_metric\s*\n\s*value:\s*)[\d\.]+",
            rf"\g<1>{mean_jerk:.3f}",
            content
        )

        # 2. Update Shields.io Badges
        content = re.sub(
            r"Catch%20Success-[\d\.]+%25-brightgreen",
            f"Catch%20Success-{success_rate:.1f}%25-brightgreen",
            content
        )
        content = re.sub(
            r"캐칭%20성공률-[\d\.]+%25-brightgreen",
            f"캐칭%20성공률-{success_rate:.1f}%25-brightgreen",
            content
        )

        # 3. Update Benchmark Tables
        content = re.sub(
            r"(\|\s*\*\*Catch Success Rate\*\*\s*\|[^\|]+\|[^\|]+\|\s*\*\*)[\d\.]+(\s*%\*\*\s*\|)",
            rf"\g<1>{success_rate:.1f}\g<2>",
            content
        )
        content = re.sub(
            r"(\|\s*\*\*캐칭 성공률\*\*\s*\|[^\|]+\|[^\|]+\|\s*\*\*)[\d\.]+(\s*%\*\*\s*\|)",
            rf"\g<1>{success_rate:.1f}\g<2>",
            content
        )

        content = re.sub(
            r"(\|\s*\*\*Average Time to Intercept\*\*\s*\|[^\|]+\|[^\|]+\|\s*\*\*)[\d\.]+(\s*s\*\*\s*\|)",
            rf"\g<1>{mean_time:.2f}\g<2>",
            content
        )
        content = re.sub(
            r"(\|\s*\*\*평균 요격 소요 시간\*\*\s*\|[^\|]+\|[^\|]+\|\s*\*\*)[\d\.]+(\s*초\*\*\s*\|)",
            rf"\g<1>{mean_time:.2f}\g<2>",
            content
        )

        content = re.sub(
            r"(\|\s*\*\*Peak Projectile Speed\*\*\s*\|[^\|]+\|[^\|]+\|\s*\*\*)[\d\.]+(\s*m/s\*\*\s*\|)",
            rf"\g<1>{peak_speed:.2f}\g<2>",
            content
        )
        content = re.sub(
            r"(\|\s*\*\*비행 물체 최고 속도\*\*\s*\|[^\|]+\|[^\|]+\|\s*\*\*)[\d\.]+(\s*m/s\*\*\s*\|)",
            rf"\g<1>{peak_speed:.2f}\g<2>",
            content
        )

        content = re.sub(
            r"(\|\s*\*\*Torque Jerk Metric\*\*\s*\|[^\|]+\|[^\|]+\|\s*\*\*)[\d\.]+(\s*\(Smooth\)\*\*\s*\|)",
            rf"\g<1>{mean_jerk:.3f}\g<2>",
            content
        )
        content = re.sub(
            r"(\|\s*\*\*모터 저크\(Jerk\) 부드러움\*\*\s*\|[^\|]+\|[^\|]+\|\s*\*\*)[\d\.]+(\s*\(매우 부드러움\)\*\*\s*\|)",
            rf"\g<1>{mean_jerk:.3f}\g<2>",
            content
        )

        with open(fname, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[Auto-Benchmark] Updated {fname} with empirical results!")


def evaluate(model_path="checkpoints/aloha_toss_catch_final.zip", num_episodes=10, output_gif="assets/aloha_toss_catch_demo.gif", update_readme=False):
    print("======================================================================")
    print(f"🚀 Running Empirical Benchmark & Evaluation on AlohaTossCatch-v0")
    print(f"• Model Path     : {model_path}")
    print(f"• Test Episodes  : {num_episodes}")
    print(f"• Update README  : {update_readme}")
    print("======================================================================")

    env = gym.make("AlohaTossCatch-v0")
    
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print("✅ Loaded trained PPO policy checkpoint successfully!")
    else:
        print(f"⚠️ Warning: Checkpoint '{model_path}' not found. Evaluating random policy baseline.")
        model = None

    unwrapped = env.unwrapped

    all_frames = []
    success_count = 0
    intercept_times = []
    peak_speeds = []
    jerks = []
    episode_records = []

    # Pass 1: Run physical simulation to evaluate policy and find the best showcase episode
    for ep in range(num_episodes):
        seed_val = 1000 + ep
        obs, info = env.reset(seed=seed_val)
        ep_reward = 0.0
        step = 0
        prev_torques = None
        ep_peak_speed = 0.0

        while True:
            step += 1
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Track peak velocity
            proj_vel = unwrapped.data.qvel[16:19]
            speed = float(np.linalg.norm(proj_vel))
            if speed > ep_peak_speed:
                ep_peak_speed = speed

            # Track torque jerk
            current_torques = unwrapped.data.actuator_force.copy()
            if prev_torques is not None:
                jerk_val = float(np.mean(np.abs(current_torques - prev_torques)))
                jerks.append(jerk_val)
            prev_torques = current_torques

            if terminated or truncated:
                is_succ = bool(info.get("is_success", False))
                if is_succ:
                    success_count += 1
                    intercept_times.append(step * 0.02)  # 50Hz control loop = 0.02s per step

                peak_speeds.append(ep_peak_speed)
                episode_records.append({
                    "ep": ep,
                    "seed": seed_val,
                    "reward": ep_reward,
                    "steps": step,
                    "is_success": is_succ
                })
                yaw_val = info.get('toss_yaw_deg', 0.0)
                print(f"  [Ep {ep+1:03d}/{num_episodes:03d}] Yaw: {yaw_val:+5.1f}° | Steps: {step:03d} | Reward: {ep_reward:+6.2f} | Success: {'YES' if is_succ else 'NO'} | Reason: {info.get('reason', info.get('termination_reason'))}")
                break

    # Select the best episode for visual showcase (Priority: is_success True -> highest reward)
    best_ep = max(episode_records, key=lambda x: (x["is_success"], x["reward"]))
    print(f"\n🎬 Rendering Best Showcase Episode: Ep {best_ep['ep']+1} (Seed: {best_ep['seed']}, Success: {best_ep['is_success']}, Reward: {best_ep['reward']:+.2f})")

    # Pass 2: Re-render the best episode in high-fidelity 60fps HUD
    engine = VisualEngine(unwrapped.model, unwrapped.data, width=1280, height=720)
    obs, info = env.reset(seed=best_ep["seed"])
    step = 0
    while True:
        step += 1
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        proj_vel = unwrapped.data.qvel[16:19]

        # Direct physics contact
        has_contact_visual = False
        for c_i in range(unwrapped.data.ncon):
            con = unwrapped.data.contact[c_i]
            if (con.geom1 == unwrapped.proj_geom_id and con.geom2 in unwrapped.basket_geom_ids) or \
               (con.geom2 == unwrapped.proj_geom_id and con.geom1 in unwrapped.basket_geom_ids):
                has_contact_visual = True
                break

        telemetry = {
            "step": step,
            "phase": info.get("phase", 0),
            "proj_pos": unwrapped.data.site_xpos[unwrapped.site_projectile].copy(),
            "proj_vel": proj_vel.copy(),
            "dist_to_catch": float(info.get("dist_proj_to_catch", 0.0)),
            "reward": float(reward),
            "contact": has_contact_visual,
            "is_success": bool(info.get("is_success", False))
        }

        frame = engine.render_frame(telemetry, camera_name="front_overview")
        all_frames.append(frame)

        if terminated or truncated:
            # If successfully caught, simulate & render 30 post-catch hold frames (~1.0s at 30fps)
            if best_ep["is_success"]:
                import mujoco
                print("🏆 Success detected! Rendering 30 post-catch hold frames to showcase secure basket retention...")
                for hold_step in range(1, 31):
                    for _ in range(unwrapped.frame_skip):
                        mujoco.mj_step(unwrapped.model, unwrapped.data)
                    
                    proj_vel = unwrapped.data.qvel[16:19]
                    has_hold_contact = False
                    for c_i in range(unwrapped.data.ncon):
                        con = unwrapped.data.contact[c_i]
                        if (con.geom1 == unwrapped.proj_geom_id and con.geom2 in unwrapped.basket_geom_ids) or \
                           (con.geom2 == unwrapped.proj_geom_id and con.geom1 in unwrapped.basket_geom_ids):
                            has_hold_contact = True
                            break

                    hold_telemetry = {
                        "step": step + hold_step,
                        "phase": 3,
                        "proj_pos": unwrapped.data.site_xpos[unwrapped.site_projectile].copy(),
                        "proj_vel": proj_vel.copy(),
                        "dist_to_catch": float(np.linalg.norm(unwrapped.data.site_xpos[unwrapped.site_projectile] - unwrapped.data.site_xpos[unwrapped.site_right_catch])),
                        "reward": float(reward),
                        "contact": has_hold_contact,
                        "is_success": True
                    }
                    hold_frame = engine.render_frame(hold_telemetry, camera_name="front_overview")
                    all_frames.append(hold_frame)
            break

    # Save visual showcase GIF & Preview PNG
    if len(all_frames) > 0:
        import cv2
        # Save preview PNG showing the ball resting in basket
        preview_frame_idx = max(0, len(all_frames) - 10)
        cv2.imwrite("assets/cockpit_preview.png", all_frames[preview_frame_idx])
        print("📸 Saved high-resolution preview screenshot to assets/cockpit_preview.png")

        if output_gif:
            engine.save_gif(all_frames, output_path=output_gif, fps=30)
            # Also sync cockpit_preview.gif
            engine.save_gif(all_frames, output_path="assets/cockpit_preview.gif", fps=30)

    engine.close()
    env.close()

    # Calculate empirical statistics
    actual_sr = (success_count / num_episodes) * 100.0
    actual_time = float(np.mean(intercept_times)) if intercept_times else 0.0
    actual_jerk = float(np.mean(jerks)) if jerks else 0.0
    actual_speed = float(np.max(peak_speeds)) if peak_speeds else 0.0

    print("\n======================================================================")
    print(f"📊 EMPIRICAL BENCHMARK RESULTS ({num_episodes} Test Episodes)")
    print("======================================================================")
    print(f"• Intercept Success Rate : {actual_sr:6.1f} % ({success_count}/{num_episodes})")
    print(f"• Mean Time to Intercept : {actual_time:6.2f} s")
    print(f"• Peak Projectile Speed  : {actual_speed:6.2f} m/s")
    print(f"• Mean Torque Jerk Metric: {actual_jerk:6.3f}")
    print("======================================================================")

    if update_readme:
        update_readme_files(actual_sr, actual_time, actual_jerk, actual_speed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/aloha_toss_catch_final.zip")
    parser.add_argument("--episodes", type=int, default=5, help="Number of benchmark test episodes")
    parser.add_argument("--output", type=str, default="assets/aloha_toss_catch_demo.gif")
    parser.add_argument("--update_readme", action="store_true", help="Auto-inject real metrics into README.md & README_KR.md")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        output_gif=args.output,
        update_readme=args.update_readme
    )
