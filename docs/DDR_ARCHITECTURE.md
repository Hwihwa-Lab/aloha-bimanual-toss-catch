# 🏛️ Design Decision Records (DDR) & Architecture Specification

## DDR-001: 14-DOF Bimanual Kinematic & Dynamic Decomposition
* **Context:** Standard robot arms in Gymnasium (e.g. Fetch, Panda) focus on single-arm quasi-static manipulation. ALOHA 14-DOF dual-arm allows asymmetric task specialisation.
* **Decision:** We designate the **Left Arm (7-DOF)** as the High-Acceleration Ballistic Thrower with 2-finger servo gripper, and the **Right Arm (7-DOF)** as the Compliant Receptor equipped with a soft cushion paddle (`cushion_geom`, high friction 2.0, high contact damping).
* **Consequence:** Eliminates the need for dual-gripper finger precision in catching, shifting the learning burden to spatial intercept and damping control.

---

## DDR-002: 4-Phase State Machine & Continuous Reward Shaping
* **Context:** Pure sparse reward ($+1$ for catch, $0$ otherwise) in a 16D continuous action space leads to a vanishing gradient and zero exploration efficiency.
* **Decision:** We implement a 4-phase state machine inside `PhaseRewardEngine`:
  1. `Phase 0 (Grasp)`: Attract left hand to projectile, reward closing fingers.
  2. `Phase 1 (Toss)`: Reward velocity alignment toward apex ($Z = 1.25\text{m}$, $X = 0\text{m}$).
  3. `Phase 2 (Flight)`: Predict parabolic trajectory, reward right catcher pre-positioning.
  4. `Phase 3 (Cushion)`: Detect touch sensor trigger, penalize relative velocity, award $+50$ bonus on stable retention.
* **Consequence:** Solves exploration failure while maintaining clean continuous transitions between ballistic flight and contact retention.

---

## DDR-003: Headless EGL Offscreen 60fps Telemetry HUD
* **Context:** Standard Gymnasium `render(mode="human")` fails on remote cloud instances (Google Colab, headless servers).
* **Decision:** We utilize `MUJOCO_GL="egl"` combined with MuJoCo's native `Renderer` and OpenCV overlay engine (`render/hud_overlay.py`).
* **Consequence:** Guarantees 1080p 60fps telemetry overlays (phase banner, ballistics dashboard, wrist PiP cameras) renderable on both local Windows and Colab cloud servers without GUI display servers (X11).

---

## DDR-004: Domain Randomization for Sim-to-Real Transfer
* **Context:** Rigid physics policies overfit to exact initial positions and mass parameters.
* **Decision:** In each `reset()`, initial projectile position is perturbed by $\pm 1\text{cm}$, surface friction randomized, and contact solref damping perturbed.
* **Consequence:** Produces robust, disturbance-resilient control policies ready for physical ALOHA hardware deployment.

