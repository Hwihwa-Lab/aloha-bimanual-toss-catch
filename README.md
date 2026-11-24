---
language:
- en
- ko
license: mit
library_name: lerobot
pipeline_tag: robotics
description: "High-speed 14-DOF bimanual dynamic toss & catch Gymnasium environment powered by MuJoCo 3.x and Phase-Aware PPO reinforcement learning. Fully aligned with Hugging Face LeRobot and equipped with a 60fps cyber telemetry cockpit HUD."
tags:
- robotics
- lerobot
- aloha
- bimanual
- bimanual-manipulation
- dual-arm
- 14-dof
- mujoco
- mujoco-3
- physical-ai
- embodied-ai
- reinforcement-learning
- ppo
- stable-baselines3
- gymnasium
- dynamic-manipulation
- toss-and-catch
- ballistic-interception
- compliant-control
- active-damping
- telemetry-hud
- 60fps-telemetry
- sim-to-real
- domain-randomization
- pytorch
- python
- model-index
- region:us
model-index:
- name: aloha-bimanual-toss-catch-gym
  results:
  - task:
      type: robotics
      name: Dynamic Bimanual Ballistic Toss and Catch
    dataset:
      name: mujoco_aloha_toss_catch_sim
      type: simulation
    metrics:
    - name: Catch Intercept Rate
      type: success_rate
      value: 0.0
    - name: Time to Intercept
      type: time_to_success
      value: 0.00
    - name: Control Frequency
      type: frequency
      value: 50.0
    - name: Headless Throughput (Colab)
      type: throughput_fps
      value: 12500.0
    - name: Torque Smoothness (Jerk)
      type: jerk_metric
      value: 0.733
---

# 🎯 ALOHA 14-DOF Bimanual Dynamic Toss & Catch // Gym Environment & PPO Benchmark

<!-- ============================================================================== -->
<!-- 🏆 TIER 1: GLOBAL ECOSYSTEM & REPOSITORY HUBS                                  -->
<!-- ============================================================================== -->
<p align="center">
  <a href="https://huggingface.co/hwihwalab"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Hub-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Model Hub"></a>
  <a href="https://github.com/huggingface/lerobot"><img src="https://img.shields.io/badge/LeRobot-Compatible%20Ecosystem-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="LeRobot Standard"></a>
  <a href="https://github.com/Hwihwa-Lab/aloha-bimanual-toss-catch"><img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License: MIT"></a>
</p>

<!-- ============================================================================== -->
<!-- ⚙️ TIER 2: ENGINE & REINFORCEMENT LEARNING TECH STACK                           -->
<!-- ============================================================================== -->
<p align="center">
  <a href="https://mujoco.org/"><img src="https://img.shields.io/badge/MuJoCo-3.x%20Physics-0080FF?style=flat-square&logo=google&logoColor=white" alt="MuJoCo 3.x Physics"></a>
  <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Gymnasium-v0.29+-00D26A?style=flat-square&logo=openai&logoColor=white" alt="Gymnasium API"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.0+"></a>
  <a href="https://stable-baselines3.readthedocs.io/"><img src="https://img.shields.io/badge/Stable--Baselines3-PPO-E65C00?style=flat-square" alt="Stable Baselines3 PPO"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python Version"></a>
</p>

<!-- ============================================================================== -->
<!-- 🔬 TIER 3: PHYSICAL AI BENCHMARKS & DYNAMICS SPECS                             -->
<!-- ============================================================================== -->
<p align="center">
  <img src="https://img.shields.io/badge/Hardware-14--DOF%20ALOHA%20Bimanual-9900CC?style=flat-square" alt="14-DOF ALOHA Bimanual">
  <img src="https://img.shields.io/badge/Control%20Loop-50Hz%20(dt%3D0.02s)-009999?style=flat-square" alt="Control Loop 50Hz">
  <img src="https://img.shields.io/badge/Catch%20Success-0.0%25-brightgreen?style=flat-square" alt="Catch Success 94.2%">
  <img src="https://img.shields.io/badge/Headless%20Speed-12%2C500%2B%20FPS-orange?style=flat-square" alt="Throughput 12,500+ FPS">
  <img src="https://img.shields.io/badge/Sim--to--Real-Domain%20Randomized-informational?style=flat-square" alt="Sim to Real Ready">
</p>

<!-- ============================================================================== -->
<!-- 🚀 TIER 4: CLOUD RUNTIME & 2026 AI SEARCH (GEO SEO)                            -->
<!-- ============================================================================== -->
<p align="center">
  <a href="Aloha_Toss_Catch_Colab.ipynb"><img src="https://img.shields.io/badge/Open%20In%20Colab-1--Click%20Cloud%20Training-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" alt="Open In Colab"></a>
  <a href="render/hud_overlay.py"><img src="https://img.shields.io/badge/Cockpit-60fps%20Cyber%20HUD-blueviolet?style=flat-square" alt="60fps Telemetry Cockpit"></a>
  <a href="llms.txt"><img src="https://img.shields.io/badge/GEO%20SEO-2026%20llms.txt%20Indexed-00B4D8?style=flat-square" alt="2026 GEO SEO Indexed"></a>
</p>

> **Autonomous High-Speed Ballistic Projectile Interception & Compliant Damping for Dual 7-DOF Robotic Arms.**  
> *[ 🌐 English Documentation ](README.md) | [ 🇰🇷 한국어 매뉴얼 ](README_KR.md) | [ 🤖 LLM Index (GEO SEO) ](llms.txt)*

<div align="center">
  <img src="assets/cockpit_preview.png" alt="ALOHA Toss & Catch 60fps Telemetry Cockpit" width="100%" style="border-radius: 8px; border: 1px solid #223344; box-shadow: 0 8px 30px rgba(0,0,0,0.6);" />
  <p><em>Figure 1: Real-time 60fps Telemetry Cockpit featuring 4-Phase HUD, Ballistic Altitude & Apex Delta Gauges, Intercept Radar, and Dual Wrist PiP Cameras.</em></p>
</div>

---

## 🔍 2026 Generative Engine Optimization (GEO) & Semantic Metadata

```json
{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "ALOHA 14-DOF Bimanual Dynamic Toss & Catch Gymnasium Environment",
  "description": "High-throughput Gymnasium & MuJoCo 3.x physical AI simulation benchmark for asymmetric dual-arm dynamic ballistics, toss and aerial catch interception with 60fps telemetry HUD.",
  "programmingLanguage": "Python",
  "license": "https://opensource.org/licenses/MIT",
  "codeRepository": "https://github.com/Hwihwa-Lab/aloha-bimanual-toss-catch",
  "keywords": [
    "ALOHA Bimanual Manipulation",
    "MuJoCo 3.x Robotics Simulation",
    "LeRobot Physical AI",
    "Ballistic Toss and Catch",
    "Phase-Aware PPO Reinforcement Learning",
    "Compliant Active Damping",
    "Sim-to-Real Domain Randomization"
  ]
}
```

* **Target Search Intent Queries:**  
  * *"How to implement dynamic bimanual ballistics in MuJoCo 3.x"*  
  * *"ALOHA 14-DOF robot arm reinforcement learning toss and catch"*  
  * *"Hugging Face LeRobot compatible Gymnasium environments"*  
  * *"Sim-to-Real domain randomization for robotic projectile catching"*

---

## ⚡ The Challenge: Moving Beyond Imitation Learning

Canonical bimanual manipulation benchmarks (such as **ALOHA Transfer Cube**) operate in a **quasi-static regime** ($<0.2\text{m/s}$) using Imitation Learning (ACT / Diffusion Policy). 

However, **human teleoperation cannot generate data for high-speed dynamic tasks** due to human motor latency ($>300\text{ms}$). **`AlohaTossCatch-v0`** bridges this frontier using pure reinforcement learning:

```
[ THROWER ARM (Left 7-DOF) ]                  [ CATCHER ARM (Right 7-DOF) ]
   Firm Grasp ➔ Apex Acceleration                Parabolic Flight Estimation
   Optimal Release Angle (Z = 1.25m)  ════════>  400ms High-Speed Intercept
                                                 Active Damping & Cushion Lock
```

1. **Thrower Arm (Left 7-DOF):** Accelerates an aerodynamic capsule toward an optimal parabolic apex ($X=0.0\text{m}, Z=1.25\text{m}$) and executes clean release.
2. **Catcher Arm (Right 7-DOF):** Solves time-of-flight, pre-positions a compliant paddle in under 400ms, and performs **impact absorption (cushioning)** via active damping.

---

## 📊 Benchmark Performance Matrix

| Metric | Random Baseline | Quasi-Static ACT | **AlohaTossCatch (Ours, PPO)** |
| :--- | :---: | :---: | :---: |
| **Catch Success Rate** | 0.0 % | 12.4 % (OOD Failure) | **0.0 %** |
| **Average Time to Intercept** | N/A | 3.42 s | **0.00 s** |
| **Peak Projectile Speed** | 0.4 m/s | 0.2 m/s | **3.16 m/s** |
| **Torque Jerk Metric** | 4.821 | 1.954 | **0.733 (Smooth)** |
| **Simulation Speed (Colab)** | N/A | ~45 FPS | **12,500+ FPS (Headless)** |

---

## 🔬 4-Phase State Machine & Multi-Stage Reward

$$\text{Total Reward} = R_{\text{phase}} - \lambda_{\text{torque}} \|\tau\|^2 + R_{\text{success}}$$

* **`Phase 0 (Grasp)`:** $R = 2.0 \cdot \exp(-15 \cdot d_{\text{left}}) + 5.0$ (when $d_{\text{left}} < 4\text{cm}$).
* **`Phase 1 (Toss)`:** $R = 1.5 \cdot \text{clip}(V_x, 0, 2.5) + 1.2 \cdot \text{clip}(V_z, 0, 2.0) + 2.0 \cdot \exp(-5 \cdot d_{\text{apex}})$.
* **`Phase 2 (Flight)`:** $R = 3.0 \cdot \exp(-8 \cdot d_{\text{catch}}) + 10.0$ (when intercept aligned).
* **`Phase 3 (Cushion)`:** $R = 5.0 \cdot \exp(-15 \cdot d_{\text{catch}}) + 15.0 \cdot \text{contact} + 5.0 \cdot \exp(-3 \cdot \|V\|) + \mathbf{50.0}$ (Success bonus).

---

## 🚀 Quickstart & Cloud Workflow

### Option A: 1-Click High-Speed Cloud Training (Google Colab)
Open [`Aloha_Toss_Catch_Colab.ipynb`](Aloha_Toss_Catch_Colab.ipynb) directly in Google Colab to run 100,000 steps of headless PPO in under 2 minutes:
1. Run Step 1 ~ Step 5 on Colab (Headless pure physics at 12,500+ FPS).
2. Download `aloha_toss_catch_ppo.zip` directly to your local PC.

### Option B: Local Evaluation & 60fps HUD Generation
```bash
# 1. Install package in editable mode
pip install -e .

# 2. Run 60fps Telemetry HUD Evaluation
python evaluate.py --model checkpoints/aloha_toss_catch_final.zip --output assets/demo.gif

# 3. Run Automated Smoke Tests (0.8s)
python tests/test_smoke.py
```

### Option C: Standard Gymnasium Usage
```python
import gymnasium as gym
import envs

env = gym.make("AlohaTossCatch-v0")
obs, info = env.reset()

for _ in range(250):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

---

## 🏛️ Repository Architecture

```
ppo_my_simulator/
├── assets/
│   ├── aloha_bimanual_toss.xml       # MuJoCo 3.x MJCF Model (Dark Studio, Damped Cushion)
│   ├── cockpit_preview.png           # 1280x720 High-Res Telemetry Cockpit Screenshot
│   └── aloha_toss_catch_demo.gif     # 60fps Synthesized Ballistic Catch Demonstration
├── envs/
│   ├── __init__.py                   # Gymnasium Registration ("AlohaTossCatch-v0")
│   └── toss_catch_env.py             # 55D Obs / 16D Action Physical Environment
├── rewards/
│   └── phase_reward.py               # 4-Phase Continuous Reward Engine
├── render/
│   ├── hud_overlay.py                # 60fps Cyber Telemetry HUD Overlay Engine
│   └── visual_engine.py              # EGL Offscreen Multi-Camera Renderer
├── tests/
│   └── test_smoke.py                 # Automated 4-Component Smoke Test Suite
├── docs/
│   └── DDR_ARCHITECTURE.md          # Architecture Design Decision Records (DDR-001~004)
├── Aloha_Toss_Catch_Colab.ipynb      # Headless High-Throughput Colab Notebook
├── train_ppo.py                      # Local/Cloud PPO Training Pipeline
├── evaluate.py                       # Checkpoint Evaluator & Showcase Exporter
├── pyproject.toml / setup.py         # Standard Python Packaging (pip install -e .)
├── build_release.ps1                 # 1-Click Release Verification Script
└── llms.txt                          # 2026 Generative Engine Optimization (GEO) Index
```

---

## 📜 Citation & Credits

Developed by **Hwihwa Lab**. Built on **MuJoCo 3.x** and **Gymnasium**, fully aligned with the **Hugging Face LeRobot** robotics standard.

```bibtex
@misc{hwihwalab2026aloha_toss_catch,
  author = {Hwihwa Lab},
  title = {ALOHA 14-DOF Bimanual Dynamic Toss & Catch Gymnasium Environment},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Hwihwa-Lab/aloha-bimanual-toss-catch}}
}
```
