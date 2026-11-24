"""
Custom Gymnasium Environment: AlohaTossCatchEnv
A 14-DOF Bimanual Robotic Toss & Catch Dynamic Physical Environment powered by MuJoCo 3.x.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from rewards.phase_reward import PhaseRewardEngine

class AlohaTossCatchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, xml_path=None, render_mode=None, frame_skip=10, max_episode_steps=250):
        super().__init__()

        if xml_path is None:
            xml_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "assets",
                "aloha_bimanual_toss.xml"
            )

        self.xml_path = xml_path
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Load MuJoCo Model & Data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache ID lookups
        self.site_left_hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site")
        self.site_right_catch = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_catch_center")
        self.site_projectile = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "projectile_center")
        self.body_projectile = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "projectile")
        self.sensor_touch = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "sensor_catcher_touch")

        # Action Space: 16 actuators
        # Left: 7 arm motors + 2 finger positions
        # Right: 7 arm motors
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Observation Space: 55 continuous features
        # 9 left joint pos + 9 left joint vel = 18
        # 7 right joint pos + 7 right joint vel = 14
        # projectile pos(3) + vel(3) + quat(4) + angvel(3) = 13
        # left_hand_to_proj (3)
        # proj_to_right_catch (3)
        # phase one-hot (4)
        obs_dim = 18 + 14 + 13 + 3 + 3 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reward_engine = PhaseRewardEngine()
        self.trajectory_history = []
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.trajectory_history = []
        self.reward_engine.reset()

        # Reset MuJoCo physics state
        mujoco.mj_resetData(self.model, self.data)

        # Set default joint poses (Solved Inverse Kinematics for Toss & Catch)
        # Left Arm initial pose: holding cube at [-0.25, 0.0, 0.88]
        self.data.qpos[0] = 0.0    # left_waist
        self.data.qpos[1] = 0.866  # left_shoulder
        self.data.qpos[2] = 0.0    # left_shoulder_roll
        self.data.qpos[3] = 2.138  # left_elbow
        self.data.qpos[4] = 0.0    # left_forearm_roll
        self.data.qpos[5] = 1.050  # left_wrist_pitch
        self.data.qpos[6] = 0.0    # left_wrist_roll
        self.data.qpos[7] = 0.012  # finger1 closed
        self.data.qpos[8] = 0.012  # finger2 closed

        # Right Arm initial pose: ready at catching zone [0.25, 0.0, 0.88]
        self.data.qpos[9] = 0.0    # right_waist
        self.data.qpos[10] = -0.908 # right_shoulder
        self.data.qpos[11] = 0.0   # right_shoulder_roll
        self.data.qpos[12] = -2.176 # right_elbow
        self.data.qpos[13] = 0.0   # right_forearm_roll
        self.data.qpos[14] = -1.009 # right_wrist_pitch
        self.data.qpos[15] = 0.0   # right_wrist_roll

        # Domain Randomization: Projectile position slightly randomized (+- 1cm)
        proj_qpos_start = 16  # index of freejoint qpos
        noise_x = self.np_random.uniform(-0.01, 0.01)
        noise_y = self.np_random.uniform(-0.01, 0.01)
        self.data.qpos[proj_qpos_start + 0] = -0.25 + noise_x  # x
        self.data.qpos[proj_qpos_start + 1] = 0.0 + noise_y    # y
        self.data.qpos[proj_qpos_start + 2] = 0.88             # z
        self.data.qpos[proj_qpos_start + 3:proj_qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]

        # Zero out velocities
        self.data.qvel[:] = 0.0

        # Step forward to stabilize contacts
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {"phase": self.reward_engine.current_phase}
        return obs, info

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply action control
        # Map motor actions: [-1, 1] mapped to ctrl
        # Note fingers are position actuators: ctrl [0, 0.045]
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        # Arm motors: direct torque scaling
        ctrl[:7] = action[:7]
        # Gripper fingers: map [-1, 1] to [0.0, 0.045]
        ctrl[7] = (action[7] + 1.0) * 0.5 * 0.045
        ctrl[8] = (action[8] + 1.0) * 0.5 * 0.045
        # Right Arm motors
        ctrl[9:16] = action[9:16]

        self.data.ctrl[:] = ctrl

        # Advance physics simulation for frame_skip sub-steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Extract environment states
        left_hand_pos = self.data.site_xpos[self.site_left_hand].copy()
        right_catch_pos = self.data.site_xpos[self.site_right_catch].copy()
        proj_pos = self.data.site_xpos[self.site_projectile].copy()
        
        # Freejoint linear velocity
        proj_vel = self.data.qvel[16:19].copy()

        # Touch sensor check
        touch_val = self.data.sensordata[0] if self.model.nsensordata > 0 else 0.0
        has_contact = bool(touch_val > 0.05)

        # Gripper width
        gripper_w = float(self.data.qpos[7] + self.data.qpos[8])

        # Record trajectory point for visualization
        self.trajectory_history.append(proj_pos.copy())
        if len(self.trajectory_history) > 100:
            self.trajectory_history.pop(0)

        # Compute Phase-aware reward
        env_data = {
            "projectile_pos": proj_pos,
            "projectile_vel": proj_vel,
            "left_hand_pos": left_hand_pos,
            "right_catch_pos": right_catch_pos,
            "left_gripper_width": gripper_w,
            "catcher_contact": has_contact,
            "torques": self.data.actuator_force.copy(),
            "action_rate": float(np.mean(np.abs(action)))
        }

        reward, reward_info = self.reward_engine.compute(env_data)

        # Terminations & Truncations
        terminated = False
        truncated = False

        # Termination 1: Out of bounds / dropped below table height
        if proj_pos[2] < 0.68:
            reward -= 20.0
            terminated = True
            reward_info["termination_reason"] = "dropped_below_table"

        # Termination 2: Successfully secured in catcher
        if reward_info.get("is_success", False):
            terminated = True
            reward_info["termination_reason"] = "perfect_catch_success"

        # Truncation: Max episode steps reached
        if self.current_step >= self.max_episode_steps:
            truncated = True
            reward_info["termination_reason"] = "timeout"

        obs = self._get_obs()
        return obs, reward, terminated, truncated, reward_info

    def _get_obs(self):
        # 1. Left Arm: 9 qpos + 9 qvel = 18
        left_qpos = self.data.qpos[:9]
        left_qvel = self.data.qvel[:9]

        # 2. Right Arm: 7 qpos + 7 qvel = 14
        right_qpos = self.data.qpos[9:16]
        right_qvel = self.data.qvel[9:16]

        # 3. Projectile: 3 pos + 3 vel + 4 quat + 3 angvel = 13
        proj_pos = self.data.site_xpos[self.site_projectile]
        proj_vel = self.data.qvel[16:19]
        proj_quat = self.data.qpos[19:23]  # freejoint quat
        proj_angvel = self.data.qvel[19:22]

        # 4. Relative Vectors (3 + 3 = 6)
        left_hand_pos = self.data.site_xpos[self.site_left_hand]
        right_catch_pos = self.data.site_xpos[self.site_right_catch]
        left_to_proj = proj_pos - left_hand_pos
        proj_to_catch = right_catch_pos - proj_pos

        # 5. Phase One-Hot (4)
        phase_one_hot = np.zeros(4, dtype=np.float32)
        phase_one_hot[self.reward_engine.current_phase] = 1.0

        obs = np.concatenate([
            left_qpos, left_qvel,
            right_qpos, right_qvel,
            proj_pos, proj_vel, proj_quat, proj_angvel,
            left_to_proj, proj_to_catch,
            phase_one_hot
        ]).astype(np.float32)

        return obs

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera="front_overview")
            return self.renderer.render()
        elif self.render_mode == "human":
            pass

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
