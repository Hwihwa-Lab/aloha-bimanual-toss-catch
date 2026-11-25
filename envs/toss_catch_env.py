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

        # Collision Geoms for Direct Physics Contact Detection
        self.proj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "projectile_geom")
        self.basket_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["right_cushion_base", "right_rim_fwd", "right_rim_bwd", "right_rim_left", "right_rim_right"]
        ]

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

        # Kinematic Nominal Ballistic Toss Keyframes for Left Arm
        self.q_left_ready = np.array([0.0, 0.866, 0.0, 2.138, 0.0, 1.050, 0.0], dtype=np.float64)
        self.q_left_swing = np.array([0.0, 0.450, 0.0, 1.400, 0.0, 0.400, 0.0], dtype=np.float64)
        self.q_left_release = np.array([0.0, 0.100, 0.0, 0.600, 0.0, 0.050, 0.0], dtype=np.float64)

        # Optimal Upright Facing Catch Pose for Right Arm (Perfect Horizontal Bowl facing UP, Aligned at X=0.35, Z=0.88)
        self.q_right_catch_nominal = np.array([-0.4873, 1.1973, 0.0992, 2.4, -2.2945, -1.1899, -0.2695], dtype=np.float64)
        self.cur_q_right_target = self.q_right_catch_nominal.copy()
        self.cur_q_left_swing = self.q_left_swing.copy()
        self.cur_q_left_release = self.q_left_release.copy()

        # Action scales (Left: fine residual, Right: active compliant residual intercept)
        self.action_scale = np.array([
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,  # left arm fine residual
            0.0, 0.0,                                   # fingers (kinematic timed release)
            0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08   # right arm compliant residual intercept
        ], dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.trajectory_history = []
        self.reward_engine.reset()

        # Reset MuJoCo physics state
        mujoco.mj_resetData(self.model, self.data)

        # Set default joint poses
        self.data.qpos[:7] = self.q_left_ready.copy()
        self.data.qpos[7] = 0.0
        self.data.qpos[8] = 0.0
        self.data.qpos[9:16] = self.q_right_catch_nominal.copy()

        # Domain Randomization: Left Arm Variable Toss Angles & Projectile position
        options = options or {}
        if "toss_yaw" in options:
            self.toss_yaw = float(options["toss_yaw"])
        else:
            self.toss_yaw = self.np_random.uniform(-0.16, 0.16)   # ±9.2 degrees variable toss

        if "toss_pitch" in options:
            self.toss_pitch = float(options["toss_pitch"])
        else:
            self.toss_pitch = self.np_random.uniform(-0.015, 0.015)
        
        self.cur_q_left_swing = np.array([
            self.toss_yaw * 0.5, 0.450 + self.toss_pitch * 0.5, 0.0, 1.400, 0.0, 0.400, 0.0
        ], dtype=np.float64)
        self.cur_q_left_release = np.array([
            self.toss_yaw, 0.100 + self.toss_pitch, 0.0, 0.600, 0.0, 0.050, 0.0
        ], dtype=np.float64)
        self.cur_q_right_target = self.q_right_catch_nominal.copy()

        proj_qpos_start = 16  # index of freejoint qpos
        noise_x = self.np_random.uniform(-0.003, 0.003)
        noise_y = self.np_random.uniform(-0.003, 0.003)
        self.data.qpos[proj_qpos_start + 0] = -0.218 + noise_x  # x
        self.data.qpos[proj_qpos_start + 1] = 0.0 + noise_y    # y
        self.data.qpos[proj_qpos_start + 2] = 0.879            # z
        self.data.qpos[proj_qpos_start + 3:proj_qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]

        # Zero out velocities
        self.data.qvel[:] = 0.0

        # Initialize position actuator targets to match starting joint poses
        self.data.ctrl[:7] = self.q_left_ready.copy()
        self.data.ctrl[7] = 0.0
        self.data.ctrl[8] = 0.0
        self.data.ctrl[9:16] = self.q_right_catch_nominal.copy()

        # Step forward to stabilize contacts
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {
            "phase": self.reward_engine.current_phase,
            "toss_yaw_deg": float(np.degrees(self.toss_yaw)),
            "toss_pitch_deg": float(np.degrees(self.toss_pitch))
        }
        return obs, info

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1. Left Arm Dynamic Variable Toss Generator
        nominal_ctrl = np.zeros(16, dtype=np.float64)
        if self.current_step < 4:
            nominal_ctrl[:7] = self.q_left_ready
            nominal_ctrl[7] = 0.0
            nominal_ctrl[8] = 0.0
        elif self.current_step < 11:
            u = (self.current_step - 4) / 7.0
            s = 0.5 * (1.0 - np.cos(u * np.pi))
            nominal_ctrl[:7] = (1.0 - s) * self.q_left_ready + s * self.cur_q_left_swing
            nominal_ctrl[7] = 0.0
            nominal_ctrl[8] = 0.0
        elif self.current_step < 18:
            u = (self.current_step - 11) / 7.0
            s = 0.5 * (1.0 - np.cos(u * np.pi))
            nominal_ctrl[:7] = (1.0 - s) * self.cur_q_left_swing + s * self.cur_q_left_release
            nominal_ctrl[7] = 0.045
            nominal_ctrl[8] = 0.045
        else:
            nominal_ctrl[:7] = self.cur_q_left_release
            nominal_ctrl[7] = 0.045
            nominal_ctrl[8] = 0.045

        # 2. Right Arm Active Trajectory Interceptor (Multi-Axis Ballistic Predictor)
        proj_pos = self.data.site_xpos[self.site_projectile].copy()
        proj_vel = self.data.qvel[16:19].copy()

        if self.current_step >= 15:  # Airborne tracking phase
            g = 9.81
            z_catch = 0.8898
            vz = proj_vel[2]
            disc = vz**2 + 2.0 * g * max(0.0, proj_pos[2] - z_catch)
            if disc >= 0:
                t_impact = (vz + np.sqrt(disc)) / g
                pred_x = proj_pos[0] + proj_vel[0] * t_impact
                pred_y = proj_pos[1] + proj_vel[1] * t_impact

                # Active Interception: Waist tracks Y, Elbow tracks X
                delta_waist = -pred_y / 0.201
                delta_elbow = (pred_x - 0.30) / 0.222

                # Update nominal catching pose dynamically toward landing target
                self.cur_q_right_target[0] = self.q_right_catch_nominal[0] + np.clip(delta_waist, -1.2, 1.2)
                self.cur_q_right_target[3] = self.q_right_catch_nominal[3] + np.clip(delta_elbow * 0.5, -0.3, 0.3)
                self.cur_q_right_target[5] = self.q_right_catch_nominal[5] - np.clip(delta_elbow * 0.5, -0.3, 0.3) * 0.5

        nominal_ctrl[9:16] = self.cur_q_right_target

        # 3. Hybrid Nominal + Residual RL Control
        target_ctrl = nominal_ctrl + action * self.action_scale
        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(target_ctrl, ctrl_min, ctrl_max)

        # Advance physics simulation for frame_skip sub-steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Extract environment states
        left_hand_pos = self.data.site_xpos[self.site_left_hand].copy()
        right_catch_pos = self.data.site_xpos[self.site_right_catch].copy()
        proj_pos = self.data.site_xpos[self.site_projectile].copy()
        
        # Freejoint linear velocity
        proj_vel = self.data.qvel[16:19].copy()

        # Direct Physics Contact Check via MuJoCo Collision Engine
        has_contact = False
        for c_i in range(self.data.ncon):
            con = self.data.contact[c_i]
            g1, g2 = con.geom1, con.geom2
            if (g1 == self.proj_geom_id and g2 in self.basket_geom_ids) or \
               (g2 == self.proj_geom_id and g1 in self.basket_geom_ids):
                has_contact = True
                break
        
        # Fallback to site touch sensor if non-zero
        if not has_contact and self.model.nsensordata > 0:
            has_contact = bool(self.data.sensordata[0] > 0.05)

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
        reward_info["toss_yaw_deg"] = float(np.degrees(self.toss_yaw))
        reward_info["toss_pitch_deg"] = float(np.degrees(self.toss_pitch))

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
