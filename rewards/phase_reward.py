"""
Phase-Aware Reward Engine for Bimanual Dynamic Toss & Catch.
Decomposes the continuous ballistic task into 4 distinct physical phases:
1. Grasping & Stabilization
2. Ballistic Acceleration & Clean Release
3. Flight Trajectory Tracking & Pre-positioning
4. Compliant Cushioning & Contact Retention
"""
import numpy as np

class PhaseRewardEngine:
    def __init__(self):
        # Target locations in world coordinates
        self.toss_launch_pos = np.array([-0.25, 0.0, 0.88])
        self.apex_target_pos = np.array([0.0, 0.0, 1.25])
        self.catch_target_pos = np.array([0.25, 0.0, 0.88])

        # Phase tracking
        self.current_phase = 0  # 0: GRASP, 1: TOSS, 2: FLIGHT, 3: CATCH

    def reset(self):
        self.current_phase = 0

    def compute(self, env_data):
        """
        env_data dict containing:
        - projectile_pos: np.ndarray (3,)
        - projectile_vel: np.ndarray (3,)
        - left_hand_pos: np.ndarray (3,)
        - right_catch_pos: np.ndarray (3,)
        - left_gripper_width: float
        - catcher_contact: bool
        - torques: np.ndarray
        - action_rate: float
        """
        proj_pos = env_data["projectile_pos"]
        proj_vel = env_data["projectile_vel"]
        left_pos = env_data["left_hand_pos"]
        right_pos = env_data["right_catch_pos"]
        gripper_w = env_data["left_gripper_width"]
        has_contact = env_data["catcher_contact"]
        torques = env_data["torques"]

        reward = 0.0
        info = {
            "phase": self.current_phase,
            "dist_left_to_proj": np.linalg.norm(left_pos - proj_pos),
            "dist_proj_to_catch": np.linalg.norm(proj_pos - right_pos),
            "proj_speed": np.linalg.norm(proj_vel),
            "is_success": False
        }

        dist_left = info["dist_left_to_proj"]
        dist_catch = info["dist_proj_to_catch"]

        # Phase Transition Checks
        if self.current_phase == 0:
            # Grasp Phase: reward reaching and securing object
            reward += 2.0 * np.exp(-15.0 * dist_left)
            if dist_left < 0.04 and gripper_w < 0.025:
                reward += 5.0
                self.current_phase = 1  # Transition to Toss

        elif self.current_phase == 1:
            # Toss Phase: reward moving forward/upward toward the apex
            dist_to_apex = np.linalg.norm(proj_pos - self.apex_target_pos)
            vel_x_reward = np.clip(proj_vel[0], 0.0, 2.5) * 1.5
            vel_z_reward = np.clip(proj_vel[2], 0.0, 2.0) * 1.2
            reward += vel_x_reward + vel_z_reward + 2.0 * np.exp(-5.0 * dist_to_apex)

            # Detect release when gripper opens or projectile clears left hand
            if dist_left > 0.08 and proj_vel[0] > 0.5:
                reward += 10.0
                self.current_phase = 2  # Transition to Flight

        elif self.current_phase == 2:
            # Flight Phase: projectile airborne toward right catcher
            # Analytical ballistic landing predictor guides right arm intercept
            g = 9.81
            z_catch = 0.88
            vz = proj_vel[2]
            disc = vz**2 + 2.0 * g * max(0.0, proj_pos[2] - z_catch)
            t_impact = (vz + np.sqrt(disc)) / g if disc >= 0 else 0.1
            t_impact = np.clip(t_impact, 0.0, 1.2)
            predicted_landing = np.array([
                proj_pos[0] + proj_vel[0] * t_impact,
                proj_pos[1] + proj_vel[1] * t_impact,
                z_catch
            ])
            dist_to_pred = np.linalg.norm(right_pos - predicted_landing)
            reward += 4.0 * np.exp(-10.0 * dist_to_pred) + 2.5 * np.exp(-6.0 * dist_catch)
            
            # Bonus for high arc crossing the apex region
            if proj_pos[0] > -0.1 and proj_pos[0] < 0.2 and proj_pos[2] > 1.05:
                reward += 2.0

            # Transition to Catch when projectile approaches catcher pad
            if dist_catch < 0.12 or has_contact:
                reward += 10.0
                self.current_phase = 3  # Transition to Catch

        elif self.current_phase == 3:
            # Catch Phase: Contact absorption, damping, retention
            reward += 5.0 * np.exp(-15.0 * dist_catch)
            if has_contact:
                reward += 15.0
                # Kinetic damping bonus: penalize projectile speed inside catcher
                proj_speed = np.linalg.norm(proj_vel)
                reward += 5.0 * np.exp(-3.0 * proj_speed)

                if dist_catch < 0.05 and proj_speed < 0.25:
                    reward += 50.0  # Massive success bonus
                    info["is_success"] = True

        # Energy & Jerk Regularization (prevent violent motor shudder)
        torque_penalty = 0.0005 * np.sum(np.square(torques))
        reward -= torque_penalty

        info["phase"] = self.current_phase
        return reward, info
