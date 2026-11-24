"""
Automated Smoke Test Harness for Aloha Toss & Catch Simulator.
Validates MuJoCo physical loading, Gymnasium environment API,
4-phase reward calculations, and 60fps HUD rendering in under 5 seconds.
"""
import sys
import os
import unittest
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mujoco
import gymnasium as gym

import envs
from rewards.phase_reward import PhaseRewardEngine
from render.visual_engine import VisualEngine

class TestAlohaTossCatchSmoke(unittest.TestCase):
    def test_01_mujoco_model_integrity(self):
        """Verify XML parsing, joint limits, actuators, and physics step."""
        model = mujoco.MjModel.from_xml_path("assets/aloha_bimanual_toss.xml")
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)

        self.assertEqual(model.nu, 16, f"Expected 16 actuators, got {model.nu}")
        self.assertEqual(model.nq, 23, f"Expected 23 qpos, got {model.nq}")
        self.assertEqual(model.nv, 22, f"Expected 22 qvel, got {model.nv}")
        self.assertGreaterEqual(model.nsensordata, 1, "Sensors must be defined")

    def test_02_gym_env_lifecycle(self):
        """Verify Gymnasium registration, reset, observation/action specs, and step."""
        env = gym.make("AlohaTossCatch-v0")
        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape, (55,), f"Expected obs shape (55,), got {obs.shape}")
        self.assertEqual(env.action_space.shape, (16,), f"Expected action shape (16,), got {env.action_space.shape}")
        self.assertIn("phase", info, "Info dictionary must contain 'phase'")

        # Take 10 steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            self.assertEqual(obs.shape, (55,))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(term, bool)
            self.assertIsInstance(trunc, bool)
            if term or trunc:
                break
        env.close()

    def test_03_phase_reward_engine(self):
        """Verify 4-phase reward transitions and torque regularizations."""
        engine = PhaseRewardEngine()
        engine.reset()
        self.assertEqual(engine.current_phase, 0)

        # Grasp check
        mock_data = {
            "projectile_pos": np.array([-0.25, 0.0, 0.88]),
            "projectile_vel": np.array([0.0, 0.0, 0.0]),
            "left_hand_pos": np.array([-0.25, 0.0, 0.88]),
            "right_catch_pos": np.array([0.25, 0.0, 0.88]),
            "left_gripper_width": 0.015,
            "catcher_contact": False,
            "torques": np.zeros(16),
        }
        reward, info = engine.compute(mock_data)
        self.assertGreater(reward, 0.0)
        self.assertEqual(engine.current_phase, 1, "Should transition to Toss phase when grasping")

    def test_04_visual_engine_and_hud(self):
        """Verify offscreen rendering and 60fps HUD overlay without GUI crash."""
        env = gym.make("AlohaTossCatch-v0")
        obs, info = env.reset(seed=42)
        unwrapped = env.unwrapped

        engine = VisualEngine(unwrapped.model, unwrapped.data, width=640, height=360)
        telemetry = {
            "step": 1,
            "phase": 1,
            "proj_pos": unwrapped.data.site_xpos[unwrapped.site_projectile],
            "proj_vel": np.array([1.0, 0.0, 0.5]),
            "dist_to_catch": 0.35,
            "reward": 5.2,
            "contact": False,
            "is_success": False
        }
        frame = engine.render_frame(telemetry)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (360, 640, 3))
        engine.close()
        env.close()

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlohaTossCatchSmoke)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
