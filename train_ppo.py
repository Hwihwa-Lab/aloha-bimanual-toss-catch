"""
PPO Training Pipeline for Aloha Bimanual Dynamic Toss & Catch.
Uses Stable-Baselines3 with Phase-Aware Reward Shaping and Episode Callbacks.
"""
import os
import argparse
import gymnasium as gym
import envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

class TossCatchMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.success_count = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        # Check infos from step
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_rew = info["episode"]["r"]
                self.episode_rewards.append(ep_rew)
                self.total_episodes += 1
                if info.get("is_success", False):
                    self.success_count += 1

                if self.total_episodes % 10 == 0:
                    sr = (self.success_count / self.total_episodes) * 100.0
                    mean_r = sum(self.episode_rewards[-10:]) / 10.0
                    print(f"[Training Eval] Episodes: {self.total_episodes} | Mean Reward (last 10): {mean_r:+.2f} | Success Rate: {sr:.1f}%")
        return True

def make_env():
    def _init():
        env = gym.make("AlohaTossCatch-v0")
        env = Monitor(env)
        return env
    return _init

def train(timesteps=50000, lr=3e-4, batch_size=64, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"=== Starting PPO Training on AlohaTossCatch-v0 ===")
    print(f"Total Timesteps: {timesteps} | Learning Rate: {lr} | Batch Size: {batch_size}")

    vec_env = DummyVecEnv([make_env()])

    # Hyperparameters tuned for continuous contact & ballistic manipulation
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        n_steps=1024,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
    )

    metrics_cb = TossCatchMetricsCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=max(5000, timesteps // 5),
        save_path=save_dir,
        name_prefix="aloha_toss_catch_ppo"
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[metrics_cb, checkpoint_cb],
        progress_bar=False
    )

    final_model_path = os.path.join(save_dir, "aloha_toss_catch_final.zip")
    model.save(final_model_path)
    print(f"=== PPO Training Complete! Model saved to {final_model_path} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        lr=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

