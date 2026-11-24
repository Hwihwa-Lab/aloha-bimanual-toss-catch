from gymnasium.envs.registration import register
from envs.toss_catch_env import AlohaTossCatchEnv

register(
    id="AlohaTossCatch-v0",
    entry_point="envs.toss_catch_env:AlohaTossCatchEnv",
    max_episode_steps=250,
)

__all__ = ["AlohaTossCatchEnv"]

