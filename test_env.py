import gymnasium as gym
import envs

def main():
    print("=== Testing AlohaTossCatch-v0 Gymnasium Environment ===")
    env = gym.make("AlohaTossCatch-v0")
    obs, info = env.reset(seed=42)
    print("Observation shape:", obs.shape)
    print("Action space shape:", env.action_space.shape)
    print("Initial info:", info)

    for i in range(15):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        phase = info.get("phase", 0)
        dist_left = info.get("dist_left_to_proj", 0.0)
        dist_catch = info.get("dist_proj_to_catch", 0.0)
        print(f"Step {i+1:02d} | Phase: {phase} | Reward: {reward:+.3f} | DistLeft: {dist_left:.3f}m | DistCatch: {dist_catch:.3f}m")
        if terminated or truncated:
            print("Episode ended early:", info.get("termination_reason"))
            break

    env.close()
    print("SUCCESS: Environment validated!")

if __name__ == "__main__":
    main()

