import gymnasium as gym
import highway_env

gym.register_envs(highway_env)
env = gym.make("parking-v0", render_mode="human")
obs, info = env.reset()
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()