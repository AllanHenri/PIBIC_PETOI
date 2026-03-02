from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_schedule_fn
from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class
parallel_env = 1
env = make_vec_env(lambda: OpenCatGymEnv(render_mode="human"), n_envs=parallel_env)

custom_objects = {
    "clip_range": get_schedule_fn(0.2),
    "lr_schedule": get_schedule_fn(3e-4),
}

model = PPO.load("trained/trained_agent_PPO", env=env, custom_objects=custom_objects)

obs = env.reset()
sum_reward = 0

for _ in range(500):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    sum_reward += reward
    env.render()
    if done[0]:
        print("Reward", sum_reward[0])
        sum_reward = 0
        obs = env.reset()
