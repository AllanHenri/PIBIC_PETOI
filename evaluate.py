import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_schedule_fn
from opencat_gym_env import OpenCatGymEnv

def make_env():
    return OpenCatGymEnv(render_mode="human")

env = DummyVecEnv([make_env])

custom_objects = {
    "clip_range": get_schedule_fn(0.2),
    "lr_schedule": get_schedule_fn(3e-4),
}

policy_kwargs = dict(net_arch=[256, 256])

model = PPO.load("trained/opencat_gym_esp32_trained_controller",
                 env=env,
                 custom_objects=custom_objects,
                 policy_kwargs=policy_kwargs)

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
