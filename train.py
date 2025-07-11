from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from opencat_gym_env import OpenCatGymEnv

if __name__ == "__main__":
    parallel_env = 8

    env = make_vec_env(OpenCatGymEnv,
                       n_envs=parallel_env,
                       vec_env_cls=SubprocVecEnv)

    custom_arch = dict(net_arch=[256, 256])

    model = PPO('MlpPolicy', env, seed=42,
                policy_kwargs=custom_arch,
                n_steps=int(2048 * 8 / parallel_env),
                device='cuda',
                verbose=1).learn(2e6)

    model.save("trained/opencat_gym_esp32_trained_controller")



