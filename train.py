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
                n_steps=int(1024 * 8 / parallel_env),
                device='cuda',
                verbose=1).learn(2e6)

    model.save("trained/opencat_gym_esp32_trained_controller")

    # Depois do model.learn(...) e antes de fechar o env
    # 1) Extrai recompensas dos últimos episódios registrados no buffer
    rewards = [info["r"] for info in model.ep_info_buffer if "r" in info]
    episodes = list(range(1, len(rewards) + 1))

    # 2) Calcula média móvel (janela de N episódios)
    import numpy as np
    window = 20  # ajustável: quantos episódios pra média
    ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ma_episodes = list(range(window, len(rewards) + 1))

    # 3) Plota tudo junto
    import matplotlib.pyplot as plt

    plt.plot(episodes, rewards,       label="Recompensa por episódio")
    plt.plot(ma_episodes, ma,         label=f"Média móvel ({window})")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa")
    plt.title("Curva de Aprendizagem Rápida")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #preciso criar grafico com os resultados do treinamento
    env.close()
