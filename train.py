import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from opencat_gym_env import OpenCatGymEnv

# --- CONFIGURAÇÕES ---
LOG_DIR = "logs/"
TB_LOG_DIR = "logs/tensorboard/"
MODEL_DIR = "trained/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Wrapper para criar cada env com Monitor

def make_monitored_env(rank, log_dir):
    def _init():
        env = OpenCatGymEnv()
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
    return _init

if __name__ == "__main__":
    # 1) Criação do vetor de ambientes paralelos com Monitor
    parallel_env = 8
    env_fns = [make_monitored_env(i, LOG_DIR) for i in range(parallel_env)]
    env = SubprocVecEnv(env_fns)

    # 2) Definição da arquitetura customizada e do agente
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        'MlpPolicy',
        env,
        seed=42,
        policy_kwargs=policy_kwargs,
        n_steps=int(2048 * parallel_env / parallel_env),
        tensorboard_log=TB_LOG_DIR,
        verbose=1
    )

    # 3) Treinamento (2e6 timesteps)
    model.learn(2_000_000)
    model.save(os.path.join(MODEL_DIR, "opencat_gym_esp32_trained_controller"))

    # 4) Plot da curva de aprendizagem
    # Carrega todos os CSV gerados pelo Monitor
    results = load_results(LOG_DIR)
    x, y = ts2xy(results, 'timesteps')

    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Curva de Aprendizagem - OpenCatGymEnv")
    plt.tight_layout()
    # Salva figura e exibe
    plt.savefig(os.path.join(MODEL_DIR, "learning_curve.png"))
    plt.show()
