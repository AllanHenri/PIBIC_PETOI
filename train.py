import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
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
    train_env = SubprocVecEnv(env_fns)

    # 2) Ambiente de avaliação (pode ser um único env)
    eval_env = DummyVecEnv([lambda: OpenCatGymEnv()])

    # 3) Definição da arquitetura customizada e do agente
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        'MlpPolicy',
        train_env,
        seed=42,
        policy_kwargs=policy_kwargs,
        n_steps=int(2048 * parallel_env / parallel_env),
        tensorboard_log=TB_LOG_DIR,
        verbose=1
    )

    # 4) Callbacks:
    #    a) EvalCallback para salvar o *melhor* modelo com base na recompensa média
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,    # onde o best model será salvo como best_model.zip
        log_path=LOG_DIR,                  # onde salvar logs de avaliação
        eval_freq=50_000,                  # a cada quantos timesteps avaliar
        deterministic=True,
        render=False
    )
    #    b) (Opcional) CheckpointCallback para salvar periódicamente
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path=MODEL_DIR,
        name_prefix='checkpoint'
    )
    callback = CallbackList([eval_callback, checkpoint_callback])

    # 5) Treinamento (2e6 timesteps) com callbacks
    model.learn(total_timesteps=2_000_000, callback=callback)

    # O melhor modelo já está salvo em trained/best_model.zip
    # Você pode carregá‑lo depois com:
    # best_model = PPO.load(os.path.join(MODEL_DIR, "best_model.zip"))

    # 6) Plot da curva de aprendizagem (como antes)
    results = load_results(LOG_DIR)
    x, y = ts2xy(results, 'timesteps')
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Curva de Aprendizagem - OpenCatGymEnv")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "learning_curve.png"))
    plt.show()
