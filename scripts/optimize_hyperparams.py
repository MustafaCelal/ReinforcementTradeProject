"""
Optuna Hiperparametre Optimizasyonu
===================================
PPO modeli için en iyi hiperparametreleri bulur.
Kullanım: python scripts/optimize_hyperparams.py --trials 20 --symbol EURUSD=X
"""

import sys
import os
import argparse
import json
import numpy as np

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.core.environment import ForexTradingEnv
import config.settings as cfg


def create_env(df, feature_cols, is_eval=False):
    """Environment factory fonksiyonu"""
    return ForexTradingEnv(
        df=df,
        window_size=cfg.WINDOW_SIZE,
        sl_options=cfg.SL_OPTIONS,
        tp_options=cfg.TP_OPTIONS,
        spread_pips=cfg.SPREAD_PIPS,
        commission_pips=cfg.COMMISSION_PIPS,
        max_slippage_pips=cfg.MAX_SLIPPAGE_PIPS,
        random_start=not is_eval,
        min_episode_steps=1000 if not is_eval else 300,
        episode_max_steps=2000 if not is_eval else None,
        feature_columns=feature_cols,
        initial_equity_usd=cfg.INITIAL_EQUITY_USD,
        lot_size=cfg.LOT_SIZE_MICRO,
    )


def evaluate_model(model, eval_env, n_episodes=3):
    """Model performansını değerlendir"""
    total_equity = 0.0
    
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_out = eval_env.step(action)
            
            if len(step_out) == 4:
                obs, _, dones, infos = step_out
                done = bool(dones[0])
            else:
                obs, _, terminated, truncated, infos = step_out
                done = bool(terminated[0] or truncated[0])
        
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        total_equity += info.get("equity_usd", cfg.INITIAL_EQUITY_USD)
    
    return total_equity / n_episodes


def objective(trial: optuna.Trial, train_df, test_df, feature_cols, n_timesteps=100000):
    """Optuna objective fonksiyonu - her trial için çağrılır"""
    
    # Hiperparametreleri sample et
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    n_epochs = trial.suggest_int("n_epochs", 3, 15)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    
    # Network architecture
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch_map = {
        "small": [dict(pi=[128, 128], vf=[128, 128])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "large": [dict(pi=[512, 256], vf=[512, 256])],
    }
    net_arch = net_arch_map[net_arch_choice]
    
    # Environment oluştur
    train_env = DummyVecEnv([lambda: create_env(train_df, feature_cols, is_eval=False)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    eval_env = DummyVecEnv([lambda: create_env(test_df, feature_cols, is_eval=True)])
    
    # Model oluştur
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=0,
    )
    
    # Eğitim
    try:
        model.learn(total_timesteps=n_timesteps)
    except Exception as e:
        print(f"Training failed: {e}")
        return cfg.INITIAL_EQUITY_USD  # Return initial equity as failure
    
    # Değerlendirme
    final_equity = evaluate_model(model, eval_env, n_episodes=3)
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return final_equity


def main():
    parser = argparse.ArgumentParser(description="Optuna PPO Hyperparameter Optimization")
    parser.add_argument("--symbol", type=str, default="EURUSD=X")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--period", type=str, default="max")
    parser.add_argument("--trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps per trial")
    parser.add_argument("--study-name", type=str, default="ppo_optimization")
    args = parser.parse_args()
    
    print(f"📊 Optuna Hiperparametre Optimizasyonu Başlatılıyor...")
    print(f"   Sembol: {args.symbol}, Trial sayısı: {args.trials}")
    
    # Veri yükle
    print(f"Veri indiriliyor: {args.symbol}...")
    raw_df = DataLoader.download_yfinance(symbol=args.symbol, interval=args.interval, period=args.period)
    df, feature_cols = DataProcessor.add_indicators(raw_df)
    
    # Train/Test split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train bars: {len(train_df)}, Test bars: {len(test_df)}")
    
    # Optuna study oluştur
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",  # Final equity'yi maksimize et
        sampler=sampler,
        pruner=pruner,
    )
    
    # Optimizasyon
    study.optimize(
        lambda trial: objective(trial, train_df, test_df, feature_cols, args.timesteps),
        n_trials=args.trials,
        show_progress_bar=True,
    )
    
    # Sonuçlar
    print("\n" + "="*60)
    print("🏆 EN İYİ PARAMETRELER")
    print("="*60)
    
    best_params = study.best_params
    best_value = study.best_value
    
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n  Best Final Equity: ${best_value:.2f}")
    print(f"  Net Profit: ${best_value - cfg.INITIAL_EQUITY_USD:.2f}")
    
    # En iyi parametreleri kaydet
    output_path = os.path.join(cfg.OUTPUTS_DIR, "best_hyperparams.json")
    with open(output_path, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_equity": best_value,
            "symbol": args.symbol,
            "n_trials": args.trials,
        }, f, indent=2)
    
    print(f"\n✅ En iyi parametreler kaydedildi: {output_path}")
    
    # Top 5 trials
    print("\n📈 TOP 5 TRIALS:")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_trials = trials_df.nlargest(5, "value")
        print(top_trials[["number", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
