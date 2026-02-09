import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Proje kök dizinini sys.path'e ekle (src ve config modüllerini bulabilmek için)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.core.environment import ForexTradingEnv
from src.utils.callbacks import TensorboardCallback, BestModelEvalCallback
import config.settings as cfg


def evaluate_model(model: PPO, eval_env: DummyVecEnv, deterministic: bool = True):
    obs = eval_env.reset()
    equity_curve = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        # use equity from info (state *before* DummyVecEnv reset)
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity



# --- DEFAULT SETTINGS (Overridden by argparse) ---
DEFAULT_SYMBOL = "EURUSD=X"
DEFAULT_TOTAL_TIMESTEPS = 600000
DEFAULT_LOAD_EXISTING = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--steps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--load-existing", type=bool, default=DEFAULT_LOAD_EXISTING)
    parser.add_argument("--load-path", type=str, default=None, help="Path to a model to load as starting point")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (e.g. 1h, 15m, 1d)")
    parser.add_argument("--period", type=str, default="max", help="Data period (e.g. max, 2y, 1y, 6mo)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    symbol = args.symbol
    total_timesteps = args.steps
    load_existing = args.load_existing

    # Live Data Download for Training
    print(f"Eğitim için {symbol} verileri indiriliyor (Aralık: {args.interval}, Periyot: {args.period})...")
    raw_df = DataLoader.download_yfinance(symbol=symbol, interval=args.interval, period=args.period) 
    df, feature_cols = DataProcessor.add_indicators(raw_df)

    # Time split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Eğitim bar sayısı: {len(train_df)}")
    print(f"Test bar sayısı   : {len(test_df)}")
    
    # ---- Feature Scaling (MinMaxScaler) ----
    print("\n🔄 Feature Scaling (0-1 normalizasyonu)...")
    scaler = DataProcessor.create_scaler(train_df, feature_cols)
    train_df = DataProcessor.scale_features(train_df, feature_cols, scaler)
    test_df = DataProcessor.scale_features(test_df, feature_cols, scaler)
    
    # Save scaler for inference/testing
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_{symbol.replace('=', '_').replace('-', '_')}.pkl")
    DataProcessor.save_scaler(scaler, scaler_path)
    print(f"   ✅ Scaler kaydedildi: {scaler_path}")

    # ---- Env factories ----
    # ATR Multipliers for dynamic SL/TP
    SL_OPTS = cfg.TradingConfig.SL_OPTIONS
    TP_OPTS = cfg.TradingConfig.TP_OPTIONS
    WIN = cfg.TradingConfig.WINDOW_SIZE

    # Train env: random starts to reduce memorization
    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=cfg.TradingConfig.SPREAD_PIPS,
            commission_pips=cfg.TradingConfig.COMMISSION_PIPS,
            max_slippage_pips=cfg.TradingConfig.MAX_SLIPPAGE_PIPS,
            random_start=True,
            min_episode_steps=1000,
            episode_max_steps=2000,
            feature_columns=feature_cols,
            hold_reward_weight=cfg.TradingConfig.HOLD_REWARD_WEIGHT,
            open_penalty_pips=cfg.TradingConfig.OPEN_PENALTY_PIPS,
            time_penalty_pips=cfg.TradingConfig.TIME_PENALTY_PIPS,
            unrealized_delta_weight=cfg.TradingConfig.UNREALIZED_DELTA_WEIGHT,
            sharpe_reward_weight=cfg.TradingConfig.SHARPE_REWARD_WEIGHT,
            drawdown_penalty_weight=cfg.TradingConfig.DRAWDOWN_PENALTY_WEIGHT,
            initial_equity_usd=cfg.TradingConfig.INITIAL_EQUITY_USD,
            lot_size=cfg.TradingConfig.LOT_SIZE_MICRO
        )

    # Train-eval env: deterministic start, NO random starts (so curve is stable/reproducible)
    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=cfg.TradingConfig.SPREAD_PIPS,
            commission_pips=cfg.TradingConfig.COMMISSION_PIPS,
            max_slippage_pips=cfg.TradingConfig.MAX_SLIPPAGE_PIPS,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=cfg.TradingConfig.HOLD_REWARD_WEIGHT,
            open_penalty_pips=cfg.TradingConfig.OPEN_PENALTY_PIPS,
            time_penalty_pips=cfg.TradingConfig.TIME_PENALTY_PIPS,
            unrealized_delta_weight=cfg.TradingConfig.UNREALIZED_DELTA_WEIGHT,
            sharpe_reward_weight=cfg.TradingConfig.SHARPE_REWARD_WEIGHT,
            drawdown_penalty_weight=cfg.TradingConfig.DRAWDOWN_PENALTY_WEIGHT,
            initial_equity_usd=cfg.TradingConfig.INITIAL_EQUITY_USD,
            lot_size=cfg.TradingConfig.LOT_SIZE_MICRO
        )

    # Test-eval env: deterministic
    def make_test_eval_env():
        return ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=cfg.TradingConfig.SPREAD_PIPS,
            commission_pips=cfg.TradingConfig.COMMISSION_PIPS,
            max_slippage_pips=cfg.TradingConfig.MAX_SLIPPAGE_PIPS,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=cfg.TradingConfig.HOLD_REWARD_WEIGHT,
            open_penalty_pips=cfg.TradingConfig.OPEN_PENALTY_PIPS,
            time_penalty_pips=cfg.TradingConfig.TIME_PENALTY_PIPS,
            unrealized_delta_weight=cfg.TradingConfig.UNREALIZED_DELTA_WEIGHT,
            sharpe_reward_weight=cfg.TradingConfig.SHARPE_REWARD_WEIGHT,
            drawdown_penalty_weight=cfg.TradingConfig.DRAWDOWN_PENALTY_WEIGHT,
            initial_equity_usd=cfg.TradingConfig.INITIAL_EQUITY_USD,
            lot_size=cfg.TradingConfig.LOT_SIZE_MICRO
        )

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env]) # Original test_eval_env

    # ---- Model ----
    # Determine which model to load
    target_best_model_path = os.path.join(cfg.MODELS_DIR, f"model_{symbol.replace('=', '_').replace('-', '_')}_best.zip")
    source_model_path = args.load_path if args.load_path else target_best_model_path
    
    # CRITICAL: Path for VecNormalize stats
    stats_path = os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl")

    if load_existing and os.path.exists(source_model_path):
        print(f"🔄 Model yükleniyor: {source_model_path}")
        
        # If loading an existing model, we need to load its VecNormalize stats if it was trained with it
        # For now, we'll assume if a model is loaded, its env is already set up correctly.
        model = PPO.load(source_model_path, env=train_vec_env)
        # If the loaded model was trained with VecNormalize, we need to load the stats
        # This is a placeholder, actual loading logic might be more complex
        if os.path.exists(os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl")):
            train_vec_env = VecNormalize.load(os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl"), train_vec_env)
            # Update the model's environment to the normalized one
            model.set_env(train_vec_env)
        else:
            print("⚠️ Warning: No VecNormalize stats found for loaded model. Training might be inconsistent.")
    else:
        if args.load_path:
            print(f"⚠️ Uyarı: Belirtilen model bulunamadı: {args.load_path}")
        
        train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )

        model = PPO(
            policy="MlpPolicy",
            env=train_vec_env,
            verbose=1,
            tensorboard_log=os.path.join(cfg.BASE_DIR, "tensorboard_log"),
            policy_kwargs=policy_kwargs,
            learning_rate=cfg.linear_learning_rate_schedule,  # Linear Learning Rate Scheduler
            batch_size=cfg.TradingConfig.BATCH_SIZE,
            ent_coef=cfg.TradingConfig.ENT_COEF,         # Increased exploration to help model find better cycles
            n_epochs=cfg.TradingConfig.N_EPOCHS,
            n_steps=cfg.TradingConfig.N_STEPS
        )

    # ---- Checkpoints ----
    ckpt_dir = os.path.join(cfg.BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix=f"ppo_{symbol.replace('=', '_').replace('-', '_')}"
    )

    # ---- Best Model Evaluation (Otomatik) ----
    best_model_save_path = os.path.join(cfg.MODELS_DIR, f"model_{symbol.replace('=', '_').replace('-', '_')}_best_evals.zip")
    best_model_eval_callback = BestModelEvalCallback(
        eval_env=test_eval_env,
        eval_freq=50_000,  # Her 50k timestep'te değerlendir
        best_model_save_path=best_model_save_path,
        verbose=1
    )

    tensorboard_callback = TensorboardCallback()
    callback = CallbackList([checkpoint_callback, best_model_eval_callback, tensorboard_callback])

    # ---- Train ----
    print("\n🚀 Eğitim başlıyor... (EvalCallback en kârlı modeli otomatik kaydedecek)")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # ---- Eğitim Tamamlandı ----
    print(f"\n✅ Eğitim tamamlandı!")
    print(f"   📊 Best Model (OOS) kaydedildi: {best_model_save_path}")
    
    # Load the best model from evaluation
    if os.path.exists(best_model_save_path):
        print(f"🔄 Best model yükleniyor: {best_model_save_path}")
        best_model = PPO.load(best_model_save_path, env=train_vec_env)
    else:
        print("⚠️  Warning: Best model from evaluation not found. Using last trained model.")
        best_model = model
    
    # CRITICAL: Save VecNormalize stats for test/inference phase
    stats_path = os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl")
    train_vec_env.save(stats_path)
    
    # Save final best model as default path (for backward compatibility)
    best_model.save(target_best_model_path)

    # ---- Plot BOTH: in-sample vs out-of-sample ----
    equity_curve_train, final_equity_train = evaluate_model(best_model, train_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(best_model, test_eval_env)

    print(f"[IS Eval]  Final equity (train): {final_equity_train:.2f}")
    print(f"[OOS Eval] Final equity (test) : {final_equity_test:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train, label="Train (in-sample) equity")
    plt.plot(equity_curve_test, label="Test (out-of-sample) equity")
    plt.title("Equity Curves: In-sample vs Out-of-sample (Best Model)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
