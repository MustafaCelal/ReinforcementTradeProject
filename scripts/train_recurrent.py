"""
RecurrentPPO (LSTM) ile Eğitim
==============================
Hafıza tabanlı politika ağı kullanarak modeli eğitir.
LSTM sayesinde zaman serisi bağlamını daha iyi öğrenir.

Kullanım: python scripts/train_recurrent.py --symbol EURUSD=X --steps 300000
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.core.environment import ForexTradingEnv
from src.utils.callbacks import BestModelEvalCallback, TensorboardCallback
import config.settings as cfg


def evaluate_model(model, eval_env, deterministic=True):
    """RecurrentPPO modeli için değerlendirme (LSTM state yönetimi dahil)"""
    obs = eval_env.reset()
    equity_curve = []
    
    # LSTM states için - RecurrentPPO None olarak başlar
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    while True:
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic
        )
        
        step_out = eval_env.step(action)
        
        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        
        episode_starts = dones if len(step_out) == 4 else np.array([done])
        
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)
        
        if done:
            break
    
    return equity_curve, float(equity_curve[-1])


def main():
    parser = argparse.ArgumentParser(description="RecurrentPPO (LSTM) Training")
    parser.add_argument("--symbol", type=str, default="EURUSD=X")
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--period", type=str, default="max")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--load-path", type=str, default=None)
    args = parser.parse_args()
    
    symbol = args.symbol
    total_timesteps = args.steps
    
    print(f"🧠 RecurrentPPO (LSTM) Eğitimi Başlatılıyor...")
    print(f"   Sembol: {symbol}, Timesteps: {total_timesteps:,}")
    
    # Veri yükle
    print(f"Veri indiriliyor: {symbol}...")
    raw_df = DataLoader.download_yfinance(symbol=symbol, interval=args.interval, period=args.period)
    df, feature_cols = DataProcessor.add_indicators(raw_df)
    
    # Train/Test split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Eğitim bar sayısı: {len(train_df)}")
    print(f"Test bar sayısı  : {len(test_df)}")
    
    # ---- Feature Scaling (MinMaxScaler) ----
    print("\n🔄 Feature Scaling (0-1 normalizasyonu)...")
    scaler = DataProcessor.create_scaler(train_df, feature_cols)
    train_df = DataProcessor.scale_features(train_df, feature_cols, scaler)
    test_df = DataProcessor.scale_features(test_df, feature_cols, scaler)
    
    # Save scaler for inference/testing
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_{symbol.replace('=', '_').replace('-', '_')}.pkl")
    DataProcessor.save_scaler(scaler, scaler_path)
    print(f"   ✅ Scaler kaydedildi: {scaler_path}")
    
    # Environment ayarları
    WIN = cfg.TradingConfig.WINDOW_SIZE
    SL_OPTS = cfg.TradingConfig.SL_OPTIONS
    TP_OPTS = cfg.TradingConfig.TP_OPTIONS
    
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
            min_episode_steps=500,  # LSTM için daha kısa episodlar
            episode_max_steps=1000,
            feature_columns=feature_cols,
            hold_reward_weight=cfg.TradingConfig.HOLD_REWARD_WEIGHT,
            open_penalty_pips=cfg.TradingConfig.OPEN_PENALTY_PIPS,
            time_penalty_pips=cfg.TradingConfig.TIME_PENALTY_PIPS,
            unrealized_delta_weight=cfg.TradingConfig.UNREALIZED_DELTA_WEIGHT,
            sharpe_reward_weight=cfg.TradingConfig.SHARPE_REWARD_WEIGHT,
            drawdown_penalty_weight=cfg.TradingConfig.DRAWDOWN_PENALTY_WEIGHT,
            initial_equity_usd=cfg.TradingConfig.INITIAL_EQUITY_USD,
            lot_size=cfg.TradingConfig.LOT_SIZE_MICRO,
        )
    
    def make_eval_env(test=False):
        return ForexTradingEnv(
            df=test_df if test else train_df,
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
            lot_size=cfg.TradingConfig.LOT_SIZE_MICRO,
        )
    
    train_vec_env = DummyVecEnv([make_train_env])
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    train_eval_env = DummyVecEnv([lambda: make_eval_env(test=False)])
    test_eval_env = DummyVecEnv([lambda: make_eval_env(test=True)])
    
    # Model oluştur veya yükle
    model_path = os.path.join(cfg.MODELS_DIR, f"recurrent_{symbol.replace('=', '_').replace('-', '_')}_best.zip")
    
    if args.load_path and os.path.exists(args.load_path):
        print(f"🔄 Model yükleniyor: {args.load_path}")
        model = RecurrentPPO.load(args.load_path, env=train_vec_env)
    else:
        print("📦 Yeni RecurrentPPO modeli oluşturuluyor...")
        
        policy_kwargs = dict(
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
            net_arch=dict(pi=[256], vf=[256]),
        )
        
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=train_vec_env,
            verbose=1,
            tensorboard_log=os.path.join(cfg.BASE_DIR, "tensorboard_log"),
            policy_kwargs=policy_kwargs,
            learning_rate=cfg.linear_learning_rate_schedule,  # Linear Learning Rate Scheduler
            n_steps=128,      # LSTM için daha düşük (memory)
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
        )
    
    # Checkpoint callback
    ckpt_dir = os.path.join(cfg.BASE_DIR, "checkpoints", "recurrent")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=ckpt_dir,
        name_prefix=f"recurrent_{symbol.replace('=', '_').replace('-', '_')}"
    )
    
    # ---- Best Model Evaluation (Otomatik) ----
    best_model_save_path = os.path.join(cfg.MODELS_DIR, f"recurrent_{symbol.replace('=', '_').replace('-', '_')}_best_evals.zip")
    best_model_eval_callback = BestModelEvalCallback(
        eval_env=test_eval_env,
        eval_freq=25_000,  # Her 25k timestep'te değerlendir (iç loop daha kısa olduğu için)
        best_model_save_path=best_model_save_path,
        verbose=1
    )
    
    tensorboard_callback = TensorboardCallback()
    callback = CallbackList([checkpoint_callback, best_model_eval_callback, tensorboard_callback])
    
    # Eğitim
    print("\n🚀 Eğitim başlıyor... (EvalCallback en kârlı modeli otomatik kaydedecek)")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # ---- Eğitim Tamamlandı ----
    print(f"\n✅ Eğitim tamamlandı!")
    print(f"   📊 Best Model (OOS) kaydedildi: {best_model_save_path}")
    
    # Load the best model from evaluation
    if os.path.exists(best_model_save_path):
        print(f"🔄 Best model yükleniyor: {best_model_save_path}")
        model = RecurrentPPO.load(best_model_save_path, env=train_vec_env)
    else:
        print("⚠️  Warning: Best model from evaluation not found.")
    
    # Model kaydet (final)
    model.save(model_path)
    
    # VecNormalize istatistiklerini kaydet
    stats_path = os.path.join(cfg.MODELS_DIR, "recurrent_vec_normalize.pkl")
    train_vec_env.save(stats_path)
    
    print(f"\n✅ Model kaydedildi: {model_path}")
    
    # Değerlendirme
    print("\n📊 Model Değerlendirmesi...")
    equity_curve_train, final_eq_train = evaluate_model(model, train_eval_env)
    equity_curve_test, final_eq_test = evaluate_model(model, test_eval_env)
    
    print(f"[Train] Final Equity: ${final_eq_train:.2f}")
    print(f"[Test]  Final Equity: ${final_eq_test:.2f}")
    print(f"[Test]  Net Profit  : ${final_eq_test - cfg.INITIAL_EQUITY_USD:.2f}")
    
    # Plot
    if not args.no_plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(equity_curve_train)
        plt.title(f"Train Equity - Final: ${final_eq_train:.2f}")
        plt.xlabel("Steps")
        plt.ylabel("Equity ($)")
        
        plt.subplot(1, 2, 2)
        plt.plot(equity_curve_test, color='orange')
        plt.title(f"Test Equity - Final: ${final_eq_test:.2f}")
        plt.xlabel("Steps")
        plt.ylabel("Equity ($)")
        
        plt.suptitle(f"RecurrentPPO (LSTM) - {symbol}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
