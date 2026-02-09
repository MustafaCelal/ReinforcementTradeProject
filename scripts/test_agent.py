import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Proje kök dizinini sys.path'e ekle (src ve config modüllerini bulabilmek için)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.core.environment import ForexTradingEnv
from src.utils.reporting import calculate_performance_metrics, print_fancy_summary
from src.ui.visualizer import plot_trading_results
import config.settings as cfg
import os


# --- DEFAULT SETTINGS (Overridden by argparse) ---
DEFAULT_SYMBOL = "EURUSD=X"
DEFAULT_PERIOD = "1y"
SOURCE = 'live'
DETERMINISTIC = True

def run_one_episode(model, vec_env, deterministic=True):
    obs = vec_env.reset()
    equity_curve = [vec_env.get_attr("equity_usd")[0]] # Start with initial equity
    closed_trades = []
    
    print("\n🤖 Bot stratejiyi uygulamaya başladı...")
    print("-----------------------------------------")

    step_count = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        # Verbose Logging (Support multiple events per step)
        events = infos[0].get("events", [])
        if events:
            for event_info in events:
                event = event_info.get("event")
                step = event_info.get("step")
                
                if event == "OPEN":
                    side = "ALIŞ (Long)" if event_info['position'] == 1 else "SATIŞ (Short)"
                    print(f"🔹 [Adım {step}] Pozisyon Açıldı: {side}")
                    print(f"   | Giriş: {event_info['entry_price']:.5f} | SL: {event_info['sl_price']:.5f} | TP: {event_info['tp_price']:.5f}")
                
                elif event == "CLOSE":
                    reason = event_info.get("reason", "Manuel")
                    pips = event_info.get("net_pips", 0)
                    icon = "✅" if pips > 0 else "❌"
                    print(f"{icon} [Adım {step}] Pozisyon Kapatıldı ({reason})")
                    print(f"   | Çıkış: {event_info['exit_price']:.5f} | Net Pip: {pips:.1f} | Bakiye: ${event_info['equity_usd']:,.2f}")
                    closed_trades.append(event_info)

        if done:
            # Environment has auto-reset. Grab final equity from info.
            final_equity = infos[0].get("equity_usd", equity_curve[-1])
            equity_curve.append(final_equity)
            break
        
        # If not done, regular step equity
        equity_curve.append(vec_env.get_attr("equity_usd")[0])
        
        step_count += 1
        if step_count % 100 == 0:
            print(f"⏳ Analiz devam ediyor... ({step_count} mum tarandı)")

    print("-----------------------------------------")
    print("🏁 Analiz tamamlandı.")
    return equity_curve, closed_trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--period", type=str, default=DEFAULT_PERIOD)
    parser.add_argument("--model-path", type=str, default=None, help="Path to specific model file (.zip)")
    parser.add_argument("--no-plot", action="store_true", help="Disable interactive plot display")
    args = parser.parse_args()

    symbol = args.symbol
    period = args.period
    
    model_name = f"model_{symbol.replace('=', '_').replace('-', '_')}_best"
    
    if SOURCE == 'csv':
        file_path = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
        df, feature_cols = load_and_preprocess_data(file_path)
        # Split for OOS
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
    else:
        # Live / Yahoo Finance
        print(f"Test için {symbol} verileri indiriliyor ({period})...")
        raw_df = DataLoader.download_yfinance(symbol=symbol, interval="1h", period=period)
        test_df, feature_cols = DataProcessor.add_indicators(raw_df)

    # Load scaler if available (to match training preprocessing)
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_{symbol.replace('=', '_').replace('-', '_')}.pkl")
    if os.path.exists(scaler_path):
        print(f"🔄 Feature Scaler yükleniyor: {scaler_path}")
        scaler = DataProcessor.load_scaler(scaler_path)
        test_df = DataProcessor.scale_features(test_df, feature_cols, scaler)
        print(f"   ✅ Test verileri scaled")
    else:
        print(f"⚠️  Scaler bulunamadı: {scaler_path}")

    # Must match training params
    SL_OPTS = cfg.TradingConfig.SL_OPTIONS
    TP_OPTS = cfg.TradingConfig.TP_OPTIONS
    WIN = cfg.TradingConfig.WINDOW_SIZE

    test_env = ForexTradingEnv(
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
            initial_equity_usd=cfg.TradingConfig.INITIAL_EQUITY_USD,
            lot_size=cfg.TradingConfig.LOT_SIZE_MICRO
    )

    vec_test_env = DummyVecEnv([lambda: test_env])
    
    # Wrap with VecNormalize for consistency with training
    stats_path = os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl")
    if os.path.exists(stats_path):
        print(f"🔄 Normalizasyon verileri yükleniyor: {stats_path}")
        vec_test_env = VecNormalize.load(stats_path, vec_test_env)
        vec_test_env.training = False # Do not update stats during test
        vec_test_env.norm_reward = False
    else:
        print("⚠️ Uyarı: Normalizasyon dosyası bulunamadı! Tahminler hatalı olabilir.")

    # Load model
    load_path = args.model_path if args.model_path else os.path.join(cfg.MODELS_DIR, f"{model_name}.zip")
    
    try:
        print(f"🔄 Test için model yükleniyor: {load_path}")
        model = PPO.load(load_path, env=vec_test_env)
        # Ensure model uses the normalized env
        model.set_env(vec_test_env)
    except Exception as e:
        print(f"❌ Model yüklenemedi ({load_path}): {e}")
        return

    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # Save trades and Print Summary
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        
        # COMPLETELY REMOVE End of Data trades from history (User request)
        trades_df = trades_df[trades_df["reason"] != "END_OF_DATA"]
        
        if not trades_df.empty:
            trades_df.to_csv(cfg.TRADE_HISTORY_FILE, index=False)
            
            initial_eq = equity_curve[0]
            final_eq = equity_curve[-1]
            summary = calculate_performance_metrics(trades_df, initial_eq, final_eq)
            print_fancy_summary(summary)
            
            # Plot and save
            plot_trading_results(test_df, trades_df, title=f"{symbol} Trading Results (Test)", save_path=cfg.TRADING_CHART_FILE)
        else:
            print("\n⚠️  Stratejiye uygun (SL/TP/Manüel) bir işlem gerçekleşmedi.")
    else:
        print("\n⚠️  Seçilen veri aralığında hiç işlem gerçekleşmedi.")

    # Plot equity and save
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity (Test)")
    plt.title(f"Equity Curve - {symbol} Evaluation")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    
    # Save a separate equity chart if needed, or just rely on visualizer's candle chart
    # Here we just handle the show() behavior
    if not args.no_plot:
        plt.show()
    else:
        plt.close() # Close figures to free memory in headless mode


if __name__ == "__main__":
    main()
