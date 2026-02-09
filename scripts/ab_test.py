"""
A/B Testing Script
==================
Compare two pre-trained models on the same test dataset.
Generates a comparative report and equity curve plot.

Usage:
python scripts/ab_test.py --model-a models/model_a.zip --model-b models/model_b.zip --symbol EURUSD=X
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.core.environment import ForexTradingEnv
import config.settings as cfg

def create_eval_env(df, feature_cols):
    """Create evaluation environment"""
    return ForexTradingEnv(
        df=df,
        window_size=cfg.WINDOW_SIZE,
        sl_options=cfg.SL_OPTIONS,
        tp_options=cfg.TP_OPTIONS,
        spread_pips=cfg.SPREAD_PIPS,
        commission_pips=cfg.COMMISSION_PIPS,
        max_slippage_pips=cfg.MAX_SLIPPAGE_PIPS,
        random_start=False,
        episode_max_steps=None,
        feature_columns=feature_cols,
        initial_equity_usd=cfg.INITIAL_EQUITY_USD,
        lot_size=cfg.LOT_SIZE_MICRO,
    )

def evaluate_model(model, env, name="Model"):
    """Evaluate a single model"""
    print(f"🔄 Evaluating {name}...")
    obs = env.reset()
    
    equity_curve = []
    trades = []
    
    # LSTM support
    is_recurrent = isinstance(model, RecurrentPPO)
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    while True:
        if is_recurrent:
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        step_out = env.step(action)
        
        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
            
        episode_starts = dones if len(step_out) == 4 else np.array([done])
            
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        equity_curve.append(info.get("equity_usd", cfg.INITIAL_EQUITY_USD))
        
        # Collect closed trades
        events = info.get("events", [])
        for e in events:
            if e.get("event") == "CLOSE":
                e["model"] = name
                trades.append(e)
        
        if done:
            break
            
    final_equity = equity_curve[-1]
    net_profit = final_equity - cfg.INITIAL_EQUITY_USD
    return {
        "name": name,
        "equity_curve": equity_curve,
        "final_equity": final_equity,
        "net_profit": net_profit,
        "trades": trades,
        "total_trades": len(trades)
    }

def calculate_metrics(result):
    """Calculate detailed metrics from evaluation result"""
    trades = pd.DataFrame(result["trades"])
    
    metrics = {
        "Model": result["name"],
        "Final Equity": f"${result['final_equity']:.2f}",
        "Net Profit": f"${result['net_profit']:.2f}",
        "Total Trades": result["total_trades"],
    }
    
    if len(trades) > 0:
        winners = trades[trades["net_pips"] > 0]
        win_rate = len(winners) / len(trades) * 100
        
        avg_win = winners["net_pips"].mean() if len(winners) > 0 else 0
        avg_loss = trades[trades["net_pips"] <= 0]["net_pips"].mean() if len(trades) > len(winners) else 0
        profit_factor = abs(winners["net_pips"].sum() / trades[trades["net_pips"] <= 0]["net_pips"].sum()) if avg_loss != 0 else float('inf')
        
        metrics["Win Rate"] = f"{win_rate:.1f}%"
        metrics["Profit Factor"] = f"{profit_factor:.2f}"
        metrics["Avg Win (Pips)"] = f"{avg_win:.1f}"
        metrics["Avg Loss (Pips)"] = f"{avg_loss:.1f}"
        
        # Drawdown calculation from equity curve
        equity = np.array(result["equity_curve"])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min() * 100
        metrics["Max Drawdown"] = f"{max_dd:.2f}%"
        
    else:
        metrics.update({
            "Win Rate": "0%", "Profit Factor": "0.00", 
            "Avg Win (Pips)": "0.0", "Avg Loss (Pips)": "0.0",
            "Max Drawdown": "0.00%"
        })
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="A/B Testing for Trading Models")
    parser.add_argument("--model-a", type=str, required=True, help="Path to Model A (.zip)")
    parser.add_argument("--model-b", type=str, required=True, help="Path to Model B (.zip)")
    parser.add_argument("--symbol", type=str, default="EURUSD=X")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--period", type=str, default="1y")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    print(f"⚔️ A/B Test Başlatılıyor: {args.symbol}")
    print(f"   Model A: {os.path.basename(args.model_a)}")
    print(f"   Model B: {os.path.basename(args.model_b)}")
    
    # Data loading
    print(f"Veri indiriliyor: {args.symbol}...")
    raw_df = DataLoader.download_yfinance(symbol=args.symbol, interval=args.interval, period=args.period)
    df, feature_cols = DataProcessor.add_indicators(raw_df)
    
    # Use last 20% for testing (or custom logic)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    print(f"Test veri seti: {len(test_df)} bar")
    
    # Load scaler if available
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_{args.symbol.replace('=', '_').replace('-', '_')}.pkl")
    if os.path.exists(scaler_path):
        print(f"🔄 Feature Scaler yükleniyor...")
        scaler = DataProcessor.load_scaler(scaler_path)
        test_df = DataProcessor.scale_features(test_df, feature_cols, scaler)
    else:
        print(f"⚠️  Scaler bulunamadı: {scaler_path}")
    
    # Create environments
    def make_env():
        return create_eval_env(test_df, feature_cols)
        
    env_a = DummyVecEnv([make_env])
    env_b = DummyVecEnv([make_env])
    
    # Load Models
    # Try loading as PPO, if fails try RecurrentPPO
    def load_any_model(path, env):
        try:
            return PPO.load(path, env=env)
        except:
            try:
                print(f"⚠️ PPO load failed, trying RecurrentPPO for {path}")
                return RecurrentPPO.load(path, env=env)
            except Exception as e:
                print(f"❌ Could not load model {path}: {e}")
                sys.exit(1)
                
    model_a = load_any_model(args.model_a, env_a)
    model_b = load_any_model(args.model_b, env_b)
    
    # Evaluate
    res_a = evaluate_model(model_a, env_a, name="Model A")
    res_b = evaluate_model(model_b, env_b, name="Model B")
    
    # Metrics
    metrics_a = calculate_metrics(res_a)
    metrics_b = calculate_metrics(res_b)
    
    # Display Report
    df_metrics = pd.DataFrame([metrics_a, metrics_b])
    print("\n📊 KARŞILAŞTIRMALI RAPOR")
    print("="*60)
    print(df_metrics.T.to_string())
    print("="*60)
    
    # Winner logic
    profit_a = res_a['net_profit']
    profit_b = res_b['net_profit']
    
    if profit_a > profit_b:
        print(f"🏆 KAZANAN: Model A (+${profit_a - profit_b:.2f} fark)")
    elif profit_b > profit_a:
        print(f"🏆 KAZANAN: Model B (+${profit_b - profit_a:.2f} fark)")
    else:
        print("🤝 BERABERE")
        
    # Plotting
    if not args.no_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(res_a['equity_curve'], label=f"Model A ({os.path.basename(args.model_a)})", color='blue')
        plt.plot(res_b['equity_curve'], label=f"Model B ({os.path.basename(args.model_b)})", color='orange')
        plt.axhline(y=cfg.INITIAL_EQUITY_USD, color='gray', linestyle='--')
        plt.title(f"A/B Test Equity Comparison - {args.symbol}")
        plt.xlabel("Steps")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = os.path.join(cfg.OUTPUTS_DIR, "ab_test_result.png")
        plt.savefig(output_file)
        print(f"\n📈 Grafik kaydedildi: {output_file}")
        plt.show()

if __name__ == "__main__":
    main()
