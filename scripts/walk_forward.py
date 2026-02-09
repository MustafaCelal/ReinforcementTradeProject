"""
Walk-Forward Validation
=======================
Rolling window ile daha gerçekci model değerlendirmesi.
Her pencerede yeni model eğitilir ve bir sonraki dönemde test edilir.

Kullanım: python scripts/walk_forward.py --symbol EURUSD=X --windows 4
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.core.environment import ForexTradingEnv
import config.settings as cfg


def create_env(df, feature_cols, random_start=True, episode_max_steps=None):
    """Environment factory"""
    return ForexTradingEnv(
        df=df,
        window_size=cfg.WINDOW_SIZE,
        sl_options=cfg.SL_OPTIONS,
        tp_options=cfg.TP_OPTIONS,
        spread_pips=cfg.SPREAD_PIPS,
        commission_pips=cfg.COMMISSION_PIPS,
        max_slippage_pips=cfg.MAX_SLIPPAGE_PIPS,
        random_start=random_start,
        min_episode_steps=500,
        episode_max_steps=episode_max_steps,
        feature_columns=feature_cols,
        initial_equity_usd=cfg.INITIAL_EQUITY_USD,
        lot_size=cfg.LOT_SIZE_MICRO,
    )


def evaluate_model(model, eval_env):
    """Model performansını değerlendir"""
    obs = eval_env.reset()
    equity_curve = []
    trades = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = eval_env.step(action)
        
        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        equity_curve.append(info.get("equity_usd", cfg.INITIAL_EQUITY_USD))
        
        # İşlem sayısını takip et
        events = info.get("events", [])
        trades += sum(1 for e in events if e.get("event") == "CLOSE")
        
        if done:
            break
    
    final_equity = float(equity_curve[-1])
    profit = final_equity - cfg.INITIAL_EQUITY_USD
    
    return {
        "equity_curve": equity_curve,
        "final_equity": final_equity,
        "profit": profit,
        "trades": trades,
        "profit_pct": (profit / cfg.INITIAL_EQUITY_USD) * 100
    }


def walk_forward_split(df, n_windows=4, train_ratio=0.7):
    """
    Walk-forward split şeması oluştur
    
    Örnek (4 window):
        Window 1: [Train: 0-55%] [Test: 55-70%]
        Window 2: [Train: 15-70%] [Test: 70-85%]
        Window 3: [Train: 30-85%] [Test: 85-100%]
        Window 4: [Train: 0-70%] [Test: 70-100%] (full validation)
    """
    n = len(df)
    splits = []
    
    # Son window: full validation (geleneksel split)
    final_train_end = int(n * train_ratio)
    
    # Sliding window splits
    window_size = int(n * train_ratio)  # Her window bu kadar veri kullanır
    step = int(n * (1 - train_ratio) / (n_windows - 1)) if n_windows > 1 else 0
    
    for i in range(n_windows - 1):
        train_start = i * step
        train_end = train_start + int(window_size * 0.75)
        test_start = train_end
        test_end = min(train_start + window_size, n)
        
        if test_end > n:
            break
            
        splits.append({
            "window": i + 1,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
    
    # Final full validation window
    splits.append({
        "window": n_windows,
        "train_start": 0,
        "train_end": final_train_end,
        "test_start": final_train_end,
        "test_end": n,
    })
    
    return splits


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--symbol", type=str, default="EURUSD=X")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--period", type=str, default="max")
    parser.add_argument("--windows", type=int, default=4, help="Number of validation windows")
    parser.add_argument("--timesteps", type=int, default=200000, help="Timesteps per window")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    print(f"📊 Walk-Forward Validation Başlatılıyor...")
    print(f"   Sembol: {args.symbol}")
    print(f"   Window sayısı: {args.windows}")
    
    # Veri yükle
    print(f"\nVeri indiriliyor: {args.symbol}...")
    raw_df = DataLoader.download_yfinance(symbol=args.symbol, interval=args.interval, period=args.period)
    df, feature_cols = DataProcessor.add_indicators(raw_df)
    
    print(f"Toplam bar sayısı: {len(df)}")
    
    # Split'leri oluştur
    splits = walk_forward_split(df, n_windows=args.windows)
    
    print("\n📋 Walk-Forward Split Planı:")
    for s in splits:
        print(f"   Window {s['window']}: Train[{s['train_start']}:{s['train_end']}] "
              f"-> Test[{s['test_start']}:{s['test_end']}]")
    
    # Her window için eğit ve test et
    results = []
    
    for split in splits:
        window_num = split["window"]
        print(f"\n{'='*60}")
        print(f"🔄 Window {window_num}/{len(splits)}")
        print(f"{'='*60}")
        
        # Veri split
        train_df = df.iloc[split["train_start"]:split["train_end"]].copy()
        test_df = df.iloc[split["test_start"]:split["test_end"]].copy()
        
        print(f"Train bars: {len(train_df)}, Test bars: {len(test_df)}")
        
        if len(train_df) < cfg.WINDOW_SIZE + 50 or len(test_df) < cfg.WINDOW_SIZE + 50:
            print(f"⚠️ Window {window_num} atlanıyor - yetersiz veri")
            continue
        
        # Environment oluştur
        train_env = DummyVecEnv([lambda: create_env(train_df, feature_cols, random_start=True, episode_max_steps=1500)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        eval_env = DummyVecEnv([lambda: create_env(test_df, feature_cols, random_start=False, episode_max_steps=None)])
        
        # Model eğit
        print(f"🚀 Eğitim başlıyor (Window {window_num})...")
        
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=0,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            learning_rate=0.0001,
            batch_size=128,
            n_steps=2048,
            ent_coef=0.02,
        )
        
        model.learn(total_timesteps=args.timesteps)
        
        # Test et
        print(f"📊 Test ediliyor (Window {window_num})...")
        result = evaluate_model(model, eval_env)
        result["window"] = window_num
        result["train_bars"] = len(train_df)
        result["test_bars"] = len(test_df)
        results.append(result)
        
        print(f"   Final Equity: ${result['final_equity']:.2f}")
        print(f"   Profit: ${result['profit']:.2f} ({result['profit_pct']:.1f}%)")
        print(f"   Trades: {result['trades']}")
        
        # Cleanup
        train_env.close()
        eval_env.close()
    
    # Sonuçları özetle
    print("\n" + "="*70)
    print("📈 WALK-FORWARD VALİDASYON SONUÇLARI")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print(results_df[["window", "test_bars", "trades", "profit", "profit_pct"]].to_string(index=False))
        
        print(f"\n📊 Aggregate Metrikler:")
        print(f"   Ortalama Profit: ${results_df['profit'].mean():.2f}")
        print(f"   Toplam Profit: ${results_df['profit'].sum():.2f}")
        print(f"   Ortalama Profit %: {results_df['profit_pct'].mean():.1f}%")
        print(f"   Kârlı Window: {len(results_df[results_df['profit'] > 0])}/{len(results_df)}")
        print(f"   Ortalama Trade/Window: {results_df['trades'].mean():.1f}")
        
        # Consistency score
        win_rate = len(results_df[results_df['profit'] > 0]) / len(results_df)
        consistency = "🟢 İyi" if win_rate >= 0.7 else ("🟡 Orta" if win_rate >= 0.5 else "🔴 Zayıf")
        print(f"   Tutarlılık: {consistency} ({win_rate*100:.0f}% windows profitable)")
        
        # Sonuçları kaydet
        output_path = os.path.join(cfg.OUTPUTS_DIR, "walk_forward_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Sonuçlar kaydedildi: {output_path}")
        
        # Plot
        if not args.no_plot and len(results) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Window profits
            ax1 = axes[0, 0]
            colors = ['green' if p > 0 else 'red' for p in results_df['profit']]
            ax1.bar(results_df['window'], results_df['profit'], color=colors)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax1.set_xlabel('Window')
            ax1.set_ylabel('Profit ($)')
            ax1.set_title('Profit per Window')
            
            # Cumulative profit
            ax2 = axes[0, 1]
            cumulative = results_df['profit'].cumsum()
            ax2.plot(results_df['window'], cumulative, marker='o')
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax2.set_xlabel('Window')
            ax2.set_ylabel('Cumulative Profit ($)')
            ax2.set_title('Cumulative Profit')
            ax2.fill_between(results_df['window'], 0, cumulative, alpha=0.3)
            
            # Equity curves (last window)
            ax3 = axes[1, 0]
            if results and 'equity_curve' in results[-1]:
                ax3.plot(results[-1]['equity_curve'])
                ax3.axhline(y=cfg.INITIAL_EQUITY_USD, color='red', linestyle='--', label='Initial')
                ax3.set_xlabel('Steps')
                ax3.set_ylabel('Equity ($)')
                ax3.set_title(f'Last Window Equity Curve')
                ax3.legend()
            
            # Trade count per window
            ax4 = axes[1, 1]
            ax4.bar(results_df['window'], results_df['trades'], color='steelblue')
            ax4.set_xlabel('Window')
            ax4.set_ylabel('Trade Count')
            ax4.set_title('Trades per Window')
            
            plt.suptitle(f'Walk-Forward Validation - {args.symbol}', fontsize=14)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
