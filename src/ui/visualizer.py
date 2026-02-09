import pandas as pd
import mplfinance as mpf
import numpy as np

def plot_trading_results(df: pd.DataFrame, trades_df: pd.DataFrame, title: str = "Forex Trading Results", save_path: str = 'trading_chart.png'):
    """
    Plots the OHLC chart with buy/sell markers using mplfinance.
    """
    if trades_df.empty:
        print("No trades to plot.")
        return

    # Prepare marker data
    # Create empty arrays for markers
    buys = np.full(len(df), np.nan)
    sells = np.full(len(df), np.nan)
    
    # Fill markers based on trades
    for _, trade in trades_df.iterrows():
        step = int(trade['step'])
        if step < len(df):
            if trade['position'] == 1: # Long
                buys[step] = df.iloc[step]['Low'] * 0.9995 # Offset below low
            else: # Short
                sells[step] = df.iloc[step]['High'] * 1.0005 # Offset above high

    # Add markers to extra plots
    apds = [
        mpf.make_addplot(buys, type='scatter', markersize=100, marker='^', color='green'),
        mpf.make_addplot(sells, type='scatter', markersize=100, marker='v', color='red')
    ]

    # Save to file or show
    # Note: Using 'binance' style for a modern look
    print(f"Grafik oluşturuluyor: {title}")
    
    # Slice markers for the last 300 steps
    plot_df = df.tail(300)
    buys_slice = buys[-300:]
    sells_slice = sells[-300:]
    
    plot_apds = []
    if not np.all(np.isnan(buys_slice)):
        plot_apds.append(mpf.make_addplot(buys_slice, type='scatter', markersize=100, marker='^', color='green'))
    if not np.all(np.isnan(sells_slice)):
        plot_apds.append(mpf.make_addplot(sells_slice, type='scatter', markersize=100, marker='v', color='red'))

    mpf.plot(plot_df, type='candle', style='charles', 
             addplot=plot_apds if plot_apds else None, 
             title=title, 
             volume=True, figratio=(12, 8),
             savefig=save_path)
    
    print(f"Grafik '{save_path}' olarak kaydedildi.")
