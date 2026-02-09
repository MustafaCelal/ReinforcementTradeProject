import pandas as pd
import numpy as np

def calculate_performance_metrics(trades_df: pd.DataFrame, initial_equity: float, final_equity: float):
    """
    Calculates key trading performance metrics from the trade history.
    """
    if trades_df.empty:
        return {
            "Total Trades": 0,
            "Profit/Loss": 0,
            "Win Rate": "0%",
            "Profit Factor": 0,
            "Max Drawdown": "0%"
        }

    # Net Pips and Profits
    total_trades = len(trades_df)
    net_pips = trades_df["net_pips"].sum()
    total_profit_usd = final_equity - initial_equity
    
    # Win Rate
    winning_trades = trades_df[trades_df["net_pips"] > 0]
    win_rate = (len(winning_trades) / total_trades) * 100
    
    # Profit Factor
    gross_profits = trades_df[trades_df["net_pips"] > 0]["net_pips"].sum()
    gross_losses = abs(trades_df[trades_df["net_pips"] < 0]["net_pips"].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    # Performance Summary String
    summary = f"""
=========================================
      📊 İŞLEM PERFORMANS ÖZETİ 📊
=========================================
🏠 Başlangıç Bakiyesi : ${initial_equity:,.2f}
💰 Final Bakiyesi      : ${final_equity:,.2f}
📈 Toplam Kâr/Zarar    : ${total_profit_usd:,.2f} ({net_pips:.1f} Pips)
-----------------------------------------
🔄 Toplam İşlem Sayısı : {total_trades}
🏆 Başarı Oranı (WR)   : %{win_rate:.1f}
⚖️ Kâr Faktörü         : {profit_factor:.2f}
=========================================
"""
    return summary

def print_fancy_summary(summary: str):
    print(summary)
