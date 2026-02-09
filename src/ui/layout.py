# src/ui/layout.py

import streamlit as st
import pandas as pd
from PIL import Image
import time
import os
import config.settings as cfg

def show_context_info(symbol: str, period: str, model_name: str):
    """
    Renders a context box with time and session information.
    """
    st.info(f"📅 **Bağlam Bilgisi:** Bu test **{time.strftime('%Y-%m-%d %H:%M')}** tarihinde, **{symbol}** paritesi üzerinde **{period}** veri seti kullanılarak gerçekleştirilmiştir.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**🔍 Sembol:** {symbol}")
    with c2:
        st.write(f"**🕒 Periyot:** {period}")
    with c3:
        st.write(f"**🧠 Model:** {model_name}")
    st.markdown("---")

def show_performance_metrics(df_trades: pd.DataFrame):
    """
    Renders the 4 metric cards for backtest results.
    """
    st.subheader("📊 Gelişmiş Performans Özeti")
    
    total_trades = len(df_trades)
    net_pips = df_trades["net_pips"].sum()
    wins = df_trades[df_trades["net_pips"] > 0]
    losses = df_trades[df_trades["net_pips"] < 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    
    # Financial metrics (on full data)
    initial_usd = 100.0
    final_usd = df_trades["equity_usd"].iloc[-1]
    net_profit_usd = final_usd - initial_usd

    # Profit Factor calculation (on strategy trades for honesty)
    gross_profit = wins["net_pips"].sum()
    gross_loss = abs(losses["net_pips"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)

    # Row 1
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Toplam İşlem", f"{total_trades} Adet")
    with m_col2:
        st.metric("Başarı Oranı (Win Rate)", f"%{win_rate:.1f}", delta=f"{len(wins)} Galibiyet")
    with m_col3:
        st.metric("Final Bakiyesi", f"${final_usd:,.2f}", delta=f"${net_profit_usd:,.2f}")
    with m_col4:
        st.metric("Net Kazanç (Pip)", f"{net_pips:,.1f} Pip", delta=f"{len(losses)} Mağlubiyet", delta_color="inverse")

    # Row 2 (Detailed stats)
    st.markdown(" ")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.metric("Kâr Faktörü", f"{profit_factor:.2f}", help="Brüt Kâr / Brüt Zarar (1.5+ iyidir)")
    with s_col2:
        avg_pip = net_pips / total_trades if total_trades > 0 else 0
        st.metric("Ort. Pip / İşlem", f"{avg_pip:.1f} Pip")
    with s_col3:
        st.metric("Eğitim Başlangıç", f"${initial_usd:,.0f}")
    with s_col4:
        status = "✅ Kârda" if net_profit_usd > 0 else "❌ Zararda"
        st.metric("Sonuç Durumu", status)

def show_results_area(symbol: str):
    """
    Renders the chart and trade history table side by side.
    """
    st.markdown("---")
    col_img, col_stats = st.columns([2, 1])
    
    with col_img:
        st.subheader("📈 İşlem Grafiği")
        if os.path.exists(cfg.TRADING_CHART_FILE):
            image = Image.open(cfg.TRADING_CHART_FILE)
            # Use a unique key for the image to bypass Streamlit's old image cache if any
            st.image(image, caption=f"Sonuçlar ({symbol}) - {time.strftime('%H:%M:%S')}", width="stretch")
        else:
            st.info("Grafik henüz oluşturulmadı. Lütfen bir test çalıştırın.")
            
    with col_stats:
        st.subheader("📜 İşlem Geçmişi")
        if os.path.exists(cfg.TRADE_HISTORY_FILE):
            df_total = pd.read_csv(cfg.TRADE_HISTORY_FILE)
            # Filter out forced end-of-data rows for a cleaner strategy view
            df_strategy = df_total[df_total["reason"] != "END_OF_DATA"]
            
            # Clean up and reorder columns for better UI display
            display_cols = ["entry_time", "exit_time", "position", "entry_price", "exit_price", "net_pips", "reason"]
            # Ensure columns exist before filtering
            existing_cols = [c for c in display_cols if c in df_strategy.columns]
            df_display = df_strategy[existing_cols].copy()
            
            # Map position to text with robust mapping (handles int, float, str)
            if "position" in df_display.columns:
                # Add a raw indicator column for debugging
                df_display["Raw"] = df_display["position"].astype(str)
                pos_map = {
                    1: "🟢 ALIS (Long)", 1.0: "🟢 ALIS (Long)", "1": "🟢 ALIS (Long)", "1.0": "🟢 ALIS (Long)",
                    -1: "🔴 SATIS (Short)", -1.0: "🔴 SATIS (Short)", "-1": "🔴 SATIS (Short)", "-1.0": "🔴 SATIS (Short)"
                }
                df_display["position"] = df_display["position"].replace(pos_map)
            
            # Reorder with Raw column
            display_cols = ["entry_time", "exit_time", "position", "Raw", "entry_price", "exit_price", "net_pips", "reason"]
            existing_cols = [c for c in display_cols if c in df_display.columns]
            df_display = df_display[existing_cols]

            # Rename columns for localized UI
            df_display.columns = ["Giris Zamani", "Cikis Zamani", "Islem Yonu", "Ham", "Giris Fiyati", "Cikis Fiyati", "Kar/Zarar (Pip)", "Kapanis Nedeni"]

            st.dataframe(
                df_display, 
                hide_index=True,
                width="stretch"
            )
            st.info(f"💡 Tabloda test süresince gerçekleşen tüm işlemler ({len(df_strategy)} adet) listelenmektedir.")
        else:
            st.info("İşlem geçmişi bulunamadı.")
