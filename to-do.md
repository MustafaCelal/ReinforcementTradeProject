# Proje Görev Listesi (TO-DO)

Bu dosya projenin gelişim sürecini takip etmek için kullanılacaktır.

## 🔴 Yüksek Öncelikli (Gelecek Aşama)
- [ ] **Çoklu Döviz Eğitimi:** Multi-currency training ile genel model oluşturma.
- [ ] **Ensemble Model:** Farklı zaman dilimlerinde eğitilmiş modellerin birleşimi.

## 🟡 Orta Öncelikli
- [ ] **Live Trading Bridge:** MetaTrader 5 (MT5) üzerinden canlı işlem entegrasyonu.
- [ ] **Haber Entegrasyonu:** Ekonomik takvim verilerinin (High Impact News) modele eklenmesi.

## 🟢 Düşük Öncelikli / Uzun Vadeli
- [ ] **Order Book Verileri:** Market microstructure entegrasyonu.
- [ ] **Docker Container:** Production-ready deployment.

## ✅ Tamamlananlar
- [x] Temel `ForexTradingEnv` ortamının oluşturulması.
- [x] PPO algoritması ile eğitim ve test altyapısı.
- [x] Teknik gösterge (RSI, ATR, MA) entegrasyonu.
- [x] **Otomatik Veri Çekme:** `yfinance` entegrasyonu.
- [x] **Görsel Raporlama:** `mplfinance` ile mum grafiklerinde buy/sell okları.
- [x] **Performans Raporu:** Win Rate, Profit Factor ve Net Pip istatistikleri.
- [x] **Verbose Logging:** Gerçek zamanlı işlem anlatımı (Open/Close logları).
- [x] **Strateji İyileştirme:** Risk-odaklı ödül fonksiyonu.
- [x] **Sürekli Eğitim:** Önceden eğitilmiş modelden devam etme (Fine-tuning).
- [x] **Web Dashboard:** `Streamlit` ile interaktif kontrol paneli.
- [x] **ATR Tabanlı SL/TP:** Volatiliteye duyarlı stop-loss ve take-profit.
- [x] **Sharpe Ratio Ödülü:** Risk-ayarlı ödül mekanizması.
- [x] **Optuna Entegrasyonu:** Hiperparametre optimizasyonu.
- [x] **RecurrentPPO (LSTM):** Hafıza tabanlı politika ağı desteği.
- [x] **Walk-Forward Validation:** Rolling window ile gerçekci backtest.
- [x] README ve Proje dökümantasyonu.
