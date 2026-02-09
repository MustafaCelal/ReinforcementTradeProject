# Reinforcement Trading Bot - Professional Edition

Bu proje, Takviyeli Öğrenme (Reinforcement Learning) algoritmalarını kullanarak finansal piyasalarda otomatik işlem yapan modüler ve profesyonel bir sistemdir.

## 🚀 Proje Hakkında

Proje, fiyat hareketlerini analiz ederek kârlı alım-satım kararları vermeyi öğrenen bir **PPO (Proximal Policy Optimization)** ajanı üzerine kuruludur.

### Temel Özellikler
- **Modüler Mimari:** SOLID prensiplerine uygun, katmanlı (Data, Core, UI, Utils) yapı.
- **Dinamik Veri:** Farklı zaman dilimleri (15m, 1h, 1d) ve periyotlarda eğitim ve test desteği.
- **Transfer Learning:** Bir paritede eğitilmiş modeli başka bir pariteye aktarma (Tecrübe Aktarımı).
- **Gerçekçi Simülasyon:** 100 USD başlangıç bakiyesi, Mikro Lot (0.01) ve gerçek piyasa maliyetleri.
- **Merkezi Yönetim:** Tüm bakiye ve strateji ayarları `config/settings.py` üzerinden yönetilir.

## 🧠 Model ve Eğitim Detayları

Bu bot, piyasa verilerini analiz ederek en uygun aksiyonu seçmek için derin pekiştirmeli öğrenme kullanır.

### RL Algoritması
- **Algoritma:** PPO (Proximal Policy Optimization) - [Stable Baselines3](https://stable-baselines3.readthedocs.io/) kütüphanesi kullanılmaktadır.
- **Alternatif:** RecurrentPPO (LSTM) - `sb3-contrib` ile hafıza tabanlı politika ağı desteği.
- **Ağ Yapısı:** Multi-Layer Perceptron (MLP) ile ikişer adet 256 nöronluk gizli katman (Policy ve Value ağları için).
- **Normalizasyon:** Eğitim stabilitesi için `VecNormalize` (Observation & Reward normalization) kullanılmaktadır.

### Ödül (Reward) Fonksiyonu
Sistem, sadece kâr/zarara odaklanmak yerine şu faktörleri içeren gelişmiş bir ödül mekanizması kullanır:
- **Gerçekleşen PnL (Pips):** İşlem kapandığında kâr/zarar baz alınır.
- **Maliyetler:** Spread, komisyon ve kayma (slippage) maliyetleri ödülden düşülür.
- **Sharpe Ratio Bonusu:** Risk-ayarlı performansı teşvik eden rolling Sharpe hesaplaması. *(YENİ)*
- **Ödül Şekillendirme (Reward Shaping):**
    - **Overtrading Cezası:** Gereksiz işlem açılmasını önlemek için her işlem açılışında sabit pip cezası.
    - **Holding Bonusu:** Kârlı pozisyonda kalınan her bar için küçük bir teşvik primi.
    - **Zaman Maliyeti (Time Penalty):** Pozisyonda beklenen her bar için küçük bir ceza (stagnasyonu önlemek için).
    - **Trend Uyumu:** 20 ve 50 periyotluk hareketli ortalamaların (MA) yönüne göre trend ile uyumlu işlemlere bonus, ters işlemlere ceza.
    - **ATR Tabanlı SL/TP:** Volatiliteye duyarlı dinamik stop-loss ve take-profit seviyeleri.
    - **Asimetrik Kayıp Ağırlığı:** Zararlı işlemler, kârlı işlemlere göre daha yüksek çarpanla (2.5x) cezalandırılarak modelin daha temkinli olması sağlanır.

### Gözlem (State) Uzayı
Model, her adımda şu verileri içeren geçmişe dönük bir pencere (Sliding Window size: 30) görür:
- **Teknik Göstergeler:** RSI, ATR, MA Eğimleri, MA Farkı (Spread), MACD, Bollinger Bant Genişliği.
- **İçsel Durum (Agent State):** Mevcut pozisyon (-1: Short, 0: Flat, 1: Long), işlemde geçen süre, gerçekleşmemiş kâr/zarar (scaled unrealized PnL).

### Aksiyon Uzayı
Bot, ayrık (discrete) bir aksiyon uzayına sahiptir:
- **0: HOLD** - Hiçbir şey yapma veya pozisyonu koru.
- **1: CLOSE** - Mevcut açık pozisyonu kapat.
- **2..N: OPEN** - Yeni bir pozisyon aç (Yön: Long/Short, parametreler: SL ve TP opsiyonları).

## 📁 Dosya Yapısı

- **`scripts/`**: Ana giriş noktaları:
  - `train_agent.py` - Standart PPO eğitimi
  - `train_recurrent.py` - RecurrentPPO (LSTM) eğitimi *(YENİ)*
  - `test_agent.py` - Model testi
  - `optimize_hyperparams.py` - Optuna ile hiperparametre optimizasyonu *(YENİ)*
  - `walk_forward.py` - Walk-forward validation *(YENİ)*
- **`src/`**: Çekirdek iş mantığı ve modüller.
  - `data/`: Veri indirme ve işleme (Loader & Processor).
  - `core/`: RL Ortamı (Environment).
  - `ui/`: Arayüz bileşenleri ve görselleştirme.
  - `utils/`: Raporlama ve yardımcı araçlar.
- **`models/`**: Eğitilmiş bot modelleri (`.zip`).
- **`outputs/`**: Backtest sonuçları (Grafikler ve CSV raporları).
- **`config/`**: Merkezi yapılandırma ayarları.

## 🛠️ Gelişmiş Özellikler

### Optuna Hiperparametre Optimizasyonu
En iyi model parametrelerini otomatik bulma:
```bash
python scripts/optimize_hyperparams.py --trials 20 --symbol EURUSD=X
```

### RecurrentPPO (LSTM) Eğitimi
Hafıza tabanlı model eğitimi:
```bash
python scripts/train_recurrent.py --symbol EURUSD=X --steps 300000
```

### Walk-Forward Validation
Daha gerçekci model değerlendirmesi:
```bash
python scripts/walk_forward.py --symbol EURUSD=X --windows 4
```

## ⚙️ Hızlı Kurulum ve Başlatma

Bu projede karmaşık terminal komutlarıyla uğraşmanıza gerek yoktur. Her şeyi otomatik hale getirdik:

1.  Proje klasöründeki **`TradingBot_Baslat.command`** dosyasına çift tıklayın.
2.  **İlk çalıştırmada:** Program gerekli sanal ortamı (`venv`) otomatik kuracak ve kütüphaneleri yükleyecektir (bu işlem birkaç dakika sürebilir).
3.  **Sonraki çalıştırmalarda:** Saniyeler içinde Dashboard açılacaktır.

Bu işlem sonrası açılan kontrol paneli üzerinden sembol seçebilir, eğitimi başlatabilir veya backtest sonuçlarını anlık olarak izleyebilirsiniz.

---
*Bu proje eğitim amaçlıdır. Finansal tavsiye niteliği taşımaz.*
