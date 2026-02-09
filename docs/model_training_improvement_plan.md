# Model Eğitimi İyileştirme Planı

## Sorun Özeti

Model %70-80 başarılı kapanış oranına sahip olmasına rağmen zarar ediyor. **Ana Neden:** Risk/Ödül dengesizliği - küçük karlar, büyük zararlar.

### Tespit Edilen Sorunlar

| Sorun | Dosya | Mevcut Değer | Etki |
|-------|-------|--------------|------|
| Dengesiz SL/TP seçenekleri | `settings.py` | 5-120 pip aynı | Model 5 pip TP, 120 pip SL seçebilir |
| Zayıf zarar cezası | `environment.py:466` | 1.5x multiplier | Zararları yeterince hissettirmiyor |
| R:R kontrolü yok | `environment.py` | Yok | Kötü oranlar cezalandırılmıyor |
| Shaping devre dışı | `train_agent.py:100-103` | Tümü 0.0 | Ara sinyaller yok |

---

## Proposed Changes

### Faz 1: Reward Function İyileştirmeleri

---

#### [MODIFY] [environment.py](file:///Users/mcguler/MyProjects/ReinforcementTrading_Part_1/src/core/environment.py)

**Değişiklik 1: R:R Ratio Tracking & Penalty (Satır 233-263, `_open_position`)**
```python
# Pozisyon açarken R:R oranını kaydet
self.current_rr_ratio = tp_pips / sl_pips
# Kötü R:R için penalty döndür
rr_penalty = 0.0
if self.current_rr_ratio < 1.0:
    rr_penalty = 1.0  # Çok kötü
elif self.current_rr_ratio < 1.5:
    rr_penalty = 0.3  # Kötü
```

**Değişiklik 2: Asymmetric Loss Multiplier Artırımı (Satır 466-469)**
```diff
-        if reward_pips < 0:
-            reward = float(reward_pips) * 1.5 * self.reward_scale 
+        if reward_pips < 0:
+            reward = float(reward_pips) * 2.5 * self.reward_scale
```

**Değişiklik 3: R:R Bazlı Bonus/Penalty İşlem Kapanışında (Satır 265-303, `_close_position`)**
```python
# Kar alınırken iyi R:R ödüllendir
if net_pips > 0 and hasattr(self, 'current_rr_ratio'):
    if self.current_rr_ratio >= 2.0:
        net_pips *= 1.2  # %20 bonus
    elif self.current_rr_ratio >= 1.5:
        net_pips *= 1.1  # %10 bonus
```

**Değişiklik 4: Drawdown Penalty Güçlendirme (Satır 471-475)**
```diff
-            if unreal_pips < -20:
-                reward -= 0.05 * self.reward_scale 
+            # SL'nin %50'sine ulaşırsa daha agresif ceza
+            if hasattr(self, 'current_sl_pips') and unreal_pips < -self.current_sl_pips * 0.5:
+                reward -= 0.2 * self.reward_scale
+            elif unreal_pips < -20:
+                reward -= 0.1 * self.reward_scale
```

---

#### [MODIFY] [settings.py](file:///Users/mcguler/MyProjects/ReinforcementTrading_Part_1/config/settings.py)

**Değişiklik: İyi R:R Zorlayan SL/TP Seçenekleri**
```diff
-SL_OPTIONS = [5, 10, 15, 25, 30, 60, 90, 120]
-TP_OPTIONS = [5, 10, 15, 25, 30, 60, 90, 120]
+# Risk:Reward >= 1.5 zorunlu kılmak için
+SL_OPTIONS = [10, 15, 20, 25, 30]
+TP_OPTIONS = [20, 30, 40, 50, 60, 75]
```

---

#### [MODIFY] [train_agent.py](file:///Users/mcguler/MyProjects/ReinforcementTrading_Part_1/scripts/train_agent.py)

**Değişiklik: Reward Shaping Parametrelerini Aktifleştir**
```diff
-            hold_reward_weight=0.0,
-            open_penalty_pips=0.0,
-            time_penalty_pips=0.0,
-            unrealized_delta_weight=0.0
+            hold_reward_weight=0.01,      # Karlı pozisyonu tutma bonusu
+            open_penalty_pips=0.5,        # Aşırı işlem yapma cezası  
+            time_penalty_pips=0.01,       # Uzun süre beklememe teşviki
+            unrealized_delta_weight=0.02  # Ara PnL sinyali
```

---

### Faz 2: İleriye Dönük Eğitim Stratejisi

> [!IMPORTANT]
> Bu faz Faz 1 tamamlandıktan ve test edildikten sonra uygulanmalıdır.

#### Aşama 2.1: Curriculum Learning
- Önce düşük volatilite dönemlerinde eğit
- Sonra yüksek volatilite dönemlerine geç
- `DataProcessor`'a volatilite filtresi ekle

#### Aşama 2.2: Multi-Timeframe Özellikler
- 1H veriye ek olarak 4H ve Daily trend bilgisi ekle
- `feature_columns`'a üst zaman dilimi indikatörleri dahil et

#### Aşama 2.3: Profit Factor Metriği
- Eğitim sırasında win rate yerine Profit Factor takibi
- `Profit Factor = Toplam Kar / Toplam Zarar`
- En iyi modeli PF'ye göre seç (mevcut equity bazlı seçim yerine)

#### Aşama 2.4: Ensemble Trading
- Farklı hiperparametrelerle 3-5 model eğit
- Çoğunluk oylamasıyla karar ver

---

## Verification Plan

### Otomatik Testler

> [!NOTE]
> `tests/` dizini şu anda boş. İyileştirme sonrası manuel doğrulama yapılacak.

**Test 1: Environment Reward Hesaplaması**
```bash
cd /Users/mcguler/MyProjects/ReinforcementTrading_Part_1
python -c "
from src.core.environment import ForexTradingEnv
import pandas as pd
import numpy as np

# Dummy data
df = pd.DataFrame({
    'Open': np.random.randn(500).cumsum() + 1.1,
    'High': np.random.randn(500).cumsum() + 1.12,
    'Low': np.random.randn(500).cumsum() + 1.08,
    'Close': np.random.randn(500).cumsum() + 1.1,
})

env = ForexTradingEnv(df=df, sl_options=[15,20,25], tp_options=[30,40,50], window_size=30)
obs, info = env.reset()
print('Environment created successfully')
print(f'Action space: {env.action_space}')
print(f'Observation shape: {obs.shape}')
"
```

### Manuel Doğrulama (Kullanıcı Tarafından)

1. **Kısa Eğitim Testi**
   ```bash
   cd /Users/mcguler/MyProjects/ReinforcementTrading_Part_1
   python scripts/train_agent.py --steps 50000 --no-plot
   ```
   - Eğitim hatasız tamamlanmalı
   - Equity curve grafiğini incele

2. **Trade History Analizi**
   - `outputs/trade_history_output.csv` dosyasını kontrol et
   - Ortalama kar işlem pip / ortalama zarar işlem pip oranına bak
   - Bu oran 1.5'in üzerinde olmalı

3. **Tensorboard Monitoring**
   ```bash
   tensorboard --logdir=tensorboard_log
   ```
   - Reward curve'ün istikrarlı artışını kontrol et
