# Model Eğitimi İyileştirme - Walkthrough

## Yapılan Değişiklikler

### 1. Risk:Reward Oranı Takibi
- Pozisyon açılırken `current_rr_ratio`, `current_sl_pips`, `current_tp_pips` kaydediliyor
- Trade event'lerine `rr_ratio` eklendi

### 2. R:R Bazlı Bonus/Penalty Sistemi

| Durum | R:R Oranı | Etki |
|-------|-----------|------|
| Karlı işlem | ≥ 2.0 | +20% bonus |
| Karlı işlem | ≥ 1.5 | +10% bonus |
| Zararlı işlem | < 1.0 | +30% ceza |
| Zararlı işlem | < 1.5 | +15% ceza |

### 3. Asymmetric Loss Multiplier
```diff
- Zarar: 1.5x
+ Zarar: 2.5x
```

### 4. Dinamik Drawdown Penalty
| Unrealized Loss | Penalty |
|-----------------|---------|
| > %50 SL | -0.20 |
| > %30 SL | -0.10 |
| > 10 pip | -0.05 |

### 5. SL/TP Seçenekleri

```diff
- SL: [5, 10, 15, 25, 30, 60, 90, 120]
- TP: [5, 10, 15, 25, 30, 60, 90, 120]
+ SL: [10, 15, 20, 25, 30]
+ TP: [20, 30, 40, 50, 60, 75]
```

### 6. Reward Shaping Aktif

| Parametre | Eski | Yeni |
|-----------|------|------|
| hold_reward_weight | 0.0 | 0.01 |
| open_penalty_pips | 0.0 | 0.5 |
| time_penalty_pips | 0.0 | 0.01 |
| unrealized_delta_weight | 0.0 | 0.02 |

---

## Doğrulama Sonuçları

```
✅ Environment created successfully
✅ Action space size: 62 (30 SL/TP combo × 2 yön)
✅ R:R tracking: 7.50 (10 SL / 75 TP)
✅ Open penalty: -1.275 (0.5 pip × 2.5 scale)
✅ Hold rewards: ~-0.015/bar (time cost + delta)
✅ Trade close event'ine rr_ratio eklendi
```

---

## Sonraki Adımlar

1. **Yeni modeli eğit:**
   ```bash
   python scripts/train_agent.py --steps 100000
   ```

2. **Tensorboard ile takip:**
   ```bash
   tensorboard --logdir=tensorboard_log
   ```

3. **Trade history'yi analiz et** - Ortalama kar/zarar pip oranına bak
