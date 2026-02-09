# trading_env.py

from __future__ import annotations

import numpy as np

# Prefer gymnasium if available (SB3 supports it), fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False

from src.utils.logger import TradeLogger


class ForexTradingEnv(gym.Env):
    """
    RL Forex Trading Environment (Position-Persistent)

    Key properties:
      - Observation: rolling window of features + 3 state features (position, time_in_trade, unrealized_pnl_pips)
      - Actions:
          0: HOLD (do nothing)
          1: CLOSE (close position if any)
          2..: OPEN (direction + SL + TP), only effective when flat
      - Position persistence: once open, position remains until:
          - agent sends CLOSE, or
          - SL/TP hit intrabar
      - Friction: spread + commission + optional slippage
      - Reward:
          - realized PnL (pips) minus costs (pips) on closes
          - optional shaping via delta unrealized PnL (pips) while holding
      - Random episode start to reduce memorization / overfit
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        sl_options=None,
        tp_options=None,
        feature_columns = None,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,              # cost in pips per round-trip (approx)
        commission_pips: float = 0.0,          # cost in pips per round-trip
        max_slippage_pips: float = 0.0,        # random extra pips (0..max) applied on fills
        lot_size: float = 100000.0,            # 1.0 lot = 100k units (for equity in $)
        reward_scale: float = 1.0,             # optional scaling of rewards
        unrealized_delta_weight: float = 0.02, # shaping weight on delta-unrealized while holding
        random_start: bool = True,
        min_episode_steps: int = 300,          # minimum steps per episode (for random starts)
        episode_max_steps: int | None = None,  # optional cap (truncation)
        feature_mean: np.ndarray | None = None, # optional normalization (train-fitted)
        feature_std: np.ndarray | None = None,  # optional normalization (train-fitted)
        allow_flip: bool = False,               # if True, OPEN while in position flips (close+open). Default False.
        hold_reward_weight: float = 0.005,   # tuned below
        open_penalty_pips: float = 0.5,      # NEW: penalty per open
        time_penalty_pips: float = 0.02,     # NEW: cost per bar in a trade
        initial_equity_usd: float = 10000.0, # NEW: starting balance
        sharpe_reward_weight: float = 0.1,   # NEW: Sharpe ratio based reward weight
        drawdown_penalty_weight: float = 0.5,  # NEW: Maximum Drawdown penalty weight
        log_trades: bool = False,            # NEW: enable detailed file logging
    ):
        super().__init__()

        # Reset index but KEEP it as a column to have access to time/date info
        self.df = df.reset_index(drop=False)
        # Identify the time column name (usually 'index' after reset_index or 'Time')
        self.time_col = "index" if "index" in self.df.columns else ("Time" if "Time" in self.df.columns else None)
        
        self.n_steps = len(self.df)

        if feature_columns is None:
            self.feature_columns = list(self.df.columns)  # fallback: everything
        else:
            self.feature_columns = list(feature_columns)

        if sl_options is None or tp_options is None:
            raise ValueError("sl_options and tp_options must be provided (e.g. [15,20,30]).")
        self.sl_options = list(sl_options)
        self.tp_options = list(tp_options)

        if self.n_steps <= window_size + 2:
            raise ValueError("Dataframe is too short for the given window_size.")

        self.window_size = int(window_size)
        self.pip_value = float(pip_value)

        # Friction
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)

        # Equity accounting (approx): for EURUSD 1 pip per 1 lot ≈ $10
        # pip_value (price) * lot_size (units) ≈ $ per 1.0 price move.
        # 1 pip = pip_value price move, so $/pip ≈ pip_value * lot_size.
        self.lot_size = float(lot_size)
        self.usd_per_pip = self.pip_value * self.lot_size

        # Reward handling
        self.reward_scale = float(reward_scale)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)
        self.sharpe_reward_weight = float(sharpe_reward_weight)
        self.drawdown_penalty_weight = float(drawdown_penalty_weight)

        # Episode handling
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps if episode_max_steps is None else int(episode_max_steps)

        # Optional normalization (fit on train only, pass arrays here)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.allow_flip = bool(allow_flip)
        self.initial_equity_usd = float(initial_equity_usd)
        
        # Logging
        self.log_trades = log_trades
        if self.log_trades:
            self.logger = TradeLogger()
        else:
            self.logger = None

        # --- Actions ---
        # 0: HOLD
        # 1: CLOSE
        # 2..: OPEN(direction, sl, tp)
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:  # 0=short, 1=long
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))

        self.action_space = spaces.Discrete(len(self.action_map))

        # Observation features: df columns + 3 state features
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )

        # Internal state
        self._reset_state()

    # ----------------------------
    # Core Helpers
    # ----------------------------

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        # Position state
        self.position = 0              # 0=flat, +1=long, -1=short
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        
        # R:R tracking for reward shaping
        self.current_rr_ratio = 1.0
        self.current_sl_pips = 0.0
        self.current_tp_pips = 0.0

        # Accounting
        self.equity_usd = self.initial_equity_usd
        self.peak_equity_usd = self.initial_equity_usd  # Track peak for drawdown
        self.max_drawdown_pct = 0.0  # Track maximum drawdown %
        
        # Overtrading tracking
        self.last_trade_end_step = -100 # Initialize to avoid penalty on first trade

        # Logging
        self.equity_curve = []
        self.events = [] # List of events (OPEN/CLOSE) that happened this step
        
        # Sharpe Ratio tracking
        self.returns_history = []  # Track trade returns for rolling Sharpe calculation

    def _get_state_features(self):
        # position in [-1,0,1], time normalized, unrealized in pips (scaled)
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 1000.0
        unreal_pips = float(self._compute_unrealized_pips()) if self.position != 0 else 0.0
        unreal_scaled = unreal_pips / 100.0  # prevent huge magnitudes
        return np.array([pos, t_norm, unreal_scaled], dtype=np.float32)

    def _compute_unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close_price = float(self.df.loc[self.current_step, "Close"])
        if self.position == 1:
            pnl_price = close_price - self.entry_price
        else:
            pnl_price = self.entry_price - close_price
        return pnl_price / self.pip_value

    def _apply_optional_normalization(self, obs: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return obs
        mean = self.feature_mean.reshape(1, 1, -1)
        std = self.feature_std.reshape(1, 1, -1)
        std = np.where(std == 0, 1.0, std)
        return (obs - mean) / std

    def _get_observation(self):
        start = self.current_step - self.window_size
        if start < 0:
            start = 0

        obs_df = self.df.iloc[start:self.current_step].copy()
        # use only selected feature columns for the agent
        obs_df = obs_df[self.feature_columns]

        # If empty (safety), use the first row repeated
        if len(obs_df) == 0:
            base = np.tile(self.df.iloc[0].values.astype(np.float32), (self.window_size, 1))
        else:
            base = obs_df.values.astype(np.float32)
            if base.shape[0] < self.window_size:
                pad_rows = self.window_size - base.shape[0]
                pad = np.tile(base[0], (pad_rows, 1))
                base = np.vstack([pad, base])

        # Append state features (same for each row)
        state_feat = self._get_state_features()
        state_block = np.tile(state_feat, (self.window_size, 1))
        obs = np.hstack([base, state_block]).astype(np.float32)

        # Optional normalization (only if user passes train-fitted mean/std matching obs dims)
        obs = self._apply_optional_normalization(obs)

        return obs

    def _sample_slippage_pips(self) -> float:
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _cost_pips_round_trip(self) -> float:
        # Simple friction model (round-trip)
        return self.spread_pips + self.commission_pips

    def _calculate_sharpe_bonus(self) -> float:
        """
        Calculate a reward bonus/penalty based on rolling Sharpe Ratio.
        Encourages consistent, risk-adjusted returns over volatile performance.
        """
        if len(self.returns_history) < 10:
            return 0.0
        
        # Use last 50 trades for rolling calculation (or all if less)
        recent_returns = np.array(self.returns_history[-50:])
        
        if recent_returns.std() == 0:
            return 0.0
        
        # Annualized Sharpe (assuming ~252 trading days)
        sharpe = recent_returns.mean() / recent_returns.std() * np.sqrt(252)
        
        # Bound the bonus to prevent extreme values
        bonus = np.clip(sharpe * self.sharpe_reward_weight, -1.0, 1.0)
        
        return float(bonus)

    def _open_position(self, direction: int, sl_multiplier: float, tp_multiplier: float):
        # Fetch ATR for dynamic SL/TP
        atr_val = float(self.df.loc[self.current_step, "atr_14"])
        atr_pips = atr_val / self.pip_value
        
        # Guard against zero ATR
        if atr_pips <= 0:
            atr_pips = 10.0 # Fallback
            
        sl_pips = sl_multiplier * atr_pips
        tp_pips = tp_multiplier * atr_pips

        # Entry on current close + slippage; costs applied on close (round-trip model)
        close_price = float(self.df.loc[self.current_step, "Close"])
        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        if direction == 1:  # long
            entry = close_price + slip_price
            sl_price = entry - sl_pips * self.pip_value
            tp_price = entry + tp_pips * self.pip_value
            self.position = 1
        else:               # short
            entry = close_price - slip_price
            sl_price = entry + sl_pips * self.pip_value
            tp_price = entry - tp_pips * self.pip_value
            self.position = -1

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.entry_time = str(self.df.loc[self.current_step, self.time_col]) if self.time_col else self.current_step
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        
        # Track R:R ratio for reward shaping
        self.current_sl_pips = sl_pips
        self.current_tp_pips = tp_pips
        self.current_rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 1.0

        self.events.append({
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "entry_time": self.entry_time,
            "rr_ratio": self.current_rr_ratio,
            "atr_pips": float(atr_pips)
        })

    def _close_position(self, reason: str, exit_price: float):
        # Realized pips
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value

        # Costs in pips (round-trip)
        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips
        
        # Apply R:R ratio bonus/penalty to net_pips
        rr_ratio = getattr(self, 'current_rr_ratio', 1.0)
        if net_pips > 0:
            # Reward good R:R trades more
            if rr_ratio >= 2.0:
                net_pips *= 1.2  # 20% bonus for excellent R:R
            elif rr_ratio >= 1.5:
                net_pips *= 1.1  # 10% bonus for good R:R
        else:
            # Penalize bad R:R losing trades more
            if rr_ratio < 1.0:
                net_pips *= 1.3  # 30% extra penalty for terrible R:R
            elif rr_ratio < 1.5:
                net_pips *= 1.15  # 15% extra penalty for bad R:R

        # Update equity in USD
        self.equity_usd += net_pips * self.usd_per_pip

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_time": self.entry_time,
            "exit_time": str(self.df.loc[self.current_step, self.time_col]) if self.time_col else self.current_step,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(realized_pips),
            "cost_pips": float(cost_pips),
            "net_pips": float(net_pips),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
            "rr_ratio": float(rr_ratio),
        }

        # Reset position state
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0
        self.current_rr_ratio = 1.0
        self.current_sl_pips = 0.0
        self.current_tp_pips = 0.0
        self.last_trade_end_step = self.current_step # Track when trade ended

        self.events.append(trade_info)
        
        # Track return for Sharpe calculation (as percentage of entry)
        trade_return = net_pips / 100.0  # Normalize to percentage-like value
        self.returns_history.append(trade_return)
        
        # Apply Sharpe bonus to the realized pips
        sharpe_bonus = self._calculate_sharpe_bonus()
        net_pips_with_sharpe = net_pips + sharpe_bonus
        
        return net_pips_with_sharpe

    def _check_sl_tp_intrabar_and_maybe_close(self) -> float:
        """
        Checks SL/TP on the *next bar* range [Low, High].
        Conservative rule if both touched: assume SL hits first (worst case).
        Returns realized net pips if closed; otherwise None.
        """
        if self.position == 0:
            return None

        # If last bar, close on close
        if self.current_step >= self.n_steps - 2:
            exit_price = float(self.df.loc[self.current_step, "Close"])
            # Penalty for forced close to encourage agent taking initiative
            net_pips = self._close_position("END_OF_DATA", exit_price) - 2.0 
            return net_pips

        next_high = float(self.df.loc[self.current_step + 1, "High"])
        next_low = float(self.df.loc[self.current_step + 1, "Low"])

        if self.position == 1:
            sl_hit = next_low <= self.sl_price
            tp_hit = next_high >= self.tp_price
            if sl_hit and tp_hit:
                # conservative: SL first
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)
        else:
            sl_hit = next_high >= self.sl_price
            tp_hit = next_low <= self.tp_price
            if sl_hit and tp_hit:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)

        return None

    # ----------------------------
    # Gym API
    # ----------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_state()

        # Choose start
        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            if max_start <= self.window_size:
                self.current_step = self.window_size
            else:
                self.current_step = int(np.random.randint(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        obs = self._get_observation()

        if _GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action: int):
        # Reset event list for this specific step
        self.events = []

        if self.terminated or self.truncated:
            # If someone steps after done, just return current obs with 0 reward
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1

        # Reward components
        reward_pips = 0.0
        info = {}

        act_type, direction, sl_pips, tp_pips = self.action_map[int(action)]

        # 1) Apply action logic
        if act_type == "HOLD":
            pass

        elif act_type == "CLOSE":
            if self.position != 0:
                # Close at current close (with slippage)
                close_price = float(self.df.loc[self.current_step, "Close"])
                slip_pips = self._sample_slippage_pips()
                slip_price = slip_pips * self.pip_value
                exit_price = close_price - slip_price if self.position == 1 else close_price + slip_price
                reward_pips += self._close_position("MANUAL_CLOSE", exit_price)

        elif act_type == "OPEN":
            if self.position == 0:
                self._open_position(direction=direction, sl_multiplier=sl_pips, tp_multiplier=tp_pips)
                # penalty for opening a trade to discourage overtrading
                penalty = self.open_penalty_pips
                
                # Sık işlem açma cezası (Overtrading)
                # Eğer son işlemden sonra 20 bar geçmemişse ek ceza uygula
                steps_since_last = self.current_step - self.last_trade_end_step
                if steps_since_last < 20:
                    recency_penalty = (20 - steps_since_last) * 0.1 # Daha taze ise daha çok ceza
                    penalty += recency_penalty
                
                reward_pips -= penalty
            else:
                if self.allow_flip:
                    close_price = float(self.df.loc[self.current_step, "Close"])
                    reward_pips += self._close_position("FLIP_CLOSE", close_price)
                    self._open_position(direction=direction, sl_multiplier=sl_pips, tp_multiplier=tp_pips)
                    reward_pips -= self.open_penalty_pips * 1.5 # Flipping is more expensive


        # 2) If position is open, check SL/TP on next bar intrabar
        realized_now = self._check_sl_tp_intrabar_and_maybe_close()
        if realized_now is not None:
            reward_pips += realized_now

        # 3) If still open, apply reward shaping based on delta-unrealized pips
        # 3) If still open, apply reward shaping
        if self.position != 0:
            self.time_in_trade += 1

            unreal_now = self._compute_unrealized_pips()
            delta_unreal = unreal_now - self.prev_unrealized_pips

            # (a) Hold reward: Bonus for staying in a profitable trade
            # Increased weight to encourage 'letting profits run'
            if unreal_now > 0:
                reward_pips += self.hold_reward_weight * (unreal_now / 10.0) # Scaled bonus

            # (b) Trend Alignment Bonus: Reward trades moving with the 20/50 MA trend
            # ma_spread is in the observations. We can access it via df if needed
            ma_spread = self.df.loc[self.current_step, "ma_spread"]
            if (self.position == 1 and ma_spread > 0) or (self.position == -1 and ma_spread < 0):
                reward_pips += 0.05  # Small continuous bonus for trend alignment
            else:
                reward_pips -= 0.02  # Small penalty for fighting the trend

            # (c) Optional shaping on change in unrealized
            if self.unrealized_delta_weight != 0.0:
                reward_pips += self.unrealized_delta_weight * delta_unreal

            # (d) Time cost per bar to avoid infinite stagnation
            reward_pips -= self.time_penalty_pips

            self.prev_unrealized_pips = unreal_now



        # 4) Advance time
        self.current_step += 1

        # 5) Termination / truncation
        if self.current_step >= self.n_steps - 1:
            self.terminated = True

        if self.episode_max_steps is not None and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        # 6) Log equity and track drawdown
        self.equity_curve.append(float(self.equity_usd))
        
        # Track peak equity for drawdown calculation
        if self.equity_usd > self.peak_equity_usd:
            self.peak_equity_usd = self.equity_usd
        
        # Calculate current drawdown %
        if self.peak_equity_usd > 0:
            current_drawdown_pct = ((self.peak_equity_usd - self.equity_usd) / self.peak_equity_usd) * 100.0
            self.max_drawdown_pct = max(self.max_drawdown_pct, current_drawdown_pct)
        else:
            current_drawdown_pct = 0.0

        # 7) Build observation
        obs = self._get_observation()

        # 8) Final reward scaling and drawdown penalty
        # ENHANCEMENT: Penalize losses more heavily to encourage better Risk:Reward
        if reward_pips < 0:
            reward = float(reward_pips) * 2.5 * self.reward_scale  # Increased from 1.5x to 2.5x
        else:
            reward = float(reward_pips) * self.reward_scale

        # ===== MAXIMUM DRAWDOWN PENALTY (Hayatta Kalma İçgüdüsü) =====
        # Büyük drawdown'lar modele cezalandırılır, "agresif risk alma" engellenir
        if current_drawdown_pct > 0:
            # Drawdown severity penalty
            # 5% DD: -0.1, 10% DD: -0.4, 20% DD: -1.6, 30% DD: -3.6, 40% DD: -6.4
            drawdown_penalty = (current_drawdown_pct / 10.0) ** 2 * self.drawdown_penalty_weight
            reward -= drawdown_penalty
            
            # Extra penalty if approaching catastrophic loss (équity < 50% of initial)
            equity_ratio = self.equity_usd / self.initial_equity_usd
            if equity_ratio < 0.5:
                catastrophic_penalty = (0.5 - equity_ratio) * 10.0 * self.drawdown_penalty_weight
                reward -= catastrophic_penalty

        # Stronger penalty for unrealized loss while holding to discourage "hope" trading
        if self.position != 0:
            unreal_pips = self._compute_unrealized_pips()
            current_sl = getattr(self, 'current_sl_pips', 20.0)
            # More aggressive penalty when approaching SL
            if unreal_pips < -current_sl * 0.5:  # Past 50% of SL
                reward -= 0.2 * self.reward_scale
            elif unreal_pips < -current_sl * 0.3:  # Past 30% of SL
                reward -= 0.1 * self.reward_scale
            elif unreal_pips < -10:  # General drawdown penalty
                reward -= 0.05 * self.reward_scale

        if self.logger:
            log_entry = {
                "step": self.current_step,
                "equity": float(self.equity_usd),
                "reward": float(reward),
                "action": int(action),
                "position": int(self.position),
                "pnl_pips": float(reward_pips),
            }
            if self.events:
                log_entry["events"] = self.events
                # Separate detailed log for trade events
                for e in self.events:
                    self.logger.info(f"Event at step {self.current_step}: {e}")
            
            # Detailed step log (optional, can be heavy)
            # self.logger.log_step(log_entry)

        # 9) Info
        info.update({
            "equity_usd": float(self.equity_usd),
            "position": int(self.position),
            "time_in_trade": int(self.time_in_trade),
            "reward_pips": float(reward_pips),
            "events": self.events
        })

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            done = bool(self.terminated or self.truncated)
            return obs, reward, done, info

    def render(self):
        print(
            f"Step={self.current_step} | Equity=${self.equity_usd:,.2f} | "
            f"Pos={self.position} | Entry={self.entry_price} | SL={self.sl_price} | TP={self.tp_price}"
        )
