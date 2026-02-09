import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.environment import ForexTradingEnv
from config.settings import TradingConfig

class TestForexTradingEnv:
    @pytest.fixture
    def mock_df(self):
        # Create a dummy dataframe with enough rows for window size
        # Increased to 1000 to avoid issues with min_episode_steps default (300)
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="1h")
        data = {
            "Open": np.full(1000, 1.1000),
            "High": np.full(1000, 1.1010),
            "Low": np.full(1000, 1.0990),
            "Close": np.full(1000, 1.1000),
            "Volume": np.full(1000, 1000),
            "rsi_14": np.full(1000, 50.0),
            "atr_14": np.full(1000, 0.0020), # 20 pips
            "ma_20_slope": np.zeros(1000),
            "ma_50_slope": np.zeros(1000),
            "close_ma20_diff": np.zeros(1000),
            "close_ma50_diff": np.zeros(1000),
            "ma_spread": np.zeros(1000),
            "ma_spread_slope": np.zeros(1000),
            "macd": np.zeros(1000),
            "macd_h": np.zeros(1000),
            "bb_width": np.full(1000, 0.01)
        }
        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.fixture
    def env(self, mock_df):
        feature_cols = [
            "rsi_14", "atr_14", "ma_20_slope", "ma_50_slope"
        ]
        return ForexTradingEnv(
            df=mock_df,
            window_size=TradingConfig.WINDOW_SIZE,
            sl_options=TradingConfig.SL_OPTIONS,
            tp_options=TradingConfig.TP_OPTIONS,
            spread_pips=TradingConfig.SPREAD_PIPS,
            commission_pips=TradingConfig.COMMISSION_PIPS,
            max_slippage_pips=TradingConfig.MAX_SLIPPAGE_PIPS,
            feature_columns=feature_cols,
            initial_equity_usd=TradingConfig.INITIAL_EQUITY_USD,
            lot_size=TradingConfig.LOT_SIZE_MICRO,
            random_start=False # Deterministic for testing
        )

    def test_initialization(self, env):
        assert env is not None
        assert env.window_size == TradingConfig.WINDOW_SIZE
        assert len(env.sl_options) == len(TradingConfig.SL_OPTIONS)

    def test_reset(self, env):
        obs, info = env.reset()
        
        # Check observation shape: (window_size, num_features + state_features)
        # num_features = 4 (mock) + 3 (state) = 7
        assert obs.shape == (TradingConfig.WINDOW_SIZE, 4 + 3)
        assert isinstance(info, dict)

    def test_step(self, env):
        env.reset()
        action = 0 # HOLD
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (TradingConfig.WINDOW_SIZE, 4 + 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_open_position(self, env):
        env.reset()
        # Find an OPEN action index (action >= 2)
        # 0: HOLD, 1: CLOSE, 2+: OPEN...
        action = 2 
        
        obs, reward, term, trunc, info = env.step(action)
        
        # Should be in position now (unless rejected for some reason, but typically successful)
        assert env.position != 0
        assert env.entry_price is not None

    def test_close_position(self, env):
        env.reset()
        # Open first
        env.step(2) 
        assert env.position != 0
        
        # Close
        action = 1 # CLOSE
        obs, reward, term, trunc, info = env.step(action)
        
        assert env.position == 0
        assert env.entry_price is None

if __name__ == "__main__":
    pytest.main()
