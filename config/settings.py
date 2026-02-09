import os

# --- Base Directory (Root of the project) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Trading Settings ---
SYMBOL_DISPLAY_MAP = {
    "EURUSD=X": "EUR/USD (Euro/Dolar)",
    "CHF=X": "CHF/USD (İsviçre Frangı/Dolar)",
    "EURCHF=X": "EUR/CHF (Euro/İsviçre Frangı)",
    "GBPUSD=X": "GBP/USD (Sterlin/Dolar)",
    "BTC-USD": "Kripto: Bitcoin",
    "ETH-USD": "Kripto: Ethereum",
    "GC=F": "Altın (Gold)",
    "SI=F": "Gümüş (Silver)",
    "CL=F": "Petrol (Crude Oil)"
}
AVAILABLE_SYMBOLS = list(SYMBOL_DISPLAY_MAP.keys())
DEFAULT_SYMBOL = "EURUSD=X"

# --- Environment & Trading Hyperparameters ---
class TradingConfig:
    # General
    WINDOW_SIZE = 30
    STRATEGY_INTERVAL = "1h"
    STRATEGY_PERIOD = "max"
    
    # Risk Management (ATR Multipliers)
    SL_OPTIONS = [1.5, 2.0, 3.0]
    TP_OPTIONS = [2.0, 3.0, 4.5, 6.0]
    
    # Environment Friction
    SPREAD_PIPS = 1.0
    COMMISSION_PIPS = 0.0
    MAX_SLIPPAGE_PIPS = 0.2
    
    # Account
    INITIAL_EQUITY_USD = 100.0  # Default check
    LOT_SIZE_MICRO = 1000.0     # 0.01 Lot
    
    # Reward Shaping Weights
    HOLD_REWARD_WEIGHT = 0.02
    OPEN_PENALTY_PIPS = 1.5
    TIME_PENALTY_PIPS = 0.01
    UNREALIZED_DELTA_WEIGHT = 0.02
    SHARPE_REWARD_WEIGHT = 0.1
    
    # PPO/Training Hyperparameters
    TOTAL_TIMESTEPS = 600_000
    LEARNING_RATE = 0.0001
    ENT_COEF = 0.02
    BATCH_SIZE = 128
    N_EPOCHS = 10
    N_STEPS = 2048

# Backwards compatibility for existing imports (if any)
WINDOW_SIZE = TradingConfig.WINDOW_SIZE
SL_OPTIONS = TradingConfig.SL_OPTIONS
TP_OPTIONS = TradingConfig.TP_OPTIONS
SPREAD_PIPS = TradingConfig.SPREAD_PIPS
COMMISSION_PIPS = TradingConfig.COMMISSION_PIPS
MAX_SLIPPAGE_PIPS = TradingConfig.MAX_SLIPPAGE_PIPS
INITIAL_EQUITY_USD = TradingConfig.INITIAL_EQUITY_USD
LOT_SIZE_MICRO = TradingConfig.LOT_SIZE_MICRO
DEFAULT_TOTAL_TIMESTEPS = TradingConfig.TOTAL_TIMESTEPS


# --- File Paths (Fixed to BASE_DIR) ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

MODEL_PREFIX = "model_"
TRADE_HISTORY_FILE = os.path.join(OUTPUTS_DIR, "trade_history_output.csv")
TRADING_CHART_FILE = os.path.join(OUTPUTS_DIR, "trading_chart.png")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
