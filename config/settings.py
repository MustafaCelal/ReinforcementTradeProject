import os

# --- Base Directory (Root of the project) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Trading Settings ---
SYMBOL_DISPLAY_MAP = {
    "EURUSD=X": "EUR/USD (Euro/Dolar)",
    "CHF=X": "CHF/USD (İsviçre Frangı/Dolar)",
    "EURCHF=X":"EUR/CHF (Euro/İsviçre Frangı)",
    "GBPUSD=X": "GBP/USD (Sterlin/Dolar)",
    "BTC-USD": "Kripto: Bitcoin",
    "ETH-USD": "Kripto: Ethereum",
    "GC=F": "Altın (Gold)",
    "SI=F": "Gümüş (Silver)",
    "CL=F": "Petrol (Crude Oil)"
    # "^GSPC": "Endeks: S&P 500",
    # "AAPL": "Hisse: Apple",
    # "TSLA": "Hisse: Tesla"
}
AVAILABLE_SYMBOLS = list(SYMBOL_DISPLAY_MAP.keys())
DEFAULT_SYMBOL = "EURUSD=X"

# --- Hyperparameters ---
DEFAULT_TOTAL_TIMESTEPS = 600000
WINDOW_SIZE = 30
SL_OPTIONS = [1.5, 2.0, 3.0]  # ATR Multipliers for dynamic SL
TP_OPTIONS = [2.0, 3.0, 4.5, 6.0]  # ATR Multipliers for dynamic TP (keeping R:R >= 1.5)

# --- Environment Settings ---
SPREAD_PIPS = 1.0
COMMISSION_PIPS = 0.0
MAX_SLIPPAGE_PIPS = 0.2
INITIAL_EQUITY_USD = 100.0
LOT_SIZE_MICRO = 1000.0  # 0.01 Lot

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
