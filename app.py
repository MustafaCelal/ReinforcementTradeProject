import streamlit as st
import os
import subprocess
import pandas as pd
from PIL import Image
import time
import json
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.ui.layout import show_performance_metrics, show_results_area, show_context_info
import config.settings as cfg

# --- UI Configuration ---
st.set_page_config(page_title="Reinforcement Trading Pro", layout="wide", page_icon="📈")

# Custom CSS for better look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4e73df;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Reinforcement Trading Pro Dashboard")
st.markdown("*Gelişmiş Takviyeli Öğrenme ve Finansal Analiz Platformu*")

# --- Sidebar Configuration ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2583/2583151.png", width=100)
st.sidebar.header("🕹️ Kontrol Paneli")

symbol = st.sidebar.selectbox(
    "Aktif Sembol", 
    options=cfg.AVAILABLE_SYMBOLS,
    format_func=lambda x: cfg.SYMBOL_DISPLAY_MAP.get(x, x),
    index=cfg.AVAILABLE_SYMBOLS.index(cfg.DEFAULT_SYMBOL)
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Araçlar")
if st.sidebar.button("🌐 TensorBoard Başlat/Aç"):
    # Tensorboard usually runs on 6006
    st.sidebar.success("TensorBoard linki: [http://localhost:6006](http://localhost:6006)")
    subprocess.Popen(["tensorboard", "--logdir", os.path.join(cfg.BASE_DIR, "tensorboard_log")])

# --- Main Tabs ---
tab_train, tab_test, tab_ab, tab_opt, tab_wf = st.tabs([
    "🏋️ Eğitim", "🧪 Backtest", "⚔️ A/B Test", "🎯 Optuna", "🔄 Walk-Forward"
])

# --- TAB 1: TRAINING ---
with tab_train:
    st.header("🧠 Model Eğitimi")
    col1, col2 = st.columns(2)
    
    with col1:
        algo_type = st.radio("Algoritma", ["PPO (Standart)", "RecurrentPPO (LSTM)"], help="LSTM hafıza tabanlı modeller zaman serilerinde daha başarılı olabilir.")
        steps = st.number_input("Eğitim Adımı", min_value=10000, max_value=5000000, value=600000, step=100000)
        
    with col2:
        interval = st.selectbox("Zaman Dilimi", ["15m", "30m", "1h", "4h", "1d"], index=2)
        period = st.selectbox("Geçmiş Veri", ["1y", "2y", "5y", "max"], index=0)
        log_detailed = st.checkbox("Detaylı Loglama (JSONL)", value=True, help="Tüm metrikleri ve kararları dosyaya kaydeder.")

    if st.button("🚀 Eğitimi Başlat"):
        script = "train_recurrent.py" if "LSTM" in algo_type else "train_agent.py"
        train_script = os.path.join(cfg.SCRIPTS_DIR, script)
        
        cmd = ["python", train_script, "--symbol", symbol, "--steps", str(steps), "--interval", interval, "--period", period, "--no-plot"]
        
        with st.status(f"🏃 {algo_type} Eğitiliyor...", expanded=True) as status:
            log_area = st.empty()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            output = ""
            for line in process.stdout:
                output += line
                log_area.code(output[-2000:], language="bash") # Show last 2000 chars
            
            process.wait()
            if process.returncode == 0:
                status.update(label="✅ Eğitim Tamamlandı!", state="complete")
                st.balloons()
            else:
                status.update(label="❌ Eğitim Hata ile Kesildi!", state="error")

# --- TAB 2: BACKTEST ---
with tab_test:
    st.header("🧪 Strateji Testi")
    
    # List model files
    model_files = [f for f in os.listdir(cfg.MODELS_DIR) if f.endswith(".zip")]
    selected_model = st.selectbox("Test Edilecek Model", options=model_files)
    test_period = st.selectbox("Test Periyodu", ["1mo", "3mo", "6mo", "1y", "max"], index=1)
    
    if st.button("🔍 Backtest Çalıştır"):
        test_script = os.path.join(cfg.SCRIPTS_DIR, "test_agent.py")
        model_path = os.path.join(cfg.MODELS_DIR, selected_model)
        cmd = ["python", test_script, "--symbol", symbol, "--period", test_period, "--model-path", model_path, "--no-plot"]
        
        with st.spinner("Backtest yapılıyor..."):
            subprocess.run(cmd)
            st.rerun()

    # Results area (from layout.py)
    if os.path.exists(cfg.TRADE_HISTORY_FILE):
        df_trades = pd.read_csv(cfg.TRADE_HISTORY_FILE)
        show_performance_metrics(df_trades)
        show_results_area(symbol)

# --- TAB 3: A/B TEST ---
with tab_ab:
    st.header("⚔️ A/B Model Karşılaştırma")
    st.info("İki farklı modeli aynı veri seti üzerinde paralel olarak yarıştırın.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        model_a = st.selectbox("Model A (Kontrol)", options=model_files, key="ab_a")
    with col_b:
        model_b = st.selectbox("Model B (Varyant)", options=model_files, key="ab_b")
        
    ab_period = st.selectbox("Karşılaştırma Periyodu", ["1mo", "3mo", "6mo", "1y"], index=1)
    
    if st.button("⚔️ Düelloyu Başlat"):
        path_a = os.path.join(cfg.MODELS_DIR, model_a)
        path_b = os.path.join(cfg.MODELS_DIR, model_b)
        ab_script = os.path.join(cfg.SCRIPTS_DIR, "ab_test.py")
        
        cmd = ["python", ab_script, "--model-a", path_a, "--model-b", path_b, "--symbol", symbol, "--period", ab_period, "--no-plot"]
        
        with st.spinner("Modeller yarıştırılıyor..."):
            subprocess.run(cmd)
            
            # Show result image if exists
            res_img = os.path.join(cfg.OUTPUTS_DIR, "ab_test_result.png")
            if os.path.exists(res_img):
                st.image(Image.open(res_img))
                # Winner usually printed in results or we can parse from logs if we add it
                st.success("A/B Test tamamlandı. Yukarıdaki grafikte equity karşılaştırmasını görebilirsiniz.")

# --- TAB 4: OPTUNA ---
with tab_opt:
    st.header("🎯 Hiperparametre Optimizasyonu")
    st.markdown("Optuna kullanarak en kârlı PPO parametrelerini otomatik bulun.")
    
    trials = st.slider("Deneme Sayısı (Trials)", 5, 50, 10)
    opt_timesteps = st.number_input("Her Deneme İçin Adım", value=50000, step=10000)
    
    if st.button("🎯 Optimizasyonu Başlat"):
        opt_script = os.path.join(cfg.SCRIPTS_DIR, "optimize_hyperparams.py")
        cmd = ["python", opt_script, "--symbol", symbol, "--trials", str(trials), "--timesteps", str(opt_timesteps)]
        
        with st.status("🎯 Parametreler Optimize Ediliyor...", expanded=True) as status:
            log_area = st.empty()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            output = ""
            for line in process.stdout:
                output += line
                log_area.code(output[-2000:])
            process.wait()
            
            if process.returncode == 0:
                # Load results
                res_path = os.path.join(cfg.OUTPUTS_DIR, "best_hyperparams.json")
                if os.path.exists(res_path):
                    with open(res_path, 'r') as f:
                        best_params = json.load(f)
                    st.success("Optimizasyon Tamamlandı!")
                    st.json(best_params)
            else:
                st.error("Optimizasyon başarısız oldu.")

# --- TAB 5: WALK-FORWARD ---
with tab_wf:
    st.header("🔄 Walk-Forward Validation")
    st.markdown("Rolling window yöntemiyle modelin gelecekteki veriye karşı tutarlılığını ölçün.")
    
    n_windows = st.slider("Pencere Sayısı (Windows)", 2, 6, 4)
    wf_steps = st.number_input("Pencere Başına Eğitim", value=100000, step=50000)
    
    if st.button("🔄 Validasyonu Çalıştır"):
        wf_script = os.path.join(cfg.SCRIPTS_DIR, "walk_forward.py")
        cmd = ["python", wf_script, "--symbol", symbol, "--windows", str(n_windows), "--timesteps", str(wf_steps), "--no-plot"]
        
        with st.status("🔄 Rolling Validation Yapılıyor...", expanded=True) as status:
            log_area = st.empty()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            output = ""
            for line in process.stdout:
                output += line
                log_area.code(output[-2000:])
            process.wait()
            
            # Show results if saved
            wf_res = os.path.join(cfg.OUTPUTS_DIR, "walk_forward_results.csv")
            if os.path.exists(wf_res):
                st.dataframe(pd.read_csv(wf_res))

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info(f"Sistem Zamanı: {time.strftime('%H:%M:%S')}")
