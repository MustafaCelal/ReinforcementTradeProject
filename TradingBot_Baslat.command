#!/bin/bash
cd "$(dirname "$0")"

# 1. Sanal ortam kontrolü ve kurulumu
if [ ! -d "venv" ]; then
    echo "⚙️  İlk kurulum yapılıyor, lütfen bekleyin (Sanal ortam oluşturuluyor)..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "📦 Kütüphanelerin güncelliği kontrol ediliyor..."
pip install -r Requirements.txt --quiet
echo "✅ Kütüphaneler hazır!"

# 2. Uygulamayı başlat
echo "🚀 Dashboard başlatılıyor..."
streamlit run app.py --server.headless true
