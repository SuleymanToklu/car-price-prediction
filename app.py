import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import requests  # Yeni kütüphane
from tqdm import tqdm  # İlerleme çubuğu için

# --- Sayfa Ayarları ve Başlık ---
st.set_page_config(page_title="Araç Fiyat Tahmini", page_icon="🚗", layout="wide")
st.title("🚗 AI Destekli Araç Fiyat Tahmin Uygulaması")
st.markdown("Bu uygulama, verdiğiniz özelliklere göre ikinci el bir aracın tahmini piyasa değerini hesaplar.")
st.markdown("---")


# --- Model ve Pipeline'ı Yükleme (İndirme Mantığı ile) ---
@st.cache_resource
def download_and_load_pipeline():
    """
    Checks for the pipeline file. If not found, downloads it from a public URL.
    Then loads and returns the pipeline.
    """
    MODEL_DIR = "saved_pipeline"
    MODEL_FILE = "saved_pipeline/price_prediction_pipeline.joblib"
    LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    MODEL_URL = "https://github.com/SuleymanToklu/car-price-prediction/releases/download/v1.0/price_prediction_pipeline.joblib"

    # Gerekli klasörün var olduğundan emin ol
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Eğer model dosyası lokalde yoksa, indir
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info("Eğitilmiş model dosyası bulunamadı. Buluttan indiriliyor...")

        try:
            # İlerleme çubuğu ile dosyayı indir
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            progress_bar = st.progress(0, text="İndirme işlemi başladı...")

            with open(LOCAL_MODEL_PATH, "wb") as file:
                downloaded_size = 0
                for data in response.iter_content(block_size):
                    downloaded_size += len(data)
                    file.write(data)
                    # İlerleme çubuğunu güncelle
                    progress = min(downloaded_size / total_size_in_bytes, 1.0)
                    progress_bar.progress(progress, text=f"İndiriliyor... {int(progress * 100)}%")

            progress_bar.empty()  # İlerleme çubuğunu kaldır
            st.success("Model başarıyla indirildi!")

        except requests.exceptions.RequestException as e:
            st.error(f"Model indirilemedi: {e}")
            st.error(f"Lütfen URL'nin doğru olduğundan emin olun: {MODEL_URL}")
            return None

    # Modeli yükle
    try:
        pipeline = joblib.load(LOCAL_MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu: {e}")
        return None


pipeline = download_and_load_pipeline()

# --- Kullanıcı Girdi Alanları ---
if pipeline:
    st.header("Aracın Özelliklerini Giriniz")

    col1, col2 = st.columns(2)

    with col1:
        vehicle_age = st.slider("Aracın Yaşı", min_value=0, max_value=50, value=10,
                                help="Aracın model yılına göre yaşı.")
        odometer = st.slider("Kilometre (km)", min_value=0, max_value=400000, value=100000, step=1000,
                             help="Aracın yaptığı toplam kilometre.")
        condition_options = ['new', 'like new', 'excellent', 'good', 'fair', 'salvage']
        condition = st.selectbox("Durumu", options=condition_options, index=2, help="Aracın genel kondisyonu.")
        fuel_options = ['gas', 'diesel', 'hybrid', 'other', 'electric']
        fuel = st.selectbox("Yakıt Türü", options=fuel_options, index=0)

    with col2:
        transmission_options = ['automatic', 'manual', 'other']
        transmission = st.selectbox("Vites Türü", options=transmission_options, index=0)
        drive_options = ['4wd', 'fwd', 'rwd']
        drive = st.selectbox("Çekiş", options=drive_options, index=0,
                             help="Aracın çekiş tipi (4 çeker, önden çekiş, arkadan itiş).")
        state_options = ['al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks',
                         'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny',
                         'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv',
                         'wi', 'wy']
        state = st.selectbox("Eyalet", options=state_options, index=4, help="Aracın bulunduğu ABD eyaleti.")

    st.markdown("---")

    if st.button("Fiyatı Tahmin Et", type="primary", use_container_width=True):
        try:
            expected_features = pipeline.named_steps['preprocessor'].get_feature_names_out()
            # OneHotEncoder sonrası oluşan sütun isimlerini temizle (sadece ana isimleri al)
            original_features = set(feat.split('_')[0] for feat in expected_features)
        except Exception:
            st.error("Modelin özellik isimleri alınamadı. Lütfen pipeline yapısını kontrol edin.")
            st.stop()

        input_data_from_user = {'odometer': odometer, 'condition': condition, 'fuel': fuel,
                                'transmission': transmission, 'drive': drive, 'state': state,
                                'vehicle_age': vehicle_age}

        # Sadece modelin beklediği orijinal sütunları kullanarak DataFrame oluştur
        prediction_df = pd.DataFrame([input_data_from_user])

        st.subheader("Modele Gönderilen Ham Veri:")
        st.dataframe(prediction_df)

        prediction = pipeline.predict(prediction_df)
        predicted_price = prediction[0]

        st.subheader("AI Tahmin Sonucu")
        st.metric(label="Tahmini Piyasa Değeri", value=f"${predicted_price:,.2f}")
        st.success("Tahmin başarıyla tamamlandı!")
        st.info(
            "Not: Bu tahmin, yaklaşık %66 doğruluk oranına sahip bir model tarafından yapılmıştır ve sadece bir referans değeridir.",
            icon="💡")

else:
    st.warning("Uygulama başlatılamadı. Model dosyası yüklenemedi.")