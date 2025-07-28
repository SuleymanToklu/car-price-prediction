import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import requests
from tqdm import tqdm

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
    # Model dosyasını doğrudan ana dizine kaydedeceğiz.
    LOCAL_MODEL_PATH = "price_prediction_pipeline.joblib"
    MODEL_URL = "https://github.com/SuleymanToklu/car-price-prediction/releases/download/v1.0/price_prediction_pipeline.joblib"

    # Eğer model dosyası lokalde yoksa, indir
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info("Eğitilmiş model dosyası bulunamadı. Buluttan indiriliyor...")
        
        try:
            # İlerleme çubuğu ile dosyayı indir
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status() 

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024 # 1 Kibibyte
            
            progress_bar = st.progress(0, text="İndirme işlemi başladı...")
            
            with open(LOCAL_MODEL_PATH, "wb") as file:
                downloaded_size = 0
                for data in response.iter_content(block_size):
                    downloaded_size += len(data)
                    file.write(data)
                    # İlerleme çubuğunu güncelle
                    progress = min(downloaded_size / total_size_in_bytes, 1.0) if total_size_in_bytes > 0 else 1.0
                    progress_bar.progress(progress, text=f"İndiriliyor... {int(progress * 100)}%")
            
            progress_bar.empty() # İlerleme çubuğunu kaldır
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
        vehicle_age = st.slider("Aracın Yaşı", min_value=0, max_value=50, value=10, help="Aracın model yılına göre yaşı.")
        odometer = st.slider("Kilometre (km)", min_value=0, max_value=400000, value=100000, step=1000, help="Aracın yaptığı toplam kilometre.")
        condition_options = ['new', 'like new', 'excellent', 'good', 'fair', 'salvage']
        condition = st.selectbox("Durumu", options=condition_options, index=2, help="Aracın genel kondisyonu.")
        fuel_options = ['gas', 'diesel', 'hybrid', 'other', 'electric']
        fuel = st.selectbox("Yakıt Türü", options=fuel_options, index=0)

    with col2:
        transmission_options = ['automatic', 'manual', 'other']
        transmission = st.selectbox("Vites Türü", options=transmission_options, index=0)
        drive_options = ['4wd', 'fwd', 'rwd']
        drive = st.selectbox("Çekiş", options=drive_options, index=0, help="Aracın çekiş tipi (4 çeker, önden çekiş, arkadan itiş).")
        state_options = ['al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy']
        state = st.selectbox("Eyalet", options=state_options, index=4, help="Aracın bulunduğu ABD eyaleti.")
        
    st.markdown("---")

    if st.button("Fiyatı Tahmin Et", type="primary", use_container_width=True):
        
        # --- HATA DÜZELTME: Bu bölüm sorunu çözmek için güncellendi ---
        try:
            # 1. Modelin eğitildiği sırada gördüğü tüm sütun isimlerini al
            expected_features = pipeline.feature_names_in_
        except AttributeError:
            st.error("Bu pipeline dosyası beklenen özellik listesini içermiyor. Lütfen scikit-learn 1.0+ ile eğitilmiş bir model kullanın.")
            st.stop()

        # 2. Kullanıcıdan gelen girdileri bir sözlükte topla
        user_input_data = {
            'odometer': odometer,
            'condition': condition,
            'fuel': fuel,
            'transmission': transmission,
            'drive': drive,
            'state': state,
            'vehicle_age': vehicle_age
        }
        
        # 3. Modelin beklediği tüm sütunları içeren boş bir DataFrame oluştur
        prediction_input_df = pd.DataFrame(columns=expected_features)
        prediction_input_df.loc[0] = np.nan # Tek bir satır oluştur ve NaN ile doldur

        # 4. Bu boş DataFrame'i kullanıcının girdiği verilerle doldur
        # Girilmeyen sütunlar (VIN gibi) NaN olarak kalacak ve pipeline'daki Imputer bunu halledecek.
        for col, value in user_input_data.items():
            if col in prediction_input_df.columns:
                prediction_input_df.at[0, col] = value
        
        st.subheader("Modele Gönderilen Ham Veri (Önizleme):")
        st.dataframe(prediction_input_df)

        try:
            # 5. Pipeline'ı kullanarak tahmini yap
            prediction = pipeline.predict(prediction_input_df)
            predicted_price = prediction[0]

            # Tahmini şık bir şekilde gösterelim
            st.subheader("AI Tahmin Sonucu")
            st.metric(label="Tahmini Piyasa Değeri", value=f"${predicted_price:,.2f}")
            st.success("Tahmin başarıyla tamamlandı!")
            st.info("Not: Bu tahmin, yaklaşık %66 doğruluk oranına sahip bir model tarafından yapılmıştır ve sadece bir referans değeridir.", icon="💡")

        except Exception as e:
            st.error(f"Tahmin sırasında bir hata oluştu: {e}")

else:
    st.warning("Uygulama başlatılamadı. Model dosyası yüklenemedi.")
