import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Sayfa Ayarları ve Başlık ---
st.set_page_config(page_title="Araç Fiyat Tahmini", page_icon="🚗", layout="wide")
st.title("🚗 AI Destekli Araç Fiyat Tahmin Uygulaması")
st.markdown("Bu uygulama, verdiğiniz özelliklere göre ikinci el bir aracın tahmini piyasa değerini hesaplar.")
st.markdown("---")


# --- Model ve Pipeline'ı Yükleme ---
# @st.cache_resource, bu fonksiyonun sonucunu önbelleğe alır.
# Bu sayede, büyük model dosyası her etkileşimde yeniden yüklenmez, uygulama hızlanır.
@st.cache_resource
def load_pipeline():
    """Loads the saved pipeline object from the file."""
    try:
        pipeline = joblib.load("saved_pipeline/price_prediction_pipeline.joblib")
        return pipeline
    except FileNotFoundError:
        st.error(
            "Kaydedilmiş pipeline dosyası 'saved_pipeline/price_prediction_pipeline.joblib' bulunamadı. Lütfen önce `main_training_pipeline.py` script'ini çalıştırın.")
        return None


pipeline = load_pipeline()

# --- Kullanıcı Girdi Alanları ---
if pipeline:
    st.header("Aracın Özelliklerini Giriniz")

    # Arayüzü daha düzenli hale getirmek için sütunlar oluşturalım
    col1, col2 = st.columns(2)

    with col1:
        # Sayısal Girdiler
        vehicle_age = st.slider("Aracın Yaşı", min_value=0, max_value=50, value=10,
                                help="Aracın model yılına göre yaşı.")
        odometer = st.slider("Kilometre (km)", min_value=0, max_value=400000, value=100000, step=1000,
                             help="Aracın yaptığı toplam kilometre.")

        # Kategorik Girdiler - 1
        condition_options = ['new', 'like new', 'excellent', 'good', 'fair', 'salvage']
        condition = st.selectbox("Durumu", options=condition_options, index=2, help="Aracın genel kondisyonu.")

        fuel_options = ['gas', 'diesel', 'hybrid', 'other', 'electric']
        fuel = st.selectbox("Yakıt Türü", options=fuel_options, index=0)

    with col2:
        # Kategorik Girdiler - 2
        transmission_options = ['automatic', 'manual', 'other']
        transmission = st.selectbox("Vites Türü", options=transmission_options, index=0)

        drive_options = ['4wd', 'fwd', 'rwd']
        drive = st.selectbox("Çekiş", options=drive_options, index=0,
                             help="Aracın çekiş tipi (4 çeker, önden çekiş, arkadan itiş).")

        state_options = [
            'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi',
            'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi',
            'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc',
            'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut',
            'vt', 'va', 'wa', 'wv', 'wi', 'wy'
        ]
        state = st.selectbox("Eyalet", options=state_options, index=4, help="Aracın bulunduğu ABD eyaleti.")

    st.markdown("---")

    # --- Tahmin Butonu ve Sonuç Gösterimi ---
    if st.button("Fiyatı Tahmin Et", type="primary", use_container_width=True):

        # HATA DÜZELTME:
        # Model, eğitim sırasında gördüğü tüm sütunları bekler.
        # Önce modelin beklediği tüm sütun isimlerini alalım.
        try:
            expected_features = pipeline.feature_names_in_
        except AttributeError:
            st.error(
                "Bu pipeline dosyası, beklenen özellik listesini içermiyor. Lütfen scikit-learn 1.0+ ile oluşturulmuş bir pipeline kullanın.")
            st.stop()  # Hata durumunda uygulamayı durdur

        # Kullanıcıdan gelen girdileri bir sözlükte toplayalım.
        input_data_from_user = {
            'odometer': odometer,
            'condition': condition,
            'fuel': fuel,
            'transmission': transmission,
            'drive': drive,
            'state': state,
            'vehicle_age': vehicle_age
        }

        # Modelin beklediği tüm sütunları içeren, ama içi boş (NaN) bir DataFrame oluşturalım.
        prediction_df = pd.DataFrame(columns=expected_features)
        prediction_df.loc[0] = np.nan

        # Şimdi bu boş DataFrame'i, kullanıcının girdiği verilerle dolduralım.
        # Kullanıcının girmediği sütunlar (VIN gibi) NaN olarak kalacak.
        # Pipeline'daki SimpleImputer bu NaN değerlerini otomatik olarak dolduracaktır.
        for col, value in input_data_from_user.items():
            if col in prediction_df.columns:
                prediction_df[col] = value

        st.subheader("Modele Gönderilen Ham Veri (Önizleme):")
        st.dataframe(prediction_df)

        # Pipeline'ı kullanarak tahmini yapalım
        prediction = pipeline.predict(prediction_df)
        predicted_price = prediction[0]

        # Tahmini şık bir şekilde gösterelim
        st.subheader("AI Tahmin Sonucu")
        st.metric(label="Tahmini Piyasa Değeri", value=f"${predicted_price:,.2f}")
        st.success("Tahmin başarıyla tamamlandı!")

        st.info(
            "Not: Bu tahmin, yaklaşık %66 doğruluk oranına sahip bir model tarafından yapılmıştır ve sadece bir referans değeridir.",
            icon="💡")

else:
    st.warning("Uygulama başlatılamadı. Model dosyası yüklenemedi.")
