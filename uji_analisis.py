import streamlit as st
import joblib
from uji_analisis_process import preprocess_text

def uji_analisis():
    # Load model dan vectorizer
    model = joblib.load("./assets/NEW_optimized_hybrid_model.pkl")
    count_vectorizer = joblib.load("./assets/count_vectorizer.pkl")
    tfidf_transformer = joblib.load("./assets/tfidf_transformer.pkl")

    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            margin-bottom: 20px;
        }
        .result-box {
            background-color: #262730;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'><h1>Uji Analisis Sentimen</h1></div>", unsafe_allow_html=True)

    # Input user
    text_input = st.text_area("Masukkan ulasan:", height=120)

    if st.button("Analyze"):
        if text_input.strip():
            cleaned_text = preprocess_text(text_input)

            # Vektorisasi
            text_counts = count_vectorizer.transform([cleaned_text])
            transformed_text = tfidf_transformer.transform(text_counts).toarray()

            # Prediksi
            prediction = model.predict(transformed_text)[0]
            probabilities = model.predict_proba(transformed_text)[0]

            label_mapping = {0: "Negatif", 1: "Netral", 2: "Positif"}
            label = label_mapping.get(prediction, "Tidak diketahui")

            prob_output = "".join([
                f"<p><strong>{label_mapping[i]}</strong>: {prob:.4f}</p>"
                for i, prob in enumerate(probabilities)
            ])

            st.markdown(f"""
                <div class='result-box'>
                    <h4>Hasil Prediksi: <span style='color:#00d4ff'>{label}</span></h4>
                    <h5>Probabilitas:</h5>
                    {prob_output}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Silakan masukkan teks sebelum menganalisis!")
