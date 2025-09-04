# 📊 Sentiment Analysis  

Proyek ini merupakan implementasi **Optimasi Analisis Sentimen Ulasan Mobile Fashion Retail App ** menggunakan pendekatan **Hybrid Model (Naïve Bayes + SVM)**.  
Pipeline mencakup preprocessing, pelatihan model, evaluasi, hingga pembuatan aplikasi interaktif berbasis **Streamlit** untuk uji analisis dan visualisasi data.  

---

## 📂 Project Structure
```bash
SentimenAnalysis/
├── app.py                  # Streamlit main app (menu & UI)
├── processing_data.py
├── skripsi.ipynb           # Jupyter Notebook (model building & experiments)
├── uji_analisis.py
├── uji_analisis_process.py
└── .gitignore
```
## 🚀 How to Run
1. Clone Repository
   ```bash
   git clone https://github.com/FikriAfifK/SentimenAnalysis.git
   cd SentimenAnalysis

2. Jalankan Streamlit App
   ```bash
   streamlit run app.py

## 🛠 Features
- Data Preprocessing: pembersihan teks, tokenisasi, stopword removal, stemming.
- Model Training: eksperimen dengan Naïve Bayes, SVM, serta Hybrid Model.
- Streamlit App:
  - Processing Data → menampilkan hasil preprocessing dataset.
  - Uji Analisis → input teks ulasan untuk diprediksi sentimennya.
  - Visualisasi Data → grafik distribusi sentimen & insight lainnya.
