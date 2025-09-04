# 📊 Sentiment Analysis  

This project is an implementation of **Optimization of Sentiment Analysis on Mobile Fashion Retail App Reviews** using a **Hybrid Model (Naïve Bayes + SVM)** approach.
The pipeline includes preprocessing, model training, evaluation, and the creation of a **Streamlit**-based interactive application for data analysis and visualization testing.

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

2. Run Streamlit App
   ```bash
   streamlit run app.py

## 🛠 Features
- Data Preprocessing: text cleaning, tokenization, stopword removal, stemming.
- Model Training: experiments with Naïve Bayes, SVM, and Hybrid Model.
- Streamlit App:
  - Data Processing → displays the results of dataset preprocessing.
  - Analysis Test → input review text to predict its sentiment.
  - Data Visualization → sentiment distribution graph & other insights.
