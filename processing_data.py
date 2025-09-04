import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Halaman Preprocessing Data
def processing():
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Processing Data</h1>
            <p>Halaman ini menampilkan hasil preprocessing data berdasarkan dataset ulasan mobile fashion retail app.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dropdown Pilihan Sub-Menu
    sub_menu = st.selectbox("Pilih Proses", ["Dataset", "Preprocessing", "Labeling", "TF-IDF", "SMOTE"], index=0)

    # Menampilkan Data Sesuai Sub-Menu yang Dipilih
    if sub_menu in ["Dataset", "Preprocessing", "Labeling", "SMOTE"]:
        file_map = {
            "Dataset": "./data/hasilEDA.csv",
            "Preprocessing": "./data/hasilprepro_new.csv",
            "Labeling": "./data/dataset_labeled (5).csv",
            "SMOTE": "./data/df_resampled.csv",
        }

        st.subheader(f"Hasil {sub_menu}")
        df = pd.read_csv(file_map[sub_menu])
        if sub_menu == "Preprocessing":
            df = df[["cleaning", "case_folding", "tokenize", "normalize", "negation", "stopword", "stemming"]]
            
        elif sub_menu == "Labeling":
            df = df[["ENtranslated", "compound_score", "sentimen"]]

        # Pagination
        rows_per_page = 50
        page = st.number_input("Halaman", min_value=1, max_value=(len(df) // rows_per_page) + 1, step=1)

        # Tampilkan data sesuai halaman
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        st.dataframe(df.iloc[start_idx:end_idx])
        if sub_menu == "Labeling":
            st.subheader("Distribusi Sentimen")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.countplot(x=df['sentimen'], palette="coolwarm", ax=ax1)
                ax1.set_xlabel("Sentimen")
                ax1.set_ylabel("Jumlah Ulasan")
                ax1.set_title("Distribusi Sentimen")
                st.pyplot(fig1)
                
                # Menghitung jumlah sentimen positif, netral, negatif, dan total
                jumlah_positif = df[df['sentimen'] == 'Positif'].shape[0]
                jumlah_netral = df[df['sentimen'] == 'Netral'].shape[0]
                jumlah_negatif = df[df['sentimen'] == 'Negatif'].shape[0]
                jumlah_total = df.shape[0]

                # Menampilkan hasil
                st.write(f"Jumlah sentimen positif  : {jumlah_positif}")
                st.write(f"Jumlah sentimen netral   : {jumlah_netral}")
                st.write(f"Jumlah sentimen negatif  : {jumlah_negatif}")
                st.write(f"Jumlah sentimen total    : {jumlah_total}")
        
        elif sub_menu == "SMOTE":
            st.subheader("Distribusi Kelas Sebelum & Sesudah SMOTE")
            df = pd.read_csv("./data/dataset_labeled (5).csv")
            y = df["sentimen"].values
            X = pd.read_csv("./data/hasil_tfidf_manual.csv")
            X = X.to_numpy()
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            y_train_series = pd.Series(y_train)
            y_train_resampled_series = pd.Series(y_train_resampled)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Sebelum SMOTE")
                st.write(y_train_series.value_counts())
            with col2:
                st.write("### Setelah SMOTE")
                st.write(y_train_resampled_series.value_counts())

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.barplot(x=y_train_series.value_counts().index, y=y_train_series.value_counts().values, palette="Blues", ax=axes[0])
            axes[0].set_title("Distribusi Kelas Sebelum SMOTE", fontsize=14, fontweight="bold")
            axes[0].set_xlabel("Kelas Sentimen", fontsize=12)
            axes[0].set_ylabel("Jumlah Sampel", fontsize=12)
            sns.barplot(x=y_train_resampled_series.value_counts().index, y=y_train_resampled_series.value_counts().values, palette="Greens", ax=axes[1])
            axes[1].set_title("Distribusi Kelas Setelah SMOTE", fontsize=14, fontweight="bold")
            axes[1].set_xlabel("Kelas Sentimen", fontsize=12)
            axes[1].set_ylabel("Jumlah Sampel", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)

    elif sub_menu == "TF-IDF":
        st.subheader("Hasil Konversi TF-IDF")
        
        file_choice = st.radio("Pilih hasil yang ingin ditampilkan:", ["TF", "IDF", "TF-IDF"])
        file_map = {
            "TF": "./data/hasil_tf_manual.csv",
            "IDF": "./data/hasil_idf_manual.csv",
            "TF-IDF": "./data/hasil_tfidf_manual.csv",
        }

        df = pd.read_csv(file_map[file_choice])
        
        # Pagination
        rows_per_page = 50
        page = st.number_input("Halaman", min_value=1, max_value=(len(df) // rows_per_page) + 1, step=1, key="tfidf_page")

        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        st.dataframe(df.iloc[start_idx:end_idx])
        st.subheader("Hasil Pembagian Data")
        col1, col2 = st.columns(2)
        with col1:
                df = pd.read_csv("./data/dataset_labeled (5).csv")
                y = df["sentimen"].values
                X = pd.read_csv("./data/hasil_tfidf_manual.csv")
                X = X.to_numpy()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Visualisasi distribusi data train & test
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(['Data Training', 'Data Testing'], [len(X_train), len(X_test)], color=['blue', 'orange'])

                # Tambahkan label jumlah di atas bar
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 50, f'{height}', ha='center', va='bottom', fontsize=12, fontweight='bold')

                ax.set_title('Distribusi Data Training dan Testing', fontsize=14, fontweight='bold')
                ax.set_xlabel('Jenis Data', fontsize=12)
                ax.set_ylabel('Jumlah Data', fontsize=12)
                ax.set_ylim(0, max(len(X_train), len(X_test)) * 1.2)
                st.pyplot(fig)
