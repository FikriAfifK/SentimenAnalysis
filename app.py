import streamlit as st
from processing_data import processing
from uji_analisis import uji_analisis

# Konfigurasi tampilan
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header("Main Menu")
menu = st.sidebar.radio("Pilih Menu", ["Home", "Processing Data", "Uji Analisis"], index=0)

if menu == 'Home':
            st.session_state.page = "home"

elif menu == 'Processing Data':
            st.session_state.page = "processing"

elif menu == 'Uji Analisis':
            st.session_state.page = "uji_analisis"

# Halaman Utama
def home():
    st.markdown(
        """
        <div style='display: flex; justify-content: center; align-items: center; height: 60vh; text-align: center; flex-direction: column;'>
            <h1>Sistem Analisis Sentimen Ulasan</h1>
            <p>Selamat datang! Sistem ini digunakan untuk menganalisis sentimen ulasan terhadap aplikasi mobile fashion retail.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Routing halaman
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home()
elif st.session_state.page == "processing":
    processing()
elif st.session_state.page == "uji_analisis":
    uji_analisis()
