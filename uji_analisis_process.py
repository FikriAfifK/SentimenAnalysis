import re
import pandas as pd
import emoji
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from deep_translator import GoogleTranslator

# Download tokenizer untuk nltk
nltk.download('punkt_tab')

# ========== CLEANING ==========
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    text = re.sub(r"<.*?>", '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\d", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ========== TRANSLATING ==========
def translate_to_indonesian(text):
    try:
        if isinstance(text, str) and text.strip() != "":  # Pastikan teks bukan NaN atau kosong
            translated_text = GoogleTranslator(source='auto', target='id').translate(text)
            return translated_text
        else:
            return text
    except Exception as e:
        return text

# ========== CASE FOLDING ==========
def case_folding(text):
    return text.lower() if isinstance(text, str) else text

# ========== TOKENIZING ==========
def tokenize(text):
    return word_tokenize(text) if isinstance(text, str) else []

# ========== NORMALIZATION ==========
try:
    data_kamus = pd.read_excel('./data/kamuskatabaku.xlsx')
except FileNotFoundError:
    data_kamus = pd.DataFrame({'tidak_baku': [], 'kata_baku': []})

def normalisasi(text, kamus):
    kalimat_final = []
    for kata in text:
        kata_benar = kamus[kamus['tidak_baku'] == kata]['kata_baku'].values
        kalimat_final.append(kata_benar[0] if len(kata_benar) > 0 else kata)
    return kalimat_final

# ========== NEGATION HANDLING ==========
df_antonym = pd.read_csv("./data/kamus_antonim_clear.csv")
negation_dict = dict(zip(df_antonym["Negasi"], df_antonym["Antonim"]))

def negation_handling(text_tokens):
    text = ' '.join(text_tokens)
    text = re.sub(r'\b(tidak|bukan|jangan|belum|tanpa|gak|enggak|tak|kurang)\s+(\w+)', r'tidak_\2', text)
    tokens = text.split()
    replaced_tokens = [negation_dict.get(token, token) for token in tokens]
    return replaced_tokens

# ========== STOPWORD REMOVAL ==========
try:
    stopwords_found_df = pd.read_csv('./data/stopwords_found.csv')
    stopwords_found = set(stopwords_found_df['stopword'].dropna().tolist())
except FileNotFoundError:
    stopwords_found = set()

stop_words_factory = StopWordRemoverFactory()
stop_words_sastrawi = set(stop_words_factory.get_stop_words())

all_stopwords = stop_words_sastrawi.union(stopwords_found)
stopword_remover = stop_words_factory.create_stop_word_remover()

def remove_stopwords(text):
    return stopword_remover.remove(' '.join(text)).split() if isinstance(text, list) else text

# ========== STEMMING ==========
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_text(text):
    if isinstance(text, list):
        text = ' '.join(text)
    return ' '.join([stemmer.stem(word) for word in text.split()])

# ========== FINAL PREPROCESSING PIPELINE ==========
def preprocess_text(text):
    text = clean_text(text)
    text = case_folding(text)
    text = tokenize(text)
    text = normalisasi(text, data_kamus)
    text = negation_handling(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text