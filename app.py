import os
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.express as px

# --- KÜTÜPHANE KONTROLÜ ---
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "xgboost", "streamlit", "plotly", "shap"])
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==============================================================================
# SAYFA YAPILANDIRMASI VE OKUNABİLİRLİK (CSS)
# ==============================================================================
st.set_page_config(
    page_title="10 Bölümlü Modüler Veri Bilimi Platformu", 
    page_icon="🔬", 
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        color: #1F2937;
    }
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 800; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .section-box { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; margin-bottom: 15px; color: #1F2937; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔬 10 Bölümlü Kontrollü Veri Bilimi ve İstatistiksel Denetim Platformu</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Her aşamada tam kontrol sağlayın, modelleri özelleştirin ve sonuçların istatistiksel güvenilirliğini test edin.</p>', unsafe_allow_html=True)

# ==============================================================================
# ÖZEL ÖZELLİK MÜHENDİSLİĞİ (Scikit-Learn Uyumlu Transformer)
# ==============================================================================
class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_mappings=None):
        # Scikit-learn kuralı: Parametre üzerinde constructor içinde modifikasyon yapılmaz.
        self.ordinal_mappings = ordinal_mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        ref_date = pd.to_datetime('today')
        
        if 'ise_giris_tarihi' in X_out.columns:
            X_out['ise_giris_tarihi'] = pd.to_datetime(X_out['ise_giris_tarihi'], errors='coerce')
            X_out['kidem_yili'] = (ref_date - X_out['ise_giris_tarihi']).dt.days / 365.25
            X_out.drop(columns=['ise_giris_tarihi'], inplace=True)
            
        if 'dogum_tarihi' in X_out.columns:
            X_out['dogum_tarihi'] = pd.to_datetime(X_out['dogum_tarihi'], errors='coerce')
            X_out['yas'] = (ref_date - X_out['dogum_tarihi']).dt.days / 365.25
            X_out.drop(columns=['dogum_tarihi'], inplace=True)

        if 'maas' in X_out.columns and 'kidem_yili' in X_out.columns:
            X_out['maas_kidem_orani'] = X_out['maas'] / (X_out['kidem_yili'] + 0.1)

        return X_out

@st.cache_data
def load_data(file):
    if file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        return pd.read_csv(file)

# ==============================================================================
# BÖLÜM SEÇİCİ (SIDEBAR)
# ==============================================================================
st.sidebar.markdown("## 🧭 10 Aşamalı Proje Rehberi")
current_section = st.sidebar.radio(
    "Gitmek İstediğiniz Bölümü Seçin:",
    [
        "Bölüm 1: Kurulum ve Proje Ayarları",
        "Bölüm 2: Veriyi Yükle ve Denetle",
        "Bölüm 3: Hedef ve Özellik Hazırlama",
        "Bölüm 4: Train / Test Bölme",
        "Bölüm 5: Ön İşleme Pipeline'ı",
        "Bölüm 6: Sızıntı Avcısı",
        "Bölüm 7: Model Karşılaştırma (CV)",
        "Bölüm 8: Hiperparametre Ayarı",
        "Bölüm 9: Test Değerlendirmesi & Voting",
        "Bölüm 10: İstatistiksel Güvenilirlik & Yorumlama"
    ]
)

# ==============================================================================
# BÖLÜM 1: Kurulum ve Proje Ayarları
# ==============================================================================
if current_section == "Bölüm 1: Kurulum ve Proje Ayarları":
    st.header("Bölüm 1: Kurulum ve Proje Ayarları")
    st.markdown("""
    <div class="section-box">
    <h3>Proje Yapılandırma Bilgilendirmesi</h3>
    <p>Bu bölümde proje için gerekli olan Python kütüphaneleri kontrol edilir. 
    Sonraki adımlarda veri setinizi yükleyerek uçtan uca makine öğrenmesi pipeline'ını yönetebilirsiniz.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("💡 Sol menüden **Bölüm 2: Veriyi Yükle ve Denetle** adımına geçebilirsiniz.")

# ==============================================================================
# BÖLÜM 2: Veriyi Yükle ve Denetle
# ==============================================================================
elif current_section == "Bölüm 2: Veriyi Yükle ve Denetle":
    st.header("Bölüm 2: Veriyi Yükle ve Denetle")
    uploaded_file = st.file_uploader("Veri Setinizi Yükleyin (.xlsx veya .csv)", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        df_raw = load_data(uploaded_file)
        st.session_state['df_raw'] = df_raw
        st.success("Veri başarıyla yüklendi!")
        st.dataframe(df_raw.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Satır Sayısı", df_raw.shape[0])
        with col2:
            st.metric("Toplam Sütun Sayısı", df_raw.shape[1])
            
        st.subheader("Eksik Değer ve Tip Kontrolü")
        info_df = pd.DataFrame({
            "Veri Tipi": df_raw.dtypes.astype(str),
            "Eksik Değer Sayısı": df_raw.isnull().sum(),
            "Eksik Oranı (%)": (df_raw.isnull().mean() * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)

# ==============================================================================
# BÖLÜM 3: Hedef ve Özellik Hazırlama
# ==============================================================================
elif current_section == "Bölüm 3: Hedef ve Özellik Hazırlama":
    st.header("Bölüm 3: Hedef ve Özellik Hazırlama")
    if 'df_raw' in st
