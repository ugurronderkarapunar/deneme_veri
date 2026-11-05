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
st.markdown('<p class="sub-header">Önce veri setinizi yükleyin, sütun adlarını, tiplerini ve eksik değerleri yönetin, ardından analiz sürecine geçin.</p>', unsafe_allow_html=True)

# ==============================================================================
# ÖZEL ÖZELLİK MÜHENDİSLİĞİ (Scikit-Learn Uyumlu Transformer)
# ==============================================================================
class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_mappings=None):
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
# BÖLÜM SEÇİCİ (SIDEBAR) - VERİ YÜKLENMEDEN DİĞERLERİ KİLİTLİ
# ==============================================================================
st.sidebar.markdown("## 🧭 10 Aşamalı Proje Rehberi")

data_loaded = 'df_raw' in st.session_state

if not data_loaded:
    st.sidebar.warning("⚠️ Devam etmek için önce **Bölüm 2'den** veri setinizi yüklemelisiniz.")
    available_sections = [
        "Bölüm 2: Veriyi Yükle ve Denetle (Zorunlu İlk Adım)",
        "Bölüm 1: Kurulum ve Proje Ayarları"
    ]
else:
    available_sections = [
        "Bölüm 2: Veriyi Yükle ve Denetle",
        "Bölüm 1: Kurulum ve Proje Ayarları",
        "Bölüm 3: Hedef ve Özellik Hazırlama",
        "Bölüm 4: Train / Test Bölme",
        "Bölüm 5: Ön İşleme Pipeline'ı",
        "Bölüm 6: Sızıntı Avcısı",
        "Bölüm 7: Model Karşılaştırma (CV)",
        "Bölüm 8: Hiperparametre Ayarı",
        "Bölüm 9: Test Değerlendirmesi & Voting",
        "Bölüm 10: İstatistiksel Güvenilirlik & Yorumlama"
    ]

current_section = st.sidebar.radio("Gitmek İstediğiniz Bölümü Seçin:", available_sections)

if not data_loaded and "Bölüm 2" not in current_section:
    current_section = "Bölüm 2: Veriyi Yükle ve Denetle (Zorunlu İlk Adım)"

# ==============================================================================
# BÖLÜM 2: Veriyi Yükle, Denetle ve Manipüle Et (Gelişmiş Stüdyo)
# ==============================================================================
if "Bölüm 2: Veriyi Yükle" in current_section:
    st.header("📂 Bölüm 2: Gelişmiş Veri Yükleme ve Temizlik Stüdyosu")
    st.markdown("""
    <div class="section-box">
    <h3>Veri Sağlığı, Sütun Yönetimi ve Temizlik Paneli</h3>
    <p>Veri setinizi yükleyin; mükerrer kayıtları temizleyin, sabit sütunları ayıklayın, sütun adlarını ve tiplerini profesyonelce yönetin.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Veri Setinizi Yükleyin (.xlsx / .csv)", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        if 'df_raw' not in st.session_state or st.sidebar.button("🔄 Veriyi Sıfırla / Yeniden Yükle"):
            st.session_state['df_raw'] = load_data(uploaded_file)
            st.success("Veri başarıyla yüklendi ve hafızaya alındı!")
            
        df_working = st.session_state['df_raw'].copy()
        
        # --- HIZLI TEMİZLİK ARAÇLARI ÇUĞU ---
        st.subheader("⚡ Hızlı Veri Hijyeni İşlemleri")
        col_h1, col_h2, col_h3 = st.columns(3)
        
        with col_h1:
            dup_count = df_working.duplicated().sum()
            if st.button(f"🧹 Mükerrer Satırları Temizle ({dup_count} adet)"):
                if dup_count > 0:
                    df_working = df_working.drop_duplicates().reset_index(drop=True)
                    st.session_state['df_raw'] = df_working
                    st.success(f"{dup_count} adet yinelenen satır silindi!")
                    st.rerun()
                else:
                    st.info("Veri setinde yinelenen satır yok.")
                    
        with col_h2:
            constant_cols = [c for c in df_working.columns if df_working[c].nunique() <= 1]
            if st.button(f"🗑️ Sabit/Boş Sütunları At ({len(constant_cols)} adet)"):
                if constant_cols:
                    df_working = df_working.drop(columns=constant_cols)
                    st.session_state['df_raw'] = df_working
                    st.success(f"Sabit sütunlar atıldı: {constant_cols}")
                    st.rerun()
                else:
                    st.info("Sabit değerli sütun bulunamadı.")
                    
        with col_h3:
            csv_export = df_working.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Temiz Veriyi İndir (CSV)", data=csv_export, file_name="temizlenmis_veri.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("📊 Ham Veri Önizlemesi ve Boyutları")
        st.dataframe(df_working.head(), use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Toplam Satır (Gözlem)", df_working.shape[0])
        with col_m2:
            st.metric("Toplam Sütun (Değişken)", df_working.shape[1])
            
        st.markdown("---")
