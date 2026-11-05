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
import plotly.figure_factory as ff

# --- SHAP GLOBAL TANIMI VE KÜTÜPHANE KONTROLÜ ---
global SHAP_AVAILABLE
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

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==============================================================================
# STREAMLIT ARAYÜZ YAPILANDIRMASI VE MODERN CSS
# ==============================================================================
st.set_page_config(
    page_title="Kurumsal Otomatik ML & AI Platformu", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.3rem; color: #1E3A8A; font-weight: 800; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .card { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚡ Kurumsal Otomatik Veri Bilimi Platformu</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Modern interaktif grafikler, toplu skorlama, gelişmiş Voting ve SHAP açıklanabilirliği.</p>', unsafe_allow_html=True)

# ==============================================================================
# 1. ÖZEL ÖZELLİK MÜHENDİSLİĞİ
# ==============================================================================
class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_mappings=None):
        self.ordinal_mappings = ordinal_mappings or {}

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

def build_preprocessor(X_sample, ordinal_mappings):
    numeric_cols = X_sample.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_sample.select_dtypes(include=["object", "category"]).columns.tolist()
    
    binary_cols = [col for col in X_sample.columns if X_sample[col].nunique() == 2]
    pure_numeric_cols = [col for col in numeric_cols if col not in binary_cols]
    ordinal_cols = list(ordinal_mappings.keys())
    nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]

    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    
    ordinal_categories = [ordinal_mappings[c] for c in ordinal_cols if c in X_sample.columns]
    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    nominal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    binary_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ordinal', OrdinalEncoder())])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, pure_numeric_cols),
        ('ord', ordinal_transformer, [c for c in ordinal_cols if c in X_sample.columns]),
        ('nom', nominal_transformer, nominal_cols),
        ('bin', binary_transformer, binary_cols)
    ], remainder='drop')
    
    return preprocessor

# ==============================================================================
# 2. YAN MENÜ (SIDEBAR)
# ==============================================================================
st.sidebar.markdown("### ⚙️ Veri Yapılandırması")
uploaded_file = st.sidebar.file_uploader("Eğitim Veri Seti Yükle (.xlsx / .csv)", type=["xlsx", "csv"])

target_col = st.sidebar.text_input("Hedef Sütun (Target)", value="calisma_durumu")
drop_cols_input = st.sidebar.text_input("Silinecek ID Sütunları", value="sicil_no")
leakage_cols_input = st.sidebar.text_input("Sızıntı (Leakage) Sütunları", value="cikis_tarihi, cikis_nedeni, aktif_calisan_mi")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model ve Hiperparametreler")
use_lr = st.sidebar.checkbox("Logistic Regression", value=True)
lr_c = st.sidebar.slider("LR - Düzenlileştirme (C)", 0.01, 10.0, 1.0)

use_rf = st.sidebar.checkbox("Random Forest", value=True)
rf_n_estimators = st.sidebar.slider("RF - Ağaç Sayısı", 50, 300, 150)
rf_max_depth = st.sidebar.slider("RF - Maksimum Derinlik", 3, 25, 10)

use_gb = st.sidebar.checkbox("Gradient Boosting", value=True)
gb_lr = st.sidebar.slider("GB - Öğrenme Hızı", 0.01, 0.3, 0.1)

use_xgb = st.sidebar.checkbox("XGBoost", value=True if XGB_AVAILABLE else False, disabled=not XGB_AVAILABLE)

# ==============================================================================
# 3. ANA AKIŞ VE SEKMELER
# ==============================================================================
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    
    drop_list = [c.strip() for c in drop_cols_input.split(",") if c.strip()]
    leak_list = [c.strip() for c in leakage_cols_input.split(",") if c.strip()]
    all_drops = drop_list + leak_list
    df_clean = df_raw.drop(columns=[c for c in all_drops if c in df_raw.columns], errors='ignore')
    
    if target_col not in df_clean.columns:
        st.error(f"Hata: '{target_col}' sütunu veri setinde bulunamadı!")
    else:
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        ordinal_mappings = {
            "egitim_durumu": ['Lise', 'Ön Lisans', 'Lisans', 'Yüksek Lisans', 'Doktora'],
            "kademe": ['Uzman Yardımcısı', 'Uzman', 'Kıdemli Uzman', 'Yönetici', 'Direktör']
        }
        
        fe = HRFeatureEngineer(ordinal_mappings=ordinal_mappings)
        X_train_fe = fe.fit_transform(X_train)
        X_test_fe = fe.transform(X_test)
        preprocessor = build_preprocessor(X_train_fe, ordinal_mappings)

        # Sekmeli Modern Yapı
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📂 Modern EDA (Keşif)", 
            "⚙️ Model & Voting Eğitimi", 
            "📈 İstatistiksel Liderlik", 
            "🔍 SHAP Analizi",
            "🎯 Bireysel Simülasyon",
            "📁 Toplu Skorlama (Batch)"
        ])

        with tab1:
            st.subheader("Veri Seti Karakteristiği ve İnteraktif Dağılımlar")
            st.dataframe(df_raw.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Hedef Değişken Sınıf Oranları:**")
                val_counts = y.value_counts().reset_index()
                val_counts.columns = ['Sınıf', 'Adet']
                fig_pie = px.pie(val_counts, names='Sınıf', values='Adet', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.write("**Eksik Değer Analizi:**")
                missing_df = df_clean.isnull().sum().reset_index()
                missing_df.columns = ['Sütun', 'Eksik Sayısı']
                missing_df = missing_df[missing_df['Eks
