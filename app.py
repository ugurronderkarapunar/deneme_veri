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
        st.session_state['df_raw'] = load_data(uploaded_file)
        st.success("Veri başarıyla yüklendi ve hafızaya alındı!")
        
    if 'df_raw' in st.session_state:
        df_working = st.session_state['df_raw'].copy()
        
        if st.sidebar.button("🔄 Veriyi Sıfırla / Yeniden Yükle"):
            del st.session_state['df_raw']
            if 'df_clean' in st.session_state:
                del st.session_state['df_clean']
            if 'X_train' in st.session_state:
                del st.session_state['X_train']
            st.rerun()
            
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
        st.subheader("🛠️ İnteraktif Veri ve Sütun Yönetim Paneli")
        
        # 1. RENAME STUDIO (Sütun Adı Düzenleyici)
        with st.expander("✍️ Sütun Adı Düzenleyici (Rename Studio)", expanded=False):
            st.info("💡 Sütun adlarındaki boşlukları, Türkçe karakterleri otomatik temizleyebilir veya her sütunu tek tek yeniden adlandırabilirsiniz.")
            
            if st.button("🧹 Tüm Sütun Adlarını Otomatik Temizle (Boşlukları al, küçük harfe çevir)"):
                new_cols = {}
                tr_map = {'ç':'c', 'ğ':'g', 'ı':'i', 'ö':'o', 'ş':'s', 'ü':'u', 'İ':'i', 'Ç':'c', 'Ğ':'g', 'Ö':'o', 'Ş':'s', 'Ü':'u'}
                for col in df_working.columns:
                    cleaned = col.strip().lower().replace(" ", "_")
                    for k, v in tr_map.items():
                        cleaned = cleaned.replace(k, v)
                    new_cols[col] = cleaned
                df_working = df_working.rename(columns=new_cols)
                st.session_state['df_raw'] = df_working
                st.success("✨ Sütun adları otomatik olarak temizlendi!")
                st.rerun()
                
            st.write("---")
            st.write("**Manuel Sütun Adı Güncelleme:**")
            new_column_names = {}
            for col in df_working.columns:
                new_name = st.text_input(f"Eski: `{col}`", value=col, key=f"rename_{col}")
                new_column_names[col] = new_name
                
            if st.button("💾 Sütun Adı Değişikliklerini Kaydet", type="primary"):
                if len(set(new_column_names.values())) < len(new_column_names.values()):
                    st.error("⚠️ Hata: Sütun adları benzersiz olmalıdır! Aynı adı birden fazla sütuna veremezsiniz.")
                else:
                    df_working = df_working.rename(columns=new_column_names)
                    st.session_state['df_raw'] = df_working
                    st.success("✨ Sütun adları başarıyla güncellendi!")
                    st.rerun()

        # 2. DTYPE CONVERTER (Veri Tipi Düzenleyici)
        with st.expander("⚙️ Sütun Veri Tiplerini Güncelle", expanded=False):
            st.info("💡 Sütunların veri tiplerini (`int64`, `float64`, `object`, `category`) ihtiyacınıza göre dönüştürebilirsiniz.")
            
            edited_cols_data = []
            for col in df_working.columns:
                current_type = str(df_working[col].dtype)
                col_type_target = st.selectbox(
                    f"'{col}' sütunu veri tipi:", 
                    ["Mevcut (" + current_type + ")", "int64 (Sayısal)", "float64 (Ondalıklı)", "object (Metin)", "category (Kategorik)"],
                    key=f"dtype_{col}"
                )
                edited_cols_data.append((col, col_type_target))
                
            if st.button("🔧 Veri Tiplerini Güncelle", type="primary"):
                for col, target in edited_cols_data:
                    try:
                        if "int64" in target:
                            df_working[col] = pd.to_numeric(df_working[col], errors='coerce').fillna(0).astype('int64')
                        elif "float64" in target:
                            df_working[col] = pd.to_numeric(df_working[col], errors='coerce').astype('float64')
                        elif "object" in target:
                            df_working[col] = df_working[col].astype(str)
                        elif "category" in target:
                            df_working[col] = df_working[col].astype('category')
                    except Exception as e:
                        st.warning(f"'{col}' sütunu dönüştürülemedi: {e}")
                
                st.session_state['df_raw'] = df_working
                st.success("✨ Veri tipleri başarıyla güncellendi!")
                st.rerun()

        # 3. MISSING VALUE HANDLER (Eksik Değer Müdahale Aracı)
        with st.expander("🧹 Eksik Değer (NaN) Temizleme Stüdyosu", expanded=False):
            st.info("💡 Veri setindeki eksik hücrelere anlık müdahale edebilirsiniz.")
            impute_strategy = st.selectbox(
                "Eksik Değer Doldurma / Temizleme Stratejisi",
                ["Hiçbiri (Olduğu Gibi Bırak)", "Boş Satırları Tamamen Sil (dropna)", "Sayısal: Medyan / Kategorik: En Sık Görülen", "Sayısal: 0 ile Doldur"]
            )
            if st.button("🚀 Eksik Değer Stratejisini Uygula"):
                if "Sil" in impute_strategy:
                    df_working = df_working.dropna().reset_index(drop=True)
                elif "Medyan" in impute_strategy:
                    for col in df_working.select_dtypes(include=[np.number]).columns:
                        df_working[col] = df_working[col].fillna(df_working[col].median())
                    for col in df_working.select_dtypes(include=['object', 'category']).columns:
                        mode_val = df_working[col].mode()[0] if not df_working[col].mode().empty else "Bilinmiyor"
                        df_working[col] = df_working[col].fillna(mode_val)
                elif "0 ile Doldur" in impute_strategy:
                    df_working = df_working.fillna(0)
                
                st.session_state['df_raw'] = df_working
                st.success("✨ Eksik değer stratejisi başarıyla uygulandı!")
                st.rerun()

        st.markdown("---")
        st.subheader("📋 Veri Seti Kalite ve Eksik Değer Karnesi")
        info_df = pd.DataFrame({
            "Veri Tipi": df_working.dtypes.astype(str),
            "Eksik Değer Sayısı": df_working.isnull().sum(),
            "Eksik Oranı (%)": (df_working.isnull().mean() * 100).round(2),
            "Benzersiz (Unique) Değer": df_working.nunique()
        })
        st.dataframe(info_df, use_container_width=True)
    else:
        st.info("👈 Lütfen devam etmek için yukarıdan veri setinizi yükleyin.")

# ==============================================================================
# BÖLÜM 1: Kurulum ve Proje Ayarları
# ==============================================================================
elif current_section == "Bölüm 1: Kurulum ve Proje Ayarları":
    st.header("Bölüm 1: Kurulum ve Proje Ayarları")
    st.markdown("""
    <div class="section-box">
    <h3>Proje Yapılandırma ve Meta-Veri Yönetimi</h3>
    <p>Bu bölümde projenizin temel çalışma parametrelerini, seed değerlerini ve ortam yapılandırmasını yönetebilirsiniz.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        project_name = st.text_input("Proje Adı", value="HR Attrition & Churn Analytics SaaS")
        problem_type = st.selectbox("Problem Tipi", ["Binary Classification (İkili Sınıflandırma)", "Regression (Sayısal Tahmin)"])
    with col_set2:
        global_seed = st.number_input("Global Random Seed (Tekrarlanabilirlik)", value=42, step=1)
        target_metric = st.selectbox("Ana Optimizasyon Metriği", ["ROC-AUC", "F1-Score", "Accuracy"])
        
    st.session_state['project_config'] = {
        "name": project_name,
        "type": problem_type,
        "seed": global_seed,
        "metric": target_metric
    }
    
    st.markdown("---")
    st.subheader("💻 Sistem ve Ortam Diagnostiği")
    sys_col1, sys_col2, sys_col3 = st.columns(3)
    with sys_col1:
        st.metric("Python Sürümü", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    with sys_col2:
        st.metric("XGBoost Durumu", "Aktif ✅" if XGB_AVAILABLE else "Devre Dışı ❌")
    with sys_col3:
        st.metric("SHAP Açıklanabilirlik", "Hazır 🚀" if SHAP_AVAILABLE else "Yükleniyor...")

# ==============================================================================
# BÖLÜM 3: Hedef ve Özellik Hazırlama
# ==============================================================================
elif current_section == "Bölüm 3: Hedef ve Özellik Hazırlama":
    st.header("Bölüm 3: Hedef ve Özellik Hazırlama")
    df = st.session_state['df_raw']
    
    target_col = st.selectbox("Hedef Değişkeni (Target) Seçin:", df.columns, index=min(len(df.columns)-1, 0))
    drop_cols = st.multiselect("Silinecek / Gereksiz ID Sütunlarını Seçin:", df.columns.tolist())
    
    st.session_state['target_col'] = target_col
    st.session_state['drop_cols'] = drop_cols
    
    st.markdown("---")
    st.subheader("Özellik Mühendisliği Kararları")
    apply_fe = st.checkbox("Özel Özellik Mühendisliği Uygula (Tarih farkları, kıdem, maaş oranları)", value=True)
    st.session_state['apply_fe'] = apply_fe
    
    if st.button("Hedef ve Özellikleri Onayla"):
        df_clean = df.drop(columns=drop_cols, errors='ignore')
        st.session_state['df_clean'] = df_clean
        st.success("Özellik hazırlama adımı tamamlandı!")

# ==============================================================================
# BÖLÜM 4: Train / Test Bölme
# ==============================================================================
elif current_section == "Bölüm 4: Train / Test Bölme":
    st.header("Bölüm 4: Train / Test Bölme")
    df_clean = st.session_state['df_clean']
    target_col = st.session_state['target_col']
    
    test_size = st.slider("Test Seti Oranı (%)", 10, 40, 25, 5) / 100.0
    stratify_choice = st.checkbox("Stratified Split (Sınıf oranlarını koruyarak böl)", value=True)
    
    if st.button("Veriyi Train / Test Olarak Böl"):
        X = df_clean.drop(columns=[target_col])
        y_raw = df_clean[target_col]
        
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
        st.session_state['label_encoder'] = le
        
        strat = y if stratify_choice else None
        seed_val = st.session_state.get('project_config', {}).get('seed', 42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed_val, stratify=strat)
        
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        
        st.success(f
