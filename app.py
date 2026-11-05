import os
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# --- OTOMATİK KÜTÜPHANE KONTROLÜ ---
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==============================================================================
# STREAMLIT ARAYÜZ YAPILANDIRMASI VE MODERN CSS STİLLERİ
# ==============================================================================
st.set_page_config(
    page_title="Kurumsal Otomatik ML Platformu", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel Modern Tasarım CSS Kodları
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚡ Kurumsal Otomatik Veri Bilimi Platformu</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Uçtan uca veri temizliği, çoklu model mimarisi, Voting, Optuna ve SHAP açıklanabilirliği tek platformda.</p>', unsafe_allow_html=True)

# ==============================================================================
# 1. ÖZEL ÖZELLİK MÜHENDİSLİĞİ (Custom Transformer)
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

# ==============================================================================
# 2. ÖNBELLEKLENMİŞ VERİ YÜKLEME VE ÖN İŞLEME
# ==============================================================================
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
# 3. YAN MENÜ (SIDEBAR) - MODEL & HİPERPARAMETRE YÖNETİMİ
# ==============================================================================
st.sidebar.markdown("### ⚙️ Veri Yapılandırması")
uploaded_file = st.sidebar.file_uploader("Veri Seti Yükle (.xlsx / .csv)", type=["xlsx", "csv"])

target_col = st.sidebar.text_input("Hedef Sütun (Target)", value="calisma_durumu")
drop_cols_input = st.sidebar.text_input("Silinecek ID Sütunları", value="sicil_no")
leakage_cols_input = st.sidebar.text_input("Sızıntı (Leakage) Sütunları", value="cikis_tarihi, cikis_nedeni, aktif_calisan_mi")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Seçimi ve Hiperparametreler")

# Model 1: Logistic Regression
use_lr = st.sidebar.checkbox("Logistic Regression", value=True)
lr_c = st.sidebar.slider("LR - Düzenlileştirme (C)", 0.01, 10.0, 1.0, help="Küçük değerler aşırı öğrenmeyi engeller.")

# Model 2: Random Forest
use_rf = st.sidebar.checkbox("Random Forest", value=True)
rf_n_estimators = st.sidebar.slider("RF - Ağaç Sayısı", 50, 300, 150)
rf_max_depth = st.sidebar.slider("RF - Maksimum Derinlik", 3, 25, 10)

# Model 3: Gradient Boosting
use_gb = st.sidebar.checkbox("Gradient Boosting", value=True)
gb_lr = st.sidebar.slider("GB - Öğrenme Hızı (Learning Rate)", 0.01, 0.3, 0.1)

# Model 4: XGBoost
use_xgb = st.sidebar.checkbox("XGBoost", value=True if XGB_AVAILABLE else False, disabled=not XGB_AVAILABLE)

# ==============================================================================
# 4. ANA AKIŞ VE SEKMELİ (TABS) YAPI
# ==============================================================================
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    
    # Sızıntı Temizliği
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

        # Sekmeler (Modern UI)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📂 Keşifsel Veri Analizi (EDA)", 
            "⚙️ Model Eğitimi & Voting", 
            "📈 İstatistiksel Değerlendirme", 
            "🔍 SHAP Model Açıklanabilirliği",
            "🎯 Bireysel Risk Simülasyonu"
        ])

        with tab1:
            st.subheader("Veri Seti Karakteristiği ve Dağılımlar")
            st.dataframe(df_raw.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Hedef Değişken Sınıf Oranları:**")
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.countplot(x=y, ax=ax, palette="crest")
                st.pyplot(fig)
            with col2:
                st.write("**Eksik Değer Matrisi:**")
                missing_df = pd.DataFrame({"Eksik Sayısı": df_clean.isnull().sum()})
                st.dataframe(missing_df[missing_df["Eksik Sayısı"] > 0], use_container_width=True)

        with tab2:
            st.subheader("Topluluk (Voting) Model Yapılandırması")
            st.markdown("Seçtiğin modellerin güçlerini birleştirerek daha kararlı tahminler üreten `VotingClassifier` kuruyoruz.")
            
            if st.button("🚀 Modelleri Eğit / Güncelle", type="primary"):
                with st.spinner("Modeller eğitiliyor ve boru hattı (Pipeline) kuruluyor..."):
                    active_estimators = []
                    if use_lr:
                        active_estimators.append(('lr', LogisticRegression(C=lr_c, max_iter=1000, random_state=42, class_weight='balanced')))
                    if use_rf:
                        active_estimators.append(('rf', RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42, class_weight='balanced')))
                    if use_gb:
                        active_estimators.append(('gb', GradientBoostingClassifier(learning_rate=gb_lr, random_state=42)))
                    if use_xgb and XGB_AVAILABLE:
                        active_estimators.append(('xgb', XGBClassifier(random_state=42)))

                    if not active_estimators:
                        st.warning("Lütfen sol menüden en az bir model seçin!")
                    else:
                        voting_clf = VotingClassifier(estimators=active_estimators, voting='soft')
                        final_pipeline = Pipeline([
                            ('fe', HRFeatureEngineer(ordinal_mappings=ordinal_mappings)),
                            ('prep', preprocessor),
                            ('model', voting_clf)
                        ])

                        final_pipeline.fit(X_train_fe, y_train)
                        
                        # Oturum Hafızasına Kayıt (Gereksiz yeniden eğitimi önler)
                        st.session_state['trained_pipeline'] = final_pipeline
                        st.session_state['X_train_fe'] = X_train_fe
                        st.session_state['X_test_fe'] = X_test_fe
                        st.session_state['y_test'] = y_test
                        st.session_state['X_test_raw'] = X_test
                        
                        st.success("Tüm modeller başarıyla eğitildi ve oturum hafızasına kaydedildi!")

        with tab3:
            st.subheader("Model Performansı ve İstatistiksel Yorumlar")
            if 'trained_pipeline' in st.session_state:
                pipeline = st.session_state['trained_pipeline']
                xtest = st.session_state['X_test_fe']
                ytest = st.session_state['y_test']
                
                y_pred = pipeline.predict(xtest)
                y_proba = pipeline.predict_proba(xtest)[:, 1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sınıflandırma Raporu:**")
                    report = classification_report(ytest, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                with col2:
                    st.write("**Karışıklık Matrisi (Confusion Matrix):**")
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    sns.heatmap(confusion_matrix(ytest, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                    st.pyplot(fig)
                
                auc = roc_auc_score(ytest, y_proba)
                f1 = f1_score(ytest, y_pred, average='weighted')
                
                st.markdown(f"""
                <div class="card">
                    <h4>📊 Otomatik İstatistiksel Analiz Raporu</h4>
                    <ul>
                        <li><b>ROC-AUC Skoru ({auc:.4f}):</b> Modelin pozitif ve negatif sınıfları birbirinden ayırt etme kapasitesi oldukça başarılıdır.</li>
                        <li><b>F1-Score ({f1:.4f}):</b> Dengesiz sınıflar göz önüne alındığında modelin genel kararlılığını doğrular.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("⚠️ Lütfen önce 'Model Eğitimi' sekmesinden modeli çalıştırın.")

        with tab4:
            st.subheader("🔍 Model Açıklanabilirliği (SHAP Değerleri)")
            st.markdown("Modelin kararları hangi özelliklere (maaş, kıdem, yaş vb.) dayanarak verdiğini inceleyin.")
            if 'trained_pipeline' in st.session_state and SHAP_AVAILABLE:
                if st.button("SHAP Analizini Hesapla"):
                    with st.spinner("SHAP değerleri hesaplanıyor..."):
                        # Alt modeli veya ağaç tabanlı bileşeni alarak özet çıkaralım
                        model_step = st.session_state['trained_pipeline'].named_steps['model']
                        prep_step = st.session_state['trained_pipeline'].named_steps['prep']
                        fe_step = st.session_state['trained_pipeline'].named_steps['fe']
                        
                        transformed_data = prep_step.transform(fe_step.transform(X_test))
                        
                        # Eğer Voting içinde Random Forest varsa onun üzerinden SHAP çıkaralım
                        rf_est = dict(model_step.estimators_).get('rf', None)
                        if rf_est:
                            explainer = shap.TreeExplainer(rf_est)
                            shap_values = explainer.shap_values(transformed_data)
                            
                            fig, ax = plt.subplots(figsize=(8, 5))
                            shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, transformed_data, show=False)
                            st.pyplot(fig)
                        else:
                            st.warning("SHAP görselleştirmesi için model grubunda Random Forest bulunmalıdır.")
            else:
                st.info("Lütfen önce modeli eğitin (SHAP kütüphanesinin yüklü olduğundan emin olun).")

        with tab5:
            st.subheader("🎯 Bireysel Çalışan Risk Simülasyonu")
            if 'trained_pipeline' in st.session_state:
                sample_idx = st.slider("Test Setinden Çalışan Seçin", 0, len(st.session_state['X_test_raw'])-1, 0)
                single_row = st.session_state['X_test_raw'].iloc[[sample_idx]]
                st.dataframe(single_row, use_container_width=True)
                
                if st.button("Risk Skorunu Hesapla", type="primary"):
                    pred_prob = st.session_state['trained_pipeline'].predict_proba(st.session_state['X_test_fe'].iloc[[sample_idx]])[:, 1][0]
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric(label="Ayrılma (Churn) Olasılığı", value=f"%{pred_prob*100:.1f}")
                    with col_m2:
                        if pred_prob > 0.5:
                            st.error("⚠️ Yüksek Risk Grubu!")
                        else:
                            st.success("✅ Düşük Risk / Güvenli.")
            else:
                st.info("Simülasyon için önce modeli eğtmelisiniz.")
else:
    st.info("👈 Başlamak için lütfen sol menüden veri setinizi yükleyin.")
