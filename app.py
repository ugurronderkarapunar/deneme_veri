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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
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
st.markdown('<p class="sub-header">Modern interaktif grafikler, toplu skorlama, gelişmiş Voting ve kusursuz veri ön işleme.</p>', unsafe_allow_html=True)

# ==============================================================================
# 1. KUSURSUZ ÖZEL ÖZELLİK MÜHENDİSLİĞİ (Custom Transformer)
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
    df_clean = df_raw.drop(columns=[c for c in all_drops if c in df_clean.columns], errors='ignore')
    
    if target_col not in df_clean.columns:
        st.error(f"Hata: '{target_col}' sütunu veri setinde bulunamadı!")
    else:
        X = df_clean.drop(columns=[target_col])
        y_raw = df_clean[target_col]
        
        # --- KUSURSUZ HEDEF DEĞİŞKEN (TARGET) DÖNÜŞÜMÜ ---
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
        st.sidebar.info(f"Sınıf Eşleşmeleri: {dict(zip(le.classes_, range(len(le.classes_))))}")

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
                val_counts = y_raw.value_counts().reset_index()
                val_counts.columns = ['Sınıf', 'Adet']
                fig_pie = px.pie(val_counts, names='Sınıf', values='Adet', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.write("**Eksik Değer Analizi:**")
                missing_df = df_clean.isnull().sum().reset_index()
                missing_df.columns = ['Sütun', 'Eksik Sayısı']
                missing_df = missing_df[missing_df['Eksik Sayısı'] > 0]
                if not missing_df.empty:
                    fig_miss = px.bar(missing_df, x='Sütun', y='Eksik Sayısı', color='Eksik Sayısı', color_continuous_scale='Reds')
                    st.plotly_chart(fig_miss, use_container_width=True)
                else:
                    st.success("Tebrikler! Veri setinde eksik değer bulunmuyor.")

        with tab2:
            st.subheader("Topluluk (Voting) Model Yapılandırıcı")
            st.markdown("Seçilen tüm algoritmaların olasılıklarını birleştirerek en kararlı kararları üreten mimari.")
            
            if st.button("🚀 Modelleri Eğit ve Kaydet", type="primary"):
                with st.spinner("Modeller eğitiliyor..."):
                    active_estimators = []
                    individual_models = {}
                    
                    if use_lr:
                        m = LogisticRegression(C=lr_c, max_iter=1000, random_state=42, class_weight='balanced')
                        active_estimators.append(('lr', m))
                        individual_models['Logistic Regression'] = m
                    if use_rf:
                        m = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42, class_weight='balanced')
                        active_estimators.append(('rf', m))
                        individual_models['Random Forest'] = m
                    if use_gb:
                        m = GradientBoostingClassifier(learning_rate=gb_lr, random_state=42)
                        active_estimators.append(('gb', m))
                        individual_models['Gradient Boosting'] = m
                    if use_xgb and XGB_AVAILABLE:
                        m = XGBClassifier(random_state=42)
                        active_estimators.append(('xgb', m))
                        individual_models['XGBoost'] = m

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
                        
                        model_scores = {}
                        for name, model in individual_models.items():
                            p = Pipeline([('fe', HRFeatureEngineer(ordinal_mappings=ordinal_mappings)), ('prep', preprocessor), ('model', model)])
                            p.fit(X_train_fe, y_train)
                            sc = roc_auc_score(y_test, p.predict_proba(X_test_fe)[:, 1])
                            model_scores[name] = sc

                        st.session_state['trained_pipeline'] = final_pipeline
                        st.session_state['X_test_fe'] = X_test_fe
                        st.session_state['y_test'] = y_test
                        st.session_state['X_test_raw'] = X_test
                        st.session_state['model_scores'] = model_scores
                        
                        st.success("Tüm modeller başarıyla eğitildi!")

        with tab3:
            st.subheader("Model Karşılaştırma Liderlik Tablosu & Performans")
            if 'trained_pipeline' in st.session_state:
                pipeline = st.session_state['trained_pipeline']
                xtest = st.session_state['X_test_fe']
                ytest = st.session_state['y_test']
                
                threshold = st.slider("İnteraktif Karar Eşiği (Threshold Optimization)", 0.1, 0.9, 0.5, 0.05)
                
                y_proba = pipeline.predict_proba(xtest)[:, 1]
                y_pred_custom = (y_proba >= threshold).astype(int)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Tekli Modellerin ROC-AUC Karşılaştırması:**")
                    scores_df = pd.DataFrame(list(st.session_state['model_scores'].items()), columns=['Model', 'ROC-AUC'])
                    voting_auc = roc_auc_score(ytest, y_proba)
                    scores_df.loc[len(scores_df)] = ['Voting Ensemble (Seçilen)', voting_auc]
                    scores_df = scores_df.sort_values(by='ROC-AUC', ascending=False)
                    
                    fig_bar = px.bar(scores_df, x='Model', y='ROC-AUC', color='ROC-AUC', color_continuous_scale='Blues')
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with col2:
                    st.write("**Modern İnteraktif Karışıklık Matrisi (Heatmap):**")
                    cm = confusion_matrix(ytest, y_pred_custom)
                    # px.imshow kullanarak modern ve hatasız interaktif ısı haritası
                    fig_cm = px.imshow(
                        cm, 
                        text_auto=True, 
                        labels=dict(x="Tahmin Edilen", y="Gerçek Sınıf", color="Adet"),
                        x=['Tahmin: 0', 'Tahmin: 1'], 
                        y=['Gerçek: 0', 'Gerçek: 1'],
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("⚠️ Lütfen önce modeli eğitin.")

        with tab4:
            st.subheader("🔍 SHAP Model Açıklanabilirliği")
            
            if 'trained_pipeline' in st.session_state:
                if not SHAP_AVAILABLE:
                    st.warning("⚠️ SHAP kütüphanesi ortamda eksik. Otomatik yükleniyor, lütfen bekleyin...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
                        import shap
                        SHAP_AVAILABLE = True
                        st.success("SHAP başarıyla yüklendi!")
                    except Exception as e:
                        st.error(f"SHAP yüklenemedi: {e}")

                if SHAP_AVAILABLE:
                    if st.button("SHAP Önem Grafiğini Üret"):
                        with st.spinner("SHAP değerleri hesaplanıyor..."):
                            model_step = st.session_state['trained_pipeline'].named_steps['model']
                            prep_step = st.session_state['trained_pipeline'].named_steps['prep']
                            fe_step = st.session_state['trained_pipeline'].named_steps['fe']
                            
                            transformed_data = prep_step.transform(fe_step.transform(X_test))
                            rf_est = dict(model_step.estimators_).get('rf', None)
                            if rf_est:
                                explainer = shap.TreeExplainer(rf_est)
                                shap_values = explainer.shap_values(transformed_data)
                                
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(8, 5))
                                shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, transformed_data, show=False)
                                st.pyplot(fig)
                            else:
                                st.warning("SHAP grafiği için modeller arasına Random Forest eklemelisiniz.")
            else:
                st.info("⚠️ Lütfen önce 'Model & Voting Eğitimi' sekmesinden modeli eğitin.")

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
                st.info("Simülasyon için önce modeli eğitmelisiniz.")

        with tab6:
            st.subheader("📁 Toplu Skorlama ve Rapor İndirme (Batch Prediction)")
            st.markdown("Yeni veya etiketlenmemiş çalışan verilerinizi yükleyerek tüm şirket için risk skorlarını tek seferde hesaplayın.")
            batch_file = st.file_uploader("Test / Yeni Veri Seti Yükle (.xlsx / .csv)", type=["xlsx", "csv"], key="batch")
            
            if batch_file is not None:
                df_batch = load_data(batch_file)
                st.dataframe(df_batch.head(), use_container_width=True)
                
                if st.button("Tüm Listeyi Skorla"):
                    if 'trained_pipeline' in st.session_state:
                        pipeline = st.session_state['trained_pipeline']
                        probs = pipeline.predict_proba(df_batch)[:, 1]
                        df_batch['Ayrilma_Olasiligi'] = [f"%{p*100:.2f}" for p in probs]
                        df_batch['Risk_Durumu'] = ["Yüksek Risk" if p > 0.5 else "Güvenli" for p in probs]
                        
                        st.success("Skorlama tamamlandı!")
                        st.dataframe(df_batch, use_container_width=True)
                        
                        csv = df_batch.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Skorlanmış Raporu İndir (CSV)", data=csv, file_name="risk_analiz_raporu.csv", mime="text/csv")
                    else:
                        st.error("Önce model eğitimi yapmalısınız!")
else:
    st.info("👈 Başlamak için sol menüden veri setinizi yükleyin.")
