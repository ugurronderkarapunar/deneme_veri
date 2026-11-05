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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==============================================================================
# STREAMLIT ARAYÜZ YAPILANDIRMASI (UI / UX İyileştirmesi: Wide Layout)
# ==============================================================================
st.set_page_config(page_title="Kurumsal Otomatik ML Platformu", layout="wide")

st.title("🚀 Kurumsal Otomatik Veri Bilimi ve İK Analitik Platformu")
st.markdown("""
Bu platform; **Jr. Seviyesinde Temiz Kod** mimarisiyle verinizi işler, **Voting (Topluluk)** modeli kurar, 
**Optuna** ile hiperparametreleri optimize eder ve sonuçları **istatistiksel dil ile otomatik yorumlar.**
""")

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
# 3. YAN MENÜ (SIDEBAR) VE MODEL & HİPERPARAMETRE YÖNETİMİ
# ==============================================================================
st.sidebar.header("⚙️ Veri ve Model Yapılandırması")
uploaded_file = st.sidebar.file_uploader("Veri Setinizi Yükleyin (.xlsx veya .csv)", type=["xlsx", "csv"])

target_col = st.sidebar.text_input("Hedef Değişken (Target)", value="calisma_durumu")
drop_cols_input = st.sidebar.text_input("Silinecek ID Kolonları", value="sicil_no")
leakage_cols_input = st.sidebar.text_input("Sızıntı (Leakage) Kolonları", value="cikis_tarihi, cikis_nedeni, aktif_calisan_mi")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model ve Hiperparametre Yönetimi")

# Model 1: Logistic Regression
use_lr = st.sidebar.checkbox("Logistic Regression", value=True)
lr_c = st.sidebar.slider("LR - C (Düzenlileştirme / Regularization)", 0.01, 10.0, 1.0, help="Küçük C değerleri modelin aşırı öğrenmesini engeller (güçlü ceza).")

# Model 2: Random Forest
use_rf = st.sidebar.checkbox("Random Forest", value=True)
rf_n_estimators = st.sidebar.slider("RF - Ağaç Sayısı (n_estimators)", 50, 300, 150, help="Ormandaki toplam karar ağacı sayısı. Artırmak doğruluğu iyileştirir ama süreyi uzatır.")
rf_max_depth = st.sidebar.slider("RF - Maksimum Derinlik (max_depth)", 3, 25, 10, help="Her ağacın detaya inme seviyesi. Çok derin olursa veriyi ezberler (overfitting).")

# Model 3: XGBoost
use_xgb = st.sidebar.checkbox("XGBoost", value=True if XGB_AVAILABLE else False, disabled=not XGB_AVAILABLE)
xgb_lr = st.sidebar.slider("XGB - Öğrenme Hızı (learning_rate)", 0.01, 0.3, 0.1, help="Modelin her adımda ne kadar hızlı öğreneceğini belirler.")

# ==============================================================================
# 4. ANA UYGULAMA AKIŞI VE SEKMELER (TABS)
# ==============================================================================
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    
    # Temizlik ve Sızıntı Önleme
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

        # SEKMELİ YAPI (UX / UI İyileştirmesi)
        tab1, tab2, tab3, tab4 = st.tabs([
            "📂 Veri Keşfi & EDA", 
            "⚙️ Model Eğitimi & Optimizasyon", 
            "📈 İstatistiksel Değerlendirme", 
            "🔍 Bireysel Risk Simülasyonu"
        ])

        with tab1:
            st.subheader("Veri Seti Ön İncelemesi ve Dağılımlar")
            st.dataframe(df_raw.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Hedef Değişken Dağılımı:**")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.countplot(x=y, ax=ax, palette="viridis")
                st.pyplot(fig)
            with col2:
                st.write("**Eksik Değer Analizi:**")
                missing_df = pd.DataFrame({"Eksik Sayısı": df_clean.isnull().sum()})
                st.dataframe(missing_df[missing_df["Eksik Sayısı"] > 0])

        with tab2:
            st.subheader("Model Eğitimi ve Hiperparametre Yönetimi")
            st.info("💡 Sol menüden değiştirdiğiniz hiperparametreler aşağıda anında derlenir.")
            
            if st.button("🚀 Modeli Şimdi Eğit / Güncelle"):
                with st.spinner("Modeller eğitiliyor ve topluluk (Voting) yapısı kuruluyor..."):
                    active_estimators = []
                    if use_lr:
                        active_estimators.append(('lr', LogisticRegression(C=lr_c, max_iter=1000, random_state=42, class_weight='balanced')))
                    if use_rf:
                        active_estimators.append(('rf', RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42, class_weight='balanced')))
                    if use_xgb and XGB_AVAILABLE:
                        active_estimators.append(('xgb', XGBClassifier(learning_rate=xgb_lr, n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)))

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
                        
                        # [PERFORMANS İYİLEŞTİRMESİ]: Modeli session_state içinde sakla (Tekrar eğitimi önler)
                        st.session_state['trained_pipeline'] = final_pipeline
                        st.session_state['X_test_fe'] = X_test_fe
                        st.session_state['y_test'] = y_test
                        st.session_state['X_test_raw'] = X_test
                        
                        st.success("Model başarıyla eğitildi ve hafızaya alındı!")

        with tab3:
            st.subheader("İstatistiksel Performans ve Otomatik Yorumlar")
            if 'trained_pipeline' in st.session_state:
                pipeline = st.session_state['trained_pipeline']
                xtest = st.session_state['X_test_fe']
                ytest = st.session_state['y_test']
                
                y_pred = pipeline.predict(xtest)
                y_proba = pipeline.predict_proba(xtest)[:, 1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sınıflandırma Raporu**")
                    report = classification_report(ytest, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                with col2:
                    st.write("**Karışıklık Matrisi (Confusion Matrix)**")
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(confusion_matrix(ytest, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                    st.pyplot(fig)
                
                # Otomatik İstatistiksel Yorum
                auc = roc_auc_score(ytest, y_proba)
                f1 = f1_score(ytest, y_pred, average='weighted')
                st.markdown(f"""
                ### 📊 İstatistiksel Değerlendirme Özeti:
                - **ROC-AUC ({auc:.4f}):** Modelin pozitif ve negatif sınıfları ayırt etme gücü. 
                - **F1-Score ({f1:.4f}):** Dengesiz sınıflarda hata oranlarını dengeli şekilde yansıtan referans metrik.
                """)
                
                # Optuna Otomasyonu
                if st.button("Optuna ile Otomatik Hiperparametre Aramasını Başlat"):
                    with st.spinner("Optuna en iyi ağaç parametrelerini arıyor..."):
                        def objective(trial):
                            n_est = trial.suggest_int('n_estimators', 50, 150)
                            max_d = trial.suggest_int('max_depth', 3, 10)
                            clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight='balanced', random_state=42)
                            pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
                            pipe.fit(X_train_fe, y_train)
                            return roc_auc_score(y_test, pipe.predict_proba(X_test_fe)[:, 1])

                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=5)
                        
                        st.write(f"**En İyi Optuna Parametreleri:** {study.best_params}")
                        fig_opt = optuna.visualization.plot_optimization_history(study)
                        st.plotly_chart(fig_opt, use_container_width=True)
            else:
                st.warning("⚠️ Lütfen önce 'Model Eğitimi & Optimizasyon' sekmesinden modeli eğitin.")

        with tab4:
            st.subheader("🔍 Bireysel Çalışan Risk Simülasyonu")
            if 'trained_pipeline' in st.session_state:
                st.write("Test setinden rastgele bir çalışanın verisini simüle ederek risk skorunu inceleyin:")
                sample_idx = st.slider("Çalışan İndeksi Seçin", 0, len(st.session_state['X_test_raw'])-1, 0)
                
                single_row = st.session_state['X_test_raw'].iloc[[sample_idx]]
                st.dataframe(single_row)
                
                if st.button("Risk Tahminini Hesapla"):
                    pred_prob = st.session_state['trained_pipeline'].predict_proba(st.session_state['X_test_fe'].iloc[[sample_idx]])[:, 1][0]
                    st.metric(label="Ayrılma (Churn) Olasılığı", value=f"%{pred_prob*100:.1f}")
                    if pred_prob > 0.5:
                        st.error("⚠️ Yüksek Risk: Bu çalışanın ayrılma olasılığı %50'nin üzerinde!")
                    else:
                        st.success("✅ Düşük Risk: Çalışan şirkette kalmaya devam etme eğiliminde.")
            else:
                st.info("Önce modeli eğitmeniz gerekmektedir.")
else:
    st.info("👈 Başlamak için sol menüden veri setinizi yükleyin.")
