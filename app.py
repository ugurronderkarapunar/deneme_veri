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

# Streamlit Kütüphanesi
import streamlit as st

# --- OTOMATİK KÜTÜPHANE KONTROLÜ ---
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "xgboost", "streamlit"])
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==============================================================================
# STREAMLIT ARAYÜZ YAPILANDIRMASI
# ==============================================================================
st.set_page_config(page_title="Otomatik ML & İK Analiz Platformu", layout="wide")

st.title("🚀 Otomatik Veri Bilimi ve Model Optimizasyon Platformu")
st.markdown("""
Bu web uygulaması; **Jr. Seviyesinde Temiz Kod** mimarisiyle verinizi işler, **Voting (Topluluk)** modeli kurar, 
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
# 2. YARDIMCI FONKSİYONLAR
# ==============================================================================
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

def automatic_result_interpreter(y_test, y_pred, y_proba):
    st.subheader("📈 Otomatik İstatistiksel ve ML Performans Yorumları")
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    st.write(f"- **ROC-AUC Skoru ({auc:.4f}):** Modelin sınıfları birbirinden ayırma yeteneğini gösterir. 0.85 üzeri mükemmeldir.")
    st.write(f"- **F1-Score ({f1:.4f}):** Hassasiyet ve duyarlılığın dengeli harmonik ortalamasıdır.")
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    st.write(f"- **Hata Analizi:** Model {tp} doğru pozitif, {tn} doğru negatif yakaladı. **{fp} adet yanlış alarm (FP)** ve **{fn} adet kaçırılan risk (FN)** üretti.")

# ==============================================================================
# 3. YAN MENÜ (SIDEBAR) - KULLANICI KONTROLÜ
# ==============================================================================
st.sidebar.header("⚙️ Proje ve Model Ayarları")
uploaded_file = st.sidebar.file_uploader("Veri Setinizi Yükleyin (.xlsx veya .csv)", type=["xlsx", "csv"])

target_col = st.sidebar.text_input("Hedef Değişken (Target)", value="calisma_durumu")
drop_cols_input = st.sidebar.text_input("Silinecek ID Kolonları (virgülle ayırın)", value="sicil_no")
leakage_cols_input = st.sidebar.text_input("Sızıntı Yapacak Kolonlar (virgülle ayırın)", value="cikis_tarihi, cikis_nedeni, aktif_calisan_mi")

st.sidebar.subheader("🤖 Voting Model Seçimi")
use_lr = st.sidebar.checkbox("Logistic Regression", value=True)
use_rf = st.sidebar.checkbox("Random Forest", value=True)
use_xgb = st.sidebar.checkbox("XGBoost", value=True if XGB_AVAILABLE else False)

st.sidebar.subheader("🎛️ Hiperparametre Ayarları")
n_estimators = st.sidebar.slider("Random Forest - Ağaç Sayısı (n_estimators)", 50, 300, 150)
max_depth = st.sidebar.slider("Random Forest - Maksimum Derinlik (max_depth)", 3, 20, 10)

# ==============================================================================
# 4. ANA ÇALIŞTIRMA MANTIĞI
# ==============================================================================
if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
        
    st.success("Veri başarıyla yüklendi!")
    st.dataframe(df.head())
    
    if st.button("🚀 Modeli Eğit ve Analizi Başlat"):
        with st.spinner("Veriler temizleniyor, özellikler türetiliyor ve model eğitiliyor..."):
            # Temizlik
            drop_list = [c.strip() for c in drop_cols_input.split(",") if c.strip()]
            leak_list = [c.strip() for c in leakage_cols_input.split(",") if c.strip()]
            all_drops = drop_list + leak_list
            
            df_clean = df.drop(columns=[c for c in all_drops if c in df.columns], errors='ignore')
            
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

                # Voting Estimators
                active_estimators = []
                if use_lr:
                    active_estimators.append(('lr', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')))
                if use_rf:
                    active_estimators.append(('rf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced')))
                if use_xgb and XGB_AVAILABLE:
                    active_estimators.append(('xgb', XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)))

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
                    
                    # Tahminler
                    y_pred = final_pipeline.predict(X_test_fe)
                    y_proba = final_pipeline.predict_proba(X_test_fe)[:, 1]
                    
                    # Sonuçları Göster
                    st.success("Model eğitimi tamamlandı!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📊 Sınıflandırma Raporu")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())
                        
                    with col2:
                        st.subheader("📉 Karışıklık Matrisi (Confusion Matrix)")
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                        st.pyplot(fig)

                    automatic_result_interpreter(y_test, y_pred, y_proba)
                    
                    # Optuna Optimizasyon Grafikleri
                    st.subheader("📊 Hiperparametre Optimizasyon ve Etki Grafikleri (Optuna)")
                    def objective(trial):
                        n_est = trial.suggest_int('n_estimators', 50, 200)
                        max_d = trial.suggest_int('max_depth', 3, 12)
                        clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight='balanced', random_state=42)
                        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
                        pipe.fit(X_train_fe, y_train)
                        return roc_auc_score(y_test, pipe.predict_proba(X_test_fe)[:, 1])

                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=8)
                    
                    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
                    st.pyplot(fig1)
                    
                    fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
                    st.pyplot(fig2)
else:
    st.info("👈 Başlamak için sol menüden veri setinizi yükleyin.")

# ==============================================================================
# 5. [İSTENEN ÖZELLİK 3]: PROJE REHBERİ VE YÖNLENDİRME (ROADMAP)
# ==============================================================================
st.markdown("---")
st.subheader("🧭 Proje Rehberi: Bu Adımdan Sonra Ne Yapmalıyım?")
st.markdown("""
Çalıştığınız projenin türüne göre sonraki adımlarınız şunlar olmalıdır:
1. **Mevcut Projeniz (Tabular Classification / Churn):** 
   * *Bu adımdan sonra:* Eşik değeri optimizasyonu (Threshold Tuning) yapın ve **SHAP** kütüphanesiyle modelin hangi özelliğe bakarak karar verdiğini açıklayın.
2. **Eğer Projeniz Regresyon (Fiyat / Maaş Tahmini) Olsaydı:**
   * *Bu adımdan sonra:* Accuracy yerine **RMSE** ve **MAE** metriklerini takip edin; hata payı dağılım grafiklerini (Residual Plots) çizin.
3. **Eğer Projeniz Zaman Serisi (Satış / Hava Durumu) Olsaydı:**
   * *Bu adımdan sonra:* Standart `train_test_split` yerine veriyi kronolojik olarak bölün ve gecikme (Lag) özellikleri türetin.
""")

st.markdown("---")
st.subheader("⚠️ Düşebileceğiniz Kritik Hatalar (Pitfalls)")
st.markdown("""
- **Veri Sızıntısı (Data Leakage):** Gelecekteki bilgiyi (örneğin işten ayrılma tarihi) modele vermeyin.
- **Test Setini Kirletmek:** Ölçeklendirme (`StandardScaler`) işlemini tüm veriye değil, sadece eğitim setine (`fit_transform`) uygulamalısınız (Yukarıdaki Pipeline bunu sizin için otomatik yapar).
- **Dengesiz Sınıf Yanılgısı:** Azınlık sınıfı göz ardı etmemek için `class_weight='balanced'` parametresini mutlaka aktif tutun.
""")
