import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# Kütüphane Kontrolleri (Graceful Fallback)
try:
    import xgboost as XGB
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as LGB
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    from sklearn.pipeline import Pipeline as ImbPipeline

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ==============================================================================
# 1. KISMI & İLERİ SEVİYE VERİ TEMİZLİĞİ VE STANDARDİZASYON FONKSİYONLARI
# ==============================================================================
def advanced_data_cleaning(df, config):
    """
    Kirli veriyi temizleyen, metinleri standartlaştıran, 
    anomalileri ve mükerrer kayıtları (duplicates) ayırt eden senior temizlik motoru.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # 1. Mükerrer (Duplicate) Satır Temizliği
    if config.get("drop_duplicates", True):
        df_clean.drop_duplicates(inplace=True)

    # 2. Sütun İsimlerinin Standardizasyonu (Boşluk, Türkçe karakter düzeltme)
    df_clean.columns = [str(col).strip().lower().replace(' ', '_') for col in df_clean.columns]

    # 3. Metin (String) Değerlerin Standardizasyonu (Strip & Title Case)
    cat_cols_raw = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols_raw:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # "nan" veya boş string kalıntılarını NaN'e çevir
        df_clean[col] = df_clean[col].replace(['nan', 'None', '', 'NULL', 'null'], np.nan)

    # 4. Sayısal Kolonlardaki Bozuk Verilerin ve Anomalilerin Temizlenmesi
    numeric_cols_raw = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols_raw:
        # Sonsuz değerleri NaN yap
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        
        # Mantık dışı anomali filtreleri (Örn: Yaş < 0 veya Yaş > 120 ise NaN yap)
        if 'yas' in col or 'age' in col:
            df_clean.loc[(df_clean[col] < 15) | (df_clean[col] > 100), col] = np.nan
        if 'maas' in col or 'salary' in col or 'income' in col:
            df_clean.loc[df_clean[col] < 0, col] = np.nan

    cleaned_rows = len(df_clean)
    print(f">>> Veri Temizliği Tamamlandı: {initial_rows} satırdan {cleaned_rows} satıra ulaşıldı.")
    return df_clean

# ==============================================================================
# 2. ÖZELLİK MÜHENDİSLİĞİ VE ESNEK PREPROCESSOR
# ==============================================================================
class EnterpriseFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        ref_date = pd.to_datetime('today')
        
        # Akıllı Tarih Mühendisliği
        if self.config.get("enable_date_fe", True):
            date_cols = [col for col in X_out.columns if 'tarih' in col or 'date' in col]
            for col in date_cols:
                try:
                    X_out[col] = pd.to_datetime(X_out[col], errors='coerce')
                    new_name = col.replace('_tarihi', '_yil').replace('_date', '_years')
                    X_out[new_name] = (ref_date - X_out[col]).dt.days / 365.25
                    X_out.drop(columns=[col], inplace=True)
                except:
                    pass

        # IQR ile Outlier Baskılama
        if self.config.get("enable_outlier", True):
            numeric_cols = X_out.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                q1, q3 = X_out[col].quantile(0.01), X_out[col].quantile(0.99)
                X_out[col] = X_out[col].clip(lower=q1, upper=q3)

        return X_out

def enforce_model_schema(df_user, expected_columns):
    df_aligned = pd.DataFrame()
    for col in expected_columns:
        if col in df_user.columns:
            df_aligned[col] = df_user[col]
        else:
            df_aligned[col] = np.nan
    return df_aligned

# ==============================================================================
# 3. DİNAMİK EĞİTİM VE MODEL YARIŞI MOTORU
# ==============================================================================
def train_enterprise_pipeline(file_path, config):
    ext = os.path.splitext(file_path)[-1].lower()
    df_raw = pd.read_excel(file_path) if ext in ['.xlsx', '.xls'] else pd.read_csv(file_path)
    
    # Kapsamlı Veri Temizliği ve Standardizasyon Adımı
    df = advanced_data_cleaning(df_raw, config)

    # Kolon isimleri temizlendiği için config'deki isimleri de uyumlu hale getir
    target = config["target_column"].strip().lower().replace(' ', '_')
    drop_cols = [c.strip().lower().replace(' ', '_') for c in config["drop_columns"]]
    leak_cols = [c.strip().lower().replace(' ', '_') for c in config["leakage_columns"]]

    to_remove = set(drop_cols + leak_cols)
    df.drop(columns=[c for c in to_remove if c in df.columns], inplace=True)

    X, y = df.drop(columns=[target]), df[target]

    # Hedef Değişkeni Güvenli Kodlama (0 ve 1)
    if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype('category').cat.codes
    y = np.where(y == y.min(), 0, 1)
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=42, stratify=y
    )

    fe = EnterpriseFeatureEngineer(config)
    X_train_fe = fe.fit_transform(X_train)

    num_cols = X_train_fe.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()
    bin_cols = [c for c in X_train_fe.columns if X_train_fe[c].nunique() == 2]
    pure_num = [c for c in num_cols if c not in bin_cols]
    
    ordinal_mappings_clean = {k.strip().lower().replace(' ', '_'): v for k, v in config["ordinal_mappings"].items()}
    ord_cols = [c for c in ordinal_mappings_clean.keys() if c in X_train_fe.columns]
    nom_cols = [c for c in cat_cols if c not in ord_cols]

    # Scaler Seçimi
    if config["scaler_type"] == "Standard (Z-Score)":
        scaler = StandardScaler()
    elif config["scaler_type"] == "MinMax (0-1)":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline_Step([('imp', SimpleImputer(strategy=config["imputation_strategy"])), ('scaler', scaler)]), pure_num),
        ('ord', Pipeline_Step([('imp', SimpleImputer(strategy='most_frequent')), ('ord', OrdinalEncoder(categories=[ordinal_mappings_clean[c] for c in ord_cols], handle_unknown='use_encoded_value', unknown_value=-1))]), ord_cols),
        ('nom', Pipeline_Step([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), nom_cols),
        ('bin', Pipeline_Step([('imp', SimpleImputer(strategy='most_frequent')), ('bin', OrdinalEncoder())]), bin_cols)
    ], remainder='drop')

    # Model Havuzu
    model_candidates = {}
    if "Random Forest" in config["selected_models"]:
        model_candidates["Random Forest"] = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    if "Logistic Regression" in config["selected_models"]:
        model_candidates["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    if "Hist Gradient Boosting" in config["selected_models"]:
        model_candidates["Hist Gradient Boosting"] = HistGradientBoostingClassifier(random_state=42)
    if XGB_AVAILABLE and "XGBoost" in config["selected_models"]:
        model_candidates["XGBoost"] = XGB.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    if LGB_AVAILABLE and "LightGBM" in config["selected_models"]:
        model_candidates["LightGBM"] = LGB.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    if not model_candidates:
        model_candidates["Random Forest"] = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

    # Model Yarışı (CV)
    best_score = -1
    best_model_name = None
    best_pipeline = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in model_candidates.items():
        steps = [('fe', fe), ('prep', preprocessor)]
        if config["use_smote"] and IMBLEARN_AVAILABLE:
            steps.append(('smote', SMOTE(random_state=42)))
        steps.append(('clf', clf))
        
        pipe = ImbPipeline(steps)
        scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = pipe.predict_proba(X_train.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y_train.iloc[va_idx], preds))
        
        mean_auc = np.mean(scores)
        if mean_auc > best_score:
            best_score = mean_auc
            best_model_name = name
            best_pipeline = pipe.fit(X_train, y_train)

    # Optuna Hiperparametre Optimizasyonu
    if OPTUNA_AVAILABLE and config["optuna_trials"] > 0:
        def objective(trial):
            if best_model_name == "Random Forest":
                clf = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 4, 15),
                    random_state=42, class_weight='balanced', n_jobs=-1
                )
            else:
                clf = model_candidates[best_model_name]

            steps = [('fe', fe), ('prep', preprocessor)]
            if config["use_smote"] and IMBLEARN_AVAILABLE:
                steps.append(('smote', SMOTE(random_state=42)))
            steps.append(('clf', clf))
            p = ImbPipeline(steps)
            p.fit(X_train, y_train)
            return roc_auc_score(y_test, p.predict_proba(X_test)[:, 1])

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config["optuna_trials"])
        
        if best_model_name == "Random Forest":
            final_clf = RandomForestClassifier(**study.best_params, random_state=42, class_weight='balanced', n_jobs=-1)
        else:
            final_clf = model_candidates[best_model_name]

        steps = [('fe', fe), ('prep', preprocessor)]
        if config["use_smote"] and IMBLEARN_AVAILABLE:
            steps.append(('smote', SMOTE(random_state=42)))
        steps.append(('clf', final_clf))
        best_pipeline = ImbPipeline(steps).fit(X_train, y_train)

    # Karar Eşiği Optimizasyonu
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    if config["threshold_strategy"] == "Youden Index (Optimum ROC)":
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    elif config["threshold_strategy"] == "Precision-Recall Balance (F1-Optimal)":
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
    else:
        optimal_threshold = 0.50

    bundle = {
        "pipeline": best_pipeline,
        "threshold": optimal_threshold,
        "columns": X.columns.tolist(),
        "best_model_name": best_model_name,
        "best_cv_auc": best_score
    }
    joblib.dump(bundle, config["model_output_path"])
    return bundle

from sklearn.pipeline import Pipeline as Pipeline_Step

# ==============================================================================
# 4. STREAMLIT ARAYÜZ MİMARİSİ
# ==============================================================================
st.set_page_config(page_title="Enterprise Senior ML Engine", layout="wide")

def main():
    st.title("🏆 Enterprise Senior ML, Data Cleaning & MLOps Engine")
    st.markdown("Veri temizliği, standardizasyon, EDA, model yarışı ve Optuna optimizasyonunu tek ekranda birleştiren profesyonel portal.")

    model_output_path = "enterprise_clean_engine.joblib"

    with st.sidebar:
        st.header("📂 1. Veri Yükleme")
        uploaded_train_file = st.file_uploader("Eğitim Verisi (.xlsx, .csv)", type=["xlsx", "csv"])

    if uploaded_train_file is not None:
        ext = os.path.splitext(uploaded_train_file.name)[-1].lower()
        df_raw = pd.read_excel(uploaded_train_file) if ext in ['.xlsx', '.xls'] else pd.read_csv(uploaded_train_file)
        
        # Önizleme için temizlik simülasyonu
        df_preview = df_raw.copy()
        df_preview.columns = [str(col).strip().lower().replace(' ', '_') for col in df_preview.columns]

        st.success(f"Veri Yüklendi! Ham Boyut: {df_raw.shape}")

        with st.expander("📊 Veri Sağlık & Kirli Veri Raporu (Data Cleaning Insights)"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Eksik Değer Analizi:**")
                missing_df = pd.DataFrame({
                    "Eksik Sayısı": df_preview.isnull().sum(),
                    "Eksik Oranı (%)": (df_preview.isnull().mean() * 100).round(2)
                })
                st.dataframe(missing_df[missing_df["Eksik Sayısı"] > 0], use_container_width=True)
            with c2:
                st.write("**Mükerrer (Duplicate) Satır Sayısı:**", df_preview.duplicated().sum())
                st.write("**Toplam Sütun Sayısı:**", df_preview.shape[1])

        st.markdown("---")
        st.header("⚙️ 2. Veri Temizliği ve Konfigürasyon Paneli")
        
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            target_column = st.selectbox("Hedef (Target) Kolon", df_preview.columns)
            drop_columns = st.multiselect("Atılacak Kolonlar (ID vb.)", df_preview.columns, default=[])
        with tc2:
            potential_leakage = [c for c in df_preview.columns if 'tarih' in c or 'cikis' in c]
            leakage_columns = st.multiselect("Sızıntı (Leakage) Kolonları", df_preview.columns, default=potential_leakage)
            imputation_strategy = st.selectbox("Eksik Veri Doldurma", ["median", "mean", "most_frequent"])
        with tc3:
            scaler_type = st.selectbox("Ölçekleme (Scaler)", ["Standard (Z-Score)", "MinMax (0-1)", "Robust"])
            threshold_strategy = st.selectbox("Eşik (Threshold) Stratejisi", ["Youden Index (Optimum ROC)", "Precision-Recall Balance (F1-Optimal)", "Sabit (0.50)"])

        st.markdown("#### 🧹 Gelişmiş Temizlik ve Model Tercihleri")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            drop_duplicates = st.checkbox("Mükerrer Satırları Otomatik Temizle", value=True)
            enable_outlier = st.checkbox("IQR Outlier Baskılama", value=True)
        with mc2:
            use_smote = st.checkbox("SMOTE (Sınıf Dengeleme)", value=True)
            available_models = ["Random Forest", "Logistic Regression", "Hist Gradient Boosting"]
            if XGB_AVAILABLE: available_models.append("XGBoost")
            if LGB_AVAILABLE: available_models.append("LightGBM")
            selected_models = st.multiselect("Yarıştırılacak Modeller", available_models, default=["Random Forest", "Logistic Regression"])
        with mc3:
            optuna_trials = st.slider("Optuna Deneme Sayısı (0=Kapalı)", 0, 30, 5)
            test_size = st.slider("Test Seti Oranı", 0.1, 0.4, 0.2)

        ordinal_mappings = {
            "egitim_durumu": ['Lise', 'Ön Lisans', 'Lisans', 'Yüksek Lisans', 'Doktora'],
            "kademe": ['Uzman Yardımcısı', 'Uzman', 'Kıdemli Uzman', 'Yönetici', 'Direktör']
        }

        if st.button("🚀 Veriyi Temizle, Modelleri Yarıştır ve Eğit", type="primary"):
            config = {
                "target_column": target_column,
                "drop_columns": drop_columns,
                "leakage_columns": leakage_columns,
                "drop_duplicates": drop_duplicates,
                "imputation_strategy": imputation_strategy,
                "scaler_type": scaler_type,
                "threshold_strategy": threshold_strategy,
                "selected_models": selected_models,
                "use_smote": use_smote,
                "enable_outlier": enable_outlier,
                "optuna_trials": optuna_trials,
                "test_size": test_size,
                "enable_date_fe": True,
                "ordinal_mappings": ordinal_mappings,
                "model_output_path": model_output_path
            }

            with st.spinner("Veri temizleniyor, standartlaştırılıyor ve modeller eğitiliyor..."):
                temp_path = "enterprise_clean_temp.xlsx"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_train_file.getbuffer())
                
                bundle = train_enterprise_pipeline(temp_path, config)
                st.success(f"Başarılı! 🏆 Kazanan Model: {bundle['best_model_name']} (CV AUC: {bundle['best_cv_auc']:.4f})")

    if os.path.exists(model_output_path):
        st.markdown("---")
        st.header("📁 3. Toplu Tahminleme ve Sonuç Paneli")
        bundle = joblib.load(model_output_path)
        model = bundle["pipeline"]
        threshold = bundle["threshold"]
        expected_cols = bundle["columns"]

        uploaded_test_file = st.file_uploader("Tahmin Edilecek Yeni Veri Dosyası (.xlsx, .csv)", type=["xlsx", "csv"], key="test_u")

        if uploaded_test_file is not None:
            user_df = pd.read_excel(uploaded_test_file) if uploaded_test_file.name.endswith('.xlsx') else pd.read_csv(uploaded_test_file)
            
            if st.button("🔮 Tahminleri ve Risk Skorlarını Hesapla"):
                aligned_df = enforce_model_schema(user_df, expected_cols)
                probas = model.predict_proba(aligned_df)[:, 1]
                preds = (probas >= threshold).astype(int)

                user_df["Risk Skoru (%)"] = np.round(probas * 100, 2)
                user_df["Tahmin Edilen Durum"] = np.where(preds == 1, "Yüksek Risk", "Normal")

                st.success("Tahminler başarıyla tamamlandı!")
                st.dataframe(user_df, use_container_width=True)

                csv = user_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Sonuçları İndir (CSV)", data=csv, file_name="temiz_veri_tahmin_sonuclari.csv", mime="text/csv")

if __name__ == "__main__":
    main()
