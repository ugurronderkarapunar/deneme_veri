import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Optuna kontrolü
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ==============================================================================
# 1. KULLANICI KONTROL MERKEZİ (CONFIG)
# ==============================================================================
CONFIG = {
    "target_column": "calisma_durumu",
    "drop_columns": ["sicil_no"],
    "leakage_columns": ["cikis_tarihi", "cikis_nedeni", "aktif_calisan_mi"],
    "ordinal_mappings": {
        "egitim_durumu": ['Lise', 'Ön Lisans', 'Lisans', 'Yüksek Lisans', 'Doktora'],
        "kademe": ['Uzman Yardımcısı', 'Uzman', 'Kıdemli Uzman', 'Yönetici', 'Direktör']
    },
    "test_size": 0.20,
    "random_state": 42,
    "optuna_trials": 5,  # Hızlı deneme sayısı
    "model_output_path": "ultimate_engine_bundle.joblib"
}

# ==============================================================================
# 2. AKILLI ÖZELLİK MÜHENDİSLİĞİ VE ŞEMA UYUMLULAŞTIRMA (SCHEMA ENFORCEMENT)
# ==============================================================================
class UltimatePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_mappings=None):
        self.ordinal_mappings = ordinal_mappings or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        ref_date = pd.to_datetime('today')
        
        # Tarih kolonlarını otomatik yıla çevirir
        date_cols = [col for col in X_out.columns if 'tarih' in col.lower() or 'date' in col.lower()]
        for col in date_cols:
            try:
                X_out[col] = pd.to_datetime(X_out[col], errors='coerce')
                new_name = col.replace('_tarihi', '_yil').replace('_date', '_years')
                X_out[new_name] = (ref_date - X_out[col]).dt.days / 365.25
                X_out.drop(columns=[col], inplace=True)
            except:
                pass

        # IQR ile Outlier Baskılama
        numeric_cols = X_out.select_dtypes(include=["number"]).columns
        X_out[numeric_cols] = X_out[numeric_cols].replace([np.inf, -np.inf], np.nan)
        for col in numeric_cols:
            q1, q3 = X_out[col].quantile(0.01), X_out[col].quantile(0.99)
            X_out[col] = X_out[col].clip(lower=q1, upper=q3)

        return X_out

def enforce_model_schema(df_user, expected_columns):
    """
    Kullanıcı farklı formatta Excel yüklese bile sistemi patlatmaz; 
    eksik kolonlara NaN basar, fazlalıkları eler, sırayı eşitler.
    """
    df_aligned = pd.DataFrame()
    for col in expected_columns:
        if col in df_user.columns:
            df_aligned[col] = df_user[col]
        else:
            df_aligned[col] = np.nan
    return df_aligned

# ==============================================================================
# 3. OTOMATİK EĞİTİM VE OPTUNA MOTORU
# ==============================================================================
def train_pipeline(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    df = pd.read_excel(file_path) if ext in ['.xlsx', '.xls'] else pd.read_csv(file_path)
    
    to_remove = set(CONFIG["drop_columns"] + CONFIG["leakage_columns"])
    df.drop(columns=[c for c in to_remove if c in df.columns], inplace=True)

    target = CONFIG["target_column"]
    X, y = df.drop(columns=[target]), df[target]
    if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=y
    )

    fe = UltimatePreprocessor(CONFIG["ordinal_mappings"])
    X_train_fe = fe.fit_transform(X_train)

    num_cols = X_train_fe.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()
    bin_cols = [c for c in X_train_fe.columns if X_train_fe[c].nunique() == 2]
    pure_num = [c for c in num_cols if c not in bin_cols]
    ord_cols = [c for c in CONFIG["ordinal_mappings"].keys() if c in X_train_fe.columns]
    nom_cols = [c for c in cat_cols if c not in ord_cols]

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline_Step([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), pure_num),
        ('ord', Pipeline_Step([('imp', SimpleImputer(strategy='most_frequent')), ('ord', OrdinalEncoder(categories=[CONFIG["ordinal_mappings"][c] for c in ord_cols], handle_unknown='use_encoded_value', unknown_value=-1))]), ord_cols),
        ('nom', Pipeline_Step([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), nom_cols),
        ('bin', Pipeline_Step([('imp', SimpleImputer(strategy='most_frequent')), ('bin', OrdinalEncoder())]), bin_cols)
    ], remainder='drop')

    # Optuna Optimizasyonu
    if OPTUNA_AVAILABLE:
        def objective(trial):
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 4, 15),
                random_state=CONFIG["random_state"], class_weight='balanced', n_jobs=-1
            )
            from sklearn.pipeline import Pipeline as SkPipeline
            p = SkPipeline([('fe', fe), ('prep', preprocessor), ('clf', clf)])
            p.fit(X_train, y_train)
            preds = p.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=CONFIG["optuna_trials"])
        best_params = study.best_params
        classifier = RandomForestClassifier(**best_params, random_state=CONFIG["random_state"], class_weight='balanced', n_jobs=-1)
    else:
        classifier = RandomForestClassifier(n_estimators=100, random_state=CONFIG["random_state"], class_weight='balanced')

    from sklearn.pipeline import Pipeline as SkPipeline
    final_pipeline = SkPipeline([
        ('fe', fe),
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    final_pipeline.fit(X_train, y_train)

    # Optimum Eşik Tespiti (Youden Index)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    bundle = {
        "pipeline": final_pipeline,
        "threshold": optimal_threshold,
        "columns": X.columns.tolist()
    }
    joblib.dump(bundle, CONFIG["model_output_path"])
    return bundle

from sklearn.pipeline import Pipeline as Pipeline_Step

# ==============================================================================
# 4. STREAMLIT WEB ARAYÜZÜ (KULLANICI PANELİ)
# ==============================================================================
st.set_page_config(page_title="Universal ML Portalı", layout="wide")

def main():
    st.title("🤖 Tak-Çalıştır Evrensel ML ve Tahmin Portalı")
    
    with st.sidebar:
        st.header("⚙️ Model Yönetimi")
        uploaded_train_file = st.file_uploader("Eğitim Verisi Yükle (.xlsx, .csv)", type=["xlsx", "csv"])
        if uploaded_train_file is not None:
            if st.button("Modeli Eğit ve Kaydet"):
                with st.spinner("Model eğitiliyor..."):
                    # Geçici olarak kaydet
                    with open("temp_train.xlsx", "wb") as f:
                        f.write(uploaded_train_file.getbuffer())
                    train_pipeline("temp_train.xlsx")
                    st.success("Model başarıyla eğitildi!")

    if not os.path.exists(CONFIG["model_output_path"]):
        st.warning("⚠️ Lütfen sol menüden eğitim verisi yükleyip modeli eğitin.")
        return

    bundle = joblib.load(CONFIG["model_output_path"])
    model = bundle["pipeline"]
    threshold = bundle["threshold"]
    expected_cols = bundle["columns"]

    st.markdown("### 📁 Toplu Tahminleme (Excel / CSV)")
    uploaded_test_file = st.file_uploader("Tahmin edilecek veriyi yükleyin (Kolonlar farklı olsa bile uyarlanır)", type=["xlsx", "csv"], key="test")

    if uploaded_test_file is not None:
        user_df = pd.read_excel(uploaded_test_file) if uploaded_test_file.name.endswith('.xlsx') else pd.read_csv(uploaded_test_file)
        st.write("Yüklenen Ham Veri Önizlemesi:", user_df.head(3))

        if st.button("Tahminleri Çalıştır"):
            # Şema Uyumlandırma (Schema Alignment)
            aligned_df = enforce_model_schema(user_df, expected_cols)
            
            probas = model.predict_proba(aligned_df)[:, 1]
            preds = (probas >= threshold).astype(int)

            user_df["Risk Skoru (%)"] = np.round(probas * 100, 2)
            user_df["Tahmin Edilen Durum"] = np.where(preds == 1, "Riskli / Pozitif", "Normal / Negatif")

            st.success("Tahminler başarıyla tamamlandı!")
            st.dataframe(user_df, use_container_width=True)

            csv = user_df.to_csv(index=False).encode('utf-8')
            st.download_button("Sonuçları İndir (CSV)", data=csv, file_name="ml_tahmin_sonuclari.csv", mime="text/csv")

if __name__ == "__main__":
    main()
