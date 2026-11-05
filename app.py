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

# İmbanced-learn SMOTE kontrolü
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    from sklearn.pipeline import Pipeline as ImbPipeline

# Optuna kontrolü
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ==============================================================================
# 1. ESNEK ÖZELLİK MÜHENDİSLİĞİ VE ŞEMA UYUMLULAŞTIRMA
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
# 2. DİNAMİK EĞİTİM VE OPTUNA MOTORU
# ==============================================================================
def train_pipeline(file_path, params):
    ext = os.path.splitext(file_path)[-1].lower()
    df = pd.read_excel(file_path) if ext in ['.xlsx', '.xls'] else pd.read_csv(file_path)
    
    # Kullanıcının interaktif seçtiği drop kolonları temizle
    to_remove = set(params["drop_columns"] + params["leakage_columns"])
    df.drop(columns=[c for c in to_remove if c in df.columns], inplace=True)

    target = params["target_column"]
    X, y = df.drop(columns=[target]), df[target]

    # Metinsel hedefleri güvenli bir şekilde 0 ve 1'e kodlama
    if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype('category').cat.codes
    
    y = np.where(y == y.min(), 0, 1)
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=42, stratify=y
    )

    fe = UltimatePreprocessor(params["ordinal_mappings"])
    X_train_fe = fe.fit_transform(X_train)

    num_cols = X_train_fe.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()
    bin_cols = [c for c in X_train_fe.columns if X_train_fe[c].nunique() == 2]
    pure_num = [c for c in num_cols if c not in bin_cols]
    ord_cols = [c for c in params["ordinal_mappings"].keys() if c in X_train_fe.columns]
    nom_cols = [c for c in cat_cols if c not in ord_cols]

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline_Step([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), pure_num),
        ('ord', Pipeline_Step([('imp', SimpleImputer(strategy='most_frequent')), ('ord', OrdinalEncoder(categories=[params["ordinal_mappings"][c] for c in ord_cols], handle_unknown='use_encoded_value', unknown_value=-1))]), ord_cols),
        ('nom', Pipeline_Step([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), nom_cols),
        ('bin', Pipeline_Step([('imp', SimpleImputer(strategy='most_frequent')), ('bin', OrdinalEncoder())]), bin_cols)
    ], remainder='drop')

    # Optuna Optimizasyonu
    if OPTUNA_AVAILABLE and params["optuna_trials"] > 0:
        def objective(trial):
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 4, 15),
                random_state=42, class_weight='balanced', n_jobs=-1
            )
            steps = [('fe', fe), ('prep', preprocessor)]
            if params["use_smote"] and IMBLEARN_AVAILABLE:
                steps.append(('smote', SMOTE(random_state=42)))
            steps.append(('clf', clf))
            p = ImbPipeline(steps)
            
            p.fit(X_train, y_train)
            preds = p.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=params["optuna_trials"])
        best_params = study.best_params
        classifier = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced', n_jobs=-1)
    else:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

    steps = [('fe', fe), ('preprocessor', preprocessor)]
    if params["use_smote"] and IMBLEARN_AVAILABLE:
        steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('classifier', classifier))
    
    final_pipeline = ImbPipeline(steps)
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
    joblib.dump(bundle, params["model_output_path"])
    return bundle

from sklearn.pipeline import Pipeline as Pipeline_Step

# ==============================================================================
# 3. STREAMLIT İNTERAKTİF WEB ARAYÜZÜ
# ==============================================================================
st.set_page_config(page_title="Esnek Tak-Çalıştır ML Portalı", layout="wide")

def main():
    st.title("🤖 Esnek ve İnteraktif Tak-Çalıştır ML Portalı")
    st.markdown("Verinizi yükleyin; hedef sütunu, atılacak kolonları ve eğitim tercihlerinizi arayüz üzerinden dinamik olarak belirleyin.")

    model_output_path = "flexible_engine_bundle.joblib"

    with st.sidebar:
        st.header("⚙️ 1. Veri Yükleme")
        uploaded_train_file = st.file_uploader("Eğitim Verisi Yükle (.xlsx, .csv)", type=["xlsx", "csv"])

    if uploaded_train_file is not None:
        ext = os.path.splitext(uploaded_train_file.name)[-1].lower()
        df_preview = pd.read_excel(uploaded_train_file) if ext in ['.xlsx', '.xls'] else pd.read_csv(uploaded_train_file)
        
        st.success(f"Veri başarıyla yüklendi! Boyut: {df_preview.shape[0]} satır, {df_preview.shape[1]} kolon.")
        
        with st.expander("🔍 Yüklenen Verinin İlk 5 Satırı"):
            st.dataframe(df_preview.head())

        st.markdown("---")
        st.header("🎛️ 2. Esnek Konfigürasyon ve Tercihler")
        
        col1, col2 = st.columns(2)
        with col1:
            target_column = st.selectbox("Hedef (Target) Sütun Seçin", df_preview.columns)
            drop_columns = st.multiselect("Atılacak Kolonlar (ID vb.)", df_preview.columns, default=[])
        with col2:
            potential_leakage = [c for c in df_preview.columns if 'tarih' in c.lower() or 'date' in c.lower() or 'cikis' in c.lower()]
            leakage_columns = st.multiselect("Sızıntı (Leakage) Olabilecek Kolonlar", df_preview.columns, default=potential_leakage)
            use_smote = st.checkbox("SMOTE (Sınıf Dengeleme) Uygulansın mı?", value=True)

        opt_trials = st.slider("Optuna Hiperparametre Deneme Sayısı (0 = Kapalı / Hızlı)", 0, 20, 5)
        test_size = st.slider("Test Verisi Oranı", 0.1, 0.4, 0.2)

        # Sıralı kategorik tanımları (isteğe bağlı esneklik)
        ordinal_mappings = {
            "egitim_durumu": ['Lise', 'Ön Lisans', 'Lisans', 'Yüksek Lisans', 'Doktora'],
            "kademe": ['Uzman Yardımcısı', 'Uzman', 'Kıdemli Uzman', 'Yönetici', 'Direktör']
        }

        if st.button("🚀 Modeli Bu Ayarlarla Eğit ve Kaydet", type="primary"):
            params = {
                "target_column": target_column,
                "drop_columns": drop_columns,
                "leakage_columns": leakage_columns,
                "use_smote": use_smote,
                "optuna_trials": opt_trials,
                "test_size": test_size,
                "ordinal_mappings": ordinal_mappings,
                "model_output_path": model_output_path
            }
            
            with st.spinner("Model esnek ayarlara göre eğitiliyor ve optimize ediliyor..."):
                # Geçici kaydet ve eğit
                temp_path = "temp_dynamic_train.xlsx"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_train_file.getbuffer())
                
                train_pipeline(temp_path, params)
                st.success("Model başarıyla eğitildi ve kaydedildi!")

    # Model varsa tahmin sekmesini aç
    if os.path.exists(model_output_path):
        st.markdown("---")
        st.header("📁 3. Toplu Tahminleme ve Sonuç Ekranı")
        bundle = joblib.load(model_output_path)
        model = bundle["pipeline"]
        threshold = bundle["threshold"]
        expected_cols = bundle["columns"]

        uploaded_test_file = st.file_uploader("Tahmin edilecek yeni veri dosyasını yükleyin (.xlsx, .csv)", type=["xlsx", "csv"], key="test_file")

        if uploaded_test_file is not None:
            user_df = pd.read_excel(uploaded_test_file) if uploaded_test_file.name.endswith('.xlsx') else pd.read_csv(uploaded_test_file)
            st.write("Yüklenen Tahmin Verisi Önizlemesi:", user_df.head(3))

            if st.button("🔮 Tahminleri Çalıştır"):
                aligned_df = enforce_model_schema(user_df, expected_cols)
                probas = model.predict_proba(aligned_df)[:, 1]
                preds = (probas >= threshold).astype(int)

                user_df["Risk Skoru (%)"] = np.round(probas * 100, 2)
                user_df["Tahmin Edilen Durum"] = np.where(preds == 1, "Riskli / Pozitif", "Normal / Negatif")

                st.success("Tahminler başarıyla tamamlandı!")
                st.dataframe(user_df, use_container_width=True)

                csv = user_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Sonuçları İndir (CSV)", data=csv, file_name="esnek_ml_tahmin_sonuclari.csv", mime="text/csv")
    else:
        st.info("💡 Devam etmek için lütfen yukarıdan eğitim verinizi yükleyin ve ayarları yapılandırıp modeli eğitin.")

if __name__ == "__main__":
    main()
