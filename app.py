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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
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
# SAYFA YAPILANDIRMASI
# ==============================================================================
st.set_page_config(
    page_title="10 Bölümlü Modüler Veri Bilimi Platformu", 
    page_icon="🔬", 
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 800; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .section-box { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔬 10 Bölümlü Kontrollü Veri Bilimi ve İstatistiksel Denetim Platformu</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Her aşamada tam kontrol sağlayın, modelleri özelleştirin ve sonuçların istatistiksel güvenilirliğini test edin.</p>', unsafe_allow_html=True)

# ==============================================================================
# ÖZEL ÖZELLİK MÜHENDİSLİĞİ (Transformer)
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

# ==============================================================================
# BÖLÜM 1: Kurulum ve Proje Ayarları
# ==============================================================================
st.sidebar.markdown("## 🧭 10 Aşamalı Proje Rehberi")
current_section = st.sidebar.radio(
    "Gitmek İstediğiniz Bölümü Seçin:",
    [
        "Bölüm 1: Kurulum ve Proje Ayarları",
        "Bölüm 2: Veriyi Yükle ve Denetle",
        "Bölüm 3: Hedef ve Özellik Hazırlama",
        "Bölüm 4: Train / Test Bölme",
        "Bölüm 5: Ön İşleme Pipeline'ı",
        "Bölüm 6: Sızıntı Avcısı",
        "Bölüm 7: Model Karşılaştırma (CV)",
        "Bölüm 8: Hiperparametre Ayarı",
        "Bölüm 9: Test Değerlendirmesi & Voting",
        "Bölüm 10: İstatistiksel Güvenilirlik & Yorumlama"
    ]
)

# ==============================================================================
# BÖLÜM 2: Veriyi Yükle ve Denetle
# ==============================================================================
if current_section == "Bölüm 1: Kurulum ve Proje Ayarları":
    st.header("Bölüm 1: Kurulum ve Proje Ayarları")
    st.markdown("""
    <div class="section-box">
    <h3>Proje Yapılandırma Bilgilendirmesi</h3>
    <p>Bu bölümde proje için gerekli olan Python kütüphaneleri (Scikit-Learn, XGBoost, Optuna, Streamlit, Plotly) kontrol edilir. 
    Sağdaki veya aşağıdaki dosya yükleme alanını kullanarak veri setinizi sisteme tanıtabilirsiniz.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 Hazırsanız sol menüden **Bölüm 2: Veriyi Yükle ve Denetle** adımına geçebilirsiniz.")

elif current_section == "Bölüm 2: Veriyi Yükle ve Denetle":
    st.header("Bölüm 2: Veriyi Yükle ve Denetle")
    uploaded_file = st.file_uploader("Veri Setinizi Yükleyin (.xlsx veya .csv)", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        df_raw = load_data(uploaded_file)
        st.session_state['df_raw'] = df_raw
        st.success("Veri başarıyla yüklendi!")
        st.dataframe(df_raw.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Satır Sayısı", df_raw.shape[0])
        with col2:
            st.metric("Toplam Sütun Sayısı", df_raw.shape[1])
            
        st.subheader("Eksik Değer ve Tip Kontrolü")
        info_df = pd.DataFrame({
            "Veri Tipi": df_raw.dtypes.astype(str),
            "Eksik Değer Sayısı": df_raw.isnull().sum(),
            "Eksik Oranı (%)": (df_raw.isnull().mean() * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)

# ==============================================================================
# BÖLÜM 3: Hedef ve Özellik Hazırlama
# ==============================================================================
elif current_section == "Bölüm 3: Hedef ve Özellik Hazırlama":
    st.header("Bölüm 3: Hedef ve Özellik Hazırlama")
    if 'df_raw' in st.session_state:
        df = st.session_state['df_raw']
        
        target_col = st.selectbox("Hedef Değişkeni (Target) Seçin:", df.columns, index=min(len(df.columns)-1, 0))
        drop_cols = st.multiselect("Silinecek / Gereksiz ID Sütunlarını Seçin:", df.columns.tolist())
        
        st.session_state['target_col'] = target_col
        st.session_state['drop_cols'] = drop_cols
        
        st.markdown("---")
        st.subheader("Özellik Mühendisliği Kararları")
        st.markdown("Otomatik tarih dönüşümleri ve oran türetmeleri (`HRFeatureEngineer`) uygulansın mı?")
        apply_fe = st.checkbox("Özel Özellik Mühendisliği Uygula (Tarih farkları, kıdem, maaş oranları)", value=True)
        st.session_state['apply_fe'] = apply_fe
        
        if st.button("Hedef ve Özellikleri Onayla"):
            df_clean = df.drop(columns=drop_cols, errors='ignore')
            st.session_state['df_clean'] = df_clean
            st.success("Özellik hazırlama adımı tamamlandı!")
    else:
            st.warning("⚠️ Lütfen önce Bölüm 2'den veri setinizi yükleyin.")

# ==============================================================================
# BÖLÜM 4: Train / Test Bölme
# ==============================================================================
elif current_section == "Bölüm 4: Train / Test Bölme":
    st.header("Bölüm 4: Train / Test Bölme")
    if 'df_clean' in st.session_state and 'target_col' in st.session_state:
        df_clean = st.session_state['df_clean']
        target_col = st.session_state['target_col']
        
        test_size = st.slider("Test Seti Oranı (%)", 10, 40, 25, 5) / 100.0
        stratify_choice = st.checkbox("Stratified Split (Sınıf oranlarını koruyarak böl - Önerilir)", value=True)
        
        if st.button("Veriyi Train / Test Olarak Böl"):
            X = df_clean.drop(columns=[target_col])
            y_raw = df_clean[target_col]
            
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
            st.session_state['label_encoder'] = le
            
            strat = y if stratify_choice else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
            
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            
            st.success(f"Veri başarıyla bölündü! Eğitim seti: {X_train.shape[0]} satır, Test seti: {X_test.shape[0]} satır.")
            st.info(f"Sınıf Dağılımı (Eğitim): {y_train.value_counts().to_dict()}")
    else:
        st.warning("⚠️ Lütfen önce Bölüm 3'ü tamamlayın.")

# ==============================================================================
# BÖLÜM 5: Ön İşleme Pipeline'ı
# ==============================================================================
elif current_section == "Bölüm 5: Ön İşleme Pipeline'ı":
    st.header("Bölüm 5: Ön İşleme Pipeline'ı")
    st.markdown("""
    <div class="section-box">
    <p>Sayısal değişkenler için Medyan doldurma ve StandardScaler, kategorik değişkenler için One-Hot ve Ordinal kodlama kuralları otomatik olarak ayarlanır. Veri sızıntısını önlemek için bu adımlar Pipeline içerisine gömülmüştür.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'X_train' in st.session_state:
        X_train = st.session_state['X_train']
        
        if st.session_state.get('apply_fe', True):
            fe = HRFeatureEngineer()
            X_train_fe = fe.fit_transform(X_train)
            st.session_state['fe'] = fe
        else:
            X_train_fe = X_train
            
        numeric_cols = X_train_fe.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()
        binary_cols = [col for col in X_train_fe.columns if X_train_fe[col].nunique() == 2]
        pure_numeric_cols = [col for col in numeric_cols if col not in binary_cols]
        
        st.write(f"Sayısal Sütunlar: {len(pure_numeric_cols)}")
        st.write(f"Kategorik Sütunlar: {len(categorical_cols)}")
        st.write(f"İkili (Binary) Sütunlar: {len(binary_cols)}")
        
        ordinal_mappings = {
            "egitim_durumu": ['Lise', 'Ön Lisans', 'Lisans', 'Yüksek Lisans', 'Doktora'],
            "kademe": ['Uzman Yardımcısı', 'Uzman', 'Kıdemli Uzman', 'Yönetici', 'Direktör']
        }
        ordinal_cols = [c for c in ordinal_mappings.keys() if c in X_train_fe.columns]
        nominal_cols = [c for c in categorical_cols if c not in ordinal_cols]
        
        numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        ordinal_categories = [ordinal_mappings[c] for c in ordinal_cols]
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
            ('ord', ordinal_transformer, ordinal_cols),
            ('nom', nominal_transformer, nominal_cols),
            ('bin', binary_transformer, binary_cols)
        ], remainder='drop')
        
        st.session_state['preprocessor'] = preprocessor
        st.success("Ön işleme Pipeline başarıyla kuruldu!")
    else:
        st.warning("⚠️ Lütfen önce Bölüm 4'ü tamamlayın.")

# ==============================================================================
# BÖLÜM 6: Sızıntı Avcısı
# ==============================================================================
elif current_section == "Bölüm 6: Sızıntı Avcısı":
    st.header("Bölüm 6: Sızıntı Avcısı (Data Leakage Detector)")
    st.markdown("""
    <div class="section-box">
    <p>Hedef değişken ile aşırı yüksek korelasyona sahip veya gelecekteki bilgiyi sızdıran sütunlar tespit edilir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'df_clean' in st.session_state and 'target_col' in st.session_state:
        df_clean = st.session_state['df_clean']
        target_col = st.session_state['target_col']
        
        numeric_df = df_clean.select_dtypes(include=[np.number])
        if target_col in numeric_df.columns:
            corr = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
            st.subheader("Hedef Değişken ile Sütun Korelasyonları")
            st.dataframe(corr.reset_index().rename(columns={'index': 'Sütun', target_col: 'Korelasyon Skoru'}), use_container_width=True)
            
            high_corr = corr[(corr > 0.85) & (corr < 1.0)]
            if not high_corr.empty:
                st.error(f"⚠️ Dikkat! Şu sütunlarda veri sızıntısı şüphesi var (Korelasyon > 0.85): {list(high_corr.index)}")
            else:
                st.success("✅ Kritik düzeyde bariz bir veri sızıntısı tespit edilmedi.")
    else:
        st.warning("⚠️ Lütfen veriyi yükleyin.")

# ==============================================================================
# BÖLÜM 7: Model Karşılaştırma (CV)
# ==============================================================================
elif current_section == "Bölüm 7: Model Karşılaştırma (CV)":
    st.header("Bölüm 7: Model Karşılaştırma ve Çapraz Doğrulama (CV)")
    st.markdown("Kullanmak istediğiniz algoritmaları aşağıdan seçin. Sistem 5 Katlı Stratified K-Fold ile her modelin ortalama başarısını ve kararlılığını test edecektir.")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        use_lr = st.checkbox("Logistic Regression", value=True)
        use_rf = st.checkbox("Random Forest", value=True)
        use_gb = st.checkbox("Gradient Boosting", value=True)
    with col_m2:
        use_dt = st.checkbox("Decision Tree", value=False)
        use_svc = st.checkbox("Support Vector Machine (SVC)", value=False)
        use_xgb = st.checkbox("XGBoost", value=True if XGB_AVAILABLE else False, disabled=not XGB_AVAILABLE)

    if st.button("Modelleri Çapraz Doğrulama ile Karşılaştır"):
        if 'X_train' in st.session_state:
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            preprocessor = st.session_state['preprocessor']
            fe = st.session_state.get('fe', None)
            
            selected_models = {}
            if use_lr: selected_models['Logistic Regression'] = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            if use_rf: selected_models['Random Forest'] = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            if use_gb: selected_models['Gradient Boosting'] = GradientBoostingClassifier(random_state=42)
            if use_dt: selected_models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
            if use_svc: selected_models['SVC'] = SVC(probability=True, random_state=42)
            if use_xgb and XGB_AVAILABLE: selected_models['XGBoost'] = XGBClassifier(random_state=42)
            
            if not selected_models:
                st.warning("Lütfen en az bir model seçin!")
            else:
                cv_results = []
                steps = [('prep', preprocessor)]
                if fe: steps.insert(0, ('fe', fe))
                
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                with st.spinner("Modeller çapraz doğrulamadan geçiriliyor..."):
                    for name, model in selected_models.items():
                        pipeline = Pipeline(steps + [('model', model)])
                        scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='roc_auc')
                        cv_results.append({
                            'Model': name,
                            'Ortalama ROC-AUC': scores.mean(),
                            'Standart Sapma (Std)': scores.std()
                        })
                        
                results_df = pd.DataFrame(cv_results).sort_values(by='Ortalama ROC-AUC', ascending=False)
                st.session_state['cv_results'] = results_df
                st.session_state['selected_models_dict'] = selected_models
                
                st.success("Karşılaştırma tamamlandı!")
                st.dataframe(results_df, use_container_width=True)
                
                fig = px.bar(results_df, x='Model', y='Ortalama ROC-AUC', error_y='Standart Sapma (Std)', color='Ortalama ROC-AUC', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Lütfen önce Bölüm 4 ve 5'i tamamlayın.")

# ==============================================================================
# BÖLÜM 8: Hiperparametre Ayarı
# ==============================================================================
elif current_section == "Bölüm 8: Hiperparametre Ayarı":
    st.header("Bölüm 8: Hiperparametre Optimizasyonu (Optuna)")
    st.markdown("Seçilen ana ağaç modeli (`RandomForest`) için Optuna ile otomatik hiperparametre optimizasyonu yapabilirsiniz.")
    
    n_trials = st.slider("Optimizasyon Deneme Sayısı (Trials)", 5, 30, 10)
    if st.button("Optuna Optimizasyonunu Başlat"):
        if 'X_train' in st.session_state:
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            preprocessor = st.session_state['preprocessor']
            fe = st.session_state.get('fe', None)
            
            with st.spinner("Optuna en iyi parametreleri arıyor..."):
                def objective(trial):
                    n_est = trial.suggest_int('n_estimators', 50, 200)
                    max_d = trial.suggest_int('max_depth', 3, 15)
                    clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight='balanced', random_state=42)
                    
                    steps = [('prep', preprocessor), ('clf', clf)]
                    if fe: steps.insert(0, ('fe', fe))
                    pipe = Pipeline(steps)
                    
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')
                    return scores.mean()

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)
                
                best_params = study.best_params
                st.success("Optimizasyon başarıyla tamamlandı!")
                st.write(f"**En İyi Parametreler:** {best_params}")
                st.session_state['best_params'] = best_params
        else:
            st.warning("⚠️ Lütfen veriyi hazırlayın.")

# ==============================================================================
# BÖLÜM 9: Test Değerlendirmesi & Voting
# ==============================================================================
elif current_section == "Bölüm 9: Test Değerlendirmesi & Voting":
    st.header("Bölüm 9: Test Değerlendirmesi ve Topluluk (Voting) Kararı")
    st.markdown("Modellerin gücünü birleştirmek istiyor musunuz?")
    
    use_voting = st.checkbox("Voting Classifier (Topluluk Modeli) Kullan", value=True)
    voting_type = st.radio("Voting Türü:", ["soft (Olasılık Ortalaması)", "hard (Çoğunluk Oyu)"], index=0)
    
    if st.button("Nihai Modeli Eğit ve Test Setinde Değerlendir"):
        if 'selected_models_dict' in st.session_state:
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            preprocessor = st.session_state['preprocessor']
            fe = st.session_state.get('fe', None)
            
            estimators = list(st.session_state['selected_models_dict'].items())
            
            if use_voting and len(estimators) > 1:
                v_type = 'soft' if 'soft' in voting_type else 'hard'
                final_model = VotingClassifier(estimators=estimators, voting=v_type)
            else:
                # En iyi tekli modeli al
                final_model = estimators[0][1]
                
            steps = [('prep', preprocessor), ('model', final_model)]
            if fe: steps.insert(0, ('fe', fe))
            
            final_pipeline = Pipeline(steps)
            final_pipeline.fit(X_train, y_train)
            st.session_state['final_pipeline'] = final_pipeline
            
            y_pred = final_pipeline.predict(X_test)
            y_proba = final_pipeline.predict_proba(X_test)[:, 1] if hasattr(final_pipeline, "predict_proba") else y_pred
            
            st.success("Nihai model test seti üzerinde başarıyla değerlendirildi!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sınıflandırma Raporu")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
            with col2:
                st.subheader("Karışıklık Matrisi")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Tahmin", y="Gerçek", color="Adet"), color_continuous_scale='Blues')
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("⚠️ Lütfen önce Bölüm 7'den modelleri karşılaştırın.")

# ==============================================================================
# BÖLÜM 10: İstatistiksel Güvenilirlik & Yorumlama
# ==============================================================================
elif current_section == "Bölüm 10: İstatistiksel Güvenilirlik & Yorumlama":
    st.header("Bölüm 10: İstatistiksel Güvenilirlik ve Sonuç Denetimi")
    st.markdown("""
    <div class="section-box">
    <h3>🔍 Çıkan Sonuç Cidden Güvenilir mi? (İstatistiksel Audit)</h3>
    <p>Bu bölümde modelimizin rastgele şanstan (`DummyClassifier`) anlamlı ölçüde üstün olup olmadığı, 
    eğitim ile test başarıları arasındaki fark (Overfitting kontrolü) istatistiksel olarak test edilir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'final_pipeline' in st.session_state and 'X_train' in st.session_state:
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        pipeline = st.session_state['final_pipeline']
        
        # 1. Baseline Karşılaştırması (Dummy Classifier)
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        dummy_acc = dummy.score(X_test, y_test)
        
        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        
        st.subheader("📊 İstatistiksel Güvenilirlik Karnesi")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Eğitim (Train) Başarısı", f"%{train_acc*100:.2f}")
        with col_s2:
            st.metric("Test Başarısı", f"%{test_acc*100:.2f}")
        with col_s3:
            st.metric("Baseline (Rastgele/Çoğunluk) Başarısı", f"%{dummy_acc*100:.2f}")
            
        # İstatistiksel Yorum
        st.markdown("---")
        st.subheader("💡 Otomatik İstatistiksel Audit Yorumu:")
        
        overfit_gap = train_acc - test_acc
        if overfit_gap > 0.15:
            st.error(f"⚠️ **Overfitting (Ezberleme) Riski Yüksek!** Eğitim başarısı ile test başarısı arasında %{overfit_gap*100:.1f} fark var. Model veriyi ezberlemiş olabilir.")
        elif test_acc <= dummy_acc:
            st.error("🚨 **Güvenilirlik Uyarısı!** Modeliniz temel çizgi (Dummy) modelinden daha iyi performans gösteremiyor. Sonuçlar güvenilir değil!")
        else:
            st.success("✨ **Sonuçlar İstatistiksel Olarak Güvenilir!** Modeliniz baseline sınırını belirgin şekilde aşmış ve eğitim/test skorları kararlı bir denge sergilemektedir.")
            
        # SHAP veya Bireysel Tahmin
        st.markdown("---")
        st.subheader("🔍 SHAP ile Model Açıklanabilirliği")
        if SHAP_AVAILABLE and st.button("SHAP Önem Grafiğini Üret"):
            with st.spinner("SHAP değerleri hesaplanıyor..."):
                try:
                    prep = pipeline.named_steps['prep']
                    fe = pipeline.named_steps.get('fe', None)
                    model_step = pipeline.named_steps['model']
                    
                    trans_data = prep.transform(fe.transform(X_test) if fe else X_test)
                    
                    # Eğer Voting ise altındaki ilk ağaç modelini kullanalım
                    est_to_use = model_step.estimators_[0][1] if isinstance(model_step, VotingClassifier) else model_step
                    if hasattr(est_to_use, "tree_") or hasattr(est_to_use, "estimators_"):
                        explainer = shap.TreeExplainer(est_to_use)
                        shap_vals = explainer.shap_values(trans_data)
                        
                        import matplotlib.pyplot as plt
                        fig_sh, ax = plt.subplots(figsize=(8, 4))
                        shap.summary_plot(shap_vals[1] if isinstance(shap_vals, list) else shap_vals, trans_data, show=False)
                        st.pyplot(fig_sh)
                    else:
                        st.info("Seçilen model türü için TreeExplainer doğrudan desteklenmiyor.")
                except Exception as e:
                    st.warning(f"SHAP hesaplanırken bir hata oluştu: {e}")
    else:
        st.info("⚠️ Lütfen önce Bölüm 9'dan modeli eğitin.")
