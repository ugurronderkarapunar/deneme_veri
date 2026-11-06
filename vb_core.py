"""
vb_core.py - Veri Bilimi Platformu çekirdeği (arayüzden bağımsız).
Sınıflandırma (ikili) ve regresyon için: temizleme, özellik üretme, ön işleme,
model karşılaştırma, Optuna ayarı, Voting/Stacking, değerlendirme ve yorum üretimi.
Pickle uyumluluğu için sınıflar (FeatureBuilder vb.) bu modülde durur.
"""
import io
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder)
from sklearn.model_selection import (train_test_split, GroupShuffleSplit, StratifiedKFold, StratifiedGroupKFold,
                                     KFold, GroupKFold, TimeSeriesSplit, cross_validate, cross_val_score,
                                     cross_val_predict, learning_curve, permutation_test_score)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
                              ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor, VotingClassifier,
                              VotingRegressor, StackingClassifier, StackingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve, mean_squared_error,
                             mean_absolute_error, r2_score, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance, partial_dependence

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

# ----------------------------------------------------------------------------
# 1) Metin / sayı yardımcıları
# ----------------------------------------------------------------------------
_TR = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocIGUSOC")


def normalize_text(value):
    """'  İstanbul ' -> 'istanbul' (Türkçe harf, büyük/küçük harf, boşluk farklarını giderir)."""
    if pd.isna(value):
        return value
    return re.sub(r"\s+", " ", str(value).translate(_TR).strip().lower())


def parse_tr_number(value):
    """'1.250,50 TL' -> 1250.5 ; '%20' -> 20 ; '3.5 gün' -> 3.5 ; çevrilemezse NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = re.sub(r"[^0-9,.\-]", "", str(value))
    if s in ("", "-", ".", ","):
        return np.nan
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_dataframe(df, strip=True, drop_dups=True, numeric_text_cols=(), normalize_cols=(), ranges=None, date_cols=()):
    """Seçilen temizlik adımlarını uygular. Dönüş: (yeni df, açıklama listesi)."""
    d = df.copy()
    log = []
    if strip:
        for c in d.select_dtypes(include="object").columns:
            d[c] = d[c].astype("string").str.strip().astype(object)
        log.append("Metin sütunlarındaki baştaki/sondaki boşluklar silindi.")
    for c in numeric_text_cols:
        before = d[c].isna().sum()
        d[c] = d[c].map(parse_tr_number)
        log.append(f"'{c}' metinden sayıya çevrildi (Türkçe format destekli). Yeni eksik: {int(d[c].isna().sum() - before)}")
    for c in normalize_cols:
        before = d[c].nunique()
        d[c] = d[c].map(normalize_text)
        log.append(f"'{c}' normalize edildi: {before} -> {d[c].nunique()} benzersiz değer.")
    for c, (lo, hi) in (ranges or {}).items():
        if c in d.columns:
            mask = (d[c] < lo) | (d[c] > hi)
            d.loc[mask, c] = np.nan
            log.append(f"'{c}': [{lo}, {hi}] dışındaki {int(mask.sum())} değer eksik (NaN) yapıldı.")
    for c in date_cols:
        d[c] = pd.to_datetime(d[c], errors="coerce", dayfirst=True)
        log.append(f"'{c}' tarih tipine çevrildi (çevrilemeyen: {int(d[c].isna().sum())}).")
    if drop_dups:
        n = len(d)
        d = d.drop_duplicates().reset_index(drop=True)
        log.append(f"Tekrar eden satır silindi: {n - len(d)}")
    return d, log


# ----------------------------------------------------------------------------
# 2) Veri denetimi + otomatik yorum
# ----------------------------------------------------------------------------
LEAK_WORDS = ["sonuc", "iade", "cikis", "ayril", "iptal", "churn", "kapanis", "kapali", "odeme_gecik", "outcome", "result"]


def looks_like_id(name):
    n = name.lower()
    return n.endswith("id") or n.endswith("_no") or "sicil" in n


def looks_like_date(series):
    if str(series.dtype).startswith("datetime"):
        return True
    if series.dtype != "object":
        return False
    sample = series.dropna().astype(str).head(50)
    if len(sample) == 0:
        return False
    return pd.to_datetime(sample, errors="coerce", dayfirst=True).notna().mean() > 0.9


def looks_like_numeric_text(series):
    if series.dtype != "object":
        return False
    sample = series.dropna().astype(str).head(100)
    if len(sample) == 0:
        return False
    return sample.map(parse_tr_number).notna().mean() > 0.9 and sample.str.contains(r"\d").mean() > 0.9


def audit_data(df, target=None, task="classification", drop_cols=(), date_cols=()):
    """Dönüş: dict(table, comments, steps, id_like, constant, dates, numeric_text, leak_names)."""
    comments, steps = [], []
    rows, cols = df.shape
    table = pd.DataFrame({"Tip": df.dtypes.astype(str), "Eksik %": (df.isna().mean() * 100).round(1),
                          "Benzersiz": df.nunique()})
    comments.append(f"{rows} satır ve {cols} sütun var.")

    miss = (df.isna().mean() * 100)
    miss = miss[[c for c in miss.index if c not in drop_cols]]
    heavy = miss[miss > 30].index.tolist()
    mid = miss[(miss > 5) & (miss <= 30)].index.tolist()
    if heavy:
        comments.append(f"%30'dan fazla eksik sütun: {heavy}. Bilgi taşımayabilir; ya da eksikliğin kendisi anlamlıdır "
                        "(örn. çıkış tarihi sadece ayrılanlarda dolu).")
        steps.append("Çok eksik sütunlar için karar ver: at ya da 'eksik mi?' bilgisini taşıdığını düşün.")
    if mid:
        comments.append(f"%5-30 eksik sütunlar: {mid}. Medyanla doldurma dağılımı daraltır; bu yüzden 'eksikti' bayrağı "
                        "ekleme seçeneği (Bölüm 5) açık tutulmalı.")

    constant = [c for c in df.columns if df[c].nunique() <= 1]
    id_like = [c for c in df.columns if c != target and c not in drop_cols and df[c].nunique() == rows
               and (df[c].dtype == "object" or looks_like_id(c))]
    if constant:
        comments.append(f"Sabit (tek değerli) sütunlar: {constant}. Varyansı sıfır, hiçbir şey öğretmez.")
        steps.append(f"Bu sütunları at: {constant}")
    if id_like:
        comments.append(f"Kimlik benzeri sütunlar: {id_like}. Her satırda farklı; model ezberler.")
        steps.append(f"Bu sütunları at: {id_like}")

    dates = [c for c in df.columns if c not in date_cols and c not in drop_cols and c != target and looks_like_date(df[c])]
    if dates:
        comments.append(f"Tarih gibi görünen sütunlar: {dates}. Ham tarih modele girmez; yıl/ay/gün gibi sayılara çevrilmeli.")
        steps.append(f"Bölüm 3'te tarih sütunu olarak işaretle: {dates}. Veri zamana bağlıysa 'time' bölmeyi seç.")

    num_text = [c for c in df.columns if c not in drop_cols and c != target and c not in id_like and not looks_like_id(c)
                and looks_like_numeric_text(df[c]) and not looks_like_date(df[c])]
    if num_text:
        comments.append(f"Sayı gibi görünen ama metin olan sütunlar: {num_text} (örn. '1.250,00 TL'). Bu hâliyle kategorik sayılır.")
        steps.append(f"Bölüm 2'deki temizlikte 'metni sayıya çevir' seçeneğini kullan: {num_text}")

    leak_names = [c for c in df.columns if c != target and c not in drop_cols and any(k in c.lower() for k in LEAK_WORDS)]
    if leak_names:
        comments.append(f"İsmi sonuçla ilgili görünen sütunlar: {leak_names}. Hedef belli olduktan SONRA oluşuyorlarsa "
                        "model geleceği görür (veri sızıntısı).")
        steps.append("Her biri için sor: 'Tahmin anında bu bilgiyi biliyor olur muydum?' Hayırsa at.")

    dups = int(df.duplicated().sum())
    if dups:
        comments.append(f"{dups} tamamen tekrar eden satır var; aynı satır hem train hem test'e düşerse skor şişer.")
        steps.append("Temizlikte 'tekrar eden satırları sil' seçeneğini aç.")

    if target and target in df.columns:
        t = df[target].dropna()
        if task == "classification":
            shares = t.value_counts(normalize=True)
            minority, count = shares.min(), int(t.value_counts().min())
            if shares.shape[0] > 2:
                comments.append(f"Hedefte {shares.shape[0]} sınıf var; bu platform ikili hedefle çalışır. Hedefi 0/1'e indirge ya da regresyon seç.")
            elif minority < 0.10:
                comments.append(f"Hedef ciddi dengesiz (azınlık %{minority*100:.1f}). Hep çoğunluğu tahmin eden model bile "
                                f"%{(1-minority)*100:.0f} accuracy alır; accuracy KULLANMA, PR-AUC/recall bak.")
            elif minority < 0.30:
                comments.append(f"Hedef orta dengesiz (azınlık %{minority*100:.1f}); PR-AUC ve F1'e bak.")
            else:
                comments.append(f"Hedef dengeli sayılır (azınlık %{minority*100:.1f}).")
            per_feature = count / max(cols - 1, 1)
            if per_feature < 10:
                comments.append(f"Azınlık sınıfında {count} örnek, ~{cols-1} sütun var (örnek/sütun={per_feature:.1f}). "
                                "Pratik kural ~10 olay/sütun; altında ezber riski yüksek.")
                steps.append("Sütun sayısını azalt ya da daha basit model kullan.")
        else:
            tn = pd.to_numeric(t, errors="coerce").dropna()
            if len(tn):
                sk = stats.skew(tn)
                q1, q3 = tn.quantile(.25), tn.quantile(.75)
                out = ((tn < q1 - 1.5 * (q3 - q1)) | (tn > q3 + 1.5 * (q3 - q1))).mean() * 100
                comments.append(f"Hedef çarpıklığı {sk:.2f}; ortalama {tn.mean():.1f}, medyan {tn.median():.1f}; IQR uç değer oranı %{out:.1f}.")
                if abs(sk) > 1:
                    comments.append("Çarpıklık |1|'den büyük: uzun kuyruk büyük değerlerdeki hataların RMSE'yi domine etmesine yol açar.")
                    steps.append("Bölüm 1'de 'log dönüşümü (hedef)' seçeneğini dene.")
    return dict(table=table, comments=comments, steps=steps, id_like=id_like, constant=constant, dates=dates,
                numeric_text=num_text, leak_names=leak_names)


# ----------------------------------------------------------------------------
# 3) Özellik üretme (genel, tarih sabit referansla)
# ----------------------------------------------------------------------------
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Tarih sütunlarını sayıya çevirir; oran / fark / log özellikleri üretir. Referans tarih sabittir (today KULLANILMAZ)."""

    def __init__(self, date_cols=None, reference_date=None, ratios=None, diffs=None, logs=None):
        self.date_cols = date_cols
        self.reference_date = reference_date
        self.ratios = ratios
        self.diffs = diffs
        self.logs = logs

    def fit(self, X, y=None):
        cols = [c for c in (self.date_cols or []) if c in X.columns]
        if self.reference_date:
            self.ref_ = pd.Timestamp(self.reference_date)
        elif cols:
            mx = [pd.to_datetime(X[c], errors="coerce", dayfirst=True).max() for c in cols]
            self.ref_ = max([m for m in mx if pd.notna(m)], default=pd.Timestamp("2000-01-01"))
        else:
            self.ref_ = pd.Timestamp("2000-01-01")
        return self

    def transform(self, X):
        d = X.copy()
        for c in (self.date_cols or []):
            if c not in d.columns:
                continue
            t = pd.to_datetime(d[c], errors="coerce", dayfirst=True)
            d[c + "_yil"] = t.dt.year
            d[c + "_ay"] = t.dt.month
            d[c + "_haftanin_gunu"] = t.dt.dayofweek
            d[c + "_ref_gun"] = (self.ref_ - t).dt.days
            d = d.drop(columns=c)
        for r in (self.ratios or []):
            if r["a"] in d.columns and r["b"] in d.columns:
                d[r["name"]] = pd.to_numeric(d[r["a"]], errors="coerce") / pd.to_numeric(d[r["b"]], errors="coerce").replace(0, np.nan)
        for r in (self.diffs or []):
            if r["a"] in d.columns and r["b"] in d.columns:
                d[r["name"]] = pd.to_numeric(d[r["a"]], errors="coerce") - pd.to_numeric(d[r["b"]], errors="coerce")
        for c in (self.logs or []):
            if c in d.columns:
                d[c + "_log"] = np.log1p(pd.to_numeric(d[c], errors="coerce").clip(lower=0))
        return d


# ----------------------------------------------------------------------------
# 4) Ön işleme
# ----------------------------------------------------------------------------
class Winsorizer(TransformerMixin, BaseEstimator):
    """Uç değerleri train'deki alt/üst yüzdeliklere çeker (sınırlar SADECE train'den öğrenilir)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(a, self.lower, axis=0)
        self.hi_ = np.nanquantile(a, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Kategoriyi train'deki görülme oranıyla değiştirir (çok sınıflı sütunlar için basit, sızıntısız)."""

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.maps_ = [df[c].value_counts(normalize=True).to_dict() for c in df.columns]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        out = np.zeros(df.shape, dtype=float)
        for i, c in enumerate(df.columns):
            out[:, i] = df[c].map(self.maps_[i]).fillna(0.0).to_numpy(dtype=float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


DEFAULT_PREP = dict(num_impute="median", winsor=(0.01, 0.99), scaler="auto", rare_freq=0.01, high_card_threshold=15,
                    high_card_method="target", missing_indicator=True, overrides={}, ordinals={})


def group_columns(X, prep):
    """Sütunları tiplerine ayırır. prep['overrides'] ile elle değiştirilebilir."""
    groups = {"numeric": [], "binary": [], "ordinal": [], "nominal": [], "high": [], "drop": []}
    ordinals = prep.get("ordinals", {})
    overrides = prep.get("overrides", {})
    for c in X.columns:
        s = X[c]
        if c in overrides:
            groups[overrides[c]].append(c)
        elif c in ordinals:
            groups["ordinal"].append(c)
        elif str(s.dtype).startswith("datetime"):
            groups["drop"].append(c)
        elif pd.api.types.is_bool_dtype(s):
            groups["binary"].append(c)
        elif pd.api.types.is_numeric_dtype(s):
            groups["binary" if s.nunique(dropna=True) <= 2 else "numeric"].append(c)
        else:
            groups["high" if s.nunique() > prep["high_card_threshold"] else "nominal"].append(c)
    return groups


def build_preprocessor(X, prep, scale, task, seed=42):
    groups = group_columns(X, prep)
    kind = prep["scaler"]
    if kind == "auto":
        kind = "standard" if scale else "none"
    scalers = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
    num_steps = []
    if prep["winsor"]:
        num_steps.append(("winsor", Winsorizer(prep["winsor"][0], prep["winsor"][1])))
    num_steps.append(("impute", SimpleImputer(strategy=prep["num_impute"], add_indicator=prep["missing_indicator"])))
    if kind in scalers:
        num_steps.append(("scale", scalers[kind]()))
    parts = []
    if groups["numeric"]:
        parts.append(("num", Pipeline(num_steps), groups["numeric"]))
    if groups["binary"]:
        parts.append(("bin", SimpleImputer(strategy="most_frequent"), groups["binary"]))
    if groups["ordinal"]:
        cats = []
        for c in groups["ordinal"]:
            order = prep.get("ordinals", {}).get(c)
            cats.append(order if order else sorted(X[c].dropna().unique().tolist()))
        parts.append(("ord", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("enc", OrdinalEncoder(categories=cats, handle_unknown="use_encoded_value",
                                                              unknown_value=-1))]), groups["ordinal"]))
    if groups["nominal"]:
        parts.append(("nom", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Bilinmiyor")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=prep["rare_freq"],
                                                                sparse_output=False, drop="if_binary"))]), groups["nominal"]))
    if groups["high"] and prep["high_card_method"] != "drop":
        if prep["high_card_method"] == "target":
            from sklearn.preprocessing import TargetEncoder
            enc = TargetEncoder(target_type="binary" if task == "classification" else "continuous", random_state=seed)
        else:
            enc = FrequencyEncoder()
        parts.append(("high", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Bilinmiyor")),
                                        ("enc", enc)]), groups["high"]))
    return ColumnTransformer(parts, remainder="drop")


# ----------------------------------------------------------------------------
# 5) Model havuzu ve pipeline kurma
# ----------------------------------------------------------------------------
NEEDS_SCALING = {"Logistic Regression", "Ridge", "ElasticNet", "SVC", "SVR", "KNN"}


def model_zoo(task, seed=42, balanced=True, n_estimators=200, pos_weight=1.0):
    cw = "balanced" if balanced else None
    if task == "classification":
        zoo = {
            "Dummy": DummyClassifier(strategy="prior"),
            "Logistic Regression": LogisticRegression(max_iter=3000, class_weight=cw),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight=cw, random_state=seed),
            "Random Forest": RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=3, class_weight=cw, random_state=seed),
            "Extra Trees": ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=3, class_weight=cw, random_state=seed),
            "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
            "HistGradientBoosting": HistGradientBoostingClassifier(class_weight=cw, random_state=seed),
            "SVC": SVC(probability=True, class_weight=cw, random_state=seed),
            "KNN": KNeighborsClassifier(n_neighbors=15),
        }
        if XGB_OK:
            zoo["XGBoost"] = XGBClassifier(n_estimators=n_estimators, learning_rate=0.1, max_depth=5, subsample=0.9,
                                           eval_metric="logloss", scale_pos_weight=pos_weight if balanced else 1.0,
                                           random_state=seed, verbosity=0)
    else:
        zoo = {
            "Dummy": DummyRegressor(strategy="mean"),
            "Ridge": Ridge(alpha=1.0),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=seed),
            "Random Forest": RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, random_state=seed),
            "Extra Trees": ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=3, random_state=seed),
            "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
            "HistGradientBoosting": HistGradientBoostingRegressor(random_state=seed),
            "SVR": SVR(C=1.0),
            "KNN": KNeighborsRegressor(n_neighbors=15),
        }
        if XGB_OK:
            zoo["XGBoost"] = XGBRegressor(n_estimators=n_estimators, learning_rate=0.1, max_depth=5, subsample=0.9,
                                          random_state=seed, verbosity=0)
    return zoo


def build_pipeline(X_train, fe_cfg, prep, name, estimator, task, seed=42, log_target=False):
    """FeatureBuilder -> ön işleme -> model. Ön işleme sütunları FE çıktısından belirlenir."""
    fe = FeatureBuilder(**fe_cfg)
    Xf = clone(fe).fit(X_train).transform(X_train)
    ct = build_preprocessor(Xf, prep, scale=(name in NEEDS_SCALING), task=task, seed=seed)
    est = estimator
    if task == "regression" and log_target:
        est = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("fe", fe), ("prep", ct), ("model", est)])


def param_prefix(task, log_target):
    return "model__regressor__" if (task == "regression" and log_target) else "model__"


# ----------------------------------------------------------------------------
# 6) Bölme ve çapraz doğrulama
# ----------------------------------------------------------------------------
def split_data(X, y, task, mode="random", test_size=0.2, seed=42, group=None, time=None, stratify=True):
    if mode == "group" and group is not None:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, group))
        return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], group.iloc[tr]
    if mode == "time" and time is not None:
        order = time.sort_values().index
        Xs, ys = X.loc[order], y.loc[order]
        cut = int(len(Xs) * (1 - test_size))
        return Xs.iloc[:cut], Xs.iloc[cut:], ys.iloc[:cut], ys.iloc[cut:], None
    strat = y if (task == "classification" and stratify) else None
    a, b, c, d = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    return a, b, c, d, None


def get_cv(task, mode, k, seed, groups=None):
    if mode == "time":
        return TimeSeriesSplit(k)
    if task == "classification":
        return StratifiedGroupKFold(k, shuffle=True, random_state=seed) if groups is not None else StratifiedKFold(k, shuffle=True, random_state=seed)
    return GroupKFold(k) if groups is not None else KFold(k, shuffle=True, random_state=seed)


CLF_METRICS = {"average_precision": "PR-AUC", "roc_auc": "ROC-AUC", "f1": "F1", "balanced_accuracy": "Balanced Accuracy",
               "recall": "Recall", "precision": "Precision"}
REG_METRICS = {"neg_root_mean_squared_error": "RMSE", "neg_mean_absolute_error": "MAE", "r2": "R²"}


def default_primary(task):
    return "average_precision" if task == "classification" else "neg_root_mean_squared_error"


def higher_is_better(metric):
    return not metric.startswith("neg_")


def show_score(value, metric):
    return value if higher_is_better(metric) else -value


def scoring_table(task):
    if task == "classification":
        return {"PR-AUC": "average_precision", "ROC-AUC": "roc_auc", "F1": "f1", "Recall": "recall",
                "Precision": "precision", "Bal.Acc": "balanced_accuracy"}
    return {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R²": "r2"}


def compare_models(pipes, X, y, cv, groups, task, primary, progress=None):
    """Her model için CV (test + train skor). Dönüş: (tablo, fold_skorları sözlüğü, hatalar)."""
    scoring = dict(scoring_table(task))
    scoring["ANA"] = primary
    rows, folds, errors = [], {}, {}
    for i, (name, pipe) in enumerate(pipes.items()):
        t0 = time.time()
        try:
            res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=scoring, return_train_score=True, n_jobs=-1)
        except Exception as e:
            errors[name] = str(e)[:300]
            continue
        ana_cv = show_score(res["test_ANA"].mean(), primary)
        ana_tr = show_score(res["train_ANA"].mean(), primary)
        row = {"Model": name, "ANA (CV)": ana_cv, "Std": res["test_ANA"].std(), "ANA (Train)": ana_tr,
               "Fark (Train-CV)": abs(ana_tr - ana_cv), "Süre (sn)": round(time.time() - t0, 1)}
        for label, sc in scoring_table(task).items():
            row[label] = show_score(res["test_" + label].mean() if label in res else res["test_" + label].mean(), sc) \
                if ("test_" + label) in res else np.nan
        rows.append(row)
        folds[name] = res["test_ANA"]
        if progress:
            progress(i + 1, len(pipes), name)
    return pd.DataFrame(rows), folds, errors


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau-Bengio düzeltmeli eşleşmiş t-testi: CV katlarının bağımsız olmadığını hesaba katar."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if k < 2 or var == 0:
        return np.nan, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return t, 2 * (1 - stats.t.cdf(abs(t), k - 1))


def model_comparison_comments(table, folds, primary, n_train, k):
    comments, steps = [], []
    if table.empty:
        return ["Hiçbir model değerlendirilemedi."], []
    hib = higher_is_better(primary)
    real = table[table["Model"] != "Dummy"]
    if real.empty:
        return ["Sadece Dummy çalıştı; en az bir gerçek model seç."], []
    best_row = real.sort_values("ANA (CV)", ascending=not hib).iloc[0]
    dummy = table[table["Model"] == "Dummy"]
    if not dummy.empty:
        d = float(dummy["ANA (CV)"].iloc[0])
        comments.append(f"Dummy (hiçbir şey öğrenmeyen referans) skoru {d:.3f}; en iyi model {best_row['Model']}: "
                        f"{best_row['ANA (CV)']:.3f} ± {best_row['Std']:.3f}.")
        if abs(best_row["ANA (CV)"] - d) < 2 * best_row["Std"]:
            comments.append("En iyi model Dummy'den ANLAMLI uzaklaşmıyor (fark < 2 std): güçlü sinyal yok ya da hedef/veri sorunlu.")
            steps.append("Bölüm 3'e dön: yeni özellikler türet, hedef tanımını gözden geçir.")
    ordered = real.sort_values("ANA (CV)", ascending=not hib)
    if len(ordered) > 1 and best_row["Model"] in folds:
        second = ordered.iloc[1]["Model"]
        n_test = max(int(n_train / max(k - 1, 1)), 1)
        t, p = corrected_ttest(folds[best_row["Model"]], folds[second], n_train, n_test)
        comments.append(f"{best_row['Model']} vs {second}: düzeltilmiş t-testi p={p:.3f}. " +
                        ("Fark istatistiksel olarak anlamlı (p<0.05)." if p < 0.05 else
                         "Fark anlamlı DEĞİL: eşitlikte daha basit/yorumlanabilir modeli seç."))
    for _, r in real.iterrows():
        if r["Fark (Train-CV)"] > 0.15 * max(abs(r["ANA (Train)"]), 1e-9) and r["Fark (Train-CV)"] > 2 * r["Std"]:
            comments.append(f"{r['Model']}: train {r['ANA (Train)']:.3f} vs CV {r['ANA (CV)']:.3f}. Büyük fark = ezber (overfit, yüksek varyans).")
    comments.append("Okuma kuralı: train ve CV ikisi de düşükse model az öğreniyor (bias); train yüksek CV düşükse ezber (varyans).")
    return comments, steps


# ----------------------------------------------------------------------------
# 7) Sızıntı avcısı
# ----------------------------------------------------------------------------
def leakage_scan(X_train, y_train, fe_cfg, prep, task, seed=42, progress=None):
    """Her özelliği tek başına sığ bir ağaca verir; tek sütunla aşırı yüksek skor = şüpheli."""
    fe = FeatureBuilder(**fe_cfg).fit(X_train)
    Xf = fe.transform(X_train)
    metric, thr = ("roc_auc", 0.75) if task == "classification" else ("r2", 0.70)
    rows = []
    for i, c in enumerate(Xf.columns):
        single = Xf[[c]]
        ct = build_preprocessor(single, prep, scale=False, task=task, seed=seed)
        tree = DecisionTreeClassifier(max_depth=3, random_state=seed) if task == "classification" else DecisionTreeRegressor(max_depth=3, random_state=seed)
        try:
            cv = StratifiedKFold(3, shuffle=True, random_state=seed) if task == "classification" else KFold(3, shuffle=True, random_state=seed)
            s = cross_val_score(Pipeline([("p", ct), ("m", tree)]), single, y_train, cv=cv, scoring=metric).mean()
        except Exception:
            s = np.nan
        rows.append((c, s))
        if progress:
            progress(i + 1, len(Xf.columns))
    out = pd.DataFrame(rows, columns=["Sütun", "Tek başına skor"]).sort_values("Tek başına skor", ascending=False)
    out["Şüpheli"] = out["Tek başına skor"] > thr
    return out, metric, thr


# ----------------------------------------------------------------------------
# 8) Optuna ayarı
# ----------------------------------------------------------------------------
def _sp(t, lo=None, hi=None, log=False, choices=None):
    return {"type": t, "low": lo, "high": hi, "log": log, "choices": choices}


SEARCH_SPACES = {
    "Random Forest": {"n_estimators": _sp("int", 100, 500), "max_depth": _sp("int", 3, 20), "min_samples_leaf": _sp("int", 1, 30),
                      "max_features": _sp("float", 0.2, 1.0)},
    "Extra Trees": {"n_estimators": _sp("int", 100, 500), "max_depth": _sp("int", 3, 20), "min_samples_leaf": _sp("int", 1, 30),
                    "max_features": _sp("float", 0.2, 1.0)},
    "Gradient Boosting": {"n_estimators": _sp("int", 50, 400), "learning_rate": _sp("float", 0.01, 0.3, True), "max_depth": _sp("int", 2, 6),
                          "subsample": _sp("float", 0.5, 1.0), "min_samples_leaf": _sp("int", 1, 50)},
    "HistGradientBoosting": {"learning_rate": _sp("float", 0.01, 0.3, True), "max_depth": _sp("int", 2, 12), "max_leaf_nodes": _sp("int", 8, 64),
                             "min_samples_leaf": _sp("int", 10, 100), "l2_regularization": _sp("float", 0.001, 10, True)},
    "XGBoost": {"n_estimators": _sp("int", 50, 500), "learning_rate": _sp("float", 0.01, 0.3, True), "max_depth": _sp("int", 2, 10),
                "subsample": _sp("float", 0.5, 1.0), "colsample_bytree": _sp("float", 0.4, 1.0), "reg_lambda": _sp("float", 0.01, 10, True)},
    "Logistic Regression": {"C": _sp("float", 0.001, 100, True)},
    "Ridge": {"alpha": _sp("float", 0.001, 1000, True)},
    "ElasticNet": {"alpha": _sp("float", 0.0001, 10, True), "l1_ratio": _sp("float", 0.0, 1.0)},
    "Decision Tree": {"max_depth": _sp("int", 2, 20), "min_samples_leaf": _sp("int", 1, 50)},
    "SVC": {"C": _sp("float", 0.01, 100, True), "gamma": _sp("float", 0.0001, 1.0, True)},
    "SVR": {"C": _sp("float", 0.01, 100, True), "gamma": _sp("float", 0.0001, 1.0, True)},
    "KNN": {"n_neighbors": _sp("int", 3, 50), "weights": _sp("cat", choices=["uniform", "distance"])},
}

PARAM_INFO = {
    "n_estimators": "Ağaç sayısı. Çok ağaç = daha kararlı tahmin ama yavaş. Ormanlarda ezberi artırmaz; boosting'de çok olursa ezber riski vardır.",
    "max_depth": "Ağacın sorabileceği en fazla soru sayısı. Derin = karmaşık kuralları öğrenir ama ezberler; sığ = basit kalır, sinyal kaçırabilir.",
    "min_samples_leaf": "Bir yaprağın en az kaç örnek içermesi gerektiği. Büyük = gürültüye takılmaz (genel), küçük = detaya/gürültüye takılır.",
    "max_features": "Her bölünmede bakılacak sütun oranı. Düşük = ağaçlar farklılaşır (ormanın gücü); çok düşük = iyi sütunu kaçırır.",
    "learning_rate": "Her adımda ne kadar hızlı öğrendiği. Küçük = yavaş ama sağlam (çok ağaç gerekir); büyük = hızlı ama dalgalanma/ezber riski.",
    "subsample": "Her ağaç için kullanılan satır oranı. 1'den küçük = ağaçlar farklılaşır, ezber azalır.",
    "colsample_bytree": "Her ağaç için kullanılan sütun oranı. Düşük = çeşitlilik artar, ezber azalır.",
    "max_leaf_nodes": "Bir ağaçtaki en fazla yaprak sayısı (karmaşıklık sınırı). Çok yaprak = ezber riski.",
    "l2_regularization": "Aşırı büyük ağırlıklara ceza. Büyük = model sakinleşir; çok büyük olursa az öğrenir.",
    "reg_lambda": "L2 cezası. Büyük = model sakinleşir, ezber azalır.",
    "C": "Düzenleme gücünün TERSİ. Küçük C = güçlü ceza (basit model); büyük C = esnek (ezber riski).",
    "alpha": "Ceza gücü. Büyük = katsayılar küçülür (basit model); küçük = esnek.",
    "l1_ratio": "ElasticNet'te L1 (sütun eleme) payı. 1 = Lasso gibi, 0 = Ridge gibi.",
    "gamma": "RBF çekirdeğinin ne kadar 'yerel' bakacağı. Büyük = çok yerel (ezber), küçük = pürüzsüz.",
    "n_neighbors": "KNN'de kaç komşuya bakılacağı. Az = gürültüye hassas (ezber); çok = fazla düzleşir.",
    "weights": "Komşuların oyu eşit mi (uniform), yakın olan daha mı ağırlıklı (distance)?",
}

TUNING_MISTAKES = [
    "Test setiyle ayar yapmak: test en sonda TEK kez kullanılır; ayarın hepsi train içi CV ile yapılır.",
    "Sadece en iyi skora bakmak: ± std'yi ve train-CV farkını gör; küçük farklar şanstır.",
    "Çok az deneme ya da dar aralık: en iyi değer aralığın kenarında çıkıyorsa aralığı genişlet.",
    "Hiperparametreyle veri sorununu çözmeye çalışmak: sızıntı, yanlış hedef ya da zayıf özellik ayarla düzelmez.",
    "Yanlış metrikle ayar: dengesiz veride accuracy yerine PR-AUC; regresyonda hedefe uygun hata ölçütü.",
    "Ön işlemeyi CV dışında yapmak: bu platformda her şey Pipeline içinde, bu hata otomatik önlenir.",
]


def run_tuning(make_pipe, X, y, cv, groups, primary, space, n_trials, seed, prefix, progress=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []

    def objective(trial):
        params = {}
        for p, s in space.items():
            if s["type"] == "int":
                params[p] = trial.suggest_int(p, int(s["low"]), int(s["high"]), log=bool(s.get("log")))
            elif s["type"] == "float":
                params[p] = trial.suggest_float(p, float(s["low"]), float(s["high"]), log=bool(s.get("log")))
            else:
                params[p] = trial.suggest_categorical(p, s["choices"])
        pipe = make_pipe().set_params(**{prefix + k: v for k, v in params.items()})
        res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=primary, return_train_score=True, n_jobs=-1)
        rows.append({"deneme": trial.number, **params, "CV": show_score(res["test_score"].mean(), primary),
                     "Std": res["test_score"].std(), "Train": show_score(res["train_score"].mean(), primary)})
        if progress:
            progress(trial.number + 1, n_trials)
        return res["test_score"].mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials = pd.DataFrame(rows)
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        imp = {}
    best = make_pipe().set_params(**{prefix + k: v for k, v in study.best_params.items()})
    best.fit(X, y)
    return dict(best_params=study.best_params, best_score=show_score(study.best_value, primary), trials=trials,
                importance=imp, model=best)


def edge_warnings(best_params, space):
    out = []
    for p, v in best_params.items():
        s = space[p]
        if s["type"] in ("int", "float") and s["high"] > s["low"]:
            width = (np.log(s["high"]) - np.log(s["low"])) if s.get("log") else (s["high"] - s["low"])
            pos = ((np.log(v) - np.log(s["low"])) if s.get("log") else (v - s["low"])) / width
            if pos < 0.05 or pos > 0.95:
                out.append(p)
    return out


# ----------------------------------------------------------------------------
# 9) Voting / Stacking
# ----------------------------------------------------------------------------
def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def build_ensemble(kind, members, task, weights=None, voting="soft", seed=42):
    ests = [(slug(n), clone(p)) for n, p in members.items()]
    if kind == "Voting":
        if task == "classification":
            return VotingClassifier(estimators=ests, voting=voting, weights=weights)
        return VotingRegressor(estimators=ests, weights=weights)
    if task == "classification":
        return StackingClassifier(estimators=ests, final_estimator=LogisticRegression(max_iter=3000), cv=3)
    return StackingRegressor(estimators=ests, final_estimator=Ridge(alpha=1.0), cv=3)


# ----------------------------------------------------------------------------
# 10) Değerlendirme
# ----------------------------------------------------------------------------
def bootstrap_ci(fn, y, p, n=300, seed=42, clf=True):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if clf and len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def choose_threshold(model, X, y, cv, groups, cost_fn, cost_fp):
    """Eşiği test'e bakmadan, train'in out-of-fold olasılıklarından maliyete göre seçer."""
    oof = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba", n_jobs=-1)[:, 1]
    cands = np.linspace(0.02, 0.98, 97)
    costs = []
    for t in cands:
        tn, fp, fn, tp = confusion_matrix(y, oof >= t, labels=[0, 1]).ravel()
        costs.append(fn * cost_fn + fp * cost_fp)
    return float(cands[int(np.argmin(costs))]), cands, np.array(costs)


def lift_table(y, p):
    t = pd.DataFrame({"p": np.asarray(p), "y": np.asarray(y)})
    t["Dilim"] = pd.qcut(t["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    g = t.groupby("Dilim", observed=True).agg(Satır=("y", "size"), Pozitif=("y", "sum"), Oran=("y", "mean"))
    prev = t["y"].mean()
    g["Lift"] = g["Oran"] / prev
    g["Kümülatif yakalama %"] = (g["Pozitif"].cumsum() / max(g["Pozitif"].sum(), 1) * 100).round(1)
    return g.reset_index()


def eval_classification(model, X_test, y_test, thr, n_boot=300, seed=42):
    has_proba = hasattr(model, "predict_proba")
    res = {"has_proba": has_proba}
    if has_proba:
        p = model.predict_proba(X_test)[:, 1]
        pred = (p >= thr).astype(int)
        res.update(p=p, roc=roc_auc_score(y_test, p), pr=average_precision_score(y_test, p), brier=brier_score_loss(y_test, p),
                   roc_ci=bootstrap_ci(roc_auc_score, y_test, p, n_boot, seed), pr_ci=bootstrap_ci(average_precision_score, y_test, p, n_boot, seed))
        fpr, tpr, _ = roc_curve(y_test, p)
        pre, rec, _ = precision_recall_curve(y_test, p)
        res.update(roc_curve=(fpr, tpr), pr_curve=(rec, pre), lift=lift_table(y_test, p))
        try:
            frac, mean_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
            res["calib"] = (mean_pred, frac)
        except Exception:
            res["calib"] = None
    else:
        pred = model.predict(X_test)
    res["pred"] = pred
    res["prevalence"] = float(np.mean(y_test))
    res["cm"] = confusion_matrix(y_test, pred, labels=[0, 1])
    res["report"] = pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T
    res["f1"] = f1_score(y_test, pred, zero_division=0)
    return res


def eval_regression(model, X_test, y_test, y_train, n_boot=300, seed=42):
    p = model.predict(X_test)
    err = np.asarray(y_test) - p
    rmse = mean_squared_error(y_test, p) ** 0.5
    base = mean_squared_error(y_test, np.full(len(y_test), np.mean(y_train))) ** 0.5
    q = pd.DataFrame({"y": np.asarray(y_test), "hata": np.abs(err)})
    q["Çeyrek"] = pd.qcut(q["y"], 4, labels=["Q1 (düşük)", "Q2", "Q3", "Q4 (yüksek)"], duplicates="drop")
    return dict(p=p, err=err, rmse=rmse, mae=mean_absolute_error(y_test, p), r2=r2_score(y_test, p), baseline_rmse=base,
                rmse_ci=bootstrap_ci(lambda a, b: mean_squared_error(a, b) ** 0.5, y_test, p, n_boot, seed, clf=False),
                quartile=q.groupby("Çeyrek", observed=True)["hata"].mean().reset_index(),
                rho=float(stats.spearmanr(p, np.abs(err))[0]), skew=float(stats.skew(err)), kurt=float(stats.kurtosis(err)))


def classification_comments(res, thr, cv_ref=None, cv_std=None, primary="average_precision"):
    c, steps = [], []
    prev = res["prevalence"]
    if res["has_proba"]:
        roc, pr = res["roc"], res["pr"]
        band = "çok zayıf" if roc < .6 else "zayıf-orta" if roc < .7 else "kabul edilebilir" if roc < .8 else "iyi" if roc < .9 else "çok iyi"
        c.append(f"ROC-AUC = rastgele bir pozitifin rastgele bir negatiften yüksek skor alma olasılığı: {roc:.2f} ({band}).")
        c.append(f"PR-AUC {pr:.3f}; rastgele referans {prev:.3f} (≈{pr/prev:.1f} kat). " +
                 ("Dengesiz veride asıl bu oranı oku." if prev < 0.3 else "Veri dengeli olduğu için ROC-AUC ile birlikte oku."))
        w = res["pr_ci"][1] - res["pr_ci"][0]
        c.append(f"PR-AUC %95 güven aralığı {res['pr_ci'][0]:.2f}-{res['pr_ci'][1]:.2f} (genişlik {w:.2f}): " +
                 ("belirsizlik yüksek, tek rakam gibi sunma." if w > .10 else "aralık dar, skor güvenilir.") +
                 " İki modelin aralıkları üst üste biniyorsa fark iddia edemezsin.")
        top = res["lift"].iloc[0]
        c.append(f"En yüksek skorlu %10'luk dilim pozitiflerin %{top['Kümülatif yakalama %']:.0f}'ini içeriyor (lift {top['Lift']:.1f}x): "
                 "iş tarafına 'ilk %10'u izleyin' diyebileceğin somut çıktı.")
        if res["brier"] > prev * (1 - prev):
            c.append("Brier skoru 'hep ortalamayı söyle' referansından kötü: olasılıklar güvenilir değil.")
        if abs(res["p"].mean() - prev) > 0.05:
            c.append(f"Ortalama tahmin olasılığı ({res['p'].mean():.2f}) gerçek oranın ({prev:.2f}) uzağında: class_weight olasılıkları kaydırır. "
                     "Sıralama (ROC/PR) güvenilir, olasılığın kendisi yüzde değildir.")
            steps.append("Olasılığı doğrudan yüzde/beklenen kayıp için kullanacaksan modeli kalibre et (CalibratedClassifierCV).")
        if roc > 0.95:
            c.append("ROC-AUC 0.95 üstü gerçek hayatta çok nadirdir: önce veri sızıntısından şüphelen.")
            steps.append("Bölüm 6 sızıntı tablosunu ve Bölüm 10 önem listesini tekrar incele.")
        if cv_ref is not None and primary == "average_precision" and res["pr"] < cv_ref - 2 * (cv_std or 0):
            c.append(f"Test PR-AUC ({res['pr']:.3f}) CV tahmininin ({cv_ref:.3f}) belirgin altında: ayar/seçimde hafif ezber ya da dönem farkı olabilir.")
    tn, fp, fn, tp = res["cm"].ravel()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    c.append(f"Eşik {thr:.2f}'de uyarıların %{prec*100:.0f}'i gerçek pozitif (precision), pozitiflerin %{rec*100:.0f}'ini yakalıyorsun (recall). "
             "Eşik düşerse recall artar, precision düşer; maliyet oranı karar verir.")
    steps.append("Skor yetersizse Bölüm 3'te yeni özellik türet; yeterliyse eşik/maliyeti iş birimiyle teyit et ve küçük bir A/B testi planla.")
    return c, steps


def regression_comments(res, cv_ref=None, cv_std=None, log_target=False):
    c, steps = [], []
    c.append(f"R² = {res['r2']:.2f}: hedefteki değişimin yaklaşık %{res['r2']*100:.0f}'i açıklanıyor. RMSE baseline'ın %{res['rmse']/res['baseline_rmse']*100:.0f}'ine indi.")
    ratio = res["rmse"] / max(res["mae"], 1e-9)
    c.append(f"RMSE/MAE = {ratio:.2f} (≈1.25 düzenli hata; >1.5 birkaç çok büyük hata var).")
    if ratio > 1.5:
        steps.append("En büyük 20 artığa bak: veri hatası mı, gerçek uç değer mi?")
    c.append(f"Ortalama artık {res['err'].mean():.2f} (RMSE'nin %{abs(res['err'].mean())/res['rmse']*100:.0f}'i): sıfıra yakınsa sistematik fazla/az tahmin yok.")
    if res["rho"] > 0.3:
        c.append(f"Tahmin büyüdükçe |hata| büyüyor (Spearman r={res['rho']:.2f}): hata varyansı sabit değil (heteroskedastisite).")
        steps.append("Bölüm 1'de log dönüşümünü dene." if not log_target else "Log açık; yüksek aralık için ayrı model/özellik düşün.")
    c.append(f"Artık çarpıklığı {res['skew']:.2f}, basıklığı {res['kurt']:.2f} (normalde ≈0, 0). Büyük sapma = bazı gruplarda sistematik hata.")
    c.append(f"RMSE %95 güven aralığı {res['rmse_ci'][0]:.2f} - {res['rmse_ci'][1]:.2f}.")
    if cv_ref is not None and res["rmse"] > cv_ref + 2 * (cv_std or 0):
        c.append(f"Test RMSE ({res['rmse']:.2f}) CV tahmininin ({cv_ref:.2f}) belirgin üstünde: ezber ya da dönem farkı olabilir.")
    if res["r2"] < 0.2:
        c.append("R² düşük: özelliklerde hedefi açıklayan güçlü bilgi yok.")
        steps.append("Yeni özellik türet ya da eksik bir bilgi kaynağı ara.")
    steps.append("En kötü hata çeyreğini iş birimiyle konuş: o aralıkta tahminler karar için yeterince güvenilir mi?")
    return c, steps


def pdp_data(model, X, col):
    Xf = X.copy()
    if pd.api.types.is_numeric_dtype(Xf[col]):
        Xf[col] = Xf[col].astype(float)
    r = partial_dependence(model, Xf, features=[col], kind="average", grid_resolution=20)
    return np.asarray(r["grid_values"][0]), np.asarray(r["average"][0])


# ----------------------------------------------------------------------------
# 11) Demo veri, yol haritası, dışa aktarma
# ----------------------------------------------------------------------------
def demo_data(kind="classification", n=3000, seed=42):
    rng = np.random.default_rng(seed)
    if kind == "classification":
        yas = np.clip(rng.normal(36, 11, n), 18, 75).round()
        uyelik = rng.choice(["Standart", "Premium"], n, p=[.8, .2])
        sehir = rng.choice(["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"], n, p=[.4, .2, .15, .13, .12])
        siparis = rng.poisson(8, n) + 1
        harcama = (siparis * np.exp(rng.normal(5.3, .5, n))).round(2)
        iade = rng.binomial(siparis, 0.08)
        puan = np.clip(np.round(rng.normal(4.0, .9, n) - 0.3 * iade), 1, 5)
        destek = rng.poisson(1, n)
        gun = rng.integers(1, 1200, n)
        z = -0.45 * (siparis - 8) - 0.9 * (puan - 4) + 0.5 * iade + 0.4 * destek - 1.0 * (uyelik == "Premium") + rng.normal(0, 0.6, n) - 0.4
        churn = rng.random(n) < 1 / (1 + np.exp(-z))
        kayit = pd.Timestamp("2026-10-31") - pd.to_timedelta(gun, unit="D")
        df = pd.DataFrame({"musteri_id": [f"M{100000+i}" for i in range(n)], "kayit_tarihi": kayit.strftime("%d.%m.%Y"),
                           "yas": yas, "sehir": sehir, "uyelik": uyelik, "toplam_siparis": siparis, "toplam_harcama": harcama,
                           "iade_sayisi": iade, "puan": puan, "destek_talebi": destek,
                           "hesap_durumu": np.where(churn, "Kapalı", "Aktif"), "churn": np.where(churn, "Evet", "Hayır")})
        df.loc[rng.random(n) < .1, "puan"] = np.nan
        df.loc[rng.random(n) < .04, "yas"] = np.nan
        df["toplam_harcama"] = df["toplam_harcama"].map(lambda v: f"{v:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")
                                                         if rng.random() < .2 else v)
        mask = rng.random(n) < .1
        df.loc[mask, "sehir"] = df.loc[mask, "sehir"].str.upper()
        df.loc[rng.random(n) < .01, "yas"] = 999
        df = pd.concat([df, df.sample(40, random_state=1)]).reset_index(drop=True)
        return df
    m2 = np.clip(rng.normal(110, 40, n), 35, 350).round()
    oda = np.clip(np.round(m2 / 35 + rng.normal(0, .6, n)), 1, 8)
    bina_yasi = rng.integers(0, 45, n)
    semt = rng.choice(["Kadıköy", "Beşiktaş", "Üsküdar", "Pendik", "Esenyurt", "Sarıyer", "Maltepe"], n)
    carpan = pd.Series(semt).map({"Kadıköy": 1.5, "Beşiktaş": 1.7, "Üsküdar": 1.4, "Pendik": .85, "Esenyurt": .65, "Sarıyer": 1.6, "Maltepe": 1.0}).to_numpy()
    kat = rng.integers(0, 15, n)
    fiyat = (m2 * 65000 * carpan * np.exp(-0.008 * bina_yasi) * (1 + 0.01 * kat) * np.exp(rng.normal(0, .12, n))).round(-3)
    ilan = pd.Timestamp("2026-10-31") - pd.to_timedelta(rng.integers(1, 365, n), unit="D")
    df = pd.DataFrame({"ilan_id": np.arange(1, n + 1), "ilan_tarihi": ilan.strftime("%d.%m.%Y"), "semt": semt, "m2": m2, "oda_sayisi": oda,
                       "bina_yasi": bina_yasi, "kat": kat, "isitma": rng.choice(["Kombi", "Merkezi", "Klima"], n), "fiyat": fiyat})
    df.loc[rng.random(n) < .05, "bina_yasi"] = np.nan
    df.loc[rng.integers(0, n, 8), "m2"] = 5000
    return df


YOL_HARITASI = {
    "Sınıflandırma": ["Bölüm 1-11'in hepsi geçerli.",
                      "Hedef dengesizse (azınlık <%10): accuracy'yi bırak, PR-AUC/recall kullan; eşiği maliyete göre seç.",
                      "Bölüm 9'dan sonra: Voting/ayar kazancı std içindeyse özellik mühendisliğine dön.",
                      "Bölüm 10'dan sonra: eşik/maliyeti iş birimiyle doğrula, A/B testi planla, izleme kur."],
    "Regresyon": ["Bölüm 1-11'in hepsi geçerli (eşik seçimi hariç).",
                  "Hedef çarpıksa log dönüşümünü dene.",
                  "Bölüm 9'dan sonra: hata çeyreği tablosuna bak; en kötü aralığı iş birimiyle konuş ve belirsizlik (tahmin aralığı) iste."],
    "Kümeleme (hedef yok)": ["Bu platform denetimli öğrenme içindir. Kullanılabilen: Bölüm 2, 5 (ölçekleme şart).",
                             "Sonra: PCA, KMeans + elbow/silhouette; kümeleri iş diline çevir (her kümenin ortalama profili)."],
    "Zaman serisi": ["Bölüm 4'te 'time' bölmeyi seç (rastgele bölme YASAK).",
                     "Bölüm 3'te gecikme (lag), hareketli ortalama gibi geçmişten türeyen özellikleri kendin üretmelisin (bu platformda hazır yok).",
                     "Baseline: 'son değeri tekrarla'. Model onu geçmiyorsa karmaşıklığa değmez."],
    "Metin (NLP)": ["Bu platformda metin vektörleştirme yok; TF-IDF için ayrı bir pipeline gerekir.",
                    "Önce lojistik regresyon ile baseline kur."],
}


def to_bytes(obj):
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()
