"""
Veri Bilimi Platformu (Streamlit) - her tablo verisi projesi için TEK şablon.
Çalıştırma:  streamlit run app.py      (vb_core.py aynı klasörde olmalı)
"""
import io
import json
import importlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.model_selection import learning_curve, permutation_test_score
from sklearn.inspection import permutation_importance

import vb_core as vb

st.set_page_config(page_title="Veri Bilimi Platformu", page_icon="🔬", layout="wide")
S = st.session_state

SECTIONS = ["1. Kurulum ve Ayarlar", "2. Veri: Yükle, Temizle, Denetle", "3. Hedef ve Özellikler", "4. Train / Test Bölme",
            "5. Ön İşleme", "6. Sızıntı Avcısı", "7. Model Karşılaştırma", "8. Hiperparametre Ayarı",
            "9. Voting / Stacking ve Test", "10. İstatistiksel Güvenilirlik ve Yorum", "11. Dışa Aktar ve Tahmin"]

DEFAULT_CFG = dict(task="classification", seed=42, test_size=20, cv_folds=5, balanced=True, n_est=200, log_target=False,
                   cost_fn=5.0, cost_fp=1.0, primary=None, target=None, pos_label=None, drop_cols=[], date_cols=[],
                   ref_date=None, ordinals={}, ratios=[], diffs=[], logs=[], split_mode="random", group_col=None,
                   time_col=None, prep={})


# ----------------------------------------------------------------------------
# Yardımcılar
# ----------------------------------------------------------------------------
def init_state():
    if "cfg" not in S:
        S.cfg = json.loads(json.dumps(DEFAULT_CFG))
        S.notes = []
        S.done = set()
        S.tuned = {}
        S.test_uses = 0


def arrow_safe(df):
    """Karışık tipli (sayı+metin) sütunları metne çevirir; Streamlit tablo hatasını önler."""
    if not isinstance(df, pd.DataFrame):
        return df
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == "object" and d[c].map(type).nunique() > 1:
            d[c] = d[c].map(lambda v: v if pd.isna(v) else str(v))
    return d


def show_df(df, **kw):
    df = arrow_safe(df)
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def show_fig(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def is_clf():
    return S.cfg["task"] == "classification"


def primary():
    return S.cfg["primary"] or vb.default_primary(S.cfg["task"])


def box(comments, steps, title="YORUM (veri bilimi + istatistik)"):
    """Yorum + sonraki adım kutusu. Sonraki adımlar sidebar'da birikir."""
    st.markdown(f"**{title}**")
    if not comments:
        st.success("Belirgin bir sorun görünmüyor.")
    for c in comments:
        st.info(c)
    if steps:
        st.markdown("**SONRAKİ ADIM**")
        for s in steps:
            st.warning(s)
            if s not in S.notes:
                S.notes.append(s)


def need(*keys):
    for k in keys:
        if k not in S:
            st.warning("Bu bölüm için önce önceki adımları tamamla (soldaki listede ✅ olmayan adımlar).")
            st.stop()


def reset_from(step):
    """Bir üst adım değişince sonraki adımların sonuçlarını temizler."""
    order = {3: ["X", "y", "classes", "extra"], 4: ["X_train", "X_test", "y_train", "y_test", "groups_train"],
             7: ["cv_table", "cv_folds_scores"], 8: ["tuned"], 9: ["final", "eval"]}
    for s in sorted(order):
        if s >= step:
            for k in order[s]:
                S.pop(k, None)
    S.tuned = {}
    S.done = {d for d in S.done if d < step}


def W(fn, label, key, default, **kw):
    """Ayar widget'ı: değer S.cfg içinde kalıcıdır (bölüm değişse de kaybolmaz)."""
    S.cfg[key] = fn(label, value=S.cfg.get(key, default), key="w_" + key, **kw)
    return S.cfg[key]


def fe_cfg():
    c = S.cfg
    return dict(date_cols=c["date_cols"], reference_date=c["ref_date"], ratios=c["ratios"], diffs=c["diffs"], logs=c["logs"])


def prep_cfg():
    p = dict(vb.DEFAULT_PREP)
    p.update(S.cfg.get("prep", {}))
    p["ordinals"] = S.cfg["ordinals"]
    if isinstance(p.get("winsor"), list):
        p["winsor"] = tuple(p["winsor"])
    return p


def pos_weight():
    if is_clf() and "y_train" in S:
        pos = max(int(S.y_train.sum()), 1)
        return (len(S.y_train) - pos) / pos
    return 1.0


def zoo():
    return vb.model_zoo(S.cfg["task"], S.cfg["seed"], S.cfg["balanced"], S.cfg["n_est"], pos_weight())


def make_pipe(name, estimator=None):
    est = clone(estimator) if estimator is not None else clone(zoo()[name])
    return vb.build_pipeline(S.X_train, fe_cfg(), prep_cfg(), name, est, S.cfg["task"], S.cfg["seed"], S.cfg["log_target"])


def get_cv():
    return vb.get_cv(S.cfg["task"], S.cfg["split_mode"], S.cfg["cv_folds"], S.cfg["seed"], S.get("groups_train"))


def prefix():
    return vb.param_prefix(S.cfg["task"], S.cfg["log_target"])


def how_to(text):
    with st.expander("ℹ️ Bu bölüm ne yapar? Sık yapılan hatalar"):
        st.markdown(text)


# ----------------------------------------------------------------------------
# Bölüm 1: Kurulum
# ----------------------------------------------------------------------------
def sec1():
    st.header("1. Kurulum ve Ayarlar")
    how_to("Projenin türünü ve genel ayarları burada seçersin. **Kod yazmana gerek yok**; her proje için sadece bu ayarlar ve sonraki "
           "bölümlerdeki seçimler değişir. Tüm ayarları JSON olarak indirip sonraki projede yükleyebilirsin.\n\n"
           "**Sık hata:** Görevi yanlış seçmek (sayısal hedefe sınıflandırma ya da tersi).")
    c1, c2, c3 = st.columns(3)
    with c1:
        old = S.cfg["task"]
        task = st.radio("Görev türü", ["classification", "regression"], index=0 if old == "classification" else 1,
                        format_func=lambda x: "Sınıflandırma (ikili 0/1)" if x == "classification" else "Regresyon (sayısal hedef)", key="w_task")
        if task != old:
            S.cfg["task"] = task
            S.cfg["primary"] = None
            reset_from(3)
        W(st.number_input, "Rastgelelik tohumu (seed)", "seed", 42, step=1)
        W(st.slider, "Test seti oranı (%)", "test_size", 20, min_value=10, max_value=40, step=5)
    with c2:
        W(st.slider, "Çapraz doğrulama kat sayısı", "cv_folds", 5, min_value=3, max_value=10)
        W(st.slider, "Ağaç sayısı (ormanlar/boosting)", "n_est", 200, min_value=50, max_value=600, step=50)
        if is_clf():
            W(st.checkbox, "Dengesiz sınıf: class_weight='balanced'", "balanced", True)
        else:
            W(st.checkbox, "Hedefe log dönüşümü uygula (sağa çarpık hedef için)", "log_target", False)
    with c3:
        opts = vb.CLF_METRICS if is_clf() else vb.REG_METRICS
        keys = list(opts)
        cur = primary()
        S.cfg["primary"] = st.selectbox("Ana metrik (seçim/ayar bununla yapılır)", keys, index=keys.index(cur) if cur in keys else 0,
                                        format_func=lambda k: opts[k], key="w_primary")
        if is_clf():
            W(st.number_input, "Kaçırılan 1 pozitifin maliyeti", "cost_fn", 5.0, min_value=0.0, step=1.0)
            W(st.number_input, "1 gereksiz alarmın maliyeti", "cost_fp", 1.0, min_value=0.0, step=1.0)
    st.divider()
    st.subheader("Kütüphane durumu")
    rows = []
    for m in ["sklearn", "optuna", "xgboost", "shap", "plotly", "streamlit"]:
        try:
            mod = importlib.import_module(m)
            rows.append({"Kütüphane": m, "Durum": "✅ yüklü", "Sürüm": getattr(mod, "__version__", "?")})
        except Exception:
            rows.append({"Kütüphane": m, "Durum": "❌ yok (pip install " + m + ")", "Sürüm": "-"})
    show_df(pd.DataFrame(rows))
    st.subheader("Hangi projede hangi adımlar? (Yol haritası)")
    tip = st.selectbox("Proje tipi", list(vb.YOL_HARITASI), key="w_roadmap")
    for line in vb.YOL_HARITASI[tip]:
        st.write("➡️", line)
    st.divider()
    st.subheader("Ayarları kaydet / yükle")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.download_button("Ayarları indir (JSON)", json.dumps(S.cfg, ensure_ascii=False, indent=1, default=str),
                           file_name="proje_ayarlari.json", mime="application/json")
    with cc2:
        up = st.file_uploader("Ayar dosyası yükle", type=["json"], key="cfg_upload")
        if up is not None and st.button("Ayarları uygula", key="apply_cfg"):
            S.cfg.update(json.loads(up.getvalue().decode("utf-8")))
            reset_from(3)
            st.success("Ayarlar yüklendi.")
    S.done.add(1)
    st.info("Sonraki adım: **2. Veri** bölümünde dosyanı yükle (ya da demo veriyi dene).")


# ----------------------------------------------------------------------------
# Bölüm 2: Veri
# ----------------------------------------------------------------------------
def load_uploaded(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(file)
        sheet = st.selectbox("Sayfa", xl.sheet_names, key="w_sheet") if len(xl.sheet_names) > 1 else xl.sheet_names[0]
        return xl.parse(sheet)
    sep = st.selectbox("Ayraç", [",", ";", "\t", "|"], key="w_sep")
    enc = st.selectbox("Kodlama", ["utf-8", "cp1254", "latin1"], key="w_enc")
    dec = st.selectbox("Ondalık işareti", [".", ","], key="w_dec")
    return pd.read_csv(file, sep=sep, encoding=enc, decimal=dec)


def set_dataset(df, target=None, task=None):
    S.df_raw = df
    S.df = df.copy()
    S.clean_log = []
    S.cfg["target"] = target
    S.cfg["drop_cols"], S.cfg["date_cols"], S.cfg["ordinals"] = [], [], {}
    S.cfg["ratios"], S.cfg["diffs"], S.cfg["logs"] = [], [], []
    if task:
        S.cfg["task"] = task
        S.cfg["primary"] = None
    S.cfg["pos_label"] = None
    reset_from(3)
    S.done = {1, 2}


def sec2():
    st.header("2. Veri: Yükle, Temizle, Denetle")
    how_to("Veriyi yükle, **kirli veriyi** temizle ve otomatik denetimi oku. Denetim; eksik veri, kimlik/sabit sütun, tarih ve "
           "sızıntı şüphesi, hedef dengesizliği gibi konularda yorum ve sonraki adım önerir.\n\n"
           "**Sık hatalar:** Hedefi sızdıran sütunları fark etmemek; tekrar eden satırları silmemek (aynı satır train ve test'e düşer); "
           "'1.250,00 TL' gibi sayıları metin bırakmak.")
    up = st.file_uploader("Veri dosyası (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"], key="data_upload")
    if up is not None:
        df = load_uploaded(up)
        if st.button("Bu dosyayı projeye yükle", key="btn_load"):
            set_dataset(df)
            st.success(f"Yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun.")
    c1, c2 = st.columns(2)
    if c1.button("Demo veri (sınıflandırma: müşteri kaybı)", key="btn_demo_clf"):
        set_dataset(vb.demo_data("classification"), target="churn", task="classification")
    if c2.button("Demo veri (regresyon: konut fiyatı)", key="btn_demo_reg"):
        set_dataset(vb.demo_data("regression"), target="fiyat", task="regression")
    if "df_raw" not in S:
        st.info("Başlamak için dosya yükle ya da demo veriyi seç.")
        st.stop()

    df = S.df
    t1, t2, t3 = st.tabs(["Denetim ve Yorum", "Temizlik Araçları", "Dağılımlar"])
    with t1:
        m1, m2, m3 = st.columns(3)
        m1.metric("Satır", df.shape[0])
        m2.metric("Sütun", df.shape[1])
        m3.metric("Eksik hücre %", f"{df.isna().mean().mean()*100:.1f}")
        show_df(df.head(10))
        cols = list(df.columns)
        cur = S.cfg["target"] if S.cfg["target"] in cols else cols[-1]
        S.cfg["target"] = st.selectbox("Hedef sütun (denetim için)", cols, index=cols.index(cur), key="w_target_audit")
        audit = vb.audit_data(df, S.cfg["target"], S.cfg["task"], S.cfg["drop_cols"], S.cfg["date_cols"])
        S.audit = audit
        show_df(audit["table"])
        box(audit["comments"], audit["steps"])
        S.done.add(2)
    with t2:
        st.caption("Temizlik HER ZAMAN ham veriden başlar; seçimlerini değiştirip tekrar uygulayabilirsin.")
        raw = S.df_raw
        a1, a2 = st.columns(2)
        strip = a1.checkbox("Metinlerdeki baştaki/sondaki boşlukları sil", True, key="w_strip")
        dups = a2.checkbox("Tekrar eden satırları sil", True, key="w_dups")
        objs = raw.select_dtypes(include="object").columns.tolist()
        sug_num = S.get("audit", {}).get("numeric_text", [])
        num_text = st.multiselect("Metni sayıya çevir ('1.250,00 TL', '%20', '3,5 gün'...)", objs, default=[c for c in sug_num if c in objs], key="w_numtext")
        norm = st.multiselect("Kategori yazımlarını birleştir (İstanbul/ISTANBUL/Istanbul -> aynı)", [c for c in objs if c not in num_text], key="w_norm")
        sug_dates = S.get("audit", {}).get("dates", [])
        dts = st.multiselect("Tarih sütunlarını tarih tipine çevir", raw.columns.tolist(), default=[c for c in sug_dates if c in raw.columns], key="w_dtcols")
        st.markdown("**Geçerli değer aralığı (dışındakiler eksik yapılır)**")
        numeric_cols = raw.select_dtypes("number").columns.tolist()
        rules = S.get("range_rules", {})
        r1, r2, r3, r4 = st.columns([2, 1, 1, 1])
        rc = r1.selectbox("Sütun", numeric_cols if numeric_cols else ["-"], key="w_rc")
        rlo = r2.number_input("Alt sınır", value=0.0, key="w_rlo")
        rhi = r3.number_input("Üst sınır", value=100.0, key="w_rhi")
        if r4.button("Kural ekle", key="btn_rule") and numeric_cols:
            rules[rc] = (rlo, rhi)
            S.range_rules = rules
        if rules:
            st.write("Kurallar:", {k: list(v) for k, v in rules.items()})
            if st.button("Kuralları temizle", key="btn_rule_clear"):
                S.range_rules = {}
        if st.button("Temizliği uygula", key="btn_clean"):
            d2, log = vb.clean_dataframe(raw, strip, dups, num_text, norm, S.get("range_rules", {}), dts)
            S.df, S.clean_log = d2, log
            reset_from(3)
            S.done = {1, 2}
            st.rerun()          # denetimi temizlenmiş veriyle yeniden hesapla
        for line in S.get("clean_log", []):
            st.write("✔", line)
    with t3:
        col = st.selectbox("Sütun", df.columns.tolist(), key="w_dist_col")
        if pd.api.types.is_numeric_dtype(df[col]):
            show_fig(px.histogram(df, x=col, marginal="box", title=f"{col} dağılımı"))
        else:
            vc = df[col].value_counts().head(25).reset_index()
            vc.columns = [col, "adet"]
            show_fig(px.bar(vc, x=col, y="adet", title=f"{col} sınıf dağılımı (ilk 25)"))
        miss = (df.isna().mean() * 100).sort_values(ascending=False)
        miss = miss[miss > 0]
        if len(miss):
            show_fig(px.bar(miss.reset_index(), x="index", y=0, labels={"index": "Sütun", "0": "Eksik %"}, title="Eksik değer oranları"))


# ----------------------------------------------------------------------------
# Bölüm 3: Hedef ve özellikler
# ----------------------------------------------------------------------------
def sec3():
    st.header("3. Hedef ve Özellikler")
    how_to("Hedefi tanımla, modele girmeyecek sütunları (kimlik + **sızıntı**) seç, tarihleri ve sıralı kategorileri belirt, istersen yeni özellik türet. "
           "Hiçbiri veriyi değiştirmez; hepsi pipeline'a gömülür ve yeni veride aynen uygulanır.\n\n"
           "**Sık hatalar:** Sonuçla aynı anda/sonra oluşan sütunu bırakmak (sızıntı); tarihte `today` kullanmak (burada sabit referans tarihi var); "
           "hedefi yanlış tanımlamak (pozitif sınıf ne?).")
    need("df")
    df = S.df
    cols = list(df.columns)
    tgt = S.cfg["target"] if S.cfg["target"] in cols else cols[-1]
    new_t = st.selectbox("Hedef sütun", cols, index=cols.index(tgt), key="w_target")
    if new_t != S.cfg["target"]:
        S.cfg["target"] = new_t
        S.cfg["pos_label"] = None
        reset_from(3)
    tser = df[new_t].dropna()
    if is_clf():
        classes = sorted(tser.unique().tolist(), key=str)
        if len(classes) != 2:
            st.error(f"Hedefte {len(classes)} sınıf var. Sınıflandırma için hedef ikili olmalı; hedefi 0/1'e indirge ya da 'Regresyon' seç.")
        else:
            cur = S.cfg["pos_label"] if S.cfg["pos_label"] in classes else classes[-1]
            S.cfg["pos_label"] = st.selectbox("Pozitif sınıf (1 kabul edilecek, yani 'olay')", classes, index=classes.index(cur), key="w_pos")
            st.caption(f"Kodlama: {S.cfg['pos_label']} → 1, diğeri → 0 (alfabetik sıraya bağlı değil, senin seçtiğin).")
    elif not pd.api.types.is_numeric_dtype(tser):
        st.error("Regresyon için hedef sayısal olmalı (Bölüm 2'de metni sayıya çevirebilirsin).")

    audit = S.get("audit", {})
    default_drop = [c for c in (audit.get("id_like", []) + audit.get("constant", [])) if c in cols and c != new_t]
    cur_drop = [c for c in (S.cfg["drop_cols"] or default_drop) if c in cols and c != new_t]
    S.cfg["drop_cols"] = st.multiselect("Modele GİRMEYECEK sütunlar (kimlik + sızıntı)", [c for c in cols if c != new_t], default=cur_drop, key="w_drop")
    leaks = [c for c in audit.get("leak_names", []) if c not in S.cfg["drop_cols"]]
    if leaks:
        st.warning(f"İsimleri sonuçla ilgili görünen sütunlar henüz atılmadı: {leaks}. 'Tahmin anında bu bilgiyi biliyor olur muydum?' diye sor.")

    remaining = [c for c in cols if c != new_t and c not in S.cfg["drop_cols"]]
    st.subheader("Tarih sütunları")
    sug = [c for c in audit.get("dates", []) if c in remaining]
    S.cfg["date_cols"] = st.multiselect("Tarih sütunları (yıl/ay/haftanın günü/referanstan gün farkı türetilir)", remaining,
                                        default=[c for c in (S.cfg["date_cols"] or sug) if c in remaining], key="w_dates")
    if S.cfg["date_cols"]:
        use_ref = st.checkbox("Sabit referans tarihi kullan (önerilir: tahmin tarihi/veri alım tarihi)", value=bool(S.cfg["ref_date"]), key="w_useref")
        if use_ref:
            mx = pd.concat([pd.to_datetime(df[c], errors="coerce", dayfirst=True) for c in S.cfg["date_cols"]]).max()
            default_ref = pd.Timestamp(S.cfg["ref_date"]) if S.cfg["ref_date"] else (mx if pd.notna(mx) else pd.Timestamp.today())
            S.cfg["ref_date"] = str(st.date_input("Referans tarih", value=default_ref.date(), key="w_ref").isoformat())
        else:
            S.cfg["ref_date"] = None
            st.caption("Referans verilmezse eğitim verisindeki en son tarih kullanılır (sabit kalır, `today` kullanılmaz).")

    st.subheader("Sıralı (ordinal) kategoriler")
    cat_cols = [c for c in remaining if df[c].dtype == "object" and c not in S.cfg["date_cols"]]
    ords = st.multiselect("Doğal sırası olan kategorik sütunlar (eğitim düzeyi, kademe...)", cat_cols,
                          default=[c for c in S.cfg["ordinals"] if c in cat_cols], key="w_ords")
    new_ords = {}
    for c in ords:
        uniq = [str(u) for u in df[c].dropna().unique()]
        saved = S.cfg["ordinals"].get(c) or sorted(uniq)
        txt = st.text_input(f"'{c}' sırası (küçükten büyüğe, virgülle)", value=", ".join(saved), key=f"w_ord_{c}")
        order = [x.strip() for x in txt.split(",") if x.strip()]
        unknown = [u for u in uniq if u not in order]
        if unknown:
            st.warning(f"'{c}': sırada olmayan değerler var: {unknown[:6]} (bunlar -1 olarak kodlanır).")
        new_ords[c] = order
    S.cfg["ordinals"] = new_ords

    st.subheader("Yeni özellik türetme (isteğe bağlı)")
    num_cols = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
        a = f1.selectbox("A", num_cols, key="w_fa")
        b = f2.selectbox("B", num_cols, index=min(1, len(num_cols) - 1), key="w_fb")
        kind = f3.selectbox("İşlem", ["oran (A/B)", "fark (A-B)"], key="w_fk")
        if f4.button("Ekle", key="btn_addfe"):
            if kind.startswith("oran"):
                S.cfg["ratios"].append({"a": a, "b": b, "name": f"{a}_per_{b}"})
            else:
                S.cfg["diffs"].append({"a": a, "b": b, "name": f"{a}_minus_{b}"})
    S.cfg["logs"] = st.multiselect("Log dönüşümü uygulanacak (çarpık) sayısal sütunlar (yeni sütun olarak eklenir)", num_cols,
                                   default=[c for c in S.cfg["logs"] if c in num_cols], key="w_logs")
    made = [r["name"] for r in S.cfg["ratios"] + S.cfg["diffs"]]
    if made:
        st.write("Türetilen özellikler:", made)
        if st.button("Türetilenleri temizle", key="btn_clearfe"):
            S.cfg["ratios"], S.cfg["diffs"] = [], []

    if st.button("Hedef ve özellikleri onayla", key="btn_prepare"):
        d = df[df[new_t].notna()].copy()
        if is_clf():
            if d[new_t].nunique() != 2:
                st.error("İkili hedef gerekli.")
                st.stop()
            y = (d[new_t] == S.cfg["pos_label"]).astype(int)
        else:
            y = pd.to_numeric(d[new_t], errors="coerce")
            ok = y.notna()
            d, y = d[ok], y[ok]
        extra_cols = [c for c in {S.cfg.get("group_col"), S.cfg.get("time_col")} if c and c in d.columns]
        S.extra = d[extra_cols].copy() if extra_cols else pd.DataFrame(index=d.index)
        S.X = d.drop(columns=[c for c in S.cfg["drop_cols"] + [new_t] if c in d.columns])
        S.y = y
        reset_from(4)
        S.done = {1, 2, 3}
        comments, steps = [], []
        comments.append(f"Modele {S.X.shape[1]} ham sütun ve {S.X.shape[0]} satır girecek.")
        if is_clf():
            comments.append(f"Pozitif sınıf oranı %{y.mean()*100:.1f}. Hedefin tanımı sonucu tamamen değiştirir; tanımı iş birimiyle teyit et.")
        steps.append("Sızıntı kontrolünü (Bölüm 6) atlama: skorlar şüpheli derecede yüksekse sebep genelde sızıntıdır.")
        box(comments, steps)
    if "X" in S:
        show_df(S.X.head(8))


# ----------------------------------------------------------------------------
# Bölüm 4: Bölme
# ----------------------------------------------------------------------------
def sec4():
    st.header("4. Train / Test Bölme")
    how_to("Test seti modelin **hiç görmediği** veridir; sadece en sonda kullanılır. Ayar ve seçimler train içindeki çapraz doğrulamayla yapılır.\n\n"
           "**Sık hatalar:** Aynı kişi/müşteri birden çok satırdaysa rastgele bölmek (sızıntı → 'group' seç); zamana bağlı veride rastgele bölmek "
           "(geleceği görmüş olursun → 'time' seç); test setiyle model seçmek.")
    need("X", "y")
    mode = st.radio("Bölme türü", ["random", "group", "time"], index=["random", "group", "time"].index(S.cfg["split_mode"]),
                    format_func=lambda m: {"random": "Rastgele", "group": "Gruba göre (aynı kişi tek tarafta)", "time": "Zamana göre (eski→yeni)"}[m],
                    horizontal=True, key="w_mode")
    S.cfg["split_mode"] = mode
    raw_cols = list(S.df.columns)
    if mode == "group":
        S.cfg["group_col"] = st.selectbox("Grup sütunu (örn. müşteri_id)", raw_cols, key="w_group")
    if mode == "time":
        S.cfg["time_col"] = st.selectbox("Zaman sütunu", raw_cols, key="w_time")
    stratify = st.checkbox("Sınıf oranını koru (stratified)", True, key="w_strat") if is_clf() else False
    if st.button("Veriyi böl", key="btn_split"):
        ex = pd.DataFrame(index=S.X.index)
        if mode == "group":
            ex[S.cfg["group_col"]] = S.df.loc[S.X.index, S.cfg["group_col"]] if S.cfg["group_col"] in S.df.columns else np.nan
        if mode == "time":
            ex[S.cfg["time_col"]] = pd.to_datetime(S.df.loc[S.X.index, S.cfg["time_col"]], errors="coerce", dayfirst=True)
        grp = ex[S.cfg["group_col"]] if mode == "group" else None
        tm = ex[S.cfg["time_col"]] if mode == "time" else None
        a, b, c, d, g = vb.split_data(S.X, S.y, S.cfg["task"], mode, S.cfg["test_size"] / 100, S.cfg["seed"], grp, tm, stratify)
        S.X_train, S.X_test, S.y_train, S.y_test, S.groups_train = a, b, c, d, g
        reset_from(7)
        S.done = {1, 2, 3, 4}
    if "X_train" in S:
        c1, c2 = st.columns(2)
        c1.metric("Train satır", S.X_train.shape[0])
        c2.metric("Test satır", S.X_test.shape[0])
        comments, steps = [f"Mod: {mode}. Test seti sadece Bölüm 9'da, bir kez kullanılacak."], []
        if is_clf():
            comments.append(f"Pozitif oran train %{S.y_train.mean()*100:.1f}, test %{S.y_test.mean()*100:.1f}.")
            npos = int(S.y_test.sum())
            if npos:
                comments.append(f"Test'te {npos} pozitif var. Recall gibi bir oranın %95 güven aralığı yaklaşık ±%{1.96*np.sqrt(.25/npos)*100:.0f} genişliğinde (~1/√n).")
                if npos < 50:
                    steps.append("Test'teki pozitif sayısı az: skorları tek rakam değil güven aralığıyla oku (Bölüm 9 verir).")
        idl = [c for c in S.cfg["drop_cols"] if vb.looks_like_id(c) and S.df[c].nunique() < 0.9 * len(S.df)]
        if idl and mode == "random":
            comments.append(f"{idl} sütununda tekrar eden değerler var: aynı kişi birden çok satırda olabilir; rastgele bölmede hem train hem test'e düşer.")
            steps.append(f"Bölme türünü 'Gruba göre' yap ve grup sütunu olarak '{idl[0]}' seç.")
        box(comments, steps)


# ----------------------------------------------------------------------------
# Bölüm 5: Ön işleme
# ----------------------------------------------------------------------------
def sec5():
    st.header("5. Ön İşleme")
    how_to("Eksik doldurma, aykırı baskılama, ölçekleme ve kodlama pipeline'ın içindedir; **sadece train'den öğrenilir**, test ve yeni veriye aynen uygulanır "
           "(sızıntı olmaz).\n\n**Sık hatalar:** Ölçekleyiciyi tüm veriye fit etmek; kategorileri train+test birlikte kodlamak; ağaç modellerine gereksiz ölçekleme yapmak.")
    need("X_train")
    p = dict(vb.DEFAULT_PREP)
    p.update(S.cfg.get("prep", {}))
    c1, c2, c3 = st.columns(3)
    with c1:
        p["num_impute"] = st.selectbox("Sayısal eksik doldurma", ["median", "mean"], index=["median", "mean"].index(p["num_impute"]), key="w_impute")
        p["missing_indicator"] = st.checkbox("'Eksikti' bayrağı ekle", p["missing_indicator"], key="w_missind")
        use_w = st.checkbox("Uç değerleri baskıla (winsorize)", bool(p["winsor"]), key="w_usew")
        if use_w:
            lo, hi = st.slider("Yüzdelik sınırlar", 0.0, 1.0, tuple(p["winsor"]) if p["winsor"] else (0.01, 0.99), 0.005, key="w_wins")
            p["winsor"] = (lo, hi)
        else:
            p["winsor"] = None
    with c2:
        p["scaler"] = st.selectbox("Ölçekleme", ["auto", "standard", "robust", "minmax", "none"], index=["auto", "standard", "robust", "minmax", "none"].index(p["scaler"]),
                                   help="auto: sadece ölçek isteyen modeller (lojistik, SVM, KNN...) için standart ölçekleme", key="w_scaler")
        p["rare_freq"] = st.slider("Nadir kategori eşiği (altı 'diğer')", 0.0, 0.2, float(p["rare_freq"]), 0.005, key="w_rare")
    with c3:
        p["high_card_threshold"] = st.slider("Çok sınıflı sayılma eşiği", 5, 100, int(p["high_card_threshold"]), key="w_hct")
        p["high_card_method"] = st.selectbox("Çok sınıflı kategori yöntemi", ["target", "frequency", "drop"], index=["target", "frequency", "drop"].index(p["high_card_method"]),
                                             help="target: kategorinin hedef ortalaması (CV ile sızıntısız); frequency: görülme oranı; drop: at", key="w_hcm")
    S.cfg["prep"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items() if k not in ("overrides", "ordinals")}
    p["ordinals"] = S.cfg["ordinals"]
    Xf = vb.FeatureBuilder(**fe_cfg()).fit(S.X_train).transform(S.X_train)
    auto = vb.group_columns(Xf, {**p, "overrides": {}})
    st.subheader("Sütun tipleri (gerekirse değiştir)")
    rows = [{"Sütun": c, "Otomatik": g, "Seçim": S.cfg.get("prep_over", {}).get(c, g)} for g, cs in auto.items() for c in cs]
    edited = st.data_editor(pd.DataFrame(rows), column_config={"Seçim": st.column_config.SelectboxColumn(options=["numeric", "binary", "ordinal", "nominal", "high", "drop"])},
                            disabled=["Sütun", "Otomatik"], hide_index=True, key="w_coltypes")
    over = {r["Sütun"]: r["Seçim"] for _, r in edited.iterrows() if r["Seçim"] != r["Otomatik"]}
    S.cfg["prep_over"] = {r["Sütun"]: r["Seçim"] for _, r in edited.iterrows()}
    S.cfg["prep"]["overrides"] = over
    p["overrides"] = over
    ct = vb.build_preprocessor(Xf, p, True, S.cfg["task"], S.cfg["seed"])
    try:
        out = ct.fit_transform(Xf, S.y_train)
        st.success(f"Ön işleme çıktısı: {out.shape[0]} satır × {out.shape[1]} sayısal sütun (ölçekli örnek önizleme).")
    except Exception as e:
        st.error(f"Ön işleme kurulamadı: {e}")
    groups = vb.group_columns(Xf, p)
    comments, steps = ["Eksik sayısal değerler MEDYANLA doldurulur (uç değere ortalamadan dayanıklı) ve 'eksikti' bayrağı eklenir. Ölçekleme z=(x−μ)/σ sadece doğrusal/mesafe tabanlı modeller içindir; ağaçlar ölçekten etkilenmez."], []
    num = groups["numeric"]
    if num:
        sk = Xf[num].skew().abs().sort_values(ascending=False)
        skew_cols = sk[sk > 1].index.tolist()
        if skew_cols:
            comments.append(f"Çok çarpık sayısal sütunlar (|skew|>1): {skew_cols}. Ağaçlarda sorun değil; doğrusal modelde log dönüşümü iyileştirebilir (Bölüm 3).")
        if len(num) > 1:
            cm = Xf[num].corr().abs()
            pairs = [(a, b, round(cm.loc[a, b], 2)) for i, a in enumerate(num) for b in num[i + 1:] if cm.loc[a, b] >= 0.8]
            if pairs:
                comments.append(f"Birbirine çok bağlı (|r|≥0.8) çiftler: {pairs}. Doğrusal modelde katsayıları kararsızlaştırır; ağaçlarda önem puanını böler.")
                steps.append("Çiftlerden birini at ya da Bölüm 10'da önemleri birlikte yorumla.")
            if len(num) <= 25:
                show_fig(px.imshow(Xf[num].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Korelasyon (sayısal sütunlar)"))
    box(comments, steps)
    S.done.add(5)


# ----------------------------------------------------------------------------
# Bölüm 6: Sızıntı avcısı
# ----------------------------------------------------------------------------
def sec6():
    st.header("6. Sızıntı Avcısı")
    how_to("Her özelliği **tek başına** sığ bir ağaca verip skorlar. Tek bir sütunla aşırı yüksek skor, o sütunun hedefle aynı anda/sonra oluştuğunu düşündürür.\n\n"
           "**Sınırı:** birden fazla sütunun birleşiminden doğan sızıntıyı (örn. kıdem + kıdeme bağlı hak) yakalayamaz; bunun için Bölüm 10'daki önem listesini sorgula.")
    need("X_train")
    if st.button("Taramayı başlat", key="btn_leak"):
        bar = st.progress(0.0)
        res, metric, thr = vb.leakage_scan(S.X_train, S.y_train, fe_cfg(), prep_cfg(), S.cfg["task"], S.cfg["seed"],
                                           progress=lambda i, n: bar.progress(i / n))
        S.leak = (res, metric, thr)
    if "leak" in S:
        res, metric, thr = S.leak
        fig = px.bar(res.head(25).iloc[::-1], x="Tek başına skor", y="Sütun", orientation="h", color="Şüpheli", color_discrete_map={True: "crimson", False: "steelblue"},
                     title=f"Tek başına {metric} (eşik {thr})")
        fig.add_vline(x=thr, line_dash="dash")
        show_fig(fig)
        sus = res[res["Şüpheli"]]["Sütun"].tolist()
        comments = [f"Ölçüt: tek sütunla {metric} > {thr} şüpheli. {metric}=0.5 (R²=0) katkı yok, 1 hedefi birebir veriyor demektir."]
        steps = []
        if sus:
            comments.append(f"ŞÜPHELİ özellikler: {sus}. Ya gerçekten güçlü sinyal ya da sızıntı.")
            origin = [c for c in S.X_train.columns if any(s == c or s.startswith(c + "_") for s in sus)]
            drop = st.multiselect("Modelden atılacak ORİJİNAL sütunlar (önerilenler seçili)", S.X_train.columns.tolist(), default=origin, key="w_leakdrop")
            if st.button("Seçilenleri at (train/test'ten kaldır)", key="btn_leakdrop"):
                S.X_train, S.X_test = S.X_train.drop(columns=drop), S.X_test.drop(columns=drop)
                S.cfg["drop_cols"] = list(dict.fromkeys(S.cfg["drop_cols"] + drop))
                S.X = S.X.drop(columns=[c for c in drop if c in S.X.columns])
                S.pop("leak", None)
                reset_from(7)
                st.success(f"Atıldı: {drop}. Taramayı tekrar çalıştır.")
            steps.append("Her şüpheli için sor: 'Tahmin anında bu değeri biliyor olur muydum?' Hayırsa at ve taramayı tekrarla.")
        else:
            comments.append("Tek başına aşırı güçlü sütun yok. Bu sızıntı olmadığının KANITI DEĞİL, sadece ilk eleme.")
        show_df(res)
        box(comments, steps)
        S.done.add(6)


# ----------------------------------------------------------------------------
# Bölüm 7: Model karşılaştırma
# ----------------------------------------------------------------------------
def sec7():
    st.header("7. Model Karşılaştırma (Çapraz Doğrulama)")
    how_to("Birkaç modeli aynı çapraz doğrulama katlarında karşılaştırır. **Dummy** (hiçbir şey öğrenmeyen referans) her zaman dahildir: "
           "bir model Dummy'yi anlamlı geçmiyorsa öğrenmiyor demektir.\n\n**Sık hatalar:** Tek bir skora bakmak (±std ve train-CV farkına da bak); "
           "farkı test etmeden 'bu model daha iyi' demek (aşağıda düzeltilmiş t-testi var); dengesiz veride accuracy'ye bakmak.")
    need("X_train")
    names = [n for n in zoo() if n != "Dummy"]
    default = [n for n in ["Logistic Regression", "Ridge", "Random Forest", "HistGradientBoosting", "XGBoost"] if n in names]
    chosen = st.multiselect("Karşılaştırılacak modeller", names, default=S.get("chosen", default), key="w_models")
    st.caption("SVC/KNN büyük veride yavaştır. Süre kolonuna bak.")
    if st.button("Modelleri karşılaştır", key="btn_compare"):
        S.chosen = chosen
        pipes = {"Dummy": make_pipe("Dummy")}
        for n in chosen:
            pipes[n] = make_pipe(n)
        bar = st.progress(0.0, text="Başlıyor...")
        table, folds, errors = vb.compare_models(pipes, S.X_train, S.y_train, get_cv(), S.get("groups_train"), S.cfg["task"], primary(),
                                                 progress=lambda i, n, name: bar.progress(i / n, text=f"{name} bitti"))
        S.cv_table, S.cv_folds_scores, S.cv_errors = table, folds, errors
        S.tuned = {}
        S.done.add(7)
    if "cv_table" in S:
        table = S.cv_table
        for n, e in S.get("cv_errors", {}).items():
            st.error(f"{n} çalışmadı: {e}")
        show_df(table.style.format({c: "{:.3f}" for c in table.columns if c not in ("Model", "Süre (sn)")}))
        lab = vb.CLF_METRICS.get(primary(), vb.REG_METRICS.get(primary(), primary()))
        fig = go.Figure()
        fig.add_bar(x=table["Model"], y=table["ANA (CV)"], error_y=dict(type="data", array=table["Std"]), name=f"CV {lab}")
        fig.add_scatter(x=table["Model"], y=table["ANA (Train)"], mode="markers", marker=dict(size=11, symbol="diamond"), name="Train (ezber ölçüsü)")
        fig.update_layout(title=f"CV ± std ve train skoru ({lab})", yaxis_title=lab)
        show_fig(fig)
        n_train = len(S.X_train)
        c, s = vb.model_comparison_comments(table, S.cv_folds_scores, primary(), n_train, S.cfg["cv_folds"])
        real = table[table["Model"] != "Dummy"]
        if not real.empty:
            best = real.sort_values("ANA (CV)", ascending=not vb.higher_is_better(primary())).iloc[0]
            if is_clf() and "ROC-AUC" in real.columns and best["ROC-AUC"] > 0.85:
                c.append(f"{best['Model']} CV ROC-AUC={best['ROC-AUC']:.2f}: gerçek iş problemlerinde 0.85 üstü nadirdir. Tek sütun taraması birden fazla sütunun "
                         "BİRLEŞİMİNDEN doğan sızıntıyı yakalayamaz.")
                s.append("Önem listesinin başındaki sütunları (Bölüm 10) tek tek sorgula: 'Tahmin anında biliyor olur muydum?'")
            if (not is_clf()) and "R²" in real.columns and best["R²"] > 0.95:
                c.append(f"{best['Model']} CV R²={best['R²']:.2f}: çok yüksek; hedefin türevi bir sütun olabilir.")
        box(c, s)


# ----------------------------------------------------------------------------
# Bölüm 8: Hiperparametre ayarı
# ----------------------------------------------------------------------------
def sec8():
    st.header("8. Hiperparametre Ayarı (Optuna)")
    how_to("Seçtiğin modelin ayarlarını Optuna (akıllı arama) ile train içi CV'de bulur. Her denemenin sonucu **anında grafiklenir**; "
           "parametrelerin ne işe yaradığı basit dille yazılır.\n\n**Sık hatalar:** Test setiyle ayar yapmak; sadece en iyi skora bakmak; aralığın kenarında "
           "çıkan değeri kabul etmek; ayarla veri sorununu çözmeye çalışmak.")
    need("X_train")
    pool = [n for n in vb.SEARCH_SPACES if n in zoo()]
    chosen = S.get("chosen", pool)
    cands = [n for n in pool if n in chosen] or pool
    name = st.selectbox("Ayarlanacak model", cands, key="w_tune_model")
    space = {k: dict(v) for k, v in vb.SEARCH_SPACES[name].items()}
    st.markdown("**Arama aralıkları (istersen değiştir)**")
    cols = st.columns(min(len(space), 4))
    for i, (p, s) in enumerate(space.items()):
        with cols[i % len(cols)]:
            if s["type"] == "cat":
                s["choices"] = st.multiselect(p, s["choices"], default=s["choices"], key=f"w_sp_{name}_{p}")
            else:
                cast = int if s["type"] == "int" else float
                s["low"] = st.number_input(f"{p} min", value=cast(s["low"]), key=f"w_lo_{name}_{p}")
                s["high"] = st.number_input(f"{p} max", value=cast(s["high"]), key=f"w_hi_{name}_{p}")
    n_trials = st.slider("Deneme sayısı", 5, 60, 15, key="w_trials")
    if st.button("Optuna'yı başlat", key="btn_tune"):
        bar = st.progress(0.0, text="Aranıyor...")
        res = vb.run_tuning(lambda: make_pipe(name), S.X_train, S.y_train, get_cv(), S.get("groups_train"), primary(), space, n_trials, S.cfg["seed"], prefix(),
                            progress=lambda i, n: bar.progress(i / n, text=f"Deneme {i}/{n}"))
        res["space"] = space
        S.tuned[name] = res
        S.done.add(8)
    if name in S.tuned:
        res = S.tuned[name]
        tr = res["trials"]
        lab = vb.CLF_METRICS.get(primary(), vb.REG_METRICS.get(primary(), primary()))
        hib = vb.higher_is_better(primary())
        best_so_far = tr["CV"].cummax() if hib else tr["CV"].cummin()
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure()
            fig.add_scatter(x=tr["deneme"], y=tr["CV"], mode="markers", name="deneme CV")
            fig.add_scatter(x=tr["deneme"], y=best_so_far, mode="lines", name="şimdiye kadarki en iyi")
            fig.update_layout(title="Optimizasyon geçmişi", xaxis_title="Deneme", yaxis_title=lab)
            show_fig(fig)
        with g2:
            if res["importance"]:
                imp = pd.Series(res["importance"]).sort_values()
                show_fig(px.bar(imp, orientation="h", title="Hangi parametre skoru en çok etkiledi?", labels={"value": "Önem", "index": "Parametre"}))
        params = list(res["best_params"])
        st.markdown("**Parametre etkisi (nokta = deneme, mavi: CV, kırmızı: train; ikisi ayrışıyorsa ezber var)**")
        pc = st.columns(2)
        for i, p in enumerate(params):
            f = go.Figure()
            f.add_scatter(x=tr[p], y=tr["CV"], mode="markers", name="CV", marker=dict(color="royalblue"))
            f.add_scatter(x=tr[p], y=tr["Train"], mode="markers", name="Train", marker=dict(color="tomato", symbol="x"))
            if res["space"][p].get("log"):
                f.update_xaxes(type="log")
            f.update_layout(title=p, yaxis_title=lab, height=300)
            with pc[i % 2]:
                show_fig(f)
        st.success(f"En iyi CV skoru: {res['best_score']:.4f}")
        st.markdown("**Seçilen parametreler ve ne işe yaradıkları**")
        edge = vb.edge_warnings(res["best_params"], res["space"])
        for p, v in res["best_params"].items():
            st.write(f"• **{p} = {v if not isinstance(v, float) else round(v, 5)}** — {vb.PARAM_INFO.get(p, '')}")
        if st.button("Öğrenme eğrisini çiz", key="btn_lc"):
            sizes, trn, cvs = learning_curve(clone(res["model"]), S.X_train, S.y_train, cv=get_cv(), groups=S.get("groups_train"),
                                             train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], scoring=primary(), n_jobs=-1)
            S.lc = (sizes, vb.show_score(trn.mean(axis=1), primary()), vb.show_score(cvs.mean(axis=1), primary()))
        if "lc" in S:
            sizes, trn, cvs = S.lc
            f = go.Figure()
            f.add_scatter(x=sizes, y=trn, mode="lines+markers", name="train")
            f.add_scatter(x=sizes, y=cvs, mode="lines+markers", name="CV")
            f.update_layout(title="Öğrenme eğrisi: daha çok veri işe yarar mı?", xaxis_title="Eğitim örnek sayısı", yaxis_title=lab)
            show_fig(f)
        comments, steps = [], []
        before = S.get("cv_table")
        if before is not None and name in before["Model"].values:
            old = float(before[before["Model"] == name]["ANA (CV)"].iloc[0])
            oldstd = float(before[before["Model"] == name]["Std"].iloc[0])
            gain = abs(res["best_score"] - old)
            better = (res["best_score"] > old) == hib
            comments.append(f"Ayar öncesi CV {old:.3f} → sonrası {res['best_score']:.3f} ({'iyileşti' if better else 'iyileşmedi'}, fark {gain:.3f}).")
            if gain < oldstd:
                comments.append("Kazanç fold std'sinden küçük: ayarın katkısı gürültü düzeyinde. Daha çok deneme yerine yeni özellik türetmek genelde daha verimli.")
                steps.append("Bölüm 3'te oran/fark/log özellikleri türet, sonra Bölüm 7'yi tekrar çalıştır.")
        best_row = tr.sort_values("CV", ascending=not hib).iloc[0]
        comments.append(f"En iyi denemede train-CV farkı {abs(best_row['Train'] - best_row['CV']):.3f}. Fark büyükse ezber var: derinliği küçült, yaprak boyutunu büyüt, ceza artır.")
        if edge:
            comments.append(f"UYARI: {edge} değeri arama aralığının KENARINDA çıktı; gerçek en iyi değer aralığın dışında olabilir.")
            steps.append(f"{edge} için aralığı genişletip aramayı tekrar çalıştır.")
        box(comments, steps)
        st.markdown("**Hiperparametre ayarında sık yapılan hatalar**")
        for m in vb.TUNING_MISTAKES:
            st.write("✗", m)


# ----------------------------------------------------------------------------
# Bölüm 9: Voting / Stacking ve test
# ----------------------------------------------------------------------------
def candidate_models():
    """Ayarlanmışsa ayarlı sürüm, değilse varsayılan ayarlı pipeline."""
    out = {}
    for n in S.get("chosen", []):
        out[n] = make_pipe(n, S.tuned[n]["model"].named_steps["model"] if n in S.tuned else None) if n not in S.tuned else clone(S.tuned[n]["model"])
    return out


def sec9():
    st.header("9. Voting / Stacking ve Test Değerlendirmesi")
    how_to("Tek model, **Voting** (modellerin oylarını/ortalamasını alır) ve **Stacking** (üst bir model tahminleri birleştirir) CV ile karşılaştırılır; "
           "fark std'yi geçmiyorsa **basit model** seçilir. Sonra final model **test setinde bir kez** değerlendirilir.\n\n"
           "**Sık hatalar:** Test setini defalarca kullanıp 'en iyi'yi seçmek (aşağıda sayaç var); birbirine çok benzeyen modelleri oylamak (kazanç yok); "
           "hard voting'de olasılık beklemek.")
    need("X_train", "cv_table")
    cands = candidate_models()
    if not cands:
        st.warning("Önce Bölüm 7'de model seç.")
        st.stop()
    st.write("Aday modeller:", {n: ("ayarlı" if n in S.tuned else "varsayılan") for n in cands})
    members = st.multiselect("Topluluğa girecek modeller (çeşitlilik için doğrusal + ağaç + boosting karıştır)", list(cands), default=list(cands)[:3], key="w_members")
    kinds = ["Yok (sadece tek model)", "Voting", "Stacking"]
    kind = st.radio("Topluluk türü", kinds, horizontal=True, key="w_kind")
    voting = "soft"
    weights = None
    if kind == "Voting":
        if is_clf():
            voting = st.radio("Voting", ["soft", "hard"], horizontal=True, help="soft: olasılık ortalaması (ROC/PR için gerekli); hard: çoğunluk oyu", key="w_vtype")
        wtxt = st.text_input("Ağırlıklar (virgülle, üye sırasıyla; boş = eşit)", "", key="w_weights")
        if wtxt.strip():
            try:
                weights = [float(x) for x in wtxt.split(",")]
                if len(weights) != len(members):
                    st.error("Ağırlık sayısı üye sayısına eşit olmalı.")
                    weights = None
            except ValueError:
                st.error("Ağırlıklar sayı olmalı.")
    if st.button("Adayları CV ile karşılaştır", key="btn_cands"):
        pool = {f"Tek: {n}": clone(p) for n, p in cands.items()}
        if kind != "Yok (sadece tek model)" and len(members) >= 2:
            pool[kind] = vb.build_ensemble(kind, {n: cands[n] for n in members}, S.cfg["task"], weights, voting, S.cfg["seed"])
        table, folds, errors = vb.compare_models(pool, S.X_train, S.y_train, get_cv(), S.get("groups_train"), S.cfg["task"], primary())
        S.cand_pool, S.cand_table, S.cand_folds, S.cand_err = pool, table, folds, errors
    if "cand_table" in S:
        for n, e in S.cand_err.items():
            st.error(f"{n} çalışmadı: {e}")
        table = S.cand_table
        show_df(table.style.format({c: "{:.3f}" for c in table.columns if c not in ("Model", "Süre (sn)")}))
        hib = vb.higher_is_better(primary())
        ordered = table.sort_values("ANA (CV)", ascending=not hib)
        singles = ordered[ordered["Model"].str.startswith("Tek:")]
        recommended = singles.iloc[0]["Model"]
        comments, steps = [], []
        ens = ordered[~ordered["Model"].str.startswith("Tek:")]
        if not ens.empty and not singles.empty:
            es, ss = ens.iloc[0], singles.iloc[0]
            diff = abs(es["ANA (CV)"] - ss["ANA (CV)"])
            better = (es["ANA (CV)"] > ss["ANA (CV)"]) == hib
            t, pval = vb.corrected_ttest(S.cand_folds[es["Model"]], S.cand_folds[ss["Model"]], len(S.X_train), max(len(S.X_train) // S.cfg["cv_folds"], 1))
            if better and diff > ss["Std"] and pval < 0.10:
                recommended = es["Model"]
                comments.append(f"{es['Model']}, en iyi tek modelden {diff:.3f} daha iyi (fold std {ss['Std']:.3f}, düzeltilmiş t-testi p={pval:.3f}): topluluk önerilir.")
            else:
                comments.append(f"{es['Model']} ile en iyi tek model arasındaki fark ({diff:.3f}) std'den ({ss['Std']:.3f}) küçük / anlamsız (p={pval:.3f}) ya da topluluk daha kötü. "
                                "Kazanç kanıtlanamadığı için basit tek model önerilir.")
            comments.append("Voting/Stacking neden işe yarar? Modellerin hataları farklı yerlerde olursa ortalama alınca hatalar birbirini götürür (varyans düşer). "
                            "Modeller birbirine çok benziyorsa kazanç neredeyse sıfırdır; yorumlanabilirlik ve hız kaybına değmez.")
            steps.append("Topluluğa çok benzer modeller koyma; doğrusal + ağaç + boosting çeşitlilik sağlar.")
        else:
            comments.append("Topluluk denenmedi; en iyi tek model önerilir.")
        box(comments, steps)
        final_name = st.selectbox("Final model", list(S.cand_pool), index=list(S.cand_pool).index(recommended), key="w_final")
        if st.button("Final modeli eğit (train)", key="btn_final"):
            model = clone(S.cand_pool[final_name]).fit(S.X_train, S.y_train)
            row = table[table["Model"] == final_name].iloc[0]
            thr, grid, costs = None, None, None
            if is_clf() and hasattr(model, "predict_proba"):
                thr, grid, costs = vb.choose_threshold(clone(S.cand_pool[final_name]), S.X_train, S.y_train, get_cv(), S.get("groups_train"),
                                                       S.cfg["cost_fn"], S.cfg["cost_fp"])
            S.final = dict(model=model, name=final_name, cv=float(row["ANA (CV)"]), cv_std=float(row["Std"]), thr=thr, grid=grid, costs=costs)
            S.pop("eval", None)
            S.done.add(9)
    if "final" in S:
        f = S.final
        st.success(f"Final model: {f['name']} (CV {f['cv']:.3f} ± {f['cv_std']:.3f})")
        if is_clf() and f["thr"] is not None:
            st.caption(f"Eşik, test'e bakılmadan train'in out-of-fold tahminlerinden maliyete göre seçildi: {f['thr']:.2f} "
                       f"(kaçırılan pozitif maliyeti {S.cfg['cost_fn']}, gereksiz alarm {S.cfg['cost_fp']}).")
            thr = st.slider("Eşik (istersen elle değiştir)", 0.02, 0.98, float(f["thr"]), 0.01, key="w_thr")
            S.final["thr"] = thr
            fc = go.Figure()
            fc.add_scatter(x=f["grid"], y=f["costs"], mode="lines", name="toplam maliyet (OOF)")
            fc.add_vline(x=thr, line_dash="dash")
            fc.update_layout(title="Eşik - maliyet eğrisi", xaxis_title="Eşik", yaxis_title="Maliyet")
            show_fig(fc)
        st.warning(f"Test seti şimdiye kadar {S.test_uses} kez kullanıldı. Her kullanım, test skorunu iyimser yapar (sonuca göre model değiştirirsen test artık 'görülmemiş' değildir). "
                   "Test skoruna bakıp model seçmeyi bırak; asıl karar CV'ye dayanmalı.")
        if st.button("Test setinde değerlendir", key="btn_eval"):
            S.test_uses += 1
            thr = S.final["thr"] if S.final["thr"] is not None else 0.5
            if is_clf():
                S.eval = vb.eval_classification(f["model"], S.X_test, S.y_test, thr)
            else:
                S.eval = vb.eval_regression(f["model"], S.X_test, S.y_test, S.y_train)
            S.done.add(9)
        if "eval" in S:
            show_eval()


def show_eval():
    r, f = S.eval, S.final
    if is_clf():
        c1, c2, c3 = st.columns(3)
        if r["has_proba"]:
            c1.metric("ROC-AUC", f"{r['roc']:.3f}", help=f"%95 aralık {r['roc_ci'][0]:.3f}-{r['roc_ci'][1]:.3f}")
            c2.metric("PR-AUC", f"{r['pr']:.3f}", help=f"%95 aralık {r['pr_ci'][0]:.3f}-{r['pr_ci'][1]:.3f}; rastgele {r['prevalence']:.3f}")
            c3.metric("Brier", f"{r['brier']:.3f}", help=f"'hep ortalamayı söyle' referansı {r['prevalence']*(1-r['prevalence']):.3f}")
            g1, g2, g3 = st.columns(3)
            fpr, tpr = r["roc_curve"]
            fr = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines"))
            fr.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
            fr.update_layout(title="ROC", height=300)
            rec, pre = r["pr_curve"]
            fp = go.Figure(go.Scatter(x=rec, y=pre, mode="lines"))
            fp.add_hline(y=r["prevalence"], line_dash="dash")
            fp.update_layout(title="Precision-Recall", height=300)
            with g1:
                show_fig(fr)
            with g2:
                show_fig(fp)
            with g3:
                if r["calib"] is not None:
                    fc = go.Figure(go.Scatter(x=r["calib"][0], y=r["calib"][1], mode="lines+markers"))
                    fc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
                    fc.update_layout(title="Kalibrasyon", xaxis_title="tahmin", yaxis_title="gerçek oran", height=300)
                    show_fig(fc)
            st.markdown("**Risk dilimi tablosu (1 = en yüksek skorlu %10)**")
            show_df(r["lift"].round(3))
        else:
            st.info("Hard voting olasılık üretmez: ROC/PR/kalibrasyon hesaplanamadı.")
        cm = r["cm"]
        show_fig(px.imshow(cm, text_auto=True, x=["Tahmin 0", "Tahmin 1"], y=["Gerçek 0", "Gerçek 1"], color_continuous_scale="Blues", title="Karışıklık matrisi"))
        show_df(r["report"].round(3))
        c, s = vb.classification_comments(r, f["thr"] or 0.5, f["cv"], f["cv_std"], primary())
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{r['rmse']:.3f}", help=f"%95 aralık {r['rmse_ci'][0]:.3f}-{r['rmse_ci'][1]:.3f}; baseline {r['baseline_rmse']:.3f}")
        c2.metric("MAE", f"{r['mae']:.3f}")
        c3.metric("R²", f"{r['r2']:.3f}")
        g1, g2, g3 = st.columns(3)
        with g1:
            fa = px.scatter(x=S.y_test, y=r["p"], labels={"x": "Gerçek", "y": "Tahmin"}, title="Gerçek vs tahmin", opacity=0.4)
            lo, hi = float(min(np.min(S.y_test), r["p"].min())), float(max(np.max(S.y_test), r["p"].max()))
            fa.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="red", dash="dash"))
            show_fig(fa)
        with g2:
            show_fig(px.histogram(x=r["err"], nbins=40, title="Artıklar (gerçek − tahmin)"))
        with g3:
            show_fig(px.scatter(x=r["p"], y=np.abs(r["err"]), labels={"x": "Tahmin", "y": "|hata|"}, title="Tahmin büyüdükçe hata?", opacity=0.4))
        st.markdown("**Gerçek değer çeyreğine göre ortalama mutlak hata**")
        show_df(r["quartile"].round(3))
        c, s = vb.regression_comments(r, f["cv"], f["cv_std"], S.cfg["log_target"])
    box(c, s)
    out = pd.DataFrame({"gercek": np.asarray(S.y_test), "tahmin": r["p"] if (not is_clf() or r["has_proba"]) else r["pred"]}, index=S.y_test.index)
    st.download_button("Test tahminlerini indir (CSV)", out.to_csv().encode("utf-8"), "test_tahminleri.csv", "text/csv")


# ----------------------------------------------------------------------------
# Bölüm 10: İstatistiksel güvenilirlik ve yorum
# ----------------------------------------------------------------------------
def tree_member(model):
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
                                  GradientBoostingClassifier, GradientBoostingRegressor)
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    trees = (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
             GradientBoostingRegressor, DecisionTreeClassifier, DecisionTreeRegressor)
    try:
        from xgboost import XGBClassifier, XGBRegressor
        trees = trees + (XGBClassifier, XGBRegressor)
    except Exception:
        pass
    pipes = list(model.named_estimators_.values()) if hasattr(model, "named_estimators_") else [model]
    for p in pipes:
        inner = p.named_steps["model"]
        inner = getattr(inner, "regressor_", inner)
        if isinstance(inner, trees):
            return p, inner
    return None, None


def sec10():
    st.header("10. İstatistiksel Güvenilirlik ve Yorumlama")
    how_to("Bu sonuç **gerçekten güvenilir mi?** Burada modelin şansı geçip geçmediği (permütasyon testi), baseline'dan anlamlı üstün olup olmadığı, ezber (train-CV-test farkı), "
           "kalibrasyon ve modelin neye baktığı incelenir.\n\n**Sık hatalar:** Önem puanını nedensellik sanmak; korelasyonlu sütunların önemini tek tek yorumlamak; "
           "SHAP/önem listesinde bir sütun baskınsa sızıntıyı düşünmemek.")
    need("final", "eval")
    r, f = S.eval, S.final
    comments, steps = [], []
    st.subheader("Güvenilirlik karnesi")
    if is_clf() and r["has_proba"]:
        prev = r["prevalence"]
        sig_pr = r["pr_ci"][0] > prev
        sig_roc = r["roc_ci"][0] > 0.5
        c1, c2, c3 = st.columns(3)
        c1.metric("Test PR-AUC", f"{r['pr']:.3f}", delta=f"baseline {prev:.3f}")
        c2.metric("CV PR-AUC (final)", f"{f['cv']:.3f}" if primary() == "average_precision" else "-")
        c3.metric("Test ROC-AUC", f"{r['roc']:.3f}", delta="baseline 0.500")
        comments.append(f"PR-AUC alt güven sınırı {r['pr_ci'][0]:.3f} {'>' if sig_pr else '≤'} baseline {prev:.3f}: model rastgeleden " + ("istatistiksel olarak anlamlı biçimde iyi." if sig_pr else "ayırt edilemiyor."))
        comments.append(f"ROC-AUC alt sınırı {r['roc_ci'][0]:.3f} {'>' if sig_roc else '≤'} 0.5: " + ("sıralama gücü şans değil." if sig_roc else "şans düzeyinde olabilir."))
        if primary() == "average_precision" and r["pr"] < f["cv"] - 2 * f["cv_std"]:
            comments.append(f"Test PR-AUC ({r['pr']:.3f}), CV tahmininin ({f['cv']:.3f}) 2 std altında: seçim sürecinde ezber ya da dönem farkı.")
            steps.append("Veri zamana bağlıysa 'time' bölmeyle tekrar dene; değilse modeli sadeleştir.")
    elif not is_clf():
        sig = r["rmse_ci"][1] < r["baseline_rmse"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Test RMSE", f"{r['rmse']:.3f}", delta=f"baseline {r['baseline_rmse']:.3f}", delta_color="off")
        c2.metric("R²", f"{r['r2']:.3f}")
        c3.metric("RMSE güven aralığı", f"{r['rmse_ci'][0]:.2f}-{r['rmse_ci'][1]:.2f}")
        comments.append(f"RMSE üst güven sınırı {r['rmse_ci'][1]:.2f} {'<' if sig else '≥'} baseline {r['baseline_rmse']:.2f}: model ortalamayı tahmin etmekten " + ("anlamlı biçimde iyi." if sig else "ayırt edilemiyor."))
    box(comments, steps)

    t1, t2, t3, t4 = st.tabs(["Permütasyon testi", "Özellik önemi", "Kısmi bağımlılık", "SHAP"])
    with t1:
        st.caption("Hedef etiketleri karıştırılıp model tekrar tekrar eğitilir. Gerçek skor, karıştırılmış skorların neredeyse hepsinden iyiyse sonuç şans değildir (p küçük).")
        npm = st.slider("Permütasyon sayısı (yavaş!)", 10, 100, 20, key="w_nperm")
        if st.button("Permütasyon testini çalıştır", key="btn_perm"):
            with st.spinner("Çalışıyor..."):
                score, perms, p = permutation_test_score(clone(S.cand_pool[f["name"]]), S.X_train, S.y_train, scoring=primary(), cv=get_cv(),
                                                         groups=S.get("groups_train"), n_permutations=npm, random_state=S.cfg["seed"], n_jobs=-1)
            S.perm = (vb.show_score(score, primary()), vb.show_score(np.asarray(perms), primary()), p)
        if "perm" in S:
            score, perms, p = S.perm
            fh = px.histogram(x=perms, nbins=20, title="Karıştırılmış etiketlerle elde edilen skorlar (şans dağılımı)")
            fh.add_vline(x=score, line_color="red", annotation_text="gerçek skor")
            show_fig(fh)
            st.info(f"Permütasyon p-değeri = {p:.4f}. " + ("p < 0.05: model şansın üstünde öğrenmiş." if p < 0.05 else "p ≥ 0.05: şans düzeyinden ayırt edilemiyor; sonuca güvenme."))
    with t2:
        if st.button("Permütasyon önemini hesapla (test seti)", key="btn_imp"):
            ri = permutation_importance(f["model"], S.X_test, S.y_test, scoring=primary(), n_repeats=8, random_state=S.cfg["seed"], n_jobs=-1)
            S.imp = pd.DataFrame({"Özellik": S.X_test.columns, "Önem": ri.importances_mean, "Std": ri.importances_std}).sort_values("Önem", ascending=False)
        if "imp" in S:
            imp = S.imp
            top = imp.head(15).iloc[::-1]
            show_fig(px.bar(top, x="Önem", y="Özellik", orientation="h", error_x="Std", title="Permütasyon önemi (sütun karıştırılınca metrikteki düşüş)"))
            total = imp["Önem"].clip(lower=0).sum()
            ic, isteps = [], []
            if total > 0:
                share = imp["Önem"].iloc[0] / total * 100
                ic.append(f"En önemli özellik '{imp['Özellik'].iloc[0]}' toplam önemin %{share:.0f}'ini taşıyor.")
                if share > 50:
                    ic.append("Tek özellik önemin yarısından fazlasını taşıyor: bir sütuna aşırı bağımlı. Sızıntı olabilir mi? Ya da gerçekten baskın bir sinyal mi?")
                    isteps.append("Baskın sütunu çıkarıp tekrar eğit; skor çökerse sütun ya sızıntıdır ya da tek sinyaldir.")
            ic.append("Önem ≈ 0 ya da negatif: sütun katkı vermiyor (gürültü). Korelasyonlu sütunlar önemi paylaşır, tek tek küçük görünür.")
            ic.append("ÖNEMLİ: Bu bir İLİŞKİdir, NEDENSELLİK değil. 'X'i değiştirirsek sonuç değişir' demek için kontrollü deney (A/B test) gerekir.")
            isteps.append("En önemli 3 özelliği iş birimiyle konuş: mantıklı mı, aksiyona çevrilebilir mi?")
            box(ic, isteps)
    with t3:
        cols = [c for c in S.X_test.columns if pd.api.types.is_numeric_dtype(S.X_test[c])]
        if cols:
            col = st.selectbox("Özellik", cols, key="w_pdp")
            if st.button("Çiz", key="btn_pdp"):
                S.pdp = (col,) + vb.pdp_data(f["model"], S.X_test.head(1000), col)
            if "pdp" in S:
                c_, gx, gy = S.pdp
                show_fig(px.line(x=gx, y=gy, markers=True, labels={"x": c_, "y": "ortalama tahmin"}, title=f"Kısmi bağımlılık: {c_}"))
                st.caption("Diğer özellikler sabit tutulmadan değil, ortalaması alınarak bakılır; etkileşimleri gizleyebilir. İlişkidir, nedensellik değil.")
    with t4:
        try:
            import shap
            import matplotlib.pyplot as plt
            pipe, inner = tree_member(f["model"])
            if inner is None:
                st.info("SHAP (TreeExplainer) ağaç tabanlı bir model gerektirir. Final modelde ağaç modeli yok; permütasyon önemini kullan.")
            elif st.button("SHAP grafiğini üret", key="btn_shap"):
                with st.spinner("SHAP hesaplanıyor..."):
                    xs = S.X_test.head(500)
                    xt = pipe.named_steps["prep"].transform(pipe.named_steps["fe"].transform(xs))
                    names = pipe.named_steps["prep"].get_feature_names_out()
                    sv = shap.TreeExplainer(inner).shap_values(xt)
                    if isinstance(sv, list):
                        sv = sv[1]
                    elif getattr(sv, "ndim", 2) == 3:
                        sv = sv[:, :, 1]
                    plt.figure()
                    shap.summary_plot(sv, xt, feature_names=names, show=False)
                    st.pyplot(plt.gcf())
                    plt.close()
        except ImportError:
            st.info("SHAP yüklü değil: pip install shap")
        except Exception as e:
            st.warning(f"SHAP hesaplanamadı: {e}")
    S.done.add(10)


# ----------------------------------------------------------------------------
# Bölüm 11: Dışa aktar ve tahmin
# ----------------------------------------------------------------------------
def build_report():
    c = S.cfg
    lines = ["# Proje Raporu", "", f"- Görev: {c['task']}", f"- Hedef: {c['target']}" + (f" (pozitif sınıf: {c['pos_label']})" if is_clf() else ""),
             f"- Satır/sütun (model girdisi): {S.X.shape if 'X' in S else '-'}", f"- Atılan sütunlar: {c['drop_cols']}", f"- Bölme: {c['split_mode']}, test %{c['test_size']}, CV {c['cv_folds']} kat",
             f"- Ana metrik: {primary()}", ""]
    if "cv_table" in S:
        lines += ["## Model karşılaştırma (CV)", "```", S.cv_table.round(3).to_string(index=False), "```", ""]
    if "final" in S:
        lines += ["## Final model", f"- {S.final['name']} | CV {S.final['cv']:.3f} ± {S.final['cv_std']:.3f}", f"- Eşik: {S.final['thr']}"]
    if "eval" in S:
        r = S.eval
        if is_clf() and r["has_proba"]:
            lines.append(f"- Test ROC-AUC {r['roc']:.3f} ({r['roc_ci'][0]:.3f}-{r['roc_ci'][1]:.3f}), PR-AUC {r['pr']:.3f} ({r['pr_ci'][0]:.3f}-{r['pr_ci'][1]:.3f}), rastgele {r['prevalence']:.3f}")
        elif not is_clf():
            lines.append(f"- Test RMSE {r['rmse']:.3f} ({r['rmse_ci'][0]:.3f}-{r['rmse_ci'][1]:.3f}), MAE {r['mae']:.3f}, R² {r['r2']:.3f}, baseline RMSE {r['baseline_rmse']:.3f}")
        lines.append(f"- Test seti {S.test_uses} kez kullanıldı.")
    lines += ["", "## Yapılacaklar / öneriler"] + [f"- {n}" for n in S.notes]
    return "\n".join(lines)


def sec11():
    st.header("11. Dışa Aktar ve Tahmin")
    how_to("Modeli, ayarları ve raporu indir; yeni veriyi toplu ya da tek tek skorla.\n\n"
           "**Sık hatalar:** Eksik sütunlu veriyi sessizce NaN ile doldurup tahmin etmek (burada uyarı verilir); model dosyasını `vb_core.py` olmadan yüklemeye çalışmak; "
           "modeli canlıya aldıktan sonra izlememek.")
    need("final")
    refit = st.checkbox("Üretim için train+test ile yeniden eğit (seçimler bittiği için önerilir)", True, key="w_refit")
    if "bundle" not in S or st.button("Paketi hazırla", key="btn_bundle"):
        model = clone(S.cand_pool[S.final["name"]])
        model.fit(pd.concat([S.X_train, S.X_test]) if refit else S.X_train, pd.concat([S.y_train, S.y_test]) if refit else S.y_train)
        S.bundle = dict(model=model, threshold=S.final["thr"], task=S.cfg["task"], columns=list(S.X_train.columns), pos_label=S.cfg["pos_label"],
                        cfg=json.loads(json.dumps(S.cfg, default=str)), name=S.final["name"])
    c1, c2, c3 = st.columns(3)
    c1.download_button("Modeli indir (.joblib)", vb.to_bytes(S.bundle), "model_paketi.joblib")
    c2.download_button("Ayarları indir (.json)", json.dumps(S.cfg, ensure_ascii=False, indent=1, default=str), "proje_ayarlari.json")
    c3.download_button("Raporu indir (.md)", build_report(), "proje_raporu.md")
    st.caption("Model dosyasını kullanmak için: `import vb_core, joblib; paket = joblib.load('model_paketi.joblib'); paket['model'].predict_proba(yeni_df)`")
    expected = S.bundle["columns"]
    tb, ts = st.tabs(["Toplu tahmin (dosya)", "Tek kayıt tahmini"])
    with tb:
        up = st.file_uploader("Tahmin edilecek veri (.xlsx/.csv)", type=["xlsx", "xls", "csv"], key="pred_upload")
        if up is not None:
            newdf = pd.read_excel(up) if up.name.lower().endswith(("xlsx", "xls")) else pd.read_csv(up)
            missing = [c for c in expected if c not in newdf.columns]
            extra = [c for c in newdf.columns if c not in expected]
            if missing:
                st.error(f"Eksik sütunlar: {missing}. Eksik sütunlar sessizce NaN yapılırsa tahmin kalitesi bozulabilir.")
            if extra:
                st.info(f"Modelin kullanmadığı sütunlar (yok sayılır): {extra}")
            go_on = st.checkbox("Eksik sütunları NaN kabul ederek yine de devam et", False, key="w_force") if missing else True
            if go_on and st.button("Tahmin et", key="btn_predict"):
                Xn = newdf.reindex(columns=expected)
                out = newdf.copy()
                m, thr = S.bundle["model"], S.bundle["threshold"]
                if S.cfg["task"] == "classification" and hasattr(m, "predict_proba"):
                    p = m.predict_proba(Xn)[:, 1]
                    out["risk_olasiligi"] = p.round(4)
                    out["aksiyon"] = (p >= (thr if thr is not None else 0.5)).astype(int)
                else:
                    out["tahmin"] = m.predict(Xn)
                show_df(out.head(50))
                st.download_button("Sonuçları indir (CSV)", out.to_csv(index=False).encode("utf-8"), "tahminler.csv", "text/csv")
    with ts:
        with st.form("single"):
            vals = {}
            for c in expected:
                s = S.X_train[c]
                if pd.api.types.is_numeric_dtype(s):
                    vals[c] = st.number_input(c, value=float(s.median()) if s.notna().any() else 0.0, key=f"w_single_{c}")
                elif s.nunique() <= 40:
                    vals[c] = st.selectbox(c, sorted(s.dropna().astype(str).unique()), key=f"w_single_{c}")
                else:
                    vals[c] = st.text_input(c, value=str(s.dropna().iloc[0]) if s.notna().any() else "", key=f"w_single_{c}")
            ok = st.form_submit_button("Tahmin et")
        if ok:
            row = pd.DataFrame([vals])
            m, thr = S.bundle["model"], S.bundle["threshold"]
            if S.cfg["task"] == "classification" and hasattr(m, "predict_proba"):
                p = float(m.predict_proba(row)[0, 1])
                st.metric("Risk olasılığı (sıralama için, yüzde değil)", f"{p:.3f}", delta="Aksiyon: EVET" if p >= (thr or 0.5) else "Aksiyon: hayır", delta_color="off")
            else:
                st.metric("Tahmin", f"{float(m.predict(row)[0]):,.2f}")
    S.done.add(11)


# ----------------------------------------------------------------------------
# Ana akış
# ----------------------------------------------------------------------------
def main():
    init_state()
    st.sidebar.title("🔬 Veri Bilimi Platformu")
    st.sidebar.caption("Her tablo verisi projesi için tek şablon: kod yazmadan, istatistiksel kontrollerle.")
    section = st.sidebar.radio("Bölüm", SECTIONS, key="nav")
    st.sidebar.markdown("**İlerleme**")
    for i, name in enumerate(SECTIONS, start=1):
        st.sidebar.write(("✅ " if i in S.done else "⬜ ") + name)
    with st.sidebar.expander(f"Sonraki adım önerileri ({len(S.notes)})"):
        for n in S.notes:
            st.write("•", n)
        if st.button("Önerileri temizle", key="btn_clear_notes"):
            S.notes = []
    if st.sidebar.button("Projeyi sıfırla", key="btn_reset"):
        for k in list(S.keys()):
            if k != "nav":
                del S[k]
        st.rerun()
    {SECTIONS[0]: sec1, SECTIONS[1]: sec2, SECTIONS[2]: sec3, SECTIONS[3]: sec4, SECTIONS[4]: sec5, SECTIONS[5]: sec6,
     SECTIONS[6]: sec7, SECTIONS[7]: sec8, SECTIONS[8]: sec9, SECTIONS[9]: sec10, SECTIONS[10]: sec11}[section]()


main()
