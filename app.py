# ==============================================================================
# BÖLÜM 2: Veriyi Yükle, Denetle ve Manipüle Et (Gelişmiş Stüdyo)
# ==============================================================================
if "Bölüm 2: Veriyi Yükle" in current_section:
    st.header("📂 Bölüm 2: Gelişmiş Veri Yükleme ve Temizlik Stüdyosu")
    st.markdown("""
    <div class="section-box">
    <h3>Veri Sağlığı, Sütun Yönetimi ve Temizlik Paneli</h3>
    <p>Veri setinizi yükleyin; mükerrer kayıtları temizleyin, sabit sütunları ayıklayın, sütun adlarını ve tiplerini profesyonelce yönetin.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Veri Setinizi Yükleyin (.xlsx / .csv)", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        if 'df_raw' not in st.session_state or st.sidebar.button("🔄 Veriyi Sıfırla / Yeniden Yükle"):
            st.session_state['df_raw'] = load_data(uploaded_file)
            st.success("Veri başarıyla yüklendi ve hafızaya alındı!")
            
        df_working = st.session_state['df_raw'].copy()
        
        # --- HIZLI TEMİZLİK ARAÇLARI ÇUĞU ---
        st.subheader("⚡ Hızlı Veri Hijyeni İşlemleri")
        col_h1, col_h2, col_h3 = st.columns(3)
        
        with col_h1:
            dup_count = df_working.duplicated().sum()
            if st.button(f"🧹 Mükerrer Satırları Temizle ({dup_count} adet)"):
                if dup_count > 0:
                    df_working = df_working.drop_duplicates().reset_index(drop=True)
                    st.session_state['df_raw'] = df_working
                    st.success(f"{dup_count} adet yinelenen satır silindi!")
                    st.rerun()
                else:
                    st.info("Veri setinde yinelenen satır yok.")
                    
        with col_h2:
            constant_cols = [c for c in df_working.columns if df_working[c].nunique() <= 1]
            if st.button(f"🗑️ Sabit/Boş Sütunları At ({len(constant_cols)} adet)"):
                if constant_cols:
                    df_working = df_working.drop(columns=constant_cols)
                    st.session_state['df_raw'] = df_working
                    st.success(f"Sabit sütunlar atıldı: {constant_cols}")
                    st.rerun()
                else:
                    st.info("Sabit değerli sütun bulunamadı.")
                    
        with col_h3:
            # Temizlenmiş Veriyi İndir
            csv_export = df_working.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Temiz Veriyi İndir (CSV)", data=csv_export, file_name="temizlenmis_veri.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("📊 Ham Veri Önizlemesi ve Boyutları")
        st.dataframe(df_working.head(), use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Toplam Satır (Gözlem)", df_working.shape[0])
        with col_m2:
            st.metric("Toplam Sütun (Değişken)", df_working.shape[1])
            
        st.markdown("---")
        st.subheader("🛠️ İnteraktif Veri ve Sütun Yönetim Paneli")
        
        # 1. RENAME STUDIO (Sütun Adı Düzenleyici)
        with st.expander("✍️ Sütun Adı Düzenleyici (Rename Studio)", expanded=False):
            st.info("💡 Sütun adlarındaki boşlukları, Türkçe karakterleri otomatik temizleyebilir veya her sütunu tek tek yeniden adlandırabilirsiniz.")
            
            if st.button("🧹 Tüm Sütun Adlarını Otomatik Temizle (Boşlukları al, küçük harfe çevir)"):
                new_cols = {}
                tr_map = {'ç':'c', 'ğ':'g', 'ı':'i', 'ö':'o', 'ş':'s', 'ü':'u', 'İ':'i', 'Ç':'c', 'Ğ':'g', 'Ö':'o', 'Ş':'s', 'Ü':'u'}
                for col in df_working.columns:
                    cleaned = col.strip().lower().replace(" ", "_")
                    for k, v in tr_map.items():
                        cleaned = cleaned.replace(k, v)
                    new_cols[col] = cleaned
                df_working = df_working.rename(columns=new_cols)
                st.session_state['df_raw'] = df_working
                st.success("✨ Sütun adları otomatik olarak temizlendi!")
                st.rerun()
                
            st.write("---")
            st.write("**Manuel Sütun Adı Güncelleme:**")
            new_column_names = {}
            for col in df_working.columns:
                new_name = st.text_input(f"Eski: `{col}`", value=col, key=f"rename_{col}")
                new_column_names[col] = new_name
                
            if st.button("💾 Sütun Adı Değişikliklerini Kaydet", type="primary"):
                if len(set(new_column_names.values())) < len(new_column_names.values()):
                    st.error("⚠️ Hata: Sütun adları benzersiz olmalıdır! Aynı adı birden fazla sütuna veremezsiniz.")
                else:
                    df_working = df_working.rename(columns=new_column_names)
                    st.session_state['df_raw'] = df_working
                    st.success("✨ Sütun adları başarıyla güncellendi!")
                    st.rerun()

        # 2. DTYPE CONVERTER (Veri Tipi Düzenleyici)
        with st.expander("⚙️ Sütun Veri Tiplerini Güncelle", expanded=False):
            st.info("💡 Sütunların veri tiplerini (`int64`, `float64`, `object`, `category`) ihtiyacınıza göre dönüştürebilirsiniz.")
            
            edited_cols_data = []
            for col in df_working.columns:
                current_type = str(df_working[col].dtype)
                col_type_target = st.selectbox(
                    f"'{col}' sütunu veri tipi:", 
                    ["Mevcut (" + current_type + ")", "int64 (Sayısal)", "float64 (Ondalıklı)", "object (Metin)", "category (Kategorik)"],
                    key=f"dtype_{col}"
                )
                edited_cols_data.append((col, col_type_target))
                
            if st.button("🔧 Veri Tiplerini Güncelle", type="primary"):
                for col, target in edited_cols_data:
                    try:
                        if "int64" in target:
                            df_working[col] = pd.to_numeric(df_working[col], errors='coerce').fillna(0).astype('int64')
                        elif "float64" in target:
                            df_working[col] = pd.to_numeric(df_working[col], errors='coerce').astype('float64')
                        elif "object" in target:
                            df_working[col] = df_working[col].astype(str)
                        elif "category" in target:
                            df_working[col] = df_working[col].astype('category')
                    except Exception as e:
                        st.warning(f"'{col}' sütunu dönüştürülemedi: {e}")
                
                st.session_state['df_raw'] = df_working
                st.success("✨ Veri tipleri başarıyla güncellendi!")
                st.rerun()

        st.markdown("---")
        st.subheader("📋 Veri Seti Kalite ve Eksik Değer Karnesi")
        info_df = pd.DataFrame({
            "Veri Tipi": df_working.dtypes.astype(str),
            "Eksik Değer Sayısı": df_working.isnull().sum(),
            "Eksik Oranı (%)": (df_working.isnull().mean() * 100).round(2),
            "Benzersiz (Unique) Değer": df_working.nunique()
        })
        st.dataframe(info_df, use_container_width=True)
    elif not data_loaded:
        st.info("👈 Lütfen devam etmek için yukarıdan veri setinizi yükleyin.")
