import streamlit as st
import pandas as pd
from google import genai
from sklearn.tree import DecisionTreeClassifier

# --- 1. AYARLAR VE STİL ---
st.set_page_config(page_title="VetAI Pro", layout="wide", page_icon="🐾")

st.markdown("""
    <style>
    .scroll-container {
        height: 450px;
        overflow-y: auto;
        padding: 20px;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        background-color: #ffffff;
        margin-bottom: 15px;
        box-shadow: inset 0 0 5px rgba(0,0,0,0.05);
    }
    .stButton button {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

GEMINI_API_KEY = "sizin_api_anahtariniz_buraya"
MODEL_NAME = "gemini-2.5-flash"

def get_ai_client():
    return genai.Client(api_key=GEMINI_API_KEY)

# --- 2. MODEL EĞİTİMİ ---
@st.cache_resource
def train_model():
    try:
        path = r"C:\Users\evinn\Desktop\veterinerlik\dataset_filled.csv"
        df = pd.read_csv(path)
        
        def label_diagnosis(row):
            rbc_high = row['RBC'] > 6.0; hct_high = row['HCT'] > 50.0; hgb_high = row['HGB'] > 17.5
            rbc_low = row['RBC'] < 4.0; hct_low = row['HCT'] < 36.0; hgb_low = row['HGB'] < 12.0
            is_polistemi = rbc_high or hct_high or hgb_high
            is_anemi = rbc_low or hct_low or hgb_low
            
            wbc_high = row['WBC'] > 11.0; wbc_low = row['WBC'] < 4.5
            plt_high = row['PLT'] > 450.0; plt_low = row['PLT'] < 150.0
            neup_high = row['NEUp'] > 75.0; neup_low = row['NEUp'] < 40.0
            lymn_low = row['LYMn'] < 1.0; mcv_high = row['MCV'] > 100.0
            mcv_low = row['MCV'] < 80.0; mch_low = row['MCH'] < 27.0
            mchc_low = row['MCHC'] < 32.0; rdw_high = row['RDWCV'] > 14.5

            if is_polistemi:
                if neup_high and lymn_low: return "Polistemi: Stres"
                if wbc_high and plt_high: return "Polistemia Vera"
                return "Polistemi (Diger)"
            if is_anemi:
                if mcv_high: return "Anemi: B12 Eksikligi"
                if mcv_low and mch_low and mchc_low and rdw_high: return "Anemi: Demir Eksikligi"
                if mcv_low and mch_low: return "Anemi: Kronik Kan Kaybi"
                if wbc_high and plt_high and not mcv_low and not mcv_high: return "Anemi: Akut Kan Kaybi"
                if wbc_low and plt_low and neup_low and lymn_low: return "Anemi: Aplastik/FeLV"
                return "Anemi (Diger)"
            return "Normal"

        df['Diagnosis'] = df.apply(label_diagnosis, axis=1)
        features = ['WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'NEUp', 'LYMn', 'RDWCV']
        X = df[features]
        y = df['Diagnosis']
        
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X, y)
        return clf, features
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None, None

clf, features_list = train_model()

# --- 3. SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = ["Kritik değerler hangileri?", "Tedavi önerileri nedir?", "Beslenme nasıl olmalı?"]

# --- 4. ARAYÜZ TASARIMI ---
st.title("🐾 VetAI Pro: Klinik Karar Destek Sistemi")

# Hastalık Bilgi Kütüphanesi
with st.expander("📚 Hastalık Bilgi Kütüphanesi (Hızlı Tanı Rehberi)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**ANEMİ**\n- **Demir Eksikliği:** Düşük MCV, MCH, MCHC + Yüksek RDW.\n- **B12:** Yüksek MCV (Makrositik).\n- **Akut Kayıp:** WBC ve PLT yüksekliğiyle beraber.")
    with c2:
        st.warning("**POLİSTEMİ**\n- **Stres:** Yüksek NEUp + Düşük LYMn.\n- **Vera:** RBC, WBC ve PLT'nin hepsinde artış.")
    with c3:
        st.error("**KRİTİK UYARILAR**\n- **PLT < 150:** Kanama riski.\n- **WBC > 11:** Enfeksiyon veya inflamasyon.\n- **WBC < 4.5:** Bağışıklık baskılanması (FeLV vb.).")

# Yan Menü
with st.sidebar:
    st.header("🩸 Hemogram Girişi")
    inputs = {}
    inputs['WBC'] = st.slider("WBC (Beyaz Kan)", 0.0, 50.0, 7.5)
    inputs['RBC'] = st.slider("RBC (Kırmızı Kan)", 0.0, 15.0, 5.5)
    inputs['HGB'] = st.slider("HGB (Hemoglobin)", 0.0, 25.0, 14.0)
    inputs['HCT'] = st.slider("HCT (Hematokrit)", 0.0, 70.0, 42.0)
    inputs['MCV'] = st.number_input("MCV", value=88.0)
    inputs['MCH'] = st.number_input("MCH", value=30.0)
    inputs['MCHC'] = st.number_input("MCHC", value=34.0)
    inputs['PLT'] = st.number_input("PLT (Trombosit)", value=300.0)
    inputs['NEUp'] = st.number_input("NEUp %", value=65.0)
    inputs['LYMn'] = st.number_input("LYMn", value=2.5)
    inputs['RDWCV'] = st.number_input("RDW-CV", value=13.5)
    
    if st.button("ANALİZİ BAŞLAT", type="primary"):
        st.session_state.chat_history = []
        st.session_state.do_analysis = True

# Ana Panel
if "do_analysis" in st.session_state:
    input_df = pd.DataFrame([inputs])
    prediction = clf.predict(input_df)[0]
    
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        st.subheader("📊 Tanı Çıktısı")
        st.success(f"**Ön Tanı:** {prediction}")
        st.bar_chart(input_df.T)

    with col_right:
        st.subheader("💬 Klinik Konsültasyon")

        # İlk analizi yap (henüz geçmiş yoksa)
        if not st.session_state.chat_history:
            with st.spinner("İlk analiz yapılıyor..."):
                client = get_ai_client()
                p = f"Tahlil: {inputs}, Tanı: {prediction}. Klinik yorum yap ve tedavi adımlarını belirt."
                res = client.models.generate_content(model=MODEL_NAME, contents=[p])
                st.session_state.chat_history.append({"role": "assistant", "content": res.text})

        # Chat geçmişini HTML olarak scroll box içinde render et
        chat_html = ""
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                bubble = f"""
                <div style="background:#dce8ff;border-radius:10px;padding:12px 16px;margin-bottom:10px;color:#0d1b2a;">
                    <b>🤖 VetAI:</b><br>{msg["content"].replace(chr(10), "<br>")}
                </div>"""
            else:
                bubble = f"""
                <div style="background:#d4edda;border-radius:10px;padding:12px 16px;margin-bottom:10px;text-align:right;color:#0d1b2a;">
                    <b>👤 Sen:</b><br>{msg["content"].replace(chr(10), "<br>")}
                </div>"""
            chat_html += bubble

        st.markdown(
            f'<div class="scroll-container" id="chat-box">{chat_html}</div>',
            unsafe_allow_html=True
        )

        # Dinamik Soru Butonları
        st.write("💡 **Soru Önerileri:**")
        b_cols = st.columns(3)
        for i, q in enumerate(st.session_state.suggested_questions):
            if b_cols[i].button(q, key=f"btn_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("AI Yanıtlıyor..."):
                    client = get_ai_client()
                    full_p = f"Veriler: {inputs}. Soru: {q}. Yanıtla ve sonunda '---' ayıracıyla 3 adet kısa, yeni soru önerisi yaz."
                    res = client.models.generate_content(model=MODEL_NAME, contents=[full_p])
                    
                    parts = res.text.split("---")
                    answer = parts[0].strip()
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    if len(parts) > 1:
                        new_qs = [s.strip().replace("- ", "") for s in parts[1].strip().split("\n") if s.strip()][:3]
                        if len(new_qs) == 3: st.session_state.suggested_questions = new_qs
                st.rerun()

        # Manuel Sohbet Girişi
        if prompt := st.chat_input("Tahlille ilgili soru sor..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("AI Yanıtlıyor..."):
                client = get_ai_client()
                res = client.models.generate_content(model=MODEL_NAME, contents=[f"Veriler: {inputs}. Soru: {prompt}"])
                st.session_state.chat_history.append({"role": "assistant", "content": res.text})
            st.rerun()
else:
    st.info("Lütfen sol panelden tahlil değerlerini girin ve 'Analizi Başlat'a tıklayın.")