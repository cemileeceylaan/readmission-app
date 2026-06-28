import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Sayfa Ayarları ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hasta Yeniden Yatış Tahmini",
    page_icon="🏥",
    layout="wide"
)

# ── Model Yükle ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("readmission_model.pkl")
    encoders = joblib.load("readmission_encoders.pkl")
    return model, encoders

model, encoders = load_model()

# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────
def age_to_num(a):
    if a < 10:   return 5
    elif a < 20: return 15
    elif a < 30: return 25
    elif a < 40: return 35
    elif a < 50: return 45
    elif a < 60: return 55
    elif a < 70: return 65
    elif a < 80: return 75
    elif a < 90: return 85
    else:        return 95

def categorize_diag(diag):
    try:
        code = float(str(diag).split(".")[0])
        if str(diag).startswith("250"):  return "Diyabet"
        elif 390 <= code <= 459:         return "Dolaşım"
        elif 460 <= code <= 519:         return "Solunum"
        elif 520 <= code <= 579:         return "Sindirim"
        elif 800 <= code <= 999:         return "Yaralanma"
        elif 710 <= code <= 739:         return "Kas-İskelet"
        elif 580 <= code <= 629:         return "İdrar Yolu"
        elif 140 <= code <= 239:         return "Kanser"
        elif 290 <= code <= 319:         return "Ruh Sağlığı"
        elif 240 <= code <= 279:         return "Endokrin"
        else:                            return "Diğer"
    except:
        return "Diğer"

FEATURES = [
    "age_num", "gender", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_diagnoses",
    "number_outpatient", "number_emergency", "number_inpatient",
    "total_prior_visits", "has_prior_inpatient", "has_prior_emergency",
    "num_med_changed", "num_med_up", "num_med_down", "num_med_steady",
    "num_med_total", "on_insulin", "insulin_increased", "diabetesMed",
    "change", "diag_1_cat", "diag_2_cat", "diag_3_cat",
    "admission_type_id", "admission_source_group", "discharge_group", "race",
]

# ── Mapping Sözlükleri ────────────────────────────────────────────────────
gender_map = {"Kadın": "Female", "Erkek": "Male"}

race_map = {
    "Kafkas"       : "Caucasian",
    "Afro-Amerikan": "AfricanAmerican",
    "Hispanic"     : "Hispanic",
    "Asyalı"       : "Asian",
    "Diğer"        : "Other",
    "Bilinmiyor"   : "Unknown",
}

discharge_map = {
    "Eve"          : "Eve",
    "Eve + Bakım"  : "Eve_bakim",
    "Tesis/Bakımevi": "Tesis",
    "Diğer"        : "Diger",
}

admission_map = {
    "Acil Servis"  : "Acil",
    "Sevk"         : "Sevk",
    "Diğer"        : "Diger",
}

diag_map = {
    "Diyabet (250)"          : "250.01",
    "Kalp Krizi (410)"       : "410",
    "Pnömoni (486)"          : "486",
    "Kalp Yetmezliği (428)"  : "428",
    "Depresyon (296)"        : "296",
    "Kırık (820)"            : "820",
    "Hipertansiyon (401)"    : "401",
    "Böbrek Yetmezliği (585)": "585",
}

# ── Arayüz ────────────────────────────────────────────────────────────────
st.title("🏥 Hasta Yeniden Yatış Risk Tahmini")
st.markdown("**Hastaneden taburcu olan bir hastanın 30 gün içinde yeniden yatış riskini tahmin eden yapay zeka modeli**")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Hasta Bilgileri")
    age        = st.slider("Yaş", 0, 100, 65)
    gender_tr  = st.selectbox("Cinsiyet", list(gender_map.keys()))
    race_tr    = st.selectbox("Irk / Etnik Köken", list(race_map.keys()))

with col2:
    st.subheader("🏥 Klinik Bilgiler")
    time_in_hospital   = st.slider("Hastanede Kalış Süresi (gün)", 1, 14, 4)
    num_lab_procedures = st.slider("Laboratuvar Prosedür Sayısı", 1, 132, 43)
    num_procedures     = st.slider("Tıbbi Prosedür Sayısı", 0, 6, 1)
    num_medications    = st.slider("Kullanılan İlaç Sayısı", 1, 81, 15)
    number_diagnoses   = st.slider("Teşhis Sayısı", 1, 16, 7)
    diag_tr            = st.selectbox("Ana Teşhis", list(diag_map.keys()))
    admission_type_map = {
    "Acil"              : 1,
    "Elektif (Planlı)"  : 2,
    "Yenidoğan"         : 3,
    "Travma"            : 4,
    "Diğer"             : 5,
    "Bilinmiyor"        : 6}

admission_type_tr  = st.selectbox("Yatış Tipi", list(admission_type_map.keys()))
admission_type_id  = admission_type_map[admission_type_tr]
admission_tr       = st.selectbox("Yatış Kaynağı", list(admission_map.keys()))
discharge_tr       = st.selectbox("Taburcu Yeri", list(discharge_map.keys()))

with col3:
    st.subheader("💊 İlaç & Geçmiş Ziyaretler")
    number_inpatient = st.slider("Önceki Hastane Yatış Sayısı", 0, 21, 0)
    number_emergency = st.slider("Önceki Acil Başvuru Sayısı", 0, 76, 0)
    number_outpatient= st.slider("Önceki Poliklinik Ziyareti", 0, 42, 0)
    on_insulin       = st.checkbox("İnsülin Kullanıyor")
    insulin_changed  = st.checkbox("İnsülin Dozu Değiştirildi")
    num_med_changed  = st.slider("Değiştirilen İlaç Sayısı", 0, 10, 0)
    diabetes_tr      = st.selectbox("Diyabet İlacı Var mı?", ["Evet", "Hayır"])

st.divider()

# ── Tahmin Butonu ─────────────────────────────────────────────────────────
if st.button("🔍 Risk Analizi Yap", use_container_width=True, type="primary"):

    # Mapping uygula
    gender       = gender_map[gender_tr]
    race         = race_map[race_tr]
    discharge_group   = discharge_map[discharge_tr]
    admission_source  = admission_map[admission_tr]
    diabetes_med      = "Yes" if diabetes_tr == "Evet" else "No"
    diag_1            = diag_map[diag_tr]

    # Feature oluştur
    features = {
        "age_num"               : age_to_num(age),
        "gender"                : encoders["gender"].transform([gender])[0],
        "time_in_hospital"      : time_in_hospital,
        "num_lab_procedures"    : num_lab_procedures,
        "num_procedures"        : num_procedures,
        "num_medications"       : num_medications,
        "number_diagnoses"      : number_diagnoses,
        "number_outpatient"     : number_outpatient,
        "number_emergency"      : number_emergency,
        "number_inpatient"      : number_inpatient,
        "total_prior_visits"    : number_outpatient + number_emergency + number_inpatient,
        "has_prior_inpatient"   : int(number_inpatient > 0),
        "has_prior_emergency"   : int(number_emergency > 0),
        "num_med_changed"       : num_med_changed,
        "num_med_up"            : int(insulin_changed and on_insulin),
        "num_med_down"          : 0,
        "num_med_steady"        : num_med_changed,
        "num_med_total"         : int(diabetes_med == "Yes"),
        "on_insulin"            : int(on_insulin),
        "insulin_increased"     : int(insulin_changed),
        "diabetesMed"           : encoders["diabetesMed"].transform([diabetes_med])[0],
        "change"                : encoders["change"].transform(
                                    ["Ch" if num_med_changed > 0 else "No"])[0],
        "diag_1_cat"            : encoders["diag_1_cat"].transform(
                                    [categorize_diag(diag_1)])[0],
        "diag_2_cat"            : encoders["diag_2_cat"].transform(["Diğer"])[0],
        "diag_3_cat"            : encoders["diag_3_cat"].transform(["Diğer"])[0],
        "admission_type_id"     : admission_type_id,
        "admission_source_group": encoders["admission_source_group"].transform(
                                    [admission_source])[0],
        "discharge_group"       : encoders["discharge_group"].transform(
                                    [discharge_group])[0],
        "race"                  : encoders["race"].transform([race])[0],
    }

    input_df = pd.DataFrame([features])[FEATURES]
    proba    = model.predict_proba(input_df)[0][1]

    # ── Sonuç Göster ──────────────────────────────────────────────────────
    st.subheader("📊 Tahmin Sonucu")
    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.metric("🎯 Risk Skoru", f"%{proba*100:.1f}")

    with col_r2:
        st.metric("📏 Karar Eşiği", "%30.0")

    with col_r3:
        st.metric("📊 Karar",
                  "🔴 Yüksek Risk" if proba >= 0.30 else "🟢 Düşük Risk")

    # Progress bar
    st.progress(float(proba))

    if proba >= 0.30:
        st.error(f"""
        🔴 **YÜKSEK RİSK — %{proba*100:.1f}**

        Bu hasta 30 gün içinde yeniden yatış riski taşıyor.
        Taburculuk sonrası yakın takip önerilir.
        """)
    else:
        st.success(f"""
        🟢 **DÜŞÜK RİSK — %{proba*100:.1f}**

        Bu hastanın yeniden yatış riski düşük görünüyor.
        Standart taburculuk prosedürleri uygulanabilir.
        """)

    # ── SHAP ──────────────────────────────────────────────────────────────
    st.subheader("🔍 Model Neden Bu Kararı Verdi? (SHAP Analizi)")
    st.markdown("Aşağıdaki grafik hangi faktörlerin riski artırıp azalttığını gösterir.")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    FEATURE_NAMES_TR = [
    "Yaş", "Cinsiyet", "Hastanede Kalış (gün)", "Lab Prosedür Sayısı",
    "Tıbbi Prosedür Sayısı", "İlaç Sayısı", "Teşhis Sayısı",
    "Poliklinik Ziyareti", "Acil Başvurusu", "Hastane Yatışı",
    "Toplam Önceki Ziyaret", "Önceki Yatış Var mı", "Önceki Acil Var mı",
    "Değişen İlaç Sayısı", "Artan İlaç", "Azalan İlaç", "Sabit İlaç",
    "Toplam İlaç", "İnsülin Kullanımı", "İnsülin Artışı",
    "Diyabet İlacı", "İlaç Değişimi", "Ana Teşhis", "2. Teşhis", "3. Teşhis",
    "Yatış Tipi", "Yatış Kaynağı", "Taburcu Yeri", "Irk",
    ]

    shap.plots.waterfall(
        shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0].values,
        feature_names=FEATURE_NAMES_TR,
    ),
    show=False
    )
    
    st.pyplot(fig)
    plt.close()

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
*🤖 Model: XGBoost | 📊 Veri: 99.000+ hasta kaydı (130 ABD Hastanesi, 1999-2008) | 📈 AUC: 0.67*

*⚠️ Bu uygulama eğitim amaçlıdır, tıbbi karar verme aracı olarak kullanılmamalıdır.*
""")