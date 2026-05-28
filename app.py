import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Medical Cost Predictor", page_icon="🏥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #f1f5f9; }
.block-container { padding: .8rem 1.5rem 1rem !important; }

/* ── Top nav bar ── */
.topbar {
    background: white;
    border-radius: 12px;
    padding: .75rem 1.4rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 1px 6px rgba(0,0,0,.08);
    margin-bottom: 1rem;
}
.topbar-left { display:flex; align-items:center; gap:10px; }
.topbar-icon { background:#0ea5e9; color:white; width:36px; height:36px;
               border-radius:9px; display:flex; align-items:center;
               justify-content:center; font-size:1.1rem; }
.topbar-title { font-size:1.05rem; font-weight:800; color:#0f172a; }
.topbar-sub   { font-size:.72rem; color:#64748b; margin-top:1px; }
.topbar-stats { display:flex; gap:10px; }
.stat-chip    { background:#f0f9ff; color:#0369a1; border:1px solid #bae6fd;
                border-radius:8px; padding:4px 12px; font-size:.75rem; font-weight:700; }

/* ── Card ── */
.card {
    background: white; border-radius: 12px;
    padding: 1rem 1.1rem; box-shadow: 0 1px 6px rgba(0,0,0,.07);
    margin-bottom: .8rem;
}
.card-head {
    font-size: .7rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 1.2px; color: #0ea5e9;
    border-bottom: 2px solid #e0f2fe; padding-bottom: 6px; margin-bottom: .7rem;
}

/* ── Condition pills ── */
.cond-pill-on  { display:inline-flex; align-items:center; gap:5px;
                 background:#dbeafe; color:#1d4ed8; border:1.5px solid #93c5fd;
                 border-radius:8px; padding:5px 12px; font-size:.8rem; font-weight:600; }
.cond-pill-off { display:inline-flex; align-items:center; gap:5px;
                 background:#f8fafc; color:#94a3b8; border:1.5px solid #e2e8f0;
                 border-radius:8px; padding:5px 12px; font-size:.8rem; font-weight:600; }

/* ── Result panel ── */
.result-top {
    background: linear-gradient(135deg,#0c4a6e,#0ea5e9);
    border-radius:12px; padding:1.2rem 1.4rem; color:white;
    margin-bottom:.8rem;
}
.result-label { font-size:.7rem; opacity:.75; text-transform:uppercase; letter-spacing:1px; }
.result-cost  { font-size:2.6rem; font-weight:900; letter-spacing:-1px; line-height:1.1; }
.result-month { font-size:.85rem; opacity:.75; margin-top:.2rem; }

/* Risk */
.risk-L { background:#dcfce7; color:#15803d; padding:3px 12px; border-radius:99px;
           font-size:.78rem; font-weight:700; display:inline-block; margin-top:.5rem; }
.risk-M { background:#fef9c3; color:#a16207; padding:3px 12px; border-radius:99px;
           font-size:.78rem; font-weight:700; display:inline-block; margin-top:.5rem; }
.risk-H { background:#fee2e2; color:#b91c1c; padding:3px 12px; border-radius:99px;
           font-size:.78rem; font-weight:700; display:inline-block; margin-top:.5rem; }

/* Summary table */
.sum-row { display:flex; justify-content:space-between; align-items:center;
           padding:5px 0; border-bottom:1px solid #f1f5f9; font-size:.82rem; }
.sum-key  { color:#64748b; }
.sum-val  { font-weight:700; color:#0f172a; }

/* Active conditions */
.act-cond { background:#eff6ff; border-radius:8px; padding:.6rem .9rem;
            font-size:.8rem; color:#1d4ed8; margin-top:.6rem; line-height:1.8; }

/* Placeholder */
.placeholder {
    background:white; border-radius:12px; border:2px dashed #bae6fd;
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:320px; color:#94a3b8; text-align:center;
    padding:2rem;
}
.placeholder .icon { font-size:2.5rem; margin-bottom:.6rem; }
.placeholder .msg  { font-size:.9rem; font-weight:600; color:#64748b; }
.placeholder .sub  { font-size:.78rem; margin-top:.3rem; }

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg,#0369a1,#0ea5e9) !important;
    color:white !important; font-weight:800 !important; font-size:1rem !important;
    border-radius:10px !important; padding:.65rem !important;
    border:none !important; width:100% !important;
    box-shadow:0 4px 14px rgba(14,165,233,.35) !important;
    letter-spacing:.3px !important;
}
div.stButton > button:hover {
    background:linear-gradient(135deg,#075985,#0284c7) !important;
}

/* BMI meter colours */
.bmi-U { color:#6366f1; font-weight:700; }
.bmi-N { color:#16a34a; font-weight:700; }
.bmi-W { color:#d97706; font-weight:700; }
.bmi-O { color:#dc2626; font-weight:700; }

label { font-size:.82rem !important; font-weight:600 !important; color:#334155 !important; }
div[data-testid="stVerticalBlock"] > div { margin-bottom:0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def make_data(n=10000):
    np.random.seed(42)
    age=np.random.randint(18,65,n); bmi=np.round(np.random.normal(28,6,n).clip(15,52),1)
    child=np.random.randint(0,6,n); sex=np.random.choice(['male','female'],n)
    smoke=np.random.choice(['yes','no'],n,p=[0.22,0.78])
    region=np.random.choice(['northeast','southeast','southwest','northwest'],n)
    d=np.random.choice([1,0],n,p=[0.12,0.88]); bp=np.random.choice([1,0],n,p=[0.18,0.82])
    ast=np.random.choice([1,0],n,p=[0.09,0.91]); sk=np.random.choice([1,0],n,p=[0.10,0.90])
    ca=np.random.choice([1,0],n,p=[0.05,0.95]); inf=np.random.choice([1,0],n,p=[0.07,0.93])
    al=np.random.choice([1,0],n,p=[0.14,0.86])
    cost=(age*260+bmi*330+child*420+(sex=='male')*900+(smoke=='yes')*23800
          +np.where(region=='northeast',1400,900)+d*5000+bp*3500+ast*3000
          +sk*1200+ca*13000+inf*2500+al*800+(smoke=='yes')*ca*6000+d*bp*2500
          +np.random.normal(0,1500,n)).clip(1000,120000).round(2)
    return pd.DataFrame({'age':age,'sex':sex,'bmi':bmi,'children':child,'smoker':smoke,
        'region':region,'diabetes':d,'blood_pressure':bp,'asthma':ast,
        'skin_allergy':sk,'cancer_history':ca,'chronic_infection':inf,
        'allergies':al,'charges':cost})

@st.cache_resource(show_spinner="Training AI model...")
def train(_df):
    X,y=_df.drop(columns=['charges']),_df['charges']
    cat=['sex','smoker','region']; num=[c for c in X.columns if c not in cat]
    pre=ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore'),cat),('n','passthrough',num)])
    pipe=Pipeline([('pre',pre),('m',RandomForestRegressor(200,random_state=42,n_jobs=-1))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
    return pipe, mean_absolute_error(yte,p), r2_score(yte,p)

df=make_data(); model,mae,r2=train(df)

# ── Top Nav ────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-icon">🏥</div>
    <div>
      <div class="topbar-title">Medical Insurance Cost Predictor</div>
      <div class="topbar-sub">Marmikkumar Chaudhary · 251370680002 · GTU PGDDS Mini Project 2025-26</div>
    </div>
  </div>
  <div class="topbar-stats">
    <div class="stat-chip">🤖 Random Forest</div>
    <div class="stat-chip">R² {r2:.3f}</div>
    <div class="stat-chip">MAE ${mae:,.0f}</div>
    <div class="stat-chip">13 Parameters</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────
left, right = st.columns([1.15, 0.85], gap="medium")

with left:
    # ── Patient Info ──
    st.markdown('<div class="card-head">🧑‍⚕️ Patient Information</div>', unsafe_allow_html=True)
    r1, r2c, r3 = st.columns(3)
    age      = r1.number_input("Age (years)", 18, 64, 35, step=1)
    sex      = r2c.selectbox("Biological Sex", ["Male", "Female"])
    children = r3.number_input("No. of Dependants", 0, 5, 0, step=1)

    r4, r5, r6 = st.columns(3)
    bmi    = r4.number_input("BMI", 15.0, 52.0, 26.0, step=0.1, format="%.1f")
    smoker = r5.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
    region = r6.selectbox("Coverage Region",
                          ["Northeast","Southeast","Southwest","Northwest"])

    # BMI indicator
    if bmi < 18.5:
        bmi_label = f'<span class="bmi-U">🔵 Underweight ({bmi:.1f})</span>'
    elif bmi < 25:
        bmi_label = f'<span class="bmi-N">🟢 Normal ({bmi:.1f})</span>'
    elif bmi < 30:
        bmi_label = f'<span class="bmi-W">🟡 Overweight ({bmi:.1f})</span>'
    else:
        bmi_label = f'<span class="bmi-O">🔴 Obese ({bmi:.1f})</span>'
    st.markdown(f"BMI Status: {bmi_label}", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Diagnosed Conditions ──
    st.markdown('<div class="card-head">🩺 Diagnosed Medical Conditions</div>',
                unsafe_allow_html=True)
    st.caption("Select all currently diagnosed or pre-existing conditions:")

    cx1, cx2, cx3, cx4 = st.columns(4)
    diabetes  = cx1.checkbox("🩸 Diabetes",       help="Type 1 or Type 2")
    bp        = cx2.checkbox("❤️ Hypertension",    help="High Blood Pressure > 140/90")
    asthma    = cx3.checkbox("🌬️ Asthma",          help="Chronic respiratory condition")
    skin      = cx4.checkbox("🧴 Skin Condition",  help="Eczema, psoriasis, dermatitis")

    cx5, cx6, cx7, _ = st.columns(4)
    cancer    = cx5.checkbox("🎗️ Cancer History",  help="Past or active cancer diagnosis")
    infection = cx6.checkbox("🦠 Chronic Infection",help="Long-term or recurring infections")
    allergies = cx7.checkbox("🤧 Allergies",        help="Food, drug, environmental allergies")

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("🔍  Calculate Insurance Cost Estimate", use_container_width=True)

# ── Right: Result ───────────────────────────────────
with right:
    if predict:
        inp = pd.DataFrame([{
            'age': age, 'sex': sex.lower(), 'bmi': bmi, 'children': children,
            'smoker': 'yes' if smoker == 'Smoker' else 'no',
            'region': region.lower(),
            'diabetes': int(diabetes), 'blood_pressure': int(bp),
            'asthma': int(asthma), 'skin_allergy': int(skin),
            'cancer_history': int(cancer), 'chronic_infection': int(infection),
            'allergies': int(allergies)
        }])
        cost  = float(model.predict(inp)[0])
        risk  = "L" if cost < 8000 else "M" if cost < 22000 else "H"
        rlabel= {"L":"🟢 Low Risk","M":"🟡 Moderate Risk","H":"🔴 High Risk"}[risk]

        # ── Cost card ──
        st.markdown(f"""
        <div class="result-top">
            <div class="result-label">Estimated Annual Insurance Cost</div>
            <div class="result-cost">${cost:,.2f}</div>
            <div class="result-month">${cost/12:,.0f} / month &nbsp;·&nbsp; ${cost/365:,.1f} / day</div>
            <div class="risk-{risk}">{rlabel}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Patient summary ──
        st.markdown('<div class="card-head">📋 Patient Record Summary</div>',
                    unsafe_allow_html=True)

        bmi_cat = ("Underweight" if bmi<18.5 else "Normal" if bmi<25
                   else "Overweight" if bmi<30 else "Obese")

        rows = [
            ("Age",        f"{age} years"),
            ("Sex",        sex),
            ("BMI",        f"{bmi:.1f} — {bmi_cat}"),
            ("Dependants", str(children)),
            ("Smoking",    smoker),
            ("Region",     region),
        ]
        for k, v in rows:
            st.markdown(
                f'<div class="sum-row"><span class="sum-key">{k}</span>'
                f'<span class="sum-val">{v}</span></div>',
                unsafe_allow_html=True)

        # ── Active conditions ──
        active = [n for flag, n in [
            (diabetes, "🩸 Diabetes"), (bp, "❤️ Hypertension"),
            (asthma, "🌬️ Asthma"), (skin, "🧴 Skin Condition"),
            (cancer, "🎗️ Cancer History"), (infection, "🦠 Chronic Infection"),
            (allergies, "🤧 Allergies"), (smoker == "Smoker", "🚬 Smoker"),
            (bmi >= 30, "⚖️ Obesity")
        ] if flag]

        if active:
            st.markdown(
                '<div class="act-cond"><strong>Active Risk Factors:</strong><br>'
                + " &nbsp;&nbsp; ".join(active) + "</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="act-cond" style="color:#166534;background:#f0fdf4">'
                '✅ No major risk factors detected</div>',
                unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder">
            <div class="icon">📋</div>
            <div class="msg">No estimate yet</div>
            <div class="sub">Fill in the patient details on the left<br>
            then click <strong>Calculate Insurance Cost Estimate</strong></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#94a3b8;font-size:.72rem;padding:.4rem 0">
🏥 Medical Insurance Cost Predictor &nbsp;·&nbsp; GTU PGDDS Mini Project 2025-26 &nbsp;·&nbsp; For Educational Use Only
</div>
""", unsafe_allow_html=True)
