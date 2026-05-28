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
html, body, .stApp { background: #F7F9FC; font-family: 'Segoe UI', sans-serif; }
.block-container { padding: 1rem 2rem 0.5rem !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Header — light grey-blue, no harsh colours */
.hdr {
    background: #E8EEF6;
    border: 1px solid #D0DAE8;
    border-radius: 12px; padding: 0.85rem 1.4rem;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1rem;
}
.hdr-title { color: #2C3E6B; font-size: 1.1rem; font-weight: 700; }
.hdr-sub   { color: #6B7FA3; font-size: 0.7rem; margin-top: 2px; }
.hdr-chips { display: flex; gap: 7px; }
.hchip {
    background: #D6E2F0; color: #3A5080;
    border: 1px solid #BFCFE3;
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.7rem; font-weight: 600;
}

/* Form card — plain white */
.card {
    background: #FFFFFF;
    border-radius: 12px; padding: 1.1rem 1.3rem;
    border: 1px solid #DDE4EE;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.sec-title {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: #5A7099; margin: 0 0 0.6rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #DDE4EE;
}
.hr { border: none; border-top: 1px solid #EEF1F6; margin: 0.7rem 0; }

/* BMI pill — very pastel */
.bmi-pill {
    display: inline-block; border-radius: 20px;
    padding: 3px 12px; font-size: 0.73rem; font-weight: 600;
    margin-top: 3px;
}

/* Result panel — light steel, NOT dark */
.result {
    background: #EEF3FA;
    border-radius: 12px; padding: 1.2rem 1.3rem;
    border: 1px solid #C8D6E8;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.r-sub-label {
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 1.2px; color: #7A90B0; margin-bottom: 3px;
}
.r-cost   { font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;
            line-height: 1; color: #1E3A5F; }
.r-period { font-size: 0.8rem; color: #6B7FA3; margin-top: 4px; }
.r-hr     { border: none; border-top: 1px solid #D0DAE8; margin: 0.7rem 0; }

.risk-low  { background:#E6F4EC; color:#2E7D4F; border-radius:20px; padding:3px 12px;
             font-size:.73rem; font-weight:600; display:inline-block; margin-top:6px; }
.risk-mid  { background:#FDF4E3; color:#8A6000; border-radius:20px; padding:3px 12px;
             font-size:.73rem; font-weight:600; display:inline-block; margin-top:6px; }
.risk-high { background:#FCEAEA; color:#8B2020; border-radius:20px; padding:3px 12px;
             font-size:.73rem; font-weight:600; display:inline-block; margin-top:6px; }

.srow { display:flex; justify-content:space-between; padding:4px 0;
        border-bottom:1px solid #E4EAF2; font-size:0.78rem; }
.sk { color:#7A90B0; }
.sv { font-weight:600; color:#1E3A5F; }
.cond-box {
    background: #E4EBF5;
    border-radius: 8px; padding: 7px 10px;
    margin-top: 7px; font-size: 0.75rem; line-height: 1.9;
    color: #3A5080;
}

/* Placeholder */
.placeholder {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 300px; text-align: center;
}
.ph-icon { font-size: 2.6rem; margin-bottom: 0.5rem; opacity: 0.45; }
.ph-msg  { font-size: 0.88rem; font-weight: 600; color: #4A6080; }
.ph-hint { font-size: 0.74rem; color: #8A9BBB; margin-top: 4px; }

/* Button — muted blue, not loud */
div.stButton > button {
    background: #4A6FA5 !important;
    color: #fff !important; font-weight: 700 !important;
    font-size: 0.95rem !important; border-radius: 9px !important;
    padding: 0.55rem !important; width: 100% !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(74,111,165,0.25) !important;
    margin-top: 0.4rem !important;
}
div.stButton > button:hover { background: #3A5A8F !important; }

label { font-size: 0.78rem !important; font-weight: 600 !important; color: #374151 !important; }
div[data-testid="stCheckbox"] label { font-size: 0.8rem !important; color: #374151 !important; }
div[data-testid="column"] { padding: 0 5px !important; }
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

@st.cache_resource(show_spinner="Training model...")
def train(_df):
    X,y=_df.drop(columns=['charges']),_df['charges']
    cat=['sex','smoker','region']; num=[c for c in X.columns if c not in cat]
    pre=ColumnTransformer([('c',OneHotEncoder(handle_unknown='ignore'),cat),('n','passthrough',num)])
    pipe=Pipeline([('pre',pre),('m',RandomForestRegressor(200,random_state=42,n_jobs=-1))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
    return pipe, mean_absolute_error(yte,p), r2_score(yte,p)

df=make_data(); model,mae,r2=train(df)

# ── Header ─────────────────────────────────────────
st.markdown(f"""
<div class="hdr">
  <div>
    <div class="hdr-title">🏥 Medical Cost Predictor</div>
    <div class="hdr-sub">Chaudhary Marmikkumar Ashvinbhai · 251370680002 · GTU PGDDS Mini Project 2025-26 · Guide: Komal Prajapati</div>
  </div>
  <div class="hdr-chips">
    <span class="hchip">🤖 Random Forest</span>
    <span class="hchip">R² {r2:.3f}</span>
    <span class="hchip">MAE ${mae:,.0f}</span>
    <span class="hchip">13 Features · 10K Records</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two columns ────────────────────────────────────
left, right = st.columns([6, 4], gap="medium")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<p class="sec-title">👤 Patient Details</p>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    age      = a.number_input("Age (years)", 18, 64, 35, step=1)
    sex      = b.selectbox("Sex", ["Male","Female"])
    children = c.number_input("No. of Dependants", 0, 5, 0, step=1)

    d1,d2,d3 = st.columns(3)
    bmi    = d1.number_input("BMI", 15.0, 52.0, 26.0, step=0.1, format="%.1f")
    smoker = d2.selectbox("Smoking Status", ["Non-Smoker","Smoker"])
    region = d3.selectbox("Region", ["Northeast","Southeast","Southwest","Northwest"])

    if   bmi < 18.5: pcol,ptxt = "#EDE9FE;color:#5B21B6", f"🔵 Underweight  ({bmi:.1f})"
    elif bmi < 25.0: pcol,ptxt = "#D1FAE5;color:#065F46", f"🟢 Normal  ({bmi:.1f})"
    elif bmi < 30.0: pcol,ptxt = "#FEF3C7;color:#92400E", f"🟡 Overweight  ({bmi:.1f})"
    else:            pcol,ptxt = "#FFE4E6;color:#9F1239", f"🔴 Obese  ({bmi:.1f})"
    st.markdown(f'<span class="bmi-pill" style="background:{pcol}">BMI Status: {ptxt}</span>',
                unsafe_allow_html=True)

    st.markdown('<hr class="hr"/>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">🩺 Diagnosed Medical Conditions</p>', unsafe_allow_html=True)
    st.caption("Select every condition that currently applies to this patient:")

    e1,e2,e3,e4 = st.columns(4)
    diabetes  = e1.checkbox("🩸 Diabetes")
    bp        = e2.checkbox("❤️ Hypertension")
    asthma    = e3.checkbox("🌬️ Asthma")
    skin      = e4.checkbox("🧴 Skin Allergy")

    f1,f2,f3,f4 = st.columns(4)
    cancer    = f1.checkbox("🎗️ Cancer History")
    infection = f2.checkbox("🦠 Chronic Infection")
    allergies = f3.checkbox("🤧 Allergies")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("")
    predict = st.button("💰  Calculate Medical Cost Estimate", use_container_width=True)

with right:
    st.markdown('<div class="result">', unsafe_allow_html=True)

    if predict:
        inp = pd.DataFrame([{
            'age':age,'sex':sex.lower(),'bmi':bmi,'children':children,
            'smoker':'yes' if smoker=='Smoker' else 'no',
            'region':region.lower(),
            'diabetes':int(diabetes),'blood_pressure':int(bp),
            'asthma':int(asthma),'skin_allergy':int(skin),
            'cancer_history':int(cancer),'chronic_infection':int(infection),
            'allergies':int(allergies)
        }])
        cost  = float(model.predict(inp)[0])
        risk  = "low" if cost<8000 else "mid" if cost<22000 else "high"
        rbadge= {"low":"🟢 Low Risk","mid":"🟡 Moderate Risk","high":"🔴 High Risk"}[risk]
        bmi_c = "Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese"

        active=[n for flag,n in [
            (diabetes,"🩸 Diabetes"),(bp,"❤️ Hypertension"),(asthma,"🌬️ Asthma"),
            (skin,"🧴 Skin Allergy"),(cancer,"🎗️ Cancer"),(infection,"🦠 Infection"),
            (allergies,"🤧 Allergies"),(smoker=="Smoker","🚬 Smoker"),(bmi>=30,"⚖️ Obesity")
        ] if flag]

        st.markdown(f"""
        <div class="r-sub-label">Estimated Annual Medical Cost</div>
        <div class="r-cost">${cost:,.2f}</div>
        <div class="r-period">${cost/12:,.0f} per month &nbsp;·&nbsp; ${cost/365:,.1f} per day</div>
        <div class="risk-{risk}">{rbadge}</div>
        <hr class="r-hr"/>
        <div class="r-sub-label">Patient Summary</div>
        <div class="srow"><span class="sk">Age</span><span class="sv">{age} years</span></div>
        <div class="srow"><span class="sk">Sex</span><span class="sv">{sex}</span></div>
        <div class="srow"><span class="sk">BMI</span><span class="sv">{bmi:.1f} — {bmi_c}</span></div>
        <div class="srow"><span class="sk">Dependants</span><span class="sv">{children}</span></div>
        <div class="srow"><span class="sk">Smoking</span><span class="sv">{smoker}</span></div>
        <div class="srow"><span class="sk">Region</span><span class="sv">{region}</span></div>
        <hr class="r-hr"/>
        <div class="r-sub-label">Active Risk Factors</div>
        {"<div class='cond-box'>" + "  ·  ".join(active) + "</div>"
         if active else
         "<div class='cond-box' style='color:#2E7D4F'>✅ No major risk factors detected</div>"}
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder">
            <div class="ph-icon">📋</div>
            <div class="ph-msg">Your estimate will appear here</div>
            <div class="ph-hint">Fill in the patient details on the left<br>
            then click <b>Calculate Medical Cost Estimate</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
