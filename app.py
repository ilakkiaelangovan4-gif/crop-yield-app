import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score)
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# ═══════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS (Dark Theme)
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="Smart Crop AI",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* === GLOBAL TEXT COLOR FIX === */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
        color: #FFFFFF !important;
    }
    /* Force ALL text white */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp li, .stApp td, .stApp th, .stApp caption,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    /* Radio button labels */
    .stRadio label span, .stRadio div {
        color: #FFFFFF !important;
    }
    /* Slider labels & values */
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] div,
    [data-testid="stSlider"] span,
    .stSlider label, .stSlider span {
        color: #FFFFFF !important;
    }
    /* Expander text */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] div {
        color: #FFFFFF !important;
    }
    /* Metric labels and values */
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] div,
    [data-testid="stMetric"] span,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    /* Selectbox, text input */
    .stSelectbox label, .stTextInput label,
    .stSelectbox div, .stTextInput div {
        color: #FFFFFF !important;
    }
    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown strong, .stMarkdown em {
        color: #FFFFFF !important;
    }
    /* Data table headers */
    .stDataFrame th {
        color: #FFFFFF !important;
    }
    /* Sidebar caption */
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #AAAAAA !important;
    }
    /* Primary button styling */
    .stButton > button[kind="primary"],
    .stButton > button[kind="primaryFormSubmit"] {
        background: linear-gradient(90deg, #00b09b, #96c93d) !important;
        color: white !important; border: none !important;
        font-size: 1.1em !important; font-weight: bold !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="primaryFormSubmit"]:hover {
        opacity: 0.9 !important;
    }
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important; border: none !important;
    }
    /* Alert / Success / Warning / Info box text */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] strong,
    [data-testid="stAlert"] a,
    .stAlert p, .stAlert span, .stAlert div {
        color: #1a1a2e !important;
    }
    /* All generic buttons (non-primary) */
    .stButton > button {
        color: #FFFFFF !important;
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.2) !important;
    }
    /* Text input / selectbox inside expanders */
    .stTextInput input, .stSelectbox select {
        color: #1a1a2e !important;
    }

    .big-metric {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 30px; border-radius: 15px; text-align: center;
        color: white !important; font-size: 28px; font-weight: bold;
        box-shadow: 0 8px 32px rgba(0, 176, 155, 0.3);
    }
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 20px; margin: 10px 0;
        color: #FFFFFF !important;
    }
    .interpretation-card {
        background: rgba(0,176,155,0.1);
        border: 1px solid rgba(0,176,155,0.3);
        border-radius: 10px; padding: 20px; margin: 10px 0;
        color: #FFFFFF !important;
    }
    .crop-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 20px 40px; border-radius: 15px; text-align: center;
        color: white !important; font-size: 24px; font-weight: bold;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .header-title {
        text-align: center; font-size: 2.5em; font-weight: 800;
        background: linear-gradient(90deg, #00b09b, #96c93d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .header-sub {
        text-align: center; color: #CCCCCC !important; font-size: 1.1em; margin-bottom: 30px;
    }
    /* Progress bar track */
    .stProgress > div > div {
        background-color: rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# LOAD DATA & TRAIN MODELS
# ═══════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\ie\Downloads\Crop_recommendation.csv")
    np.random.seed(42)
    df['Yield'] = (
        0.05 * df['N'] + 0.03 * df['P'] + 0.04 * df['K'] +
        0.1 * df['temperature'] + 0.02 * df['humidity'] +
        0.5 * df['ph'] + 0.01 * df['rainfall'] +
        np.random.normal(0, 0.5, len(df))
    )
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    return df, le

@st.cache_resource
def train_all_models(_df, _le):
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    df = _df.copy()

    # --- Regression Models (Yield) ---
    X_r = df[feature_cols]; y_r = df['Yield']
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42); rf.fit(X_tr, y_tr)
    lr = LinearRegression(); lr.fit(X_tr, y_tr)
    dt = DecisionTreeRegressor(random_state=42); dt.fit(X_tr, y_tr)

    reg_models = {'Random Forest': rf, 'Linear Regression': lr, 'Decision Tree': dt}
    reg_results = {}
    for name, m in reg_models.items():
        p = m.predict(X_te)
        reg_results[name] = {
            'MAE': mean_absolute_error(y_te, p),
            'RMSE': np.sqrt(mean_squared_error(y_te, p)),
            'R2': r2_score(y_te, p), 'predictions': p
        }

    # --- Classification Model (Crop Recommendation) ---
    X_c = df[feature_cols]; y_c = df['label_encoded']
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42); clf.fit(Xc_tr, yc_tr)
    cls_acc = accuracy_score(yc_te, clf.predict(Xc_te))

    return reg_models, reg_results, X_te, y_te, feature_cols, clf, cls_acc

df, le = load_data()
reg_models, reg_results, X_te, y_te, feat_cols, clf_model, cls_acc = train_all_models(df, le)

# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════
def get_yield_level(v):
    if v < 11:   return "🔴 Low Yield", "#FF4B4B"
    elif v < 16: return "🟡 Medium Yield", "#FFC107"
    else:        return "🟢 High Yield", "#00CC66"

def create_gauge(value, mn=7, mx=24):
    _, color = get_yield_level(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': "Predicted Yield", 'font': {'size': 20, 'color': 'white'}},
        number={'font': {'size': 40, 'color': 'white'}, 'suffix': ' t/ha'},
        gauge={
            'axis': {'range': [mn, mx], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': color}, 'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [mn, 11], 'color': 'rgba(255,75,75,0.2)'},
                {'range': [11, 16], 'color': 'rgba(255,193,7,0.2)'},
                {'range': [16, mx], 'color': 'rgba(0,204,102,0.2)'}
            ],
            'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.8, 'value': value}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'color': 'white'}, height=300, margin=dict(t=50, b=20, l=30, r=30))
    return fig

def get_interpretation(inp, importances, fcols):
    vals = inp.iloc[0]
    imp_sorted = sorted(zip(fcols, importances), key=lambda x: x[1], reverse=True)
    lines = ["**📊 Smart Interpretation:**\n", "**Top Contributing Factors:**"]
    for f, im in imp_sorted[:3]:
        lines.append(f"- **{f}** = {vals[f]:.1f} (importance: {im*100:.1f}%)")

    n, p, k = vals['N'], vals['P'], vals['K']
    temp, hum, ph, rain = vals['temperature'], vals['humidity'], vals['ph'], vals['rainfall']
    lines.append("\n**🧠 Domain Insights:**")
    if n > 80:  lines.append("- ✅ High Nitrogen — excellent for leafy crops (rice, maize)")
    elif n < 30: lines.append("- ⚠️ Low Nitrogen — may limit growth for nitrogen-hungry crops")
    if p > 80:  lines.append("- ✅ High Phosphorus — promotes strong root development")
    if k > 100: lines.append("- ✅ High Potassium — good disease resistance")
    if temp > 35: lines.append("- 🔥 Very high temperature — heat stress risk")
    elif temp < 15: lines.append("- ❄️ Low temperature — may slow growth")
    if ph < 5.5: lines.append("- ⚠️ Acidic soil — consider liming")
    elif ph > 8: lines.append("- ⚠️ Alkaline soil — nutrient availability may be reduced")
    else: lines.append("- ✅ Soil pH in optimal range (5.5–8.0)")
    if rain > 200: lines.append("- 🌧️ High rainfall — suitable for water-loving crops")
    elif rain < 50: lines.append("- 🏜️ Very low rainfall — irrigation recommended")
    return "\n".join(lines)

def get_warnings(n, p, k, temp, hum, ph, rain):
    w = []
    if rain < 30: w.append("⚠️ Very low rainfall (<30mm) — yield may drop without irrigation.")
    if temp > 40: w.append("🔥 Extreme temperature (>40°C) — heat stress likely.")
    if temp < 10: w.append("❄️ Very low temperature (<10°C) — frost risk for tropical crops.")
    if ph < 4:   w.append("⚠️ Extremely acidic soil (pH<4) — most crops won't survive.")
    if ph > 9.5: w.append("⚠️ Extremely alkaline soil (pH>9.5) — severe nutrient lockout.")
    if n == 0 and p == 0 and k == 0: w.append("🚫 All nutrients zero — soil appears infertile.")
    if hum > 95: w.append("💧 Very high humidity (>95%) — fungal disease risk.")
    return w

def fetch_weather(city):
    try:
        geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1", timeout=5).json()
        if 'results' not in geo or not geo['results']: return None
        r = geo['results'][0]
        lat, lon, loc = r['latitude'], r['longitude'], r.get('name', city)
        w = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,rain", timeout=5
        ).json()['current']
        return {'city': loc, 'temperature': w['temperature_2m'],
                'humidity': w['relative_humidity_2m'],
                'rainfall': max(w.get('rain', 50.0) * 30, 20.0),
                'lat': lat, 'lon': lon}
    except: return None

def gen_report(inp, preds, crop, ylevel, interp):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r = ["=" * 60, "    SMART CROP AI — PREDICTION REPORT", "=" * 60, f"Generated: {now}\n",
         "--- INPUT PARAMETERS ---"]
    for c in inp.columns: r.append(f"  {c:15s}: {inp[c].values[0]:.2f}")
    r.append("\n--- YIELD PREDICTIONS ---")
    for m, v in preds.items(): r.append(f"  {m:20s}: {v:.4f} t/ha")
    r.append(f"\n  Yield Level: {ylevel}")
    r.append(f"\n--- RECOMMENDED CROP ---\n  {crop}")
    r.append(f"\n--- INTERPRETATION ---\n{interp.replace('**','').replace('- ','  * ')}")
    r += ["\n" + "=" * 60, "  Smart Crop AI | Powered by ML", "=" * 60]
    return "\n".join(r)

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
st.sidebar.markdown('<p style="text-align:center;font-size:2.5em;">🌾</p>', unsafe_allow_html=True)
st.sidebar.title("Smart Crop AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("🧭 Navigate", [
    "📊 Dataset Overview", "📈 EDA & Visualizations",
    "🤖 Model Performance", "🔮 Predict & Recommend",
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** 2,200 samples · 22 crops")
st.sidebar.markdown(f"**Regression R²:** {max(r['R2'] for r in reg_results.values()):.4f}")
st.sidebar.markdown(f"**Classification Acc:** {cls_acc*100:.1f}%")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · Plotly")

# ═══════════════════════════════════════════
# PAGE 1: Dataset Overview
# ═══════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.markdown('<div class="header-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Exploring the Crop Recommendation Dataset</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔢 Rows", df.shape[0])
    c2.metric("📋 Columns", df.shape[1])
    c3.metric("🌱 Crops", df['label'].nunique())
    c4.metric("❌ Missing", df.isnull().sum().sum())
    c5.metric("🔁 Duplicates", df.duplicated().sum())
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📄 Data Preview", "📊 Statistics", "🌾 Crop Distribution"])
    with tab1: st.dataframe(df.head(15), use_container_width=True)
    with tab2: st.dataframe(df.describe().round(2), use_container_width=True)
    with tab3:
        cc = df['label'].value_counts().reset_index(); cc.columns = ['Crop', 'Count']
        fig = go.Figure(go.Bar(x=cc['Crop'], y=cc['Count'],
            marker=dict(color=cc['Count'], colorscale='Viridis'),
            text=cc['Count'], textposition='outside'))
        fig.update_layout(title='Samples per Crop', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=500,
            xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════
# PAGE 2: EDA
# ═══════════════════════════════════════════
elif page == "📈 EDA & Visualizations":
    st.markdown('<div class="header-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Understanding patterns in environmental parameters</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📦 Boxplots", "🔥 Correlation", "🌾 Yield/Crop"])
    ncols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    with tab1:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8)); fig.patch.set_facecolor('#0E1117')
        for i, c in enumerate(ncols):
            axes.flat[i].set_facecolor('#0E1117')
            sns.histplot(df[c], kde=True, ax=axes.flat[i], color=sns.color_palette("husl", 7)[i])
            axes.flat[i].set_title(c, fontweight='bold', color='white')
            axes.flat[i].tick_params(colors='white')
            axes.flat[i].xaxis.label.set_color('white'); axes.flat[i].yaxis.label.set_color('white')
        axes.flat[-1].axis('off'); plt.tight_layout(); st.pyplot(fig)

    with tab2:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8)); fig.patch.set_facecolor('#0E1117')
        for i, c in enumerate(ncols):
            axes.flat[i].set_facecolor('#0E1117')
            sns.boxplot(y=df[c], ax=axes.flat[i], color=sns.color_palette("pastel")[i])
            axes.flat[i].set_title(c, fontweight='bold', color='white')
            axes.flat[i].tick_params(colors='white'); axes.flat[i].yaxis.label.set_color('white')
        axes.flat[-1].axis('off'); plt.tight_layout(); st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 6)); fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
        sns.heatmap(df[ncols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f',
                    ax=ax, square=True, annot_kws={'color': 'white'})
        ax.tick_params(colors='white')
        plt.title('Correlation Heatmap', color='white', fontsize=14, fontweight='bold')
        st.pyplot(fig)

    with tab4:
        cy = df.groupby('label')['Yield'].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=cy.index, y=cy.values,
            marker=dict(color=cy.values, colorscale='RdYlGn'),
            text=[f'{v:.1f}' for v in cy.values], textposition='outside'))
        fig.update_layout(title='Average Yield per Crop', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=500,
            xaxis=dict(tickangle=45), yaxis=dict(title='Yield (t/ha)'))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════
# PAGE 3: Model Performance
# ═══════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<div class="header-title">🤖 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Comparing Regression & Classification Models</div>', unsafe_allow_html=True)

    st.subheader("📈 Regression Models (Yield Prediction)")
    best_r2 = max(r['R2'] for r in reg_results.values())
    cols = st.columns(3)
    for i, (nm, res) in enumerate(reg_results.items()):
        with cols[i]:
            badge = " 🏆" if res['R2'] == best_r2 else ""
            st.markdown(f"### {nm}{badge}")
            st.metric("MAE", f"{res['MAE']:.4f}")
            st.metric("RMSE", f"{res['RMSE']:.4f}")
            st.metric("R² Score", f"{res['R2']:.4f}")

    comp = pd.DataFrame([{'Model': n, 'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R²': r['R2']}
                         for n, r in reg_results.items()])
    fig = go.Figure()
    for met, clr in [('MAE','#FF6B6B'),('RMSE','#4ECDC4'),('R²','#45B7D1')]:
        fig.add_trace(go.Bar(name=met, x=comp['Model'], y=comp[met], marker_color=clr,
            text=[f'{v:.3f}' for v in comp[met]], textposition='outside'))
    fig.update_layout(barmode='group', title='Model Comparison', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎯 Actual vs Predicted")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te.values, y=reg_results['Random Forest']['predictions'],
            mode='markers', marker=dict(color='#00CC66', opacity=0.5, size=6), name='Predictions'))
        fig.add_trace(go.Scatter(x=[y_te.min(), y_te.max()], y=[y_te.min(), y_te.max()],
            mode='lines', line=dict(color='red', dash='dash'), name='Perfect'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=400, xaxis_title='Actual', yaxis_title='Predicted')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🌟 Feature Importance")
        imp = reg_models['Random Forest'].feature_importances_
        fi = pd.DataFrame({'Feature': feat_cols, 'Importance': imp}).sort_values('Importance')
        fig = go.Figure(go.Bar(x=fi['Importance'], y=fi['Feature'], orientation='h',
            marker=dict(color=fi['Importance'], colorscale='Viridis'),
            text=[f'{v:.1%}' for v in fi['Importance']], textposition='outside'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🌱 Classification Model (Crop Recommendation)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Random Forest")
    c2.metric("Accuracy", f"{cls_acc*100:.1f}%")
    c3.metric("Classes", f"{df['label'].nunique()} crops")

# ═══════════════════════════════════════════
# PAGE 4: PREDICT & RECOMMEND (MAIN PAGE)
# ═══════════════════════════════════════════
elif page == "🔮 Predict & Recommend":
    st.markdown('<div class="header-title">🔮 Predict & Recommend</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Enter environmental parameters to predict yield & get crop recommendations</div>', unsafe_allow_html=True)

    # Mode toggle
    mode = st.radio("**Select Mode:**",
                    ["🌾 Predict Yield", "🌱 Recommend Crop", "🌾🌱 Both"],
                    horizontal=True, index=2)
    st.markdown("---")

    # Weather API auto-fill
    with st.expander("☁️ **Auto-fill from Weather API** (Enter city name)", expanded=False):
        wc1, wc2 = st.columns([3, 1])
        with wc1: city = st.text_input("Enter City Name", placeholder="e.g. Chennai, Mumbai, London")
        with wc2: fetch_btn = st.button("🌐 Fetch Weather", use_container_width=True)
        if fetch_btn and city:
            w = fetch_weather(city)
            if w:
                st.success(f"✅ Weather for **{w['city']}** — Temp: {w['temperature']}°C, Humidity: {w['humidity']}%, Rain: {w['rainfall']:.0f}mm/month")
                st.session_state['wt'] = w['temperature']
                st.session_state['wh'] = w['humidity']
                st.session_state['wr'] = w['rainfall']
            else:
                st.error("❌ City not found. Check spelling.")

    # Input sliders (2-column layout)
    st.subheader("📥 Input Parameters")
    c1, c2 = st.columns(2)
    dt_t = st.session_state.get('wt', 25.0)
    dt_h = st.session_state.get('wh', 70.0)
    dt_r = st.session_state.get('wr', 150.0)

    with c1:
        st.markdown("**🧪 Soil Nutrients**")
        n = st.slider("Nitrogen (N)", 0, 140, 50, help="Nitrogen content ratio in soil")
        p = st.slider("Phosphorus (P)", 5, 145, 50, help="Phosphorus content ratio in soil")
        k = st.slider("Potassium (K)", 5, 205, 50, help="Potassium content ratio in soil")
        ph = st.slider("Soil pH", 3.5, 10.0, 6.5, step=0.1, help="pH level of the soil")

    with c2:
        st.markdown("**🌤️ Weather Conditions**")
        temp = st.slider("Temperature (°C)", 8.0, 44.0, min(max(float(dt_t), 8.0), 44.0), step=0.5)
        humidity = st.slider("Humidity (%)", 14.0, 100.0, min(max(float(dt_h), 14.0), 100.0), step=0.5)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, min(max(float(dt_r), 20.0), 300.0), step=1.0)

    # Input validation warnings
    for w in get_warnings(n, p, k, temp, humidity, ph, rainfall):
        st.warning(w)

    st.markdown("---")

    # PREDICT BUTTON
    if st.button("🚀 Predict Now", type="primary", use_container_width=True):
        inp = pd.DataFrame({'N': [n], 'P': [p], 'K': [k], 'temperature': [temp],
                            'humidity': [humidity], 'ph': [ph], 'rainfall': [rainfall]})

        all_preds = {nm: m.predict(inp)[0] for nm, m in reg_models.items()}
        best_nm = max(reg_results.keys(), key=lambda k: reg_results[k]['R2'])
        best_pred = all_preds[best_nm]
        ylevel, ycolor = get_yield_level(best_pred)

        crop_code = clf_model.predict(inp)[0]
        crop_proba = clf_model.predict_proba(inp)[0]
        crop_name = le.inverse_transform([crop_code])[0]
        top3 = [(le.inverse_transform([i])[0], crop_proba[i]) for i in np.argsort(crop_proba)[::-1][:3]]

        st.markdown("---")

        # === YIELD ===
        if mode in ["🌾 Predict Yield", "🌾🌱 Both"]:
            st.subheader("🌾 Yield Prediction Results")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(create_gauge(best_pred), use_container_width=True)
                st.markdown(f'<div style="text-align:center;font-size:1.3em;">{ylevel}</div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown("**📊 All Model Predictions:**")
                pdf = pd.DataFrame([
                    {'Model': nm, 'Yield (t/ha)': f"{v:.4f}", '': '🏆' if nm == best_nm else ''}
                    for nm, v in all_preds.items()])
                st.dataframe(pdf, use_container_width=True, hide_index=True)
                fig = go.Figure(go.Bar(x=list(all_preds.keys()), y=list(all_preds.values()),
                    marker=dict(color=['#00CC66','#4ECDC4','#FF6B6B']),
                    text=[f'{v:.2f}' for v in all_preds.values()], textposition='outside'))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'), height=250, margin=dict(t=20,b=20), yaxis_title='Yield')
                st.plotly_chart(fig, use_container_width=True)

        # === CROP RECOMMENDATION ===
        if mode in ["🌱 Recommend Crop", "🌾🌱 Both"]:
            st.subheader("🌱 Crop Recommendation")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(
                    f'<div class="crop-badge">🏆 Recommended Crop<br/><br/>'
                    f'<span style="font-size:2em;">{crop_name.upper()}</span></div>',
                    unsafe_allow_html=True)
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {top3[0][1]*100:.1f}%")
            with c2:
                st.markdown("**🥈 Top 3 Crop Recommendations:**")
                medals = ["🥇", "🥈", "🥉"]
                for i, (cr, pr) in enumerate(top3):
                    st.markdown(f"{medals[i]} **{cr.title()}** — {pr*100:.1f}%")
                    st.progress(pr)

        # === INTERPRETATION ===
        st.markdown("---")
        st.subheader("🧠 Smart Interpretation")
        importances = reg_models['Random Forest'].feature_importances_
        interp = get_interpretation(inp, importances, feat_cols)
        st.markdown(interp)

        # === DOWNLOAD REPORT ===
        st.markdown("---")
        report = gen_report(inp, all_preds, crop_name, ylevel, interp)
        st.download_button("📥 Download Prediction Report", data=report,
            file_name=f"crop_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain", use_container_width=True)

        with st.expander("📋 Your Input Parameters", expanded=False):
            st.dataframe(inp, use_container_width=True, hide_index=True)
