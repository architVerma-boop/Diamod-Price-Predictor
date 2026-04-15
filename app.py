import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --bg: #0a0e1a;
  --surface: #111827;
  --surface2: #1a2234;
  --accent: #00d4ff;
  --accent2: #7c3aed;
  --accent3: #10b981;
  --warn: #f59e0b;
  --danger: #ef4444;
  --text: #e2e8f0;
  --muted: #64748b;
  --border: #1e293b;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Hero Banner ── */
.hero-banner {
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
  border: 1px solid #312e81;
  border-radius: 16px;
  padding: 2rem 2.5rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse at 20% 50%, rgba(124,58,237,0.15) 0%, transparent 60%),
              radial-gradient(ellipse at 80% 50%, rgba(0,212,255,0.1) 0%, transparent 60%);
  pointer-events: none;
}
.hero-title {
  font-family: 'Space Mono', monospace;
  font-size: 2.2rem;
  font-weight: 700;
  background: linear-gradient(90deg, #00d4ff, #7c3aed, #10b981);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0; letter-spacing: -0.02em;
}
.hero-sub {
  color: var(--muted); font-size: 0.95rem; margin-top: 0.4rem;
  font-family: 'Space Mono', monospace; letter-spacing: 0.05em;
}

/* ── Step Pipeline Strip ── */
.pipeline-strip {
  display: flex;
  gap: 0;
  margin-bottom: 2rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  position: relative;
}
.step-item {
  flex: 1;
  padding: 0.8rem 0.5rem;
  text-align: center;
  position: relative;
  cursor: default;
  transition: all 0.3s ease;
  border-right: 1px solid var(--border);
}
.step-item:last-child { border-right: none; }
.step-item.active {
  background: linear-gradient(180deg, rgba(0,212,255,0.12) 0%, rgba(0,212,255,0.04) 100%);
  border-bottom: 2px solid var(--accent);
}
.step-item.done {
  background: linear-gradient(180deg, rgba(16,185,129,0.08) 0%, transparent 100%);
  border-bottom: 2px solid var(--accent3);
}
.step-item.locked { opacity: 0.35; }
.step-number {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem; font-weight: 700;
  color: var(--muted); letter-spacing: 0.1em;
  display: block; margin-bottom: 0.25rem;
}
.step-item.active .step-number { color: var(--accent); }
.step-item.done .step-number { color: var(--accent3); }
.step-label {
  font-size: 0.7rem; font-weight: 600;
  color: var(--muted); display: block;
  line-height: 1.3;
}
.step-item.active .step-label { color: var(--text); }
.step-item.done .step-label { color: var(--accent3); }
.step-icon { font-size: 1.1rem; display: block; margin-bottom: 0.15rem; }

/* ── Section Card ── */
.section-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}
.section-header {
  display: flex; align-items: center; gap: 0.75rem;
  margin-bottom: 1.25rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
}
.section-badge {
  background: linear-gradient(135deg, var(--accent2), var(--accent));
  color: white; font-family: 'Space Mono', monospace;
  font-size: 0.65rem; font-weight: 700;
  padding: 0.2rem 0.6rem; border-radius: 20px;
  letter-spacing: 0.1em;
}
.section-title {
  font-size: 1.05rem; font-weight: 600;
  color: var(--text); margin: 0;
}

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-card {
  flex: 1; min-width: 140px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px; padding: 1rem;
  position: relative; overflow: hidden;
}
.metric-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.metric-card.blue::before { background: var(--accent); }
.metric-card.purple::before { background: var(--accent2); }
.metric-card.green::before { background: var(--accent3); }
.metric-card.amber::before { background: var(--warn); }
.metric-label { font-size: 0.7rem; color: var(--muted); font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: var(--text); margin-top: 0.25rem; }
.metric-sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.15rem; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
  color: white !important; border: none !important;
  border-radius: 8px !important; font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  padding: 0.5rem 1.5rem !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.02em !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Secondary button */
.btn-secondary > button {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div {
  background: var(--surface2) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
.stRadio > div { gap: 1rem !important; }
.stRadio label { color: var(--text) !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 8px !important; overflow: hidden !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--surface2) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-weight: 600 !important;
}
.streamlit-expanderContent {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
}

/* ── Alerts ── */
.info-box {
  background: rgba(0,212,255,0.08);
  border: 1px solid rgba(0,212,255,0.3);
  border-radius: 8px; padding: 0.75rem 1rem;
  color: #7dd3fc; font-size: 0.875rem; margin-bottom: 1rem;
}
.warn-box {
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.3);
  border-radius: 8px; padding: 0.75rem 1rem;
  color: #fcd34d; font-size: 0.875rem; margin-bottom: 1rem;
}
.success-box {
  background: rgba(16,185,129,0.08);
  border: 1px solid rgba(16,185,129,0.3);
  border-radius: 8px; padding: 0.75rem 1rem;
  color: #6ee7b7; font-size: 0.875rem; margin-bottom: 1rem;
}
.danger-box {
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.3);
  border-radius: 8px; padding: 0.75rem 1rem;
  color: #fca5a5; font-size: 0.875rem; margin-bottom: 1rem;
}

/* ── Progress / Divider ── */
.step-divider {
  border: none; border-top: 1px dashed var(--border);
  margin: 1.5rem 0;
}

/* ── Checkbox ── */
.stCheckbox label { color: var(--text) !important; font-size: 0.9rem !important; }

/* ── Slider ── */
.stSlider > div > div > div { background: var(--accent2) !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface2) !important;
  border-radius: 8px !important; gap: 0 !important;
  padding: 0.25rem !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  border-radius: 6px !important;
  font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent2) !important;
  color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Lazy imports (only when needed) ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_sklearn_imports():
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
    from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.svm import SVC, SVR
    from sklearn.cluster import KMeans
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                  mean_squared_error, r2_score, mean_absolute_error,
                                  roc_auc_score, roc_curve)
    return True

# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "step": 0,
        "problem_type": None,
        "df_raw": None,
        "df": None,
        "target": None,
        "features": None,
        "df_clean": None,
        "outlier_indices": [],
        "selected_features": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "model_name": None,
        "model": None,
        "cv_scores": None,
        "y_pred": None,
        "best_model": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
S = st.session_state

# ── Pipeline steps definition ─────────────────────────────────────────────────
STEPS = [
    ("🎯", "Problem\nType"),
    ("📂", "Data\nInput"),
    ("🔍", "EDA"),
    ("🔧", "Data\nEngineering"),
    ("⚡", "Feature\nSelection"),
    ("✂️", "Data\nSplit"),
    ("🤖", "Model\nSelection"),
    ("🏋️", "Training\n& KFold"),
    ("📊", "Performance\nMetrics"),
    ("🎛️", "Hyper-\nParameter Tuning"),
]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🧬 ML Pipeline Studio</div>
  <div class="hero-sub">// END-TO-END MACHINE LEARNING WORKFLOW ORCHESTRATOR</div>
</div>
""", unsafe_allow_html=True)

# ── Pipeline Strip ────────────────────────────────────────────────────────────
strip_html = '<div class="pipeline-strip">'
for i, (icon, label) in enumerate(STEPS):
    if i < S.step:
        cls = "done"
        num_label = "✓"
    elif i == S.step:
        cls = "active"
        num_label = f"STEP {i+1:02d}"
    else:
        cls = "locked"
        num_label = f"STEP {i+1:02d}"
    strip_html += f"""
    <div class="step-item {cls}">
      <span class="step-icon">{icon}</span>
      <span class="step-number">{num_label}</span>
      <span class="step-label">{label.replace(chr(10), '<br>')}</span>
    </div>"""
strip_html += "</div>"
st.markdown(strip_html, unsafe_allow_html=True)

# ── Plotly theme helper ───────────────────────────────────────────────────────
def dark_fig(fig, height=400):
    fig.update_layout(
        paper_bgcolor="rgba(17,24,39,0)",
        plot_bgcolor="rgba(17,24,39,0)",
        font=dict(family="DM Sans", color="#94a3b8", size=12),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    return fig

PALETTE = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b", "#ef4444",
           "#8b5cf6", "#06b6d4", "#34d399", "#fbbf24", "#f87171"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Problem Type
# ══════════════════════════════════════════════════════════════════════════════
if S.step == 0:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <span class="section-badge">STEP 01</span>
      <span class="section-title">🎯 Define Problem Type</span>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="metric-card blue">', unsafe_allow_html=True)
        st.markdown("### 📈 Regression")
        st.markdown("Predict **continuous numerical** values. Best for forecasting, price prediction, demand estimation, and trend analysis.")
        st.markdown("**Algorithms:** Linear Regression, SVR, Random Forest Regressor")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card purple">', unsafe_allow_html=True)
        st.markdown("### 🏷️ Classification")
        st.markdown("Predict **discrete categories** or labels. Best for spam detection, image recognition, medical diagnosis, churn prediction.")
        st.markdown("**Algorithms:** Logistic Regression, SVM, Random Forest Classifier, KMeans")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    prob = st.radio("Select your problem type:", ["Classification", "Regression"], horizontal=True, key="prob_radio")
    if st.button("Confirm Problem Type →"):
        S.problem_type = prob
        S.step = 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Input
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-header">
      <span class="section-badge">STEP 02</span>
      <span class="section-title">📂 Data Input — {S.problem_type}</span>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 Upload CSV", "📋 Paste Data"])

    with tab1:
        uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df = pd.read_csv(uploaded)
            S.df_raw = df
            st.markdown(f'<div class="success-box">✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns</div>', unsafe_allow_html=True)

    with tab2:
        sample_csv = """age,income,education,purchased
25,45000,16,1
32,62000,18,1
28,38000,14,0
45,85000,16,1
22,28000,12,0
38,71000,20,1
55,92000,18,1
29,41000,14,0"""
        pasted = st.text_area("Paste CSV data:", value=sample_csv, height=150)
        if st.button("Parse CSV Data"):
            try:
                df = pd.read_csv(io.StringIO(pasted))
                S.df_raw = df
                st.markdown(f'<div class="success-box">✅ Parsed {df.shape[0]:,} rows × {df.shape[1]} columns</div>', unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.markdown(f'<div class="danger-box">❌ Parse error: {e}</div>', unsafe_allow_html=True)

    if S.df_raw is not None:
        df = S.df_raw
        st.markdown("<hr class='step-divider'>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1], gap="medium")
        with col_left:
            st.markdown("#### Select Target Feature")
            target = st.selectbox("Target column:", df.columns.tolist(), key="target_sel")
            features_opts = [c for c in df.columns if c != target]
            features = st.multiselect("Feature columns (select for PCA view):", features_opts, default=features_opts[:min(6, len(features_opts))], key="feat_sel")

        with col_right:
            # Dataset shape metrics
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            missing = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card blue">
                <div class="metric-label">Rows</div>
                <div class="metric-value">{df.shape[0]:,}</div>
              </div>
              <div class="metric-card purple">
                <div class="metric-label">Columns</div>
                <div class="metric-value">{df.shape[1]}</div>
              </div>
              <div class="metric-card green">
                <div class="metric-label">Numeric</div>
                <div class="metric-value">{len(num_cols)}</div>
              </div>
              <div class="metric-card amber">
                <div class="metric-label">Missing</div>
                <div class="metric-value">{missing}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # PCA Visualization
            if len(features) >= 2:
                st.markdown("<br>**📊 PCA Shape of Data**", unsafe_allow_html=True)
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import LabelEncoder
    
                    pca_df = df[features].copy()
                    # Encode categoricals
                    for c in pca_df.select_dtypes(exclude=np.number).columns:
                        pca_df[c] = LabelEncoder().fit_transform(pca_df[c].astype(str))
                    pca_df = pca_df.fillna(pca_df.mean())
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(pca_df)
                    n_comp = min(3, Xs.shape[1], Xs.shape[0])
                    pca = PCA(n_components=n_comp)
                    comps = pca.fit_transform(Xs)
                    expl = pca.explained_variance_ratio_
    
                    color_vals = df[target].astype(str) if target else None
    
                    if n_comp >= 3:
                        fig_pca = px.scatter_3d(
                            x=comps[:, 0], y=comps[:, 1], z=comps[:, 2],
                            color=color_vals,
                            labels={"x": f"PC1 ({expl[0]*100:.1f}%)", "y": f"PC2 ({expl[1]*100:.1f}%)", "z": f"PC3 ({expl[2]*100:.1f}%)"},
                            color_discrete_sequence=PALETTE, opacity=0.85
                        )
                    else:
                        fig_pca = px.scatter(
                            x=comps[:, 0], y=comps[:, 1],
                            color=color_vals,
                            labels={"x": f"PC1 ({expl[0]*100:.1f}%)", "y": f"PC2 ({expl[1]*100:.1f}%)"},
                            color_discrete_sequence=PALETTE, opacity=0.85
                        )
                    dark_fig(fig_pca, 450)
                    fig_pca.update_layout(title=f"PCA Projection — {sum(expl)*100:.1f}% variance explained")
                    st.plotly_chart(fig_pca, use_container_width=True)

                # Variance bar
                    fig_var = px.bar(
                        x=[f"PC{i+1}" for i in range(len(expl))],
                        y=expl * 100,
                        color_discrete_sequence=[PALETTE[0]],
                        labels={"x": "Component", "y": "Explained Variance (%)"}
                    )
                    dark_fig(fig_var, 200)
                    st.plotly_chart(fig_var, use_container_width=True)
                except Exception as e:
                    st.markdown(f'<div class="warn-box">⚠️ PCA failed: {e}</div>', unsafe_allow_html=True)
    
            with st.expander("🔍 Preview Data"):
                st.dataframe(df.head(20), use_container_width=True)
    
            if st.button("Proceed to EDA →"):
                S.target = target
                S.features = features if features else [c for c in df.columns if c != target]
                S.df = df.copy()
                S.df_clean = df.copy()
                S.step = 2
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 2:
    df = S.df
    target = S.target
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <span class="section-badge">STEP 03</span>
      <span class="section-title">🔍 Exploratory Data Analysis</span>
    </div>""", unsafe_allow_html=True)

    # Summary stats
    stats = df.describe().T
    missing_pct = (df.isnull().sum() / len(df) * 100).rename("missing%")
    stats = stats.join(missing_pct)

    with st.expander("📋 Descriptive Statistics", expanded=True):
        st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)

    tab_dist, tab_corr, tab_box, tab_miss = st.tabs(["📊 Distributions", "🔗 Correlations", "📦 Box Plots", "❓ Missing Values"])

    with tab_dist:
        if num_cols:
            col_sel = st.selectbox("Select column:", num_cols, key="dist_col")
            fig_hist = px.histogram(df, x=col_sel, color_discrete_sequence=[PALETTE[0]],
                                     marginal="rug", nbins=40)
            dark_fig(fig_hist, 350)
            fig_hist.update_layout(title=f"Distribution of {col_sel}")
            st.plotly_chart(fig_hist, use_container_width=True)

            # Target distribution
            if target:
                fig_tgt = px.histogram(df, x=target, color_discrete_sequence=[PALETTE[1]], nbins=30)
                dark_fig(fig_tgt, 300)
                fig_tgt.update_layout(title=f"Target Distribution: {target}")
                st.plotly_chart(fig_tgt, use_container_width=True)

    with tab_corr:
        num_df = df[num_cols].fillna(0)
        if len(num_cols) > 1:
            corr = num_df.corr()
            fig_heat = px.imshow(corr, color_continuous_scale="RdBu_r",
                                  aspect="auto", zmin=-1, zmax=1,
                                  text_auto=".2f")
            dark_fig(fig_heat, 500)
            fig_heat.update_layout(title="Correlation Matrix")
            st.plotly_chart(fig_heat, use_container_width=True)

            if target and target in num_cols:
                corr_tgt = corr[target].drop(target).sort_values(key=abs, ascending=False)
                fig_bar = px.bar(x=corr_tgt.index, y=corr_tgt.values,
                                  color=corr_tgt.values,
                                  color_continuous_scale="RdBu_r",
                                  labels={"x": "Feature", "y": f"Correlation with {target}"})
                dark_fig(fig_bar, 300)
                fig_bar.update_layout(title=f"Feature Correlation with Target '{target}'")
                st.plotly_chart(fig_bar, use_container_width=True)

    with tab_box:
        if num_cols:
            cols_box = st.multiselect("Select columns for box plots:", num_cols, default=num_cols[:min(4, len(num_cols))], key="box_sel")
            if cols_box:
                fig_box = go.Figure()
                for i, c in enumerate(cols_box):
                    fig_box.add_trace(go.Box(y=df[c], name=c, marker_color=PALETTE[i % len(PALETTE)],
                                              boxmean=True))
                dark_fig(fig_box, 400)
                fig_box.update_layout(title="Box Plots — Spread & Outlier Preview")
                st.plotly_chart(fig_box, use_container_width=True)

    with tab_miss:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            fig_miss = px.bar(x=missing.index, y=missing.values,
                               color_discrete_sequence=[PALETTE[4]],
                               labels={"x": "Column", "y": "Missing Count"})
            dark_fig(fig_miss, 300)
            fig_miss.update_layout(title="Missing Values by Column")
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.markdown('<div class="success-box">✅ No missing values found in the dataset!</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            S.step = 1; st.rerun()
    with col2:
        if st.button("Proceed to Data Engineering →"):
            S.step = 3; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Data Engineering & Cleaning
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 3:
    df = S.df_clean.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <span class="section-badge">STEP 04</span>
      <span class="section-title">🔧 Data Engineering & Cleaning</span>
    </div>""", unsafe_allow_html=True)

    # Missing value imputation
    st.markdown("#### 🩹 Missing Value Imputation")
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    if missing_cols:
        imp_method = st.selectbox("Imputation method:", ["Mean", "Median", "Mode"], key="imp_method")
        if st.button("Apply Imputation"):
            for c in missing_cols:
                if df[c].dtype in [np.float64, np.int64]:
                    if imp_method == "Mean":
                        df[c].fillna(df[c].mean(), inplace=True)
                    elif imp_method == "Median":
                        df[c].fillna(df[c].median(), inplace=True)
                    else:
                        df[c].fillna(df[c].mode()[0], inplace=True)
                else:
                    df[c].fillna(df[c].mode()[0], inplace=True)
            S.df_clean = df
            st.markdown('<div class="success-box">✅ Imputation applied!</div>', unsafe_allow_html=True)
            st.rerun()
    else:
        st.markdown('<div class="info-box">ℹ️ No missing values to impute.</div>', unsafe_allow_html=True)

    st.markdown("<hr class='step-divider'>", unsafe_allow_html=True)

    # Outlier detection
    st.markdown("#### 🔎 Outlier Detection")
    if num_cols:
        outlier_method = st.selectbox("Detection method:", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"], key="out_method")
        outlier_features = st.multiselect("Features for outlier detection:", num_cols,
                                           default=num_cols[:min(4, len(num_cols))], key="out_feats")

        if st.button("Detect Outliers") and outlier_features:
            X_out = df[outlier_features].fillna(df[outlier_features].mean())

            if outlier_method == "IQR":
                mask = pd.Series([False] * len(df))
                for c in outlier_features:
                    Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
                    IQR = Q3 - Q1
                    mask |= (df[c] < Q1 - 1.5 * IQR) | (df[c] > Q3 + 1.5 * IQR)
                S.outlier_indices = df[mask].index.tolist()

            elif outlier_method == "Isolation Forest":
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
                Xs = StandardScaler().fit_transform(X_out)
                preds = IsolationForest(contamination=0.1, random_state=42).fit_predict(Xs)
                S.outlier_indices = df[preds == -1].index.tolist()

            elif outlier_method == "DBSCAN":
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler
                Xs = StandardScaler().fit_transform(X_out)
                preds = DBSCAN(eps=0.5, min_samples=5).fit_predict(Xs)
                S.outlier_indices = df[preds == -1].index.tolist()

            elif outlier_method == "OPTICS":
                from sklearn.cluster import OPTICS
                from sklearn.preprocessing import StandardScaler
                Xs = StandardScaler().fit_transform(X_out)
                preds = OPTICS(min_samples=5).fit_predict(Xs)
                S.outlier_indices = df[preds == -1].index.tolist()

            st.rerun()

        if S.outlier_indices:
            n_out = len(S.outlier_indices)
            pct = n_out / len(df) * 100
            st.markdown(f'<div class="warn-box">⚠️ Detected <b>{n_out} outliers</b> ({pct:.1f}% of data) using {outlier_method}</div>',
                         unsafe_allow_html=True)

            # Visualize outliers
            if len(outlier_features) >= 2:
                plot_df = df[outlier_features[:2]].copy()
                plot_df["_outlier"] = "Normal"
                plot_df.loc[S.outlier_indices, "_outlier"] = "Outlier"
                fig_out = px.scatter(plot_df, x=outlier_features[0], y=outlier_features[1],
                                      color="_outlier",
                                      color_discrete_map={"Normal": PALETTE[2], "Outlier": PALETTE[4]},
                                      opacity=0.7)
                dark_fig(fig_out, 380)
                fig_out.update_layout(title=f"Outlier Visualization — {outlier_method}")
                st.plotly_chart(fig_out, use_container_width=True)

            with st.expander("👁️ Preview outlier rows"):
                st.dataframe(df.loc[S.outlier_indices], use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Remove Outliers"):
                    S.df_clean = df.drop(index=S.outlier_indices).reset_index(drop=True)
                    S.outlier_indices = []
                    st.markdown(f'<div class="success-box">✅ Outliers removed. New shape: {S.df_clean.shape}</div>', unsafe_allow_html=True)
                    st.rerun()
            with col_b:
                if st.button("Keep Outliers"):
                    S.outlier_indices = []
                    st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            S.step = 2; st.rerun()
    with col2:
        if st.button("Proceed to Feature Selection →"):
            S.df_clean = df
            S.step = 4; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Feature Selection
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 4:
    df = S.df_clean.copy()
    target = S.target
    from sklearn.preprocessing import LabelEncoder

    # Encode target if classification
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    feature_cols = [c for c in df.columns if c != target and c in num_cols]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <span class="section-badge">STEP 05</span>
      <span class="section-title">⚡ Feature Selection</span>
    </div>""", unsafe_allow_html=True)

    fs_method = st.selectbox("Feature selection method:",
                               ["Variance Threshold", "Correlation (with target)", "Information Gain (with target)"],
                               key="fs_method")

    if st.button("Run Feature Selection") and feature_cols:
        try:
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y_raw = df[target].fillna(df[target].mode()[0])
            if y_raw.dtype == object:
                y = LabelEncoder().fit_transform(y_raw.astype(str))
            else:
                y = y_raw.values

            if fs_method == "Variance Threshold":
                from sklearn.feature_selection import VarianceThreshold
                thresh = st.slider("Variance threshold:", 0.0, 1.0, 0.01, key="var_thresh")
                sel = VarianceThreshold(threshold=thresh)
                sel.fit(X)
                scores = pd.Series(sel.variances_, index=feature_cols, name="Variance")
                selected = [feature_cols[i] for i, s in enumerate(sel.get_support()) if s]

            elif fs_method == "Correlation (with target)":
                scores = X.corrwith(pd.Series(y, name=target)).abs().rename("Abs Correlation")
                thresh_corr = st.slider("Min correlation threshold:", 0.0, 1.0, 0.1, key="corr_thresh")
                selected = scores[scores >= thresh_corr].index.tolist()

            else:  # Information Gain
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                if S.problem_type == "Classification":
                    mi = mutual_info_classif(X, y, random_state=42)
                else:
                    mi = mutual_info_regression(X, y, random_state=42)
                scores = pd.Series(mi, index=feature_cols, name="Information Gain")
                thresh_ig = st.slider("Min IG threshold:", 0.0, float(scores.max()) + 0.01, 0.01, key="ig_thresh")
                selected = scores[scores >= thresh_ig].index.tolist()

            # Store
            S.selected_features = selected if selected else feature_cols
            scores_df = scores.sort_values(ascending=False).reset_index()
            scores_df.columns = ["Feature", scores.name]

            fig_fs = px.bar(scores_df, x="Feature", y=scores.name,
                             color=scores.name, color_continuous_scale="Viridis",
                             labels={"Feature": "Feature", scores.name: scores.name})
            dark_fig(fig_fs, 350)
            fig_fs.update_layout(title=f"Feature Scores — {fs_method}")
            # Highlight selected
            colors = [PALETTE[2] if f in selected else PALETTE[4] for f in scores_df["Feature"]]
            fig_fs.data[0].marker.color = colors
            st.plotly_chart(fig_fs, use_container_width=True)

            st.markdown(f'<div class="success-box">✅ Selected <b>{len(selected)}</b> features: {", ".join(selected)}</div>',
                         unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="danger-box">❌ Error: {e}</div>', unsafe_allow_html=True)

    if S.selected_features is None and feature_cols:
        S.selected_features = feature_cols

    # Manual override
    if feature_cols:
        manual = st.multiselect("Or manually select features:", feature_cols,
                                 default=S.selected_features if S.selected_features else feature_cols,
                                 key="manual_feats")
        if manual:
            S.selected_features = manual

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            S.step = 3; st.rerun()
    with col2:
        if st.button("Proceed to Data Split →") and S.selected_features:
            S.step = 5; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Data Split
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 5:
    df = S.df_clean.copy()
    target = S.target
    features = S.selected_features

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <span class="section-badge">STEP 06</span>
      <span class="section-title">✂️ Train / Test Split</span>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05, key="test_size_sl")
        random_state = st.number_input("Random state:", value=42, step=1, key="rand_state")
        stratify_opt = st.checkbox("Stratify split (classification only)", value=(S.problem_type == "Classification"))

    with col2:
        n = len(df)
        n_test = int(n * test_size)
        n_train = n - n_test
        pct_train = (1 - test_size) * 100
        pct_test = test_size * 100

        fig_split = go.Figure(go.Pie(
            values=[n_train, n_test],
            labels=["Train", "Test"],
            hole=0.55,
            marker_colors=[PALETTE[0], PALETTE[1]],
            textinfo="label+percent",
            textfont_size=14,
        ))
        dark_fig(fig_split, 280)
        fig_split.update_layout(
            title="Split Ratio",
            annotations=[dict(text=f"{n:,}", x=0.5, y=0.5, font_size=22, showarrow=False,
                               font_color="#e2e8f0")]
        )
        st.plotly_chart(fig_split, use_container_width=True)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card blue">
        <div class="metric-label">Train Samples</div>
        <div class="metric-value">{n_train:,}</div>
        <div class="metric-sub">{pct_train:.0f}% of data</div>
      </div>
      <div class="metric-card purple">
        <div class="metric-label">Test Samples</div>
        <div class="metric-value">{n_test:,}</div>
        <div class="metric-sub">{pct_test:.0f}% of data</div>
      </div>
      <div class="metric-card green">
        <div class="metric-label">Features</div>
        <div class="metric-value">{len(features)}</div>
        <div class="metric-sub">selected columns</div>
      </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            S.step = 4; st.rerun()
    with col2:
        if st.button("Apply Split & Continue →"):
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            try:
                feat_cols = [c for c in features if c in df.columns and c != target]
                X = df[feat_cols].fillna(0)
                # encode non-numeric
                for c in X.select_dtypes(exclude=np.number).columns:
                    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
                y_raw = df[target].fillna(df[target].mode()[0])
                if y_raw.dtype == object or S.problem_type == "Classification":
                    y = LabelEncoder().fit_transform(y_raw.astype(str))
                else:
                    y = y_raw.values

                strat = y if stratify_opt and S.problem_type == "Classification" and len(np.unique(y)) < 20 else None
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state), stratify=strat
                )
                S.X_train, S.X_test, S.y_train, S.y_test = X_tr, X_te, y_tr, y_te
                S.step = 6
                st.rerun()
            except Exception as e:
                st.markdown(f'<div class="danger-box">❌ Split error: {e}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Model Selection
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <span class="section-badge">STEP 07</span>
      <span class="section-title">🤖 Model Selection</span>
    </div>""", unsafe_allow_html=True)

    if S.problem_type == "Classification":
        model_opts = ["Logistic Regression", "SVM (Classifier)", "Random Forest (Classifier)", "KMeans (Clustering)"]
    else:
        model_opts = ["Linear Regression", "SVM (Regressor)", "Random Forest (Regressor)"]

    model_cards = {
        "Logistic Regression": ("🔵", "Fast linear classifier. Best for linearly separable data. Low variance, high bias.", "Low", "High"),
        "SVM (Classifier)": ("🟣", "Powerful with kernel trick. Great for high-dimensional spaces.", "Medium", "Medium"),
        "Random Forest (Classifier)": ("🟢", "Ensemble of trees. Robust, handles non-linearity well.", "High", "Low"),
        "KMeans (Clustering)": ("🟡", "Unsupervised clustering. Groups data into K clusters.", "Low", "High"),
        "Linear Regression": ("🔵", "Predicts continuous values with a linear model. Fast and interpretable.", "Low", "High"),
        "SVM (Regressor)": ("🟣", "Support Vector Regression. Good for non-linear patterns.", "Medium", "Medium"),
        "Random Forest (Regressor)": ("🟢", "Ensemble regressor. High accuracy, handles complex patterns.", "High", "Low"),
    }

    cols = st.columns(len(model_opts))
    selected_model = S.model_name or model_opts[0]

    for i, (col, m) in enumerate(zip(cols, model_opts)):
        with col:
            icon, desc, comp, speed = model_cards[m]
            border = "border: 2px solid #00d4ff;" if m == selected_model else ""
            st.markdown(f"""
            <div class="metric-card {'blue' if m == selected_model else ''}" style="{border} cursor:pointer; padding:1rem;">
              <div style="font-size:1.5rem">{icon}</div>
              <div style="font-weight:700; margin:0.4rem 0; font-size:0.85rem">{m}</div>
              <div style="font-size:0.72rem; color:#94a3b8; line-height:1.5">{desc}</div>
              <div style="margin-top:0.6rem; font-size:0.68rem; color:#64748b">
                Complexity: {comp}<br>Speed: {speed}
              </div>
            </div>""", unsafe_allow_html=True)

    selected_model = st.selectbox("Select model:", model_opts,
                                   index=model_opts.index(S.model_name) if S.model_name in model_opts else 0,
                                   key="model_sel_box")

    # Extra params
    extra_params = {}
    if "SVM" in selected_model:
        kernel = st.selectbox("SVM Kernel:", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
        extra_params["kernel"] = kernel
        C = st.slider("Regularization (C):", 0.01, 100.0, 1.0, key="svm_C")
        extra_params["C"] = C
    if "Random Forest" in selected_model:
        n_est = st.slider("Number of trees:", 10, 500, 100, 10, key="rf_n_est")
        extra_params["n_estimators"] = n_est
    if "KMeans" in selected_model:
        k = st.number_input("Number of clusters (K):", 2, 20, 3, key="km_k")
        extra_params["n_clusters"] = k

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            S.step = 5; st.rerun()
    with col2:
        if st.button("Confirm Model & Continue →"):
            S.model_name = selected_model
            S.model_params = extra_params
            S.step = 7; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Training & KFold
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 7:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <span class="section-badge">STEP 08</span>
      <span class="section-title">🏋️ Model Training & K-Fold Cross Validation</span>
    </div>""", unsafe_allow_html=True)

    params = getattr(S, "model_params", {})
    k_folds = st.number_input("Number of K-Fold splits (K):", min_value=2, max_value=20, value=5, key="k_folds_n")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""
        <div class="info-box">
          📋 <b>Model:</b> {S.model_name}<br>
          📋 <b>K-Folds:</b> {k_folds}<br>
          📋 <b>Train samples:</b> {len(S.X_train):,}<br>
          📋 <b>Test samples:</b> {len(S.X_test):,}
        </div>""", unsafe_allow_html=True)

    with col2:
        # KFold diagram
        fig_kf = go.Figure()
        n_vis = min(k_folds, 10)
        for fold in range(n_vis):
            for block in range(n_vis):
                color = "#00d4ff" if block == fold else "#1a2234"
                border_color = "#00d4ff" if block == fold else "#1e293b"
                fig_kf.add_shape(type="rect",
                                  x0=block, x1=block + 0.9, y0=fold, y1=fold + 0.85,
                                  fillcolor=color, line_color=border_color)
            fig_kf.add_annotation(x=-0.3, y=fold + 0.4, text=f"Fold {fold+1}",
                                   showarrow=False, font=dict(size=9, color="#94a3b8"), xanchor="right")
        dark_fig(fig_kf, 30 * n_vis + 80)
        fig_kf.update_layout(
            title="K-Fold Splits (blue = test)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig_kf, use_container_width=True)

    if st.button("🚀 Train Model with K-Fold CV"):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import KFold, cross_val_score

        # Build model
        try:
            if S.model_name == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, **{k: v for k, v in params.items() if k not in ["kernel"]})
                scoring = "accuracy"
            elif S.model_name == "SVM (Classifier)":
                from sklearn.svm import SVC
                model = SVC(probability=True, **params)
                scoring = "accuracy"
            elif S.model_name == "Random Forest (Classifier)":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42, **params)
                scoring = "accuracy"
            elif S.model_name == "KMeans (Clustering)":
                from sklearn.cluster import KMeans
                n_cl = params.get("n_clusters", 3)
                model = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
                scoring = None
            elif S.model_name == "Linear Regression":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                scoring = "r2"
            elif S.model_name == "SVM (Regressor)":
                from sklearn.svm import SVR
                model = SVR(**params)
                scoring = "r2"
            elif S.model_name == "Random Forest (Regressor)":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42, **params)
                scoring = "r2"

            # Scale
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(S.X_train)
            X_te_s = scaler.transform(S.X_test)

            # KFold CV
            if scoring and S.model_name != "KMeans (Clustering)":
                kf = KFold(n_splits=int(k_folds), shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_tr_s, S.y_train, cv=kf, scoring=scoring)
                S.cv_scores = cv_scores
            else:
                S.cv_scores = None

            # Train final model
            with st.spinner("Training..."):
                if S.model_name == "KMeans (Clustering)":
                    model.fit(X_tr_s)
                    S.y_pred = model.predict(X_te_s)
                else:
                    model.fit(X_tr_s, S.y_train)
                    S.y_pred = model.predict(X_te_s)

            S.model = model
            S.scaler = scaler

            # CV plot
            if S.cv_scores is not None:
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Bar(
                    x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                    y=cv_scores,
                    marker_color=[PALETTE[0] if s >= cv_scores.mean() else PALETTE[4] for s in cv_scores]
                ))
                fig_cv.add_hline(y=cv_scores.mean(), line_dash="dot", line_color="#f59e0b",
                                  annotation_text=f"Mean: {cv_scores.mean():.4f}")
                dark_fig(fig_cv, 320)
                fig_cv.update_layout(title=f"K-Fold CV Scores ({scoring})")
                st.plotly_chart(fig_cv, use_container_width=True)

                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-card blue">
                    <div class="metric-label">Mean CV Score</div>
                    <div class="metric-value">{cv_scores.mean():.4f}</div>
                  </div>
                  <div class="metric-card purple">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value">{cv_scores.std():.4f}</div>
                  </div>
                  <div class="metric-card green">
                    <div class="metric-label">Best Fold</div>
                    <div class="metric-value">{cv_scores.max():.4f}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="success-box">✅ Training complete! Proceed to Performance Metrics.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="danger-box">❌ Training error: {e}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            S.step = 6; st.rerun()
    with col2:
        if st.button("View Performance Metrics →") and S.model is not None:
            S.step = 8; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Performance Metrics
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 8:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <span class="section-badge">STEP 09</span>
      <span class="section-title">📊 Performance Metrics & Overfitting Analysis</span>
    </div>""", unsafe_allow_html=True)

    try:
        scaler = S.scaler
        model = S.model
        X_tr_s = scaler.transform(S.X_train)
        X_te_s = scaler.transform(S.X_test)
        y_pred_train = model.predict(X_tr_s)
        y_pred_test = S.y_pred

        if S.problem_type == "Classification" and S.model_name != "KMeans (Clustering)":
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

            train_acc = accuracy_score(S.y_train, y_pred_train)
            test_acc = accuracy_score(S.y_test, y_pred_test)
            gap = train_acc - test_acc

            # Overfitting analysis
            if gap > 0.15:
                fit_status = "OVERFITTING"; fit_color = "danger-box"; fit_icon = "⚠️"
                fit_msg = f"Training accuracy ({train_acc:.3f}) is much higher than test ({test_acc:.3f}). Gap = {gap:.3f}"
            elif test_acc > train_acc + 0.05:
                fit_status = "UNDERFITTING"; fit_color = "warn-box"; fit_icon = "📉"
                fit_msg = f"Test accuracy exceeds train. Model may be underfitting. Consider increasing complexity."
            else:
                fit_status = "GOOD FIT"; fit_color = "success-box"; fit_icon = "✅"
                fit_msg = f"Train ({train_acc:.3f}) ≈ Test ({test_acc:.3f}). Model generalizes well."

            st.markdown(f'<div class="{fit_color}">{fit_icon} <b>{fit_status}</b>: {fit_msg}</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card blue">
                <div class="metric-label">Train Accuracy</div>
                <div class="metric-value">{train_acc:.4f}</div>
              </div>
              <div class="metric-card {'green' if gap < 0.1 else 'amber'}">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value">{test_acc:.4f}</div>
              </div>
              <div class="metric-card {'green' if gap < 0.1 else 'danger'}">
                <div class="metric-label">Gap (Train-Test)</div>
                <div class="metric-value">{gap:.4f}</div>
                <div class="metric-sub">{'✅ Healthy' if gap < 0.1 else '⚠️ Overfit risk'}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Confusion Matrix
            cm = confusion_matrix(S.y_test, y_pred_test)
            labels = np.unique(np.concatenate([S.y_test, y_pred_test]))
            fig_cm = px.imshow(cm, text_auto=True,
                                x=[str(l) for l in labels],
                                y=[str(l) for l in labels],
                                color_continuous_scale=[[0, "#0a0e1a"], [0.5, "#1e3a5f"], [1, "#00d4ff"]],
                                aspect="auto")
            dark_fig(fig_cm, 380)
            fig_cm.update_layout(title="Confusion Matrix",
                                  xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)

            # ROC Curve (binary)
            if len(labels) == 2 and hasattr(model, "predict_proba"):
                from sklearn.metrics import roc_curve, roc_auc_score
                y_prob = model.predict_proba(X_te_s)[:, 1]
                fpr, tpr, _ = roc_curve(S.y_test, y_prob)
                auc = roc_auc_score(S.y_test, y_prob)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc:.3f}",
                                              line=dict(color=PALETTE[0], width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                              line=dict(color="#64748b", dash="dash")))
                dark_fig(fig_roc, 350)
                fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig_roc, use_container_width=True)

            # Report
            with st.expander("📋 Full Classification Report"):
                report = classification_report(S.y_test, y_pred_test, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        else:  # Regression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            train_r2 = r2_score(S.y_train, y_pred_train)
            test_r2 = r2_score(S.y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(S.y_test, y_pred_test))
            mae = mean_absolute_error(S.y_test, y_pred_test)
            gap = train_r2 - test_r2

            if gap > 0.2:
                fit_status = "OVERFITTING"; fit_color = "danger-box"
                fit_msg = f"Train R² ({train_r2:.3f}) >> Test R² ({test_r2:.3f}). Gap = {gap:.3f}"
            elif test_r2 < 0.4:
                fit_status = "UNDERFITTING"; fit_color = "warn-box"
                fit_msg = f"Low test R² ({test_r2:.3f}). Model is not capturing the data well."
            else:
                fit_status = "GOOD FIT"; fit_color = "success-box"
                fit_msg = f"Train R²={train_r2:.3f}, Test R²={test_r2:.3f}. Reasonable generalization."

            st.markdown(f'<div class="{fit_color}">{"⚠️" if "FIT" in fit_status[:3] else "✅"} <b>{fit_status}</b>: {fit_msg}</div>',
                         unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card blue">
                <div class="metric-label">Train R²</div>
                <div class="metric-value">{train_r2:.4f}</div>
              </div>
              <div class="metric-card green">
                <div class="metric-label">Test R²</div>
                <div class="metric-value">{test_r2:.4f}</div>
              </div>
              <div class="metric-card purple">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{rmse:.4f}</div>
              </div>
              <div class="metric-card amber">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{mae:.4f}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Actual vs Predicted
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(x=S.y_test, y=y_pred_test, mode="markers",
                                          marker=dict(color=PALETTE[0], opacity=0.7, size=7),
                                          name="Predictions"))
            lim = [min(S.y_test.min(), y_pred_test.min()), max(S.y_test.max(), y_pred_test.max())]
            fig_avp.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                          line=dict(color=PALETTE[4], dash="dash"), name="Perfect Fit"))
            dark_fig(fig_avp, 380)
            fig_avp.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(fig_avp, use_container_width=True)

            # Residuals
            residuals = S.y_test - y_pred_test
            fig_res = px.histogram(x=residuals, nbins=40, color_discrete_sequence=[PALETTE[2]])
            dark_fig(fig_res, 280)
            fig_res.update_layout(title="Residual Distribution", xaxis_title="Residuals")
            st.plotly_chart(fig_res, use_container_width=True)

        # Learning curve (train vs CV across train sizes)
        if S.cv_scores is not None:
            st.markdown("<br>**📈 Train vs CV Comparison**", unsafe_allow_html=True)
            cv_mean = S.cv_scores.mean()
            if S.problem_type == "Classification":
                train_score = accuracy_score(S.y_train, y_pred_train)
            else:
                train_score = r2_score(S.y_train, y_pred_train)

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(name="Train Score", x=["Train", "CV Mean", "Test"],
                                          y=[train_score, cv_mean,
                                             accuracy_score(S.y_test, y_pred_test) if S.problem_type == "Classification" else test_r2],
                                          marker_color=[PALETTE[0], PALETTE[1], PALETTE[2]]))
            dark_fig(fig_compare, 280)
            fig_compare.update_layout(title="Score Comparison: Train / CV / Test")
            st.plotly_chart(fig_compare, use_container_width=True)

    except Exception as e:
        st.markdown(f'<div class="danger-box">❌ Metrics error: {e}</div>', unsafe_allow_html=True)
        import traceback; st.code(traceback.format_exc())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            S.step = 7; st.rerun()
    with col2:
        if st.button("Proceed to Hyperparameter Tuning →"):
            S.step = 9; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 9:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <span class="section-badge">STEP 10</span>
      <span class="section-title">🎛️ Hyperparameter Tuning</span>
    </div>""", unsafe_allow_html=True)

    tuning_method = st.radio("Tuning strategy:", ["GridSearchCV", "RandomizedSearchCV"], horizontal=True, key="tune_method")
    cv_tune = st.number_input("Cross-validation folds:", min_value=2, max_value=10, value=5, key="tune_cv")

    # Define param grids
    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100], "max_iter": [200, 500, 1000]},
        "SVM (Classifier)": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear", "poly"], "gamma": ["scale", "auto"]},
        "Random Forest (Classifier)": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
        "KMeans (Clustering)": {"n_clusters": [2, 3, 4, 5, 6, 7], "n_init": [10, 20]},
        "Linear Regression": {"fit_intercept": [True, False]},
        "SVM (Regressor)": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "epsilon": [0.01, 0.1, 0.5]},
        "Random Forest (Regressor)": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
    }

    grid = param_grids.get(S.model_name, {})
    st.markdown(f"**Parameter grid for {S.model_name}:**")
    grid_df = pd.DataFrame([(k, str(v)) for k, v in grid.items()], columns=["Parameter", "Values"])
    st.dataframe(grid_df, use_container_width=True, hide_index=True)

    if st.button(f"🔍 Run {tuning_method}"):
        try:
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(S.X_train)
            X_te_s = scaler.transform(S.X_test)

            # Rebuild base model
            if S.model_name == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                base = LogisticRegression()
                scoring = "accuracy"
            elif S.model_name == "SVM (Classifier)":
                from sklearn.svm import SVC
                base = SVC(probability=True)
                scoring = "accuracy"
            elif S.model_name == "Random Forest (Classifier)":
                from sklearn.ensemble import RandomForestClassifier
                base = RandomForestClassifier(random_state=42)
                scoring = "accuracy"
            elif S.model_name == "KMeans (Clustering)":
                from sklearn.cluster import KMeans
                base = KMeans(random_state=42, n_init=10)
                scoring = None
            elif S.model_name == "Linear Regression":
                from sklearn.linear_model import LinearRegression
                base = LinearRegression()
                scoring = "r2"
            elif S.model_name == "SVM (Regressor)":
                from sklearn.svm import SVR
                base = SVR()
                scoring = "r2"
            elif S.model_name == "Random Forest (Regressor)":
                from sklearn.ensemble import RandomForestRegressor
                base = RandomForestRegressor(random_state=42)
                scoring = "r2"

            if scoring is None:
                st.markdown('<div class="info-box">ℹ️ KMeans does not support standard CV scoring. Trying with inertia.</div>', unsafe_allow_html=True)
                results_list = []
                for nc in grid.get("n_clusters", [3]):
                    from sklearn.cluster import KMeans as KM
                    km = KM(n_clusters=nc, n_init=10, random_state=42)
                    km.fit(X_tr_s)
                    results_list.append({"n_clusters": nc, "inertia": km.inertia_})
                results_df = pd.DataFrame(results_list)
                fig_k = px.line(results_df, x="n_clusters", y="inertia",
                                 markers=True, color_discrete_sequence=[PALETTE[0]])
                dark_fig(fig_k, 300)
                fig_k.update_layout(title="Elbow Curve — Inertia vs K")
                st.plotly_chart(fig_k, use_container_width=True)
            else:
                with st.spinner(f"Running {tuning_method}... this may take a moment"):
                    if tuning_method == "GridSearchCV":
                        search = GridSearchCV(base, grid, cv=int(cv_tune), scoring=scoring, n_jobs=-1, verbose=0)
                    else:
                        n_iter = min(20, 1)
                        for v in grid.values():
                            n_iter *= len(v)
                        n_iter = min(20, max(5, n_iter // 4))
                        search = RandomizedSearchCV(base, grid, n_iter=n_iter, cv=int(cv_tune),
                                                    scoring=scoring, n_jobs=-1, random_state=42, verbose=0)
                    search.fit(X_tr_s, S.y_train)

                best = search.best_estimator_
                S.best_model = best

                # Before vs After
                old_pred = S.model.predict(X_te_s)
                new_pred = best.predict(X_te_s)

                if S.problem_type == "Classification":
                    from sklearn.metrics import accuracy_score
                    old_sc = accuracy_score(S.y_test, old_pred)
                    new_sc = accuracy_score(S.y_test, new_pred)
                    metric_name = "Test Accuracy"
                else:
                    from sklearn.metrics import r2_score
                    old_sc = r2_score(S.y_test, old_pred)
                    new_sc = r2_score(S.y_test, new_pred)
                    metric_name = "Test R²"

                delta = new_sc - old_sc
                delta_color = "success-box" if delta >= 0 else "warn-box"
                delta_icon = "📈" if delta >= 0 else "📉"
                st.markdown(f'<div class="{delta_color}">{delta_icon} <b>Best params found:</b> {search.best_params_}</div>',
                             unsafe_allow_html=True)

                # Comparison chart
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(name="Before Tuning", x=[metric_name], y=[old_sc],
                                          marker_color=PALETTE[1]))
                fig_cmp.add_trace(go.Bar(name="After Tuning", x=[metric_name], y=[new_sc],
                                          marker_color=PALETTE[2]))
                dark_fig(fig_cmp, 300)
                fig_cmp.update_layout(title=f"Performance: Before vs After {tuning_method}",
                                       barmode="group")
                st.plotly_chart(fig_cmp, use_container_width=True)

                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-card purple">
                    <div class="metric-label">Before Tuning</div>
                    <div class="metric-value">{old_sc:.4f}</div>
                    <div class="metric-sub">{metric_name}</div>
                  </div>
                  <div class="metric-card green">
                    <div class="metric-label">After Tuning</div>
                    <div class="metric-value">{new_sc:.4f}</div>
                    <div class="metric-sub">{metric_name}</div>
                  </div>
                  <div class="metric-card {'green' if delta >= 0 else 'amber'}">
                    <div class="metric-label">Improvement</div>
                    <div class="metric-value">{delta:+.4f}</div>
                    <div class="metric-sub">{'Improved ✅' if delta >= 0 else 'No gain ⚠️'}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # CV results heatmap if GridSearch
                if tuning_method == "GridSearchCV" and len(grid) == 2:
                    cv_res = pd.DataFrame(search.cv_results_)
                    params_keys = list(grid.keys())
                    piv_col = f"param_{params_keys[1]}"
                    piv_row = f"param_{params_keys[0]}"
                    if piv_col in cv_res.columns and piv_row in cv_res.columns:
                        try:
                            pivot = cv_res.pivot_table(index=piv_row, columns=piv_col,
                                                        values="mean_test_score")
                            fig_pivot = px.imshow(pivot, text_auto=".3f",
                                                   color_continuous_scale="Viridis", aspect="auto")
                            dark_fig(fig_pivot, 350)
                            fig_pivot.update_layout(title="Grid Search Heatmap")
                            st.plotly_chart(fig_pivot, use_container_width=True)
                        except:
                            pass

        except Exception as e:
            st.markdown(f'<div class="danger-box">❌ Tuning error: {e}</div>', unsafe_allow_html=True)
            import traceback; st.code(traceback.format_exc())

    st.markdown("<hr class='step-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="success-box">
      🎉 <b>Pipeline Complete!</b> You have successfully run the end-to-end ML pipeline.
      Use the button below to restart with a new dataset.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Metrics"):
            S.step = 8; st.rerun()
    with col2:
        if st.button("🔄 Start New Pipeline"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
