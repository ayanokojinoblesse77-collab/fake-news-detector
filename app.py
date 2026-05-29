import streamlit as st
import joblib
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
from scipy.sparse import csr_matrix, hstack

# ── Download NLTK data safely on environment bootup ────────────────────
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Page Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS Interface Styling ────────────────────────────────────────
st.markdown("""
<style>
    .stTextArea textarea { font-size: 15px; line-height: 1.6; }
    .verdict-real {
        background: linear-gradient(135deg, #1a7f37, #2ea043);
        color: white; padding: 24px 32px; border-radius: 12px;
        text-align: center; font-size: 26px; font-weight: 700;
        margin: 12px 0; box-shadow: 0 4px 12px rgba(46,160,67,0.3);
    }
    .verdict-fake {
        background: linear-gradient(135deg, #b91c1c, #dc2626);
        color: white; padding: 24px 32px; border-radius: 12px;
        text-align: center; font-size: 26px; font-weight: 700;
        margin: 12px 0; box-shadow: 0 4px 12px rgba(220,38,38,0.3);
    }
    .agree-banner {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-radius: 8px; padding: 12px 16px; margin: 12px 0;
        color: #1d4ed8; font-weight: 500;
    }
    .disagree-banner {
        background: #fff7ed; border: 1px solid #fed7aa;
        border-radius: 8px; padding: 12px 16px; margin: 12px 0;
        color: #c2410c; font-weight: 500;
    }
    .warning-banner {
        background: #fefce8; border: 1px solid #fde68a;
        border-radius: 8px; padding: 12px 16px; margin: 12px 0;
        color: #92400e; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Saved Artifacts (With Auto-Case Insensitivity Correction) ─────
@st.cache_resource
def load_models():
    # Anchor to the directory where app.py lives — fixes Streamlit Cloud path resolution
    base  = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(base)

    # Prevents FileNotFoundError caused by strict Linux case-sensitive filesystems
    lr_file     = next((f for f in files if "lr_fake_news" in f.lower() and f.endswith(".pkl")), "lr_fake_news_model.pkl")
    rf_file     = next((f for f in files if "rf_fake_news" in f.lower() and f.endswith(".pkl")), "rf_fake_news_model.pkl")
    tfidf_file  = next((f for f in files if "tfidf" in f.lower() and f.endswith(".pkl")), "tfidf_vectorizer.pkl")
    scaler_file = next((f for f in files if "scaler" in f.lower() and f.endswith(".pkl")), "scaler.pkl")

    lr     = joblib.load(os.path.join(base, lr_file))
    rf     = joblib.load(os.path.join(base, rf_file))
    tfidf  = joblib.load(os.path.join(base, tfidf_file))
    scaler = joblib.load(os.path.join(base, scaler_file))
    return lr, rf, tfidf, scaler

try:
    lr_model, rf_model, tfidf, scaler = load_models()
except Exception as e:
    st.error(f"Error loading local model files: {e}")
    st.info("Ensure your compressed model files are saved inside your tracked repository directory.")
    st.stop()


# ── Text Cleaning Pipeline ─────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    text  = text.lower()
    text  = re.sub(r'http\S+|www\S+', '', text)
    text  = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)


# ── Full Prediction Pipeline (Preserving Your Exact Engineering Math) ──
def predict_article(title_input, body_input):
    raw_combined_text = str(title_input) + " " + str(body_input)

    # Re-engineering the precise structural features used during model training
    word_count      = float(len(str(body_input).split()))
    char_count      = float(len(str(body_input)))
    title_length    = float(len(str(title_input).split()))
    exclamation_cnt = float(raw_combined_text.count('!'))
    question_cnt    = float(raw_combined_text.count('?'))
    uppercase_ratio = float(sum(1 for c in raw_combined_text if c.isupper()) / (len(raw_combined_text) + 1))

    numerical_features_single = np.array([[
        word_count, char_count, title_length,
        exclamation_cnt, question_cnt, uppercase_ratio
    ]])

    # Normalize structural features using your saved StandardScaler matrix
    scaled_numerical_features = scaler.transform(numerical_features_single)
    num_sparse = csr_matrix(scaled_numerical_features)

    # Vectorize cleaned textual array
    clean_combined_text = clean_text(raw_combined_text)
    vec_tfidf = tfidf.transform([clean_combined_text])

    # Stack sparse text matrices and dense matrices side-by-side
    vec_final = hstack([vec_tfidf, num_sparse])

    # Generate classifications and probability arrays
    lr_pred_s = lr_model.predict(vec_final)[0]
    lr_conf   = lr_model.predict_proba(vec_final)[0]
    rf_pred_s = rf_model.predict(vec_final)[0]
    rf_conf   = rf_model.predict_proba(vec_final)[0]

    return {
        'lr_label'    : 'FAKE' if lr_pred_s == 0 else 'REAL',
        'lr_real_pct' : float(lr_conf[1]) * 100,
        'lr_fake_pct' : float(lr_conf[0]) * 100,
        'rf_label'    : 'FAKE' if rf_pred_s == 0 else 'REAL',
        'rf_real_pct' : float(rf_conf[1]) * 100,
        'rf_fake_pct' : float(rf_conf[0]) * 100,
        'word_count'  : int(word_count),
        'char_count'  : int(char_count),
        'exclamation' : int(exclamation_cnt),
        'question'    : int(question_cnt),
        'caps_ratio'  : float(uppercase_ratio) * 100,
        'clean'       : clean_combined_text,
    }


# ── Feature Influence Metric Extractor ─────────────────────────────────
def get_top_features(clean_input, n=10):
    vec           = tfidf.transform([clean_input])
    feature_names = tfidf.get_feature_names_out()
    coefficients  = lr_model.coef_[0][:len(feature_names)]
    tfidf_array   = vec.toarray()[0]
    scores        = tfidf_array * coefficients
    nonzero_idx   = np.where(scores != 0)[0]

    if len(nonzero_idx) == 0:
        return [], []

    sorted_idx     = np.argsort(scores[nonzero_idx])
    sorted_nonzero = nonzero_idx[sorted_idx]
    top_fake_idx   = sorted_nonzero[:n]
    top_real_idx   = sorted_nonzero[-n:][::-1]

    top_fake = [(feature_names[i], float(scores[i])) for i in top_fake_idx]
    top_real = [(feature_names[i], float(scores[i])) for i in top_real_idx]
    return top_real, top_fake


# ── SIDEBAR INTERFACE LAYOUT ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📰 Fake News Detector")
    st.markdown("**TechCrush AI/ML Bootcamp**")
    st.markdown("Cohort 6 — Group 7")
    st.divider()

    st.markdown("### About Engine")
    st.markdown("""
This web application evaluates copy-pasted textual records using machine learning pipelines trained on the **WELFake dataset** (62,336 unique records).
    """)

    st.divider()
    st.markdown("### Model Validation Baselines")
    perf_df = pd.DataFrame({
        "Metric" : ["Accuracy", "ROC-AUC", "F1 (Fake)", "F1 (Real)"],
        "LR"     : ["95.25%", "0.9897", "0.96", "0.95"],
        "RF"     : ["95.28%", "0.9911", "0.96", "0.95"],
    })
    st.dataframe(perf_df.set_index("Metric"), use_container_width=True)


# ── MAIN APPLICATION PANEL ─────────────────────────────────────────────
st.title("📰 Fake News Detection Dashboard")
st.markdown("*Ensemble Learning Pipeline — Paste an article below to perform an analysis*")
st.divider()

col_input, col_tips = st.columns([3, 1])

with col_input:
    title_input = st.text_input("Article Headline", placeholder="Paste news headline here...")
    body_input = st.text_area("Article Body", height=220, placeholder="Paste article text character blocks here...")

with col_tips:
    st.markdown("#### Guidance")
    st.info("Input blocks containing comprehensive body paragraphs allow both classifiers to map structural patterns with optimal fidelity.")

analyse_btn = st.button("🔍 Run Machine Learning Pipeline Analysis", type="primary", use_container_width=True)

if analyse_btn:
    combined_input = (title_input.strip() + ' ' + body_input.strip()).strip()

    if not combined_input:
        st.error("Missing classification target. Provide text input strings before triggering execution.")
    elif len(combined_input.split()) < 10:
        st.markdown('<div class="warning-banner">⚠️ Short text inputs degrade classification consistency. Provide complete records for production evaluations.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Executing sparse matrix vectorization and structural tensor transforms..."):
            res = predict_article(title_input, body_input)

        st.divider()
        st.markdown("## Prediction Matrix Analysis")

        if res['lr_label'] == res['rf_label']:
            emoji = "✅" if res['lr_label'] == "REAL" else "🚨"
            st.markdown(f'<div class="agree-banner">{emoji} Model Convergence: Both models categorize this payload stream as <strong>{res["lr_label"]}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="disagree-banner">⚠️ Model Divergence: Classifiers generated conflicting targets. Cross-examine metric distributions below.</div>', unsafe_allow_html=True)

        col_lr, col_rf = st.columns(2)
        with col_lr:
            css = "verdict-real" if res['lr_label'] == "REAL" else "verdict-fake"
            st.markdown(f'<div class="{css}">Logistic Regression<br>{res["lr_label"]}</div>', unsafe_allow_html=True)
            st.markdown(f"Real Confidence: {res['lr_real_pct']:.1f}%")
            st.progress(res['lr_real_pct'] / 100)
            st.markdown(f"Fake Confidence: {res['lr_fake_pct']:.1f}%")
            st.progress(res['lr_fake_pct'] / 100)

        with col_rf:
            css = "verdict-real" if res['rf_label'] == "REAL" else "verdict-fake"
            st.markdown(f'<div class="{css}">Random Forest<br>{res["rf_label"]}</div>', unsafe_allow_html=True)
            st.markdown(f"Real Confidence: {res['rf_real_pct']:.1f}%")
            st.progress(res['rf_real_pct'] / 100)
            st.markdown(f"Fake Confidence: {res['rf_fake_pct']:.1f}%")
            st.progress(res['rf_fake_pct'] / 100)

        st.divider()
        st.markdown("## Extracted Structural Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Word Count", f"{res['word_count']:,}")
        c2.metric("Character Count", f"{res['char_count']:,}")
        c3.metric("Exclamations (!)", res['exclamation'])
        c4.metric("Questions (?)", res['question'])
        c5.metric("CAPS Density", f"{res['caps_ratio']:.1f}%")

        st.divider()
        st.markdown("## Linguistic Component Influence Graph")
        st.markdown("*Vocabulary features pulling Logistic Regression toward classifications:*")
        top_real, top_fake = get_top_features(res['clean'])

        if top_real or top_fake:
            all_features = (top_real + top_fake)
            all_features.sort(key=lambda x: x[1])
            labels = [f[0] for f in all_features]
            scores = [f[1] for f in all_features]
            colors = ['#ef4444' if s < 0 else '#2ea043' for s in scores]

            fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
            ax.barh(labels, scores, color=colors, height=0.6)
            ax.axvline(0, color='#64748b', linestyle='--')
            ax.set_xlabel('Mathematical Weight Vector')
            fake_patch = mpatches.Patch(color='#ef4444', label='Pushes toward FAKE')
            real_patch = mpatches.Patch(color='#2ea043', label='Pushes toward REAL')
            ax.legend(handles=[fake_patch, real_patch], loc='lower right', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

st.divider()
st.caption("TechCrush Bootcamp Node · Cohort 6 · Group 7 · Production Pipeline Stable Core")
