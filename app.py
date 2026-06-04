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

# ── NLTK ───────────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────
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


# ── Load model artifacts ───────────────────────────────────────────────────
# Anchored to app.py directory — required for Streamlit Cloud path resolution.
# Case-insensitive matching prevents FileNotFoundError on Linux filesystems.
@st.cache_resource
def load_models():
    base  = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(base)

    def find(keyword, fallback):
        return next(
            (f for f in files if keyword in f.lower() and f.endswith('.pkl')),
            fallback
        )

    lr     = joblib.load(os.path.join(base, find('lr_fake_news', 'lr_fake_news_model.pkl')))
    rf     = joblib.load(os.path.join(base, find('rf_fake_news', 'rf_fake_news_model.pkl')))
    tfidf  = joblib.load(os.path.join(base, find('tfidf',        'tfidf_vectorizer.pkl')))
    scaler = joblib.load(os.path.join(base, find('scaler',       'scaler.pkl')))
    return lr, rf, tfidf, scaler

try:
    lr_model, rf_model, tfidf, scaler = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.info(
        "Ensure all four .pkl files are in the same directory as app.py:\n"
        "lr_fake_news_model.pkl · rf_fake_news_model.pkl · "
        "tfidf_vectorizer.pkl · scaler.pkl"
    )
    st.stop()


# ── Text cleaning ──────────────────────────────────────────────────────────
# MUST be identical to notebook Section 4 clean_text().
# SOURCE_BIAS_PATTERN strips news outlet names to prevent the model
# classifying by source identity rather than writing style.
# Political proper nouns are intentionally NOT stripped — they appear
# in both fake and real articles and stripping them destroys context
# for short inputs.
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

SOURCE_BIAS_PATTERN = (
    r'\breuters\b|\breuter\b|\bbreitbart\b|\bnytimes\b|\bnyt\b'
    r'|\bwashington post\b|\bwashington\b|\bhuffpost\b|\bhuffington\b'
    r'|\bpolitico\b|\bfoxnews\b|\bfox news\b|\bcnn\b|\bmsnbc\b'
    r'|\bassociated press\b|\b\bap\b|\bguardian\b|\bbbc\b'
    r'|\bdaily mail\b|\bdailymail\b|\bbuzzfeed\b|\binfowars\b'
    r'|\bnewsweek\b|\btime magazine\b|\bthe hill\b|\baxios\b'
)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(SOURCE_BIAS_PATTERN, '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)


# ── Prediction pipeline ────────────────────────────────────────────────────
# Feature engineering mirrors notebook Section 4 exactly:
#   word_count, char_count, exclamation_cnt, question_cnt, uppercase_ratio
#     → body text only  (matching df['text'] in training)
#   title_length
#     → title only  (matching df['title'] in training)
#   TF-IDF
#     → title + body combined  (matching df['combined'] in training)
#   Stack order: [tfidf | numerical]  — must match hstack order in training
#
# Named DataFrame is passed to scaler.transform() to prevent the sklearn
# UserWarning seen in the notebook. The scaler was fitted on X_train[num_cols]
# (a named DataFrame), so inference must supply the same column names.
_NUM_COLS = ['word_count', 'char_count', 'title_length',
             'exclamation_cnt', 'question_cnt', 'uppercase_ratio']

def predict_article(title_input, body_input):
    body  = str(body_input)
    title = str(title_input)

    word_count      = float(len(body.split()))
    char_count      = float(len(body))
    title_length    = float(len(title.split()))
    exclamation_cnt = float(body.count('!'))
    question_cnt    = float(body.count('?'))
    uppercase_ratio = float(sum(1 for c in body if c.isupper()) / (len(body) + 1))

    numerical_df = pd.DataFrame([[
        word_count, char_count, title_length,
        exclamation_cnt, question_cnt, uppercase_ratio
    ]], columns=_NUM_COLS)

    num_sparse     = csr_matrix(scaler.transform(numerical_df))
    combined_clean = clean_text(title + ' ' + body)
    vec_tfidf      = tfidf.transform([combined_clean])
    vec_final      = hstack([vec_tfidf, num_sparse])

    lr_pred = lr_model.predict(vec_final)[0]
    lr_conf = lr_model.predict_proba(vec_final)[0]
    rf_pred = rf_model.predict(vec_final)[0]
    rf_conf = rf_model.predict_proba(vec_final)[0]

    return {
        'lr_label'    : 'FAKE' if lr_pred == 0 else 'REAL',
        'lr_real_pct' : float(lr_conf[1]) * 100,
        'lr_fake_pct' : float(lr_conf[0]) * 100,
        'rf_label'    : 'FAKE' if rf_pred == 0 else 'REAL',
        'rf_real_pct' : float(rf_conf[1]) * 100,
        'rf_fake_pct' : float(rf_conf[0]) * 100,
        'word_count'  : int(word_count),
        'char_count'  : int(char_count),
        'exclamation' : int(exclamation_cnt),
        'question'    : int(question_cnt),
        'caps_ratio'  : float(uppercase_ratio) * 100,
        'clean'       : combined_clean,
    }


# ── Feature influence ──────────────────────────────────────────────────────
# lr_model.coef_[0] layout: [tfidf weights (8000) | numerical weights (6)]
# We slice the tfidf portion only to plot vocabulary influence.
def get_top_features(clean_input, n=10):
    vec           = tfidf.transform([clean_input])
    feature_names = tfidf.get_feature_names_out()
    n_tfidf       = len(feature_names)
    tfidf_coefs   = lr_model.coef_[0][:n_tfidf]
    tfidf_array   = vec.toarray()[0]
    scores        = tfidf_array * tfidf_coefs
    nonzero_idx   = np.where(scores != 0)[0]

    if len(nonzero_idx) == 0:
        return [], []

    sorted_idx     = np.argsort(scores[nonzero_idx])
    sorted_nonzero = nonzero_idx[sorted_idx]
    top_fake       = [(feature_names[i], float(scores[i])) for i in sorted_nonzero[:n]]
    top_real       = [(feature_names[i], float(scores[i])) for i in sorted_nonzero[-n:][::-1]]
    return top_real, top_fake


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📰 Fake News Detector")
    st.markdown("**TechCrush AI/ML Bootcamp**")
    st.markdown("Cohort 6 — Group 7")
    st.divider()

    st.markdown("### About")
    st.markdown(
        "Classifies news articles as **Fake** or **Real** using an ensemble "
        "of Logistic Regression and Random Forest models trained on the "
        "**WELFake dataset** — 62,336 articles across four independent sources."
    )
    st.divider()

    st.markdown("### Model Performance")
    st.caption("Verified on held-out test set · 12,468 articles · Leakage-free + debiased pipeline")
    perf_df = pd.DataFrame({
        "Metric"  : ["Accuracy", "ROC-AUC", "F1 (Fake)", "F1 (Real)", "False Positives"],
        "LR ⭐"   : ["93.71%",   "0.9821",  "0.93",      "0.94",      "382"],
        "RF"      : ["92.80%",   "0.9810",  "0.92",      "0.94",      "419"],
    })
    st.dataframe(perf_df.set_index("Metric"), use_container_width=True)
    st.caption("⭐ LR recommended — 8.8% fewer fake articles escape detection (382 vs 419 FP)")

    st.divider()
    st.markdown("### Generalization Check (5-Fold CV)")
    cv_df = pd.DataFrame({
        "Model" : ["LR", "RF"],
        "CV AUC": ["0.9799 ± 0.0012", "0.9795 ± 0.0011"],
        "Test AUC" : ["0.9821", "0.9810"],
        "Gap"   : ["0.0022 ✅", "0.0015 ✅"],
    })
    st.dataframe(cv_df.set_index("Model"), use_container_width=True)
    st.caption("Both gaps < 0.01 — models generalize correctly, not overfit.")

    st.divider()
    st.markdown("### How to Use")
    st.markdown(
        "1. Paste the article **headline** into the first field\n"
        "2. Paste the **article body** into the second field\n"
        "3. Click **Run Analysis**\n"
        "4. Review verdicts, confidence scores, and feature influence chart"
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ════════════════════════════════════════════════════════════════════════════
st.title("📰 Fake News Detection Dashboard")
st.markdown(
    "*Logistic Regression + Random Forest · WELFake Dataset · "
    "TechCrush AI/ML Bootcamp — Cohort 6, Group 7*"
)
st.divider()

col_input, col_tips = st.columns([3, 1])

with col_input:
    title_input = st.text_input(
        "Article Headline",
        placeholder="Paste the article headline here..."
    )
    body_input = st.text_area(
        "Article Body",
        height=220,
        placeholder="Paste the full article body text here..."
    )

with col_tips:
    st.markdown("#### Tips")
    st.info(
        "**More text = better results.**  \n"
        "Paste the full article body for highest classification accuracy.  \n"
        "Headlines alone produce lower-confidence outputs."
    )
    st.warning(
        "⚠️ Educational tool only.  \n"
        "Do not use as the sole basis for editorial or publication decisions."
    )

run_btn = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

if run_btn:
    combined_check = (title_input.strip() + ' ' + body_input.strip()).strip()

    if not combined_check:
        st.error("No input provided. Paste a headline and/or article body before running.")

    elif len(combined_check.split()) < 10:
        st.markdown(
            '<div class="warning-banner">'
            '⚠️ Input is very short (&lt;10 words). Results may be unreliable. '
            'Provide a complete article body for accurate classification.'
            '</div>',
            unsafe_allow_html=True
        )

    else:
        with st.spinner("Vectorizing text and computing predictions..."):
            res = predict_article(title_input, body_input)

        st.divider()
        st.markdown("## Results")

        # ── Convergence / divergence banner ───────────────────────────────
        if res['lr_label'] == res['rf_label']:
            emoji = "✅" if res['lr_label'] == "REAL" else "🚨"
            st.markdown(
                f'<div class="agree-banner">'
                f'{emoji} Both models agree: <strong>{res["lr_label"]}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="disagree-banner">'
                '⚠️ Models disagree. Review individual confidence scores below.  '
                'Logistic Regression (⭐) is the higher-accuracy model on this dataset.'
                '</div>',
                unsafe_allow_html=True
            )

        # ── Verdict cards ─────────────────────────────────────────────────
        st.markdown("### Model Verdicts")
        col_lr, col_rf = st.columns(2)

        with col_lr:
            css = "verdict-real" if res['lr_label'] == "REAL" else "verdict-fake"
            st.markdown(
                f'<div class="{css}">Logistic Regression ⭐<br>{res["lr_label"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**Real confidence:** {res['lr_real_pct']:.1f}%")
            st.progress(res['lr_real_pct'] / 100)
            st.markdown(f"**Fake confidence:** {res['lr_fake_pct']:.1f}%")
            st.progress(res['lr_fake_pct'] / 100)
            st.caption("Accuracy: 93.71% · ROC-AUC: 0.9821 · FP: 382")

        with col_rf:
            css = "verdict-real" if res['rf_label'] == "REAL" else "verdict-fake"
            st.markdown(
                f'<div class="{css}">Random Forest<br>{res["rf_label"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**Real confidence:** {res['rf_real_pct']:.1f}%")
            st.progress(res['rf_real_pct'] / 100)
            st.markdown(f"**Fake confidence:** {res['rf_fake_pct']:.1f}%")
            st.progress(res['rf_fake_pct'] / 100)
            st.caption("Accuracy: 92.80% · ROC-AUC: 0.9810 · FP: 419")

        # ── Structural metrics ────────────────────────────────────────────
        st.divider()
        st.markdown("### Article Structural Metrics")
        st.caption("Extracted from body text — used as stylometric features alongside TF-IDF")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Word Count",       f"{res['word_count']:,}")
        c2.metric("Character Count",  f"{res['char_count']:,}")
        c3.metric("Exclamations (!)", res['exclamation'])
        c4.metric("Questions (?)",    res['question'])
        c5.metric("CAPS Density",     f"{res['caps_ratio']:.1f}%")

        # ── Linguistic influence chart ────────────────────────────────────
        st.divider()
        st.markdown("### Linguistic Feature Influence")
        st.markdown(
            "*Top vocabulary terms from this article and their influence on "
            "the Logistic Regression verdict.  \n"
            "🟢 Green = pushes toward REAL. 🔴 Red = pushes toward FAKE.*"
        )

        top_real, top_fake = get_top_features(res['clean'])

        if top_real or top_fake:
            all_features = sorted(top_real + top_fake, key=lambda x: x[1])
            labels  = [f[0] for f in all_features]
            scores  = [f[1] for f in all_features]
            colors  = ['#ef4444' if s < 0 else '#2ea043' for s in scores]

            fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
            ax.barh(labels, scores, color=colors, height=0.6)
            ax.axvline(0, color='#64748b', linestyle='--', linewidth=1)
            ax.set_xlabel('LR Coefficient × TF-IDF Weight', fontsize=10)
            ax.legend(
                handles=[
                    mpatches.Patch(color='#ef4444', label='Pushes toward FAKE'),
                    mpatches.Patch(color='#2ea043', label='Pushes toward REAL'),
                ],
                loc='lower right', fontsize=9
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info(
                "No influential vocabulary terms found. "
                "This can occur with very short or heavily numerical input text."
            )

st.divider()
st.caption(
    "TechCrush AI/ML Bootcamp · Cohort 6 · Group 7 · "
    "WELFake Dataset · Logistic Regression + Random Forest · "
    "Leakage-free pipeline · Outlet source debiasing applied"
)
