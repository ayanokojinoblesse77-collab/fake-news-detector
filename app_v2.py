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

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Is It Real? — Fake News Checker",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit default chrome */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* Page max-width and font */
    .block-container {
        max-width: 720px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2rem 0 1.2rem 0;
    }
    .hero-icon { font-size: 3rem; line-height: 1; }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.4rem 0 0.2rem 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1rem;
        opacity: 0.55;
        margin: 0;
    }

    /* Input section */
    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.5;
        margin-bottom: 0.3rem;
    }

    /* Main verdict card */
    .verdict-card {
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 1.5rem 0 1rem 0;
    }
    .verdict-card.real {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1.5px solid #86efac;
    }
    .verdict-card.fake {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 1.5px solid #fca5a5;
    }
    .verdict-card.uncertain {
        background: linear-gradient(135deg, #fef9c3, #fef08a);
        border: 1.5px solid #fde047;
    }
    .verdict-emoji { font-size: 3rem; line-height: 1; margin-bottom: 0.5rem; }
    .verdict-label {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0.2rem 0 0.4rem 0;
        letter-spacing: -0.3px;
    }
    .verdict-label.real  { color: #15803d; }
    .verdict-label.fake  { color: #b91c1c; }
    .verdict-label.uncertain { color: #854d0e; }
    .verdict-sub {
        font-size: 0.95rem;
        opacity: 0.6;
        margin: 0;
    }

    /* Confidence bar section */
    .conf-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .conf-label { width: 60px; font-weight: 600; opacity: 0.7; }
    .conf-pct { width: 44px; text-align: right; font-weight: 700; }

    /* Signal pill tags */
    .signals-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.75rem 0;
    }
    .pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .pill-red  { background: #fee2e2; color: #b91c1c; }
    .pill-grn  { background: #dcfce7; color: #15803d; }
    .pill-ylw  { background: #fef9c3; color: #854d0e; }

    /* Info box */
    .info-box {
        border-radius: 10px;
        padding: 0.9rem 1rem;
        font-size: 0.88rem;
        line-height: 1.6;
        margin: 0.75rem 0;
        opacity: 0.85;
    }
    .info-box.tip  { background: #f0f9ff; border-left: 3px solid #38bdf8; }
    .info-box.warn { background: #fff7ed; border-left: 3px solid #fb923c; }
    .info-box.note { background: #f9fafb; border-left: 3px solid #d1d5db; }

    /* Divider */
    .soft-divider {
        border: none;
        border-top: 1px solid rgba(0,0,0,0.07);
        margin: 1.5rem 0;
    }

    /* Section heading */
    .section-heading {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.4;
        margin: 1.5rem 0 0.5rem 0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        font-size: 0.78rem;
        opacity: 0.35;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(0,0,0,0.07);
    }

    /* Streamlit overrides */
    div.stButton > button {
        border-radius: 10px;
        height: 3rem;
        font-size: 1rem;
        font-weight: 700;
    }
    .stTextInput input, .stTextArea textarea {
        border-radius: 10px;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base  = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(base)
    def find(kw, fb):
        return next((f for f in files if kw in f.lower() and f.endswith('.pkl')), fb)
    lr     = joblib.load(os.path.join(base, find('lr_fake_news', 'lr_fake_news_model.pkl')))
    rf     = joblib.load(os.path.join(base, find('rf_fake_news', 'rf_fake_news_model.pkl')))
    tfidf  = joblib.load(os.path.join(base, find('tfidf',        'tfidf_vectorizer.pkl')))
    scaler = joblib.load(os.path.join(base, find('scaler',       'scaler.pkl')))
    return lr, rf, tfidf, scaler

try:
    lr_model, rf_model, tfidf, scaler = load_models()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()


# ── ML backend — identical to app.py ──────────────────────────────────────
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

def get_top_features(clean_input, n=8):
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
    top_fake = [(feature_names[i], float(scores[i])) for i in sorted_nonzero[:n]]
    top_real = [(feature_names[i], float(scores[i])) for i in sorted_nonzero[-n:][::-1]]
    return top_real, top_fake


# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-icon">🔍</div>
    <div class="hero-title">Is It Real?</div>
    <p class="hero-sub">Paste any news article below and we'll check if it looks real or fake.</p>
</div>
""", unsafe_allow_html=True)

# Input
title_input = st.text_input(
    "Headline",
    placeholder="Paste the article headline here...",
    label_visibility="collapsed"
)
st.markdown('<p class="section-label">Headline</p>', unsafe_allow_html=True)

body_input = st.text_area(
    "Article body",
    height=180,
    placeholder="Paste the article text here. The more you paste, the more accurate the result.",
    label_visibility="collapsed"
)
st.markdown('<p class="section-label">Article body</p>', unsafe_allow_html=True)

check_btn = st.button("Check this article →", type="primary", use_container_width=True)

# Tip
st.markdown("""
<div class="info-box tip">
    💡 <strong>Tip:</strong> Paste as much of the article as you can. 
    Headlines alone work, but the full article body gives a much more reliable result.
</div>
""", unsafe_allow_html=True)


# ── Results ────────────────────────────────────────────────────────────────
if check_btn:
    combined_check = (title_input.strip() + ' ' + body_input.strip()).strip()

    if not combined_check:
        st.warning("Please paste a headline or article body first.")
        st.stop()

    if len(combined_check.split()) < 10:
        st.markdown("""
        <div class="info-box warn">
            ⚠️ That's quite short. Try pasting the full article for a reliable result.
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("Analysing..."):
        res = predict_article(title_input, body_input)

    # ── Determine overall verdict ─────────────────────────────────────────
    # Use LR as primary (higher accuracy). If models disagree, show uncertain.
    lr_conf_max = max(res['lr_real_pct'], res['lr_fake_pct'])
    models_agree = res['lr_label'] == res['rf_label']

    if not models_agree:
        verdict     = "UNCERTAIN"
        card_class  = "uncertain"
        verdict_css = "uncertain"
        emoji       = "🤔"
        verdict_msg = "Our models disagree on this one."
        verdict_sub = "This article has mixed signals. Read it carefully before sharing."
    elif res['lr_label'] == 'REAL':
        verdict     = "Looks Real"
        card_class  = "real"
        verdict_css = "real"
        emoji       = "✅"
        verdict_msg = f"{res['lr_real_pct']:.0f}% confidence it's real"
        verdict_sub = "This article shows patterns consistent with real news. Always verify with a trusted source."
    else:
        verdict     = "Likely Fake"
        card_class  = "fake"
        verdict_css = "fake"
        emoji       = "🚨"
        verdict_msg = f"{res['lr_fake_pct']:.0f}% confidence it's fake"
        verdict_sub = "This article shows patterns common in fake news. Be careful before sharing."

    # ── Main verdict card ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="verdict-card {card_class}">
        <div class="verdict-emoji">{emoji}</div>
        <div class="verdict-label {verdict_css}">{verdict}</div>
        <p class="verdict-sub">{verdict_msg}</p>
        <p class="verdict-sub" style="margin-top:0.3rem; font-size:0.82rem;">{verdict_sub}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Warning signals as plain-English pills ────────────────────────────
    signals = []
    if res['exclamation'] >= 2:
        signals.append(("pill-red",  f"🔴 {res['exclamation']} exclamation marks"))
    if res['caps_ratio'] > 5:
        signals.append(("pill-red",  f"🔴 High CAPS usage ({res['caps_ratio']:.0f}%)"))
    if res['question'] >= 3:
        signals.append(("pill-ylw",  f"🟡 {res['question']} rhetorical questions"))
    if res['word_count'] < 100:
        signals.append(("pill-ylw",  f"🟡 Short article ({res['word_count']} words)"))
    if res['word_count'] >= 300:
        signals.append(("pill-grn",  f"🟢 Detailed article ({res['word_count']} words)"))
    if res['exclamation'] == 0 and res['caps_ratio'] < 3:
        signals.append(("pill-grn",  "🟢 Calm, measured tone"))

    if signals:
        pills_html = '<div class="signals-wrap">' + ''.join(
            f'<span class="pill {cls}">{label}</span>' for cls, label in signals
        ) + '</div>'
        st.markdown('<p class="section-heading">What we noticed</p>', unsafe_allow_html=True)
        st.markdown(pills_html, unsafe_allow_html=True)

    # ── Disclaimer ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="info-box warn">
        ⚠️ <strong>Always double-check.</strong> This tool is a helpful guide, not a final verdict. 
        If something seems off, search for the story on a site you already trust.
    </div>
    """, unsafe_allow_html=True)

    # ── Details expander ─────────────────────────────────────────────────
    with st.expander("See full breakdown  ↓"):
        st.markdown("##### How each model voted")

        col_lr, col_rf = st.columns(2)
        with col_lr:
            icon = "✅" if res['lr_label'] == 'REAL' else "🚨"
            st.markdown(f"**Primary model** {icon}")
            st.caption("Logistic Regression · 93.71% accurate")
            real_c = res['lr_real_pct'] / 100
            fake_c = res['lr_fake_pct'] / 100
            st.markdown(f"Real: **{res['lr_real_pct']:.1f}%**")
            st.progress(real_c)
            st.markdown(f"Fake: **{res['lr_fake_pct']:.1f}%**")
            st.progress(fake_c)

        with col_rf:
            icon = "✅" if res['rf_label'] == 'REAL' else "🚨"
            st.markdown(f"**Second model** {icon}")
            st.caption("Random Forest · 92.80% accurate")
            st.markdown(f"Real: **{res['rf_real_pct']:.1f}%**")
            st.progress(res['rf_real_pct'] / 100)
            st.markdown(f"Fake: **{res['rf_fake_pct']:.1f}%**")
            st.progress(res['rf_fake_pct'] / 100)

        st.divider()

        st.markdown("##### Words that influenced the result")
        st.caption("These words from the article pushed the result one way or the other.")

        top_real, top_fake = get_top_features(res['clean'])

        if top_real or top_fake:
            all_features = sorted(top_real + top_fake, key=lambda x: x[1])
            labels  = [f[0] for f in all_features]
            scores  = [f[1] for f in all_features]
            colors  = ['#fca5a5' if s < 0 else '#86efac' for s in scores]

            fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.4)))
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            ax.barh(labels, scores, color=colors, height=0.55, edgecolor='none')
            ax.axvline(0, color='#9ca3af', linestyle='-', linewidth=0.8)
            ax.set_xlabel('Influence score', fontsize=9, color='#6b7280')
            ax.tick_params(colors='#6b7280', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#e5e7eb')
            ax.legend(
                handles=[
                    mpatches.Patch(color='#fca5a5', label='Points toward Fake'),
                    mpatches.Patch(color='#86efac', label='Points toward Real'),
                ],
                loc='lower right', fontsize=8,
                framealpha=0.5, edgecolor='#e5e7eb'
            )
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.caption("Not enough vocabulary matched to show word-level influence.")

    # ── How it works expander ────────────────────────────────────────────
    with st.expander("How does this work?  ↓"):
        st.markdown("""
Two AI models read your article and each give their own verdict. We then combine
their answers to show you a final result.

The models were trained on over 62,000 real and fake news articles and learned 
to spot patterns like:

- **Tone** — calm, factual writing vs emotional, urgent language  
- **Punctuation** — overuse of exclamation marks and all-caps  
- **Structure** — how long the article is, how the headline is written  
- **Word choices** — specific phrases that tend to appear in fake or real stories  

**Important:** No tool is perfect. This one is about 94% accurate on the articles 
it was trained on. For anything important, always check a source you trust.
        """)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Built by Group 7 · TechCrush AI/ML Bootcamp · Cohort 6<br>
    Trained on the WELFake dataset · For educational use
</div>
""", unsafe_allow_html=True)