"""
Phishing URL Detection - Streamlit Web Application

This web application allows users to paste any URL and instantly inspect
whether it is Legitimate or Phishing using our trained Machine Learning model.
It relies purely on lexical (text-based) features and NEVER visits or requests the target site.
"""

import os
import sys
import warnings
import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_extraction import extract_features
import tldextract

_tld_extractor = tldextract.TLDExtract(cache_dir=False)

# Top authoritative domains (Reputation Layer / Defense-in-Depth)
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "amazon.com",
    "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "netflix.com",
    "github.com", "microsoft.com", "apple.com", "yahoo.com", "bing.com",
    "stackoverflow.com", "wordpress.org", "adobe.com", "quora.com", "medium.com",
    "spotify.com", "dropbox.com", "salesforce.com", "paypal.com", "ebay.com",
    "cnn.com", "nytimes.com", "bbc.com", "theguardian.com", "espn.com",
    "imdb.com", "mozilla.org", "nih.gov", "cdc.gov", "mit.edu", "stanford.edu",
    "harvard.edu", "berkeley.edu", "w3schools.com", "geeksforgeeks.org", "gitlab.com",
    "whatsapp.com", "twitch.tv", "zoom.us", "vimeo.com", "pinterest.com"
}


# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS for Clean, Presentation-Ready UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard - Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom styling for clean, professional cybersecurity appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .result-card-legit {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .result-card-phish {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #6b7280;
        text-align: center;
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Loader (Cached for instant inference)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
    if not os.path.exists(model_path):
        return None, None, None

    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        feature_names = bundle.get("feature_names", None)
        metrics = bundle.get("metrics", {})
    else:
        model = bundle
        feature_names = None
        metrics = {}

    # Set CPU execution for low-overhead single prediction
    if hasattr(model, "set_params"):
        try:
            model.set_params(device="cpu")
        except Exception:
            pass

    return model, feature_names, metrics


model, feature_names, metrics = load_trained_model()


# -----------------------------------------------------------------------------
# Sidebar: Project Info & Quick Demo Presets
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ About PhishGuard")
    st.markdown(
        "A machine learning application that detects **Phishing URLs** using "
        "**18 lexical features** extracted directly from URL structure.\n\n"
        "Trained on **60,000 balanced URLs** combining classic web patterns and modern cloud-hosted phishing datasets."
    )
    st.markdown("---")

    st.markdown("#### 📊 Model Performance")
    if metrics:
        st.write(f"• **Model:** Tuned XGBoost (GPU)")
        st.write(f"• **Accuracy:** `{metrics.get('accuracy', 0.8918)*100:.2f}%`")
        st.write(f"• **Recall:** `{metrics.get('recall', 0.9038)*100:.2f}%`")
        st.write(f"• **ROC-AUC:** `{metrics.get('auc', 0.9622)*100:.2f}%`")
    else:
        st.write("• **Model:** Tuned XGBoost (89.18% Acc)")

    st.markdown("---")
    st.markdown("#### ⚡ Quick Demo URLs")
    st.caption("Click any demo URL to paste it into the search box:")

    demo_urls = [
        "Select a demo URL...",
        "https://www.google.com",
        "https://github.com",
        "google.com/search?q=machine+learning",
        "en.wikipedia.org/wiki/Computer_security",
        "amazon.com/dp/B08N5WRWNW/ref=nav_signin",
        "https://service-mitld.firebaseapp.com/",
        "http://att-103731-107123.weeblysite.com/",
        "http://paypal-verification-account-update.xyz/login.html",
        "http://192.168.1.1/admin/login.php",
        "http://secure-login.bankofamerica.update-verification.com/login.php"
    ]
    selected_demo = st.selectbox("Presets", demo_urls, index=0)


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ PhishGuard: Phishing URL Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Enter any website URL below to inspect whether it is <b>Legitimate</b> or <b>Phishing</b>. '
    'Analysis is performed strictly on URL text syntax — the site is never contacted or opened.'
    '</div>',
    unsafe_allow_html=True
)

# Populate input from preset if selected
default_input = "" if selected_demo == "Select a demo URL..." else selected_demo

url_input = st.text_input(
    label="Enter or paste URL:",
    value=default_input,
    placeholder="e.g. https://www.google.com or https://secure-bank.example.com/login",
    help="Paste the full or partial URL here. The site will NOT be visited."
)

col1, col2 = st.columns([1, 4])
with col1:
    check_btn = st.button("🔍 Check URL", type="primary", use_container_width=True)

if check_btn or (url_input and selected_demo != "Select a demo URL..."):
    if not url_input.strip():
        st.warning("⚠️ Please enter a URL before clicking Check.")
    elif model is None:
        st.error("❌ Trained model not found! Please run `python src/evaluate.py` first to generate models/best_model.joblib.")
    else:
        # Extract features (purely lexical, NO network requests)
        raw_url = url_input.strip()
        features = extract_features(raw_url)

        # Defense-in-Depth: Check Trusted High-Reputation Authority Domain
        prepended = raw_url if "://" in raw_url else ("http://" + raw_url)
        try:
            ext = _tld_extractor(prepended)
            reg_domain = f"{ext.domain}.{ext.suffix}".lower()
        except Exception:
            reg_domain = ""

        is_trusted_authority = (
            reg_domain in TRUSTED_DOMAINS
            and features["has_at_symbol"] == 0
            and features["has_ip_address"] == 0
            and features["has_double_slash_redirect"] == 0
            and features["has_shortening_service"] == 0
            and features["num_subdomains"] <= 1
        )

        if is_trusted_authority:
            prediction = 0
            prob_legit = 0.999
            prob_phish = 0.001
            detection_source = f"Verified Authority Domain ({reg_domain})"
        else:
            # Align with model training feature columns
            if feature_names:
                df_feat = pd.DataFrame([features])[feature_names]
            else:
                df_feat = pd.DataFrame([features])

            # Predict using trained Machine Learning Model
            prediction = int(model.predict(df_feat)[0])
            probabilities = model.predict_proba(df_feat)[0]

            prob_legit = float(probabilities[0])
            prob_phish = float(probabilities[1])
            detection_source = "Lexical Machine Learning Model"

        # Display Result
        if prediction == 1:
            confidence = prob_phish * 100.0
            st.markdown(f"""
            <div class="result-card-phish">
                <h3 style="color: #b91c1c; margin-top: 0;">🚨 PHISHING DETECTED</h3>
                <p style="font-size: 1.1rem; color: #7f1d1d; margin-bottom: 8px;">
                    This URL displays suspicious patterns commonly associated with phishing attacks.
                </p>
                <p style="font-size: 1.25rem; font-weight: bold; color: #991b1b; margin-bottom: 4px;">
                    Confidence: {confidence:.2f}% Phishing Probability
                </p>
                <p style="font-size: 0.9rem; color: #991b1b; margin-bottom: 0;">
                    <b>Detection Engine:</b> {detection_source}
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_phish, text=f"Threat Probability: {confidence:.2f}%")
        else:
            confidence = prob_legit * 100.0
            st.markdown(f"""
            <div class="result-card-legit">
                <h3 style="color: #047857; margin-top: 0;">✅ LEGITIMATE (SAFE)</h3>
                <p style="font-size: 1.1rem; color: #065f46; margin-bottom: 8px;">
                    This URL aligns with typical legitimate domain and syntax structures.
                </p>
                <p style="font-size: 1.25rem; font-weight: bold; color: #065f46; margin-bottom: 4px;">
                    Confidence: {confidence:.2f}% Legitimate Probability
                </p>
                <p style="font-size: 0.9rem; color: #047857; margin-bottom: 0;">
                    <b>Detection Engine:</b> {detection_source}
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_phish, text=f"Threat Probability: {prob_phish * 100:.2f}%")

        # -------------------------------------------------------------
        # Expandable Section: Extracted Lexical Features
        # -------------------------------------------------------------
        with st.expander("🔍 View Extracted Lexical Features & Breakdown", expanded=False):
            st.markdown("#### 18 Lexical URL Characteristics Analyzed:")

            # Organize features into a readable table
            feature_descriptions = {
                "url_length": ("Total character length of the entire URL", features["url_length"]),
                "hostname_length": ("Length of domain/hostname part", features["hostname_length"]),
                "path_length": ("Length of URL path portion", features["path_length"]),
                "num_dots": ("Count of '.' in the URL", features["num_dots"]),
                "num_hyphens": ("Count of '-' in the URL", features["num_hyphens"]),
                "num_underscores": ("Count of '_' in the URL", features["num_underscores"]),
                "num_slashes": ("Count of '/' in the URL", features["num_slashes"]),
                "num_digits": ("Total number of numeric digits (0-9)", features["num_digits"]),
                "num_special_chars": ("Count of special characters (@, ?, &, =, %, etc.)", features["num_special_chars"]),
                "has_at_symbol": ("Contains '@' (Userinfo disguise flag)", "Yes (1)" if features["has_at_symbol"] else "No (0)"),
                "has_ip_address": ("Domain is a raw IP address", "Yes (1)" if features["has_ip_address"] else "No (0)"),
                "uses_https": ("Scheme begins with https://", "Yes (1)" if features["uses_https"] else "No (0)"),
                "num_subdomains": ("Number of subdomains identified", features["num_subdomains"]),
                "has_suspicious_words": ("Contains keywords (login, verify, bank, paypal, etc.)", "Yes (1)" if features["has_suspicious_words"] else "No (0)"),
                "has_shortening_service": ("Uses link shortener (e.g., bit.ly, tinyurl)", "Yes (1)" if features["has_shortening_service"] else "No (0)"),
                "has_double_slash_redirect": ("Contains '//' after protocol (Open redirect)", "Yes (1)" if features["has_double_slash_redirect"] else "No (0)"),
                "digit_ratio": ("Proportion of numeric digits in URL", features["digit_ratio"]),
                "letter_ratio": ("Proportion of alphabetic letters in URL", features["letter_ratio"])
            }

            feat_table = []
            for feat_name, (desc, val) in feature_descriptions.items():
                feat_table.append({
                    "Feature": feat_name,
                    "Extracted Value": str(val),
                    "Security Meaning": desc
                })

            st.dataframe(pd.DataFrame(feat_table), use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# Educational Warning Line
# -------------------------------------------------------------
st.markdown(
    '<div class="disclaimer">'
    '⚠️ <b>Educational Project:</b> This is an educational project. Do not rely on it as your only protection.'
    '</div>',
    unsafe_allow_html=True
)
