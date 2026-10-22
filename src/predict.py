"""
Phishing URL Detection - Prediction Script

This script implements Step 8:
1. Loads the trained best model bundle from models/best_model.joblib
2. Accepts a URL as a command-line argument: python src/predict.py "<url>"
3. Extracts lexical features, performs inference, and outputs:
   - Prediction: Legitimate or Phishing
   - Confidence Percentage
   - Key Risk Signals
4. If run without arguments, runs a built-in benchmark test across 6 diverse URLs.
"""

import sys
import os
import warnings
import joblib
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to sys.path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_extraction import extract_features


import tldextract

# Initialize offline extractor
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


def load_model(model_path="models/best_model.joblib"):
    """
    Loads the saved model bundle.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found.\n"
            "Please make sure Step 7 (src/evaluate.py) has been executed."
        )
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        feature_names = bundle.get("feature_names", None)
    else:
        model = bundle
        feature_names = None

    # Use CPU for rapid, low-overhead single-row inference
    if hasattr(model, "set_params"):
        try:
            model.set_params(device="cpu")
        except Exception:
            pass

    return model, feature_names


def predict_url(url: str, model, feature_names=None):
    """
    Predicts whether a single URL is Legitimate or Phishing.

    Combines:
    1. Defense-in-Depth Domain Authority Verification (Trusted Whitelist)
    2. Machine Learning Lexical Classifier (XGBoost / Random Forest)

    Returns:
        dict: {
            'url': str,
            'prediction': 'Legitimate' or 'Phishing',
            'label': 0 or 1,
            'confidence': float (percentage),
            'prob_legitimate': float,
            'prob_phishing': float,
            'detection_source': str,
            'features': dict
        }
    """
    feats = extract_features(url)

    # 1. Defense-in-Depth Reputation Check:
    # Check if the domain is an authoritative trusted domain WITHOUT suspicious phishing tricks
    raw_clean = str(url).strip()
    prepended = raw_clean if "://" in raw_clean else ("http://" + raw_clean)
    try:
        ext = _tld_extractor(prepended)
        registered_domain = f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        registered_domain = ""

    # Domain is trusted only if it does NOT contain deceptive obfuscation
    is_trusted_authority = (
        registered_domain in TRUSTED_DOMAINS
        and feats["has_at_symbol"] == 0
        and feats["has_ip_address"] == 0
        and feats["has_double_slash_redirect"] == 0
        and feats["has_shortening_service"] == 0
        and feats["num_subdomains"] <= 1
    )

    if is_trusted_authority:
        pred = 0
        prob_legit = 0.999
        prob_phish = 0.001
        label_str = "Legitimate"
        confidence = 99.90
        detection_source = f"Verified Authority Domain ({registered_domain})"
    else:
        if feature_names:
            df_feat = pd.DataFrame([feats])[feature_names]
        else:
            df_feat = pd.DataFrame([feats])

        pred = int(model.predict(df_feat)[0])
        probs = model.predict_proba(df_feat)[0]

        prob_legit = float(probs[0])
        prob_phish = float(probs[1])

        label_str = "Phishing" if pred == 1 else "Legitimate"
        confidence = (prob_phish if pred == 1 else prob_legit) * 100.0
        detection_source = "Lexical Machine Learning Model"

    return {
        "url": url,
        "prediction": label_str,
        "label": pred,
        "confidence": confidence,
        "prob_legitimate": prob_legit,
        "prob_phishing": prob_phish,
        "detection_source": detection_source,
        "features": feats
    }


def print_prediction_card(result: dict):
    """
    Prints a clean, user-friendly terminal report for the prediction.
    """
    url = result["url"]
    pred = result["prediction"]
    conf = result["confidence"]
    feats = result["features"]

    print("\n" + "=" * 65)
    print(" URL ANALYSIS REPORT")
    print("=" * 65)
    print(f" Analyzed URL : {url}")
    print(f" Result       : [{pred.upper()}]")
    print(f" Confidence   : {conf:.2f}%")
    print(f" Detection    : {result.get('detection_source', 'Machine Learning')}")
    print("-" * 65)
    print(" Key Lexical Indicators:")
    print(f"  - Total Length           : {feats['url_length']} characters")
    print(f"  - Subdomain Count        : {feats['num_subdomains']}")
    print(f"  - Uses HTTPS             : {'Yes' if feats['uses_https'] else 'No'}")
    print(f"  - Uses Raw IP Address    : {'Yes (Suspicious)' if feats['has_ip_address'] else 'No'}")
    print(f"  - Suspicious Keywords    : {'Yes (Found high-risk tokens)' if feats['has_suspicious_words'] else 'None'}")
    print(f"  - URL Shortening Service : {'Yes (e.g. bit.ly)' if feats['has_shortening_service'] else 'No'}")
    print(f"  - Open Redirect '//'     : {'Yes (Detected after protocol)' if feats['has_double_slash_redirect'] else 'No'}")
    print("=" * 65 + "\n")


def run_benchmark_tests(model, feature_names):
    """
    Evaluates a diverse mix of 6 realistic legitimate and phishing URLs.
    """
    test_urls = [
        "google.com/search?q=cybersecurity+threats",
        "en.wikipedia.org/wiki/Computer_security",
        "amazon.com/dp/B08N5WRWNW/ref=nav_signin",
        "http://paypal-verification-account-update.xyz/login.html",
        "http://192.168.1.1/admin/login.php",
        "http://secure-login.bankofamerica.update-verification.com/login.php"
    ]

    print("\n" + "=" * 80)
    print(" BENCHMARK TEST SUITE (6 Sample URLs)")
    print("=" * 80)

    for url in test_urls:
        res = predict_url(url, model, feature_names)
        status_tag = "[PHISHING]  " if res["prediction"] == "Phishing" else "[LEGITIMATE]"
        print(f" {status_tag} {res['confidence']:6.2f}% | {url}")

    print("=" * 80)
    print("To test any custom URL, run:")
    print('  python src/predict.py "<your-url-here>"\n')


def main():
    try:
        model, feature_names = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target_url = sys.argv[1]
        result = predict_url(target_url, model, feature_names)
        print_prediction_card(result)
    else:
        run_benchmark_tests(model, feature_names)


if __name__ == "__main__":
    main()
