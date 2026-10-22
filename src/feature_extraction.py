"""
Phishing URL Detection - Lexical Feature Extraction Module

This module extracts 18 lexical (text-based) features from URLs.
Lexical analysis inspects only the characters and structure of the URL string itself.
It does NOT make network requests, download content, or visit the target website,
keeping the detection process safe, fast, and offline-compatible.
"""

import sys
import os
import re
import ipaddress
from urllib.parse import urlparse
import pandas as pd
import tldextract

# Initialize tldextract offline without network calls
extractor = tldextract.TLDExtract(cache_dir=False)

# List of suspicious keywords frequently found in phishing campaigns
SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update",
    "bank", "confirm", "paypal"
]

# Common URL shortening domains used to hide real destinations
SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "goo.gl", "rebrand.ly", "t.co",
    "ow.ly", "is.gd", "buff.ly", "adf.ly", "bitly.com",
    "cutt.ly", "shorte.st", "tiny.cc"
}


def extract_features(url: str) -> dict:
    """
    Extracts 18 lexical numeric features from a single URL string.

    Parameters:
        url (str): The raw URL string to analyze.

    Returns:
        dict: A dictionary mapping feature names to their numeric values.
    """
    # 0. Clean and prepare URL string
    raw = str(url).strip()
    url_lower = raw.lower()

    # Prepend scheme if missing so urllib and tldextract can reliably parse
    has_scheme = "://" in raw
    prepended = raw if has_scheme else ("http://" + raw)

    try:
        parsed = urlparse(prepended)
        hostname = parsed.netloc or ""
        path = parsed.path or ""
    except Exception:
        # Fallback parsing for malformed URLs (e.g. irregular brackets or non-standard characters)
        without_scheme = raw.split("://", 1)[-1]
        parts = without_scheme.split("/", 1)
        hostname = parts[0].split("?")[0].split("#")[0]
        path = "/" + parts[1] if len(parts) > 1 else ""

    # 1. Structural Length Features
    url_length = len(raw)
    hostname_length = len(hostname)
    path_length = len(path)

    # 2. Character Counts
    num_dots = raw.count(".")
    num_hyphens = raw.count("-")
    num_underscores = raw.count("_")
    num_slashes = raw.count("/")
    num_digits = sum(c.isdigit() for c in raw)
    num_letters = sum(c.isalpha() for c in raw)

    # Special characters common in query strings and obfuscation (@ ? & = % # ~ + $ ! * ;)
    num_special_chars = len(re.findall(r"[@?&=%#~+$!*;]", raw))

    # 3. Security & Pattern Indicator Flags (Binary: 0 or 1)
    # '@' symbol often causes browsers to ignore text before it (userinfo disguise)
    has_at_symbol = 1 if "@" in raw else 0

    # Phishers sometimes use IP addresses (e.g. 192.168.1.1) instead of domain names
    host_only = hostname.split(":")[0]
    try:
        ipaddress.ip_address(host_only)
        has_ip_address = 1
    except ValueError:
        has_ip_address = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host_only) else 0

    # HTTPS usage check
    uses_https = 1 if url_lower.startswith("https://") else 0

    # Subdomain count (e.g. login.bank.com has 1 subdomain: 'login')
    try:
        ext = extractor(prepended)
        sub = ext.subdomain
        # Exclude standard 'www' prefix so legitimate websites are not penalized with extra subdomains
        subs = [s for s in sub.split(".") if s and s.lower() != "www"]
        num_subdomains = len(subs)
        registered_domain = f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        clean_host = host_only[4:] if host_only.startswith("www.") else host_only
        num_subdomains = max(0, clean_host.count(".") - 1)
        registered_domain = host_only.lower()

    # Check for presence of high-risk keywords (e.g., 'login', 'verify', 'paypal')
    has_suspicious_words = 1 if any(word in url_lower for word in SUSPICIOUS_WORDS) else 0

    # Check for known URL shortening services (e.g., bit.ly, tinyurl)
    has_shortening_service = 1 if (registered_domain in SHORTENING_SERVICES or host_only in SHORTENING_SERVICES) else 0

    # Double slash '//' appearing in the path or after the protocol (open redirect trick)
    has_double_slash_redirect = 1 if raw.rfind("//") > 7 else 0

    # 4. Ratios
    digit_ratio = round(num_digits / url_length, 4) if url_length > 0 else 0.0
    letter_ratio = round(num_letters / url_length, 4) if url_length > 0 else 0.0

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_slashes": num_slashes,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "has_at_symbol": has_at_symbol,
        "has_ip_address": has_ip_address,
        "uses_https": uses_https,
        "num_subdomains": num_subdomains,
        "has_suspicious_words": has_suspicious_words,
        "has_shortening_service": has_shortening_service,
        "has_double_slash_redirect": has_double_slash_redirect,
        "digit_ratio": digit_ratio,
        "letter_ratio": letter_ratio,
    }


def extract_features_from_dataframe(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    """
    Extracts features for all URLs in a DataFrame.
    """
    feature_records = [extract_features(url) for url in df[url_col]]
    return pd.DataFrame(feature_records)


def main():
    # If a URL argument is passed from CLI, extract and display its features
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        test_url = sys.argv[1]
        print(f"\nExtracting features for URL: '{test_url}'\n" + "-" * 50)
        feats = extract_features(test_url)
        for name, value in feats.items():
            print(f"  {name:28s} : {value}")
        print("-" * 50)
        return

    # Otherwise, process the entire cleaned dataset
    input_path = os.path.join("data", "processed", "cleaned_urls.csv")
    output_path = os.path.join("data", "processed", "features.csv")

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        print("Please ensure Step 3 EDA has completed and generated cleaned_urls.csv.")
        sys.exit(1)

    print(f"Loading cleaned dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} URLs. Extracting 18 lexical features...")

    features_df = extract_features_from_dataframe(df, url_col="url")

    # Include original URL and label
    features_df.insert(0, "url", df["url"])
    features_df["label"] = df["label"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"\nFeature extraction complete!")
    print(f"Saved {features_df.shape[0]:,} rows and {features_df.shape[1]} columns to {output_path}")
    print("\nFeature column list:")
    for col in features_df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
