# PhishGuard: Phishing URL Detection using Machine Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![XGBoost GPU](https://img.shields.io/badge/Accelerated-NVIDIA%20CUDA-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A fast, lightweight, and offline-safe machine learning system that detects whether a given website URL is **Legitimate (Safe)** or **Phishing (Malicious)** using **18 lexical (text-based) features**.

---

## 1. Problem Statement
Phishing is one of the most widespread cyber attacks, where adversaries create spoofed websites impersonating trusted entities (banks, e-commerce, cloud providers, social networks) to harvest credentials, session tokens, and credit card information. To the human eye, disguised URLs often appear genuine. 

Traditional detection methods rely on **blacklists** (e.g., Google Safe Browsing), which fail to catch newly registered zero-day phishing domains. Modern machine learning solutions analyze URL characteristics directly to identify fraudulent sites the moment they are generated.

---

## 2. Approach & Architecture
PhishGuard uses **lexical URL analysis**. The system strictly inspects the character patterns, structural counts, and linguistic tokens of the URL string itself.

- **Safe:** The target website is **never visited, queried, or downloaded**.
- **Offline Compatible:** Operates instantaneously without external network requests or DNS lookups.
- **Low Latency:** Inference executes in sub-millisecond time.

```
                    +---------------------------+
                    |       User Input URL      |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Lexical Feature Extractor |
                    |   (18 Numerical Features) |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |  Trained ML Classifier    |
                    | (Tuned XGBoost / RF Model)|
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | Verdict & Confidence (%)  |
                    |  [SAFE]  or  [PHISHING]   |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
          +-------------------+       +-------------------+
          |  Command Line CLI |       | Streamlit Web App |
          +-------------------+       +-------------------+
```

---

## 3. Dataset
We utilized the **Malicious URLs Dataset** (by Manu Siddhartha, Kaggle), comprising 651,191 raw labeled URLs.
- **Data Cleaning & Balancing:** To prevent majority-class bias and enable fast training, we cleaned duplicate URLs, mapped labels (`0 = Legitimate`, `1 = Phishing`), and sampled an evenly balanced subset of **50,000 URLs** (**25,000 Legitimate** and **25,000 Phishing**).
- **Split:** 80% Training (40,000 URLs) and 20% Testing (10,000 unseen URLs), stratified with `random_state=42`.

---

## 4. Feature Engineering (18 Lexical Features)

| # | Feature | Security Description |
|---|---|---|
| 1 | `url_length` | Total character count of URL. Phishing URLs tend to be longer. |
| 2 | `hostname_length` | Length of domain/hostname part. |
| 3 | `path_length` | Length of URL path. Phishing links often conceal tokens in deep paths. |
| 4 | `num_dots` | Count of `.`. Multiple dots often indicate deceptive subdomains. |
| 5 | `num_hyphens` | Count of `-`. Commonly used to mimic brand names (e.g. `paypal-update`). |
| 6 | `num_underscores` | Count of `_`. |
| 7 | `num_slashes` | Count of `/`. Indicates deep directory structure. |
| 8 | `num_digits` | Total count of numerical digits `0-9`. |
| 9 | `num_special_chars` | Count of special characters (`@ ? & = % # ~ + $ ! * ;`). |
| 10 | `has_at_symbol` | Flag for `@`. Browser credentials trick (disguises actual destination). |
| 11 | `has_ip_address` | Flag if domain is an IPv4 address instead of a registered domain. |
| 12 | `uses_https` | Flag if URL protocol is HTTPS. |
| 13 | `num_subdomains` | Count of subdomains (e.g. `login.bank.com` has 1 subdomain). |
| 14 | `has_suspicious_words`| Flag if URL contains keywords like `login`, `verify`, `account`, `bank`, `paypal`. |
| 15 | `has_shortening_service`| Flag for link shorteners (e.g. `bit.ly`, `tinyurl`). |
| 16 | `has_double_slash_redirect`| Flag if `//` appears after protocol (open redirect evasion). |
| 17 | `digit_ratio` | Ratio of numerical digits to total URL length. |
| 18 | `letter_ratio` | Ratio of alphabetic characters to total URL length. |

---

## 5. Model Evaluation & Benchmark Results

All models were evaluated on the **same 10,000 unseen test URLs**:

| Model | Hardware | Accuracy | Precision | Recall | F1 Score | Training Time |
|---|---|---|---|---|---|---|
| **Logistic Regression** | CPU | 75.22% | 72.57% | 81.10% | 76.60% | 0.05s |
| **Decision Tree** | CPU | 90.48% | 90.30% | 90.70% | 90.50% | 0.15s |
| **Random Forest** | 12 CPU Threads | 91.87% | 90.86% | 93.10% | 91.97% | 0.47s |
| **XGBoost** | NVIDIA RTX 3050 | 91.97% | 91.38% | 92.68% | 92.03% | 0.85s |
| **Tuned XGBoost (Final)** | **NVIDIA RTX 3050** | **92.64%** | **92.20%** | **93.16%** | **92.68%** | **18.0s (GridSearchCV)** |

- **Best Model:** Tuned XGBoost (`learning_rate=0.2`, `max_depth=8`, `n_estimators=150`).
- **ROC-AUC Score:** **0.9786 (97.86%)**
- **Test Set Confusion Matrix:**
  - True Negatives (Legitimate correctly identified): **4,606**
  - True Positives (Phishing correctly caught): **4,658**
  - False Positives (Legitimate flagged as phishing): **394**
  - False Negatives (Phishing missed): **342**

---

## 6. How to Set Up and Run

### Prerequisites
- Python 3.9+ (Windows / Linux / macOS)
- PowerShell or Terminal

### Installation
```powershell
# 1. Clone or open the repository
cd phishguard

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux/macOS

# 3. Install required libraries
pip install -r requirements.txt
```

### Running the Project

1. **Run Exploratory Data Analysis (EDA):**
   ```powershell
   jupyter notebook notebooks/exploration.ipynb
   ```

2. **Run Feature Extraction:**
   ```powershell
   python src/feature_extraction.py
   ```

3. **Train Models and Benchmark:**
   ```powershell
   python src/train.py
   ```

4. **Evaluate Best Model & Generate Evaluation Figures:**
   ```powershell
   python src/evaluate.py
   ```

5. **Test CLI Predictions:**
   ```powershell
   # Run the benchmark test suite
   python src/predict.py

   # Test any custom URL
   python src/predict.py "google.com/search?q=cybersecurity"
   python src/predict.py "http://paypal-verification-account-update.xyz/login.html"
   ```

6. **Launch the Streamlit Web Interface:**
   ```powershell
   streamlit run app/app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 7. Limitations

1. **Lexical Analysis Only:** The system analyzes URL text structure only; it cannot inspect webpage HTML, JavaScript execution, SSL/TLS certificate validity, or WHOIS domain age.
2. **Evasion by Sophisticated Attackers:** Attackers can register authentic-looking domains (homograph attacks, typo-squatting) that mimic clean lexical properties.
3. **Dataset Bias:** Training patterns depend on dataset composition. Ongoing retraining with fresh real-world threat feeds is essential.
4. **Link Shorteners:** Services like `bit.ly` hide the destination path until resolved.

---

## 8. Future Scope

1. **Content & WHOIS Integration:** Incorporate domain registration age, DNS records, and page DOM analysis for hybrid defense.
2. **Browser Extension:** Package the trained lightweight model into a Chrome/Firefox extension for real-time browsing protection.
3. **Deep Learning Character Embeddings:** Implement 1D-CNN or Bidirectional LSTM models directly on raw character sequences.
4. **Real-time Threat Feed API:** Integrate with live threat feeds (PhishTank, OpenPhish) to continuously update model weights.
