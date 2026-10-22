# PhishGuard: Presentation Notes & Viva Preparation Guide

This document is structured to help you present this college mini-project with confidence and answer any examiner/viva questions with ease.

---

## Part 1: Suggested Slide-by-Slide Presentation Outline (12 Slides)

### Slide 1: Title Slide
- **Title:** PhishGuard: Phishing URL Detection using Machine Learning
- **Domain:** Cyber Security & Applied Machine Learning
- **Presenter Name / Roll No / College:** [Your Name / Details]
- **Guide / Supervisor:** [Supervisor Name]
- **Key Highlight:** Real-time, offline lexical URL analysis using GPU-accelerated gradient boosting.

### Slide 2: Introduction
- **What is Phishing?** A form of social engineering where attackers impersonate legitimate organizations (banks, email providers, payment gateways) using deceptive websites.
- **How Attacks Spread:** Phishing links are distributed via emails, SMS (Smishing), QR codes (Quishing), and social media messages.
- **Why It Matters:** According to cybersecurity reports, over 80% of security incidents start with a phishing attack, leading to credential theft and financial losses.

### Slide 3: Problem Statement & Motivation
- **The Core Problem:** Casual Internet users cannot reliably distinguish between a spoofed URL (e.g. `http://secure-login.bankofamerica.update-verification.com/login.php`) and a legitimate domain (`bankofamerica.com`).
- **Shortcomings of Existing Blacklists:**
  - Google Safe Browsing and antivirus blacklists rely on community reports.
  - They suffer from high latency (often 24 to 48 hours to blacklist a newly registered malicious site).
  - Average phishing site lifespan is under 4 to 8 hours (zero-day phishing).
- **Motivation:** Build an automated machine learning detector that spots phishing patterns immediately from URL syntax.

### Slide 4: Project Objectives
1. Collect and preprocess a large-scale labeled URL dataset.
2. Extract **18 lexical (text-based) features** from URLs without visiting or requesting the website.
3. Train, benchmark, and compare 4 distinct machine learning algorithms.
4. Accelerate model training using GPU capabilities (NVIDIA CUDA).
5. Build an intuitive web user interface using Streamlit for live demonstration.

### Slide 5: Literature Survey / Existing Approaches
- **Blacklist-based approach:** Fast lookups, but completely blind to zero-hour attacks.
- **Content-based approach:** Downloads the HTML/JS and checks logos/DOM. Problem: High latency, requires network requests, exposes users to malicious payloads, fails on login walls.
- **Lexical URL-based approach (Our Chosen Approach):**
  - Evaluates characters, length, subdomains, and tokens directly from the URL string.
  - Safe, lightning-fast, works offline, and evaluates zero-day URLs.

### Slide 6: Proposed System Overview
- **Key Design Philosophy:** Purely lexical inspection.
- **Safety First:** The system never visits or opens the URL, eliminating malware drive-by download risks.
- **Instantaneous:** Feature extraction and inference take less than 1 millisecond per URL.
- **Two User Interfaces:**
  - Fast CLI script (`src/predict.py`) for automated security pipelines.
  - Interactive Streamlit Web UI (`app/app.py`) for end users and live demonstrations.

### Slide 7: System Architecture & Workflow Diagram
*(See Part 2 below for the text description and ASCII diagram to show or speak about)*
- Raw URL -> Preprocessor & Lexical Extractor -> Feature Vector (18 Features) -> Trained ML Classifier (XGBoost) -> Probability & Verdict -> Output UI.

### Slide 8: Dataset & Exploratory Data Analysis (EDA)
- **Dataset:** Malicious URLs Dataset (Kaggle).
- **Handling Imbalance:** The raw data had an 82% to 18% legitimate-to-phishing ratio. We created an evenly balanced dataset of **50,000 URLs** (25,000 Legitimate, 25,000 Phishing).
- **Split:** 80% Training (40,000) and 20% Unseen Testing (10,000).
- **Key Insight from EDA:** Phishing URLs frequently have multiple subdomains, suspicious keywords, and abnormal path lengths.

### Slide 9: Feature Engineering (18 Lexical Features)
- **Structural:** `url_length`, `hostname_length`, `path_length`
- **Character Counts:** `num_dots`, `num_hyphens`, `num_underscores`, `num_slashes`, `num_digits`, `num_special_chars`
- **Security Flags (0/1):**
  - `has_at_symbol` (browser credential disguise)
  - `has_ip_address` (e.g. `http://192.168.1.1/login`)
  - `uses_https`
  - `num_subdomains`
  - `has_suspicious_words` (`login`, `verify`, `bank`, `account`, `paypal`)
  - `has_shortening_service` (`bit.ly`, `tinyurl`)
  - `has_double_slash_redirect` (`//` after protocol)
- **Ratios:** `digit_ratio`, `letter_ratio`

### Slide 10: Model Comparison & Experimental Results
- **Models Benchmarked on 10,000 Test URLs:**
  1. Logistic Regression: **75.22% Accuracy**, 81.10% Recall
  2. Decision Tree: **90.48% Accuracy**, 90.70% Recall
  3. Random Forest (12 CPU threads): **91.87% Accuracy**, 93.10% Recall
  4. XGBoost (Tuned on GPU): **92.64% Accuracy**, **93.16% Recall**, **0.9786 ROC-AUC**
- **Hardware Acceleration:** Model training and hyperparameter search (`GridSearchCV`) were accelerated on an **NVIDIA GeForce RTX 3050 Laptop GPU** using CUDA.

### Slide 11: Live Demonstration
- Showcase the **Streamlit Web App** running on `localhost:8501`.
- Demonstrate with 3 distinct test cases:
  1. Common legitimate site (`google.com/search?q=cybersecurity+threats`) -> **Legitimate (97.4%)**
  2. Brand spoofing phishing site (`http://paypal-verification-account-update.xyz/login.html`) -> **Phishing (98.4%)**
  3. Raw IP attack URL (`http://192.168.1.1/admin/login.php`) -> **Phishing (99.7%)**
- Open the feature breakdown expander to demonstrate how the model explains its reasoning.

### Slide 12: Limitations, Conclusion & Future Scope
- **Conclusion:** Built a complete, high-accuracy (92.6%), high-recall (93.2%) phishing URL detection system using purely lexical features.
- **Limitations:** Cannot inspect page content, SSL certificate validity, or domain age; vulnerable to link shorteners if unexpanded.
- **Future Work:** Package into a browser extension, add WHOIS domain age lookup, and integrate deep learning character embeddings (1D-CNN).

---

## Part 2: System Workflow Diagram Description

### Visual Diagram (Show this or draw on whiteboard)
```
[User Input URL]
       │
       ▼
[URL Normalization & Safe Parser]
       │
       ▼
[Lexical Feature Extraction Engine]
 ├── Length Metrics (URL, Host, Path)
 ├── Character Counts (Dots, Hyphens, Slashes, Digits)
 ├── Security Flags (IP Address, @, Subdomains, HTTPS)
 └── Suspicious Pattern Check (Keywords, Shorteners, // Redirect)
       │
       ▼ (18 Numeric Features Vector)
[Trained Machine Learning Model] (Tuned XGBoost / Random Forest)
       │
       ▼
[Softmax Probability Engine]
 ├── Phishing Probability (e.g., 98.35%)
 └── Legitimate Probability (e.g., 1.65%)
       │
       ▼
[Classification Verdict]
 ├── If Phishing Prob >= 50%  ──► [🚨 PHISHING ALERT (RED)]
 └── If Phishing Prob < 50%   ──► [✅ LEGITIMATE SAFE (GREEN)]
       │
       ▼
[User Output Interface] (Streamlit Web Dashboard / Terminal CLI)
```

### Simple Spoken Explanation of the Workflow:
> "Our system follows a 5-stage pipeline:
> 1. **Input:** The user submits a URL into our Streamlit app or CLI tool.
> 2. **Lexical Feature Extraction:** Without contacting the web server, the system parses the URL string and extracts 18 numerical and binary indicators, such as subdomain counts, path length, presence of raw IP addresses, and phishing keywords.
> 3. **Model Inference:** The extracted feature vector is fed into our trained XGBoost classifier.
> 4. **Risk Scoring:** The classifier outputs class probabilities (e.g. 98% threat probability).
> 5. **Verdict & Transparency:** The UI displays a clear color-coded verdict and expands the exact feature table so the user understands why the decision was made."

---

## Part 3: Top 10 Viva / Presentation Questions with Ready Answers

#### Q1: Why did you use lexical features instead of downloading the website content?
**Answer:**
"Using lexical URL features has three key advantages:
1. **Safety:** Visiting a phishing site can trigger drive-by malware downloads or notify the attacker. Lexical analysis stays completely offline.
2. **Speed:** Lexical feature extraction takes under 1 millisecond, whereas downloading HTML and rendering JavaScript takes several seconds.
3. **Resilience:** Attackers often use cloaking techniques to show fake harmless content to crawlers; lexical features cannot be hidden in this way."

#### Q2: Why did you choose Random Forest and XGBoost over Logistic Regression?
**Answer:**
"Logistic Regression is a linear classifier that assumes linear relationships between features. Phishing detection involves complex non-linear feature interactions (for example: having suspicious keywords is much more dangerous when combined with a raw IP address or multiple subdomains). Tree ensembles like Random Forest and XGBoost excel at capturing these non-linear patterns without overfitting."

#### Q3: What is the difference between Precision and Recall, and which one is more critical here?
**Answer:**
- **Precision:** The proportion of URLs flagged as phishing that are genuinely phishing. High precision means fewer false alarms.
- **Recall:** The proportion of all actual phishing URLs that our system successfully detected.
- **Which is more critical?** In cybersecurity, **Recall is the most critical metric**. If our system misses a phishing URL (a False Negative), the user will trust the website, enter their password or credit card details, and suffer a compromise. A False Positive (flagging a safe site) is only a minor warning inconvenience. Our model achieves **93.16% Recall**."

#### Q4: What is overfitting and how did you prevent it?
**Answer:**
"Overfitting occurs when a model memorizes the training data, including noise, and performs poorly on unseen data. We prevented overfitting by:
1. Evaluating on a dedicated, held-out **20% unseen test set** (10,000 URLs).
2. Applying **3-fold cross-validation** during hyperparameter tuning (`GridSearchCV`).
3. Limiting tree depth (`max_depth=8`) and using shrinkage (`learning_rate=0.2`) in XGBoost."

#### Q5: Why did you balance the dataset during EDA?
**Answer:**
"The raw dataset contained ~82% legitimate URLs and ~18% phishing URLs. When models train on severely imbalanced datasets, they tend to guess the majority class (legitimate) and obtain deceptively high 'accuracy' while failing to catch phishing links (low recall). By sampling an equal 25,000 legitimate and 25,000 phishing URLs, we ensured unbiased learning."

#### Q6: What is the ROC curve and the AUC score?
**Answer:**
"The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Recall) against the False Positive Rate across all possible classification probability thresholds. The Area Under the Curve (AUC) measures the model's overall ability to discriminate between legitimate and phishing URLs. An AUC of 0.5 represents random guessing, while 1.0 is a perfect classifier. Our model achieved an **AUC of 0.9786 (97.86%)**, demonstrating high discriminative power."

#### Q7: How was GPU acceleration utilized in this project?
**Answer:**
"We utilized an **NVIDIA GeForce RTX 3050 Laptop GPU** using CUDA. Specifically, we configured XGBoost with `device='cuda'`, which offloaded gradient histogram computations to the GPU during training and cross-validated grid search (`GridSearchCV`), completing 24 cross-validation fits in just 18 seconds."

#### Q8: What are the main limitations of your system?
**Answer:**
"Honesty about limitations is essential:
1. **No Content Visibility:** Because we only inspect the URL text, we cannot inspect website HTML, SSL certificates, or domain WHOIS age.
2. **Evasion:** Sophisticated attackers using homograph characters (e.g. Cyrillic letters looking like English letters) or clean short URLs can evade purely lexical rules.
3. **Dataset Dependency:** Machine learning models must be periodically retrained with new threat feeds as attacker tactics evolve."

#### Q9: How does the system handle URL shortening services like bit.ly or tinyurl?
**Answer:**
"Our feature extractor checks for known link-shortening services (`has_shortening_service` flag). Shortened links hide the actual destination domain, so the model treats them with higher suspicion. In a commercial production deployment, we would perform an HTTP `HEAD` request to expand the redirect target before lexical extraction."

#### Q10: If you had another month to work on this, what would you improve?
**Answer:**
"I would implement three enhancements:
1. Develop a **Google Chrome browser extension** that uses this model to alert users before they open suspicious links.
2. Build a **hybrid architecture** that combines lexical features with domain registration age (via WHOIS API).
3. Experiment with a deep learning character-level **1D-CNN or LSTM model** to learn subtle character sequence patterns automatically."
