# Project Brief: Phishing URL Detection using Machine Learning

> **To the AI agent (Antigravity):** Read this entire file first. Then build the project **one step at a time**.
> Do **NOT** attempt multiple steps in one go.

---

## 0. How You Must Work (Read Carefully)

1. Complete **only one step at a time**, in the order given below.
2. After finishing a step, **stop** and give me:
   - A short summary of what you did (plain language)
   - The files you created or changed
   - How I can run or verify it myself
3. Then **wait** for me to reply `continue` before starting the next step.
4. If something is unclear, or a choice is needed (e.g., dataset), **ask me** instead of guessing.
5. Keep code **simple, well commented, and beginner friendly**. This is a college mini project and I must be able to explain every line during my presentation.
6. Do not add extra features, frameworks, or complexity that are not in this brief.
7. If a step fails, explain the error in simple words and fix it before moving on.

---

## 1. Project Overview

**Title:** Phishing URL Detection using Machine Learning

**Domain:** Cyber Security + Machine Learning

**Problem:** Phishing is a cyber attack where attackers create fake websites that imitate real ones (banks, e-commerce, social media) to steal passwords, card details and personal data. Users usually cannot tell a fake URL from a real one at a glance.

**Goal:** Build a system that takes a URL as input and predicts whether it is **Legitimate (real)** or **Phishing (fake)** using a machine learning model.

**Approach:**
1. Collect a labeled dataset of URLs (phishing vs legitimate)
2. Extract features from each URL (length, number of dots, presence of `@`, use of IP address, HTTPS, suspicious words, etc.)
3. Train and compare multiple ML models
4. Pick the best model based on evaluation metrics
5. Build a simple interface where a user pastes a URL and gets a result

**Important design choice:** Use **lexical (text based) features of the URL only**. The system must NOT visit or open the URL, so it stays safe and works offline and fast.

---

## 2. Tech Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.9+ |
| Data handling | pandas, numpy |
| ML models | scikit-learn (and optionally xgboost) |
| Visualization | matplotlib, seaborn |
| URL parsing | `urllib.parse`, `re`, `tldextract` (optional) |
| Saving model | joblib |
| User interface | Streamlit (simple web app) |
| Notebook (for exploration) | Jupyter Notebook |

---

## 3. Target Folder Structure

```
phishing-url-detection/
├── data/
│   ├── raw/                 # original dataset (do not edit)
│   └── processed/           # feature extracted dataset
├── notebooks/
│   └── exploration.ipynb    # EDA and experiments
├── src/
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── best_model.joblib
├── app/
│   └── app.py               # Streamlit app
├── reports/
│   └── figures/             # charts and confusion matrices
├── requirements.txt
└── README.md
```

---

## 4. Step by Step Plan

> Remember: finish ONE step, summarize, then wait for me to say `continue`.

---

### STEP 1: Project Setup
**Tasks**
- Create the folder structure shown above
- Create `requirements.txt` with: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit, tldextract, jupyter
- Create a virtual environment and install the requirements
- Create a basic `README.md` with the project title and a one line description

**Done when:** folders exist, packages install without error, and I can run `python --version` and import pandas in the environment.

**Stop and wait for me.**

---

### STEP 2: Dataset Collection
**Tasks**
- Help me choose a labeled phishing URL dataset. Suggest 2 or 3 options (for example, public datasets on Kaggle such as "Phishing Site URLs" or "PhiUSIIL Phishing URL Dataset") and **ask me which one to use**.
- If you cannot download it automatically, tell me exactly where to download it and where to place it (`data/raw/`).
- Load the dataset with pandas and show: shape, column names, first 5 rows, and class balance (count of phishing vs legitimate).

**Done when:** dataset is in `data/raw/` and loads correctly with a clear label column (phishing = 1, legitimate = 0).

**Stop and wait for me.**

---

### STEP 3: Data Exploration and Cleaning (EDA)
**Tasks**
- In `notebooks/exploration.ipynb`, check for missing values and duplicate URLs and remove them
- Make sure labels are encoded consistently (1 = phishing, 0 = legitimate)
- If classes are heavily imbalanced, tell me and suggest a simple fix (like sampling a balanced subset)
- Create 2 or 3 simple charts (class distribution, URL length distribution by class)
- Save the figures to `reports/figures/`

**Done when:** a clean dataframe is ready and charts are saved.

**Stop and wait for me.**

---

### STEP 4: Feature Extraction
**Tasks**
Write `src/feature_extraction.py` with a function that converts one URL into a dictionary of numeric features. Use these features:

| Feature | Meaning |
|---|---|
| `url_length` | total characters in URL |
| `hostname_length` | length of domain part |
| `path_length` | length of path |
| `num_dots` | count of `.` |
| `num_hyphens` | count of `-` |
| `num_underscores` | count of `_` |
| `num_slashes` | count of `/` |
| `num_digits` | count of digits |
| `num_special_chars` | count of `@ ? & = % # ~` etc. |
| `has_at_symbol` | 1 if `@` present |
| `has_ip_address` | 1 if domain is an IP address |
| `uses_https` | 1 if scheme is https |
| `num_subdomains` | number of subdomains |
| `has_suspicious_words` | 1 if contains words like login, verify, secure, account, update, bank, confirm, paypal |
| `has_shortening_service` | 1 if bit.ly, tinyurl, etc. |
| `has_double_slash_redirect` | 1 if `//` appears after the protocol |
| `digit_ratio` | digits divided by URL length |
| `letter_ratio` | letters divided by URL length |

Then apply it to the whole dataset and save the result as `data/processed/features.csv`.

**Done when:** I can run the function on any URL and see the feature values, and the processed CSV exists.

**Stop and wait for me.**

---

### STEP 5: Train / Test Split and Baseline Model
**Tasks**
- Split data 80% train / 20% test (use `stratify` and a fixed `random_state=42`)
- Train a baseline **Logistic Regression** model
- Print accuracy, precision, recall and F1 on the test set
- Explain briefly in simple words what each metric means and why recall matters for phishing detection

**Done when:** baseline metrics are shown and explained.

**Stop and wait for me.**

---

### STEP 6: Train and Compare Multiple Models
**Tasks**
In `src/train.py`, train and compare these models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting (or XGBoost if installed)

For each model, record accuracy, precision, recall, F1 and training time. Show the results in a **comparison table** and a **bar chart** (save to `reports/figures/`). Do light hyperparameter tuning only for the best one or two models (GridSearchCV with a small grid is enough).

**Done when:** a clear comparison table and chart exist and a best model is identified.

**Stop and wait for me.**

---

### STEP 7: Detailed Evaluation of the Best Model
**Tasks**
In `src/evaluate.py`:
- Show the confusion matrix (as a heatmap) and the classification report
- Plot the ROC curve with the AUC score
- Plot **feature importance** (which URL features matter most)
- Explain in simple words what the results mean, including false positives vs false negatives in a phishing context
- Save the best model to `models/best_model.joblib`

**Done when:** all plots are saved and the model file exists.

**Stop and wait for me.**

---

### STEP 8: Prediction Script
**Tasks**
Write `src/predict.py` that:
- Loads the saved model
- Accepts a URL from the command line
- Extracts features, predicts, and prints: **Legitimate** or **Phishing**, along with the confidence percentage
- Test it on at least 5 URLs (a mix of real-looking and suspicious-looking ones) and show the outputs

**Done when:** `python src/predict.py "<url>"` works.

**Stop and wait for me.**

---

### STEP 9: Web Interface (Streamlit)
**Tasks**
Build `app/app.py` with:
- A title and a short description
- A text box to paste a URL and a "Check URL" button
- A clear result: green for Legitimate, red for Phishing, with confidence
- An expandable section that shows the extracted features
- A small warning line: "This is an educational project. Do not rely on it as your only protection."
- Do NOT open or fetch the URL in the app

**Done when:** `streamlit run app/app.py` works and gives correct results.

**Stop and wait for me.**

---

### STEP 10: Documentation and Presentation Support
**Tasks**
- Finish `README.md`: problem statement, approach, dataset, features, results, how to run, limitations, future scope
- Create a file `PRESENTATION_NOTES.md` with:
  - A suggested slide outline (about 10 to 12 slides: Title, Introduction, Problem Statement, Objectives, Literature/Existing Methods, Proposed System, Architecture/Workflow, Dataset, Features, Models and Results, Demo, Conclusion and Future Scope)
  - A simple text description of the system workflow diagram
  - 10 likely viva or presentation questions with short answers (e.g., why Random Forest, what is precision vs recall, what is overfitting, what are the limitations)

**Done when:** both files are complete and accurate to what we built.

**Stop and wait for me.**

---

## 5. Limitations to Mention (for honesty in the presentation)

- Uses URL text features only, so it cannot see page content, certificates or domain age
- Attackers can craft URLs that avoid these patterns
- Dataset bias: the model is only as good as its training data
- Needs retraining over time as phishing tactics change

## 6. Future Scope Ideas (do NOT build unless I ask)

- Add content based and WHOIS/domain age features
- Browser extension
- Deep learning model (LSTM or CNN on URL characters)
- Real time threat feed integration

---

## 7. Final Reminder

Start with **STEP 1 only**. After finishing it, summarize and **wait for me to say `continue`**.
