# 🛡️ Phishing Email Detector

A machine-learning tool built with **Scikit-learn** that classifies emails as **Phishing** or **Safe** using both the email's text and URL-based features. It reaches **~96.7% accuracy** on a real dataset of 18,000+ emails and comes with a friendly command-line interface that explains *why* an email looks suspicious.

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-96.7%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

- **Trains on labeled emails**: phishing vs. legitimate (your own CSV or the included demo generator).
- **Feature extraction**: TF-IDF on the email text plus 13 hand-crafted URL and style features.
- **Two models compared**: Logistic Regression and Random Forest; the better one is saved automatically.
- **Evaluation**: accuracy, precision, recall and a confusion matrix (printed and saved as an image).
- **Friendly scanner**: colored verdict, risk meter (0–100%), red flags, good signs and advice.
- **Hybrid scoring**: the ML score is combined with a rule-based check, so obvious red flags are never ignored.
- **Agreement check**: if the ML model says "phishing" but the rule check finds nothing technical (normal https links to well-known domains), the verdict is downgraded to *Suspicious* instead of claiming certainty.
- **Simple CLI**: interactive menu, or one-line commands for scripting.

---

## 🧠 How it works

```
Email text
   │
   ├─► clean_text()      → lowercase, URLs replaced with a token ──► TF-IDF (1–2 word n-grams)
   │
   └─► hand_features()   → 13 numeric URL / keyword / style features ──► StandardScaler
                                          │
                                          ▼
                       Logistic Regression  /  Random Forest
                                          │
                                          ▼
                          Phishing probability (0–1)
                                          │
                  max(ML score, 0.9 × rule-based score)
                                          │
                                          ▼
                     Verdict + risk meter + red flags
```

### Extracted features

| Category | Features |
|---|---|
| **URL structure** | number of URLs, raw IP address in URL, URL shorteners (bit.ly…), suspicious TLDs (.xyz, .top, .ru…), `http://` instead of `https://`, `@` inside URL, longest URL length, dots and digits in URL |
| **Keywords** | count of words like *urgent, verify, suspend, password, login, bank, refund…* |
| **Style** | exclamation marks, ratio of CAPITAL letters, word count |
| **Text** | TF-IDF unigrams + bigrams of the cleaned email body |

### Verdict levels

| Risk score | Verdict |
|---|---|
| ≥ 65% | 🔴 **PHISHING – Dangerous** |
| 35% – 65% | 🟡 **SUSPICIOUS – Be careful** |
| < 35% | 🟢 **SAFE – Looks fine** |

---

## 📁 Project structure

```
Phishing_Email_Detector.py/
├── Phishing_Email_Detector.py     # main script (training + scanning + CLI)
├── requirements.txt         # python3 dependencies
├── confusion_matrix.png     # generated after training
├── phishing_model.joblib    # generated after training (ignored by git)
├── README.md
└── .gitignore
```

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/HackerRank7/Phishing_Email_Detector.py.git
cd Phishing_Email_Detector.py

# 2. (Recommended) create a virtual environment
python3 -m venv venv
# Windows:      venv\Scripts\activate
# Linux / Mac:  source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset

The results below use the **Phishing Email Detection** dataset from Kaggle (`Phishing_Email.csv`, by *subhajournal*).

1. Download it from Kaggle: <https://www.kaggle.com/datasets/subhajournal/phishingemails>
2. Place `Phishing_Email.csv` in the project folder.

> The dataset is **not included** in this repository. Please download it from Kaggle and follow its license terms.

The loader accepts either of these CSV layouts:

| Layout | Columns |
|---|---|
| Kaggle format | `Email Text`, `Email Type` |
| Generic | `text`, `label` (1/0 or phishing/safe) |

No dataset? Run the script without `--data` (menu option 2, press Enter) to train on built-in **synthetic demo data**. Note that accuracy on this fake data is unrealistically high.

---

## 💻 Usage

### Interactive menu (easiest)

```bash
python3 Phishing_Email_Detector.py
```

```
══════════════════════════════════════════════════════════════
  PHISHING EMAIL DETECTOR
══════════════════════════════════════════════════════════════
   1) Scan an email
   2) Train the model (on your CSV)
   3) Exit
```

### Command-line options

```bash
# Train on your dataset (saves phishing_model.joblib and confusion_matrix.png)
python3 Phishing_Email_Detector.py --data Phishing_Email.csv

# Open the scanner with the saved model (paste an email, type END to finish)
python3 Phishing_Email_Detector.py --interactive

# Scan one email directly
python3 Phishing_Email_Detector.py --predict "URGENT: verify your password at http://secure-login.xyz"

# Scan an email stored in a text file
python3 Phishing_Email_Detector.py --file email.txt
```

### Example output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [!!] VERDICT: PHISHING - DANGEROUS
  Phishing risk : ██████████████████████░░ 90%
  (ML model: 15%  |  Rule check: 100%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Things to notice (red flags):
     * Contains 1 link(s): http://secure-paypal-login.xyz/verify
     * Link has a suspicious domain ending (.xyz / .top / .ru etc.)
     * Link is not secure (http instead of https)
     * Suspicious words found: urgent, verify, suspend, password, account, login
     * Creates urgency to make you act fast
     * Asks for sensitive information (password / bank / SSN)

  Advice: Do not click any links, open attachments or share information.
```

---

## 📈 Results

Trained on **18,634 emails** (7,312 phishing / 11,322 safe), 80/20 train–test split (3,727 test emails).

| Model | Accuracy |
|---|---|
| **Logistic Regression** ✅ | **96.73%** |
| Random Forest | 96.62% |

**Confusion matrix** (Logistic Regression, test set):

| | Predicted Safe | Predicted Phishing |
|---|---|---|
| **Actually Safe** | 2,168 | 97 *(false alarms)* |
| **Actually Phishing** | 25 *(missed)* | 1,437 |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Safe | 0.99 | 0.96 | 0.97 |
| Phishing | 0.94 | 0.98 | 0.96 |

**Takeaways**
- The model catches **~98% of phishing emails** (recall 0.98), which matters most for security.
- It wrongly flags ~4% of safe emails; an acceptable trade-off for a phishing filter.

![Confusion Matrix](confusion_matrix.png)

---

## ⚠️ Limitations

- Analyzes **email body text only**. It does not inspect headers, sender reputation, SPF/DKIM, or attachments.
- Trained on one public dataset; performance on other email styles, languages or newer attack techniques may be lower.
- Public phishing datasets often contain mostly personal/corporate "safe" emails, so legitimate marketing or transactional emails (order confirmations, account notices) can be unfamiliar to the model and may be over-flagged. Adding such emails to the training data is the best fix.
- URLs are analyzed by their **structure**, not checked against live blacklists or fetched.
- The rule-based score is a heuristic and can raise the risk for legitimate emails that happen to contain many trigger words.
- This is an educational project, **not a replacement** for professional email security tools.

---

## 🔮 Future improvements

- Add email header features (sender domain, SPF/DKIM, reply-to mismatch)
- Try LinearSVC / XGBoost and character n-grams for obfuscated words (`p@ssw0rd`)
- Cross-validation and hyperparameter tuning
- Web UI (Flask / Streamlit) or a browser extension
- Integrate URL reputation APIs (e.g. PhishTank)

---

## 🤝 Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

## 📄 License

Copyright (c) 2026 <Your Name>. Distributed under the MIT License. Add a `LICENSE` file to your repository (GitHub can generate one for you).

## 🙏 Acknowledgements

- Dataset: *Phishing Email Detection* by **subhajournal** on [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails)
- Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/) and [matplotlib](https://matplotlib.org/)

## 👤 Author

[![Gaurav Bharty](https://img.shields.io/badge/Gaurav%20Bharty-orange)]()

If you found this project useful, consider giving it a ⭐ on GitHub!
