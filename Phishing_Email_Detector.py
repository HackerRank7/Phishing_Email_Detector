"""
Phishing Email Detector
-----------------------
How to run:
    python phishing_detector.py                       # opens the menu (easiest)
    python phishing_detector.py --data emails.csv     # train on your own CSV
    python phishing_detector.py --interactive         # scan emails with the saved model
    python phishing_detector.py --predict "email text"
    python phishing_detector.py --file email.txt      # scan an email stored in a text file
    python phishing_detector.py --adversarial-test    # [NEW] test against disguised (obfuscated) emails

CSV format (both work):
    columns: text,label                  (label: 1/0, phishing/safe)
    columns: Email Text,Email Type       (Kaggle "Phishing_Email.csv" format)

=====================================================================================
[CHANGELOG - HINGLISH] Is version me kya-kya naya / badla hai
=====================================================================================
Jahan bhi code me  # [NAYA]  likha hai = bilkul nayi cheez add hui hai
Jahan bhi code me  # [BADLAV] likha hai = purane code me change hua hai

 1. EXPLAINABILITY: Ab tool batata hai ki ML model ne kin words/features ki wajah se
    "phishing" ya "safe" bola (Logistic Regression ke coefficients se, SHAP ki zaroorat nahi).
       -> naya function: explain_model()   |  show_result() me print hota hai
 2. ADVERSARIAL DEFENCE: attacker ke tricks ko pakadne ke liye
       - normalize_text(): leetspeak (p@ssw0rd), spaced letters (v e r i f y),
         look-alike (homoglyph) letters (Cyrillic 'a'), zero-width characters ko theek karta hai
       - has_homoglyph_url(): URL ke domain me non-English/look-alike characters ka check
       - 2 naye hand features (ab 13 ki jagah 15 features)
       - character n-grams (TF-IDF char_wb 3-5) -> misspelling/obfuscation me madad
       - naya --adversarial-test command: obfuscated emails par tool ko test karta hai
 3. FEEDBACK LOOP: scan ke baad "Was this verdict correct?" poochta hai. Galat ho to
    feedback.csv me save hota hai aur agli training me use hota hai.
       -> naye functions: save_feedback(), load_feedback(), ask_feedback()
 4. IMBALANCE: class_weight="balanced" PEHLE SE code me tha (dono models me). Ab training
    report me class balance aur precision/recall/F1 bhi dikhte hain.
 5. IMPORTANT: features badal gaye hain, isliye purana phishing_model.joblib kaam nahi karega.
    Dobara train karna padega (menu option 2 ya --data).
=====================================================================================
"""
import argparse
import csv  # [NAYA] feedback.csv likhne ke liye
import os
import random
import re
import sys
import unicodedata  # [NAYA] unicode (look-alike) characters ko samajhne/normalize karne ke liye

import joblib
import matplotlib
matplotlib.use("Agg")  # save images without needing a screen/GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix,
                             precision_recall_fscore_support)  # [NAYA] precision/recall/F1 ke liye
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "phishing_model.joblib"  # the trained model is saved here
FEEDBACK_PATH = "feedback.csv"        # [NAYA] user ka feedback (galat verdicts) yahan save hota hai
FEEDBACK_REPEAT = 3                   # [NAYA] feedback wali email training me 3 baar daali jati hai (zyada weight)
USE_CHAR_NGRAMS = True                # [NAYA] True = character n-grams on. Training slow lage to False kar do
URL_RE = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", re.I)   # finds URLs in text
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")             # URLs that use a raw IP address
SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly")
SUSPICIOUS_TLDS = (".xyz", ".top", ".click", ".info", ".ru", ".tk", ".work", ".loan")
KEYWORDS = ("urgent", "verify", "suspend", "password", "account", "login", "confirm",
            "click", "bank", "security", "limited", "immediately", "winner", "prize",
            "invoice", "update", "alert", "unusual", "locked", "expire", "ssn", "refund")
# Well-known domains. A link counts as trusted only if it is https AND the host is exactly
# one of these (or a subdomain of it), so look-alikes such as amazon.com.evil.xyz do NOT match.
TRUSTED_DOMAINS = ("amazon.com", "google.com", "microsoft.com", "apple.com", "paypal.com",
                   "github.com", "linkedin.com", "facebook.com", "twitter.com", "x.com",
                   "youtube.com", "wikipedia.org", "dropbox.com", "zoom.us", "slack.com")

# ---------------------------------------------------------------------------
# [NAYA] Adversarial defence ke liye constants
# ---------------------------------------------------------------------------
# Leetspeak: attacker "password" ko "p@ssw0rd" likhta hai. Ye table wapas normal letters banati hai.
LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "@": "a"})

# Homoglyph: Cyrillic/Greek ke kuch letters bilkul English letters jaise dikhte hain
# (jaise Cyrillic 'а' aur English 'a'). Attacker inse "paypal" ko fake bana leta hai.
HOMOGLYPHS = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",   # Cyrillic a e o p c
    "\u0445": "x", "\u0443": "y", "\u0456": "i", "\u0458": "j", "\u0455": "s",   # Cyrillic x y i j s
    "\u0501": "d", "\u04bb": "h", "\u051b": "q", "\u051d": "w", "\u0475": "v",   # Cyrillic d h q w v
    "\u03bf": "o", "\u03b1": "a", "\u03bd": "v", "\u03c1": "p", "\u03b9": "i",   # Greek o a v p i
    "\u03ba": "k", "\u03c4": "t", "\u03c5": "u", "\u03b5": "e",                  # Greek k t u e
}
# Zero-width characters dikhte nahi hain, par words ke beech chhupa diye jaate hain ("ver\u200bify")
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")  # email address pehchanne ke liye (usme '@' ko leetspeak na samjhein)
# "v e r i f y" ya "v.e.r.i.f.y" jaisa spaced/dotted text pakadta hai (kam se kam 4 akshar)
SPACED_RE = re.compile(r"(?<![\w@$/])(?:[a-z0-9@$][ .\-_*]){3,}[a-z0-9@$](?![\w@$])", re.I)
# Leetspeak token: akshar-digit/symbol-akshar (p4ss, w0rd) ya token ki shuruaat digit se (4ccount)
LEET_TOKEN_RE = re.compile(r"[a-z][013457$@]+[a-z]|^[013457][a-z]{3,}", re.I)

# [NAYA] Explanation me hand features ke naam (hand_features() ke order me, total 15)
HAND_NAMES = ["number of links", "IP-address link", "shortened link", "suspicious domain ending",
              "insecure http link", "'@' inside link", "very long link", "many dots in link",
              "many digits in link", "suspicious keywords", "exclamation marks",
              "CAPITAL letters ratio", "email length", "look-alike characters in link",
              "disguised / obfuscated words"]


# ---------------------------------------------------------------------------
# Terminal styling (colors, bars, lines) - no extra libraries needed
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")  # let emoji/bars print on Windows too
except Exception:
    pass
if os.name == "nt":
    os.system("")  # turns on ANSI colors in the Windows terminal

_COLOR = sys.stdout.isatty()  # colors only when printing to a real terminal


def paint(text, code):
    """Wrap text in a color code (only if output is a terminal)."""
    return f"\033[{code}m{text}\033[0m" if _COLOR else str(text)


def green(t): return paint(t, "92")
def red(t): return paint(t, "91")
def yellow(t): return paint(t, "93")
def cyan(t): return paint(t, "96")
def bold(t): return paint(t, "1")
def dim(t): return paint(t, "2")


def bar(fraction, width=24, color=None):
    """Turn a value between 0 and 1 into a progress bar like ████████░░░░"""
    filled = int(round(max(0, min(1, fraction)) * width))
    s = "█" * filled + "░" * (width - filled)
    return color(s) if color else s


def mini_bar(fraction, width=10, color=None):
    """
    [NAYA] Patli, saaf bar: ■■■■■······ (bhara hua hissa rang me, khali hissa halke dots me).
    Block characters (█ ░) ki tarah ye upar-neeche chipakti nahi hai, isliye list saaf dikhti hai.
    Kam se kam 1 ■ hamesha dikhta hai, taaki chhoti value bhi nazar aaye.
    """
    filled = max(1, int(round(max(0, min(1, fraction)) * width)))
    on = "■" * filled
    return (color(on) if color else on) + dim("·" * (width - filled))


def line(char="═", width=62):
    """A horizontal divider line."""
    return char * width


def title(text):
    """Print a boxed section title."""
    print("\n" + cyan(line()))
    print(cyan("  " + bold(text)))
    print(cyan(line()))


# ---------------------------------------------------------------------------
# [NAYA] Adversarial defence: disguised (obfuscated) text ko normal banana
# ---------------------------------------------------------------------------
def _is_confusable(c):
    """[NAYA] True agar character Cyrillic/Greek ka hai (jo English jaisa dikh sakta hai)."""
    return ord(c) > 127 and unicodedata.name(c, "").startswith(("CYRILLIC", "GREEK"))


def _is_mixed_script(tok):
    """[NAYA] True agar ek hi word me English letters AUR Cyrillic/Greek letters mile hue hain (homoglyph trick)."""
    return any(c.isascii() and c.isalpha() for c in tok) and any(_is_confusable(c) for c in tok)


def _is_leet_token(tok):
    """[NAYA] True agar word leetspeak jaisa hai (p@ssw0rd, v3rify). Email address aur URL ko ignore karta hai."""
    return bool(LEET_TOKEN_RE.search(tok)) and not EMAIL_RE.search(tok) and not URL_RE.search(tok)


def _join_spaced(m):
    """[NAYA] 'v e r i f y' -> 'verify'. Sirf tab jodta hai jab usme koi akshar ho (numbers jaise '1 2 3 4' ko nahi chhedta)."""
    s = m.group()
    return re.sub(r"[ .\-_*]", "", s) if any(c.isalpha() for c in s) else s


def _fix_token(tok):
    """[NAYA] Ek word ko theek karta hai: pehle homoglyph, phir leetspeak."""
    if tok.isascii() and tok.isalpha():  # fast path: normal English word, kuch karna nahi
        return tok
    if EMAIL_RE.search(tok):             # email address ko jaisa hai waisa rehne do
        return tok
    if _is_mixed_script(tok):
        tok = "".join(HOMOGLYPHS.get(c, c) for c in tok)  # look-alike letters -> asli English letters
    if _is_leet_token(tok):
        tok = tok.translate(LEET_MAP)                      # p@ssw0rd -> password
    return tok


def normalize_text(text):
    """
    [NAYA] Attacker ke tricks hatata hai taaki model asli words dekh sake:
      1. NFKC: fullwidth / fancy unicode letters ko normal letters banata hai
      2. zero-width (invisible) characters hatata hai
      3. spaced letters jodta hai  (v e r i f y -> verify)
      4. look-alike letters theek karta hai (Cyrillic 'а' -> 'a')
      5. leetspeak theek karta hai (p@ssw0rd -> password)
    URLs ko chhua nahi jaata (unka alag check hota hai).
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = ZERO_WIDTH_RE.sub("", text)
    parts = URL_RE.split(text)  # URL_RE me ek group hai => [text, url, text, url, ...]
    for i in range(0, len(parts), 2):  # sirf text wale hisse (even index); URL (odd index) jaise hain waise
        seg = SPACED_RE.sub(_join_spaced, parts[i])
        parts[i] = re.sub(r"\S+", lambda m: _fix_token(m.group()), seg)
    return "".join(parts)


def count_obfuscation(text):
    """[NAYA] Email me kitne 'disguised' patterns hain (zero-width, spaced letters, leetspeak, look-alike words)."""
    text = str(text)
    n = len(ZERO_WIDTH_RE.findall(text))
    no_url = " ".join(URL_RE.split(text)[0::2])  # URLs hata kar sirf text
    n += sum(1 for m in SPACED_RE.finditer(no_url) if any(c.isalpha() for c in m.group()))
    for tok in no_url.split():
        if _is_mixed_script(tok) or _is_leet_token(tok):
            n += 1
    return n


def has_homoglyph_url(urls):
    """[NAYA] True agar kisi URL ke domain me non-English (look-alike) characters ya punycode ('xn--') hai."""
    for u in urls:
        h = host_of(u)
        if (not h.isascii()) or "xn--" in h:
            return True
    return False


# ---------------------------------------------------------------------------
# Feature engineering (inputs for the model)
# ---------------------------------------------------------------------------
def hand_features(texts):
    """
    Build 15 numeric features for every email:                       # [BADLAV] 13 se badhkar 15
    URL structure, keyword counts, writing-style and disguise signals.
    Returns a sparse matrix (rows = emails, columns = features).
    """
    rows = []
    for t in pd.Series(texts).fillna("").astype(str):
        urls = URL_RE.findall(t)   # all URLs in the email
        low = normalize_text(t).lower()   # [BADLAV] pehle sirf t.lower() tha; ab disguised words bhi keyword count me aate hain
        words = t.split()
        rows.append([
            len(urls),                                                   # 1. number of URLs
            int(bool(IP_URL_RE.search(t))),                              # 2. has an IP-address URL?
            int(any(s in u.lower() for u in urls for s in SHORTENERS)),  # 3. has a shortened link?
            int(any(u.lower().split("/")[2:3] and u.lower().split("/")[2].endswith(SUSPICIOUS_TLDS)
                    for u in urls if u.lower().startswith("http"))),     # 4. suspicious domain ending?
            int(any(u.lower().startswith("http://") for u in urls)),     # 5. insecure http:// link?
            int(any("@" in u for u in urls)),                            # 6. '@' inside a URL?
            max((len(u) for u in urls), default=0),                      # 7. length of the longest URL
            max((u.count(".") for u in urls), default=0),                # 8. dots in the URL (many subdomains)
            max((sum(c.isdigit() for c in u) for u in urls), default=0), # 9. digits in the URL
            sum(low.count(k) for k in KEYWORDS),                         # 10. suspicious keyword count
            t.count("!"),                                                # 11. exclamation marks
            sum(c.isupper() for c in t) / max(len(t), 1),                # 12. ratio of CAPITAL letters
            len(words),                                                  # 13. word count
            int(has_homoglyph_url(urls)),                                # 14. [NAYA] URL me look-alike characters?
            count_obfuscation(t),                                        # 15. [NAYA] disguised words ki ginti
        ])
    return csr_matrix(np.array(rows, dtype=float))


def host_of(url):
    """Return the host name of a URL, e.g. 'https://www.amazon.com/x' -> 'www.amazon.com'."""
    u = url if "://" in url else "http://" + url
    return u.split("/")[2].split("@")[-1].split(":")[0].lower()


def is_trusted(url):
    """True if the URL is https and its host is (a subdomain of) a well-known domain."""
    h = host_of(url)
    return url.lower().startswith("https://") and any(h == d or h.endswith("." + d) for d in TRUSTED_DOMAINS)


def clean_text(texts):
    """
    Lowercase the text and replace every URL with 'urltoken' (so TF-IDF learns wording, not domains).
    [BADLAV] Ab URL hatane ke baad normalize_text() bhi chalta hai, taaki p@ssw0rd / v e r i f y /
    look-alike letters jaise tricks normal words ban jayein.
    """
    out = []
    for t in pd.Series(texts).fillna(""):
        t = URL_RE.sub(" urltoken ", str(t))   # pehle URLs hatao
        out.append(normalize_text(t).lower())  # [NAYA] phir disguise hatao, phir lowercase
    return out


def build_pipeline(classifier):
    """
    Build the full ML pipeline:
      (a) TF-IDF on the email words
      (a2) [NAYA] TF-IDF on character n-grams (3-5 letters) - misspellings/obfuscation ke khilaaf madad
      (b) hand_features (URL/keyword numbers), scaled
      (c) the classifier on top (Logistic Regression / Random Forest)
    The same steps run automatically during training and prediction.
    """
    transformers = [
        ("tfidf", Pipeline([("clean", FunctionTransformer(clean_text)),
                            ("vec", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                                    min_df=2, max_features=20000, sublinear_tf=True))]),
         "text"),
    ]
    if USE_CHAR_NGRAMS:  # [NAYA] character n-gram branch
        transformers.append(
            ("tfidf_char", Pipeline([("clean", FunctionTransformer(clean_text)),
                                     ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                                             min_df=3, max_features=30000, sublinear_tf=True))]),
             "text"))
    transformers.append(
        ("hand", Pipeline([("feat", FunctionTransformer(hand_features)),
                           ("scale", StandardScaler(with_mean=False))]),
         "text"))
    features = ColumnTransformer(transformers)
    return Pipeline([("features", features), ("clf", classifier)])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path):
    """Read a CSV and return two columns: 'text' and 'label' (1 = phishing, 0 = safe)."""
    df = pd.read_csv(path)
    df = df.rename(columns={"Email Text": "text", "Email Type": "label", "body": "text"})
    df = df.dropna(subset=["text", "label"])  # drop empty rows
    lab = df["label"].astype(str).str.lower()
    df["label"] = lab.str.contains("phish|spam|1").astype(int)
    return df[["text", "label"]]


def make_synthetic(n=3000, seed=42):
    """
    Create a fake demo dataset when no real data is available.
    Some overlap is added on purpose (safe mails with links/URGENT, phishing mails without links).
    NOTE: real-world accuracy will be lower than on this demo data.
    """
    rnd = random.Random(seed)
    brands = ["PayPal", "Amazon", "Netflix", "Chase Bank", "Microsoft", "Apple", "DHL", "Google"]
    bad_domains = ["secure-{b}-login.xyz", "{b}-verify.top", "account-{b}.click", "{b}.support-center.ru",
                   "192.168.{x}.{y}", "bit.ly/{r}", "tinyurl.com/{r}"]
    good_domains = ["www.{b}.com", "docs.{b}.com", "github.com/{r}", "www.wikipedia.org/wiki/{r}",
                    "calendar.google.com", "meet.company.com/{r}"]
    phish_t = [
        "URGENT: Your {B} account has been suspended. Verify your password immediately: {u}",
        "Security alert! Unusual login detected on your {B} account. Confirm your identity here {u}",
        "Dear customer, your payment of ${n} failed. Update your billing details now to avoid closure. {u}",
        "Congratulations! You are the winner of a ${n} {B} gift card. Click {u} to claim your prize before it expires",
        "Your invoice #{n} is overdue. Open the attachment or log in at {u} to avoid penalties",
        "Final notice: your mailbox is almost full. Login to {u} to keep your emails",
        "We noticed suspicious activity. Your account is locked. Reset your password immediately",
        "Refund of ${n} pending. Provide your bank account and SSN to receive the transfer",
    ]
    safe_t = [
        "Hi team, the meeting notes from Tuesday are attached. Let me know if I missed anything.",
        "Your {B} order #{n} has shipped and will arrive on Friday. Track it at {u}",
        "Reminder: quarterly review is scheduled for next Monday at 10am. Agenda: {u}",
        "Thanks for the great lunch yesterday! Let's catch up again soon.",
        "Here is the draft of the project proposal. Please review and send comments by Thursday.",
        "Your {B} receipt for ${n}. No action is needed. View it any time at {u}",
        "Please confirm your attendance for the workshop by replying to this email. Details: {u}",
        "Security tip of the month from IT: update your software regularly. Full guide at {u}",
        "Can you update the spreadsheet with the latest numbers before the call? Thanks!",
    ]
    fill = ["Regards,", "Thanks,", "Best,", "Sincerely,", "Cheers,"]

    def url(domains, bad):
        """Build a random URL from a domain template (phishing links use http:// more often)."""
        d = rnd.choice(domains).format(b=rnd.choice(brands).lower().replace(" ", ""),
                                       x=rnd.randint(1, 250), y=rnd.randint(1, 250),
                                       r="".join(rnd.choices("abcdefg1234567", k=6)))
        return ("http://" if bad and rnd.random() < .7 else "https://") + d

    def make(templates, domains, bad):
        """Pick a template, fill in the blanks and return one email text."""
        t = rnd.choice(templates)
        body = t.format(B=rnd.choice(brands), n=rnd.randint(10, 9999), u=url(domains, bad))
        if bad and rnd.random() < .15:        # remove the link from 15% of phishing mails
            body = URL_RE.sub("", body)
        if not bad and rnd.random() < .10:    # add URGENT to 10% of safe mails
            body = "URGENT: " + body
        return body + " " + rnd.choice(fill)

    data = [(make(phish_t, bad_domains, True), 1) for _ in range(n // 2)]
    data += [(make(safe_t, good_domains, False), 0) for _ in range(n // 2)]
    rnd.shuffle(data)
    return pd.DataFrame(data, columns=["text", "label"])


# ---------------------------------------------------------------------------
# [NAYA] Feedback loop: galat verdicts ko save karna aur training me use karna
# ---------------------------------------------------------------------------
def save_feedback(text, label):
    """[NAYA] Ek corrected email (label: 1 = phishing, 0 = safe) feedback.csv me jodta hai."""
    new_file = not os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["text", "label"])  # pehli baar header likho
        w.writerow([text, int(label)])


def load_feedback():
    """[NAYA] feedback.csv padhta hai. File na ho ya khali ho to None deta hai."""
    if not os.path.exists(FEEDBACK_PATH):
        return None
    try:
        df = pd.read_csv(FEEDBACK_PATH).dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(int)
    except Exception:
        return None
    return df[["text", "label"]] if len(df) else None


def ask_feedback(text):
    """[NAYA] Scan ke baad user se poochta hai ki verdict sahi tha ya nahi. Galat ho to correct label save karta hai."""
    try:
        ans = input(cyan("\n  Was this verdict correct? (y = yes / n = no / Enter = skip): ")).strip().lower()
    except EOFError:
        return
    if ans == "y":
        print(dim("  Thanks for confirming!"))
        return
    if ans != "n":
        return
    try:
        real = input(cyan("  What is this email really? (p = phishing / s = safe): ")).strip().lower()
    except EOFError:
        return
    if real in ("p", "s"):
        save_feedback(text, 1 if real == "p" else 0)
        print(green(f"  Saved to {FEEDBACK_PATH}. It will be used the next time you train the model."))
    else:
        print(dim("  Not saved (please type p or s)."))


# ---------------------------------------------------------------------------
# Training + friendly report
# ---------------------------------------------------------------------------
def train(df, feedback_df=None):   # [BADLAV] feedback_df parameter naya hai
    """
    Split the data 80/20, train two models, keep the better one,
    and print an easy-to-read report. Also saves the model and a confusion-matrix image.
    [NAYA] feedback_df: user ke corrected emails. Ye SIRF training set me judte hain (test set me nahi),
    taaki test accuracy fake na badhe (data leakage na ho).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"]], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

    n_feedback = 0
    if feedback_df is not None and len(feedback_df):   # [NAYA] feedback ko training set me jodo
        n_feedback = len(feedback_df)
        fb = pd.concat([feedback_df[["text", "label"]]] * FEEDBACK_REPEAT, ignore_index=True)
        X_train = pd.concat([X_train, fb[["text"]]], ignore_index=True)
        y_train = pd.concat([y_train.reset_index(drop=True), fb["label"]], ignore_index=True)

    # [NOTE] class_weight="balanced" imbalance (phishing kam / safe zyada) ko handle karta hai.
    # Ye pehle se code me tha. Isse chhoti class ko zyada importance milti hai.
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                                class_weight="balanced"),
    }
    print(dim("\nTraining started... (this may take a while on big datasets)"))
    results, best_name, best_pipe, best_acc = {}, None, None, -1
    for name, clf in candidates.items():
        print(dim(f"  - training {name}..."))
        pipe = build_pipeline(clf).fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        results[name] = acc
        if acc > best_acc:  # remember the best model
            best_name, best_pipe, best_acc = name, pipe, acc

    preds = best_pipe.predict(X_test)
    (tn, fp), (fn, tp) = confusion_matrix(y_test, preds)
    n_safe, n_phish = tn + fp, fn + tp
    correct = tn + tp
    pad = 9 if _COLOR else 0  # extra characters that color codes add (for column alignment)

    # ---- Report ----
    title("TRAINING REPORT")
    print(f"  Total emails : {len(df):,}   ({red('Phishing ' + format(int(df.label.sum()), ','))}"
          f"  |  {green('Safe ' + format(int((df.label == 0).sum()), ','))})")
    ph_share = df.label.mean()   # [NAYA] class balance dikhao (imbalance ka jawab)
    print(f"  Class balance: {ph_share:.0%} phishing / {1 - ph_share:.0%} safe  "
          f"{dim('(handled with class_weight=balanced)')}")
    print(f"  Used to train: {len(X_train):,} emails")
    if n_feedback:  # [NAYA]
        print(f"  Feedback used: {n_feedback:,} corrected email(s) from {FEEDBACK_PATH} "
              f"{dim(f'(each repeated {FEEDBACK_REPEAT}x, training set only)')}")
    print(f"  Used to test : {len(X_test):,} emails  {dim('(never seen by the model during training)')}")

    print("\n" + bold("  Model comparison"))
    for name, acc in results.items():
        crown = "  <- best" if name == best_name else ""
        print(f"    {name:<20} {bar(acc, 24, green if name == best_name else None)} {acc:.2%}{crown}")

    print("\n" + bold("  Final score"))
    print(f"    Accuracy: {bold(green(f'{best_acc:.2%}'))}  -> about {round(best_acc * 100)} "
          f"out of every 100 emails are classified correctly")

    # [NAYA] Precision / Recall / F1: classes unequal ho to akele accuracy par bharosa nahi karte
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, labels=[0, 1], zero_division=0)
    print("\n" + bold("  Precision / Recall / F1  (more honest than accuracy when classes are unequal)"))
    print(f"    Safe     : precision {prec[0]:.2f}   recall {rec[0]:.2f}   F1 {f1[0]:.2f}")
    print(f"    Phishing : precision {prec[1]:.2f}   recall {rec[1]:.2f}   F1 {f1[1]:.2f}")

    print("\n" + bold("  Where is it right and wrong? (Confusion matrix)"))
    print(f"    {'':<20}{'Model said: SAFE':>22}{'Model said: PHISHING':>26}")
    print(f"    {'Actually SAFE':<20}{green(f'OK {tn:,}'):>{22 + pad}}"
          f"{yellow(f'X {fp:,} (false alarm)'):>{26 + pad}}")
    print(f"    {'Actually PHISHING':<20}{red(f'X {fn:,} (missed)'):>{22 + pad}}"
          f"{green(f'OK {tp:,}'):>{26 + pad}}")

    print("\n" + bold("  In plain words"))
    print(f"    - Of {n_phish:,} real phishing emails, the model caught {green(f'{tp:,}')} "
          f"({tp / n_phish:.1%}) and missed {red(f'{fn:,}')}.")
    print(f"    - Of {n_safe:,} real safe emails, it correctly allowed {green(f'{tn:,}')} "
          f"({tn / n_safe:.1%}) and wrongly flagged {yellow(f'{fp:,}')} as phishing.")
    print(f"    - Total: {correct:,} correct and {fp + fn:,} wrong predictions.")

    # Save the confusion-matrix image
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds),
                           display_labels=["Safe", "Phishing"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{best_name}\nAccuracy: {best_acc:.2%}")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    joblib.dump(best_pipe, MODEL_PATH)  # save the model so we don't have to retrain every time
    print(f"\n  Model saved: {bold(MODEL_PATH)}    Chart saved: {bold('confusion_matrix.png')}")
    print(cyan(line()))
    return best_pipe


# ---------------------------------------------------------------------------
# Prediction + explanation (why is it phishing?)
# ---------------------------------------------------------------------------
def predict(text, model=None):
    """Return ("Phishing" or "Safe", phishing probability between 0 and 1)."""
    model = model or joblib.load(MODEL_PATH)
    proba = model.predict_proba(pd.DataFrame({"text": [text]}))[0][1]  # [1] = phishing class
    return ("Phishing" if proba >= 0.5 else "Safe"), proba


def explain_model(model, text, top_n=5):
    """
    [NAYA] EXPLAINABILITY: ML model ne kin words/features ki wajah se ye score diya?
    Logistic Regression me har feature ka ek weight (coefficient) hota hai.
    Contribution = (is email me feature ki value) x (uska weight).
        positive  -> email ko PHISHING ki taraf dhakelta hai
        negative  -> email ko SAFE ki taraf dhakelta hai
    [BADLAV] Ab result 2 alag groups me aata hai (pehle sab ek hi list me milaa hota tha):
        "words" -> email ke asli words (verify, prize, gift card ...)
        "style" -> link / style / disguise signals (email length, exclamation marks, many dots in link ...)
    Dono groups ka apna alag (phishing_list, safe_list) hota hai, taaki bade numbers wale style features
    words ko dabaa na dein.
    Returns {"words": (phish, safe), "style": (phish, safe)} ya None (Random Forest me coefficients nahi hote).
    """
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    feats = model.named_steps["features"]
    X = feats.transform(pd.DataFrame({"text": [text]}))
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    contrib = X[0] * clf.coef_[0]

    # Har column ka naam banao (ColumnTransformer ke order me: word -> char -> hand)
    names = []
    for name, trans, _ in feats.transformers_:
        if name == "tfidf":
            names += list(trans.named_steps["vec"].get_feature_names_out())
        elif name == "tfidf_char":
            names += ["chr:" + n for n in trans.named_steps["vec"].get_feature_names_out()]
        elif name == "hand":
            names += HAND_NAMES
    if len(names) != len(contrib):
        return None

    style_names = set(HAND_NAMES)
    order = [i for i in np.argsort(contrib) if not names[i].startswith("chr:")]   # char n-grams hide

    def pick(is_style, n_phish, n_safe):
        """Ek group (words ya style) me se top phishing aur top safe contributors chuno."""
        group = [i for i in order if (names[i] in style_names) == is_style]
        phish = [(names[i], float(contrib[i])) for i in reversed(group) if contrib[i] > 0.01][:n_phish]
        safe = [(names[i], float(contrib[i])) for i in group if contrib[i] < -0.01][:n_safe]
        return phish, safe

    return {"words": pick(False, top_n, 3), "style": pick(True, 3, 3)}


def explain(text):
    """Return (red flags, good signs, rule-based score 0-1, all-links-trusted?) for an email."""
    flags, good, score = [], [], 0.0

    def flag(msg, weight):
        nonlocal score
        flags.append(msg)
        score += weight  # every red flag has its own weight

    low = normalize_text(text).lower()   # [BADLAV] pehle text.lower() tha; ab disguised keywords bhi pakde jaate hain
    urls = URL_RE.findall(text)
    https_urls = [u for u in urls if u.lower().startswith("https://")]
    trusted = bool(urls) and all(is_trusted(u) for u in urls)
    if trusted:
        good.append(f"All links use https and go to a well-known domain ({host_of(urls[0])})")

    if urls:
        flag(f"Contains {len(urls)} link(s): {urls[0][:55]}{'...' if len(urls[0]) > 55 else ''}", 0.05)
    else:
        good.append("No links in the email")
    if IP_URL_RE.search(text):
        flag("Link uses a raw IP address instead of a domain (very suspicious)", 0.5)
    if any(s in u.lower() for u in urls for s in SHORTENERS):
        flag("Shortened link (like bit.ly) hides the real destination", 0.3)
    if any(u.lower().startswith("http") and u.lower().split("/")[2].endswith(SUSPICIOUS_TLDS) for u in urls):
        flag("Link has a suspicious domain ending (.xyz / .top / .ru etc.)", 0.3)
    if any(u.lower().startswith("http://") for u in urls):
        flag("Link is not secure (http instead of https)", 0.15)
    if any("@" in u for u in urls):
        flag("Link contains '@' (a trick to hide the real site)", 0.3)
    if any(len(u) > 75 for u in urls):
        flag("Link is unusually long", 0.1)
    if has_homoglyph_url(urls):   # [NAYA] look-alike domain check
        flag("Link domain contains look-alike / non-English characters (homoglyph trick)", 0.5)
    if urls and len(https_urls) == len(urls) and len(flags) == 1 and not trusted:
        good.append("Link is secure (https)")

    n_obf = count_obfuscation(text)   # [NAYA] disguised words ka check
    if n_obf:
        flag(f"Disguised text found ({n_obf}x): e.g. p@ssw0rd, v e r i f y, hidden or look-alike letters", 0.3)

    found = [k for k in KEYWORDS if k in low]
    if len(found) >= 2:
        flag("Suspicious words found: " + ", ".join(found[:7]), 0.15)
    elif not found:
        good.append("No pressure or suspicious words found")
    if any(w in low for w in ("urgent", "immediately", "expire", "final notice", "within 24")):
        flag("Creates urgency to make you act fast", 0.15)
    if text.count("!") >= 3:
        flag(f"Too many exclamation marks ({text.count('!')})", 0.05)
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 30 and sum(c.isupper() for c in letters) / len(letters) > 0.3:
        flag("Too many CAPITAL letters (shouting style)", 0.05)
    if any(w in low for w in ("password", "ssn", "bank account", "credit card", "otp")):
        flag("Asks for sensitive information (password / bank / SSN)", 0.25)
    return flags, good, min(score, 1.0), trusted


def score_email(text, model):
    """
    [NAYA] Final risk score nikalne ka kaam ab ek jagah hai (pehle show_result ke andar tha).
    Isse show_result aur adversarial_test dono same logic use karte hain.
    Returns (p, p_model, rule, trusted, disagree, flags, good).
    """
    _, p_model = predict(text, model)
    flags, good, rule, trusted = explain(text)
    # Hybrid score: rules can RAISE the risk to cover model mistakes
    p = max(p_model, 0.9 * rule)
    # Agreement check: if the ML model screams "phishing" but the rule check finds nothing technical
    # (normal links, no sensitive requests), the model may simply be unfamiliar with this kind of
    # email. Don't claim certainty - downgrade to SUSPICIOUS and ask the user to verify.
    disagree = p_model >= 0.65 and rule < 0.25
    if disagree:
        p = min(p, 0.40 if trusted else 0.55)
    return p, p_model, rule, trusted, disagree, flags, good


def show_result(text, model):
    """Show the verdict, risk meter, reasons and advice for one email. Returns the final risk (0-1)."""
    p, p_model, rule, trusted, disagree, flags, good = score_email(text, model)   # [BADLAV] logic score_email() me gaya

    if p >= 0.65:
        verdict, icon, col = "PHISHING - DANGEROUS", "[!!]", red
        advice = "Do not click any links, open attachments or share information. Report or delete this email."
    elif p >= 0.35:
        verdict, icon, col = "SUSPICIOUS - BE CAREFUL", "[!]", yellow
        advice = "Not certain. Verify the sender through another channel (phone call / official website)."
    else:
        verdict, icon, col = "SAFE - LOOKS FINE", "[OK]", green
        advice = "No major threat found, but still be careful with links from unknown senders."

    print("\n" + col(line("━")))
    print(f"  {icon} {col(bold('VERDICT: ' + verdict))}")
    print(f"  Phishing risk : {bar(p, 24, col)} {col(bold(f'{p:.0%}'))}")
    print(dim(f"  (ML model: {p_model:.0%}  |  Rule check: {rule:.0%})"))
    print(col(line("━")))
    if disagree:
        print(yellow("\n  Note: the ML model suspects phishing, but the rule check found no technical red flags"
                     + (" and all links go to well-known domains" if trusted else "")
                     + ".\n  Rated SUSPICIOUS instead of PHISHING - please verify the sender manually."))

    if flags:
        print(bold("\n  Things to notice (red flags):"))
        for f in flags:
            print(f"     {red('*')} {f}")
    if good:
        print(bold("\n  Good signs:"))
        for g in good:
            print(f"     {green('*')} {g}")

    # [NAYA] ML model ki explanation: kin words/features ne score ko prabhavit kiya
    try:
        reasons = explain_model(model, text)
    except Exception:
        reasons = None
    if reasons is None:
        print(dim("\n  (Word-level explanation is available only when Logistic Regression is the saved model.)"))
    else:
        # [BADLAV] Ab "Top words" aur "Style & link signals" alag sections me dikhte hain.
        # Har section apni khud ki scale par (bar ki lambai apne section ke sabse bade number se tay hoti hai).
        print(bold("\n  Why the ML model said so:"))

        def show_group(heading, group):
            pro_phish, pro_safe = group
            if not (pro_phish or pro_safe):
                return
            print("\n   " + cyan(bold(heading)))
            top = max(abs(v) for _, v in pro_phish + pro_safe)   # is section ki scale
            for name, v in pro_phish:
                print(f"     {red('▲')} {name:<28} {mini_bar(abs(v) / top, 10, red)}  {red(f'{v:+.2f}')}  {dim('phishing')}")
            for name, v in pro_safe:
                print(f"     {green('▼')} {name:<28} {mini_bar(abs(v) / top, 10, green)}  {green(f'{v:+.2f}')}  {dim('safe')}")

        show_group("Top words", reasons["words"])
        show_group("Style & link signals", reasons["style"])

    print(f"\n  {bold('Advice:')} {advice}")
    print(dim("\n  Note: this is a machine-learning estimate, not a 100% guarantee. "
              "If in doubt, confirm with your IT team or bank."))
    return p   # [NAYA] risk wapas karte hain (zaroorat pade to use ho sake)


# ---------------------------------------------------------------------------
# [NAYA] Adversarial test: tool ko disguised (obfuscated) emails par parakhna
# ---------------------------------------------------------------------------
ADV_SAMPLES = [   # [NAYA] 5 sample phishing emails (inko alag-alag tareeke se disguise karke test hota hai)
    "URGENT: Your PayPal account has been suspended. Verify your password immediately: http://secure-paypal-login.xyz/verify",
    "Security alert! Unusual login detected on your Amazon account. Confirm your identity here http://amazon-verify.top/login",
    "Congratulations! You are the winner of a $500 gift card. Click http://bit.ly/a1b2c3 to claim your prize before it expires",
    "Your invoice is overdue. Login to http://account-support.click/pay and provide your bank account to avoid penalties",
    "Final notice: your mailbox is almost full. Reset your password immediately at http://mail-update.ru/reset",
]
LEET_ENCODE = str.maketrans("aeiostAEIOST", "431057431057")        # a->4 e->3 i->1 o->0 s->5 t->7
HOMO_ENCODE = str.maketrans({"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441"})  # English -> Cyrillic look-alike
ATTACK_WORDS = ("urgent", "verify", "password", "account", "login", "security", "bank", "confirm")


def _outside_urls(text, fn):
    """[NAYA] fn ko sirf text par lagata hai, URLs ko nahi chhedta."""
    parts = URL_RE.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = fn(parts[i])
    return "".join(parts)


def obf_leet(t):
    """[NAYA] Attack 1: leetspeak (URGENT -> URG3N7)."""
    return _outside_urls(t, lambda s: s.translate(LEET_ENCODE))


def obf_spaced(t):
    """[NAYA] Attack 2: keywords me akshar alag-alag (verify -> v e r i f y)."""
    def fn(s):
        for w in ATTACK_WORDS:
            s = re.sub(w, lambda m: " ".join(m.group()), s, flags=re.I)
        return s
    return _outside_urls(t, fn)


def obf_homoglyph(t):
    """[NAYA] Attack 3: English letters ki jagah Cyrillic look-alike (text me bhi aur URL ke domain me bhi)."""
    parts = URL_RE.split(t)
    for i, p in enumerate(parts):
        if i % 2 == 0:
            parts[i] = p.translate(HOMO_ENCODE)
        else:
            head, sep, rest = p.partition("://")   # 'http://' ko na chhedo, sirf domain badlo
            parts[i] = (head + sep + rest.translate(HOMO_ENCODE)) if sep else p.translate(HOMO_ENCODE)
    return "".join(parts)


def obf_zerowidth(t):
    """[NAYA] Attack 4: keywords ke beech invisible zero-width character."""
    def fn(s):
        for w in ATTACK_WORDS:
            s = re.sub(w, lambda m: m.group()[0] + "\u200b" + m.group()[1:], s, flags=re.I)
        return s
    return _outside_urls(t, fn)


def adversarial_test(model):
    """
    [NAYA] Har attack type ke liye dekhta hai ki tool kitni phishing emails pakad pata hai.
      ML only      = sirf ML model ka score >= 50%
      Full tool    = ML + rules ka final risk >= 65% (yani verdict 'PHISHING')
    """
    title("ADVERSARIAL TEST")
    attacks = [("Original (no disguise)", lambda t: t), ("Leetspeak (p4ssw0rd)", obf_leet),
               ("Spaced letters (v e r i f y)", obf_spaced), ("Look-alike letters (Cyrillic)", obf_homoglyph),
               ("Zero-width characters", obf_zerowidth)]
    total = len(ADV_SAMPLES)
    print(f"  {len(ADV_SAMPLES)} phishing emails x {len(attacks)} disguises. Higher = better.\n")
    print(f"  {'Attack':<32}{'ML only':>10}{'Full tool':>12}")
    for name, fn in attacks:
        ml_hits = full_hits = 0
        for s in ADV_SAMPLES:
            p, p_model, *_ = score_email(fn(s), model)
            ml_hits += p_model >= 0.5
            full_hits += p >= 0.65
        col = green if full_hits == total else (yellow if full_hits >= total - 1 else red)
        print(f"  {name:<32}{f'{ml_hits}/{total}':>10}{col(f'{full_hits}/{total}'):>{12 + (9 if _COLOR else 0)}}")
    print(dim("\n  Tip: In cases where the 'full tool' proves effective, include examples of that attack in the `feedback.csv` file and retrain the model."))


# ---------------------------------------------------------------------------
# Interactive screens
# ---------------------------------------------------------------------------
def read_multiline():
    """Read a multi-line email from the user. Type END on its own line to finish."""
    print(dim("  Paste the email, then type  END  on a new line and press Enter:"))
    lines = []
    while True:
        try:
            ln = input()
        except EOFError:
            break
        if ln.strip().upper() == "END":
            break
        lines.append(ln)
    return "\n".join(lines).strip()


def _load_checked(path):
    """
    [NAYA] Saved model load karta hai aur check karta hai ki wo naye code ke saath chalta hai ya nahi.
    Features badalne ki wajah se purana model fail ho sakta hai - tab user ko dobara train karne ko kehta hai.
    """
    model = joblib.load(path)
    try:
        model.predict_proba(pd.DataFrame({"text": ["test email"]}))
        return model
    except Exception:
        print(yellow("\n  The saved model was made with an older version of this tool."
                     "\n  Please train again (menu option 2)."))
        return None


def load_model_or_none():
    """Load the saved model, or tell the user to train first."""
    if os.path.exists(MODEL_PATH):
        return _load_checked(MODEL_PATH)   # [BADLAV] seedha joblib.load ki jagah _load_checked
    print(yellow(f"\n  Saved model ({MODEL_PATH}) not found. Please train first (menu option 2)."))
    return None


def scan_loop(model):
    """Keep scanning emails until the user says stop."""
    title("EMAIL SCANNER")
    while True:
        text = read_multiline()
        if not text:
            print(yellow("  Nothing was pasted."))
        else:
            show_result(text, model)
            ask_feedback(text)   # [NAYA] verdict sahi tha ya galat? (feedback loop)
        try:
            again = input(cyan("\n  Scan another email? (y/n): ")).strip().lower()
        except EOFError:
            break
        if again != "y":
            break


def _clean_path(raw):
    """
    [NAYA] User ke type/paste kiye path ko saaf karta hai:
      - aage-peeche ke spaces, double aur single quotes hatata hai ("C:\\x\\a.csv" ya 'C:\\x\\a.csv')
      - PowerShell me drag-drop se aane wala "& 'path'" format handle karta hai
      - ~ (home folder) ko asli path me badalta hai
    """
    p = raw.strip()
    if p.startswith("& "):
        p = p[2:].strip()
    p = p.strip().strip('"').strip("'").strip()
    if os.name != "nt":   # Linux/Mac terminal drag-drop me space aise aata hai: my\\ file.csv
        p = p.replace("\\ ", " ")
    return os.path.expanduser(p)


def _find_file(path):
    """
    [NAYA] File dhoondhta hai: pehle jaisa diya waisa, phir current folder me, phir script wale folder me.
    Isse sirf "Phishing_Email.csv" likhne par bhi file mil jaati hai (agar wo project folder me hai).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in (path, os.path.join(os.getcwd(), path), os.path.join(script_dir, path)):
        if os.path.isfile(candidate):
            return candidate
    return None


def _browse_for_csv():
    """[NAYA] File chunne ki window kholta hai (tkinter). Cancel karne par '' deta hai."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)   # window sabse upar aaye
        chosen = filedialog.askopenfilename(title="Select your CSV file",
                                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        root.destroy()
        return chosen or ""
    except Exception:
        print(yellow("  The file picker is not available on this system. Please type the path instead."))
        return ""


def _csv_hints():
    """[NAYA] Current aur script folder ki .csv files ki list (path galat hone par madad ke liye)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    found = []
    for folder in dict.fromkeys([os.getcwd(), script_dir]):
        try:
            found += [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".csv") and f != FEEDBACK_PATH]   # feedback.csv ko list me nahi dikhate
        except OSError:
            pass
    return found


def train_flow():
    """
    Menu option 'Train': asks for a CSV path.
    [BADLAV] Ab 3 tareeke hain: path type/paste karo, 'browse' likho (file window), ya Enter dabao (demo data).
    """
    raw = input(cyan("\n  Path to your CSV file\n"
                     "  (type 'browse' to pick a file, or press Enter for demo data): ")).strip()
    if raw.lower() == "browse":
        raw = _browse_for_csv()
        if not raw:
            print(yellow("  No file selected."))
            return None
    path = _clean_path(raw)
    if path:
        real = _find_file(path)
        if real is None:
            print(red(f"  File not found: {path}"))
            hints = _csv_hints()
            if hints:
                print(yellow("  CSV files found near this script (copy one of these paths):"))
                for h in hints[:5]:
                    print(f"     {h}")
            else:
                print(yellow("  Tip: put the CSV in the same folder as this script and type just its name, "
                             "or type 'browse'."))
            return None
        df = load_csv(real)
    else:
        print(yellow("  Using demo (fake) data - provide a real CSV for real accuracy."))
        df = make_synthetic()
    return train(df, load_feedback())   # [BADLAV] feedback bhi training me pass hota hai


def menu():
    """Main menu."""
    model = _load_checked(MODEL_PATH) if os.path.exists(MODEL_PATH) else None   # [BADLAV] _load_checked
    while True:
        title("PHISHING EMAIL DETECTOR")
        status = green("Ready") if model else yellow("not trained yet")
        print(f"  Model status: {status}\n")
        print("   1) Scan an email")
        print("   2) Train the model (on your CSV + saved feedback)")   # [BADLAV] text badla
        print("   3) Adversarial test (disguised emails)")              # [NAYA]
        print("   4) Exit")                                             # [BADLAV] pehle option 3 tha
        try:
            choice = input(cyan("\n  Choose an option (1/2/3/4): ")).strip()
        except EOFError:
            break
        if choice == "1":
            if model is None:
                model = load_model_or_none()
            if model:
                scan_loop(model)
        elif choice == "2":
            model = train_flow() or model
        elif choice == "3":   # [NAYA]
            if model is None:
                model = load_model_or_none()
            if model:
                adversarial_test(model)
        elif choice == "4":
            print(green("\n  Thank you! Stay safe.\n"))
            break
        else:
            print(red("  Invalid option. Please type 1, 2, 3 or 4."))


# ---------------------------------------------------------------------------
# Entry point (the program starts here when run from the command line)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phishing Email Detector")
    ap.add_argument("--data", help="path to a labeled CSV (to train the model)")
    ap.add_argument("--predict", help="email text to scan")
    ap.add_argument("--file", help="path to a text file containing an email")
    ap.add_argument("--interactive", action="store_true", help="open the email scanner")
    ap.add_argument("--adversarial-test", action="store_true",
                    help="test the saved model against disguised (obfuscated) phishing emails")  # [NAYA]
    args = ap.parse_args()

    if args.predict or args.file:
        model = load_model_or_none()
        if model:
            if args.file:
                with open(args.file, encoding="utf-8", errors="ignore") as f:
                    show_result(f.read(), model)
            else:
                show_result(args.predict, model)
    elif args.adversarial_test:   # [NAYA]
        m = load_model_or_none()
        if m:
            adversarial_test(m)
    elif args.interactive:
        m = load_model_or_none()
        if m:
            scan_loop(m)
    elif args.data:
        trained = train(load_csv(args.data), load_feedback())   # [BADLAV] feedback bhi pass hota hai
        if sys.stdin.isatty():
            ans = input(cyan("\n  Do you want to scan an email now? (y/n): ")).strip().lower()
            if ans == "y":
                scan_loop(trained)
        else:
            print(dim("  Tip: use --interactive to scan emails."))
    else:
        menu()
