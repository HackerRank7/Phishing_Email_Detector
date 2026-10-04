"""
Phishing Email Detector
-----------------------
How to run:
    python phishing_detector.py                       # opens the menu (easiest)
    python phishing_detector.py --data emails.csv     # train on your own CSV
    python phishing_detector.py --interactive         # scan emails with the saved model
    python phishing_detector.py --predict "email text"
    python phishing_detector.py --file email.txt      # scan an email stored in a text file

CSV format (both work):
    columns: text,label                  (label: 1/0, phishing/safe)
    columns: Email Text,Email Type       (Kaggle "Phishing_Email.csv" format)
"""
import argparse
import os
import random
import re
import sys

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
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "phishing_model.joblib"  # the trained model is saved here
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


def line(char="═", width=62):
    """A horizontal divider line."""
    return char * width


def title(text):
    """Print a boxed section title."""
    print("\n" + cyan(line()))
    print(cyan("  " + bold(text)))
    print(cyan(line()))


# ---------------------------------------------------------------------------
# Feature engineering (inputs for the model)
# ---------------------------------------------------------------------------
def hand_features(texts):
    """
    Build 13 numeric features for every email:
    URL structure, keyword counts and writing-style signals.
    Returns a sparse matrix (rows = emails, columns = features).
    """
    rows = []
    for t in pd.Series(texts).fillna("").astype(str):
        urls = URL_RE.findall(t)   # all URLs in the email
        low = t.lower()            # lowercase copy for keyword matching
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
    """Lowercase the text and replace every URL with 'urltoken' (so TF-IDF learns wording, not domains)."""
    return [URL_RE.sub(" urltoken ", str(t).lower()) for t in pd.Series(texts).fillna("")]


def build_pipeline(classifier):
    """
    Build the full ML pipeline:
      (a) TF-IDF on the email words
      (b) hand_features (URL/keyword numbers), scaled
      (c) the classifier on top (Logistic Regression / Random Forest)
    The same steps run automatically during training and prediction.
    """
    features = ColumnTransformer([
        ("tfidf", Pipeline([("clean", FunctionTransformer(clean_text)),
                            ("vec", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                                    min_df=2, max_features=20000, sublinear_tf=True))]),
         "text"),
        ("hand", Pipeline([("feat", FunctionTransformer(hand_features)),
                           ("scale", StandardScaler(with_mean=False))]),
         "text"),
    ])
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
# Training + friendly report
# ---------------------------------------------------------------------------
def train(df):
    """
    Split the data 80/20, train two models, keep the better one,
    and print an easy-to-read report. Also saves the model and a confusion-matrix image.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"]], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

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
    print(f"  Used to train: {len(X_train):,} emails")
    print(f"  Used to test : {len(X_test):,} emails  {dim('(never seen by the model during training)')}")

    print("\n" + bold("  Model comparison"))
    for name, acc in results.items():
        crown = "  <- best" if name == best_name else ""
        print(f"    {name:<20} {bar(acc, 24, green if name == best_name else None)} {acc:.2%}{crown}")

    print("\n" + bold("  Final score"))
    print(f"    Accuracy: {bold(green(f'{best_acc:.2%}'))}  -> about {round(best_acc * 100)} "
          f"out of every 100 emails are classified correctly")

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


def explain(text):
    """Return (red flags, good signs, rule-based score 0-1, all-links-trusted?) for an email."""
    flags, good, score = [], [], 0.0

    def flag(msg, weight):
        nonlocal score
        flags.append(msg)
        score += weight  # every red flag has its own weight

    low = text.lower()
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
    if urls and len(https_urls) == len(urls) and len(flags) == 1 and not trusted:
        good.append("Link is secure (https)")

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


def show_result(text, model):
    """Show the verdict, risk meter, reasons and advice for one email."""
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
    print(f"\n  {bold('Advice:')} {advice}")
    print(dim("\n  Note: this is a machine-learning estimate, not a 100% guarantee. "
              "If in doubt, confirm with your IT team or bank."))


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


def load_model_or_none():
    """Load the saved model, or tell the user to train first."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
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
        try:
            again = input(cyan("\n  Scan another email? (y/n): ")).strip().lower()
        except EOFError:
            break
        if again != "y":
            break


def train_flow():
    """Menu option 'Train': asks for a CSV path (press Enter to use demo data)."""
    path = input(cyan("\n  Path to your CSV file (leave empty for demo data): ")).strip().strip('"')
    if path:
        if not os.path.exists(path):
            print(red(f"  File not found: {path}"))
            return None
        df = load_csv(path)
    else:
        print(yellow("  Using demo (fake) data - provide a real CSV for real accuracy."))
        df = make_synthetic()
    return train(df)


def menu():
    """Main menu."""
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    while True:
        title("PHISHING EMAIL DETECTOR")
        status = green("Ready") if model else yellow("not trained yet")
        print(f"  Model status: {status}\n")
        print("   1) Scan an email")
        print("   2) Train the model (on your CSV)")
        print("   3) Exit")
        try:
            choice = input(cyan("\n  Choose an option (1/2/3): ")).strip()
        except EOFError:
            break
        if choice == "1":
            if model is None:
                model = load_model_or_none()
            if model:
                scan_loop(model)
        elif choice == "2":
            model = train_flow() or model
        elif choice == "3":
            print(green("\n  Thank you! Stay safe.\n"))
            break
        else:
            print(red("  Invalid option. Please type 1, 2 or 3."))


# ---------------------------------------------------------------------------
# Entry point (the program starts here when run from the command line)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phishing Email Detector")
    ap.add_argument("--data", help="path to a labeled CSV (to train the model)")
    ap.add_argument("--predict", help="email text to scan")
    ap.add_argument("--file", help="path to a text file containing an email")
    ap.add_argument("--interactive", action="store_true", help="open the email scanner")
    args = ap.parse_args()

    if args.predict or args.file:
        model = load_model_or_none()
        if model:
            if args.file:
                with open(args.file, encoding="utf-8", errors="ignore") as f:
                    show_result(f.read(), model)
            else:
                show_result(args.predict, model)
    elif args.interactive:
        m = load_model_or_none()
        if m:
            scan_loop(m)
    elif args.data:
        trained = train(load_csv(args.data))
        if sys.stdin.isatty():
            ans = input(cyan("\n  Do you want to scan an email now? (y/n): ")).strip().lower()
            if ans == "y":
                scan_loop(trained)
        else:
            print(dim("  Tip: use --interactive to scan emails."))
    else:
        menu()
