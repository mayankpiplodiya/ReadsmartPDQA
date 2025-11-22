import os
import re
import sqlite3
import datetime
import logging
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    flash, send_from_directory
)
from werkzeug.utils import secure_filename


try:
    from textstat import flesch_reading_ease
    HAS_TEXTSTAT = True
except Exception:
    flesch_reading_ease = None
    HAS_TEXTSTAT = False

# Gemini SDK
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    genai = None
    HAS_GENAI = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GEMINI_KEY = os.getenv("GEMINI_KEY", "c2d49092dd1f3a8313dad1648d579a55").strip()
FLASK_SECRET = os.getenv("FLASK_SECRET", "c2d49092dd1f3a8313dad1648d579a55")
app = Flask(__name__)
app.secret_key = FLASK_SECRET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("readsmart")

# configure Gemini if key provided
if HAS_GENAI and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        logger.info("Configured Gemini SDK.")
    except Exception as e:
        logger.warning("Failed to configure Gemini SDK: %s", e)
else:
    if not HAS_GENAI:
        logger.info("google.generativeai SDK not installed. Gemini disabled.")
    elif not GEMINI_KEY:
        logger.info("GEMINI_KEY not set. Gemini disabled.")

# ---------------- DB
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db_and_migrate():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      password TEXT NOT NULL
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER,
      text TEXT,
      summary TEXT,
      created_at TEXT,
      readability REAL
    )
    """)
    conn.commit()

    c.execute("SELECT id FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', 'admin'))
    conn.commit()
    conn.close()

init_db_and_migrate()

# Helpers
def naive_extractive_summarize(text, ratio=0.12, min_sents=3, max_sents=12):
    if not text:
        return ""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sents:
        return text[:1000]
    n = max(min_sents, int(len(sents) * ratio))
    n = min(n, max_sents)
    return " ".join(sents[:n]).strip()

# Gemini model discovery & selection (robust)
_cached_gemini_model = None

def list_gemini_models():
    if not (HAS_GENAI and GEMINI_KEY):
        return []
    try:
        resp = genai.list_models()
        names = []
        if isinstance(resp, dict):
            models = resp.get("models") or resp.get("model") or []
        else:
            models = getattr(resp, "models", resp) or resp
        for m in models:
            if isinstance(m, dict):
                nm = m.get("name")
            else:
                nm = getattr(m, "name", None)
            if nm:
                names.append(nm)
        return names
    except Exception as e:
        logger.warning("list_gemini_models failed: %s", e)
        return []

def pick_gemini_model():
    global _cached_gemini_model
    if _cached_gemini_model:
        return _cached_gemini_model
    names = list_gemini_models()
    logger.info("Available Gemini models: %s", names)
    preferred = [
        "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.1",
        "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5"
    ]
    for cand in preferred:
        for n in names:
            if cand in n.lower():
                _cached_gemini_model = n
                return n
    for n in names:
        if "gemini" in n.lower():
            _cached_gemini_model = n
            return n
    return None

def extract_text_from_file(path):
    path_l = (path or "").lower()
    try:
        if path_l.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif path_l.endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(path)
                return "\n".join([page.get_text("text") for page in doc])
            except Exception as e:
                logger.warning("PDF extraction failed: %s", e)
                return ""
        elif path_l.endswith(".docx"):
            try:
                import docx
                d = docx.Document(path)
                return "\n".join([p.text for p in d.paragraphs])
            except Exception as e:
                logger.warning("DOCX extraction failed: %s", e)
                return ""
    except Exception as e:
        logger.warning("extract_text_from_file error: %s", e)
    return ""

def compute_readability(text):
    if not text:
        return None
    if HAS_TEXTSTAT:
        try:
            return round(flesch_reading_ease(text), 2)
        except Exception as e:
            logger.warning("readability compute failed: %s", e)
            return None
    return None

def robust_extract_genai_text(resp):
    """Try multiple locations for generated content in the genai response."""
    if not resp:
        return None
    try:
        t = getattr(resp, "text", None)
        if t:
            return t
    except Exception:
        pass
    try:
        if isinstance(resp, dict):
            cands = resp.get("candidates") or resp.get("outputs") or []
            if isinstance(cands, list) and cands:
                first = cands[0]
                for key in ("content", "output", "text", "message"):
                    if isinstance(first, dict) and first.get(key):
                        return first.get(key)
                return str(first)
            for key in ("text", "content", "output"):
                if resp.get(key):
                    return resp.get(key)
    except Exception:
        pass
    try:
        s = str(resp)
        if s and len(s) > 0:
            return s
    except Exception:
        pass
    return None

def summarize_with_gemini(text):
    """Return summary string or raise on failure."""
    if not (HAS_GENAI and GEMINI_KEY):
        raise RuntimeError("Gemini not available")
    model_name = pick_gemini_model()
    if not model_name:
        # fallback guess
        model_name = "gemini-1.5-flash"
    parts = []
    chunk_size = 14000
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            nl = text.rfind("\n", start, end)
            dot = text.rfind(".", start, end)
            if nl > start:
                end = nl
            elif dot > start:
                end = dot + 1
        chunk = text[start:end].strip()
        start = end
        if not chunk:
            continue
        try:
            model = genai.GenerativeModel(model_name)
            prompt = (
                "Summarize the text below in 5-8 clear sentences. Use short paragraphs and make it well-structured:\n\n"
                f"{chunk}"
            )
            resp = model.generate_content(prompt)
            gen_text = robust_extract_genai_text(resp)
            if gen_text:
                parts.append(gen_text.strip())
        except Exception as e:
            logger.warning("Gemini chunk generation failed: %s", e)
            # surface the error to caller
            raise
    if not parts:
        return ""
    return "\n\n".join(parts)

def summarize_text(text):
    """Try Gemini first, fallback to naive summarizer."""
    if not text or not text.strip():
        return "No readable text found."
    if HAS_GENAI and GEMINI_KEY:
        try:
            s = summarize_with_gemini(text)
            if s:
                return s
        except Exception as e:
            logger.warning("Gemini summarize failed: %s — falling back", e)
    return naive_extractive_summarize(text)

# History DB helpers
def save_history(user_id, text, summary, readability):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO history (user_id, text, summary, created_at, readability) VALUES (?, ?, ?, ?, ?)",
        (user_id, text, summary, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), readability)
    )
    conn.commit()
    conn.close()

def get_history_for_user(user_id):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, text, summary, created_at, readability FROM history WHERE user_id = ? ORDER BY id DESC", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_history_entry(entry_id):
    conn = get_db_connection()
    r = conn.execute("SELECT id, user_id, text, summary, created_at, readability FROM history WHERE id = ?", (entry_id,)).fetchone()
    conn.close()
    return r

def readability_label(score):
    if score is None:
        return "N/A"
    try:
        s = float(score)
    except Exception:
        return "N/A"
    if s >= 90:
        return f"{s} — Very Easy"
    if s >= 80:
        return f"{s} — Easy"
    if s >= 70:
        return f"{s} — Fairly Easy"
    if s >= 60:
        return f"{s} — Standard"
    if s >= 50:
        return f"{s} — Fairly Difficult"
    if s >= 30:
        return f"{s} — Difficult"
    return f"{s} — Very Confusing"

@app.context_processor
def utility_processor():
    return dict(readability_label=readability_label)

#  Routes
@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        if username == "admin" and password == "admin":
            conn = get_db_connection()
            row = conn.execute("SELECT id, username FROM users WHERE username=?", ("admin",)).fetchone()
            conn.close()
            if row:
                session['user'] = row['username']
                session['user_id'] = row['id']
            else:
                session['user'] = "admin"
                session['user_id'] = None
            flash("Logged in as admin.", "success")
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid username or password. Use admin / admin for demo."
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET','POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        if not username or not password:
            error = "Provide both username and password."
            return render_template('signup.html', error=error)
        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            flash("Signup successful. Please login (use admin/admin for demo).", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            error = "Username already exists."
        except Exception as e:
            logger.error("Signup error: %s", e)
            error = "Signup failed."
    return render_template('signup.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    summary = None
    readability = None
    if request.method == 'POST':
        text = request.form.get('text','').strip()
        file = request.files.get('file')
        if file and file.filename:
            filename = secure_filename(file.filename)
            fpath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(fpath)
            extracted = extract_text_from_file(fpath)
            text = extracted or text or f"[Uploaded file saved as {filename}]"
        if not text:
            flash("Provide text or upload a file.", "warning")
            return render_template('dashboard.html', summary=None, readability=None)
        readability = compute_readability(text)

        if HAS_GENAI and GEMINI_KEY:
            try:
                summary = summarize_with_gemini(text)
            except Exception as e:
                logger.warning("Gemini failed: %s", e)
                flash("Gemini summarization failed; using local fallback.", "warning")
                summary = naive_extractive_summarize(text)
        else:
            summary = naive_extractive_summarize(text)

        try:
            save_history(session.get('user_id'), text, summary, readability)
        except Exception as e:
            logger.error("Failed to save history: %s", e)
            flash("Could not save history.", "warning")
    return render_template('dashboard.html', summary=summary, readability=readability, readability_score=readability)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    history_list = get_history_for_user(session.get('user_id'))
    return render_template('history.html', histories=history_list, history=history_list)

@app.route('/view_history')
def view_history():
    if 'user' not in session:
        return redirect(url_for('login'))
    history_list = get_history_for_user(session.get('user_id'))
    return render_template('view_history.html', history=history_list)

@app.route('/history/<int:entry_id>')
def view_entry(entry_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    row = get_history_entry(entry_id)
    if not row:
        flash("Entry not found.", "danger")
        return redirect(url_for('history'))
    if row['user_id'] is not None and row['user_id'] != session.get('user_id'):
        flash("Permission denied.", "danger")
        return redirect(url_for('history'))
    entry = dict(row)
    if os.path.exists(os.path.join(BASE_DIR, "templates", "view_entry.html")):
        return render_template('view_entry.html', entry=entry)
    else:
        return render_template('view_history.html', history=[entry])

@app.route('/history/<int:entry_id>/delete', methods=['POST'])
def delete_entry(entry_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    row = get_history_entry(entry_id)
    if not row:
        flash("Entry not found.", "danger")
        return redirect(url_for('history'))
    if row['user_id'] is not None and row['user_id'] != session.get('user_id'):
        flash("Permission denied.", "danger")
        return redirect(url_for('history'))
    conn = get_db_connection()
    conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()
    flash("Entry deleted.", "success")
    return redirect(url_for('history'))

@app.route('/history/<int:entry_id>/resummarize', methods=['POST'])
def resummarize_entry(entry_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    row = get_history_entry(entry_id)
    if not row:
        flash("Entry not found.", "danger")
        return redirect(url_for('history'))
    if row['user_id'] is not None and row['user_id'] != session.get('user_id'):
        flash("Permission denied.", "danger")
        return redirect(url_for('history'))
    original_text = row['text'] or ""
    new_readability = compute_readability(original_text)
    new_summary = summarize_text(original_text)
    conn = get_db_connection()
    conn.execute("UPDATE history SET summary = ?, created_at = ?, readability = ? WHERE id = ?", (new_summary, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), new_readability, entry_id))
    conn.commit()
    conn.close()
    flash("Entry re-summarized.", "success")
    return redirect(url_for('view_entry', entry_id=entry_id))

if __name__ == '__main__':
    app.run(debug=True)
