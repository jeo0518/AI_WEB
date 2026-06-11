import io
import os
import re
import unicodedata
from datetime import datetime, timezone

import torch
from docx import Document
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from pypdf import PdfReader
from transformers import AutoTokenizer, LongformerForSequenceClassification
from openai import OpenAI
from werkzeug.security import check_password_hash, generate_password_hash
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder=".", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
app.secret_key = os.environ["SECRET_KEY"]
CORS(app)

# Railway/Heroku-style URLs use the legacy "postgres://" scheme; SQLAlchemy 2 requires "postgresql://"
database_url = os.environ["DATABASE_URL"].replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class SavedFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    essay_excerpt = db.Column(db.String(400), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    raw_score = db.Column(db.Float, nullable=False)
    rationale = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()

HF_REPO = os.environ.get("HF_MODEL_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer (downloads once from HuggingFace)...")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

print(f"Loading model on {DEVICE}...")
model = LongformerForSequenceClassification.from_pretrained(HF_REPO, token=HF_TOKEN)
model.eval()
model.to(DEVICE)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
print("Ready.")


SCORE_LABELS = {
    (1.0, 1.5): "Inadequate",
    (1.5, 2.5): "Developing",
    (2.5, 3.5): "Approaching",
    (3.5, 4.5): "Proficient",
    (4.5, 5.5): "Advanced",
    (5.5, 6.0): "Exemplary",
}

def score_label(score):
    for (_, hi), label in SCORE_LABELS.items():
        if score <= hi:
            return label
    return "Exemplary"


def generate_rationale(essay: str, score: float) -> str:
    band = score_label(score)
    prompt = (
        f"An AI essay grading model scored the following essay {score}/6 ({band}).\n\n"
        f"Essay:\n{essay}\n\n"
        "Give a concise grading rationale in exactly 3 short paragraphs:\n"
        "1. Overall assessment and score justification\n"
        "2. Specific strengths (quote or reference the essay)\n"
        "3. Specific areas for improvement\n\n"
        "Be direct and constructive. Do not use headers or bullet points."
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert essay grader providing clear, specific feedback."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def extract_text_from_upload(file_storage):
    filename = file_storage.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    data = file_storage.read()

    if ext == "txt":
        return data.decode("utf-8", errors="ignore")

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(data))
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        # pypdf often emits ligatures (ﬁ, ﬂ) and spurious whitespace/line breaks
        # for PDFs that position glyphs individually (e.g. Google Docs exports).
        normalized = unicodedata.normalize("NFKC", raw_text)
        return re.sub(r"\s+", " ", normalized)

    if ext == "docx":
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")


@app.route("/")
def index():
    return app.send_static_file("index.html")


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not EMAIL_RE.match(email):
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with that email already exists."}), 409

    user = User(email=email, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()

    login_user(user)
    return jsonify({"email": user.email}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password."}), 401

    login_user(user)
    return jsonify({"email": user.email})


@app.route("/api/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"ok": True})


@app.route("/api/me")
def me():
    if current_user.is_authenticated:
        return jsonify({"email": current_user.email})
    return jsonify({"email": None})


@app.route("/api/history")
@login_required
def history():
    entries = (
        SavedFeedback.query.filter_by(user_id=current_user.id)
        .order_by(SavedFeedback.created_at.desc())
        .all()
    )
    return jsonify([
        {
            "id": entry.id,
            "essay_excerpt": entry.essay_excerpt,
            "score": entry.score,
            "raw": entry.raw_score,
            "rationale": entry.rationale,
            "created_at": entry.created_at.isoformat(),
        }
        for entry in entries
    ])


@app.route("/extract", methods=["POST"])
def extract():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file provided."}), 400

    try:
        text = extract_text_from_upload(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Could not read that file. Make sure it isn't corrupted or password-protected."}), 400

    text = text.strip()
    if not text:
        return jsonify({"error": "No text could be extracted from that file."}), 400

    return jsonify({"text": text})


@app.route("/grade", methods=["POST"])
def grade():
    data = request.get_json(silent=True) or {}
    essay = data.get("essay", "").strip()

    if not essay:
        return jsonify({"error": "Essay text is required."}), 400
    if len(essay.split()) < 10:
        return jsonify({"error": "Essay is too short to grade."}), 400

    inputs = tokenizer(
        essay,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    global_attention_mask = torch.zeros_like(inputs["input_ids"])
    global_attention_mask[:, 0] = 1
    inputs["global_attention_mask"] = global_attention_mask

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze().item()

    score = max(1.0, min(6.0, logits))
    score_rounded = round(score)

    rationale = generate_rationale(essay, score_rounded)
    raw_score = round(score, 3)

    if current_user.is_authenticated:
        excerpt = essay[:200] + ("…" if len(essay) > 200 else "")
        db.session.add(SavedFeedback(
            user_id=current_user.id,
            essay_excerpt=excerpt,
            score=score_rounded,
            raw_score=raw_score,
            rationale=rationale,
        ))
        db.session.commit()

    return jsonify({"score": score_rounded, "raw": raw_score, "rationale": rationale})


if __name__ == "__main__":
    app.run(port=5000, debug=False)
