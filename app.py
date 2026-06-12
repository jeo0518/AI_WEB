import io
import json
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
    essay_text = db.Column(db.Text, nullable=True)
    score = db.Column(db.Integer, nullable=False)
    raw_score = db.Column(db.Float, nullable=False)
    rationale = db.Column(db.Text, nullable=False)
    rubric_json = db.Column(db.Text, nullable=True)
    annotations_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()
    # Lightweight migration for columns added after the
    # original table was created in production.
    with db.engine.connect() as conn:
        conn.execute(db.text("ALTER TABLE saved_feedback ADD COLUMN IF NOT EXISTS rubric_json TEXT"))
        conn.execute(db.text("ALTER TABLE saved_feedback ADD COLUMN IF NOT EXISTS essay_text TEXT"))
        conn.execute(db.text("ALTER TABLE saved_feedback ADD COLUMN IF NOT EXISTS annotations_json TEXT"))
        conn.commit()

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


RUBRIC_CATEGORIES = [
    "Ideas & Content",
    "Organization",
    "Voice & Style",
    "Sentence Fluency",
    "Conventions",
]


SENTENCE_RE = re.compile(r'[^.!?]*[.!?]+(?:["\')\]]+)?|[^.!?]+')


def segment_essay(essay: str):
    """Split an essay into paragraphs of sentences for annotation rendering.

    Returns (paragraphs, sentences): `sentences` is a flat list of sentence
    strings, and `paragraphs` is a list of lists of 1-based indices into
    `sentences`, grouping sentences by their original paragraph.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n+", essay) if p.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [essay.strip()]

    paragraphs = []
    sentences = []
    for para in raw_paragraphs:
        para_sentences = [s.strip() for s in SENTENCE_RE.findall(para) if s.strip()]
        if not para_sentences:
            continue
        indices = []
        for s in para_sentences:
            sentences.append(s)
            indices.append(len(sentences))
        paragraphs.append(indices)

    if not paragraphs:
        paragraphs = [[]]

    return paragraphs, sentences


def generate_feedback(essay: str, score: float, sentences: list) -> dict:
    band = score_label(score)
    categories = ", ".join(RUBRIC_CATEGORIES)
    numbered_essay = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, start=1))
    prompt = (
        f"An AI essay grading model scored the following essay {score}/6 ({band}).\n\n"
        "The essay below has been split into numbered sentences for reference:\n\n"
        f"{numbered_essay}\n\n"
        "Respond with a JSON object containing exactly three keys:\n\n"
        '"rationale": a concise grading rationale in exactly 3 short paragraphs '
        "(1. overall assessment and score justification, 2. specific strengths "
        "with reference to the essay, 3. specific areas for improvement). "
        "Be direct and constructive. Write in flowing prose without mentioning "
        "sentence numbers or brackets. Do not use headers or bullet points.\n\n"
        '"rubric": an array with exactly one object for each of these categories, '
        f"in this order: {categories}. Each object must have keys "
        '"category" (the category name), "score" (an integer from 1 to 6 '
        "rating the essay on that category), and \"feedback\" (one short "
        "sentence explaining the score for that category).\n\n"
        '"annotations": an array of 4 to 8 objects that call out specific '
        "strengths or issues in individual sentences from the numbered list "
        'above. Each object must have keys "sentence" (the integer sentence '
        'number it refers to), "type" (either "strength" or "improvement"), '
        f'"category" (one of: {categories}), and "comment" (one short, specific '
        "sentence, no more than ~20 words, explaining the strength or how to "
        "improve that sentence). Spread the annotations across different parts "
        "of the essay and include a mix of strengths and improvements."
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert essay grader providing clear, specific feedback. Always respond with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1300,
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}

    try:
        rationale = parsed["rationale"].strip()
        rubric = []
        for category, item in zip(RUBRIC_CATEGORIES, parsed.get("rubric", [])):
            rubric.append({
                "category": item.get("category", category),
                "score": max(1, min(6, round(float(item.get("score", score))))),
                "feedback": item.get("feedback", "").strip(),
            })
        if len(rubric) != len(RUBRIC_CATEGORIES):
            raise ValueError("Incomplete rubric breakdown")
    except (KeyError, ValueError, TypeError):
        rationale = content
        rubric = [
            {"category": category, "score": round(score), "feedback": ""}
            for category in RUBRIC_CATEGORIES
        ]

    annotations = []
    for item in parsed.get("annotations", []) if isinstance(parsed, dict) else []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("sentence"))
        except (TypeError, ValueError):
            continue
        if not (1 <= idx <= len(sentences)):
            continue
        ann_type = item.get("type")
        if ann_type not in ("strength", "improvement"):
            continue
        comment = str(item.get("comment", "")).strip()
        if not comment:
            continue
        annotations.append({
            "sentence": idx,
            "type": ann_type,
            "category": str(item.get("category", "")).strip(),
            "comment": comment,
        })

    return {"rationale": rationale, "rubric": rubric, "annotations": annotations}


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


@app.route("/grader")
def grader():
    return app.send_static_file("grader.html")


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
    result = []
    for entry in entries:
        if entry.essay_text:
            paragraphs, sentences = segment_essay(entry.essay_text)
        else:
            paragraphs, sentences = [], []
        result.append({
            "id": entry.id,
            "essay_excerpt": entry.essay_excerpt,
            "essay_text": entry.essay_text,
            "score": entry.score,
            "raw": entry.raw_score,
            "rationale": entry.rationale,
            "rubric": json.loads(entry.rubric_json) if entry.rubric_json else None,
            "annotations": json.loads(entry.annotations_json) if entry.annotations_json else None,
            "paragraphs": paragraphs,
            "sentences": sentences,
            "created_at": entry.created_at.isoformat(),
        })
    return jsonify(result)


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

    paragraphs, sentences = segment_essay(essay)
    feedback = generate_feedback(essay, score_rounded, sentences)
    rationale = feedback["rationale"]
    rubric = feedback["rubric"]
    annotations = feedback["annotations"]
    raw_score = round(score, 3)

    if current_user.is_authenticated:
        excerpt = essay[:200] + ("…" if len(essay) > 200 else "")
        db.session.add(SavedFeedback(
            user_id=current_user.id,
            essay_excerpt=excerpt,
            essay_text=essay,
            score=score_rounded,
            raw_score=raw_score,
            rationale=rationale,
            rubric_json=json.dumps(rubric),
            annotations_json=json.dumps(annotations),
        ))
        db.session.commit()

    return jsonify({
        "score": score_rounded,
        "raw": raw_score,
        "rationale": rationale,
        "rubric": rubric,
        "annotations": annotations,
        "paragraphs": paragraphs,
        "sentences": sentences,
    })


if __name__ == "__main__":
    app.run(port=5000, debug=False)
