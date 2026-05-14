import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, LongformerForSequenceClassification
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

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


@app.route("/")
def index():
    return app.send_static_file("index.html")


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

    return jsonify({"score": score_rounded, "raw": round(score, 3), "rationale": rationale})


if __name__ == "__main__":
    app.run(port=5000, debug=False)
