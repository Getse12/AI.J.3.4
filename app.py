"""
AI Translator & Critic (Flask)

This simple Flask application demonstrates how to call an external LLM-style
HTTP API (Mentorpiece) to perform two tasks:
  1) Translate text to a selected language using a translation model
  2) Evaluate (judge) a provided translation using a judge model

Notes for QA engineers (beginner friendly):
- The app sends JSON requests to the API endpoint and expects JSON responses.
- Network errors, timeouts, and HTTP 4xx/5xx responses are handled and shown
  in the UI so you can inspect failures.
- The current environment is using the Mentorpiece API in OPEN mode (no
  Authorization header). If the API later requires a key, the code must be
  updated to add the `Authorization` header with the key from
  `os.getenv('MENTORPIECE_API_KEY')`.

Run:
  python src/app.py

"""

import os
import logging
from typing import List, Dict, Any, Optional

from flask import Flask, render_template, request, flash
import requests
from dotenv import load_dotenv

load_dotenv()

# Basic logging so QA can see what's happening in the logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
MENTORPIECE_ENDPOINT = "https://api.mentorpiece.org/v1/process-ai-request"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")


def call_llm(model_name: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call Mentorpiece API and return parsed response.

    Parameters:
      - model_name: name of the LLM model to call
      - messages: list of message objects. Each object should have a 'role'
        (typically 'user') and a 'content' string. This function will join
        messages into a prompt and send JSON in the shape required by the
        current API: {"model_name": ..., "prompt": ...}

    Returns dict with keys:
      - 'ok': bool, whether call succeeded (200)
      - 'text': string with the model response or error info
      - 'raw': full raw response text (for debugging/QA)
      - 'status_code': numeric HTTP status code
    """
    # Join messages into a single prompt (simplest approach)
    prompt_parts = []
    for m in messages:
        if isinstance(m, dict) and "content" in m:
            prompt_parts.append(str(m["content"]))
        else:
            prompt_parts.append(str(m))
    prompt = "\n".join(prompt_parts)

    payload = {"model_name": model_name, "prompt": prompt}

    headers = {"Content-Type": "application/json"}
    # NOTE: mentorpiece is currently in open mode — do NOT send Authorization
    # If in future a key is required, add:
    #   headers['Authorization'] = f"Bearer {os.getenv('MENTORPIECE_API_KEY')}"

    try:
        logger.info("Calling Mentorpiece model=%s", model_name)
        resp = requests.post(MENTORPIECE_ENDPOINT, json=payload, headers=headers, timeout=20)
        raw = resp.text
        status = resp.status_code
        if status != 200:
            # If 4xx/5xx, return helpful info for QA to inspect
            logger.warning("Mentorpiece returned HTTP %s", status)
            # Try to include JSON 'response' if available
            try:
                j = resp.json()
                text = j.get("response") or j.get("message") or raw
            except Exception:
                text = raw
            return {"ok": False, "text": text, "raw": raw, "status_code": status}

        # success path — parse JSON and extract the 'response' field
        try:
            j = resp.json()
            # per spec response is {"response": "..."}
            text = j.get("response") if isinstance(j, dict) else None
            if not text:
                # fallback to stringified JSON or raw
                text = str(j)
            return {"ok": True, "text": text, "raw": raw, "status_code": status}
        except ValueError:
            # resp.body is not JSON — return raw text
            return {"ok": True, "text": raw, "raw": raw, "status_code": status}

    except requests.RequestException as e:
        # Network/timeout error — surface to UI for QA
        logger.exception("Network error calling Mentorpiece: %s", e)
        return {"ok": False, "text": str(e), "raw": "", "status_code": None}


@app.route("/", methods=["GET"])
def index():
    """Render the main page with an input form.

    QA note: This route only renders the form. Submission happens on POST /.
    """
    return render_template("index.html")


@app.route("/", methods=["POST"])
def process():
    """Handle form submission: translate and optionally judge.

    The form includes two buttons:
      - "translate": performs a translation call
      - "judge": performs both translation and a judge call (LLM-as-a-Judge)
    """
    source_text = request.form.get("source_text", "").strip()
    language = request.form.get("language", "English")
    action = request.form.get("action")

    if not source_text:
        flash("Пожалуйста, введите исходный текст для перевода.")
        return render_template("index.html")

    # Prepare the translation prompt and call the translation model
    translate_prompt = f"Переведи на {language}: {source_text}"
    trans_resp = call_llm("Qwen/Qwen3-VL-30B-A3B-Instruct", [{"role": "user", "content": translate_prompt}])

    translation = trans_resp.get("text") if trans_resp else None
    translation_ok = trans_resp.get("ok", False) if trans_resp else False

    judge_text = None
    judge_ok = False
    judge_raw = None

    if action == "judge":
        # Build judge prompt: provide original and translated text and ask to grade
        judge_prompt = (
            f"Оцени качество перевода от 1 до 10 и коротко аргументируй.\n"
            f"Оригинал: {source_text}\nПеревод: {translation or '[нет перевода]'}"
        )
        judge_resp = call_llm("claude-sonnet-4-5-20250929", [{"role": "user", "content": judge_prompt}])
        judge_text = judge_resp.get("text") if judge_resp else None
        judge_ok = judge_resp.get("ok", False) if judge_resp else False
        judge_raw = judge_resp.get("raw") if judge_resp else None

    # Render page with results; QA can inspect raw responses
    return render_template(
        "index.html",
        source_text=source_text,
        language=language,
        translation=translation,
        translation_ok=translation_ok,
        translation_raw=trans_resp.get("raw") if trans_resp else None,
        judge_text=judge_text,
        judge_ok=judge_ok,
        judge_raw=judge_raw,
    )


if __name__ == "__main__":
    # Run in debug mode for local testing; do NOT use Flask's dev server in prod
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
