# /python/app.py
# Minimal Flask service for ticket NLP using Gemini (with safe fallback)
# Deps: Flask, Flask-CORS, google-generativeai

import os
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---- Env keys ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")      # required to use Gemini; service will fallback if missing
CLASSIFIER_KEY = os.getenv("CLASSIFIER_KEY")      # optional shared secret header: x-classifier-key

# ---- Flask ----
app = Flask(__name__)
CORS(app)

# ---- Gemini setup (lazy, lightweight) ----
model = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")  # fast & cost-efficient
        print("✅ Gemini configured")
    except Exception as e:
        print(f"⚠️  Gemini init failed: {e}; using fallback only")
        model = None
else:
    print("ℹ️  GEMINI_API_KEY not set; using fallback only")

# ---- Allowed labels ----
PRIORITIES = {"low", "medium", "high"}
DEPARTMENTS = {"account", "hardware", "network", "software", "other"}

# ---- Estimation heuristics (minutes) ----
BASE_MIN = {
    "account": 45,
    "network": 90,
    "software": 120,
    "hardware": 180,
    "other": 60,
}
PRIO_MULT = {"low": 0.75, "medium": 1.0, "high": 1.5}


def estimate_minutes(department: str, priority: str) -> int:
    base = BASE_MIN.get(department, BASE_MIN["other"])
    mult = PRIO_MULT.get(priority, 1.0)
    return int(round(base * mult))


# ---- Prompt sent to Gemini ----
SYSTEM_PROMPT = """You classify support tickets.

Return a STRICT JSON object with exactly two fields:
- "priority": one of ["low","medium","high"]
- "department": one of ["account","hardware","network","software","other"]

Guidance:
- If login/password/access → department = "account"
- If connectivity/WiFi/VPN/speed → "network"
- If device/peripherals/disk/monitor/keyboard/printer → "hardware"
- If app/OS/install/crash/error message → "software"
- Else → "other"

Priority heuristics:
- "high": outages, security risk, cannot work, data loss
- "medium": significantly blocked but has workaround
- "low": routine request or minor inconvenience

Output: ONLY the JSON object. No extra text.
"""


def _normalize(value: str, allowed: set[str], default: str) -> str:
    v = (value or "").strip().lower()
    return v if v in allowed else default


def _fallback_classify(text: str) -> dict:
    """Heuristic classifier used when Gemini is unavailable or errors."""
    t = (text or "").lower()

    # Department rules
    if any(k in t for k in ["login", "password", "2fa", "account", "signin", "reset"]):
        dept = "account"
    elif any(k in t for k in ["wifi", "network", "vpn", "latency", "internet", "dns"]):
        dept = "network"
    elif any(k in t for k in ["laptop", "keyboard", "monitor", "printer", "disk", "hardware", "battery"]):
        dept = "hardware"
    elif any(k in t for k in ["install", "crash", "error", "bug", "update", "windows", "macos", "linux", "app"]):
        dept = "software"
    else:
        dept = "other"

    # Priority rules
    if any(k in t for k in ["can't work", "cannot work", "down", "outage", "urgent", "security", "data loss", "won't boot", "won’t boot"]):
        prio = "high"
    elif any(k in t for k in ["slow", "sometimes", "intermittent", "degraded"]):
        prio = "medium"
    else:
        prio = "low"

    return {
        "priority": prio,
        "department": dept,
        "estimated_minutes": estimate_minutes(dept, prio),
    }


@app.get("/")
def root():
    return jsonify({
        "name": "Ticket NLP Service",
        "endpoints": {"GET /health": "service status", "POST /classify": "classify ticket"},
        "gemini": bool(model),
    })


@app.get("/health")
def health():
    return jsonify({"ok": True, "gemini": bool(model)})


@app.post("/classify")
def classify():
    # Optional shared-secret header
    if CLASSIFIER_KEY and request.headers.get("x-classifier-key") != CLASSIFIER_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    message = (data.get("message") or "").strip()
    text = (title + " " + message).strip()

    if not text:
        # Default neutral response
        prio, dept = "medium", "other"
        return jsonify({
            "priority": prio,
            "department": dept,
            "estimated_minutes": estimate_minutes(dept, prio),
        })

    # If Gemini is not ready, fallback
    if not model:
        return jsonify(_fallback_classify(text))

    # Try Gemini; always normalize + estimate
    try:
        user_prompt = f"Ticket message:\n{message}\n\nTitle:\n{title}\n"
        resp = model.generate_content([SYSTEM_PROMPT, user_prompt])
        raw = resp.text or ""

        # Extract JSON (be defensive if the model adds prose)
        m = re.search(r"\{[\s\S]*\}", raw)
        payload = m.group(0) if m else raw
        parsed = json.loads(payload)

        priority = _normalize(parsed.get("priority"), PRIORITIES, "medium")
        department = _normalize(parsed.get("department"), DEPARTMENTS, "other")

        return jsonify({
            "priority": priority,
            "department": department,
            "estimated_minutes": estimate_minutes(department, priority),
        })

    except Exception as e:
        # Safe fallback on any error
        print(f"⚠️  Gemini classify error: {e}")
        return jsonify(_fallback_classify(text))


if __name__ == "__main__":
    # Dev server
    port = int(os.getenv("PORT", "8001"))
    app.run(host="0.0.0.0", port=port, debug=True)
