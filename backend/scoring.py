import os
import requests
import json
import base64
import random
from datetime import datetime

# ── Gemini Multi-Key Rotation ────────────────────────────────────────────────
# To bypass 429 Quota errors, the user can provide multiple keys separated by commas.
raw_keys = os.getenv('GEMINI_API_KEY', '') or os.getenv('GOOGLE_FACT_CHECK_API_KEY', '')
GEMINI_KEYS = [k.strip() for k in raw_keys.split(',') if k.strip()]

clients = []
try:
    from google import genai
    from google.genai import types as genai_types
    for key in GEMINI_KEYS:
        clients.append(genai.Client(api_key=key))
    
    GEMINI_MODEL = 'gemini-2.5-flash'
    if clients:
        print(f"✅ Gemini Multi-Key System Active ({len(clients)} keys loaded)")
    else:
        print("⚠️ No Gemini keys found — AI analysis disabled.")
except Exception as e:
    print(f"❌ Gemini setup error: {str(e)}")
    clients = []

GOOGLE_FACT_CHECK_API_KEY = os.getenv('GOOGLE_FACT_CHECK_API_KEY', '')


# ── Strict Truth-Checking Prompts ────────────────────────────────────────────

TEXT_PROMPT = """Analyze this claim for credibility. Be extremely skeptical.
If it is a known hoax, satire, or physically impossible (e.g. humans on Mars next year), mark it FALSE.

CLAIM: \"\"\"{content}\"\"\"

Return ONLY valid JSON:
{{
  "score": <0-100>,
  "verdict": "<CREDIBLE|LIKELY_TRUE|UNCERTAIN|LIKELY_FALSE|FALSE>",
  "summary": "<one sentence punchy summary>",
  "reasoning": "<concise logic: explain WHY it is true or false using facts>",
  "flags": ["list of 1-3 specific red flags"]
}}"""

IMAGE_PROMPT = """Analyze this image for manipulation or deepfakes.
Return ONLY valid JSON:
{{
  "score": <0-100>,
  "verdict": "<AUTHENTIC|LIKELY_AUTHENTIC|UNCERTAIN|LIKELY_MANIPULATED|MANIPULATED|DEEPFAKE>",
  "summary": "<one sentence verdict>",
  "reasoning": "<technical visual analysis>",
  "flags": ["list of 1-3 visual red flags"]
}}"""


# ── Master Calculation Logic ──────────────────────────────────────────────────

from local_engine import engine as local_engine # Optimized Offline Engine

def calculate_credibility_score(text='', image_data=None, image_mime='image/jpeg'):
    """
    Main Orchestrator:
    - Text: Uses local_engine (Offline, No Quota Limit)
    - Fact Checks: Uses Google API (Database Results)
    - Image: Uses Gemini (Multimodal fallback)
    """
    try:
        # 1. IMAGE ANALYSIS (Needs Computer Vision)
        if image_data:
            ai_res = _try_gemini(_analyze_image, image_data, image_mime)
            if not ai_res:
                return _error_result("Image analysis requires an active Gemini API key.")
            return ai_res

        if not text:
            return _error_result("No text provided.")

        # 2. LOCAL TEXT ANALYSIS (Accurate Linguistic Pattern Matching)
        res = local_engine.analyze(text)
        
        # 3. GOOGLE FACT CHECK API (Database Verification)
        if GOOGLE_FACT_CHECK_API_KEY:
            fact_hit = _check_google_fact_api(text)
            if fact_hit and fact_hit.get('false_count', 0) > 0:
                res['score'] = max(5, res['score'] - 40)
                res['verdict'] = "FALSE"
                res['summary'] = "❌ Debunked by official fact-checkers."
                res['reasoning'] = "This claim was found in global fact-check databases with a FALSE rating. " + res['reasoning']
                res['flags'].append("Explicitly Debunked")

        return res

    except Exception as e:
        print(f"Scoring Orchestrator Error: {e}")
        return _error_result(f"Analysis failed: {str(e)}")


# ── Helper: Multi-Key Runner ──────────────────────────────────────────────────

def _try_gemini(func, *args, **kwargs):
    """Tries the function with multiple API keys to bypass 429 errors."""
    if not clients: return None
    
    # Randomize starting key for load balancing
    shuffled_clients = list(clients)
    random.shuffle(shuffled_clients)

    for client in shuffled_clients:
        try:
            return func(client, *args, **kwargs)
        except Exception as e:
            err = str(e)
            if "429" in err or "ResourceExhausted" in err:
                print(f"⚠️ Key Quota Exceeded (Try next key...)")
                continue
            print(f"Gemini error: {err}")
            return None
    return None


# ── Internal AI Modules ───────────────────────────────────────────────────────

def _analyze_text_gemini(client, text):
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=TEXT_PROMPT.format(content=text[:4000]),
        config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=1000)
    )
    raw = resp.text.strip()
    
    # Robust JSON extraction
    try:
        if '```json' in raw: raw = raw.split('```json')[1].split('```')[0].strip()
        elif '```' in raw: raw = raw.split('```')[1].split('```')[0].strip()
        
        # Handle cases where AI adds text before/after JSON
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end+1]
            
        data = json.loads(raw, strict=False) # strict=False allows control characters/newlines
        return {
            "score": int(data.get('score', 50)),
            "verdict": str(data.get('verdict', 'UNCERTAIN')),
            "summary": str(data.get('summary', '')),
            "reasoning": str(data.get('reasoning', '')),
            "flags": list(data.get('flags', [])),
            "claims": []
        }
    except Exception as e:
        print(f"Gemini JSON Parsing error: {e}")
        return {
            "score": 50, "verdict": "UNCERTAIN", 
            "summary": "AI response was complex. Checking external registries...",
            "reasoning": raw[:500], "flags": ["Complex Analysis"], "claims": []
        }


def _analyze_image(client, image_data, mime_type="image/jpeg"):
    if isinstance(image_data, str): image_bytes = base64.b64decode(image_data)
    else: image_bytes = image_data

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type), IMAGE_PROMPT],
        config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=1000)
    )
    raw = resp.text.strip()
    if '```json' in raw: raw = raw.split('```json')[1].split('```')[0].strip()
    data = json.loads(raw)
    return {
        "score": int(data.get('score', 50)),
        "verdict": data.get('verdict', 'UNCERTAIN'),
        "summary": data.get('summary', ''),
        "reasoning": data.get('reasoning', ''),
        "flags": data.get('flags', []),
        "content_type": "image"
    }


# ── Google Fact Check API ─────────────────────────────────────────────────────

def _check_google_fact_api(text):
    try:
        resp = requests.get(
            'https://factcheckapi.googleapis.com/v1alpha1/claims:search',
            params={'query': text[:200], 'key': GOOGLE_FACT_CHECK_API_KEY, 'pageSize': 3},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            false_count = 0
            for claim in data.get('claims', []):
                for review in claim.get('claimReview', []):
                    rating = review.get('textualRating', '').lower()
                    if any(f in rating for f in ['false', 'incorrect', 'misleading', 'pants', 'wrong']):
                        false_count += 1
            return {'false_count': false_count}
    except: pass
    return None


# ── Intent & Communication ─────────────────────────────────────────────────────

def detect_intent(text):
    res = _try_gemini(_get_intent, text)
    return res if res else 'CLAIM'

def _get_intent(client, text):
    prompt = f"Identify intent (CLAIM, GREETING, QUESTION, CONVERSATION). Text: \"{text[:300]}\". Reply ONE word."
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    intent = resp.text.strip().upper().split()[0]
    return intent if intent in ('CLAIM', 'GREETING', 'QUESTION', 'CONVERSATION') else 'CLAIM'


def generate_chat_reply(text):
    res = _try_gemini(_chat_reply, text)
    return res if res else "I'm sorry, I'm currently at my rate limit. Please add more API keys or wait 1 minute."

def _chat_reply(client, text):
    prompt = f"You are CredChecker Bot. Respond concisely to: \"{text}\"."
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return resp.text.strip()


def _error_result(msg):
    return {
        "score": 0, "verdict": "UNCERTAIN", "summary": msg,
        "reasoning": "We hit a quota limit. To fix this permanently, add a second API key in your Render settings.",
        "flags": ["Quota Exceeded"], "claims": []
    }
