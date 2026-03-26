import os
import requests
import json
import base64
from datetime import datetime
from textblob import TextBlob
import spacy

# ── Gemini 2.5 Flash setup ───────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types

    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_FACT_CHECK_API_KEY', '')
    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_MODEL  = 'gemini-2.5-flash'
        print(f"✅ Gemini AI configured (model: {GEMINI_MODEL})")
    else:
        gemini_client = None
        print("⚠️  No GEMINI_API_KEY found — AI analysis will be disabled.")
except Exception as e:
    print(f"❌ Gemini initialization error: {str(e)}")
    gemini_client = None

# ── spaCy ─────────────────────────────────────────────────────────────────────
try:
    nlp = spacy.load('en_core_web_sm')
except:
    nlp = None

GOOGLE_FACT_CHECK_API_KEY = os.getenv('GOOGLE_FACT_CHECK_API_KEY', '')
CREDIBILITY_CACHE = {}

# ── Prompts ───────────────────────────────────────────────────────────────────
TEXT_PROMPT = """You are a highly skeptical Fact-Checking AI. 
Your goal is to debunk misinformation, fake news, and propaganda.

STRICT RULES:
- If a claim sounds scientifically impossible (e.g., "Modi going to Mars next year"), mark it as FALSE immediately.
- If it's a "Forwarded" style viral message without sources, be extremely critical.
- Analyze the claim against your internal knowledge of world events, physics, and history.

CONTENT:
\"\"\"{content}\"\"\"

Return ONLY valid JSON:
{{
  "score": <0-100, where 0 is total lie, 100 is verified truth>,
  "breakdown": {{"fact_match": <0-100>, "language": <0-100>, "sentiment": <0-100>, "source_quality": <0-100>}},
  "claims": [{{"text": "<key claim>", "type": "<PERSON|ORG|STAT|MEDICAL|POLITICAL>", "verified": false, "confidence": 0.9}}],
  "flags": ["<specific red flag or logical fallacy>"],
  "verdict": "<CREDIBLE|LIKELY_TRUE|UNCERTAIN|MISLEADING|LIKELY_FALSE|FALSE>",
  "summary": "<Very clear verdict, e.g. 'This is a satirical or false claim.'>",
  "reasoning": "<Step-by-step logical debunking or verification>"
}}"""

IMAGE_PROMPT = """Analyze this image for misinformation/manipulation.
If the image contains text, extract the claims and verify them.
Look for deepfake artifacts, lighting errors, or out-of-context usage.

Return ONLY valid JSON:
{{
  "score": <0-100>,
  "breakdown": {{"visual_authenticity": <0-100>, "text_credibility": <0-100>, "context_accuracy": <0-100>, "source_quality": <0-100>}},
  "claims": [],
  "flags": [],
  "verdict": "<AUTHENTIC|LIKELY_AUTHENTIC|UNCERTAIN|LIKELY_MANIPULATED|MANIPULATED|DEEPFAKE>",
  "summary": "<verdict summary>",
  "reasoning": "<visual analysis>"
}}"""


# ── Core Analysis Logic ───────────────────────────────────────────────────────

def calculate_credibility_score(text, image_data=None, image_mime="image/jpeg"):
    """
    Multi-step Verification:
    1. Gemini Analysis (Deep Reasoning)
    2. Google Fact Check API (External Database)
    3. Conflict Resolution & Final Scoring
    """
    try:
        if image_data:
            return _analyze_image(image_data, image_mime)

        # Step 1: Gemini Deep Analysis
        gemini_res = _analyze_text_gemini(text) if gemini_client else None
        
        # Step 2: Google Fact Check API
        fact_data = _check_google_fact_api(text) if GOOGLE_FACT_CHECK_API_KEY else {"checks": [], "true_count": 0, "false_count": 0}

        # Step 3: Synthesis
        if not gemini_res:
            # If Gemini is dead, we ONLY reply if we have a hard Fact-Check hit.
            # Otherwise, we admit we don't know (stop-gap to prevent false 'Credible' responses).
            if fact_data['false_count'] > 0:
                result = _error_result("⚠️ AI analysis is currently unavailable, but this claim was FLAGGED as FALSE by human fact-checkers.")
                result['score'] = 15
                result['verdict'] = "FALSE"
                result['flags'] = [f"Found {fact_data['false_count']} debunked matches"]
            else:
                return _error_result("⚠️ Gemini AI is currently offline (Quota Exceeded). I cannot verify this claim with 100% accuracy right now.")
        else:
            result = gemini_res

        # If Fact Check API found hard proof, it OVERRIDES or HEAVILY weights the score
        if fact_data['false_count'] > 0:
            result['score'] = min(result['score'], 20)  # Drop to 'Likely False' or lower
            result['verdict'] = "FALSE" if fact_data['false_count'] > 1 else "LIKELY_FALSE"
            result['flags'].append(f"🔴 Debunked by {fact_data['false_count']} independent fact-checkers.")
            result['summary'] = f"❌ This claim has been debunked. {result['summary']}"
        
        for check in fact_data['checks'][:3]:
            result['claims'].append({
                "text": check['claim'][:120], "type": "FACT_CHECKED",
                "verified": True, "rating": check['rating'],
                "publisher": check['publisher'], "confidence": 0.98
            })

        return result

    except Exception as e:
        print(f"Scoring critical error: {e}")
        return _error_result("Analysis failed. Please try a different query.")


# ── Gemini Impl ───────────────────────────────────────────────────────────────

def _analyze_text_gemini(text):
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=TEXT_PROMPT.format(content=text[:4000]),
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=1500)
        )
        raw = resp.text.strip()
        if '```json' in raw: raw = raw.split('```json')[1].split('```')[0].strip()
        elif '```' in raw: raw = raw.split('```')[1].split('```')[0].strip()

        data = json.loads(raw)
        return {
            "score": int(data.get('score', 50)),
            "breakdown": data.get('breakdown', {}),
            "claims":    data.get('claims', [])[:5],
            "flags":     data.get('flags', [])[:5],
            "verdict":   data.get('verdict', 'UNCERTAIN'),
            "summary":   data.get('summary', ''),
            "reasoning": data.get('reasoning', ''),
        }
    except Exception as e:
        err = str(e)
        if "429" in err or "ResourceExhausted" in err:
            print("⚠️ Gemini Quota Exceeded")
        else:
            print(f"Gemini error: {err}")
        return None


def _analyze_image(image_data, mime_type="image/jpeg"):
    if not gemini_client: return _error_result("AI Vision unavailable.")
    try:
        if isinstance(image_data, str): image_bytes = base64.b64decode(image_data)
        else: image_bytes = image_data

        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type), IMAGE_PROMPT],
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=1200)
        )
        raw = resp.text.strip()
        if '```json' in raw: raw = raw.split('```json')[1].split('```')[0].strip()
        data = json.loads(raw)
        return {
            "score": int(data.get('score', 50)), "content_type": "image",
            "breakdown": data.get('breakdown', {}),
            "claims": data.get('claims', []), "flags": data.get('flags', []),
            "verdict": data.get('verdict', 'UNCERTAIN'),
            "summary": data.get('summary', ''), "reasoning": data.get('reasoning', '')
        }
    except: return _error_result("Image analysis failed.")


# ── Google Fact Check API ─────────────────────────────────────────────────────

def _check_google_fact_api(text):
    result = {"checks": [], "true_count": 0, "false_count": 0}
    try:
        # Search for exact or similar claims
        resp = requests.get(
            'https://factcheckapi.googleapis.com/v1alpha1/claims:search',
            params={'query': text[:200], 'key': GOOGLE_FACT_CHECK_API_KEY, 'pageSize': 5},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            for claim in data.get('claims', []):
                for review in claim.get('claimReview', []):
                    rating = review.get('textualRating', '').lower()
                    check = {'claim': claim.get('text', '')[:120], 'rating': rating,
                             'publisher': review.get('publisher', {}).get('name', 'Unknown')}
                    result['checks'].append(check)
                    if any(f in rating for f in ['false', 'incorrect', 'misleading', 'pants', 'wrong', 'fale']):
                        result['false_count'] += 1
    except: pass
    return result


# ── Intent & Chat ─────────────────────────────────────────────────────────────

def detect_intent(text):
    if not gemini_client: return 'CLAIM'
    try:
        prompt = f"Classify intent (CLAIM, GREETING, QUESTION, CONVERSATION): \"{text[:300]}\". Reply with ONE word."
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        intent = resp.text.strip().upper().split()[0]
        return intent if intent in ('CLAIM', 'GREETING', 'QUESTION', 'CONVERSATION') else 'CLAIM'
    except: return 'CLAIM'


def generate_chat_reply(text):
    if not gemini_client: return "I'm sorry, my AI brain is offline right now. Try again in a minute!"
    try:
        prompt = f"You are CredChecker Bot. Respond to: \"{text}\". Keep it concise and helpful."
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text.strip()
    except: return "I'm having trouble thinking right now. Please try again later."


def _error_result(msg=""):
    return {
        "score": 0,
        "breakdown": {"fact_match": 0, "language": 0, "sentiment": 0, "source_quality": 0},
        "claims": [], "flags": [msg or "Analysis error"],
        "verdict": "UNCERTAIN", "summary": msg or "Unable to analyze content.", "reasoning": ""
    }

# Legacy
def extract_claims(text): return []
def generate_summary(score, flags, claims): return ""
