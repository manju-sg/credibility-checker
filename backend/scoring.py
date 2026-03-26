import os
import requests
import json
import base64
from datetime import datetime
from textblob import TextBlob
import spacy

# ── Gemini 2.5 Flash setup (new google-genai SDK) ─────────────────────────────
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
        print("⚠️  No GEMINI_API_KEY found — falling back to NLP scoring")
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
TEXT_PROMPT = """You are a world-class fact-checker and misinformation analyst.

Analyze the following content DEEPLY and return a JSON credibility assessment.

CONTENT:
\"\"\"{content}\"\"\"

Evaluate:
1. **Fact Match** (0-100): How well-supported are the claims by known facts?
2. **Language Quality** (0-100): Neutral/objective (high) vs sensationalist/emotional/clickbait (low)?
3. **Sentiment** (0-100): Balanced reporting (high) vs extreme bias/manipulation (low)?
4. **Source Quality** (0-100): Are credible sources cited/implied?

Red flags: ALL CAPS, excessive exclamation marks, conspiracy language ("they don't want you to know"),
missing sources, logical fallacies, miracle cures, vague attribution ("experts say" without naming them),
emotional manipulation designed to provoke fear/anger/outrage.

Return ONLY valid JSON, no markdown:
{{
  "score": <0-100>,
  "breakdown": {{"fact_match": <0-100>, "language": <0-100>, "sentiment": <0-100>, "source_quality": <0-100>}},
  "claims": [{{"text": "<claim>", "type": "<PERSON|ORG|STATISTIC|MEDICAL|POLITICAL|CLAIM>", "verified": false, "confidence": 0.8}}],
  "flags": ["<specific red flag>"],
  "verdict": "<CREDIBLE|LIKELY_TRUE|UNCERTAIN|MISLEADING|LIKELY_FALSE|FALSE>",
  "summary": "<2-3 sentences>",
  "reasoning": "<detailed analysis paragraph>"
}}"""

IMAGE_PROMPT = """You are an expert at detecting misinformation, deepfakes, and manipulated media.

Analyze this image carefully for:
1. **Visual Authenticity** (0-100): Signs of manipulation, deepfake artifacts, lighting inconsistencies
2. **Text Credibility** (0-100): Any text/captions — are the claims accurate?
3. **Context Accuracy** (0-100): Is the image used in proper context? Old/recycled footage?
4. **Source Quality** (0-100): Watermarks, logos, attribution visible?

Return ONLY valid JSON, no markdown:
{{
  "score": <0-100>,
  "breakdown": {{"visual_authenticity": <0-100>, "text_credibility": <0-100>, "context_accuracy": <0-100>, "source_quality": <0-100>}},
  "claims": [{{"text": "<text found in image>", "type": "VISUAL_CLAIM", "verified": false, "confidence": 0.7}}],
  "flags": ["<specific issue>"],
  "verdict": "<AUTHENTIC|LIKELY_AUTHENTIC|UNCERTAIN|LIKELY_MANIPULATED|MANIPULATED|DEEPFAKE>",
  "summary": "<2-3 sentences>",
  "reasoning": "<detailed visual analysis>"
}}"""

INTENT_PROMPT = """Classify this WhatsApp message into exactly one category:

CLAIM — a news headline, medical claim, political statement, or something that needs fact-checking
GREETING — hello, hi, hey, good morning, etc.
QUESTION — asking "what can you do?", "how to use this?", "help", or asking about your features/capabilities
CONVERSATION — general chat, jokes, unrelated topics

Message: "{text}"

Reply with ONLY one word: CLAIM, GREETING, QUESTION, or CONVERSATION"""


CHAT_PROMPT = """You are CredChecker Bot, a helpful WhatsApp assistant that specializes in fact-checking and detecting misinformation.

User message: "{text}"

Respond naturally and helpfully. If it's a greeting, introduce yourself briefly. If they ask what you do, explain:
- You fact-check news, headlines, and claims using Gemini AI + Google Fact Check API
- You analyze images for manipulation/deepfakes
- Just send any suspicious text or image and you'll give a credibility score

Keep it concise (under 200 words), friendly, and in plain text (no markdown, no asterisks)."""


def calculate_credibility_score(text, image_data=None, image_mime="image/jpeg"):
    """Master scoring function — Gemini 2.5 Flash + Fact Check API + NLP fallback"""
    try:
        if image_data:
            return _analyze_image(image_data, image_mime)

        result = _analyze_text_gemini(text) if gemini_client else None
        if not result:
            result = _fallback_nlp_scoring(text)

        # Augment with Google Fact Check API
        if GOOGLE_FACT_CHECK_API_KEY:
            fact_data = _check_google_fact_api(text)
            if fact_data['false_count'] > 0:
                result['score'] = max(0, result['score'] - fact_data['false_count'] * 12)
                result['flags'].append(f"Fact-checkers flagged {fact_data['false_count']} false claim(s)")
            if fact_data['true_count'] > 0:
                result['score'] = min(100, result['score'] + fact_data['true_count'] * 5)
            for check in fact_data['checks'][:2]:
                result['claims'].append({
                    "text": check['claim'][:120], "type": "FACT_CHECKED",
                    "verified": True, "rating": check['rating'],
                    "publisher": check['publisher'], "confidence": 0.95
                })
        return result

    except Exception as e:
        print(f"Scoring error: {e}")
        return _error_result()


def detect_intent(text):
    """Use Gemini to classify message intent. Falls back to keyword matching."""
    if not gemini_client:
        return _keyword_intent(text)
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=INTENT_PROMPT.format(text=text[:300]),
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=10)
        )
        intent = resp.text.strip().upper().split()[0]
        if intent in ('CLAIM', 'GREETING', 'QUESTION', 'CONVERSATION'):
            return intent
        return 'CLAIM'
    except Exception as e:
        print(f"Intent detection error: {e}")
        return _keyword_intent(text)


def generate_chat_reply(text):
    """Generate a natural conversational reply using Gemini"""
    if not gemini_client:
        return _static_help_message()
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=CHAT_PROMPT.format(text=text[:500]),
            config=genai_types.GenerateContentConfig(temperature=0.7, max_output_tokens=300)
        )
        return resp.text.strip()
    except:
        return _static_help_message()


def _keyword_intent(text):
    """Simple keyword-based fallback intent classification"""
    t = text.lower().strip()
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'sup', 'hola', 'namaste']
    questions = ['what can you', 'what do you', 'how do you', 'help', 'how does', 'what is this']
    if any(t == g or t.startswith(g) for g in greetings):
        return 'GREETING'
    if any(q in t for q in questions):
        return 'QUESTION'
    if len(text) > 30:  # longer text likely a claim
        return 'CLAIM'
    return 'CONVERSATION'


def _static_help_message():
    return (
        "Hi! I'm CredChecker Bot 🔍\n\n"
        "I can help you detect misinformation!\n\n"
        "Send me:\n"
        "• Any news headline or claim to fact-check\n"
        "• An image to check for deepfakes or manipulation\n\n"
        "I'll give you an AI credibility score instantly!"
    )


# ── Gemini text analysis ──────────────────────────────────────────────────────
def _analyze_text_gemini(text):
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=TEXT_PROMPT.format(content=text[:4500]),
            config=genai_types.GenerateContentConfig(temperature=0.1, max_output_tokens=1500)
        )
        raw = resp.text.strip()
        if '```json' in raw:
            raw = raw.split('```json')[1].split('```')[0].strip()
        elif '```' in raw:
            raw = raw.split('```')[1].split('```')[0].strip()

        data = json.loads(raw)
        score = max(0, min(100, int(data.get('score', 50))))
        bd = data.get('breakdown', {})
        return {
            "score": score,
            "breakdown": {
                "fact_match":     max(0, min(100, int(bd.get('fact_match', 50)))),
                "language":       max(0, min(100, int(bd.get('language', 50)))),
                "sentiment":      max(0, min(100, int(bd.get('sentiment', 50)))),
                "source_quality": max(0, min(100, int(bd.get('source_quality', 50)))),
            },
            "claims":    data.get('claims', [])[:6],
            "flags":     data.get('flags', [])[:6],
            "verdict":   data.get('verdict', 'UNCERTAIN'),
            "summary":   data.get('summary', ''),
            "reasoning": data.get('reasoning', ''),
        }
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "ResourceExhausted" in err_msg:
            print(f"⚠️  Gemini Quota Exceeded (429): {err_msg}")
            return {
                "score": 50, "breakdown": {"fact_match": 50, "language": 50, "sentiment": 50, "source_quality": 50},
                "claims": [], "flags": ["AI Quota Exceeded — Too many requests"],
                "verdict": "UNCERTAIN", 
                "summary": "⚠️ I'm a bit overwhelmed! Please wait 60 seconds and try again. (API Quota Exceeded)", 
                "reasoning": "The Gemini API free tier has reached its rate limit. Please try again in 1 minute."
            }
        print(f"Gemini text error: {err_msg}")
        return None


# ── Gemini image analysis ─────────────────────────────────────────────────────
def _analyze_image(image_data, mime_type="image/jpeg"):
    if not gemini_client:
        return {
            "score": 50, "content_type": "image",
            "breakdown": {"visual_authenticity": 50, "text_credibility": 50, "context_accuracy": 50, "source_quality": 50},
            "claims": [], "flags": ["Gemini API not configured — cannot analyze images"],
            "verdict": "UNCERTAIN", "summary": "Image analysis requires a valid Gemini API key.", "reasoning": ""
        }
    try:
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data

        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                IMAGE_PROMPT
            ],
            config=genai_types.GenerateContentConfig(temperature=0.1, max_output_tokens=1200)
        )
        raw = resp.text.strip()
        if '```json' in raw:
            raw = raw.split('```json')[1].split('```')[0].strip()
        elif '```' in raw:
            raw = raw.split('```')[1].split('```')[0].strip()

        data = json.loads(raw)
        score = max(0, min(100, int(data.get('score', 50))))
        return {
            "score": score, "content_type": "image",
            "breakdown": data.get('breakdown', {}),
            "claims":    data.get('claims', [])[:5],
            "flags":     data.get('flags', [])[:5],
            "verdict":   data.get('verdict', 'UNCERTAIN'),
            "summary":   data.get('summary', ''),
            "reasoning": data.get('reasoning', ''),
        }
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "ResourceExhausted" in err_msg:
            print(f"⚠️  Gemini Image Quota Exceeded (429): {err_msg}")
            return {
                "score": 50, "content_type": "image", "breakdown": {}, "claims": [],
                "flags": ["Image Quota Exceeded"], "verdict": "UNCERTAIN", 
                "summary": "⚠️ Image analysis limit reached. Please wait 1 minute before sending another image.", 
                "reasoning": "Gemini's free vision quota has been exceeded. Please try again in 60 seconds."
            }
        print(f"Image analysis error: {err_msg}")
        return {
            "score": 50, "content_type": "image",
            "breakdown": {}, "claims": [],
            "flags": [f"Image analysis error: {err_msg[:80]}"],
            "verdict": "UNCERTAIN", "summary": "Could not fully analyze image.", "reasoning": ""
        }



# ── Google Fact Check API ─────────────────────────────────────────────────────
def _check_google_fact_api(text):
    result = {"checks": [], "true_count": 0, "false_count": 0}
    queries = set()
    if nlp:
        try:
            for ent in nlp(text[:600]).ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                    queries.add(ent.text.strip())
        except:
            pass
    queries.add(text[:120])
    for query in list(queries)[:3]:
        cache_key = query.lower().strip()
        if cache_key in CREDIBILITY_CACHE:
            cached = CREDIBILITY_CACHE[cache_key]
            result['checks'].extend(cached.get('checks', []))
            result['true_count'] += cached.get('true_count', 0)
            result['false_count'] += cached.get('false_count', 0)
            continue
        try:
            resp = requests.get(
                'https://factcheckapi.googleapis.com/v1alpha1/claims:search',
                params={'query': query[:100], 'key': GOOGLE_FACT_CHECK_API_KEY, 'pageSize': 3},
                timeout=4
            )
            if resp.status_code == 200:
                data = resp.json()
                local = {"checks": [], "true_count": 0, "false_count": 0}
                for claim in data.get('claims', []):
                    for review in claim.get('claimReview', []):
                        rating = review.get('textualRating', '').lower()
                        check = {'claim': claim.get('text', '')[:120], 'rating': rating,
                                 'publisher': review.get('publisher', {}).get('name', 'Unknown'),
                                 'url': review.get('url', '')}
                        local['checks'].append(check)
                        if any(t in rating for t in ['true', 'mostly true', 'correct', 'accurate']):
                            local['true_count'] += 1
                        elif any(f in rating for f in ['false', 'incorrect', 'misleading', 'pants', 'wrong']):
                            local['false_count'] += 1
                CREDIBILITY_CACHE[cache_key] = local
                result['checks'].extend(local['checks'])
                result['true_count'] += local['true_count']
                result['false_count'] += local['false_count']
        except Exception as e:
            print(f"Fact check API error: {e}")
    return result


# ── NLP fallback ──────────────────────────────────────────────────────────────
def _fallback_nlp_scoring(text):
    try:
        blob = TextBlob(text)
        polarity, subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
        lang = 100
        if any(w in text.lower() for w in ['shocking', 'breaking!!', 'miracle', 'exposed', "they don't want"]):
            lang -= 25
        caps = sum(1 for w in text.split() if w.isupper() and len(w) > 2) / max(len(text.split()), 1)
        if caps > 0.3: lang -= 30
        if text.count('!') > 3: lang -= 15
        sent  = 90 if abs(polarity) < 0.15 else 75 if abs(polarity) < 0.4 else 50
        src   = 75 if any(p in text.lower() for p in ['according to', 'study', 'research', 'source:']) else 40
        flags = []
        if subjectivity > 0.65: flags.append("Highly subjective language")
        if src < 50: flags.append("No credible sources cited")
        final = max(0, min(100, int(lang * 0.35 + sent * 0.25 + src * 0.25 + 60 * 0.15)))
        verdict = 'CREDIBLE' if final >= 75 else 'UNCERTAIN' if final >= 50 else 'MISLEADING'
        emoji = '✅' if final >= 75 else '⚠️' if final >= 50 else '❌'
        return {
            "score": final,
            "breakdown": {"fact_match": 50, "language": lang, "sentiment": sent, "source_quality": src},
            "claims": [], "flags": flags, "verdict": verdict,
            "summary": f"{emoji} Content appears {verdict.lower()}. (Basic NLP — Gemini unavailable)", "reasoning": ""
        }
    except:
        return _error_result()


def _error_result():
    return {
        "score": 50,
        "breakdown": {"fact_match": 50, "language": 50, "sentiment": 50, "source_quality": 50},
        "claims": [], "flags": ["Analysis error — please try again"],
        "verdict": "UNCERTAIN", "summary": "Unable to fully analyze content.", "reasoning": ""
    }

# Legacy
def extract_claims(text): return []
def generate_summary(score, flags, claims):
    emoji = "✅" if score >= 75 else "⚠️" if score >= 50 else "❌"
    return f"{emoji} Content appears {'credible' if score >= 75 else 'uncertain' if score >= 50 else 'likely false'}."
