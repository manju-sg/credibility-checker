import os
import requests
import json
import base64
from datetime import datetime
from textblob import TextBlob
import spacy

# ── Gemini setup ──────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv('GOOGLE_FACT_CHECK_API_KEY', '')  # reuse same key
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    else:
        gemini_model = None
except Exception as e:
    print(f"Gemini init error: {e}")
    gemini_model = None

# ── spaCy NER ─────────────────────────────────────────────────────────────────
try:
    nlp = spacy.load('en_core_web_sm')
except:
    nlp = None

GOOGLE_FACT_CHECK_API_KEY = os.getenv('GOOGLE_FACT_CHECK_API_KEY', '')
CREDIBILITY_CACHE = {}

# ── Gemini prompts ────────────────────────────────────────────────────────────
TEXT_ANALYSIS_PROMPT = """You are a world-class fact-checker and misinformation analyst with expertise in journalism, critical thinking, and source verification.

Analyze the following content DEEPLY and return a JSON credibility assessment.

CONTENT TO ANALYZE:
\"\"\"{content}\"\"\"

Evaluate these dimensions:
1. **Fact Match** (0-100): How well-supported are the claims by known facts? Are there verifiable sources implied?
2. **Language Quality** (0-100): Is the language neutral/objective (high) or sensationalist/emotional/clickbait (low)?
3. **Sentiment** (0-100): Higher = more neutral/balanced reporting. Lower = extreme bias or emotional manipulation.
4. **Source Quality** (0-100): Are credible sources cited/implied? Expert quotes? Data/statistics referenced properly?

Red flags to look for:
- ALL CAPS, excessive exclamation marks, clickbait phrases
- Conspiracy language ("they don't want you to know", "hidden truth")
- Missing sources for specific statistics or events
- Logical fallacies or circular reasoning
- Known misinformation patterns (miracle cures, election fraud, vaccine claims)
- Vague attribution ("some say", "experts claim" without naming them)
- Emotionally manipulative language designed to provoke fear/anger/outrage

Return ONLY this JSON (no markdown, no explanation):
{{
  "score": <overall credibility 0-100>,
  "breakdown": {{
    "fact_match": <0-100>,
    "language": <0-100>,
    "sentiment": <0-100>,
    "source_quality": <0-100>
  }},
  "claims": [
    {{"text": "<specific factual claim>", "type": "<PERSON|ORG|GPE|STATISTIC|MEDICAL|POLITICAL|CLAIM>", "verified": <true/false>, "confidence": <0.0-1.0>}}
  ],
  "flags": ["<specific red flag found>"],
  "verdict": "<CREDIBLE|LIKELY_TRUE|UNCERTAIN|MISLEADING|LIKELY_FALSE|FALSE>",
  "summary": "<2-3 sentences explaining the overall assessment>",
  "reasoning": "<detailed paragraph explaining the key factors in your judgment>"
}}"""

IMAGE_ANALYSIS_PROMPT = """You are an expert in detecting misinformation, manipulated media, deepfakes, and misleading visual content.

Analyze this image for credibility and authenticity. Look for:

1. **Visual Authenticity** (0-100): Signs of manipulation, deepfake artifacts, inconsistent lighting/shadows, unnatural edges, splice marks
2. **Text Credibility** (0-100): Any text/captions in the image — are the claims accurate?
3. **Context Accuracy** (0-100): Does the image appear to be used in proper context? Is it old/recycled footage presented as new?
4. **Source Quality** (0-100): Are there watermarks, logos, or attribution visible?

Return ONLY this JSON (no markdown):
{{
  "score": <overall credibility 0-100>,
  "breakdown": {{
    "visual_authenticity": <0-100>,
    "text_credibility": <0-100>,
    "context_accuracy": <0-100>,
    "source_quality": <0-100>
  }},
  "claims": [
    {{"text": "<text found in image or visual claim>", "type": "VISUAL_CLAIM", "verified": false, "confidence": 0.7}}
  ],
  "flags": ["<specific issue detected>"],
  "verdict": "<AUTHENTIC|LIKELY_AUTHENTIC|UNCERTAIN|LIKELY_MANIPULATED|MANIPULATED|DEEPFAKE>",
  "summary": "<2-3 sentences explaining the visual analysis>",
  "reasoning": "<detailed analysis of what you found>"
}}"""


def calculate_credibility_score(text, image_data=None, image_mime="image/jpeg"):
    """
    Master scoring function.
    - Uses Gemini 2.0 Flash for deep AI analysis
    - Augments with Google Fact Check API results
    - Falls back to NLP if Gemini unavailable
    """
    try:
        if image_data:
            return _analyze_image(image_data, image_mime)

        # Try Gemini first
        result = None
        if gemini_model:
            result = _analyze_text_gemini(text)

        if not result:
            result = _fallback_nlp_scoring(text)

        # Augment with Google Fact Check API
        if GOOGLE_FACT_CHECK_API_KEY:
            fact_data = _check_google_fact_api(text)
            if fact_data['false_count'] > 0:
                penalty = fact_data['false_count'] * 12
                result['score'] = max(0, result['score'] - penalty)
                result['flags'].append(f"Fact-checkers flagged {fact_data['false_count']} false claim(s)")
            if fact_data['true_count'] > 0:
                bonus = fact_data['true_count'] * 5
                result['score'] = min(100, result['score'] + bonus)
            for check in fact_data['checks'][:2]:
                result['claims'].append({
                    "text": check['claim'][:120],
                    "type": "FACT_CHECKED",
                    "verified": True,
                    "rating": check['rating'],
                    "publisher": check['publisher'],
                    "confidence": 0.95
                })

        return result

    except Exception as e:
        print(f"Scoring error: {e}")
        return _error_result()


# ── Gemini text analysis ──────────────────────────────────────────────────────
def _analyze_text_gemini(text):
    """Deep text analysis using Gemini 2.0 Flash"""
    try:
        prompt = TEXT_ANALYSIS_PROMPT.format(content=text[:4500])
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1500,
            )
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
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
        print(f"Gemini text analysis error: {e}")
        return None


# ── Gemini image analysis ─────────────────────────────────────────────────────
def _analyze_image(image_data, mime_type="image/jpeg"):
    """Analyze image using Gemini multimodal"""
    if not gemini_model:
        return {
            "score": 50, "content_type": "image",
            "breakdown": {"visual_authenticity": 50, "text_credibility": 50, "context_accuracy": 50, "source_quality": 50},
            "claims": [], "flags": ["Gemini API not configured — cannot analyze images"],
            "verdict": "UNCERTAIN", "summary": "Image analysis requires a valid Gemini API key.", "reasoning": ""
        }
    try:
        if isinstance(image_data, bytes):
            b64 = base64.b64encode(image_data).decode()
        else:
            b64 = image_data  # already base64

        image_part = {"inline_data": {"mime_type": mime_type, "data": b64}}
        response = gemini_model.generate_content(
            [IMAGE_ANALYSIS_PROMPT, image_part],
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=1200)
        )
        raw = response.text.strip()
        if '```json' in raw:
            raw = raw.split('```json')[1].split('```')[0].strip()
        elif '```' in raw:
            raw = raw.split('```')[1].split('```')[0].strip()

        data = json.loads(raw)
        score = max(0, min(100, int(data.get('score', 50))))
        return {
            "score": score,
            "content_type": "image",
            "breakdown": data.get('breakdown', {}),
            "claims":    data.get('claims', [])[:5],
            "flags":     data.get('flags', [])[:5],
            "verdict":   data.get('verdict', 'UNCERTAIN'),
            "summary":   data.get('summary', ''),
            "reasoning": data.get('reasoning', ''),
        }
    except Exception as e:
        print(f"Image analysis error: {e}")
        return {
            "score": 50, "content_type": "image",
            "breakdown": {}, "claims": [], "flags": [f"Image analysis error: {str(e)[:80]}"],
            "verdict": "UNCERTAIN", "summary": "Could not fully analyze image.", "reasoning": ""
        }


# ── Google Fact Check API ─────────────────────────────────────────────────────
def _check_google_fact_api(text):
    """Query Google Fact Check API for claims in the text"""
    result = {"checks": [], "true_count": 0, "false_count": 0}
    queries = set()

    # Extract named entities for targeted queries
    if nlp:
        try:
            doc = nlp(text[:600])
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                    queries.add(ent.text.strip())
        except:
            pass

    queries.add(text[:120])  # also query with the raw text start
    queries = list(queries)[:3]

    for query in queries:
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
                        check = {
                            'claim':     claim.get('text', '')[:120],
                            'rating':    rating,
                            'publisher': review.get('publisher', {}).get('name', 'Unknown'),
                            'url':       review.get('url', '')
                        }
                        local['checks'].append(check)
                        if any(t in rating for t in ['true', 'mostly true', 'correct', 'accurate']):
                            local['true_count'] += 1
                        elif any(f in rating for f in ['false', 'incorrect', 'misleading', 'pants', 'wrong', 'inaccurate']):
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
    """Rule-based NLP scoring when Gemini is unavailable"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        lang_score = 100
        sensational = ['shocking', 'breaking', 'unbelievable', 'miracle', 'exposed',
                       'they don\'t want', 'hidden truth', 'wake up', 'share before deleted']
        if any(w in text.lower() for w in sensational):
            lang_score -= 25
        caps_ratio = sum(1 for w in text.split() if w.isupper() and len(w) > 2) / max(len(text.split()), 1)
        if caps_ratio > 0.3:
            lang_score -= 30
        if text.count('!') > 3:
            lang_score -= 15

        sent_score = 90 if abs(polarity) < 0.15 else 75 if abs(polarity) < 0.4 else 50
        src_score = 75 if any(p in text.lower() for p in ['according to', 'study shows', 'research', 'source:', 'per ', 'citing']) else 40

        flags = []
        if subjectivity > 0.65:
            flags.append("Highly subjective language detected")
        if src_score < 50:
            flags.append("No credible sources cited")
        if caps_ratio > 0.2:
            flags.append("Excessive use of CAPS — common in sensationalist content")

        final = int(lang_score * 0.35 + sent_score * 0.25 + src_score * 0.25 + 60 * 0.15)
        final = max(0, min(100, final))
        verdict = 'CREDIBLE' if final >= 75 else 'UNCERTAIN' if final >= 50 else 'MISLEADING'
        emoji = '✅' if final >= 75 else '⚠️' if final >= 50 else '❌'

        return {
            "score": final,
            "breakdown": {"fact_match": 50, "language": lang_score, "sentiment": sent_score, "source_quality": src_score},
            "claims": [], "flags": flags, "verdict": verdict,
            "summary": f"{emoji} Content appears {verdict.lower()}. (Basic NLP analysis — Gemini unavailable)",
            "reasoning": ""
        }
    except Exception as e:
        return _error_result()


def _error_result():
    return {
        "score": 50,
        "breakdown": {"fact_match": 50, "language": 50, "sentiment": 50, "source_quality": 50},
        "claims": [], "flags": ["Analysis error — please try again"],
        "verdict": "UNCERTAIN", "summary": "Unable to fully analyze content.", "reasoning": ""
    }


# ── Legacy compatibility ───────────────────────────────────────────────────────
def extract_claims(text):
    return []

def generate_summary(score, flags, claims):
    emoji = "✅" if score >= 75 else "⚠️" if score >= 50 else "❌"
    verdict = "likely credible" if score >= 75 else "uncertain" if score >= 50 else "likely false"
    return f"{emoji} This content appears {verdict}."
