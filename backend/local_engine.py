import re
from textblob import TextBlob

import spacy

# Load spaCy small model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class VeritasLocalEngine:
    """
    Veritas AI Offline Engine:
    Detects misinformation patterns using linguistic markers, 
    sentiment polarity, and semantic structure without APIs.
    """

    MISINFO_KEYWORDS = [
        "miracle cure", "doctors hate", "shocking truth", "hidden secret",
        "conspiracy", "mainstream media won't tell", "proven to kill", 
        "instant results", "government cover-up", "leak", "exposed"
    ]

    CLICKBAIT_PATTERNS = [
        r"\d+ ways to", r"you won't believe", r"what happened next", 
        r"shocking", r"incredible", r"bizarre"
    ]

    def analyze(self, text):
        if not text: return None
        
        doc = nlp(text)
        blob = TextBlob(text)
        
        # 1. Subjectivity & Polarity (TextBlob)
        # Higher subjectivity often correlates with biased/misleading info.
        subjectivity = blob.sentiment.subjectivity * 100
        polarity = abs(blob.sentiment.polarity) * 100
        
        # 2. Clickbait & Misinfo Keywords (Pattern Match)
        keyword_hits = 0
        for kw in self.MISINFO_KEYWORDS:
            if kw.lower() in text.lower(): keyword_hits += 1
            
        clickbait_hits = 0
        for pat in self.CLICKBAIT_PATTERNS:
            if re.search(pat, text, re.IGNORECASE): clickbait_hits += 1

        # 3. Linguistic Complexity (spaCy)
        # Extremely simple or overly complex structure can be a signal.
        num_sentences = len(list(doc.sents)) or 1
        avg_sent_len = len(doc) / num_sentences
        
        # Red Flags Calculation
        flags = []
        if subjectivity > 70: flags.append("High Subjective Bias")
        if polarity > 60: flags.append("Extreme Emotional Tone")
        if keyword_hits > 0: flags.append("Misinformation Keywords Detected")
        if clickbait_hits > 0: flags.append("Clickbait Structure Detected")
        if avg_sent_len > 40: flags.append("Unusually Dense/Complex Phrasing")
        if text.isupper(): flags.append("All-Caps 'Shouting' detected")

        # Final Score Logic (0-100, where 100 is most credible)
        # Penalize for red flags and high bias
        base_score = 100
        base_score -= (subjectivity * 0.4) # Subjectivity is a big penalty
        base_score -= (keyword_hits * 15)
        base_score -= (clickbait_hits * 10)
        
        # Stability floor
        score = max(5, min(95, base_score))
        
        # Verdict logic
        if score >= 75: verdict = "CREDIBLE"
        elif score >= 50: verdict = "UNCERTAIN"
        else: verdict = "LIKELY_FALSE"

        # Breakdown
        breakdown = {
            "fact_match": 100 - (keyword_hits * 20), # Proxy for matching
            "language": 100 - (clickbait_hits * 25),
            "sentiment": 100 - subjectivity,
            "source_quality": 50 # Local can't verify source quality alone
        }

        # Reasoning
        reasoning = f"Analysis based on {len(flags)} linguistic markers. "
        if subjectivity > 50: reasoning += "The text uses highly subjective language. "
        if clickbait_hits > 0: reasoning += "Clickbait patterns were identified. "
        if not flags: reasoning += "No major red flags detected in the writing style."

        return {
            "score": int(score),
            "verdict": verdict,
            "summary": f"Local AI Analysis: {verdict.replace('_', ' ')}",
            "reasoning": reasoning,
            "flags": flags,
            "content_type": "text",
            "breakdown": breakdown
        }

# Global Instance
engine = VeritasLocalEngine()
