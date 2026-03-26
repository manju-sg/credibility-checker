import os
import requests
from textblob import TextBlob
import spacy
import json
from datetime import datetime

GOOGLE_FACT_CHECK_API_KEY = os.getenv('GOOGLE_FACT_CHECK_API_KEY', '')
nlp = spacy.load('en_core_web_sm')

CREDIBILITY_CACHE = {}  # Simple in-memory cache

def calculate_credibility_score(text):
    """
    Main scoring function.
    
    Combines multiple signals:
    - Fact-check database matching (45%)
    - Language quality analysis (25%)
    - Sentiment analysis (20%)
    - Red flag detection (10%)
    
    Returns dict with score (0-100) and detailed breakdown.
    """
    try:
        # 1. Extract claims
        claims = extract_claims(text)
        
        # 2. Fact-check claims
        fact_match_score = check_claims_against_db(claims, text)
        
        # 3. Analyze language patterns
        language_score = analyze_language(text)
        
        # 4. Sentiment analysis
        sentiment_score = analyze_sentiment(text)
        
        # 5. Red flags
        flags = detect_red_flags(text)
        red_flag_penalty = len(flags) * 10  # Each flag: -10 points
        
        # 6. Aggregate scores
        final_score = (
            0.45 * fact_match_score +
            0.25 * language_score +
            0.20 * sentiment_score +
            0.10 * max(0, 100 - red_flag_penalty)
        )
        
        final_score = int(max(0, min(100, final_score)))
        
        # 7. Generate summary
        summary = generate_summary(final_score, flags, claims)
        
        return {
            "score": final_score,
            "breakdown": {
                "fact_match": int(fact_match_score),
                "language": int(language_score),
                "sentiment": int(sentiment_score)
            },
            "claims": claims[:5],  # Return top 5 claims
            "flags": flags[:3],     # Return top 3 flags
            "summary": summary
        }
    
    except Exception as e:
        print(f"Error in calculate_credibility_score: {str(e)}")
        return {
            "score": 50,
            "breakdown": {"fact_match": 50, "language": 50, "sentiment": 50},
            "claims": [],
            "flags": ["Analysis error"],
            "summary": "Unable to fully analyze content. Try a shorter text."
        }

def extract_claims(text):
    """
    Extract factual claims using spaCy NER.
    
    Returns list of claims with metadata.
    """
    try:
        doc = nlp(text)
        claims = []
        seen_texts = set()
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'DATE', 'FAC']:
                claim_text = ent.text.strip()
                if claim_text and claim_text not in seen_texts and len(claim_text) > 2:
                    claims.append({
                        "text": claim_text,
                        "type": ent.label_,
                        "verified": False,
                        "confidence": 0.7
                    })
                    seen_texts.add(claim_text)
        
        # Extract numerical claims (percentages, numbers in context)
        for token in doc:
            if token.like_num:
                context = " ".join([t.text for t in doc[max(0, token.i-2):min(len(doc), token.i+3)]])
                if context not in seen_texts and len(context) > 5:
                    claims.append({
                        "text": context,
                        "type": "QUANTITY",
                        "verified": False,
                        "confidence": 0.6
                    })
                    seen_texts.add(context)
        
        return claims[:10]  # Limit to 10 claims
    
    except Exception as e:
        print(f"Error in extract_claims: {str(e)}")
        return []

def check_claims_against_db(claims, original_text):
    """
    Check claims against Google Fact Check API.
    
    Returns credibility score (0-100) based on fact-check matches.
    """
    if not GOOGLE_FACT_CHECK_API_KEY or not claims:
        return 70  # Default neutral score
    
    verified_true = 0
    verified_false = 0
    total_checks = 0
    
    for claim in claims[:3]:  # Check top 3 claims to save API quota
        try:
            cache_key = claim['text'].lower()
            
            # Check cache first
            if cache_key in CREDIBILITY_CACHE:
                cached = CREDIBILITY_CACHE[cache_key]
                if cached['rating'].lower() in ['true', 'mostly true']:
                    verified_true += 1
                elif cached['rating'].lower() in ['false', 'mostly false']:
                    verified_false += 1
                total_checks += 1
                continue
            
            # Query Google Fact Check API
            response = requests.get(
                'https://factcheckapi.googleapis.com/v1alpha1/claims:search',
                params={
                    'query': claim['text'][:100],  # API limit
                    'key': GOOGLE_FACT_CHECK_API_KEY
                },
                timeout=3
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('claims'):
                    review = data['claims'][0].get('claimReview', [{}])[0]
                    rating = review.get('textualRating', 'unknown').lower()
                    
                    # Cache the result
                    CREDIBILITY_CACHE[cache_key] = {
                        'rating': rating,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if rating in ['true', 'mostly true']:
                        verified_true += 1
                    elif rating in ['false', 'mostly false']:
                        verified_false += 1
                    
                    total_checks += 1
                    claim['verified'] = True
        
        except requests.exceptions.Timeout:
            print("Fact check API timeout")
        except Exception as e:
            print(f"Error checking claim '{claim['text']}': {str(e)}")
    
    # Calculate score: more true claims = higher score
    if total_checks == 0:
        return 70  # No fact-checks found
    
    score = ((verified_true / total_checks) * 100) - (verified_false * 20)
    return max(0, min(100, score))

def analyze_language(text):
    """
    Analyze language quality and patterns.
    
    Red flags:
    - ALL CAPS text
    - Excessive punctuation
    - Sensationalist words
    - Vague claims
    """
    score = 100
    
    # Check for ALL CAPS
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 1]
    if len(caps_words) > len(text.split()) * 0.3:
        score -= 30
    
    # Check for excessive punctuation
    if text.count('!') > 3 or text.count('?') > 3:
        score -= 20
    
    # Sensationalist keywords
    sensational_words = [
        'shocking', 'breaking', 'exclusive', 'stunning',
        'unbelievable', 'must read', 'you wont believe', 'exposé'
    ]
    for word in sensational_words:
        if word.lower() in text.lower():
            score -= 15
            break
    
    # Check for proper nouns and structure
    doc = nlp(text)
    noun_ratio = sum(1 for token in doc if token.pos_ in ['NOUN', 'PROPN']) / len(doc) if len(doc) > 0 else 0
    if noun_ratio < 0.1:
        score -= 10  # Too vague
    
    return max(0, min(100, score))

def analyze_sentiment(text):
    """
    Analyze sentiment. Extreme sentiment = lower credibility.
    
    Neutral sentiment indicates more objective reporting.
    """
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Neutral polarity is more credible
        if -0.15 < polarity < 0.15:
            polarity_score = 90
        elif -0.4 < polarity < 0.4:
            polarity_score = 75
        else:
            polarity_score = 55
        
        # Lower subjectivity is more credible
        if subjectivity < 0.4:
            subjectivity_score = 95
        elif subjectivity < 0.6:
            subjectivity_score = 75
        else:
            subjectivity_score = 55
        
        return (polarity_score * 0.5) + (subjectivity_score * 0.5)
    
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        return 70

def detect_red_flags(text):
    """
    Detect common misinformation red flags.
    
    Returns list of flags found.
    """
    flags = []
    
    if len(text) < 20:
        flags.append("Content too short to analyze thoroughly")
    
    if len(text) > 3000:
        flags.append("Very long content - may contain mixed claims")
    
    if '!!!' in text or '???' in text:
        flags.append("Excessive punctuation detected")
    
    if text.count('\n') > 10:
        flags.append("Fragmented text structure")
    
    # Check for emotional triggers
    emotional_words = ['hate', 'love', 'disgusting', 'amazing', 'incredible']
    if any(word in text.lower() for word in emotional_words):
        flags.append("Emotional language detected")
    
    # Check for vagueness
    vague_words = ['allegedly', 'rumor', 'apparently', 'supposedly', 'might be']
    if any(phrase in text.lower() for phrase in vague_words):
        flags.append("Unverified claims detected")
    
    # Check for lack of sources
    if not any(pattern in text.lower() for pattern in ['according to', 'source', 'study', 'research']):
        if len(text.split()) > 50:
            flags.append("No sources cited")
    
    return flags

def generate_summary(score, flags, claims):
    """Generate a human-readable summary of the analysis"""
    if score >= 75:
        credibility = "likely credible"
        emoji = "✅"
    elif score >= 50:
        credibility = "uncertain or mixed"
        emoji = "⚠️"
    else:
        credibility = "likely false or misleading"
        emoji = "❌"
    
    summary = f"{emoji} This content appears {credibility}. "
    
    if flags:
        summary += f"We detected: {', '.join(flags[:2])}. "
    
    if claims:
        summary += f"We found {len(claims)} claim(s) to verify."
    
    return summary
