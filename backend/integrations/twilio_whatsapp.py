import os
import requests
import base64
from scoring import calculate_credibility_score

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN  = os.getenv('TWILIO_AUTH_TOKEN', '')


def handle_whatsapp_message(text='', media_url=None, media_type='image/jpeg'):
    """
    Process WhatsApp message (text or image) and return a formatted reply.
    """
    try:
        # ── Image analysis ────────────────────────────────────────────────────
        if media_url:
            result = _analyze_media(media_url, media_type, caption=text)
            return _format_response(result, is_image=True)

        # ── Text too short ────────────────────────────────────────────────────
        if not text or len(text.strip()) < 5:
            return (
                "👋 Hi! I'm your *Credibility Checker Bot*.\n\n"
                "📝 *Text:* Send me any headline, claim, or article.\n"
                "🖼️ *Image:* Send me a screenshot or photo to analyze.\n\n"
                "I'll give you an AI-powered credibility score instantly using Gemini AI! 🔍"
            )

        # ── Text analysis ─────────────────────────────────────────────────────
        result = calculate_credibility_score(text)
        return _format_response(result, is_image=False)

    except Exception as e:
        print(f"WhatsApp handler error: {e}")
        return "⚠️ Error analyzing your message. Please try again with different text."


def _analyze_media(media_url, media_type, caption=''):
    """Download Twilio media and analyze with Gemini vision"""
    try:
        # Download image from Twilio (requires auth)
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None
        resp = requests.get(media_url, auth=auth, timeout=10)

        if resp.status_code != 200:
            raise Exception(f"Failed to download media: HTTP {resp.status_code}")

        image_bytes = resp.content
        result = calculate_credibility_score(
            text=caption or '[Image analysis]',
            image_data=image_bytes,
            image_mime=media_type
        )
        return result

    except Exception as e:
        print(f"Media download/analysis error: {e}")
        # Fall back to analyzing just the caption if image fails
        if caption and len(caption) > 5:
            return calculate_credibility_score(caption)
        raise


def _format_response(result, is_image=False):
    """Format the scoring result into a WhatsApp-friendly message"""
    score    = result.get('score', 50)
    breakdown = result.get('breakdown', {})
    flags    = result.get('flags', [])
    claims   = result.get('claims', [])
    verdict  = result.get('verdict', 'UNCERTAIN')
    summary  = result.get('summary', '')
    reasoning = result.get('reasoning', '')

    # Emoji + colour
    if score >= 75:
        circle, verdict_label = '🟢', 'CREDIBLE'
    elif score >= 50:
        circle, verdict_label = '🟡', 'UNCERTAIN'
    else:
        circle, verdict_label = '🔴', 'LIKELY FALSE'

    content_label = "🖼️ *Image*" if is_image else "📝 *Text*"

    msg  = f"{circle} *{verdict_label}* — {content_label} Analysis\n"
    msg += f"━━━━━━━━━━━━━━━\n"
    msg += f"📊 *Score: {score}/100*\n\n"

    # Breakdown
    if is_image:
        msg += "🔬 *Visual Analysis:*\n"
        msg += f"  • Authenticity: {breakdown.get('visual_authenticity', '–')}%\n"
        msg += f"  • Text claims: {breakdown.get('text_credibility', '–')}%\n"
        msg += f"  • Context: {breakdown.get('context_accuracy', '–')}%\n"
        msg += f"  • Source quality: {breakdown.get('source_quality', '–')}%\n\n"
    else:
        msg += "📈 *Breakdown:*\n"
        msg += f"  • Fact match: {breakdown.get('fact_match', '–')}%\n"
        msg += f"  • Language quality: {breakdown.get('language', '–')}%\n"
        msg += f"  • Sentiment: {breakdown.get('sentiment', '–')}%\n"
        msg += f"  • Source quality: {breakdown.get('source_quality', '–')}%\n\n"

    # Summary
    if summary:
        msg += f"💡 *Assessment:*\n{summary}\n\n"

    # Flags
    if flags:
        msg += f"⚠️ *Red Flags:*\n"
        for flag in flags[:3]:
            msg += f"  • {flag}\n"
        msg += "\n"

    # Fact-checked claims
    fact_claims = [c for c in claims if c.get('type') == 'FACT_CHECKED']
    if fact_claims:
        msg += f"🔎 *Fact Checks Found:*\n"
        for c in fact_claims[:2]:
            rating = c.get('rating', 'unknown')
            pub = c.get('publisher', '')
            msg += f"  • \"{c['text'][:60]}...\"\n    → *{rating}* ({pub})\n"
        msg += "\n"

    msg += f"_Powered by Gemini AI + Google Fact Check_ 🤖"
    return msg
