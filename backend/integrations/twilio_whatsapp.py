import os
import requests
import base64
from scoring import calculate_credibility_score, detect_intent, generate_chat_reply

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN  = os.getenv('TWILIO_AUTH_TOKEN', '')


def handle_whatsapp_message(text='', media_url=None, media_type='image/jpeg'):
    """
    Smart WhatsApp message handler:
    - Greetings / questions → natural Gemini reply
    - News/claims → credibility analysis
    - Images → analyze and give visual credibility report
    """
    try:
        # ── Image received ────────────────────────────────────────────────────
        if media_url:
            if not text or len(text) < 3:
                return (
                    "🖼️ *I've received your image!*\n\n"
                    "Should I analyze this for misinformation, deepfakes, or manipulation?\n\n"
                    "Reply with *'Yes'* or *'Check'* to start the AI analysis."
                )
            result = _download_and_analyze(media_url, media_type, caption=text)
            return _format_score(result, is_image=True)

        text = text.strip()

        # ── Handle simple 'Yes' for image analysis ────────────────────────────
        # (This is a bit tricky without state, but if they say 'Yes' and we just 
        #  got an image, we'd need a way to remember. For now, let's keep it simple 
        #  and just analyze if they send a prompt with the image or after.)
        
        # ── Empty message ─────────────────────────────────────────────────────
        if not text:
            return _welcome_message()

        # ── Detect intent ─────────────────────────────────────────────────────
        intent = detect_intent(text)

        if intent == 'GREETING':
            return _welcome_message()

        if intent in ('QUESTION', 'CONVERSATION'):
            return generate_chat_reply(text)

        # ── CLAIM → run credibility analysis ──────────────────────────────────
        if len(text) < 10:
            return (
                "Please send a longer message to fact-check.\n\n"
                "For example:\n"
                "\"Scientists discovered a miracle cure for cancer\"\n\n"
                "Or send me an image to analyze 🖼️"
            )

        result = calculate_credibility_score(text)
        return _format_score(result, is_image=False)

    except Exception as e:
        print(f"WhatsApp handler error: {e}")
        return "⚠️ Error processing your message. Please try again."



def _download_and_analyze(media_url, media_type, caption=''):
    """Download Twilio media and run Gemini image analysis"""
    try:
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None
        resp = requests.get(media_url, auth=auth, timeout=12)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}")
        return calculate_credibility_score(
            text=caption or '[Image analysis]',
            image_data=resp.content,
            image_mime=media_type
        )
    except Exception as e:
        print(f"Media error: {e}")
        if caption and len(caption) > 10:
            return calculate_credibility_score(caption)
        raise


def _format_score(result, is_image=False):
    """Format scoring result as a clean WhatsApp message"""
    score    = result.get('score', 50)
    bd       = result.get('breakdown', {})
    flags    = result.get('flags', [])
    claims   = result.get('claims', [])
    verdict  = result.get('verdict', 'UNCERTAIN')
    summary  = result.get('summary', '')

    if score >= 75:   circle, label = '🟢', 'CREDIBLE'
    elif score >= 50: circle, label = '🟡', 'UNCERTAIN'
    else:             circle, label = '🔴', 'LIKELY FALSE'

    content_icon = '🖼️ Image' if is_image else '📝 Text'

    lines = [
        f"{circle} *{label}* — {content_icon} Analysis",
        "━━━━━━━━━━━━━━━",
        f"📊 *Credibility Score: {score}/100*",
        f"🏷️ Verdict: {verdict.replace('_', ' ')}",
        "",
    ]

    # Breakdown
    if is_image:
        lines += [
            "🔬 *Visual Analysis:*",
            f"  • Authenticity: {bd.get('visual_authenticity', '–')}%",
            f"  • Text claims: {bd.get('text_credibility', '–')}%",
            f"  • Context: {bd.get('context_accuracy', '–')}%",
            f"  • Sources: {bd.get('source_quality', '–')}%",
            "",
        ]
    else:
        lines += [
            "📈 *Breakdown:*",
            f"  • Fact match: {bd.get('fact_match', '–')}%",
            f"  • Language: {bd.get('language', '–')}%",
            f"  • Sentiment: {bd.get('sentiment', '–')}%",
            f"  • Sources: {bd.get('source_quality', '–')}%",
            "",
        ]

    if summary:
        lines += [f"💡 {summary}", ""]

    if flags:
        lines.append("⚠️ *Red Flags:*")
        for f in flags[:3]:
            lines.append(f"  • {f}")
        lines.append("")

    fact_claims = [c for c in claims if c.get('type') == 'FACT_CHECKED']
    if fact_claims:
        lines.append("🔎 *Fact Checks:*")
        for c in fact_claims[:2]:
            lines.append(f"  • \"{c['text'][:55]}...\"")
            lines.append(f"    → {c.get('rating', 'checked')} ({c.get('publisher', '')})")
        lines.append("")

    lines.append("_Powered by Gemini 2.5 Flash + Google Fact Check AI_ 🤖")
    return "\n".join(lines)


def _welcome_message():
    return (
        "👋 Hi! I'm *CredChecker Bot* 🔍\n\n"
        "I use *Gemini 2.5 Flash AI* to detect misinformation!\n\n"
        "Here's what I can do:\n"
        "📝 *Fact-check text* — send any headline, claim, or article\n"
        "🖼️ *Analyze images* — send screenshots to detect manipulation\n"
        "🔎 *Cross-check* — I verify against Google's Fact Check database\n\n"
        "Just send me something suspicious and I'll give you a credibility score!\n\n"
        "_Example: \"Scientists found a miracle cure for cancer\"_"
    )


